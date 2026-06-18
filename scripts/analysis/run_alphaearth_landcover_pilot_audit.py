from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.bwer import BWERConfig, compute_bwer
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from scripts.analysis.check_alphaearth_export_schema import DEFAULT_INPUT, EMBEDDING_BANDS, check_alphaearth_export_schema


DEFAULT_OUTPUT = Path("outputs/alphaearth_gee_pilot_v1")
SLICE_COLUMNS = ["country_iso3", "region", "biome_or_ecoregion", "urban_rural_or_built_proxy", "income_group", "support_stratum"]
CLASS_NAMES = {
    "10": "Tree cover",
    "20": "Shrubland",
    "30": "Grassland",
    "40": "Cropland",
    "50": "Built-up",
    "60": "Bare/sparse vegetation",
    "70": "Snow and ice",
    "80": "Permanent water bodies",
    "90": "Herbaceous wetland",
    "95": "Mangroves",
    "100": "Moss and lichen",
}


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_name(value: Any) -> str:
    return CLASS_NAMES.get(str(value), str(value))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _fit_multinomial_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    seed: int = 42,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
    epochs: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x = (x_train - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(x.shape[1], n_classes))
    bias = np.zeros(n_classes, dtype=float)
    y_onehot = np.eye(n_classes)[y_train]
    n = max(1, x.shape[0])
    for _ in range(epochs):
        probs = _softmax(x @ weights + bias)
        diff = (probs - y_onehot) / n
        grad_w = x.T @ diff + l2 * weights
        grad_b = diff.sum(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias, mean, std


def _predict_logreg(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_norm = (x - mean) / std
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
    probs = _softmax(x_norm @ weights + bias)
    return np.argmax(probs, axis=1), np.max(probs, axis=1)


def _rows_to_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[str]]:
    x = np.asarray([[_float(row.get(band), 0.0) for band in EMBEDDING_BANDS] for row in rows], dtype=float)
    labels = [str(row.get("worldcover_label")) for row in rows]
    return x, labels


def _split_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train = [row for row in rows if str(row.get("split", "")).lower() == "train"]
    test = [row for row in rows if str(row.get("split", "")).lower() in {"test", "val", "validation"}]
    return train, test


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    values = []
    for cls in range(n_classes):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return float(np.mean(values)) if values else float("nan")


def _balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    recalls = []
    for cls in range(n_classes):
        total = sum(1 for yt in y_true if yt == cls)
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        if total:
            recalls.append(correct / total)
    return float(np.mean(recalls)) if recalls else float("nan")


def _support_strata(rows: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        for column in SLICE_COLUMNS:
            value = str(row.get(column, ""))
            if value:
                counts[(column, value)] = counts.get((column, value), 0) + 1
    for row in rows:
        country_support = counts.get(("country_iso3", str(row.get("country_iso3", ""))), 0)
        if not country_support:
            row["support_stratum"] = ""
        elif country_support < 20:
            row["support_stratum"] = "country_support_lt20"
        elif country_support < 50:
            row["support_stratum"] = "country_support_20_49"
        else:
            row["support_stratum"] = "country_support_ge50"


def _sample_support_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for column in ["split", "worldcover_label", *SLICE_COLUMNS]:
        if not rows or column not in rows[0]:
            continue
        counts: dict[str, int] = {}
        class_sets: dict[str, set[str]] = {}
        for row in rows:
            value = str(row.get(column, ""))
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
            class_sets.setdefault(value, set()).add(str(row.get("worldcover_label")))
        for value, count in sorted(counts.items()):
            output.append(
                {
                    "slice_variable": column,
                    "slice_value": value,
                    "sample_count": count,
                    "class_count": len(class_sets.get(value, set())),
                    "support_warning": "below_20" if count < 20 else "",
                }
            )
    return output


def _bwer_rows(audit_rows: Sequence[Mapping[str, Any]], min_samples_per_slice: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    config = BWERConfig(
        dataset="alphaearth_worldcover_pilot",
        model="numpy_multinomial_logreg",
        task="land_cover_classification",
        split="test",
        score_name="correct",
        risk_name="risk",
        min_samples_per_slice=min_samples_per_slice,
        min_slices_required=2,
        tail_fraction=0.10,
    )
    for column in SLICE_COLUMNS:
        if audit_rows and column in audit_rows[0] and any(str(row.get(column, "")).strip() for row in audit_rows):
            raw = compute_bwer(audit_rows, config, column, risk_column="risk")
            summary_rows.append({**raw.summary, "analysis_type": "raw"})
            slice_rows.extend({**row, "analysis_type": "raw"} for row in raw.by_slice)
            if column != "worldcover_class_name":
                std = compute_bwer(audit_rows, config, column, balance_variable="worldcover_class_name", risk_column="risk")
                summary_rows.append({**std.summary, "analysis_type": "standardised"})
                slice_rows.extend({**row, "analysis_type": "standardised"} for row in std.by_slice)
    if audit_rows and any(not math.isnan(_float(row.get("confidence"))) for row in audit_rows):
        ranked = sorted(audit_rows, key=lambda row: _float(row.get("confidence"), -1.0), reverse=True)
        keep = max(1, int(math.ceil(len(ranked) * 0.8)))
        retained = [dict(row) for row in ranked[:keep]]
        sel_config = BWERConfig(
            dataset="alphaearth_worldcover_pilot",
            model="numpy_multinomial_logreg",
            task="land_cover_classification",
            split="test",
            score_name="correct",
            risk_name="risk",
            min_samples_per_slice=min_samples_per_slice,
            min_slices_required=2,
            tail_fraction=0.10,
            selective_coverage=0.8,
        )
        for column in ["country_iso3", "biome_or_ecoregion", "urban_rural_or_built_proxy", "income_group"]:
            if retained and column in retained[0] and any(str(row.get(column, "")).strip() for row in retained):
                result = compute_bwer(retained, sel_config, column, risk_column="risk")
                summary_rows.append({**result.summary, "analysis_type": "selective_confidence_top80"})
                slice_rows.extend({**row, "analysis_type": "selective_confidence_top80"} for row in result.by_slice)
    return summary_rows, slice_rows


def _write_figures(output: Path, metrics: Sequence[Mapping[str, Any]], bwer_rows: Sequence[Mapping[str, Any]], slice_rows: Sequence[Mapping[str, Any]], support_rows: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures = ensure_dir(output / "figures")
    paths: dict[str, Path] = {}

    def save(fig: Any, name: str) -> None:
        png = figures / f"{name}.png"
        pdf = figures / f"{name}.pdf"
        fig.tight_layout()
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
        paths[f"{name}_png"] = png
        paths[f"{name}_pdf"] = pdf

    accuracy = _float(metrics[0].get("accuracy")) if metrics else float("nan")
    country_bwer = next((_float(row.get("bwer")) for row in bwer_rows if row.get("slice_variable") == "country_iso3" and row.get("analysis_type") == "raw"), float("nan"))
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.bar(["accuracy", "country Raw-BWER"], [accuracy, country_bwer], color=["#2F5DA8", "#A04D3A"])
    ax.set_ylim(0, 1)
    ax.set_title("AlphaEarth pilot aggregate vs BWER")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "aggregate_vs_bwer_alphaearth_pilot")

    country = [row for row in slice_rows if row.get("slice_variable") in {"country_iso3", "region"} and row.get("analysis_type") == "raw" and row.get("is_valid_slice")]
    country = sorted(country, key=lambda row: _float(row.get("balanced_risk")), reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar([str(row.get("slice_value")) for row in country], [_float(row.get("balanced_risk")) for row in country], color="#5E6C84")
    ax.set_ylabel("Risk")
    ax.set_title("Highest country/region slice risks")
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "slice_risk_by_country_or_region")

    std = [row for row in bwer_rows if row.get("analysis_type") == "standardised"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar([str(row.get("slice_variable")) for row in std], [_float(row.get("bwer")) for row in std], color="#6B8E23")
    ax.set_ylabel("Standardised-BWER")
    ax.set_title("Class-standardised BWER by slice")
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "class_standardised_bwer")

    support = [row for row in support_rows if row.get("slice_variable") in {"country_iso3", "worldcover_label"}]
    support = sorted(support, key=lambda row: _float(row.get("sample_count")), reverse=True)[:25]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar([str(row.get("slice_value")) for row in support], [_float(row.get("sample_count")) for row in support], color="#8C6D31")
    ax.set_ylabel("Samples")
    ax.set_title("Pilot support by slice")
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "support_by_slice")
    return paths


def _missing_export_outputs(output: Path, input_csv: Path, report_path: Path, status: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = {
        "alphaearth_export_schema_report": report_path,
        "alphaearth_sample_support_summary": output / "alphaearth_sample_support_summary.csv",
        "alphaearth_landcover_pilot_metrics": output / "alphaearth_landcover_pilot_metrics.csv",
        "alphaearth_bwer_summary": output / "alphaearth_bwer_summary.csv",
        "alphaearth_slice_risk_summary": output / "alphaearth_slice_risk_summary.csv",
        "alphaearth_pilot_caveats": output / "alphaearth_pilot_caveats.csv",
        "alphaearth_pilot_report": output / "alphaearth_pilot_report.md",
    }
    unavailable = {
        "status": "unavailable_missing_or_invalid_export",
        "input_csv": str(input_csv),
        "schema_status": status.get("schema_status", ""),
        "required_action": "Run the GEE/Colab export and copy the CSV to the expected local path.",
    }
    write_csv(artifacts["alphaearth_sample_support_summary"], [unavailable])
    write_csv(artifacts["alphaearth_landcover_pilot_metrics"], [unavailable])
    write_csv(artifacts["alphaearth_bwer_summary"], [unavailable])
    write_csv(artifacts["alphaearth_slice_risk_summary"], [unavailable])
    write_csv(
        artifacts["alphaearth_pilot_caveats"],
        [
            {"category": "missing_export", "caveat": f"Export table missing or invalid: {input_csv}"},
            {"category": "required_columns", "caveat": str(status.get("missing_required_columns", ""))},
            {"category": "claim_scope", "caveat": "No empirical AlphaEarth claims are supported until a valid GEE export table exists."},
        ],
    )
    artifacts["alphaearth_pilot_report"].write_text(
        "# AlphaEarth land-cover pilot audit\n\n"
        f"Local audit did not run because schema status is `{status.get('schema_status')}` for `{input_csv}`.\n\n"
        "See `alphaearth_export_schema_report.md` for required columns. No model training, BWER result, or empirical AlphaEarth claim was produced.\n",
        encoding="utf-8",
    )
    return artifacts


def run_alphaearth_landcover_pilot_audit(
    input_csv: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    min_samples_per_slice: int = 5,
    seed: int = 42,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    status, schema_report = check_alphaearth_export_schema(input_csv, output)
    if status["schema_status"] != "ok":
        return _missing_export_outputs(output, input_csv, schema_report, status)
    raw_rows = [dict(row) for row in read_csv_rows(input_csv)]
    _support_strata(raw_rows)
    train_rows, test_rows = _split_rows(raw_rows)
    if not train_rows or not test_rows:
        status = {"schema_status": "invalid_split", "missing_required_columns": "split must contain train and test/val rows"}
        return _missing_export_outputs(output, input_csv, schema_report, status)
    x_train, y_train_raw = _rows_to_arrays(train_rows)
    x_test, y_test_raw = _rows_to_arrays(test_rows)
    classes = sorted(set(y_train_raw) | set(y_test_raw), key=lambda value: int(value) if str(value).isdigit() else str(value))
    class_to_idx = {label: index for index, label in enumerate(classes)}
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    y_train = np.asarray([class_to_idx[label] for label in y_train_raw], dtype=int)
    y_test = np.asarray([class_to_idx[label] for label in y_test_raw], dtype=int)
    weights, bias, mean, std = _fit_multinomial_logreg(x_train, y_train, len(classes), seed=seed)
    pred_idx, confidence = _predict_logreg(x_test, weights, bias, mean, std)
    audit_rows: list[dict[str, Any]] = []
    for row, y_idx, p_idx, conf in zip(test_rows, y_test, pred_idx, confidence):
        label = idx_to_class[int(y_idx)]
        pred = idx_to_class[int(p_idx)]
        correct = int(label == pred)
        audit_rows.append(
            {
                **dict(row),
                "dataset": "alphaearth_worldcover_pilot",
                "task": "land_cover_classification",
                "model": "numpy_multinomial_logreg",
                "label": label,
                "class_label": _label_name(label),
                "prediction": pred,
                "predicted_class_name": _label_name(pred),
                "correct": correct,
                "risk": 1 - correct,
                "confidence": float(conf),
            }
        )
    support_rows = _sample_support_summary(raw_rows)
    metrics = [
        {
            "dataset": "alphaearth_worldcover_pilot",
            "model": "numpy_multinomial_logreg",
            "split": "test",
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "n_classes": len(classes),
            "accuracy": float(np.mean([row["correct"] for row in audit_rows])),
            "balanced_accuracy": _balanced_accuracy(y_test.tolist(), pred_idx.tolist(), len(classes)),
            "macro_f1": _macro_f1(y_test.tolist(), pred_idx.tolist(), len(classes)),
            "claim_scope": "pilot only; ESA WorldCover is map-label agreement target, not perfect ground truth",
        }
    ]
    bwer_rows, slice_rows = _bwer_rows(audit_rows, min_samples_per_slice=min_samples_per_slice)
    caveats = [
        {"category": "pilot_scope", "caveat": "This is a small AlphaEarth/GEE pilot, not a final empirical deployment claim."},
        {"category": "label_source", "caveat": "ESA WorldCover is map-label supervision/agreement target, not human ground truth."},
        {"category": "dynamic_world", "caveat": "Dynamic World confidence is diagnostic only if exported; it is not human truth."},
        {"category": "split_scope", "caveat": "Formal claims require spatial-block or location-disjoint split; random split is sanity only."},
        {"category": "social_claims", "caveat": "Do not make causal social fairness claims from this pilot."},
    ]
    artifacts = {
        "alphaearth_export_schema_report": schema_report,
        "alphaearth_audit_table": output / "alphaearth_landcover_pilot_audit_table.csv",
        "alphaearth_sample_support_summary": output / "alphaearth_sample_support_summary.csv",
        "alphaearth_landcover_pilot_metrics": output / "alphaearth_landcover_pilot_metrics.csv",
        "alphaearth_bwer_summary": output / "alphaearth_bwer_summary.csv",
        "alphaearth_slice_risk_summary": output / "alphaearth_slice_risk_summary.csv",
        "alphaearth_pilot_caveats": output / "alphaearth_pilot_caveats.csv",
        "alphaearth_pilot_report": output / "alphaearth_pilot_report.md",
    }
    write_csv(artifacts["alphaearth_audit_table"], audit_rows)
    write_csv(artifacts["alphaearth_sample_support_summary"], support_rows)
    write_csv(artifacts["alphaearth_landcover_pilot_metrics"], metrics)
    write_csv(artifacts["alphaearth_bwer_summary"], bwer_rows)
    write_csv(artifacts["alphaearth_slice_risk_summary"], slice_rows)
    write_csv(artifacts["alphaearth_pilot_caveats"], caveats)
    artifacts["alphaearth_pilot_report"].write_text(
        "# AlphaEarth land-cover pilot audit\n\n"
        "This pilot used a single CPU-friendly NumPy multinomial logistic regression over exported AlphaEarth embedding bands. No image chips were exported or used locally.\n\n"
        f"- Train rows: {len(train_rows)}\n"
        f"- Test rows: {len(test_rows)}\n"
        f"- Accuracy: {metrics[0]['accuracy']:.4f}\n"
        f"- Balanced accuracy: {metrics[0]['balanced_accuracy']:.4f}\n"
        f"- Macro-F1: {metrics[0]['macro_f1']:.4f}\n\n"
        "Claims are pilot-only unless support, split protocol, and export provenance are upgraded for a full run.\n",
        encoding="utf-8",
    )
    artifacts.update({f"figure_{name}": path for name, path in _write_figures(output, metrics, bwer_rows, slice_rows, support_rows).items()})
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local AlphaEarth land-cover pilot audit from a GEE-exported table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-samples-per-slice", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    artifacts = run_alphaearth_landcover_pilot_audit(args.input, args.out, args.min_samples_per_slice, args.seed)
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    if "alphaearth_landcover_pilot_metrics" not in artifacts:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
