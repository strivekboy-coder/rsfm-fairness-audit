from __future__ import annotations

import argparse
import math
import shutil
import sys
import zipfile
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
from scripts.analysis.check_alphaearth_full_export_schema import (
    DEFAULT_EXPORT,
    DEFAULT_GEE_ROOT,
    DEFAULT_MANIFEST,
    EMBEDDING_BANDS,
    check_alphaearth_full_export_schema,
    read_alphaearth_full_export,
)


AUDIT_ROOT = Path("outputs/alphaearth_landcover_audit_full_v1")
UNIFIED_V4_ROOT = Path("outputs/unified_paper_package_v4")
SLICE_COLUMNS = ["country_iso3", "region", "worldcover_class_name", "country_class", "region_class", "biome_or_ecoregion", "urban_rural_or_built_proxy", "income_group"]
TARGET_COVERAGES = [0.7, 0.8, 0.9]
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _fit_numpy_logreg(x_train: np.ndarray, y_train: np.ndarray, n_classes: int, seed: int = 42, epochs: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x = np.nan_to_num((x_train - mean) / std)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.01, size=(x.shape[1], n_classes))
    bias = np.zeros(n_classes)
    y_onehot = np.eye(n_classes)[y_train]
    lr = 0.18
    l2 = 1e-3
    n = max(1, x.shape[0])
    for _ in range(epochs):
        probs = _softmax(x @ weights + bias)
        diff = (probs - y_onehot) / n
        weights -= lr * (x.T @ diff + l2 * weights)
        bias -= lr * diff.sum(axis=0)
    return weights, bias, mean, std


def _predict_numpy_logreg(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, bias, mean, std = model
    return _softmax(np.nan_to_num((x - mean) / std) @ weights + bias)


def _fit_predict_proba(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, n_classes: int, seed: int) -> tuple[str, np.ndarray]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, l2_regularization=0.02, random_state=seed)
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_eval)
        if probs.shape[1] != n_classes:
            full = np.zeros((x_eval.shape[0], n_classes), dtype=float)
            for index, cls in enumerate(model.classes_):
                full[:, int(cls)] = probs[:, index]
            probs = full
        return "hist_gradient_boosting", probs
    except Exception:
        model = _fit_numpy_logreg(x_train, y_train, n_classes, seed=seed)
        return "numpy_multinomial_logreg_fallback", _predict_numpy_logreg(x_eval, model)


def _arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[str]]:
    return np.asarray([[_float(row.get(band), 0.0) for band in EMBEDDING_BANDS] for row in rows], dtype=float), [str(row.get("worldcover_label")) for row in rows]


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
    values = []
    for cls in range(n_classes):
        total = sum(1 for item in y_true if item == cls)
        if total:
            values.append(sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls) / total)
    return float(np.mean(values)) if values else float("nan")


def _prepare_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item["worldcover_class_name"] = item.get("worldcover_class_name") or CLASS_NAMES.get(str(item.get("worldcover_label")), str(item.get("worldcover_label")))
        item["country_class"] = f"{item.get('country_iso3','')}|{item.get('worldcover_class_name','')}" if item.get("country_iso3") and item.get("worldcover_class_name") else ""
        item["region_class"] = f"{item.get('region','')}|{item.get('worldcover_class_name','')}" if item.get("region") and item.get("worldcover_class_name") else ""
        output.append(item)
    return output


def _split(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = [dict(row) for row in rows if str(row.get("split", "")).lower() == "train"]
    calibration = [dict(row) for row in rows if str(row.get("split", "")).lower() in {"calibration", "calib", "val", "validation"}]
    test = [dict(row) for row in rows if str(row.get("split", "")).lower() == "test"]
    rng = np.random.default_rng(42)
    if not calibration and len(train) >= 10:
        indices = np.arange(len(train))
        rng.shuffle(indices)
        n_cal = max(1, int(len(train) * 0.15))
        cal_indices = set(indices[:n_cal].tolist())
        calibration = [row for idx, row in enumerate(train) if idx in cal_indices]
        train = [row for idx, row in enumerate(train) if idx not in cal_indices]
    if not test:
        test = [dict(row) for row in rows if str(row.get("split", "")).lower() in {"val", "validation"}]
    rng.shuffle(train)
    random_rows = [dict(row) for row in rows]
    for idx, row in enumerate(random_rows):
        row["split"] = "test" if idx % 5 == 0 else "calibration" if idx % 5 == 1 else "train"
    return train, calibration, test, random_rows


def _prediction_rows(eval_rows: Sequence[Mapping[str, Any]], labels: Sequence[str], probs: np.ndarray, classes: list[str], model_name: str, protocol: str) -> list[dict[str, Any]]:
    output = []
    for row, true_label, prob in zip(eval_rows, labels, probs):
        pred_idx = int(np.argmax(prob))
        pred_label = classes[pred_idx]
        true_idx = classes.index(true_label)
        correct = int(pred_label == true_label)
        item = {
            **dict(row),
            "dataset": "alphaearth_worldcover_full",
            "task": "land_cover_classification",
            "model": model_name,
            "split_protocol": protocol,
            "label": true_label,
            "class_label": row.get("worldcover_class_name") or CLASS_NAMES.get(str(true_label), str(true_label)),
            "prediction": pred_label,
            "predicted_class_name": CLASS_NAMES.get(str(pred_label), str(pred_label)),
            "correct": correct,
            "risk": 1 - correct,
            "confidence": float(np.max(prob)),
            "p_true": float(prob[true_idx]),
        }
        for cls, value in zip(classes, prob):
            item[f"prob_{cls}"] = float(value)
        output.append(item)
    return output


def _metrics(rows: Sequence[Mapping[str, Any]], classes: list[str], model_name: str, protocol: str, n_train: int, n_calibration: int) -> dict[str, Any]:
    y_true = [classes.index(str(row.get("label"))) for row in rows]
    y_pred = [classes.index(str(row.get("prediction"))) for row in rows]
    return {
        "dataset": "alphaearth_worldcover_full",
        "model": model_name,
        "split_protocol": protocol,
        "n_train": n_train,
        "n_calibration": n_calibration,
        "n_test": len(rows),
        "n_classes": len(classes),
        "accuracy": float(np.mean([row.get("correct") for row in rows])) if rows else "",
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred, len(classes)) if rows else "",
        "macro_f1": _macro_f1(y_true, y_pred, len(classes)) if rows else "",
    }


def _bwer_family(rows: Sequence[Mapping[str, Any]], model_name: str, protocol: str, analysis_type: str, min_support: int = 20) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    config = BWERConfig(dataset="alphaearth_worldcover_full", model=model_name, task="land_cover_classification", split=protocol, min_samples_per_slice=min_support, min_slices_required=2, tail_fraction=0.10)
    for column in SLICE_COLUMNS:
        if not rows or column not in rows[0] or not any(str(row.get(column, "")).strip() for row in rows):
            continue
        raw = compute_bwer(rows, config, column, risk_column="risk")
        summaries.append({**raw.summary, "analysis_type": analysis_type, "bwer_type": "raw"})
        slices.extend({**row, "analysis_type": analysis_type, "bwer_type": "raw"} for row in raw.by_slice)
        support.extend(raw.support_diagnostics)
        if column != "worldcover_class_name":
            std = compute_bwer(rows, config, column, balance_variable="worldcover_class_name", risk_column="risk")
            summaries.append({**std.summary, "analysis_type": analysis_type, "bwer_type": "standardised"})
            slices.extend({**row, "analysis_type": analysis_type, "bwer_type": "standardised"} for row in std.by_slice)
            support.extend(std.support_diagnostics)
    return summaries, slices, support


def _coverage_selection(rows: Sequence[Mapping[str, Any]], coverage: float) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: _float(row.get("confidence"), -1.0), reverse=True)
    keep = max(1, int(math.ceil(len(ranked) * coverage)))
    return [dict(row) for row in ranked[:keep]]


def _selective_outputs(test_rows: Sequence[Mapping[str, Any]], model_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    risk_rows = []
    bwer_rows = []
    slice_rows = []
    for coverage in TARGET_COVERAGES:
        retained = _coverage_selection(test_rows, coverage)
        risk_rows.append(
            {
                "selector": "confidence_topk",
                "coverage_target": coverage,
                "retained_coverage": len(retained) / len(test_rows) if test_rows else "",
                "mean_risk": float(np.mean([row.get("risk") for row in retained])) if retained else "",
                "n_retained": len(retained),
            }
        )
        summaries, slices, _ = _bwer_family(retained, model_name, f"selective_topk_{coverage}", "selective_topk", min_support=5)
        for row in summaries:
            row["coverage_target"] = coverage
        for row in slices:
            row["coverage_target"] = coverage
        bwer_rows.extend(summaries)
        slice_rows.extend(slices)
    return risk_rows, bwer_rows, slice_rows


def _calibrated_threshold_outputs(calibration: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]], model_name: str) -> list[dict[str, Any]]:
    rows = []
    cal_conf = sorted([_float(row.get("confidence")) for row in calibration if not math.isnan(_float(row.get("confidence")))])
    for coverage in TARGET_COVERAGES:
        if not cal_conf:
            threshold = float("nan")
            retained = []
        else:
            threshold = cal_conf[max(0, min(len(cal_conf) - 1, int(math.floor((1 - coverage) * len(cal_conf)))))]
            retained = [dict(row) for row in test_rows if _float(row.get("confidence")) >= threshold]
        summaries, _, _ = _bwer_family(retained, model_name, f"calibrated_threshold_{coverage}", "calibrated_threshold", min_support=5)
        for row in summaries:
            row["coverage_target"] = coverage
            row["confidence_threshold"] = "" if math.isnan(threshold) else threshold
            row["retained_coverage"] = len(retained) / len(test_rows) if test_rows else ""
            rows.append(row)
    return rows


def _conformal_outputs(calibration: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]], model_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cal_scores = sorted([1.0 - _float(row.get("p_true")) for row in calibration if not math.isnan(_float(row.get("p_true")))])
    coverage_rows, slice_rows, bwer_rows, set_rows = [], [], [], []
    prob_columns = sorted([key for key in test_rows[0] if key.startswith("prob_")], key=lambda key: key) if test_rows else []
    for coverage in TARGET_COVERAGES:
        alpha = 1.0 - coverage
        if cal_scores:
            q_index = min(len(cal_scores) - 1, int(math.ceil((len(cal_scores) + 1) * (1 - alpha))) - 1)
            qhat = cal_scores[q_index]
        else:
            qhat = float("nan")
        evaluated = []
        for row in test_rows:
            probs = {column.replace("prob_", ""): _float(row.get(column)) for column in prob_columns}
            pred_set = [label for label, prob in probs.items() if not math.isnan(prob) and 1.0 - prob <= qhat]
            covered = str(row.get("label")) in pred_set
            item = dict(row)
            item["conformal_covered"] = int(covered)
            item["set_size"] = len(pred_set)
            item["risk"] = 1 - int(covered)
            evaluated.append(item)
        coverage_rows.append(
            {
                "method": "split_conformal_p_true",
                "coverage_target": coverage,
                "alpha": alpha,
                "qhat": "" if math.isnan(qhat) else qhat,
                "marginal_coverage": float(np.mean([row["conformal_covered"] for row in evaluated])) if evaluated else "",
                "average_set_size": float(np.mean([row["set_size"] for row in evaluated])) if evaluated else "",
                "n_calibration": len(cal_scores),
                "n_test": len(evaluated),
            }
        )
        for column in ["country_iso3", "region", "worldcover_class_name", "income_group", "urban_rural_or_built_proxy"]:
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for row in evaluated:
                value = row.get(column)
                if value:
                    grouped.setdefault(str(value), []).append(row)
            for value, items in sorted(grouped.items()):
                slice_rows.append(
                    {
                        "coverage_target": coverage,
                        "slice_variable": column,
                        "slice_value": value,
                        "support_count": len(items),
                        "slice_coverage": float(np.mean([row["conformal_covered"] for row in items])),
                        "coverage_gap": coverage - float(np.mean([row["conformal_covered"] for row in items])),
                        "average_set_size": float(np.mean([row["set_size"] for row in items])),
                    }
                )
        summaries, _, _ = _bwer_family(evaluated, model_name, f"conformal_coverage_{coverage}", "conformal_coverage", min_support=5)
        for row in summaries:
            row["coverage_target"] = coverage
            bwer_rows.append(row)
        set_rows.append({"coverage_target": coverage, "average_set_size": coverage_rows[-1]["average_set_size"], "qhat": coverage_rows[-1]["qhat"]})
    return coverage_rows, slice_rows, bwer_rows, set_rows


def _sensitivity_rows(base_rows: Sequence[Mapping[str, Any]], model_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alpha_rows = []
    support_rows = []
    missing_rows = []
    for tail in [0.05, 0.10, 0.20]:
        config = BWERConfig(dataset="alphaearth_worldcover_full", model=model_name, task="land_cover_classification", split="test", min_samples_per_slice=5, tail_fraction=tail)
        result = compute_bwer(base_rows, config, "country_iso3", risk_column="risk") if base_rows else None
        alpha_rows.append({"tail_fraction": tail, "slice_variable": "country_iso3", "bwer": result.summary.get("bwer", "") if result else ""})
    for threshold in [5, 20, 50, 200]:
        config = BWERConfig(dataset="alphaearth_worldcover_full", model=model_name, task="land_cover_classification", split="test", min_samples_per_slice=threshold)
        result = compute_bwer(base_rows, config, "country_iso3", risk_column="risk") if base_rows else None
        support_rows.append({"min_samples_per_slice": threshold, "slice_variable": "country_iso3", "bwer": result.summary.get("bwer", "") if result else "", "n_valid_slices": result.summary.get("n_slices_valid", "") if result else ""})
    for policy in ["renormalize", "overlap", "invalidate"]:
        config = BWERConfig(dataset="alphaearth_worldcover_full", model=model_name, task="land_cover_classification", split="test", min_samples_per_slice=5, missing_balance_policy=policy)
        result = compute_bwer(base_rows, config, "country_iso3", balance_variable="worldcover_class_name", risk_column="risk") if base_rows else None
        missing_rows.append({"missing_balance_policy": policy, "slice_variable": "country_iso3", "standardised_bwer": result.summary.get("bwer", "") if result else "", "warnings": result.summary.get("warnings", "") if result else ""})
    return alpha_rows, support_rows, missing_rows


def _social_spatial(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indicators = ["income_group"]
    output = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("country_iso3"):
            grouped.setdefault((str(row.get("country_iso3")), str(row.get("income_group", ""))), []).append(row)
    for income_group in sorted({key[1] for key in grouped if key[1]}):
        items = [item for key, rows_for_key in grouped.items() if key[1] == income_group for item in rows_for_key]
        output.append({"indicator": "income_group", "indicator_value": income_group, "support_count": len(items), "mean_risk": float(np.mean([row.get("risk") for row in items])) if items else "", "claim_scope": "exploratory association only; not causal"})
    if not output:
        output.append({"indicator": "income_group", "indicator_value": "", "support_count": 0, "mean_risk": "", "claim_scope": "unavailable_missing_indicator_support"})
    return output


def _write_figures(output: Path, metrics: Sequence[Mapping[str, Any]], bwer_rows: Sequence[Mapping[str, Any]], slice_rows: Sequence[Mapping[str, Any]], selective_bwer: Sequence[Mapping[str, Any]], conformal_bwer: Sequence[Mapping[str, Any]], conformal_slice: Sequence[Mapping[str, Any]], support_rows: Sequence[Mapping[str, Any]], social_rows: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
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

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    acc = _float(metrics[0].get("accuracy")) if metrics else 0
    country_bwer = next((_float(row.get("bwer")) for row in bwer_rows if row.get("slice_variable") == "country_iso3" and row.get("bwer_type") == "raw"), 0)
    ax.bar(["accuracy", "country Raw-BWER"], [acc, country_bwer], color=["#2F5DA8", "#A04D3A"])
    ax.set_ylim(0, 1)
    ax.set_title("AlphaEarth aggregate vs BWER")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "alphaearth_aggregate_vs_bwer")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    protocol = [row for row in bwer_rows if row.get("slice_variable") == "country_iso3" and row.get("bwer_type") == "raw"]
    ax.bar([str(row.get("split_protocol", row.get("split", ""))) or str(row.get("analysis_type")) for row in protocol[:4]], [_float(row.get("bwer")) for row in protocol[:4]], color="#5E6C84")
    ax.set_ylabel("Raw-BWER")
    ax.set_title("Spatial-block vs random protocol")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "alphaearth_spatial_block_vs_random_protocol")

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    std = [row for row in bwer_rows if row.get("bwer_type") == "standardised"][:12]
    ax.bar([str(row.get("slice_variable")) for row in std], [_float(row.get("bwer")) for row in std], color="#6B8E23")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Standardised-BWER")
    ax.set_title("Class-standardised BWER")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "alphaearth_class_standardised_bwer")

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    risks = sorted([row for row in slice_rows if row.get("slice_variable") in {"country_iso3", "region"} and row.get("is_valid_slice")], key=lambda row: _float(row.get("balanced_risk")), reverse=True)[:20]
    ax.bar([str(row.get("slice_value")) for row in risks], [_float(row.get("balanced_risk")) for row in risks], color="#8C6D31")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Risk")
    ax.set_title("Country/region tail risk")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "alphaearth_country_region_tail_risk")

    for name, rows, title in [
        ("alphaearth_selective_bwer", selective_bwer, "Selective-BWER"),
        ("alphaearth_conformal_bwer", conformal_bwer, "Conformal coverage-BWER"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        subset = [row for row in rows if row.get("slice_variable") == "country_iso3"][:9]
        ax.bar([str(row.get("coverage_target")) for row in subset], [_float(row.get("bwer")) for row in subset], color="#2F5DA8")
        ax.set_ylabel("BWER")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, name)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    subset = sorted(conformal_slice, key=lambda row: _float(row.get("average_set_size")), reverse=True)[:20]
    ax.bar([str(row.get("slice_value")) for row in subset], [_float(row.get("average_set_size")) for row in subset], color="#A04D3A")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Avg set size")
    ax.set_title("Conformal set size by slice")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "alphaearth_conformal_set_size_by_slice")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar([str(row.get("indicator_value")) for row in social_rows], [_float(row.get("mean_risk")) for row in social_rows], color="#5E6C84")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Mean risk")
    ax.set_title("Exploratory social-spatial summary")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "alphaearth_social_spatial_summary")

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    subset = sorted(support_rows, key=lambda row: _float(row.get("sample_count")), reverse=True)[:25]
    ax.bar([str(row.get("slice_value")) for row in subset], [_float(row.get("sample_count")) for row in subset], color="#3A7D44")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Samples")
    ax.set_title("Support diagnostics")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "alphaearth_support_diagnostics")
    return paths


def _unavailable_outputs(output: Path, gee_root: Path, reason: str) -> dict[str, Path]:
    artifacts = _artifact_paths(output)
    unavailable = [{"status": "unavailable", "reason": reason, "required_action": "Complete full GEE export or provide shard manifest."}]
    for key, path in artifacts.items():
        if key.endswith("_report"):
            continue
        if key == "figures":
            continue
        write_csv(path, unavailable)
    artifacts["alphaearth_full_report"].write_text(
        "# AlphaEarth full land-cover audit v1\n\n"
        f"Audit unavailable: {reason}.\n\n"
        "No empirical AlphaEarth formal claim is supported until the full export is available and passes support preflight.\n",
        encoding="utf-8",
    )
    return artifacts


def _artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "alphaearth_full_metrics": output / "alphaearth_full_metrics.csv",
        "alphaearth_full_predictions": output / "alphaearth_full_predictions.csv",
        "alphaearth_full_bwer_summary": output / "alphaearth_full_bwer_summary.csv",
        "alphaearth_full_standardised_bwer": output / "alphaearth_full_standardised_bwer.csv",
        "alphaearth_full_selective_risk_summary": output / "alphaearth_full_selective_risk_summary.csv",
        "alphaearth_full_selective_bwer": output / "alphaearth_full_selective_bwer.csv",
        "alphaearth_full_calibrated_threshold_bwer": output / "alphaearth_full_calibrated_threshold_bwer.csv",
        "alphaearth_full_conformal_coverage_summary": output / "alphaearth_full_conformal_coverage_summary.csv",
        "alphaearth_full_conformal_slice_coverage": output / "alphaearth_full_conformal_slice_coverage.csv",
        "alphaearth_full_conformal_bwer": output / "alphaearth_full_conformal_bwer.csv",
        "alphaearth_full_conformal_set_size_summary": output / "alphaearth_full_conformal_set_size_summary.csv",
        "alphaearth_full_slice_risk_summary": output / "alphaearth_full_slice_risk_summary.csv",
        "alphaearth_full_support_diagnostics": output / "alphaearth_full_support_diagnostics.csv",
        "alphaearth_full_alpha_sensitivity": output / "alphaearth_full_alpha_sensitivity.csv",
        "alphaearth_full_support_sensitivity": output / "alphaearth_full_support_sensitivity.csv",
        "alphaearth_full_missing_policy_sensitivity": output / "alphaearth_full_missing_policy_sensitivity.csv",
        "alphaearth_full_protocol_contrast": output / "alphaearth_full_protocol_contrast.csv",
        "alphaearth_full_social_spatial_association": output / "alphaearth_full_social_spatial_association.csv",
        "alphaearth_full_claim_support": output / "alphaearth_full_claim_support.csv",
        "alphaearth_full_caveats": output / "alphaearth_full_caveats.csv",
        "alphaearth_full_report": output / "alphaearth_full_report.md",
    }


def run_alphaearth_full_audit(input_csv: Path = DEFAULT_EXPORT, manifest_csv: Path = DEFAULT_MANIFEST, gee_root: Path = DEFAULT_GEE_ROOT, output_dir: Path = AUDIT_ROOT, unified_v4_dir: Path = UNIFIED_V4_ROOT, seed: int = 42) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    check_alphaearth_full_export_schema(input_csv, manifest_csv, gee_root)
    rows, _ = read_alphaearth_full_export(input_csv, manifest_csv)
    if not rows:
        artifacts = _unavailable_outputs(output, gee_root, "missing_full_export")
        _write_unified_v4_placeholder(unified_v4_dir, output, "missing_full_export")
        return artifacts
    rows = _prepare_rows(rows)
    train, calibration, test, random_rows = _split(rows)
    if not train or not test:
        artifacts = _unavailable_outputs(output, gee_root, "missing_train_or_test_split")
        _write_unified_v4_placeholder(unified_v4_dir, output, "missing_train_or_test_split")
        return artifacts
    x_train, y_train_raw = _arrays(train)
    x_cal, y_cal_raw = _arrays(calibration)
    x_test, y_test_raw = _arrays(test)
    classes = sorted(set(y_train_raw) | set(y_cal_raw) | set(y_test_raw), key=lambda value: int(value) if str(value).isdigit() else str(value))
    y_train = np.asarray([classes.index(label) for label in y_train_raw], dtype=int)
    x_eval = np.vstack([x_cal, x_test]) if len(calibration) else x_test
    model_name, probs_eval = _fit_predict_proba(x_train, y_train, x_eval, len(classes), seed)
    cal_probs = probs_eval[: len(calibration)] if len(calibration) else np.empty((0, len(classes)))
    test_probs = probs_eval[len(calibration) :] if len(calibration) else probs_eval
    cal_pred = _prediction_rows(calibration, y_cal_raw, cal_probs, classes, model_name, "spatial_block") if len(calibration) else []
    test_pred = _prediction_rows(test, y_test_raw, test_probs, classes, model_name, "spatial_block")
    metrics = [_metrics(test_pred, classes, model_name, "spatial_block", len(train), len(calibration))]
    bwer_rows, slice_rows, support_rows = _bwer_family(test_pred, model_name, "spatial_block", "baseline", min_support=20)
    raw_bwer = [row for row in bwer_rows if row.get("bwer_type") == "raw"]
    std_bwer = [row for row in bwer_rows if row.get("bwer_type") == "standardised"]
    selective_risk, selective_bwer, selective_slice = _selective_outputs(test_pred, model_name)
    calibrated_bwer = _calibrated_threshold_outputs(cal_pred, test_pred, model_name)
    conformal_cov, conformal_slice, conformal_bwer, conformal_set = _conformal_outputs(cal_pred, test_pred, model_name)
    alpha_sens, support_sens, missing_sens = _sensitivity_rows(test_pred, model_name)
    random_train, random_cal, random_test, _ = _split(random_rows)
    protocol_rows = [{"protocol": "spatial_block", **metrics[0]}]
    if random_train and random_test:
        xr_train, yr_train_raw = _arrays(random_train)
        xr_test, yr_test_raw = _arrays(random_test)
        y_random_train = np.asarray([classes.index(label) for label in yr_train_raw if label in classes], dtype=int)
        if len(y_random_train) == len(yr_train_raw):
            random_model, random_probs = _fit_predict_proba(xr_train, y_random_train, xr_test, len(classes), seed)
            random_pred = _prediction_rows(random_test, yr_test_raw, random_probs, classes, random_model, "random_sanity")
            protocol_rows.append({"protocol": "random_sanity", **_metrics(random_pred, classes, random_model, "random_sanity", len(random_train), len(random_cal))})
    social_rows = _social_spatial(test_pred)
    support_preflight = read_csv_rows(gee_root / "alphaearth_full_support_preflight.csv") if (gee_root / "alphaearth_full_support_preflight.csv").exists() else []
    caveats = [
        {"category": "label_source", "caveat": "ESA WorldCover is a map-label agreement target, not perfect ground truth."},
        {"category": "dynamic_world", "caveat": "Dynamic World is diagnostic/confidence support only when exported; it is not human truth."},
        {"category": "conformal_scope", "caveat": "Split conformal uses full probability vectors and is separate from calibrated threshold diagnostics."},
        {"category": "random_split", "caveat": "Random split protocol contrast is sanity only, not deployment evidence."},
        {"category": "support", "caveat": "Formal inclusion requires support targets; do not hide quota/support limitations."},
    ]
    strong_enough = len(rows) >= 100000 and len({row.get("country_iso3") for row in rows if row.get("country_iso3")}) >= 100
    claim_support = [
        {"claim": "AlphaEarth formal land-cover BWER audit is paper-ready", "support": "supported" if strong_enough else "blocked_by_support_or_quota", "evidence": f"n_rows={len(rows)}; n_countries={len({row.get('country_iso3') for row in rows if row.get('country_iso3')})}", "caveat": "Requires 100k+ samples and 100-120 countries for formal inclusion."},
        {"claim": "Aggregate land-cover performance can be compared against deployment-tail BWER", "support": "available_from_export" if test_pred else "unavailable", "evidence": "alphaearth_full_metrics.csv and alphaearth_full_bwer_summary.csv", "caveat": "ESA WorldCover is an agreement target."},
    ]
    artifacts = _artifact_paths(output)
    write_csv(artifacts["alphaearth_full_metrics"], metrics)
    write_csv(artifacts["alphaearth_full_predictions"], cal_pred + test_pred)
    write_csv(artifacts["alphaearth_full_bwer_summary"], raw_bwer)
    write_csv(artifacts["alphaearth_full_standardised_bwer"], std_bwer)
    write_csv(artifacts["alphaearth_full_selective_risk_summary"], selective_risk)
    write_csv(artifacts["alphaearth_full_selective_bwer"], selective_bwer)
    write_csv(artifacts["alphaearth_full_calibrated_threshold_bwer"], calibrated_bwer)
    write_csv(artifacts["alphaearth_full_conformal_coverage_summary"], conformal_cov)
    write_csv(artifacts["alphaearth_full_conformal_slice_coverage"], conformal_slice)
    write_csv(artifacts["alphaearth_full_conformal_bwer"], conformal_bwer)
    write_csv(artifacts["alphaearth_full_conformal_set_size_summary"], conformal_set)
    write_csv(artifacts["alphaearth_full_slice_risk_summary"], slice_rows + selective_slice)
    write_csv(artifacts["alphaearth_full_support_diagnostics"], support_preflight or support_rows)
    write_csv(artifacts["alphaearth_full_alpha_sensitivity"], alpha_sens)
    write_csv(artifacts["alphaearth_full_support_sensitivity"], support_sens)
    write_csv(artifacts["alphaearth_full_missing_policy_sensitivity"], missing_sens)
    write_csv(artifacts["alphaearth_full_protocol_contrast"], protocol_rows)
    write_csv(artifacts["alphaearth_full_social_spatial_association"], social_rows)
    write_csv(artifacts["alphaearth_full_claim_support"], claim_support)
    write_csv(artifacts["alphaearth_full_caveats"], caveats)
    artifacts["alphaearth_full_report"].write_text(
        "# AlphaEarth full land-cover audit v1\n\n"
        f"- Rows: {len(rows)}\n"
        f"- Countries: {len({row.get('country_iso3') for row in rows if row.get('country_iso3')})}\n"
        f"- Model: {model_name}\n"
        f"- Accuracy: {metrics[0]['accuracy']:.4f}\n"
        f"- Balanced accuracy: {metrics[0]['balanced_accuracy']:.4f}\n"
        f"- Macro-F1: {metrics[0]['macro_f1']:.4f}\n"
        f"- Formal paper inclusion status: {'strong enough' if strong_enough else 'blocked by support/quota scale'}\n\n"
        "ESA WorldCover is treated as map-label agreement, not perfect ground truth. Random split is sanity only.\n",
        encoding="utf-8",
    )
    artifacts.update({f"figure_{name}": path for name, path in _write_figures(output, metrics, bwer_rows, slice_rows, selective_bwer, conformal_bwer, conformal_slice, support_preflight or support_rows, social_rows).items()})
    _write_unified_v4(unified_v4_dir, output, metrics, claim_support, strong_enough)
    return artifacts


def _write_unified_v4_placeholder(output_dir: Path, audit_dir: Path, reason: str) -> None:
    output = ensure_dir(output_dir)
    rows = [{"experiment": "AlphaEarth/GEE land-cover", "status": "blocked", "reason": reason, "source_dir": str(audit_dir)}]
    for name in ["claim_support_table_v4.csv", "experiment_status_matrix_v4.csv", "metric_scope_and_caveat_matrix_v4.csv", "unified_results_narrative_table_v4.csv", "alphaearth_summary_v4.csv"]:
        write_csv(output / name, rows)
    (output / "paper_ready_main_findings_v4.md").write_text(f"# Paper-ready main findings v4\n\nAlphaEarth is not yet empirical because {reason}.\n", encoding="utf-8")
    _freeze_v4(output)


def _write_unified_v4(output_dir: Path, audit_dir: Path, metrics: Sequence[Mapping[str, Any]], claim_support: Sequence[Mapping[str, Any]], strong_enough: bool) -> None:
    output = ensure_dir(output_dir)
    write_csv(output / "alphaearth_summary_v4.csv", metrics)
    write_csv(output / "claim_support_table_v4.csv", claim_support)
    write_csv(output / "experiment_status_matrix_v4.csv", [{"experiment": "AlphaEarth/GEE land-cover", "status": "formal_ready" if strong_enough else "pilot_or_underpowered_full", "source_dir": str(audit_dir)}])
    write_csv(output / "metric_scope_and_caveat_matrix_v4.csv", [{"metric": "AlphaEarth BWER", "scope": "land-cover map-label agreement", "caveat": "ESA WorldCover is not perfect ground truth"}])
    write_csv(output / "unified_results_narrative_table_v4.csv", [{"experiment": "AlphaEarth/GEE", "accuracy": metrics[0].get("accuracy", ""), "macro_f1": metrics[0].get("macro_f1", ""), "formal_inclusion": strong_enough}])
    for md in ["paper_ready_main_findings_v4.md", "manuscript_outline_v4.md", "thesis_chapter_outline_v4.md"]:
        (output / md).write_text("# Unified paper package v4\n\nAlphaEarth is added as the fourth deployment axis. Interpret formal strength according to `claim_support_table_v4.csv`.\n", encoding="utf-8")
    _freeze_v4(output)


def _freeze_v4(output: Path) -> None:
    zip_path = output / "rsfm_bwer_paper_freeze_v4.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob("*"):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlphaEarth full land-cover deployment-slice audit.")
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gee-root", type=Path, default=DEFAULT_GEE_ROOT)
    parser.add_argument("--out", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--unified-v4-out", type=Path, default=UNIFIED_V4_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    artifacts = run_alphaearth_full_audit(args.input, args.manifest, args.gee_root, args.out, args.unified_v4_out, args.seed)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
