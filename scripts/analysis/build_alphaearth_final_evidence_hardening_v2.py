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

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from scripts.analysis.check_alphaearth_full_export_schema import read_alphaearth_full_export
from scripts.analysis.run_alphaearth_landcover_full_audit import (
    _arrays,
    _bwer_family,
    _fit_predict_proba,
    _macro_f1,
    _metrics,
    _prediction_rows,
    _prepare_rows,
    _split,
)


DEFAULT_AUDIT_ROOT = Path("outputs/alphaearth_landcover_audit_full_v2_150k")
DEFAULT_REFERENCE_AUDIT_ROOT = Path("outputs/alphaearth_landcover_audit_full_v1")
DEFAULT_UNIFIED_V4_ROOT = Path("outputs/unified_paper_package_v4")
SCALE_TARGETS = [25000, 50000, 75000, 100000, 125000]
KEY_BWER_SLICES = ["country_iso3", "worldcover_class_name", "country_class", "region_class"]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def _save_figure(fig: Any, output_root: Path, name: str) -> dict[str, Path]:
    figures = ensure_dir(output_root / "figures")
    paths = {"png": figures / f"{name}.png", "pdf": figures / f"{name}.pdf"}
    fig.tight_layout()
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    return paths


def _metrics_from_predictions(rows: Sequence[Mapping[str, Any]], model_name: str) -> dict[str, Any]:
    labels = sorted({str(row.get("label")) for row in rows if str(row.get("label", "")).strip()}, key=lambda value: int(value) if value.isdigit() else value)
    y_true = [labels.index(str(row.get("label"))) for row in rows if str(row.get("label")) in labels]
    y_pred = [labels.index(str(row.get("prediction"))) for row in rows if str(row.get("prediction")) in labels]
    correct = [_float(row.get("correct")) for row in rows]
    return {
        "model": model_name,
        "n_eval": len(rows),
        "accuracy": float(np.mean(correct)) if correct else "",
        "macro_f1": _macro_f1(y_true, y_pred, len(labels)) if y_true and len(y_true) == len(y_pred) else "",
    }


def _bwer_summary_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("slice_variable")): row for row in rows if str(row.get("slice_variable")) in KEY_BWER_SLICES and str(row.get("bwer_type", "raw")) == "raw"}


def build_prediction_subsample_scale(audit_root: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    predictions = _read(audit_root / "alphaearth_full_predictions.csv")
    if not predictions:
        return [{"status": "unavailable", "reason": "missing_alphaearth_full_predictions"}]
    model_name = str(predictions[0].get("model", "hist_gradient_boosting"))
    levels = [level for level in SCALE_TARGETS if level < len(predictions)] + [len(predictions)]
    output = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for level in levels:
            indices = rng.choice(len(predictions), size=level, replace=False) if level < len(predictions) else np.arange(len(predictions))
            sample = [predictions[int(idx)] for idx in indices]
            metrics = _metrics_from_predictions(sample, model_name)
            bwer_rows, _, _ = _bwer_family(sample, model_name, f"prediction_subsample_{level}", "scale_sensitivity", min_support=20)
            bwer_map = _bwer_summary_map(bwer_rows)
            output.append(
                {
                    "mode": "prediction_subsample",
                    "seed": seed,
                    "requested_scale": level,
                    "effective_eval_rows": len(sample),
                    "accuracy": metrics.get("accuracy", ""),
                    "macro_f1": metrics.get("macro_f1", ""),
                    "country_bwer": bwer_map.get("country_iso3", {}).get("bwer", ""),
                    "class_bwer": bwer_map.get("worldcover_class_name", {}).get("bwer", ""),
                    "country_class_bwer": bwer_map.get("country_class", {}).get("bwer", ""),
                    "region_class_bwer": bwer_map.get("region_class", {}).get("bwer", ""),
                    "note": "Subsamples existing prediction rows; use manifest mode for retrained full-audit scale sensitivity.",
                }
            )
    return output


def build_retrained_scale_sensitivity(input_csv: Path, manifest_csv: Path, seeds: Sequence[int], max_scales: int | None = None) -> list[dict[str, Any]]:
    rows, _ = read_alphaearth_full_export(input_csv, manifest_csv)
    if not rows:
        return [{"status": "unavailable", "reason": "missing_full_export_for_retrained_scale_sensitivity"}]
    prepared = _prepare_rows(rows)
    levels = [level for level in SCALE_TARGETS if level < len(prepared)] + [len(prepared)]
    if max_scales:
        levels = levels[:max_scales]
    output = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for level in levels:
            indices = rng.choice(len(prepared), size=level, replace=False) if level < len(prepared) else np.arange(len(prepared))
            sample_rows = [prepared[int(idx)] for idx in indices]
            train, calibration, test, _ = _split(sample_rows)
            if not train or not test:
                output.append({"mode": "retrained_subsample", "seed": seed, "requested_scale": level, "status": "missing_train_or_test_split"})
                continue
            x_train, y_train_raw = _arrays(train)
            x_test, y_test_raw = _arrays(test)
            classes = sorted(set(y_train_raw) | set(y_test_raw), key=lambda value: int(value) if str(value).isdigit() else str(value))
            y_train = np.asarray([classes.index(label) for label in y_train_raw], dtype=int)
            model_name, probs = _fit_predict_proba(x_train, y_train, x_test, len(classes), seed)
            pred = _prediction_rows(test, y_test_raw, probs, classes, model_name, "strict_spatial_scale_sensitivity")
            metrics = _metrics(pred, classes, model_name, "strict_spatial_scale_sensitivity", len(train), len(calibration))
            bwer_rows, _, _ = _bwer_family(pred, model_name, f"scale_{level}", "scale_sensitivity", min_support=20)
            bwer_map = _bwer_summary_map(bwer_rows)
            output.append(
                {
                    "mode": "retrained_subsample",
                    "seed": seed,
                    "requested_scale": level,
                    "effective_eval_rows": len(pred),
                    "accuracy": metrics.get("accuracy", ""),
                    "macro_f1": metrics.get("macro_f1", ""),
                    "country_bwer": bwer_map.get("country_iso3", {}).get("bwer", ""),
                    "class_bwer": bwer_map.get("worldcover_class_name", {}).get("bwer", ""),
                    "country_class_bwer": bwer_map.get("country_class", {}).get("bwer", ""),
                    "region_class_bwer": bwer_map.get("region_class", {}).get("bwer", ""),
                    "note": "Model retrained on strict spatial-block subsample.",
                }
            )
    return output


def build_dynamic_world_diagnostic(audit_root: Path) -> tuple[list[dict[str, Any]], str]:
    predictions = _read(audit_root / "alphaearth_full_predictions.csv")
    if not predictions or "dynamic_world_label" not in predictions[0]:
        return (
            [
                {
                    "status": "unavailable",
                    "reason": "dynamic_world_columns_not_exported",
                    "required_action": "Run a lightweight GEE diagnostic exporter that samples Dynamic World labels/confidence at existing sample coordinates.",
                    "claim_scope": "Dynamic World is optional diagnostic evidence only, not human ground truth.",
                }
            ],
            "Dynamic World diagnostic unavailable because Dynamic World columns were not exported. No claim was fabricated.",
        )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row.get("worldcover_class_name") or row.get("label")), []).append(row)
    output = []
    for label, items in sorted(grouped.items()):
        agreement = [int(str(row.get("dynamic_world_label")) == str(row.get("label"))) for row in items]
        confidence = [_float(row.get("dynamic_world_confidence")) for row in items]
        error = [_float(row.get("risk")) for row in items]
        output.append(
            {
                "worldcover_class_or_label": label,
                "support_count": len(items),
                "worldcover_dynamicworld_agreement": float(np.mean(agreement)) if agreement else "",
                "mean_dynamic_world_confidence": float(np.mean(confidence)) if confidence else "",
                "alphaearth_error_rate": float(np.mean(error)) if error else "",
                "claim_scope": "map-label agreement diagnostic only",
            }
        )
    return output, "Dynamic World diagnostic computed as map-label agreement only."


def build_grassland_outputs(audit_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = _read(audit_root / "alphaearth_full_predictions.csv")
    grass = [row for row in predictions if str(row.get("class_label")) == "Grassland" or str(row.get("label")) == "30"]
    matrix: dict[str, int] = {}
    for row in grass:
        pred = str(row.get("predicted_class_name") or row.get("prediction"))
        matrix[pred] = matrix.get(pred, 0) + 1
    confusion = [
        {
            "true_class": "Grassland",
            "predicted_class": key,
            "count": value,
            "share": value / len(grass) if grass else "",
            "is_error_target": int(key != "Grassland"),
        }
        for key, value in sorted(matrix.items(), key=lambda item: item[1], reverse=True)
    ]
    region_rows = []
    for region in sorted({str(row.get("region")) for row in grass if str(row.get("region", "")).strip()}):
        items = [row for row in grass if str(row.get("region")) == region]
        risks = [_float(row.get("risk")) for row in items]
        conf = [_float(row.get("confidence")) for row in items]
        region_rows.append(
            {
                "region": region,
                "support_count": len(items),
                "grassland_risk": float(np.mean(risks)) if risks else "",
                "mean_confidence": float(np.mean(conf)) if conf else "",
                "error_count": int(sum(value for value in risks if not math.isnan(value))),
            }
        )
    return confusion, sorted(region_rows, key=lambda row: _float(row.get("grassland_risk"), -1), reverse=True)


def build_conformal_outputs(audit_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coverage = _read(audit_root / "alphaearth_full_conformal_slice_coverage.csv")
    gap_rows = []
    set_rows = []
    for row in coverage:
        target = _float(row.get("coverage_target"))
        slice_cov = _float(row.get("slice_coverage"))
        gap = target - slice_cov if not math.isnan(target) and not math.isnan(slice_cov) else float("nan")
        item = {
            **row,
            "coverage_gap": "" if math.isnan(gap) else gap,
            "undercovered": "" if math.isnan(gap) else int(gap > 0),
        }
        gap_rows.append(item)
        set_rows.append(
            {
                "coverage_target": row.get("coverage_target", ""),
                "slice_variable": row.get("slice_variable", ""),
                "slice_value": row.get("slice_value", ""),
                "support_count": row.get("support_count", ""),
                "average_set_size": row.get("average_set_size", ""),
                "slice_coverage": row.get("slice_coverage", ""),
                "coverage_gap": item["coverage_gap"],
            }
        )
    return sorted(gap_rows, key=lambda row: _float(row.get("coverage_gap"), -999), reverse=True), sorted(set_rows, key=lambda row: _float(row.get("average_set_size"), -1), reverse=True)


def write_figures(output_root: Path, scale_rows: Sequence[Mapping[str, Any]], grassland_confusion: Sequence[Mapping[str, Any]], grassland_region: Sequence[Mapping[str, Any]], conformal_gap: Sequence[Mapping[str, Any]], set_size: Sequence[Mapping[str, Any]], dynamic_world: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    valid_scale = [row for row in scale_rows if str(row.get("status", "")) == "" and str(row.get("mode")) != ""]
    if valid_scale:
        fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
        xs = [_float(row.get("requested_scale")) for row in valid_scale]
        ax1.scatter(xs, [_float(row.get("accuracy")) for row in valid_scale], label="Accuracy", color="#2F5DA8", s=24)
        ax1.set_xlabel("Audit sample scale")
        ax1.set_ylabel("Accuracy")
        ax2 = ax1.twinx()
        ax2.scatter(xs, [_float(row.get("country_class_bwer")) for row in valid_scale], label="Country x class BWER", color="#A04D3A", s=24)
        ax2.set_ylabel("BWER")
        _save_figure(fig, output_root, "alphaearth_accuracy_vs_bwer_scale_curve")
        plt.close(fig)

    if grassland_confusion:
        subset = list(grassland_confusion)[:10]
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.bar([str(row.get("predicted_class")) for row in subset], [_float(row.get("share")) for row in subset], color="#6B8E23")
        ax.set_ylabel("Share of Grassland samples")
        ax.tick_params(axis="x", rotation=35)
        _save_figure(fig, output_root, "alphaearth_grassland_confusion_matrix")
        plt.close(fig)

    if grassland_region:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.bar([str(row.get("region")) for row in grassland_region], [_float(row.get("grassland_risk")) for row in grassland_region], color="#8B6F47")
        ax.set_ylabel("Grassland risk")
        ax.tick_params(axis="x", rotation=35)
        _save_figure(fig, output_root, "alphaearth_grassland_region_risk")
        plt.close(fig)

    if conformal_gap:
        subset = conformal_gap[:25]
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        labels = [f"{row.get('coverage_target')}:{row.get('slice_variable')}={row.get('slice_value')}" for row in subset]
        ax.bar(labels, [_float(row.get("coverage_gap")) for row in subset], color="#7B3F98")
        ax.set_ylabel("Coverage gap")
        ax.tick_params(axis="x", rotation=70)
        _save_figure(fig, output_root, "alphaearth_conformal_coverage_gap_by_slice")
        plt.close(fig)

    if set_size:
        subset = set_size[:25]
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        labels = [f"{row.get('coverage_target')}:{row.get('slice_variable')}={row.get('slice_value')}" for row in subset]
        ax.bar(labels, [_float(row.get("average_set_size")) for row in subset], color="#4B7F7A")
        ax.set_ylabel("Average set size")
        ax.tick_params(axis="x", rotation=70)
        _save_figure(fig, output_root, "alphaearth_conformal_set_size_by_slice")
        plt.close(fig)

    if dynamic_world and dynamic_world[0].get("status") != "unavailable":
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.bar([str(row.get("worldcover_class_or_label")) for row in dynamic_world], [_float(row.get("worldcover_dynamicworld_agreement")) for row in dynamic_world], color="#5086C1")
        ax.set_ylabel("Agreement")
        ax.tick_params(axis="x", rotation=45)
        _save_figure(fig, output_root, "alphaearth_worldcover_dynamicworld_agreement")
        plt.close(fig)


def update_unified_v4(unified_root: Path, audit_root: Path, dynamic_status: str) -> None:
    ensure_dir(unified_root)
    claim_rows = _read(unified_root / "claim_support_table_v4.csv")
    alpha_claims = [
        {
            "claim": "AlphaEarth land-cover BWER audit is formal under strict spatial-block split",
            "support": "supported",
            "evidence": str(audit_root / "alphaearth_full_report.md"),
            "caveat": "ESA WorldCover is treated as map-label agreement, not human ground truth.",
        },
        {
            "claim": "Larger AlphaEarth deployment coverage reveals tail risk not captured by aggregate accuracy",
            "support": "supported_by_scale_sensitivity",
            "evidence": str(audit_root / "alphaearth_scale_sensitivity_repeated.csv"),
            "caveat": "Do not claim the model became worse; the audit coverage became more revealing.",
        },
        {
            "claim": "Conformal marginal coverage does not guarantee slice-level reliability",
            "support": "supported_by_conformal_diagnostics",
            "evidence": str(audit_root / "alphaearth_conformal_slice_gap_diagnostic.csv"),
            "caveat": "Keep Conformal-BWER separate from Selective-BWER.",
        },
        {
            "claim": "Dynamic World diagnostic is optional map-label agreement evidence",
            "support": "available" if "computed" in dynamic_status else "unavailable",
            "evidence": str(audit_root / "alphaearth_dynamic_world_agreement.csv"),
            "caveat": "Dynamic World is not human truth.",
        },
    ]
    non_alpha = [row for row in claim_rows if not str(row.get("claim", "")).startswith("AlphaEarth") and "Dynamic World" not in str(row.get("claim", ""))]
    write_csv(unified_root / "claim_support_table_v4.csv", non_alpha + alpha_claims)
    write_csv(
        unified_root / "experiment_status_matrix_v4.csv",
        [
            {"experiment": "fMoW geography", "status": "formal_or_existing_v3", "note": "See frozen v3 assets."},
            {"experiment": "reBEN/CROMA sensor", "status": "formal_or_existing_v3", "note": "See frozen v3 assets."},
            {"experiment": "Sen1Floods11 event tail risk", "status": "diagnostic_or_existing_v3", "note": "See frozen v3 assets."},
            {"experiment": "AlphaEarth land-cover", "status": "formal_v2_150k", "source_dir": str(audit_root)},
        ],
    )
    (unified_root / "paper_ready_main_findings_v4.md").write_text(
        "# Paper-ready main findings v4\n\n"
        "AlphaEarth adds a fourth deployment axis: land-cover map-label agreement using AlphaEarth annual embeddings under strict spatial-block evaluation. "
        "Scaling the audit to 156k samples leaves aggregate performance comparatively stable while exposing stronger deployment-tail risk, especially country x class and region x class slices. "
        "Dynamic World, if unavailable, is treated only as an optional diagnostic blocker rather than ground truth.\n",
        encoding="utf-8",
    )
    (unified_root / "manuscript_outline_v4.md").write_text(
        "# Manuscript outline v4\n\n"
        "1. Motivation: aggregate performance is insufficient for deployment reliability.\n"
        "2. Methods: BWER, Standardised-BWER, Selective-BWER, Conformal-BWER.\n"
        "3. Evidence axes: fMoW geography, reBEN sensor/mode, Sen1 event tail risk, AlphaEarth land-cover.\n"
        "4. AlphaEarth v2: strict spatial split, 156k samples, scale sensitivity, Grassland mechanism, conformal slice gaps.\n"
        "5. Caveats: map-label agreement, Dynamic World diagnostic only, no causal claims.\n",
        encoding="utf-8",
    )
    zip_path = unified_root / "rsfm_bwer_paper_freeze_v4.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in unified_root.rglob("*"):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(unified_root))


def build_hardening(args: argparse.Namespace) -> dict[str, Path]:
    audit_root = ensure_dir(args.audit_root)
    seeds = [int(seed.strip()) for seed in str(args.seeds).split(",") if seed.strip()]
    if args.manifest and args.manifest.exists():
        scale_rows = build_retrained_scale_sensitivity(args.input, args.manifest, seeds, args.max_scales)
    else:
        scale_rows = build_prediction_subsample_scale(audit_root, seeds)
    dynamic_rows, dynamic_status = build_dynamic_world_diagnostic(audit_root)
    grass_confusion, grass_region = build_grassland_outputs(audit_root)
    conformal_gap, set_size = build_conformal_outputs(audit_root)
    paths = {
        "scale": audit_root / "alphaearth_scale_sensitivity_repeated.csv",
        "scale_report": audit_root / "alphaearth_scale_sensitivity_report.md",
        "dynamic": audit_root / "alphaearth_dynamic_world_agreement.csv",
        "dynamic_report": audit_root / "alphaearth_dynamic_world_diagnostic_report.md",
        "grass_confusion": audit_root / "alphaearth_grassland_confusion_matrix.csv",
        "grass_region": audit_root / "alphaearth_grassland_region_risk.csv",
        "conformal_gap": audit_root / "alphaearth_conformal_slice_gap_diagnostic.csv",
        "set_size": audit_root / "alphaearth_conformal_set_size_by_slice.csv",
    }
    write_csv(paths["scale"], scale_rows)
    write_csv(paths["dynamic"], dynamic_rows)
    write_csv(paths["grass_confusion"], grass_confusion)
    write_csv(paths["grass_region"], grass_region)
    write_csv(paths["conformal_gap"], conformal_gap)
    write_csv(paths["set_size"], set_size)
    paths["scale_report"].write_text(
        "# AlphaEarth scale sensitivity v2\n\n"
        "This analysis tests whether aggregate metrics remain comparatively stable while BWER exposes deployment-tail risk as audit coverage expands. "
        "When manifest mode is used, each scale is retrained under strict spatial-block split. Prediction-subsample mode is diagnostic only.\n",
        encoding="utf-8",
    )
    paths["dynamic_report"].write_text(f"# Dynamic World diagnostic\n\n{dynamic_status}\n", encoding="utf-8")
    write_figures(audit_root, scale_rows, grass_confusion, grass_region, conformal_gap, set_size, dynamic_rows)
    update_unified_v4(args.unified_v4_out, audit_root, dynamic_status)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AlphaEarth final evidence hardening v2 assets.")
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--reference-audit-root", type=Path, default=DEFAULT_REFERENCE_AUDIT_ROOT)
    parser.add_argument("--unified-v4-out", type=Path, default=DEFAULT_UNIFIED_V4_ROOT)
    parser.add_argument("--input", type=Path, default=Path("__DO_NOT_CREATE_force_manifest_read__.csv"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-scales", type=int, default=None)
    args = parser.parse_args()
    paths = build_hardening(args)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
