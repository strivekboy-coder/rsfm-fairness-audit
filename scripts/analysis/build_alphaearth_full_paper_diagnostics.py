from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


DEFAULT_AUDIT_ROOT = Path("outputs/alphaearth_landcover_audit_full_v2_150k")
DEFAULT_REFERENCE_AUDIT_ROOT = Path("outputs/alphaearth_landcover_audit_full_v1")
DEFAULT_UNIFIED_V4_ROOT = Path("outputs/unified_paper_package_v4")
TARGET_COVERAGES = ["0.7", "0.8", "0.9"]
KEY_SLICE_VARIABLES = ["country_iso3", "worldcover_class_name", "country_class", "region_class", "income_group"]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _str(value: Any) -> str:
    return "" if value is None else str(value)


def _read_optional(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def _scenario_rows(
    baseline_raw: Sequence[Mapping[str, Any]],
    standardised: Sequence[Mapping[str, Any]],
    selective: Sequence[Mapping[str, Any]],
    calibrated: Sequence[Mapping[str, Any]],
    conformal: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in baseline_raw:
        item = dict(row)
        item["scenario"] = "baseline"
        item["selector"] = "none"
        item["coverage_target"] = ""
        rows.append(item)
    for row in standardised:
        item = dict(row)
        item["scenario"] = "baseline_standardised"
        item["selector"] = "none"
        item["coverage_target"] = ""
        rows.append(item)
    for row in selective:
        item = dict(row)
        item["scenario"] = "topk_selective"
        item["selector"] = "confidence_topk"
        rows.append(item)
    for row in calibrated:
        item = dict(row)
        item["scenario"] = "calibrated_threshold"
        item["selector"] = "confidence_threshold_from_calibration"
        rows.append(item)
    for row in conformal:
        item = dict(row)
        item["scenario"] = "conformal_set_coverage"
        item["selector"] = "split_conformal_p_true"
        rows.append(item)
    return rows


def build_rank_divergence(audit_root: Path) -> list[dict[str, Any]]:
    metrics = _read_optional(audit_root / "alphaearth_full_metrics.csv")
    baseline = _read_optional(audit_root / "alphaearth_full_bwer_summary.csv")
    standardised = _read_optional(audit_root / "alphaearth_full_standardised_bwer.csv")
    selective = _read_optional(audit_root / "alphaearth_full_selective_bwer.csv")
    calibrated = _read_optional(audit_root / "alphaearth_full_calibrated_threshold_bwer.csv")
    conformal = _read_optional(audit_root / "alphaearth_full_conformal_bwer.csv")
    model = _str(metrics[0].get("model")) if metrics else "unknown"
    accuracy = metrics[0].get("accuracy", "") if metrics else ""
    all_rows = _scenario_rows(baseline, standardised, selective, calibrated, conformal)
    output = []
    for row in all_rows:
        if row.get("slice_variable") not in KEY_SLICE_VARIABLES:
            continue
        output.append(
            {
                "scenario": row.get("scenario", ""),
                "selector": row.get("selector", ""),
                "coverage_target": row.get("coverage_target", ""),
                "aggregate_best_model": model,
                "aggregate_best_accuracy": accuracy,
                "bwer_best_model": model,
                "bwer_best_value": row.get("bwer", ""),
                "bwer_slice_variable": row.get("slice_variable", ""),
                "bwer_type": row.get("bwer_type", ""),
                "tail_risk": row.get("tail_risk", ""),
                "mean_risk": row.get("mean_risk", ""),
                "worst_slice": row.get("worst_slice", ""),
                "worst_slice_risk": row.get("worst_slice_risk", ""),
                "rank_divergence_status": "not_applicable_single_model",
                "interpretation": "AlphaEarth full audit uses one model, so aggregate-best versus BWER-best model divergence is not estimable; the table records within-model tail-risk persistence across scenarios.",
            }
        )
    return output


def _group(rows: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_str(row.get(k)) for k in keys)
        if all(key):
            grouped[key].append(row)
    return grouped


def build_grassland_diagnostic(audit_root: Path) -> list[dict[str, Any]]:
    predictions = _read_optional(audit_root / "alphaearth_full_predictions.csv")
    grassland = [
        row
        for row in predictions
        if _str(row.get("class_label")) == "Grassland" or _str(row.get("worldcover_class_name")) == "Grassland" or _str(row.get("label")) == "30"
    ]
    output: list[dict[str, Any]] = []
    for keys in (["country_iso3"], ["region"], ["country_iso3", "predicted_class_name"]):
        for key, items in _group(grassland, keys).items():
            support = len(items)
            if support < 5:
                continue
            risk_values = [_float(item.get("risk")) for item in items]
            confidence_values = [_float(item.get("confidence")) for item in items]
            output.append(
                {
                    "diagnostic_scope": "|".join(keys),
                    "slice_value": "|".join(key),
                    "support_count": support,
                    "error_count": int(sum(value for value in risk_values if not math.isnan(value))),
                    "risk": float(np.mean(risk_values)) if risk_values else "",
                    "mean_confidence": float(np.mean(confidence_values)) if confidence_values else "",
                    "interpretation_scope": "mechanism diagnostic for Grassland tail risk; not causal evidence",
                }
            )
    return sorted(output, key=lambda row: (_float(row.get("risk"), -1), _float(row.get("support_count"), -1)), reverse=True)


def build_grassland_confusion(audit_root: Path) -> list[dict[str, Any]]:
    predictions = _read_optional(audit_root / "alphaearth_full_predictions.csv")
    grassland = [
        row
        for row in predictions
        if _str(row.get("class_label")) == "Grassland" or _str(row.get("worldcover_class_name")) == "Grassland" or _str(row.get("label")) == "30"
    ]
    output: list[dict[str, Any]] = []
    for keys in (["predicted_class_name"], ["region", "predicted_class_name"]):
        for key, items in _group(grassland, keys).items():
            support = len(items)
            output.append(
                {
                    "diagnostic_scope": "|".join(keys),
                    "slice_value": "|".join(key),
                    "support_count": support,
                    "share_of_grassland_samples": support / len(grassland) if grassland else "",
                    "error_share_within_scope": float(np.mean([_float(item.get("risk")) for item in items])) if items else "",
                    "interpretation_scope": "Grassland confusion mechanism diagnostic; WorldCover is a map-label agreement target.",
                }
            )
    return sorted(output, key=lambda row: (_float(row.get("support_count"), -1)), reverse=True)


def build_conformal_slice_gap(audit_root: Path) -> list[dict[str, Any]]:
    rows = _read_optional(audit_root / "alphaearth_full_conformal_slice_coverage.csv")
    output = []
    for row in rows:
        coverage = _float(row.get("coverage_target"))
        slice_coverage = _float(row.get("slice_coverage"))
        gap = coverage - slice_coverage if not math.isnan(coverage) and not math.isnan(slice_coverage) else float("nan")
        output.append(
            {
                **row,
                "coverage_gap_recomputed": "" if math.isnan(gap) else gap,
                "undercovered": "" if math.isnan(gap) else int(gap > 0),
                "interpretation": "Marginal conformal coverage does not guarantee slice-level coverage reliability." if not math.isnan(gap) and gap > 0 else "Slice coverage meets or exceeds nominal target.",
            }
        )
    return sorted(output, key=lambda row: _float(row.get("coverage_gap_recomputed"), -999), reverse=True)


def _metric_lookup(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return rows[0] if rows else {}


def _bwer_by_slice(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(_str(row.get("slice_variable")), _str(row.get("bwer_type"))): row for row in rows}


def build_scale_sensitivity(audit_root: Path, reference_audit_root: Path | None = None) -> list[dict[str, Any]]:
    roots = [("current", audit_root)]
    if reference_audit_root and reference_audit_root.exists():
        roots.insert(0, ("reference", reference_audit_root))
    output: list[dict[str, Any]] = []
    for label, root in roots:
        metrics = _metric_lookup(_read_optional(root / "alphaearth_full_metrics.csv"))
        bwer_rows = _read_optional(root / "alphaearth_full_bwer_summary.csv")
        std_rows = _read_optional(root / "alphaearth_full_standardised_bwer.csv")
        by_slice = _bwer_by_slice([*bwer_rows, *std_rows])
        for slice_variable in ["country_iso3", "worldcover_class_name", "country_class", "region_class", "income_group"]:
            raw = by_slice.get((slice_variable, "raw"), {})
            std = by_slice.get((slice_variable, "standardised"), {})
            output.append(
                {
                    "scale_label": label,
                    "audit_root": str(root),
                    "n_train": metrics.get("n_train", ""),
                    "n_calibration": metrics.get("n_calibration", ""),
                    "n_test": metrics.get("n_test", ""),
                    "accuracy": metrics.get("accuracy", ""),
                    "macro_f1": metrics.get("macro_f1", ""),
                    "slice_variable": slice_variable,
                    "raw_bwer": raw.get("bwer", ""),
                    "raw_tail_risk": raw.get("tail_risk", ""),
                    "standardised_bwer": std.get("bwer", ""),
                    "standardised_tail_risk": std.get("tail_risk", ""),
                    "interpretation": "Compare aggregate stability against tail-risk discovery as audit coverage scales.",
                }
            )
    if len(roots) == 1:
        output.append(
            {
                "scale_label": "reference_unavailable",
                "audit_root": str(reference_audit_root or ""),
                "interpretation": "Provide the 98k/v1 audit root to populate direct 98k-vs-156k scale comparison.",
            }
        )
    return output


def build_spatial_split_diagnostics(audit_root: Path) -> list[dict[str, Any]]:
    predictions = _read_optional(audit_root / "alphaearth_full_predictions.csv")
    output: list[dict[str, Any]] = []
    for keys in (["split"], ["split", "country_iso3"], ["split", "worldcover_class_name"], ["split", "spatial_block_id"]):
        for key, items in _group(predictions, keys).items():
            support = len(items)
            risk_values = [_float(item.get("risk")) for item in items]
            output.append(
                {
                    "diagnostic_scope": "|".join(keys),
                    "slice_value": "|".join(key),
                    "support_count": support,
                    "mean_risk": float(np.mean(risk_values)) if risk_values else "",
                }
            )
    block_by_split: dict[str, set[str]] = defaultdict(set)
    for row in predictions:
        block = _str(row.get("spatial_block_id"))
        split = _str(row.get("split"))
        if block and split:
            block_by_split[block].add(split)
    leaked = [block for block, splits in block_by_split.items() if len(splits) > 1]
    output.append(
        {
            "diagnostic_scope": "spatial_block_leakage_check",
            "slice_value": "blocks_present_in_multiple_splits",
            "support_count": len(leaked),
            "mean_risk": "",
            "note": "Expected zero for strict spatial-block split; nonzero means split metadata should be reviewed.",
        }
    )
    return output


def _bootstrap_ci(values: Sequence[float], seed: int = 42, n_boot: int = 500) -> tuple[float, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_boot):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        estimates.append(float(np.mean(sample)))
    estimates.sort()
    return estimates[int(0.025 * len(estimates))], estimates[int(0.975 * len(estimates)) - 1]


def build_bwer_ci_summary(audit_root: Path) -> list[dict[str, Any]]:
    slice_rows = _read_optional(audit_root / "alphaearth_full_slice_risk_summary.csv")
    output = []
    grouped = _group(
        [
            row
            for row in slice_rows
            if row.get("analysis_type") == "baseline" and row.get("bwer_type") in {"raw", "standardised"} and row.get("is_valid_slice") == "True"
        ],
        ["slice_variable", "bwer_type"],
    )
    for key, items in grouped.items():
        risks = [_float(row.get("balanced_risk") if key[1] == "standardised" else row.get("raw_risk")) for row in items]
        ci_low, ci_high = _bootstrap_ci(risks)
        output.append(
            {
                "slice_variable": key[0],
                "bwer_type": key[1],
                "n_valid_slices": len(items),
                "mean_slice_risk_bootstrap_ci_low": "" if math.isnan(ci_low) else ci_low,
                "mean_slice_risk_bootstrap_ci_high": "" if math.isnan(ci_high) else ci_high,
                "bootstrap_n": 500,
                "note": "Bootstrap CI is over slice risks, not a replacement for sample-level uncertainty.",
            }
        )
    return output


def build_dynamic_world_label_agreement_placeholder(audit_root: Path) -> list[dict[str, Any]]:
    predictions = _read_optional(audit_root / "alphaearth_full_predictions.csv")
    has_dw = bool(predictions and "dynamic_world_label" in predictions[0] and "dynamic_world_confidence" in predictions[0])
    if not has_dw:
        return [
            {
                "status": "unavailable",
                "reason": "dynamic_world_columns_not_exported",
                "required_action": "Re-export GEE table with Dynamic World label/confidence columns if label-agreement diagnostics are needed.",
                "rerun_scope": "GEE table export only; no AlphaEarth model retraining required after export.",
            }
        ]
    rows = []
    grouped = _group(predictions, ["dynamic_world_label"])
    for key, items in grouped.items():
        agreement = [int(_str(item.get("dynamic_world_label")) == _str(item.get("label"))) for item in items]
        confidence = [_float(item.get("dynamic_world_confidence")) for item in items]
        rows.append(
            {
                "dynamic_world_label": key[0],
                "support_count": len(items),
                "worldcover_agreement_rate": float(np.mean(agreement)) if agreement else "",
                "mean_dynamic_world_confidence": float(np.mean(confidence)) if confidence else "",
                "interpretation_scope": "map-label agreement diagnostic only",
            }
        )
    return rows


def update_reports(audit_root: Path, unified_v4_root: Path, rank_rows: Sequence[Mapping[str, Any]]) -> None:
    report_path = audit_root / "alphaearth_full_report.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# AlphaEarth full land-cover audit v1\n"
    adopted = [
        "rank divergence table for baseline, confidence top-k, calibrated threshold, and conformal-set scenarios",
        "Grassland mechanism diagnostic",
        "Grassland confusion and regional risk diagnostic",
        "conformal marginal-versus-slice coverage gap diagnostic",
        "audit scale-sensitivity comparison where a reference audit root is available",
        "spatial split diagnostics",
        "BWER CI/sensitivity postprocess summary",
        "Dynamic World label-agreement availability check",
    ]
    report_path.write_text(
        existing.rstrip()
        + "\n\n## Paper Diagnostics v1.1\n\n"
        + "\n".join(f"- Added {item}." for item in adopted)
        + "\n- Rank divergence across models is marked not applicable because AlphaEarth full v1 uses a single classifier.\n"
        + "- Dynamic World agreement remains unavailable unless those columns were exported.\n",
        encoding="utf-8",
    )
    ensure_dir(unified_v4_root)
    write_csv(unified_v4_root / "alphaearth_rank_divergence_v4.csv", rank_rows)


def build_diagnostics(audit_root: Path, unified_v4_root: Path, reference_audit_root: Path | None = DEFAULT_REFERENCE_AUDIT_ROOT) -> dict[str, Path]:
    output = ensure_dir(audit_root)
    paths = {
        "alphaearth_full_rank_divergence": output / "alphaearth_full_rank_divergence.csv",
        "alphaearth_grassland_mechanism_diagnostic": output / "alphaearth_grassland_mechanism_diagnostic.csv",
        "alphaearth_spatial_split_diagnostics": output / "alphaearth_spatial_split_diagnostics.csv",
        "alphaearth_bwer_ci_summary": output / "alphaearth_bwer_ci_summary.csv",
        "alphaearth_dynamic_world_label_agreement_diagnostic": output / "alphaearth_dynamic_world_label_agreement_diagnostic.csv",
        "alphaearth_grassland_confusion_diagnostic": output / "alphaearth_grassland_confusion_diagnostic.csv",
        "alphaearth_conformal_slice_gap_diagnostic": output / "alphaearth_conformal_slice_gap_diagnostic.csv",
        "alphaearth_scale_sensitivity_summary": output / "alphaearth_scale_sensitivity_summary.csv",
    }
    rank_rows = build_rank_divergence(output)
    write_csv(paths["alphaearth_full_rank_divergence"], rank_rows)
    write_csv(paths["alphaearth_grassland_mechanism_diagnostic"], build_grassland_diagnostic(output))
    write_csv(paths["alphaearth_spatial_split_diagnostics"], build_spatial_split_diagnostics(output))
    write_csv(paths["alphaearth_bwer_ci_summary"], build_bwer_ci_summary(output))
    write_csv(paths["alphaearth_dynamic_world_label_agreement_diagnostic"], build_dynamic_world_label_agreement_placeholder(output))
    write_csv(paths["alphaearth_grassland_confusion_diagnostic"], build_grassland_confusion(output))
    write_csv(paths["alphaearth_conformal_slice_gap_diagnostic"], build_conformal_slice_gap(output))
    write_csv(paths["alphaearth_scale_sensitivity_summary"], build_scale_sensitivity(output, reference_audit_root))
    update_reports(output, unified_v4_root, rank_rows)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AlphaEarth full paper diagnostics from completed audit outputs.")
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--reference-audit-root", type=Path, default=DEFAULT_REFERENCE_AUDIT_ROOT)
    parser.add_argument("--unified-v4-out", type=Path, default=DEFAULT_UNIFIED_V4_ROOT)
    args = parser.parse_args()
    paths = build_diagnostics(args.audit_root, args.unified_v4_out, args.reference_audit_root)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
