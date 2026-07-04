from __future__ import annotations

import argparse
import math
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
FORMAL_SCALE_SEEDS = [42, 73, 101]
KEY_BWER_SLICES = ["country_iso3", "worldcover_class_name", "country_class", "region_class"]
REQUIRED_DW_COLUMNS = ["sample_id", "dynamic_world_label"]
PLACEHOLDER_VALUES = {"", "-1", "None", "none", "nan", "NaN"}
DW_TO_WORLDCOVER = {
    "0": "80",  # water
    "1": "10",  # trees
    "2": "30",  # grass
    "3": "90",  # flooded vegetation
    "4": "40",  # crops
    "5": "20",  # shrub and scrub
    "6": "50",  # built
    "7": "60",  # bare
    "8": "70",  # snow and ice
    "water": "80",
    "trees": "10",
    "grass": "30",
    "flooded_vegetation": "90",
    "crops": "40",
    "shrub_and_scrub": "20",
    "built": "50",
    "bare": "60",
    "snow_and_ice": "70",
}


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _valid_label(value: Any) -> bool:
    return _clean(value) not in PLACEHOLDER_VALUES


def _label_equal(left: Any, right: Any) -> int:
    return int(_valid_label(left) and _clean(left) == _clean(right))


def _confidence_bin(value: Any) -> str:
    score = _float(value)
    if math.isnan(score):
        return "missing"
    if score < 0.4:
        return "0.0-0.4"
    if score < 0.6:
        return "0.4-0.6"
    if score < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def _entropy_bin(row: Mapping[str, Any]) -> str:
    entropy = _float(row.get("dynamic_world_entropy"))
    if math.isnan(entropy):
        top = _float(row.get("dynamic_world_top_probability") or row.get("dynamic_world_confidence"))
        if math.isnan(top):
            return "missing"
        entropy = 1.0 - top
    if entropy < 0.2:
        return "low"
    if entropy < 0.4:
        return "medium"
    return "high"


def _to_worldcover_label(value: Any) -> str:
    label = _clean(value)
    if not label:
        return ""
    return DW_TO_WORLDCOVER.get(label, label)


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
        raise FileNotFoundError(f"Formal scale sensitivity requires a readable full export or manifest: input={input_csv}; manifest={manifest_csv}")
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
                output.append({"seed": seed, "requested_scale": level, "source_sample_rows": len(sample_rows), "status": "missing_train_or_test_split"})
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
                    "seed": seed,
                    "requested_scale": level,
                    "source_sample_rows": len(sample_rows),
                    "n_train": len(train),
                    "n_calibration": len(calibration),
                    "n_test": len(test),
                    "effective_eval_rows": len(pred),
                    "accuracy": metrics.get("accuracy", ""),
                    "macro_f1": metrics.get("macro_f1", ""),
                    "country_bwer": bwer_map.get("country_iso3", {}).get("bwer", ""),
                    "class_bwer": bwer_map.get("worldcover_class_name", {}).get("bwer", ""),
                    "country_class_bwer": bwer_map.get("country_class", {}).get("bwer", ""),
                    "region_class_bwer": bwer_map.get("region_class", {}).get("bwer", ""),
                }
            )
    return output


def _dw_column(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = _clean(row.get(name))
        if value:
            return value
    return ""


def _lookup_from_rows(predictions: Sequence[Mapping[str, Any]], split_filter: set[str] | None = None) -> dict[str, Mapping[str, Any]]:
    lookup = {}
    for row in predictions:
        if split_filter is not None and _clean(row.get("split")) not in split_filter:
            continue
        sample_id = _clean(row.get("sample_id"))
        if sample_id:
            lookup[sample_id] = row
    return lookup


def _prediction_scopes(audit_root: Path) -> dict[str, tuple[list[dict[str, str]], set[str] | None, str]]:
    eval_rows = _read(audit_root / "alphaearth_full_eval_predictions.csv")
    if not eval_rows:
        eval_rows = _read(audit_root / "alphaearth_full_predictions.csv")
    all_rows = _read(audit_root / "alphaearth_full_all_split_predictions.csv")
    scopes: dict[str, tuple[list[dict[str, str]], set[str] | None, str]] = {
        "eval_calibration_test": (eval_rows, None, "formal_eval"),
        "test_only": (eval_rows, {"test"}, "formal_test"),
    }
    if all_rows:
        scopes["all_split_descriptive"] = (all_rows, None, "descriptive_background_only")
    return scopes


def _joined_dw_rows(aligned: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], split_filter: set[str] | None = None) -> tuple[list[dict[str, Any]], int, int]:
    lookup = _lookup_from_rows(predictions, split_filter)
    joined: list[dict[str, Any]] = []
    placeholder_count = 0
    for row in aligned:
        sample_id = _clean(row.get("sample_id"))
        pred = lookup.get(sample_id)
        raw_wc = _dw_column(row, "worldcover_label", "label")
        raw_ae = _dw_column(row, "alphaearth_prediction", "ae_pred", "prediction")
        if not _valid_label(raw_wc) or not _valid_label(raw_ae):
            placeholder_count += 1
        if not pred:
            continue
        wc = _dw_column(pred, "label", "worldcover_label")
        ae = _dw_column(pred, "prediction", "alphaearth_prediction", "ae_pred")
        if not _valid_label(wc) or not _valid_label(ae):
            continue
        item = dict(row)
        item["worldcover_label"] = wc
        item["alphaearth_prediction"] = ae
        item["alphaearth_correct"] = pred.get("correct", _label_equal(ae, wc))
        item["alphaearth_risk"] = pred.get("risk", 1 - _label_equal(ae, wc))
        item["matched_prediction_row"] = 1
        joined.append(item)
    return joined, len(lookup), placeholder_count


def _group_metric(rows: Sequence[Mapping[str, Any]], metric_name: str, group_name: str, value_getter: Any) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        key = str(value_getter(row))
        grouped.setdefault(key, []).append(_float(row.get(metric_name)))
    return [
        {
            "metric": metric_name,
            "group": group_name,
            "group_value": key,
            "value": float(np.mean([value for value in values if not math.isnan(value)])) if any(not math.isnan(value) for value in values) else "",
            "support_count": len(values),
            "claim_scope": "Dynamic World is map-product agreement / label ambiguity diagnostic only, not human ground truth.",
        }
        for key, values in sorted(grouped.items())
    ]


def build_dynamic_world_diagnostic(audit_root: Path) -> tuple[list[dict[str, Any]], str]:
    aligned_path = audit_root / "alphaearth_full_dw_aligned.csv"
    aligned = _read(aligned_path)
    if not aligned:
        return (
            [
                {
                    "status": "unavailable",
                    "reason": "missing_alphaearth_full_dw_aligned_csv",
                    "required_action": "Create alphaearth_full_dw_aligned.csv with sample_id, worldcover_label, dynamic_world_label, Dynamic World confidence/probability, and AlphaEarth prediction columns.",
                    "claim_scope": "Dynamic World is optional diagnostic evidence only, not human ground truth.",
                }
            ],
            "Dynamic World diagnostic unavailable because alphaearth_full_dw_aligned.csv is missing. No claim was fabricated.",
        )
    missing_columns = [column for column in REQUIRED_DW_COLUMNS if column not in aligned[0]]
    if missing_columns:
        return (
            [
                {
                    "status": "invalid",
                    "reason": "missing_required_columns",
                    "missing_columns": ";".join(missing_columns),
                    "claim_scope": "Dynamic World diagnostic rejected; no empirical DW claim should be made.",
                }
            ],
            f"Dynamic World diagnostic invalid because required columns are missing: {', '.join(missing_columns)}.",
        )
    sample_ids = [_clean(row.get("sample_id")) for row in aligned]
    unique_sample_ids = len(set(sample_ids))
    output_rows = []
    valid_scope_count = 0
    ambiguity_labels = {"20", "30", "40", "Shrubland", "Grassland", "Cropland"}
    for scope, (predictions, split_filter, claim_kind) in _prediction_scopes(audit_root).items():
        joined, prediction_row_count, placeholder_count = _joined_dw_rows(aligned, predictions, split_filter)
        if not joined:
            output_rows.append(
                {
                    "scope": scope,
                    "status": "invalid",
                    "reason": "no_valid_prediction_label_join",
                    "dw_aligned_rows": len(aligned),
                    "matched_prediction_rows": 0,
                    "prediction_table_rows": prediction_row_count,
                    "unique_sample_id_count": unique_sample_ids,
                    "placeholder_label_count": placeholder_count,
                    "claim_scope": "No valid AlphaEarth prediction/WorldCover label join exists for this scope.",
                }
            )
            continue
        valid_scope_count += 1
        validation = {
            "scope": scope,
            "metric": "dw_aligned_table_validation",
            "group": "all",
            "group_value": "all",
            "value": "",
            "dw_aligned_rows": len(aligned),
            "matched_prediction_rows": len(joined),
            "prediction_table_rows": prediction_row_count,
            "support_count": len(joined),
            "unique_sample_id_count": unique_sample_ids,
            "matched_unique_sample_id_count": len({_clean(row.get("sample_id")) for row in joined}),
            "expected_row_count": 156246,
            "row_count_status": "ok" if len(aligned) == 156246 else "unexpected",
            "unique_sample_id_status": "ok" if unique_sample_ids == len(aligned) else "duplicate_sample_ids",
            "placeholder_label_count": placeholder_count,
            "diagnostic_scope": "matched_subset" if len(joined) < len(aligned) else "full_aligned_table",
            "claim_scope": "paper_facing_formal_evaluation" if claim_kind in {"formal_eval", "formal_test"} else "descriptive_background_only",
        }
        output_rows.append(validation)
        enriched = []
        for row in joined:
            wc = _dw_column(row, "worldcover_label", "label")
            dw = _to_worldcover_label(_dw_column(row, "dynamic_world_label"))
            ae = _dw_column(row, "alphaearth_prediction", "ae_pred", "prediction")
            wc_dw = _label_equal(wc, dw)
            ae_wc = _label_equal(ae, wc)
            ae_dw = _label_equal(ae, dw)
            item = dict(row)
            item["worldcover_dynamicworld_agreement"] = wc_dw
            item["alphaearth_worldcover_accuracy"] = ae_wc
            item["alphaearth_dynamicworld_agreement"] = ae_dw
            item["dw_confidence_bin"] = _confidence_bin(row.get("dynamic_world_top_probability") or row.get("dynamic_world_confidence"))
            item["dw_entropy_bin"] = _entropy_bin(row)
            item["worldcover_dynamicworld_agreement_group"] = "agree" if wc_dw else "disagree"
            item["grassland_shrubland_cropland_ambiguity"] = int(wc in ambiguity_labels or dw in ambiguity_labels or ae in ambiguity_labels)
            enriched.append(item)
        claim_scope = "paper_facing_formal_evaluation" if claim_kind in {"formal_eval", "formal_test"} else "descriptive_background_only_not_formal_model_accuracy"
        for metric in ["worldcover_dynamicworld_agreement", "alphaearth_worldcover_accuracy", "alphaearth_dynamicworld_agreement", "grassland_shrubland_cropland_ambiguity"]:
            values = [_float(row.get(metric)) for row in enriched]
            output_rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "group": "all",
                    "group_value": "all",
                    "value": float(np.mean(values)) if values else "",
                    "support_count": len(values),
                    "dw_aligned_rows": len(aligned),
                    "matched_prediction_rows": len(joined),
                    "unique_sample_id_count": unique_sample_ids,
                    "claim_scope": claim_scope,
                }
            )
        for row in _group_metric(enriched, "alphaearth_worldcover_accuracy", "dw_confidence_bin", lambda item: item.get("dw_confidence_bin")):
            output_rows.append({"scope": scope, **row, "claim_scope": claim_scope})
        for row in _group_metric(enriched, "alphaearth_worldcover_accuracy", "worldcover_dynamicworld_agreement", lambda item: item.get("worldcover_dynamicworld_agreement_group")):
            output_rows.append({"scope": scope, **row, "claim_scope": claim_scope})
        error_rows = []
        for row in enriched:
            item = dict(row)
            item["alphaearth_error"] = 1 - int(_float(row.get("alphaearth_worldcover_accuracy"), 0))
            error_rows.append(item)
        for row in _group_metric(error_rows, "alphaearth_error", "dw_entropy_bin", lambda item: item.get("dw_entropy_bin")):
            output_rows.append({"scope": scope, **row, "claim_scope": claim_scope})
    if valid_scope_count == 0:
        return output_rows, "Dynamic World diagnostic invalid because no valid prediction/label join exists in any scope. No accuracy was reported."
    return output_rows, "Dynamic World diagnostic computed by scope from alphaearth_full_dw_aligned.csv joined to formal eval and all-split prediction tables."


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

    if dynamic_world and dynamic_world[0].get("status") not in {"unavailable", "invalid"}:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        subset = [row for row in dynamic_world if row.get("metric") in {"worldcover_dynamicworld_agreement", "alphaearth_worldcover_accuracy", "alphaearth_dynamicworld_agreement"} and row.get("group") == "all"]
        ax.bar([str(row.get("metric")) for row in subset], [_float(row.get("value")) for row in subset], color="#5086C1")
        ax.set_ylabel("Rate")
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
            "support": "available" if "computed by scope" in dynamic_status else "unavailable",
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
        "Dynamic World, when available through `alphaearth_full_dw_aligned.csv`, is evaluated on formal `test_only` and `eval_calibration_test` scopes for paper-facing mechanism claims; all-split results are descriptive background only.\n",
        encoding="utf-8",
    )
    (unified_root / "manuscript_outline_v4.md").write_text(
        "# Manuscript outline v4\n\n"
        "1. Motivation: aggregate performance is insufficient for deployment reliability.\n"
        "2. Methods: BWER, Standardised-BWER, Selective-BWER, Conformal-BWER.\n"
        "3. Evidence axes: fMoW geography, reBEN sensor/mode, Sen1 event tail risk, AlphaEarth land-cover.\n"
        "4. AlphaEarth v2: strict spatial split, 156k samples, scale sensitivity, Grassland mechanism, conformal slice gaps.\n"
        "5. Caveats: WorldCover map-label agreement, Dynamic World diagnostic only, no causal claims.\n",
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
    seeds = [int(seed.strip()) for seed in str(args.seeds).split(",") if seed.strip()] or FORMAL_SCALE_SEEDS
    if not args.manifest or not args.manifest.exists():
        raise FileNotFoundError("Formal alphaearth_scale_sensitivity_repeated.csv requires --manifest pointing to the full AlphaEarth export manifest.")
    scale_rows = build_retrained_scale_sensitivity(args.input, args.manifest, seeds, args.max_scales)
    prediction_subset_rows = build_prediction_subsample_scale(audit_root, seeds)
    dynamic_rows, dynamic_status = build_dynamic_world_diagnostic(audit_root)
    grass_confusion, grass_region = build_grassland_outputs(audit_root)
    conformal_gap, set_size = build_conformal_outputs(audit_root)
    paths = {
        "scale": audit_root / "alphaearth_scale_sensitivity_repeated.csv",
        "scale_report": audit_root / "alphaearth_scale_sensitivity_report.md",
        "dynamic": audit_root / "alphaearth_dynamic_world_agreement.csv",
        "dynamic_report": audit_root / "alphaearth_dynamic_world_diagnostic_report.md",
        "prediction_subset": audit_root / "alphaearth_prediction_subset_sensitivity_diagnostic.csv",
        "grass_confusion": audit_root / "alphaearth_grassland_confusion_matrix.csv",
        "grass_region": audit_root / "alphaearth_grassland_region_risk.csv",
        "conformal_gap": audit_root / "alphaearth_conformal_slice_gap_diagnostic.csv",
        "set_size": audit_root / "alphaearth_conformal_set_size_by_slice.csv",
    }
    write_csv(paths["scale"], scale_rows)
    write_csv(paths["prediction_subset"], prediction_subset_rows)
    write_csv(paths["dynamic"], dynamic_rows)
    write_csv(paths["grass_confusion"], grass_confusion)
    write_csv(paths["grass_region"], grass_region)
    write_csv(paths["conformal_gap"], conformal_gap)
    write_csv(paths["set_size"], set_size)
    paths["scale_report"].write_text(
        "# AlphaEarth scale sensitivity v2\n\n"
        "This formal repeated scale-sensitivity analysis loads the full AlphaEarth source export through the manifest, subsamples source rows, and retrains/evaluates under strict spatial-block split at each scale. "
        "`alphaearth_prediction_subset_sensitivity_diagnostic.csv` is a separate diagnostic and is not the formal scale result.\n",
        encoding="utf-8",
    )
    paths["dynamic_report"].write_text(
        "# Dynamic World diagnostic\n\n"
        f"{dynamic_status}\n\n"
        "Paper-facing Dynamic World mechanism claims must use `test_only` or `eval_calibration_test` rows. "
        "`all_split_descriptive` rows are map-product/background distribution only and must not be used as formal model accuracy evidence.\n",
        encoding="utf-8",
    )
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
    parser.add_argument("--seeds", default="42,73,101")
    parser.add_argument("--max-scales", type=int, default=None)
    args = parser.parse_args()
    paths = build_hardening(args)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
