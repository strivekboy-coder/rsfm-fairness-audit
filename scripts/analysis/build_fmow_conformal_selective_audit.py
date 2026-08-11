from __future__ import annotations

import argparse
import ast
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer import BWERConfig, compute_bwer
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


DEFAULT_OUTPUT = Path("outputs/fmow_conformal_selective_audit_v1")
DEFAULT_DRIVE_AUDIT = Path("outputs/drive_real_audit_v1")
DEFAULT_UNIFIED_V2 = Path("outputs/unified_audit_matrix_v2")
DEFAULT_COVERAGES = (0.7, 0.8, 0.9)


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).lower() in {"nan", "none", "null"}


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if _is_missing(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_correct(row: Mapping[str, Any]) -> bool | None:
    if not _is_missing(row.get("correct")):
        value = str(row.get("correct")).strip().lower()
        if value in {"true", "1", "yes"}:
            return True
        if value in {"false", "0", "no"}:
            return False
    for left, right in (("y_true", "y_pred"), ("label", "prediction"), ("class_label", "prediction")):
        if left in row and right in row and not _is_missing(row.get(left)) and not _is_missing(row.get(right)):
            return str(row.get(left)) == str(row.get(right))
    return None


def _risk(row: Mapping[str, Any]) -> float:
    if not _is_missing(row.get("risk")):
        return _float(row.get("risk"))
    correct = _bool_correct(row)
    if correct is not None:
        return 0.0 if correct else 1.0
    return float("nan")


def _confidence(row: Mapping[str, Any]) -> float:
    for key in ("confidence", "max_probability", "max_prob", "score"):
        value = _float(row.get(key))
        if not math.isnan(value):
            return value
    return float("nan")


def _row_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("sample_id", "image_id", "extracted_path", "image_path"):
        if not _is_missing(row.get(key)):
            return str(row.get(key))
    return str(index)


def _normalise_rows(rows: Sequence[Mapping[str, Any]], run_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["run_id"] = run_id
        item["_row_index"] = index
        item["_row_id"] = _row_id(row, index)
        item["_risk"] = _risk(row)
        item["_confidence"] = _confidence(row)
        out.append(item)
    return out


def choose_group_column(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ("location_id", "country", "region", "un_region")) -> str:
    for column in preferred:
        values = {str(row.get(column)) for row in rows if not _is_missing(row.get(column))}
        if len(values) >= 2:
            return column
    return ""


def build_grouped_calibration_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    calibration_fraction: float = 0.5,
    group_column: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    materialized = [dict(row) for row in rows]
    resolved_group = group_column or choose_group_column(materialized)
    if resolved_group:
        grouped: dict[str, list[int]] = {}
        missing_indexes: list[int] = []
        for index, row in enumerate(materialized):
            value = row.get(resolved_group)
            if _is_missing(value):
                missing_indexes.append(index)
            else:
                grouped.setdefault(str(value), []).append(index)
        groups = sorted(grouped)
        rng = random.Random(seed)
        rng.shuffle(groups)
        target = max(1, int(round(len(materialized) * calibration_fraction)))
        calibration_indexes: set[int] = set()
        for group in groups:
            if len(calibration_indexes) >= target and calibration_indexes:
                break
            calibration_indexes.update(grouped[group])
        # Missing-group rows are assigned by deterministic row order after the grouped split.
        for index in missing_indexes:
            destination = "calibration" if len(calibration_indexes) < target else "test"
            if destination == "calibration":
                calibration_indexes.add(index)
        split_rows = []
        for index, row in enumerate(materialized):
            split_rows.append({**row, "calibration_split": "calibration" if index in calibration_indexes else "test", "calibration_group_column": resolved_group, "calibration_group_value": row.get(resolved_group, "")})
        overlap = False
    else:
        indexes = list(range(len(materialized)))
        random.Random(seed).shuffle(indexes)
        calibration_indexes = set(indexes[: max(1, int(round(len(indexes) * calibration_fraction)))])
        split_rows = [{**row, "calibration_split": "calibration" if index in calibration_indexes else "test", "calibration_group_column": "", "calibration_group_value": ""} for index, row in enumerate(materialized)]
        overlap = True
    n_calibration = sum(1 for row in split_rows if row["calibration_split"] == "calibration")
    report = {
        "group_column": resolved_group or "none",
        "grouped_split": bool(resolved_group),
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "n_rows": len(split_rows),
        "n_calibration": n_calibration,
        "n_test": len(split_rows) - n_calibration,
        "group_overlap_between_calibration_and_test": overlap,
    }
    return split_rows, report


def confidence_threshold_for_coverage(rows: Sequence[Mapping[str, Any]], coverage: float, confidence_column: str = "_confidence") -> float:
    values = sorted(_float(row.get(confidence_column)) for row in rows if not math.isnan(_float(row.get(confidence_column))))
    if not values:
        return float("nan")
    index = max(0, min(len(values) - 1, int(math.floor((1.0 - coverage) * (len(values) - 1)))))
    return values[index]


def retained_by_threshold(rows: Sequence[Mapping[str, Any]], threshold: float, confidence_column: str = "_confidence") -> list[dict[str, Any]]:
    return [dict(row) for row in rows if not math.isnan(_float(row.get(confidence_column))) and _float(row.get(confidence_column)) >= threshold]


def _mean(values: Sequence[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return float(np.mean(clean)) if clean else float("nan")


def selective_risk_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    coverages: Sequence[float],
    selector_name: str,
    thresholds_by_coverage: Mapping[float, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    usable = [dict(row) for row in rows if not math.isnan(_float(row.get("_confidence"))) and not math.isnan(_float(row.get("_risk")))]
    summary: list[dict[str, Any]] = []
    retained_slice: list[dict[str, Any]] = []
    high_conf: list[dict[str, Any]] = []
    for coverage in coverages:
        threshold = thresholds_by_coverage.get(float(coverage), float("nan")) if thresholds_by_coverage else confidence_threshold_for_coverage(usable, float(coverage))
        retained = retained_by_threshold(usable, threshold)
        summary.append(
            {
                "run_id": run_id,
                "selector": selector_name,
                "coverage_target": coverage,
                "confidence_threshold": threshold,
                "retained_count": len(retained),
                "total_count": len(usable),
                "retained_coverage": len(retained) / max(1, len(usable)),
                "abstention_rate": 1.0 - (len(retained) / max(1, len(usable))),
                "mean_risk": _mean([_float(row.get("_risk")) for row in retained]),
                "baseline_mean_risk": _mean([_float(row.get("_risk")) for row in usable]),
                "risk_reduction": _mean([_float(row.get("_risk")) for row in usable]) - _mean([_float(row.get("_risk")) for row in retained]),
                "confidence_column": "confidence/max_probability",
                "risk_column": "risk",
            }
        )
        for column in ("country", "region", "class_label"):
            if not usable or column not in usable[0]:
                continue
            values = sorted({str(row.get(column)) for row in usable if not _is_missing(row.get(column))})
            for value in values:
                all_slice = [row for row in usable if str(row.get(column)) == value]
                kept = [row for row in retained if str(row.get(column)) == value]
                slice_row = {
                    "run_id": run_id,
                    "selector": selector_name,
                    "coverage_target": coverage,
                    "slice_variable": column,
                    "slice_value": value,
                    "retained_count": len(kept),
                    "total_count": len(all_slice),
                    "retained_coverage": len(kept) / max(1, len(all_slice)),
                    "mean_risk": _mean([_float(row.get("_risk")) for row in kept]),
                    "baseline_mean_risk": _mean([_float(row.get("_risk")) for row in all_slice]),
                }
                retained_slice.append(slice_row)
                high_conf.append({**slice_row, "high_confidence_error": slice_row["mean_risk"]})
    return summary, retained_slice, high_conf


def _probability_vector(row: Mapping[str, Any]) -> list[float] | None:
    for key in ("probabilities", "class_probabilities", "prob_vector", "probs"):
        if _is_missing(row.get(key)):
            continue
        value = row.get(key)
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(str(value))
            except (ValueError, SyntaxError):
                continue
        if isinstance(parsed, list) and parsed and all(isinstance(item, (int, float)) for item in parsed):
            return [float(item) for item in parsed]
    return None


def true_class_probability(row: Mapping[str, Any]) -> float:
    for key in ("true_probability", "p_true", "true_class_probability", "prob_true"):
        value = _float(row.get(key))
        if not math.isnan(value):
            return value
    vector = _probability_vector(row)
    if vector is None:
        return float("nan")
    label = row.get("label", row.get("y_true"))
    try:
        index = int(label)
    except (TypeError, ValueError):
        return float("nan")
    if 0 <= index < len(vector):
        return vector[index]
    return float("nan")


def conformal_threshold_from_true_probability(calibration_rows: Sequence[Mapping[str, Any]], alpha: float) -> float:
    scores = sorted(1.0 - true_class_probability(row) for row in calibration_rows if not math.isnan(true_class_probability(row)))
    if not scores:
        return float("nan")
    rank = int(math.ceil((len(scores) + 1) * (1.0 - alpha))) - 1
    rank = max(0, min(rank, len(scores) - 1))
    return scores[rank]


def bwer_summary_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    selector: str,
    coverage: float,
    min_samples_per_slice: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for slice_variable, balance_variable, label in (("country", None, "raw"), ("country", "class_label", "standardised")):
        if not rows or slice_variable not in rows[0] or (balance_variable and balance_variable not in rows[0]):
            continue
        config = BWERConfig(
            dataset="fmow_sentinel",
            model=run_id,
            task="scene_classification",
            split="test",
            risk_name="_risk",
            tail_fraction=0.1,
            min_samples_per_slice=min_samples_per_slice,
            min_units_required=min_samples_per_slice,
            min_slices_required=2,
            missing_balance_policy="renormalize",
        )
        result = compute_bwer(rows, config, slice_variable, balance_variable=balance_variable, risk_column="_risk")
        summary = result.summary
        out.append(
            {
                "run_id": run_id,
                "selector": selector,
                "coverage_target": coverage,
                "analysis_type": label,
                "slice_variable": slice_variable,
                "balance_variable": balance_variable or "",
                "bwer": summary.get("bwer", ""),
                "mean_slice_risk": summary.get("mean_risk", ""),
                "tail_risk": summary.get("tail_risk", ""),
                "worst_slice": summary.get("worst_slice", ""),
                "tail_slices": summary.get("tail_slices", ""),
                "n_valid_slices": summary.get("n_slices_valid", ""),
                "note": "Post-hoc selective/conformal BWER over retained fMoW test rows.",
            }
        )
    return out


def support_filtered_slice_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_support: int,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row.get("run_id")),
                str(row.get("selector")),
                str(row.get("coverage_target")),
                str(row.get("slice_variable")),
            ),
            [],
        ).append(row)
    for (run_id, selector, coverage, slice_variable), items in sorted(groups.items()):
        supported = [item for item in items if _float(item.get("retained_count"), 0.0) >= min_support]
        unsupported = [item for item in items if _float(item.get("retained_count"), 0.0) < min_support]
        ranked = sorted(supported, key=lambda item: (-_float(item.get("mean_risk")), str(item.get("slice_value"))))
        for rank, item in enumerate(ranked[:top_n], start=1):
            output.append(
                {
                    "run_id": run_id,
                    "selector": selector,
                    "coverage_target": coverage,
                    "slice_variable": slice_variable,
                    "support_filter": f"retained_count >= {min_support}",
                    "n_supported_slices": len(supported),
                    "n_excluded_low_support_slices": len(unsupported),
                    "risk_rank": rank,
                    "slice_value": item.get("slice_value"),
                    "retained_count": item.get("retained_count"),
                    "total_count": item.get("total_count"),
                    "retained_coverage": item.get("retained_coverage"),
                    "mean_risk": item.get("mean_risk"),
                    "baseline_mean_risk": item.get("baseline_mean_risk", ""),
                    "note": "Worst slices are ranked after support filtering to reduce low-support country artifacts.",
                }
            )
    return output


def rank_divergence_rows(
    aggregate_rows: Sequence[Mapping[str, Any]],
    bwer_rows: Sequence[Mapping[str, Any]],
    *,
    scenario_label: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in aggregate_rows:
        key = (str(row.get("selector")), str(row.get("coverage_target")), str(row.get("run_id")))
        aggregates[key] = dict(row)
    by_case: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in bwer_rows:
        key = (str(row.get("selector")), str(row.get("coverage_target")), str(row.get("analysis_type")))
        by_case.setdefault(key, []).append(row)
    for (selector, coverage, analysis_type), items in sorted(by_case.items()):
        run_ids = sorted({str(row.get("run_id")) for row in items})
        agg_candidates = []
        bwer_candidates = []
        for run_id in run_ids:
            agg = aggregates.get((selector, coverage, run_id))
            bwer = next((row for row in items if str(row.get("run_id")) == run_id), None)
            if agg is not None:
                agg_candidates.append((run_id, _float(agg.get("mean_risk")), agg))
            if bwer is not None:
                bwer_candidates.append((run_id, _float(bwer.get("bwer")), bwer))
        agg_candidates = [item for item in agg_candidates if not math.isnan(item[1])]
        bwer_candidates = [item for item in bwer_candidates if not math.isnan(item[1])]
        if not agg_candidates or not bwer_candidates:
            continue
        aggregate_best = min(agg_candidates, key=lambda item: (item[1], item[0]))
        bwer_best = min(bwer_candidates, key=lambda item: (item[1], item[0]))
        output.append(
            {
                "scenario": scenario_label,
                "selector": selector,
                "coverage_target": coverage,
                "analysis_type": analysis_type,
                "aggregate_best_run": aggregate_best[0],
                "aggregate_best_mean_risk": aggregate_best[1],
                "bwer_best_run": bwer_best[0],
                "bwer_best": bwer_best[1],
                "rank_diverges": aggregate_best[0] != bwer_best[0],
                "compared_runs": ";".join(run_ids),
                "interpretation": "aggregate-best differs from BWER-best under this selective audit setting" if aggregate_best[0] != bwer_best[0] else "aggregate-best and BWER-best agree under this selective audit setting",
            }
        )
    return output


def _baseline_summary_for_test_rows(rows: Sequence[Mapping[str, Any]], run_id: str) -> dict[str, Any]:
    risks = [_float(row.get("_risk")) for row in rows]
    return {
        "run_id": run_id,
        "selector": "baseline_all_test",
        "coverage_target": 1.0,
        "confidence_threshold": "",
        "retained_count": len(rows),
        "total_count": len(rows),
        "retained_coverage": 1.0,
        "abstention_rate": 0.0,
        "mean_risk": _mean(risks),
        "baseline_mean_risk": _mean(risks),
        "risk_reduction": 0.0,
        "confidence_column": "none",
        "risk_column": "risk",
    }


def discover_fmow_audit_tables() -> dict[str, Path]:
    source_config = Path(__file__).resolve().parents[2] / "configs" / "analysis" / "fmow_asset_sources.json"
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    configured = payload.get("audit_tables", {})
    if not isinstance(configured, dict):
        raise ValueError(f"Invalid audit_tables in {source_config}")
    project_root = Path(__file__).resolve().parents[2]
    candidates = {
        str(run_id): [
            path if path.is_absolute() else project_root / path
            for path in (Path(str(value)) for value in values)
        ]
        for run_id, values in configured.items()
    }
    found: dict[str, Path] = {}
    for run_id, paths in candidates.items():
        for path in paths:
            if path.exists():
                found[run_id] = path
                break
    return found


def _write_figures(
    output: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    retained_rows: Sequence[Mapping[str, Any]],
    conformal_slice_rows: Sequence[Mapping[str, Any]],
    bwer_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    figures = ensure_dir(output / "figures")
    paths = {
        "coverage_vs_risk": figures / "coverage_vs_risk.png",
        "retained_coverage_by_slice": figures / "retained_coverage_by_slice.png",
        "conformal_coverage_by_slice": figures / "conformal_coverage_by_slice.png",
        "selective_bwer_comparison": figures / "selective_bwer_comparison.png",
    }
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        for path in paths.values():
            path.write_text("matplotlib unavailable\n", encoding="utf-8")
        return paths

    colors = {"resnet50_13band": "#2F5DA8", "dofa_scaled10000": "#2E8B70"}
    labels = {"resnet50_13band": "ResNet-50", "dofa_scaled10000": "DOFA"}

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for run_id in sorted({str(row.get("run_id")) for row in summary_rows}):
        items = [row for row in summary_rows if row.get("run_id") == run_id and row.get("selector") == "confidence_topk_test"]
        items = sorted(items, key=lambda row: _float(row.get("coverage_target")))
        ax.plot([_float(row.get("retained_coverage")) for row in items], [_float(row.get("mean_risk")) for row in items], marker="o", linewidth=2, color=colors.get(run_id), label=labels.get(run_id, run_id))
    ax.set_xlabel("Retained coverage")
    ax.set_ylabel("Mean risk")
    ax.set_title("fMoW selective risk")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(paths["coverage_vs_risk"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    country_rows = [row for row in support_rows if row.get("slice_variable") == "country" and str(row.get("coverage_target")) == "0.8" and row.get("selector") == "confidence_topk_test"]
    top = sorted(country_rows, key=lambda row: (-_float(row.get("mean_risk")), str(row.get("slice_value"))))[:12]
    ax.bar(range(len(top)), [_float(row.get("retained_coverage")) for row in top], color=[colors.get(str(row.get("run_id")), "#777777") for row in top])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"{row.get('slice_value')}\n{labels.get(str(row.get('run_id')), row.get('run_id'))}" for row in top], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Retained coverage")
    ax.set_title("Support-filtered high-risk countries at 80% target coverage")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(paths["retained_coverage_by_slice"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    top = sorted(conformal_slice_rows, key=lambda row: _float(row.get("total_count")), reverse=True)[:20]
    ax.bar(range(len(top)), [_float(row.get("empirical_retained_accuracy")) for row in top], color=[colors.get(str(row.get("run_id")), "#777777") for row in top])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"{row.get('slice_value')}\n{labels.get(str(row.get('run_id')), row.get('run_id'))}" for row in top], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Empirical retained accuracy")
    ax.set_title("Calibrated-threshold retained accuracy by large slices")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(paths["conformal_coverage_by_slice"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=False)
    for ax, analysis_type, title in zip(axes, ["raw", "standardised"], ["Country Raw-BWER", "Country | class Std-BWER"]):
        rows = [
            row
            for row in bwer_rows
            if row.get("analysis_type") == analysis_type
            and row.get("selector") in {"baseline_all_test", "confidence_topk_test", "calibrated_confidence_threshold_diagnostic"}
        ]
        cases = [
            ("baseline_all_test", "1.0", "Base"),
            ("confidence_topk_test", "0.7", "Top-k\n0.7"),
            ("confidence_topk_test", "0.8", "Top-k\n0.8"),
            ("confidence_topk_test", "0.9", "Top-k\n0.9"),
            ("calibrated_confidence_threshold_diagnostic", "0.7", "Cal\n0.7"),
            ("calibrated_confidence_threshold_diagnostic", "0.8", "Cal\n0.8"),
            ("calibrated_confidence_threshold_diagnostic", "0.9", "Cal\n0.9"),
        ]
        x = np.arange(len(cases), dtype=float)
        width = 0.36
        for offset, run_id in [(-width / 2, "resnet50_13band"), (width / 2, "dofa_scaled10000")]:
            heights = []
            for selector, coverage, _label in cases:
                row = next(
                    (
                        item
                        for item in rows
                        if str(item.get("selector")) == selector
                        and str(item.get("coverage_target")) == coverage
                        and item.get("run_id") == run_id
                    ),
                    {},
                )
                heights.append(_float(row.get("bwer")))
            ax.bar(x + offset, heights, color=colors.get(run_id, "#777777"), width=width)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _selector, _coverage, label in cases], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("BWER")
    handles = [Line2D([0], [0], color=colors[key], lw=6) for key in ["resnet50_13band", "dofa_scaled10000"]]
    fig.legend(handles, ["ResNet-50", "DOFA"], loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=False)
    fig.suptitle("fMoW selective BWER comparison", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(paths["selective_bwer_comparison"], dpi=180)
    plt.close(fig)
    return paths


def _write_unified_matrix_v2(
    *,
    output_dir: Path,
    rank_rows: Sequence[Mapping[str, Any]],
    selective_summary: Sequence[Mapping[str, Any]],
    bwer_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    out = ensure_dir(output_dir)
    paths = {
        "unified_rank_divergence": out / "rank_divergence_under_selective_audit.csv",
        "unified_fmow_selective_main": out / "fmow_selective_main_results.csv",
        "unified_support_filtered": out / "fmow_support_filtered_slice_summary.csv",
        "paper_ready_report": out / "paper_ready_fmow_selective_audit_report.md",
    }
    write_csv(paths["unified_rank_divergence"], rank_rows)
    main_rows = []
    for summary in selective_summary:
        bwer = next(
            (
                row
                for row in bwer_rows
                if row.get("run_id") == summary.get("run_id")
                and row.get("selector") == summary.get("selector")
                and str(row.get("coverage_target")) == str(summary.get("coverage_target"))
                and row.get("analysis_type") == "raw"
            ),
            {},
        )
        main_rows.append(
            {
                "experiment_id": "fmow_sentinel_step3_selective",
                "dataset": "fMoW-Sentinel",
                "deployment_axis": "geography_location",
                "run_id": summary.get("run_id"),
                "selector": summary.get("selector"),
                "coverage_target": summary.get("coverage_target"),
                "mean_risk": summary.get("mean_risk"),
                "risk_reduction": summary.get("risk_reduction"),
                "country_raw_bwer": bwer.get("bwer", ""),
                "country_worst_slice": bwer.get("worst_slice", ""),
                "n_valid_country_slices": bwer.get("n_valid_slices", ""),
                "claim_status": "formal_posthoc" if summary.get("selector") in {"baseline_all_test", "confidence_topk_test"} else "diagnostic_calibrated_threshold",
            }
        )
    write_csv(paths["unified_fmow_selective_main"], main_rows)
    write_csv(paths["unified_support_filtered"], support_rows)
    divergent = [row for row in rank_rows if str(row.get("rank_diverges")).lower() == "true"]
    paths["paper_ready_report"].write_text(
        "# Paper-ready fMoW selective audit report\n\n"
        "This v2 matrix addendum integrates the fMoW post-hoc selective and calibrated-threshold audit into the paper synthesis. "
        "It uses saved fMoW Step3 audit tables only; no training or inference was run.\n\n"
        "## Main takeaways\n\n"
        f"- Rank divergence rows evaluated: {len(rank_rows)}.\n"
        f"- Aggregate-best and BWER-best diverge in {len(divergent)} rows.\n"
        "- Support-filtered slice summaries rank worst countries only after retained support filtering, reducing low-support worst-slice artifacts.\n"
        "- Confidence top-k selective risk is a formal post-hoc claim from saved confidence/max_probability fields.\n"
        "- Calibrated-threshold rows are diagnostics, not full APS/RAPS conformal claims, because true-class probabilities/full probability vectors are absent.\n\n"
        "## Files\n\n"
        "- `rank_divergence_under_selective_audit.csv`\n"
        "- `fmow_selective_main_results.csv`\n"
        "- `fmow_support_filtered_slice_summary.csv`\n",
        encoding="utf-8",
    )
    return paths


def _load_contract_notes(drive_audit_dir: Path) -> list[str]:
    notes = []
    for name in ("audit_contract_report.md", "audit_contract_coverage.csv", "missing_fields_by_experiment.csv", "rerun_requirements.csv"):
        path = drive_audit_dir / name
        notes.append(f"{name}: {'found' if path.exists() else 'missing'}")
    return notes


def build_fmow_conformal_selective_audit(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    drive_audit_dir: Path = DEFAULT_DRIVE_AUDIT,
    audit_tables: Mapping[str, Path] | None = None,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    seed: int = 42,
    calibration_fraction: float = 0.5,
    min_samples_per_slice: int = 20,
    alpha: float = 0.1,
    unified_output_dir: Path = DEFAULT_UNIFIED_V2,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    resolved_tables = dict(audit_tables or {})
    if not resolved_tables:
        resolved_tables = discover_fmow_audit_tables()
    missing_runs = [run_id for run_id in ("resnet50_13band", "dofa_scaled10000") if run_id not in resolved_tables]
    if missing_runs:
        missing_path = output / "missing_fmow_audit_tables.md"
        missing_path.write_text(
            "# Missing fMoW audit tables\n\n"
            + "\n".join(f"- {run_id}" for run_id in missing_runs)
            + "\n\nExpected formal candidates are final_step3 ResNet50 location-disjoint and DOFA scaled10000 location-disjoint audit_table.csv files.\n",
            encoding="utf-8",
        )
        if len(missing_runs) == 2:
            return {"missing_fmow_audit_tables": missing_path}

    split_manifest: list[dict[str, Any]] = []
    split_reports: list[dict[str, Any]] = []
    selective_summary: list[dict[str, Any]] = []
    retained_slice_rows: list[dict[str, Any]] = []
    high_conf_rows: list[dict[str, Any]] = []
    selective_bwer_rows: list[dict[str, Any]] = []
    conformal_coverage_rows: list[dict[str, Any]] = []
    conformal_bwer_rows: list[dict[str, Any]] = []
    conformal_slice_rows: list[dict[str, Any]] = []
    support_filtered_rows: list[dict[str, Any]] = []
    available_fields: dict[str, list[str]] = {}

    for run_id, path in sorted(resolved_tables.items()):
        rows = _normalise_rows(read_csv_rows(path), run_id)
        available_fields[run_id] = sorted(rows[0].keys()) if rows else []
        split_rows, split_report = build_grouped_calibration_split(rows, seed=seed, calibration_fraction=calibration_fraction)
        split_report = {**split_report, "run_id": run_id, "source_path": str(path)}
        split_reports.append(split_report)
        for row in split_rows:
            split_manifest.append(
                {
                    "run_id": run_id,
                    "row_id": row.get("_row_id"),
                    "sample_id": row.get("sample_id", ""),
                    "image_id": row.get("image_id", ""),
                    "location_id": row.get("location_id", ""),
                    "country": row.get("country", ""),
                    "region": row.get("region", ""),
                    "class_label": row.get("class_label", ""),
                    "calibration_split": row.get("calibration_split"),
                    "calibration_group_column": row.get("calibration_group_column"),
                    "calibration_group_value": row.get("calibration_group_value"),
                }
            )
        calibration = [row for row in split_rows if row.get("calibration_split") == "calibration"]
        test = [row for row in split_rows if row.get("calibration_split") == "test"]
        baseline_row = _baseline_summary_for_test_rows(test, run_id)
        selective_summary.append(baseline_row)
        selective_bwer_rows.extend(bwer_summary_for_rows(test, run_id=run_id, selector="baseline_all_test", coverage=1.0, min_samples_per_slice=min_samples_per_slice))
        summary, retained, high_conf = selective_risk_rows(test, run_id=run_id, coverages=coverages, selector_name="confidence_topk_test")
        selective_summary.extend(summary)
        retained_slice_rows.extend(retained)
        high_conf_rows.extend(high_conf)
        for coverage in coverages:
            threshold = confidence_threshold_for_coverage(test, float(coverage))
            retained_test = retained_by_threshold(test, threshold)
            selective_bwer_rows.extend(bwer_summary_for_rows(retained_test, run_id=run_id, selector="confidence_topk_test", coverage=float(coverage), min_samples_per_slice=min_samples_per_slice))

        calibrated_thresholds = {float(coverage): confidence_threshold_for_coverage(calibration, float(coverage)) for coverage in coverages}
        conformal_method = "split_conformal_true_probability" if any(not math.isnan(true_class_probability(row)) for row in calibration) else "calibrated_confidence_threshold_diagnostic"
        qhat = conformal_threshold_from_true_probability(calibration, alpha)
        calibrated_summary, calibrated_retained, calibrated_high_conf = selective_risk_rows(
            test,
            run_id=run_id,
            coverages=coverages,
            selector_name=conformal_method,
            thresholds_by_coverage=calibrated_thresholds,
        )
        selective_summary.extend(calibrated_summary)
        retained_slice_rows.extend(calibrated_retained)
        high_conf_rows.extend(calibrated_high_conf)
        for coverage in coverages:
            threshold = calibrated_thresholds[float(coverage)]
            retained_test = retained_by_threshold(test, threshold)
            correct_values = [1.0 - _float(row.get("_risk")) for row in retained_test if not math.isnan(_float(row.get("_risk")))]
            row = {
                "run_id": run_id,
                "method": conformal_method,
                "coverage_target": coverage,
                "alpha": alpha,
                "qhat_nonconformity": qhat,
                "confidence_threshold": threshold,
                "calibration_count": len(calibration),
                "test_count": len(test),
                "retained_count": len(retained_test),
                "retained_coverage": len(retained_test) / max(1, len(test)),
                "empirical_retained_accuracy": _mean(correct_values),
                "formal_label_coverage_claim": "yes" if conformal_method == "split_conformal_true_probability" else "no",
                "note": "Only confidence/max_probability was available; this is calibrated retention, weaker than APS/RAPS conformal." if conformal_method != "split_conformal_true_probability" else "Split conformal using true-class probability.",
            }
            conformal_coverage_rows.append(row)
            for column in ("country", "region"):
                if not test or column not in test[0]:
                    continue
                for value in sorted({str(item.get(column)) for item in test if not _is_missing(item.get(column))}):
                    all_slice = [item for item in test if str(item.get(column)) == value]
                    kept = [item for item in retained_test if str(item.get(column)) == value]
                    conformal_slice_rows.append(
                        {
                            "run_id": run_id,
                            "method": conformal_method,
                            "coverage_target": coverage,
                            "slice_variable": column,
                            "slice_value": value,
                            "retained_count": len(kept),
                            "total_count": len(all_slice),
                            "retained_coverage": len(kept) / max(1, len(all_slice)),
                            "empirical_retained_accuracy": _mean([1.0 - _float(item.get("_risk")) for item in kept]),
                        }
                    )
            conformal_bwer_rows.extend(bwer_summary_for_rows(retained_test, run_id=run_id, selector=conformal_method, coverage=float(coverage), min_samples_per_slice=min_samples_per_slice))

    all_bwer_rows = selective_bwer_rows + conformal_bwer_rows
    support_filtered_rows = support_filtered_slice_summary(retained_slice_rows, min_support=min_samples_per_slice)
    rank_rows = rank_divergence_rows(selective_summary, all_bwer_rows, scenario_label="fmow_step3_location_disjoint_test_split")

    artifacts = {
        "fmow_selective_risk_summary": output / "fmow_selective_risk_summary.csv",
        "fmow_retained_coverage_by_slice": output / "fmow_retained_coverage_by_slice.csv",
        "fmow_high_confidence_error_by_slice": output / "fmow_high_confidence_error_by_slice.csv",
        "fmow_support_filtered_slice_summary": output / "fmow_support_filtered_slice_summary.csv",
        "fmow_selective_bwer_summary": output / "fmow_selective_bwer_summary.csv",
        "fmow_conformal_coverage_summary": output / "fmow_conformal_coverage_summary.csv",
        "fmow_conformal_bwer_summary": output / "fmow_conformal_bwer_summary.csv",
        "fmow_conformal_slice_coverage": output / "fmow_conformal_slice_coverage.csv",
        "rank_divergence_under_selective_audit": output / "rank_divergence_under_selective_audit.csv",
        "calibration_split_manifest": output / "calibration_split_manifest.csv",
        "calibration_split_report": output / "calibration_split_report.md",
        "fmow_conformal_selective_report": output / "fmow_conformal_selective_report.md",
    }
    write_csv(artifacts["fmow_selective_risk_summary"], selective_summary)
    write_csv(artifacts["fmow_retained_coverage_by_slice"], retained_slice_rows)
    write_csv(artifacts["fmow_high_confidence_error_by_slice"], high_conf_rows)
    write_csv(artifacts["fmow_support_filtered_slice_summary"], support_filtered_rows)
    write_csv(artifacts["fmow_selective_bwer_summary"], selective_bwer_rows)
    write_csv(artifacts["fmow_conformal_coverage_summary"], conformal_coverage_rows)
    write_csv(artifacts["fmow_conformal_bwer_summary"], conformal_bwer_rows)
    write_csv(artifacts["fmow_conformal_slice_coverage"], conformal_slice_rows)
    write_csv(artifacts["rank_divergence_under_selective_audit"], rank_rows)
    write_csv(artifacts["calibration_split_manifest"], split_manifest)

    contract_notes = _load_contract_notes(drive_audit_dir)
    risk_reductions = [row for row in selective_summary if not math.isnan(_float(row.get("risk_reduction")))]
    reduces = any(_float(row.get("risk_reduction")) > 0 for row in risk_reductions)
    fields_lines = []
    for run_id, fields in available_fields.items():
        found = [field for field in ("confidence", "max_probability", "risk", "correct", "score", "split", "location_id", "country", "region", "class_label", "probabilities", "true_probability") if field in fields]
        fields_lines.append(f"- {run_id}: {', '.join(found)}")
    artifacts["calibration_split_report"].write_text(
        "# fMoW calibration split report\n\n"
        + "\n".join(f"- {row['run_id']}: group={row['group_column']}, seed={row['seed']}, calibration={row['n_calibration']}, test={row['n_test']}, overlap={row['group_overlap_between_calibration_and_test']}" for row in split_reports)
        + "\n\nDrive audit inputs:\n"
        + "\n".join(f"- {note}" for note in contract_notes)
        + "\n",
        encoding="utf-8",
    )
    artifacts["fmow_conformal_selective_report"].write_text(
        "# fMoW conformal selective audit report\n\n"
        "This is a post-hoc audit over saved fMoW Step3 audit tables. No model training or inference was run.\n\n"
        "## Available fields\n\n"
        + "\n".join(fields_lines)
        + "\n\n## Findings\n\n"
        + f"- Confidence filtering reduces average risk in at least one fMoW run/coverage: {reduces}.\n"
        "- Geography tail risk is evaluated in `fmow_selective_bwer_summary.csv` and `fmow_conformal_bwer_summary.csv`; nonzero BWER means tail risk remains after filtering.\n"
        "- Worst-slice interpretation should use `fmow_support_filtered_slice_summary.csv` to avoid low-support country artifacts.\n"
        "- Aggregate-vs-BWER model choice under baseline, top-k selective, and calibrated-threshold settings is recorded in `rank_divergence_under_selective_audit.csv`.\n"
        "- Full APS/RAPS-style conformal was not claimed unless true-class probability or full probability vectors were present.\n"
        "- Current fMoW Drive tables expose confidence/max_probability, so conformal outputs are calibrated confidence-threshold diagnostics unless probability-vector columns are added later.\n"
        "- Formal claims: post-hoc selective risk/BWER from saved fMoW audit tables. Diagnostic claims: calibrated confidence-threshold conformal selective BWER without true-class probabilities.\n\n"
        "## Rerun status\n\n"
        "fMoW does not need model retraining or inference rerun for confidence-based selective risk. Full split conformal with label-set coverage would require saved true-class probabilities or full probability vectors.\n",
        encoding="utf-8",
    )
    unified_paths = _write_unified_matrix_v2(
        output_dir=unified_output_dir,
        rank_rows=rank_rows,
        selective_summary=selective_summary,
        bwer_rows=all_bwer_rows,
        support_rows=support_filtered_rows,
    )
    artifacts.update({f"unified_v2_{key}": value for key, value in unified_paths.items()})
    fig_paths = _write_figures(output, selective_summary, retained_slice_rows, conformal_slice_rows, all_bwer_rows, support_filtered_rows)
    artifacts.update({f"figure_{key}": value for key, value in fig_paths.items()})
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fMoW post-hoc selective and conformal-selective audit outputs.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--drive-audit-dir", type=Path, default=DEFAULT_DRIVE_AUDIT)
    parser.add_argument("--resnet-audit-table", type=Path)
    parser.add_argument("--dofa-audit-table", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--coverages", nargs="*", type=float, default=list(DEFAULT_COVERAGES))
    parser.add_argument("--min-samples-per-slice", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--unified-output-dir", type=Path, default=DEFAULT_UNIFIED_V2)
    args = parser.parse_args()
    tables: dict[str, Path] = {}
    if args.resnet_audit_table:
        tables["resnet50_13band"] = args.resnet_audit_table
    if args.dofa_audit_table:
        tables["dofa_scaled10000"] = args.dofa_audit_table
    artifacts = build_fmow_conformal_selective_audit(
        output_dir=args.out,
        drive_audit_dir=args.drive_audit_dir,
        audit_tables=tables or None,
        coverages=args.coverages,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
        min_samples_per_slice=args.min_samples_per_slice,
        alpha=args.alpha,
        unified_output_dir=args.unified_output_dir,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
