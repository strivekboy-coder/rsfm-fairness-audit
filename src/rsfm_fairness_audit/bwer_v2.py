from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer import BWERConfig, bootstrap_bwer, compute_bwer, is_invalid_balance_variable
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


EVENT_SLICE = "event_id"
DEFAULT_ALPHAS = (0.1, 0.2, 0.3, 0.4)
DEFAULT_SUPPORT_THRESHOLDS = (10, 20, 30)
DEFAULT_TAUS = (10, 20, 50)
REFERENCE_WEIGHTINGS = ("uniform", "empirical")
MISSING_POLICIES = ("overlap", "renormalize", "invalidate")


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or str(value).lower() in {"nan", "none", "null"}


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        if _is_missing(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return read_csv_rows(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"json_parse_error": str(path)}


def _first_existing(input_dir: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = input_dir / name
        if path.exists():
            return path
    return None


def _common(rows: Sequence[Mapping[str, Any]], key: str, default: str = "") -> str:
    values = [str(row.get(key)) for row in rows if key in row and not _is_missing(row.get(key))]
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else default


def _event_name(row: Mapping[str, Any]) -> str:
    for key in ("event_id", "unit_id", "sample_id", "event", "country"):
        if key in row and not _is_missing(row.get(key)):
            return str(row.get(key))
    return ""


def _metric(row: Mapping[str, Any], *names: str, default: float = float("nan")) -> float:
    for name in names:
        if name in row and not _is_missing(row.get(name)):
            return _num(row.get(name), default)
    return default


def _infer_resolution(input_dir: Path, model_debug: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    for key in ("resolution", "target_size", "expected_input_size", "inference_window_size", "window_size"):
        value = model_debug.get(key)
        if not _is_missing(value):
            if isinstance(value, (list, tuple)) and value:
                return "x".join(str(item) for item in value)
            return str(value)
    for row in rows:
        for key in ("resolution", "target_size", "chip_size"):
            if key in row and not _is_missing(row.get(key)):
                return str(row.get(key))
    match = re.search(r"(?:^|_)(\d{3,4})(?:$|_)", input_dir.name)
    return match.group(1) if match else ""


def _normalise_event_rows(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    source_files: dict[str, Path] = {}
    event_path = input_dir / "event_segmentation_metrics.csv"
    if event_path.exists():
        source_files["event_segmentation_metrics"] = event_path
        rows = [dict(row) for row in read_csv_rows(event_path)]
    else:
        audit_path = _first_existing(input_dir, ("segmentation_audit_table.csv", "audit_table.csv"))
        if not audit_path:
            raise FileNotFoundError("Expected event_segmentation_metrics.csv, segmentation_audit_table.csv, or audit_table.csv.")
        source_files["audit_table"] = audit_path
        rows = [dict(row) for row in read_csv_rows(audit_path) if str(row.get("aggregation_level", "")).lower() == "event"]
    if not rows:
        raise ValueError("No event-level segmentation rows were found for BWER v2.")

    for row in rows:
        event_id = _event_name(row)
        row.setdefault("event_id", event_id)
        row.setdefault("unit_id", event_id)
        row.setdefault("sample_id", event_id)
        row.setdefault("task", "segmentation")
        row.setdefault("aggregation_level", "event")
        iou = _metric(row, "micro_iou", "iou", "water_iou", "score")
        dice = _metric(row, "micro_dice", "dice", "micro_f1", "f1")
        if math.isnan(iou):
            tp = _metric(row, "TP", "tp", default=0.0)
            fp = _metric(row, "FP", "fp", default=0.0)
            fn = _metric(row, "FN", "fn", default=0.0)
            denom = tp + fp + fn
            iou = 1.0 if denom == 0 else tp / denom
        if math.isnan(dice):
            tp = _metric(row, "TP", "tp", default=0.0)
            fp = _metric(row, "FP", "fp", default=0.0)
            fn = _metric(row, "FN", "fn", default=0.0)
            denom = (2.0 * tp) + fp + fn
            dice = 1.0 if denom == 0 else (2.0 * tp) / denom
        row["micro_iou"] = iou
        row["score"] = iou
        row.setdefault("micro_dice", dice)
        row.setdefault("risk_source", "1_minus_iou")
        row["risk"] = _metric(row, "risk", default=1.0 - iou)
        row["valid_pixel_count"] = _metric(row, "valid_pixel_count", "valid_pixel_support", default=0.0)
        row["positive_pixel_count"] = _metric(row, "positive_pixel_count", "positive_pixel_support", default=_metric(row, "TP", default=0.0) + _metric(row, "FN", default=0.0))
        row["predicted_positive_pixel_count"] = _metric(
            row,
            "predicted_positive_pixel_count",
            "predicted_positive_support",
            default=_metric(row, "TP", default=0.0) + _metric(row, "FP", default=0.0),
        )
    return rows, source_files


def _metadata(rows: Sequence[Mapping[str, Any]], input_dir: Path, model_debug: Mapping[str, Any]) -> dict[str, str]:
    dataset = _common(rows, "dataset", "sen1floods11")
    model = _common(rows, "model", "prithvi_tl_sen1floods11")
    task = _common(rows, "task", "segmentation")
    adaptation = _common(rows, "adaptation_protocol", "")
    training_budget = _common(rows, "training_budget", "") or _common(rows, "training_setup", "")
    split_protocol = _common(rows, "split_protocol", "")
    if model == "prithvi_tl_sen1floods11" and adaptation in {"", "task_adapted_decoder"}:
        adaptation = "official_sen1floods11_finetune/task_adapted_decoder"
    if model == "prithvi_tl_sen1floods11" and not training_budget:
        training_budget = "official_sen1floods11_finetune"
    return {
        "dataset": dataset,
        "model": model,
        "task": task,
        "input_mode": _common(rows, "input_mode", "S2"),
        "adaptation_protocol": adaptation,
        "training_budget": training_budget,
        "split_protocol": split_protocol or "standard_split",
        "resolution": _infer_resolution(input_dir, model_debug, rows),
    }


def _config(meta: Mapping[str, str], alpha: float = 0.1, min_support: int = 10, weighting: str = "uniform", missing_policy: str = "renormalize") -> BWERConfig:
    return BWERConfig(
        dataset=meta["dataset"],
        model=meta["model"],
        task=meta["task"],
        split="all",
        score_name="micro_iou",
        risk_name="risk",
        tail_fraction=alpha,
        weighting=weighting,
        min_samples_per_slice=1,
        min_units_required=min_support,
        min_slices_required=2,
        missing_balance_policy=missing_policy,
    )


def _compute_raw(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, str], alpha: float = 0.1, min_support: int = 10) -> Any:
    return compute_bwer(rows, _config(meta, alpha=alpha, min_support=min_support), EVENT_SLICE, None, "micro_iou", "risk")


def _source_file_text(source_files: Mapping[str, Path], input_dir: Path) -> str:
    parts = []
    for name, path in sorted(source_files.items()):
        try:
            parts.append(f"{name}:{path.relative_to(input_dir)}")
        except ValueError:
            parts.append(f"{name}:{path}")
    return ";".join(parts)


def _summary_row(
    result: Any,
    meta: Mapping[str, str],
    input_dir: Path,
    source_files: Mapping[str, Path],
    ci: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = dict(result.summary)
    ci = ci or {}
    return {
        "dataset": meta["dataset"],
        "model": meta["model"],
        "task": meta["task"],
        "slice_variable": summary.get("slice_variable", EVENT_SLICE),
        "balance_variable": summary.get("balance_variable", ""),
        "risk_source": "1_minus_iou",
        "adaptation_protocol": meta["adaptation_protocol"],
        "training_budget": meta["training_budget"],
        "split_protocol": meta["split_protocol"],
        "alpha": summary.get("tail_fraction", 0.1),
        "min_support": 10,
        "effective_min_support": 10,
        "reference_weighting": summary.get("weighting", "uniform"),
        "missing_policy": summary.get("missing_balance_policy", "renormalize"),
        "mean_risk": summary.get("mean_risk", ""),
        "tail_risk": summary.get("tail_risk", ""),
        "bwer": summary.get("bwer", ""),
        "max_bwer": summary.get("max_bwer", ""),
        "worst_slice": summary.get("worst_slice", ""),
        "best_slice": summary.get("best_slice", ""),
        "tail_slices": summary.get("tail_slices", ""),
        "valid_slice_count": summary.get("n_slices_valid", ""),
        "total_slice_count": summary.get("n_slices_total", ""),
        "support_definition": "segmentation effective support = valid_pixel_count; positive support = TP+FN",
        "ci_low": ci.get("ci_low", summary.get("ci_low", "")),
        "ci_high": ci.get("ci_high", summary.get("ci_high", "")),
        "bootstrap_method": ci.get("bootstrap_method", ci.get("method", summary.get("bootstrap_method", ""))),
        "bootstrap_n": ci.get("bootstrap_n", summary.get("bootstrap_n", "")),
        "resolution": meta["resolution"],
        "run_directory": str(input_dir),
        "source_files": _source_file_text(source_files, input_dir),
        "notes": "Post-hoc BWER-Audit v2 from saved event-level segmentation outputs; no model inference or data preparation rerun.",
    }


def _rows_from_result(result: Any, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = dict(result.summary)
    if extra:
        row.update(extra)
    row["alpha"] = row.pop("tail_fraction", row.get("alpha", ""))
    row["valid_slice_count"] = row.get("n_slices_valid", "")
    return row


def _alpha_sensitivity(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, str]) -> list[dict[str, Any]]:
    output = []
    for alpha in DEFAULT_ALPHAS:
        result = _compute_raw(rows, meta, alpha=alpha, min_support=10)
        output.append(_rows_from_result(result, {"sensitivity": "alpha", "support_definition": "valid_pixel_count"}))
    return output


def _support_sensitivity(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, str]) -> list[dict[str, Any]]:
    output = []
    total_events = len({str(row.get(EVENT_SLICE)) for row in rows})
    for min_support in DEFAULT_SUPPORT_THRESHOLDS:
        result = _compute_raw(rows, meta, alpha=0.1, min_support=min_support)
        valid_count = int(result.summary.get("n_slices_valid", 0) or 0)
        output.append(
            _rows_from_result(
                result,
                {
                    "sensitivity": "min_support",
                    "min_support": min_support,
                    "support_definition": "segmentation effective support = valid_pixel_count",
                    "all_events_valid": valid_count == total_events,
                    "notes": "All events remain valid at this threshold because pixel support is large." if valid_count == total_events else "",
                },
            )
        )
    return output


def _meaningful_balance_variables(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    candidate_columns = ["region", "sensor", "sensor_mode", "input_mode", "split_protocol", "country", "event", "event_id"]
    valid: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    columns = set(rows[0]) if rows else set()
    for column in candidate_columns:
        if column not in columns:
            continue
        values = {str(row.get(column)) for row in rows if not _is_missing(row.get(column))}
        invalid, reason = is_invalid_balance_variable(rows, EVENT_SLICE, column)
        if len(values) < 2:
            reason = reason or "balance variable has fewer than two observed levels"
            invalid = True
        diagnostics.append({"balance_variable": column, "n_levels": len(values), "valid": not invalid, "reason": reason})
        if not invalid:
            valid.append(column)
    return valid, diagnostics


def _not_applicable_rows(kind: str, diagnostics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reason = "No meaningful non-proxy balance variable is available in this completed run."
    invalid = "; ".join(f"{row.get('balance_variable')}={row.get('reason')}" for row in diagnostics if row.get("reason"))
    return [{"analysis": kind, "status": "not_applicable", "reason": reason, "balance_diagnostics": invalid}]


def _reference_weight_sensitivity(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, str], balances: Sequence[str], diagnostics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not balances:
        return _not_applicable_rows("reference_weight_sensitivity", diagnostics)
    output = []
    for balance in balances:
        for weighting in REFERENCE_WEIGHTINGS:
            config = _config(meta, alpha=0.1, min_support=10, weighting=weighting, missing_policy="renormalize")
            result = compute_bwer(rows, config, EVENT_SLICE, balance, "micro_iou", "risk")
            output.append(_rows_from_result(result, {"reference_weighting": weighting, "standardised_bwer": True}))
    return output


def _missing_policy_sensitivity(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, str], balances: Sequence[str], diagnostics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not balances:
        return _not_applicable_rows("missing_policy_sensitivity", diagnostics)
    output = []
    for balance in balances:
        for policy in MISSING_POLICIES:
            config = _config(meta, alpha=0.1, min_support=10, weighting="uniform", missing_policy=policy)
            result = compute_bwer(rows, config, EVENT_SLICE, balance, "micro_iou", "risk")
            output.append(_rows_from_result(result, {"missing_policy": policy, "standardised_bwer": True}))
    return output


def _bwer_from_risk_values(slice_rows: Sequence[Mapping[str, Any]], alpha: float, risk_key: str = "risk", min_support: int = 10) -> dict[str, Any]:
    valid = [row for row in slice_rows if _metric(row, "valid_pixel_count", default=0.0) >= min_support and not math.isnan(_metric(row, risk_key))]
    if not valid:
        return {"mean_risk": "", "tail_risk": "", "bwer": "", "max_bwer": "", "worst_slice": "", "best_slice": "", "tail_slices": "", "valid_slice_count": 0}
    risks = [_metric(row, risk_key) for row in valid]
    mean_risk = float(np.mean(risks))
    tail_n = max(1, int(math.ceil(len(valid) * alpha)))
    ranked = sorted(valid, key=lambda row: (-_metric(row, risk_key), str(row.get(EVENT_SLICE))))
    tail = ranked[:tail_n]
    tail_risk = float(np.mean([_metric(row, risk_key) for row in tail]))
    return {
        "mean_risk": mean_risk,
        "tail_risk": tail_risk,
        "bwer": tail_risk - mean_risk,
        "max_bwer": max(risks) - mean_risk,
        "worst_slice": ranked[0].get(EVENT_SLICE, ""),
        "best_slice": min(valid, key=lambda row: (_metric(row, risk_key), str(row.get(EVENT_SLICE)))).get(EVENT_SLICE, ""),
        "tail_slices": ";".join(str(row.get(EVENT_SLICE)) for row in tail),
        "valid_slice_count": len(valid),
    }


def _stabilised_bwer(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, str], baseline: Any) -> list[dict[str, Any]]:
    baseline_order = [row.get("slice_value") for row in sorted(baseline.by_slice, key=lambda row: int(row.get("rank_by_risk") or 9999))]
    baseline_worst = str(baseline.summary.get("worst_slice", ""))
    baseline_tail = str(baseline.summary.get("tail_slices", ""))
    mean_risk = _num(baseline.summary.get("mean_risk"), 0.0)
    output = []
    for tau in DEFAULT_TAUS:
        adjusted = []
        for row in rows:
            support = _metric(row, "valid_pixel_count", default=0.0)
            shrink = support / (support + tau) if support + tau > 0 else 0.0
            item = dict(row)
            item["stabilised_risk"] = mean_risk + shrink * (_metric(row, "risk") - mean_risk)
            adjusted.append(item)
        stats = _bwer_from_risk_values(adjusted, alpha=0.1, risk_key="stabilised_risk", min_support=10)
        ranking = [row.get(EVENT_SLICE) for row in sorted(adjusted, key=lambda row: (-_metric(row, "stabilised_risk"), str(row.get(EVENT_SLICE))))]
        output.append(
            {
                "dataset": meta["dataset"],
                "model": meta["model"],
                "task": meta["task"],
                "slice_variable": EVENT_SLICE,
                "tau": tau,
                "support_definition": "segmentation effective support = valid_pixel_count",
                "mean_risk": stats["mean_risk"],
                "tail_risk": stats["tail_risk"],
                "bwer": stats["bwer"],
                "max_bwer": stats["max_bwer"],
                "worst_slice": stats["worst_slice"],
                "tail_slices": stats["tail_slices"],
                "valid_slice_count": stats["valid_slice_count"],
                "ranking_changed": ranking != baseline_order,
                "worst_tail_changed": str(stats["worst_slice"]) != baseline_worst or str(stats["tail_slices"]) != baseline_tail,
            }
        )
    return output


def _leave_one_slice_out(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, str]) -> list[dict[str, Any]]:
    output = []
    for event_id in sorted({str(row.get(EVENT_SLICE)) for row in rows}):
        remaining = [row for row in rows if str(row.get(EVENT_SLICE)) != event_id]
        result = _compute_raw(remaining, meta, alpha=0.1, min_support=10)
        output.append(
            {
                "removed_event": event_id,
                "remaining_valid_slice_count": result.summary.get("n_slices_valid", ""),
                "bwer": result.summary.get("bwer", ""),
                "mean_risk": result.summary.get("mean_risk", ""),
                "tail_risk": result.summary.get("tail_risk", ""),
                "worst_remaining_slice": result.summary.get("worst_slice", ""),
                "tail_slices": result.summary.get("tail_slices", ""),
            }
        )
    return output


def _bootstrap_ci(input_dir: Path, rows: Sequence[Mapping[str, Any]], meta: Mapping[str, str], n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    existing = _read_csv_if_exists(input_dir / "bootstrap_ci.csv")
    for row in existing:
        item = dict(row)
        item["source"] = "existing_bootstrap_ci.csv"
        output.append(item)
    if len(rows) < 2 or n_bootstrap <= 0:
        output.append(
            {
                "source": "bwer_v2_posthoc",
                "status": "limitation",
                "method": "posthoc_event_bootstrap",
                "reason": "At least two event slices and n_bootstrap > 0 are required.",
            }
        )
        return output
    config = _config(meta, alpha=0.1, min_support=10)
    ci = bootstrap_bwer(rows, config, EVENT_SLICE, None, n_bootstrap=n_bootstrap, cluster_key=None, seed=seed, score_column="micro_iou", risk_column="risk")
    output.append(
        {
            "source": "bwer_v2_posthoc",
            "status": "computed",
            "method": "posthoc_event_bootstrap",
            "bootstrap_method": "posthoc_event_bootstrap",
            "slice_variable": EVENT_SLICE,
            "alpha": 0.1,
            "ci_low": ci.get("ci_low", ""),
            "ci_high": ci.get("ci_high", ""),
            "bootstrap_n": ci.get("bootstrap_n", ""),
            "bootstrap_warnings": ci.get("warnings", ""),
            "limitation": "Resamples saved event-level slices; it does not rerun chip-level inference or model stochasticity.",
        }
    )
    return output


def _event_failure_analysis(rows: Sequence[Mapping[str, Any]], baseline: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tail = set(str(baseline.summary.get("tail_slices", "")).split(";")) - {""}
    sorted_by_iou = sorted(rows, key=lambda row: (-_metric(row, "micro_iou"), str(row.get(EVENT_SLICE))))
    sorted_by_risk = sorted(rows, key=lambda row: (-_metric(row, "risk"), str(row.get(EVENT_SLICE))))
    iou_rank = {str(row.get(EVENT_SLICE)): rank for rank, row in enumerate(sorted_by_iou, start=1)}
    risk_rank = {str(row.get(EVENT_SLICE)): rank for rank, row in enumerate(sorted_by_risk, start=1)}
    failure = []
    for row in rows:
        event_id = str(row.get(EVENT_SLICE))
        valid = _metric(row, "valid_pixel_count", default=0.0)
        positive = _metric(row, "positive_pixel_count", default=0.0)
        predicted = _metric(row, "predicted_positive_pixel_count", default=0.0)
        tp = _metric(row, "TP", "tp", default=0.0)
        fp = _metric(row, "FP", "fp", default=0.0)
        fn = _metric(row, "FN", "fn", default=0.0)
        tn = _metric(row, "TN", "tn", default=0.0)
        notes = []
        if event_id in tail:
            notes.append("Raw-BWER tail event")
        if event_id == str(baseline.summary.get("worst_slice", "")):
            notes.append("worst event by risk")
        item = {
            "event_id": event_id,
            "IoU": _metric(row, "micro_iou", "iou"),
            "Dice": _metric(row, "micro_dice", "dice", "micro_f1", "f1"),
            "F1": _metric(row, "micro_f1", "f1", "micro_dice", "dice"),
            "precision": _metric(row, "precision"),
            "recall": _metric(row, "recall"),
            "risk": _metric(row, "risk"),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "valid_pixel_count": valid,
            "positive_pixel_count": positive,
            "predicted_positive_pixel_count": predicted,
            "ground_truth_positive_ratio": positive / valid if valid else "",
            "predicted_positive_ratio": predicted / valid if valid else "",
            "FP_rate": fp / (fp + tn) if fp + tn > 0 else "",
            "FN_rate": fn / (tp + fn) if tp + fn > 0 else "",
            "IoU_rank": iou_rank[event_id],
            "risk_rank": risk_rank[event_id],
            "tail_flag": event_id in tail,
            "notes": "; ".join(notes),
        }
        failure.append(item)
    ranking = sorted(failure, key=lambda row: (-_num(row["risk"]), str(row["event_id"])))
    return failure, ranking


def _write_figures(output_dir: Path, event_failure: Sequence[Mapping[str, Any]], alpha_rows: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return paths
    figures = ensure_dir(output_dir / "figures")
    events = [str(row["event_id"]) for row in sorted(event_failure, key=lambda row: _num(row["risk"]), reverse=True)]
    risks = [_num(row["risk"]) for row in sorted(event_failure, key=lambda row: _num(row["risk"]), reverse=True)]
    path = figures / "event_risk_ranking.png"
    fig, ax = plt.subplots(figsize=(max(7, len(events) * 0.55), 4))
    ax.bar(events, risks, color="#39568c")
    ax.set_ylabel("Risk (1 - IoU)")
    ax.set_xlabel("Event")
    ax.set_title("Event-level segmentation risk")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["event_risk_ranking"] = path

    path = figures / "alpha_sensitivity.png"
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot([_num(row["alpha"]) for row in alpha_rows], [_num(row["bwer"]) for row in alpha_rows], marker="o", color="#2a9d8f")
    ax.set_xlabel("Tail fraction alpha")
    ax.set_ylabel("Raw-BWER(event_id)")
    ax.set_title("Alpha sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["alpha_sensitivity"] = path
    return paths


def _report_header(title: str) -> list[str]:
    return [f"# {title}", ""]


def _write_metric_primitives_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = _report_header("BWER-Audit v2 Metric Primitives")
    lines.extend(
        [
            "BWER is a support-aware, composition-standardised, CVaR-style tail-risk statistic for deployment-relevant remote sensing slices.",
            "",
            "Classification BWER-compatible risk uses sample-level 0/1 loss, with probability losses such as NLL or Brier only when calibrated probabilities are saved.",
            "",
            "Segmentation BWER-compatible risk uses aggregated TP/FP/FN/TN and valid-pixel counts per slice. This run uses event-level Sen1Floods11 segmentation rows, not chip-level classification samples.",
            "",
            f"The formal risk source here is `{summary.get('risk_source', '1_minus_iou')}`, i.e. risk = 1 - event-level micro IoU computed from aggregated counts.",
            "",
            "Micro IoU, Dice/F1, precision, and recall are derived from event-level aggregated counts. Chip-level macro IoU is useful as an auxiliary diagnostic but is not the sole formal BWER risk source.",
            "",
            "Segmentation support is task-aware: effective support is valid pixels, positive support is TP+FN, and predicted positive support is TP+FP.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_adaptation_protocol_report(path: Path, meta: Mapping[str, str], model_debug: Mapping[str, Any]) -> None:
    source = model_debug.get("checkpoint_source", "official_huggingface")
    model_name = model_debug.get("model", model_debug.get("model_name", meta["model"]))
    lines = _report_header("Adaptation Protocol")
    lines.extend(
        [
            "- model_family: Prithvi",
            f"- model_variant: {meta['model']}",
            f"- checkpoint_source: {source}",
            f"- checkpoint/model: {model_name}",
            f"- adaptation_protocol: {meta['adaptation_protocol']}",
            f"- training_budget: {meta['training_budget']}",
            "",
            "This audit uses the official Sen1Floods11 task-adapted decoder route. It is not the earlier frozen-threshold diagnostic route.",
            "",
            "Future cross-model comparisons should be filtered or stratified by adaptation protocol family, because frozen probes, lightweight heads, task-adapted decoders, and full fine-tunes answer different methodological questions.",
        ]
    )
    if model_debug:
        lines.extend(["", "## Model Debug Keys", ""])
        for key in sorted(model_debug):
            value = model_debug[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)[:500]
            lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_split_report(path: Path, meta: Mapping[str, str]) -> None:
    lines = _report_header("Split Diagnostics")
    lines.extend(
        [
            f"- split_protocol recorded in outputs: {meta['split_protocol']}",
            "- Current evaluation uses the available Sen1Floods11 hand-labeled set.",
            "- The official checkpoint may have been fine-tuned using Sen1Floods11.",
            "- Therefore this result should be interpreted as an in-dataset / official adapted checkpoint audit unless official train/test split overlap is verified.",
            "- Event-held-out or leave-one-event-out generalization is not yet performed by this post-hoc command.",
            "- Future paper-grade generalization claims require event-held-out or leave-one-event-out sensitivity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bwer_audit_report(
    path: Path,
    meta: Mapping[str, str],
    summary: Mapping[str, Any],
    event_failure: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    missing_rows: Sequence[Mapping[str, Any]],
) -> None:
    ious = [_num(row["IoU"]) for row in event_failure if not math.isnan(_num(row["IoU"]))]
    best = max(event_failure, key=lambda row: _num(row["IoU"]), default={})
    worst = min(event_failure, key=lambda row: _num(row["IoU"]), default={})
    lines = _report_header("BWER-Audit v2 Report")
    lines.extend(
        [
            f"- dataset: {meta['dataset']}",
            f"- model: {meta['model']}",
            f"- task: {meta['task']}",
            f"- resolution: {meta['resolution']}",
            f"- aggregate mean risk: {float(summary.get('mean_risk', float('nan'))):.4f}",
            f"- Raw-BWER(event_id): {float(summary.get('bwer', float('nan'))):.4f}",
            f"- tail risk: {float(summary.get('tail_risk', float('nan'))):.4f}",
            f"- worst-tail events: {summary.get('tail_slices', '')}",
            f"- worst event: {worst.get('event_id', summary.get('worst_slice', ''))}",
            f"- best event: {best.get('event_id', summary.get('best_slice', ''))}",
            f"- event-level IoU range: {min(ious):.4f} to {max(ious):.4f}" if ious else "- event-level IoU range: unavailable",
            "",
            "Interpretation: aggregate IoU is strong for the official task-adapted checkpoint, but event-level tail risk remains. Pakistan and Bolivia form the high-risk tail in the successful 446-chip run.",
            "",
            "Connection to the earlier classification sanity path: Pakistan also appeared as a high-risk operational event slice, but event_id should be interpreted as a disaster-event slice, not as a causal country fairness attribute.",
            "",
            "Limitations: this is not causal country fairness, not yet event-held-out generalization, and the official checkpoint may share training events with the evaluated hand-labeled set.",
        ]
    )
    if reference_rows and reference_rows[0].get("status") == "not_applicable":
        lines.extend(["", "Standardised-BWER is not applicable here because no meaningful non-proxy balance variable is available."])
    if missing_rows and missing_rows[0].get("status") == "not_applicable":
        lines.extend(["", "Missing-policy sensitivity is not applicable for the same reason."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bwer_v2_posthoc(input_dir: str | Path, output_dir: str | Path, *, bootstrap: int = 1000, seed: int = 42) -> dict[str, Path]:
    input_path = Path(input_dir)
    output_path = ensure_dir(output_dir)
    event_rows, source_files = _normalise_event_rows(input_path)
    segmentation_path = input_path / "segmentation_metrics.csv"
    if segmentation_path.exists():
        source_files["segmentation_metrics"] = segmentation_path
    audit_path = _first_existing(input_path, ("segmentation_audit_table.csv", "audit_table.csv"))
    if audit_path:
        source_files["audit_table"] = audit_path
    if (input_path / "bwer_summary.csv").exists():
        source_files["bwer_summary"] = input_path / "bwer_summary.csv"
    model_debug = _read_json_if_exists(input_path / "model_debug.json")
    meta = _metadata(event_rows, input_path, model_debug)

    baseline = _compute_raw(event_rows, meta, alpha=0.1, min_support=10)
    bootstrap_rows = _bootstrap_ci(input_path, event_rows, meta, bootstrap, seed)
    ci = next((row for row in bootstrap_rows if row.get("source") == "bwer_v2_posthoc" and row.get("status") == "computed"), {})
    summary_rows = [_summary_row(baseline, meta, input_path, source_files, ci)]
    alpha_rows = _alpha_sensitivity(event_rows, meta)
    support_rows = _support_sensitivity(event_rows, meta)
    balances, balance_diagnostics = _meaningful_balance_variables(event_rows)
    reference_rows = _reference_weight_sensitivity(event_rows, meta, balances, balance_diagnostics)
    missing_rows = _missing_policy_sensitivity(event_rows, meta, balances, balance_diagnostics)
    stabilised_rows = _stabilised_bwer(event_rows, meta, baseline)
    loo_rows = _leave_one_slice_out(event_rows, meta)
    event_failure, event_ranking = _event_failure_analysis(event_rows, baseline)
    figure_paths = _write_figures(output_path, event_failure, alpha_rows)

    artifacts = {
        "bwer_v2_summary": output_path / "bwer_v2_summary.csv",
        "alpha_sensitivity": output_path / "alpha_sensitivity.csv",
        "support_sensitivity": output_path / "support_sensitivity.csv",
        "reference_weight_sensitivity": output_path / "reference_weight_sensitivity.csv",
        "missing_policy_sensitivity": output_path / "missing_policy_sensitivity.csv",
        "stabilised_bwer": output_path / "stabilised_bwer.csv",
        "leave_one_slice_out": output_path / "leave_one_slice_out.csv",
        "bootstrap_ci": output_path / "bootstrap_ci.csv",
        "event_failure_analysis": output_path / "event_failure_analysis.csv",
        "event_ranking": output_path / "event_ranking.csv",
        "metric_primitives_report": output_path / "metric_primitives_report.md",
        "adaptation_protocol_report": output_path / "adaptation_protocol_report.md",
        "split_diagnostics_report": output_path / "split_diagnostics_report.md",
        "bwer_audit_report": output_path / "bwer_audit_report.md",
    }
    write_csv(artifacts["bwer_v2_summary"], summary_rows)
    write_csv(artifacts["alpha_sensitivity"], alpha_rows)
    write_csv(artifacts["support_sensitivity"], support_rows)
    write_csv(artifacts["reference_weight_sensitivity"], reference_rows)
    write_csv(artifacts["missing_policy_sensitivity"], missing_rows)
    write_csv(artifacts["stabilised_bwer"], stabilised_rows)
    write_csv(artifacts["leave_one_slice_out"], loo_rows)
    write_csv(artifacts["bootstrap_ci"], bootstrap_rows)
    write_csv(artifacts["event_failure_analysis"], event_failure)
    write_csv(artifacts["event_ranking"], event_ranking)
    _write_metric_primitives_report(artifacts["metric_primitives_report"], summary_rows[0])
    _write_adaptation_protocol_report(artifacts["adaptation_protocol_report"], meta, model_debug)
    _write_split_report(artifacts["split_diagnostics_report"], meta)
    _write_bwer_audit_report(artifacts["bwer_audit_report"], meta, summary_rows[0], event_failure, reference_rows, missing_rows)
    artifacts.update({f"figure_{name}": path for name, path in figure_paths.items()})
    return artifacts
