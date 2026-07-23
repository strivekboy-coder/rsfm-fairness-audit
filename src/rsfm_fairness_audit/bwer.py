from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BWERConfig:
    dataset: str
    model: str
    task: str
    split: str = "all"
    score_name: str = "score"
    risk_name: str = "risk"
    tail_fraction: float = 0.10
    weighting: str = "uniform"
    min_samples_per_slice: int = 1
    min_positive_support: int | None = None
    min_valid_pixel_support: int | None = None
    min_slices_required: int = 2
    min_units_required: int = 1
    missing_balance_policy: str = "renormalize"
    bootstrap_n: int = 0
    bootstrap_method: str = "none"
    cluster_key: str | None = None
    selective_coverage: float | None = None
    seed: int = 42


@dataclass(frozen=True)
class BWERComputation:
    summary: dict[str, Any]
    by_slice: list[dict[str, Any]]
    support_diagnostics: list[dict[str, Any]]
    warnings: list[str]


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or str(value).lower() in {"nan", "none", "null"}


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if _is_missing(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_segmentation_task(task: str) -> bool:
    return "segmentation" in str(task).lower()


def _is_multilabel_task(task: str) -> bool:
    text = str(task).lower()
    return "multilabel" in text or "multi-label" in text


def _sum_column(rows: Sequence[Mapping[str, Any]], *names: str) -> float:
    total = 0.0
    found = False
    for row in rows:
        for name in names:
            if name in row and not _is_missing(row.get(name)):
                total += max(0.0, _as_float(row.get(name), 0.0))
                found = True
                break
    return total if found else float("nan")


def _segmentation_counts_available(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    columns = set().union(*(row.keys() for row in rows))
    return {"TP", "FP", "FN"}.issubset(columns) or {"tp", "fp", "fn"}.issubset(columns)


def _segmentation_score_and_risk(
    rows: Sequence[Mapping[str, Any]],
    score_column: str | None = None,
    risk_column: str | None = None,
) -> tuple[float, float, str]:
    tp = _sum_column(rows, "TP", "tp", "true_positive", "true_positives")
    fp = _sum_column(rows, "FP", "fp", "false_positive", "false_positives")
    fn = _sum_column(rows, "FN", "fn", "false_negative", "false_negatives")
    if math.isnan(tp) or math.isnan(fp) or math.isnan(fn):
        risks = [_get_score_and_risk(row, score_column, risk_column)[1] for row in rows]
        risk = float(np.nanmean(np.asarray(risks, dtype=float)))
        return 1.0 - risk, risk, risk_column or "risk"
    iou_den = tp + fp + fn
    dice_den = (2.0 * tp) + fp + fn
    iou = 1.0 if iou_den == 0 else float(tp / iou_den)
    dice = 1.0 if dice_den == 0 else float((2.0 * tp) / dice_den)
    name = (risk_column or score_column or "iou").lower()
    if "dice" in name or name in {"f1", "f1_risk"}:
        return dice, 1.0 - dice, "1_minus_dice"
    return iou, 1.0 - iou, "1_minus_iou"


def _get_score_and_risk(row: Mapping[str, Any], score_column: str | None, risk_column: str | None) -> tuple[float, float]:
    if risk_column and risk_column in row and not _is_missing(row.get(risk_column)):
        risk = _as_float(row.get(risk_column))
        return 1.0 - risk, risk
    candidates = [score_column, "score", "correct", "water_iou", "iou", "dice", "f1", "accuracy"]
    for key in candidates:
        if key and key in row and not _is_missing(row.get(key)):
            score = _as_float(row.get(key))
            return score, 1.0 - score
    if "y_true" in row and "y_pred" in row:
        score = float(str(row.get("y_true")) == str(row.get("y_pred")))
        return score, 1.0 - score
    if "label" in row and "prediction" in row:
        score = float(str(row.get("label")) == str(row.get("prediction")))
        return score, 1.0 - score
    raise ValueError("Audit row must contain score/risk, correct, y_true/y_pred, or label/prediction.")


def _positive_support(rows: Sequence[Mapping[str, Any]]) -> int:
    if rows and any(key in rows[0] for key in ["TP", "tp", "FN", "fn"]):
        return int(_sum_column(rows, "TP", "tp") + _sum_column(rows, "FN", "fn"))
    if rows and any(key in rows[0] for key in ["positive_pixel_count", "positive_pixels", "n_positive_pixels"]):
        return int(_sum_column(rows, "positive_pixel_count", "positive_pixels", "n_positive_pixels"))
    if rows and "n_positive" in rows[0]:
        return int(sum(max(0.0, _as_float(row.get("n_positive"), 0.0)) for row in rows))
    count = 0
    for row in rows:
        value = row.get("class_label", row.get("label", row.get("y_true")))
        if str(value) in {"1", "water", "flood", "foreground", "positive", "True", "true"}:
            count += 1
    return count


def _valid_pixel_support(rows: Sequence[Mapping[str, Any]]) -> int:
    total = _sum_column(rows, "valid_pixel_count", "valid_pixels", "n_valid_pixels")
    if not math.isnan(total):
        return int(total)
    if rows and any(key in rows[0] for key in ["TP", "FP", "FN", "TN", "tp", "fp", "fn", "tn"]):
        values = [
            _sum_column(rows, "TP", "tp"),
            _sum_column(rows, "FP", "fp"),
            _sum_column(rows, "FN", "fn"),
            _sum_column(rows, "TN", "tn"),
        ]
        return int(sum(0.0 if math.isnan(value) else value for value in values))
    return len(rows)


def _predicted_positive_support(rows: Sequence[Mapping[str, Any]]) -> int:
    total = _sum_column(rows, "predicted_positive_pixel_count", "predicted_positive_pixels", "n_predicted_positive_pixels")
    if not math.isnan(total):
        return int(total)
    if rows and any(key in rows[0] for key in ["TP", "FP", "tp", "fp"]):
        return int(_sum_column(rows, "TP", "tp") + _sum_column(rows, "FP", "fp"))
    return 0


def _effective_support(rows: Sequence[Mapping[str, Any]], config: BWERConfig) -> int:
    if _is_segmentation_task(config.task):
        return _valid_pixel_support(rows)
    return len(rows)


def _sample_count(rows: Sequence[Mapping[str, Any]]) -> int:
    total = _sum_column(rows, "sample_count", "n_samples")
    if math.isnan(total) or total <= 0:
        return len(rows)
    return int(total)


def _common_value(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = {str(row.get(key)) for row in rows if key in row and not _is_missing(row.get(key))}
    if len(values) == 1:
        return next(iter(values))
    return ""


def _column_values(rows: Sequence[Mapping[str, Any]], column: str) -> list[str]:
    return [str(row.get(column)) for row in rows if column in row and not _is_missing(row.get(column))]


def is_invalid_balance_variable(rows: Sequence[Mapping[str, Any]], slice_variable: str, balance_variable: str | None) -> tuple[bool, str]:
    if not balance_variable:
        return False, ""
    if balance_variable == slice_variable:
        return True, "balance variable is identical to slice variable"
    if not rows or slice_variable not in rows[0] or balance_variable not in rows[0]:
        return False, ""
    slice_values = _column_values(rows, slice_variable)
    balance_values = _column_values(rows, balance_variable)
    if slice_values and slice_values == balance_values:
        return True, "balance variable has identical row values to slice variable"
    forward: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    for row in rows:
        if _is_missing(row.get(slice_variable)) or _is_missing(row.get(balance_variable)):
            continue
        g = str(row.get(slice_variable))
        z = str(row.get(balance_variable))
        forward.setdefault(g, set()).add(z)
        reverse.setdefault(z, set()).add(g)
    if forward and all(len(values) == 1 for values in forward.values()) and all(len(values) == 1 for values in reverse.values()):
        return True, "balance variable is a deterministic proxy of slice variable"
    return False, ""


def _group_by(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = row.get(column)
        if _is_missing(value):
            continue
        grouped.setdefault(str(value), []).append(row)
    return grouped


def _balance_weights(rows: Sequence[Mapping[str, Any]], balance_variable: str, weighting: str) -> dict[str, float]:
    grouped = _group_by(rows, balance_variable)
    if not grouped:
        return {}
    if weighting == "empirical":
        total = sum(len(items) for items in grouped.values())
        return {value: len(items) / total for value, items in grouped.items()}
    if weighting != "uniform":
        raise ValueError(f"Unsupported BWER weighting={weighting!r}. Use 'uniform' or 'empirical'.")
    weight = 1.0 / len(grouped)
    return {value: weight for value in grouped}


def _weighted_available(values: dict[str, float], weights: dict[str, float], allowed_levels: set[str] | None = None) -> tuple[float, list[str], int]:
    required = set(weights)
    if allowed_levels is not None:
        required = required & allowed_levels
    available = {key: weights[key] for key in values if key in required and not math.isnan(values[key])}
    missing = sorted(required - set(values))
    if not available:
        return float("nan"), missing, 0
    total = sum(available.values())
    return float(sum(values[key] * weight / total for key, weight in available.items())), missing, len(available)


def compute_slice_scores(
    audit_rows: Sequence[Mapping[str, Any]],
    config: BWERConfig,
    slice_variable: str,
    balance_variable: str | None = None,
    score_column: str | None = None,
    risk_column: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    warnings: list[str] = []
    support_diagnostics: list[dict[str, Any]] = []
    balance_stats = {
        "n_total_balance_levels": 0,
        "n_used_balance_levels": 0,
        "missing_gz_count": 0,
        "missing_gz_fraction": 0.0,
    }
    if not audit_rows:
        return [], [], ["Audit table is empty."], balance_stats
    if slice_variable not in audit_rows[0]:
        return [], [], [f"Missing slice column: {slice_variable}"], balance_stats
    if balance_variable and balance_variable not in audit_rows[0]:
        return [], [], [f"Missing balance column: {balance_variable}"], balance_stats
    invalid_balance, invalid_reason = is_invalid_balance_variable(audit_rows, slice_variable, balance_variable)
    if invalid_balance:
        return [], [], [f"Invalid BWER({slice_variable} | {balance_variable}): {invalid_reason}."], balance_stats
    if config.missing_balance_policy not in {"renormalize", "invalidate", "overlap"}:
        raise ValueError("missing_balance_policy must be one of: renormalize, invalidate, overlap.")

    prepared = []
    use_segmentation_counts = _is_segmentation_task(config.task) and _segmentation_counts_available(audit_rows)
    for row in audit_rows:
        if _is_missing(row.get(slice_variable)):
            continue
        if use_segmentation_counts:
            score, risk, _ = _segmentation_score_and_risk([row], score_column or config.score_name, risk_column)
        else:
            score, risk = _get_score_and_risk(row, score_column or config.score_name, risk_column or config.risk_name)
        item = dict(row)
        item["_score"] = score
        item["_risk"] = risk
        prepared.append(item)

    groups = _group_by(prepared, slice_variable)
    weights = _balance_weights(prepared, balance_variable, config.weighting) if balance_variable else {}
    required_levels = set(weights)
    basic_valid_slices: set[str] = set()
    for slice_value, items in groups.items():
        n_positive = _positive_support(items)
        effective_support = _effective_support(items, config)
        sample_count = _sample_count(items)
        is_basic_valid = sample_count >= config.min_samples_per_slice and effective_support >= config.min_units_required
        if config.min_positive_support is not None:
            is_basic_valid = is_basic_valid and n_positive >= config.min_positive_support
        if config.min_valid_pixel_support is not None and _is_segmentation_task(config.task):
            is_basic_valid = is_basic_valid and _valid_pixel_support(items) >= config.min_valid_pixel_support
        if is_basic_valid:
            basic_valid_slices.add(slice_value)
    observed_by_slice: dict[str, set[str]] = {}
    if balance_variable:
        for slice_value, items in groups.items():
            observed_by_slice[slice_value] = set(_group_by(items, balance_variable))
        if config.missing_balance_policy == "overlap" and observed_by_slice:
            # For paper-grade sensitivity analysis, the shared overlap set is
            # derived only from slices that pass basic support thresholds.
            # Tiny invalid slices should be reported, but must not shrink the
            # balance levels used by otherwise valid slices.
            candidate_observed = [levels for slice_value, levels in observed_by_slice.items() if slice_value in basic_valid_slices]
            required_levels = set.intersection(*candidate_observed) if candidate_observed else set()
            if not required_levels:
                warnings.append(f"No shared {balance_variable} levels across support-valid slices for overlap policy.")
            else:
                warnings.append(
                    f"Overlap policy uses {len(required_levels)} shared {balance_variable} levels computed after basic slice support filtering."
                )
    total_gz = len(groups) * len(weights) if balance_variable else 0
    missing_gz = 0
    used_levels: set[str] = set()
    rows: list[dict[str, Any]] = []
    for slice_value, items in sorted(groups.items(), key=lambda item: str(item[0])):
        risks = np.asarray([_as_float(row["_risk"]) for row in items], dtype=float)
        scores = np.asarray([_as_float(row["_score"]) for row in items], dtype=float)
        n_positive = _positive_support(items)
        valid_pixel_support = _valid_pixel_support(items) if _is_segmentation_task(config.task) else ""
        predicted_positive_support = _predicted_positive_support(items) if _is_segmentation_task(config.task) else ""
        effective_support = _effective_support(items, config)
        sample_count = _sample_count(items)
        if use_segmentation_counts:
            raw_score, raw_risk, risk_source = _segmentation_score_and_risk(items, score_column or config.score_name, risk_column)
        else:
            raw_score = float(np.nanmean(scores))
            raw_risk = float(np.nanmean(risks))
            risk_source = risk_column or config.risk_name
        balanced_risk = raw_risk
        support_warning = ""
        missing: list[str] = []
        if balance_variable:
            z_groups = _group_by(items, balance_variable)
            for level in sorted(weights):
                has_support = level in z_groups and len(z_groups[level]) > 0
                used = has_support and level in required_levels
                if not has_support:
                    missing_gz += 1
                if used:
                    used_levels.add(level)
                support_diagnostics.append(
                    {
                        "dataset": config.dataset,
                        "model": config.model,
                        "task": config.task,
                        "slice_variable": slice_variable,
                        "balance_variable": balance_variable,
                        "slice_value": slice_value,
                        "balance_level": level,
                        "n_units": _effective_support(z_groups.get(level, []), config),
                        "sample_count": _sample_count(z_groups.get(level, [])),
                        "valid_pixel_support": _valid_pixel_support(z_groups.get(level, [])) if _is_segmentation_task(config.task) else "",
                        "positive_pixel_support": _positive_support(z_groups.get(level, [])) if _is_segmentation_task(config.task) else "",
                        "has_support": bool(has_support),
                        "used_in_balanced_risk": bool(used),
                        "missing_balance_policy": config.missing_balance_policy,
                    }
                )
            if use_segmentation_counts:
                z_risks = {z: _segmentation_score_and_risk(z_items, score_column or config.score_name, risk_column)[1] for z, z_items in z_groups.items()}
            else:
                z_risks = {z: float(np.nanmean([_as_float(row["_risk"]) for row in z_items])) for z, z_items in z_groups.items()}
            allowed = required_levels if config.missing_balance_policy == "overlap" else None
            balanced_risk, missing, n_used = _weighted_available(z_risks, weights, allowed)
            if missing:
                support_warning = f"missing_balance_levels={','.join(missing)}"
                warnings.append(f"Slice {slice_variable}={slice_value} missing {balance_variable} levels: {', '.join(missing)}")
            if config.missing_balance_policy == "invalidate" and missing:
                balanced_risk = float("nan")
                n_used = 0
            balance_stats["n_used_balance_levels"] = max(balance_stats["n_used_balance_levels"], n_used)
        is_valid = slice_value in basic_valid_slices
        if balance_variable and config.missing_balance_policy == "invalidate" and missing:
            is_valid = False
        if balance_variable and config.missing_balance_policy == "overlap" and not required_levels:
            is_valid = False
        if config.min_positive_support is not None:
            is_valid = is_valid and n_positive >= config.min_positive_support
        if config.min_valid_pixel_support is not None and _is_segmentation_task(config.task):
            is_valid = is_valid and _valid_pixel_support(items) >= config.min_valid_pixel_support
        if not is_valid and not support_warning:
            support_warning = "below_support_threshold"
        rows.append(
            {
                "dataset": config.dataset,
                "model": config.model,
                "task": config.task,
                "split": config.split,
                "slice_variable": slice_variable,
                "balance_variable": balance_variable or "",
                "slice_value": slice_value,
                "n_units": effective_support,
                "sample_count": sample_count,
                "n_positive": n_positive,
                "valid_pixel_support": valid_pixel_support,
                "positive_pixel_support": n_positive if _is_segmentation_task(config.task) else "",
                "predicted_positive_support": predicted_positive_support,
                "risk_source": risk_source,
                "raw_score": raw_score,
                "raw_risk": raw_risk,
                "balanced_risk": balanced_risk,
                "is_valid_slice": bool(is_valid),
                "is_tail_slice": False,
                "support_warning": support_warning,
                "rank_by_risk": "",
                "ci_low": "",
                "ci_high": "",
            }
        )
    balance_stats["n_total_balance_levels"] = len(weights)
    if config.missing_balance_policy == "overlap":
        balance_stats["n_used_balance_levels"] = len(required_levels)
    elif balance_variable and not balance_stats["n_used_balance_levels"]:
        balance_stats["n_used_balance_levels"] = len(used_levels)
    balance_stats["missing_gz_count"] = missing_gz
    balance_stats["missing_gz_fraction"] = float(missing_gz / total_gz) if total_gz else 0.0
    return rows, support_diagnostics, warnings, balance_stats


def compute_tail_risk(slice_rows: Sequence[Mapping[str, Any]], tail_fraction: float) -> tuple[float, list[str]]:
    valid = [row for row in slice_rows if bool(row.get("is_valid_slice")) and not math.isnan(_as_float(row.get("balanced_risk")))]
    if not valid:
        return float("nan"), []
    tail_n = max(1, int(math.ceil(len(valid) * tail_fraction)))
    ranked = sorted(valid, key=lambda row: (-_as_float(row.get("balanced_risk")), str(row.get("slice_value"))))
    tail = ranked[:tail_n]
    return float(np.mean([_as_float(row.get("balanced_risk")) for row in tail])), [str(row.get("slice_value")) for row in tail]


def compute_max_bwer(slice_rows: Sequence[Mapping[str, Any]]) -> float:
    valid = [row for row in slice_rows if bool(row.get("is_valid_slice")) and not math.isnan(_as_float(row.get("balanced_risk")))]
    if not valid:
        return float("nan")
    risks = np.asarray([_as_float(row.get("balanced_risk")) for row in valid], dtype=float)
    return float(np.max(risks) - np.mean(risks))


def compute_bwer(
    audit_rows: Sequence[Mapping[str, Any]],
    config: BWERConfig,
    slice_variable: str,
    balance_variable: str | None = None,
    score_column: str | None = None,
    risk_column: str | None = None,
) -> BWERComputation:
    by_slice, support_diagnostics, warnings, balance_stats = compute_slice_scores(audit_rows, config, slice_variable, balance_variable, score_column, risk_column)
    valid = [row for row in by_slice if bool(row.get("is_valid_slice")) and not math.isnan(_as_float(row.get("balanced_risk")))]
    if len(valid) < config.min_slices_required:
        warnings.append(f"Only {len(valid)} valid slices for {slice_variable}; min_slices_required={config.min_slices_required}.")
    ranked = sorted(valid, key=lambda row: (-_as_float(row.get("balanced_risk")), str(row.get("slice_value"))))
    for rank, row in enumerate(ranked, start=1):
        for target in by_slice:
            if target["slice_value"] == row["slice_value"]:
                target["rank_by_risk"] = rank
                break
    tail_risk, tail_slices = compute_tail_risk(by_slice, config.tail_fraction)
    for row in by_slice:
        row["is_tail_slice"] = str(row["slice_value"]) in set(tail_slices)
    risks = np.asarray([_as_float(row.get("balanced_risk")) for row in valid], dtype=float)
    mean_risk = float(np.mean(risks)) if len(risks) else float("nan")
    bwer = float(tail_risk - mean_risk) if not math.isnan(tail_risk) and not math.isnan(mean_risk) else float("nan")
    best = min(valid, key=lambda row: (_as_float(row.get("balanced_risk")), str(row.get("slice_value"))), default={})
    worst = max(valid, key=lambda row: (_as_float(row.get("balanced_risk")), str(row.get("slice_value"))), default={})
    summary = {
        "dataset": config.dataset,
        "model": config.model,
        "task": config.task,
        "split": config.split,
        "slice_variable": slice_variable,
        "balance_variable": balance_variable or "",
        "score_name": score_column or config.score_name,
        "risk_name": risk_column or config.risk_name,
        "tail_fraction": config.tail_fraction,
        "weighting": config.weighting,
        "missing_balance_policy": config.missing_balance_policy,
        "bwer": bwer,
        "tail_risk": tail_risk,
        "mean_risk": mean_risk,
        "max_bwer": compute_max_bwer(by_slice),
        "best_slice": best.get("slice_value", ""),
        "best_slice_risk": best.get("balanced_risk", ""),
        "worst_slice": worst.get("slice_value", ""),
        "worst_slice_risk": worst.get("balanced_risk", ""),
        "n_slices_total": len(by_slice),
        "n_slices_valid": len(valid),
        "n_units": len(audit_rows),
        "min_samples_per_slice": config.min_samples_per_slice,
        "n_total_balance_levels": balance_stats["n_total_balance_levels"],
        "n_used_balance_levels": balance_stats["n_used_balance_levels"],
        "missing_gz_count": balance_stats["missing_gz_count"],
        "missing_gz_fraction": balance_stats["missing_gz_fraction"],
        "ci_low": "",
        "ci_high": "",
        "bootstrap_n": config.bootstrap_n,
        "bootstrap_method": config.bootstrap_method,
        "tail_slices": ";".join(tail_slices),
        "task_type": "segmentation" if _is_segmentation_task(config.task) else "multilabel_classification" if _is_multilabel_task(config.task) else "classification",
        "input_mode": _common_value(audit_rows, "input_mode"),
        "adaptation_protocol": _common_value(audit_rows, "adaptation_protocol"),
        "training_budget": _common_value(audit_rows, "training_budget") or _common_value(audit_rows, "training_setup"),
        "split_protocol": _common_value(audit_rows, "split_protocol"),
        "selective_coverage": "" if config.selective_coverage is None else config.selective_coverage,
        "retained_coverage": "",
        "abstention_rate": "",
        "warnings": " | ".join(sorted(set(warnings))),
    }
    return BWERComputation(summary=summary, by_slice=by_slice, support_diagnostics=support_diagnostics, warnings=warnings)


def bootstrap_bwer(
    audit_rows: Sequence[Mapping[str, Any]],
    config: BWERConfig,
    slice_variable: str,
    balance_variable: str | None = None,
    n_bootstrap: int = 1000,
    cluster_key: str | None = None,
    seed: int = 42,
    score_column: str | None = None,
    risk_column: str | None = None,
) -> dict[str, Any]:
    if n_bootstrap <= 0 or not audit_rows:
        return {"ci_low": "", "ci_high": "", "bootstrap_n": 0, "bootstrap_method": "none", "warnings": ""}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    warnings: list[str] = []
    rows = [dict(row) for row in audit_rows]
    for _ in range(n_bootstrap):
        if cluster_key and cluster_key in rows[0]:
            clusters = sorted({str(row.get(cluster_key)) for row in rows if not _is_missing(row.get(cluster_key))})
            if len(clusters) < 2:
                warnings.append("too_few_clusters")
                break
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            sample = []
            for draw_index, cluster in enumerate(sampled):
                # Every sampled cluster is a distinct bootstrap clone. Keeping
                # the original ID makes repeated draws collapse in downstream
                # group-by operations when cluster_key is also the slice key.
                clone_id = f"{cluster}__bootstrap_clone_{draw_index}"
                for source in rows:
                    if str(source.get(cluster_key)) != str(cluster):
                        continue
                    row = dict(source)
                    row["_bootstrap_source_cluster"] = str(cluster)
                    row[cluster_key] = clone_id
                    sample.append(row)
        else:
            indices = rng.integers(0, len(rows), size=len(rows))
            sample = [rows[int(index)] for index in indices]
        try:
            result = compute_bwer(sample, config, slice_variable, balance_variable, score_column, risk_column)
            value = _as_float(result.summary.get("bwer"))
            if not math.isnan(value):
                values.append(value)
        except Exception as exc:  # pragma: no cover - defensive; result recorded as warning
            warnings.append(str(exc))
    if not values:
        return {"ci_low": "", "ci_high": "", "bootstrap_n": 0, "bootstrap_method": "cluster" if cluster_key else "ordinary", "warnings": ";".join(sorted(set(warnings)))}
    return {
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "bootstrap_n": len(values),
        "bootstrap_method": "cluster" if cluster_key else "ordinary",
        "warnings": ";".join(sorted(set(warnings))),
    }


def create_interaction_slice(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], name: str | None = None) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    target = name or "__".join(columns)
    for row in output:
        if all(column in row and not _is_missing(row.get(column)) for column in columns):
            row[target] = "__".join(str(row.get(column)) for column in columns)
    return output


def compute_bwer_family(
    audit_rows: Sequence[Mapping[str, Any]],
    config: BWERConfig,
    slice_variables: Sequence[str],
    balance_variables: Sequence[str | None],
    n_bootstrap: int = 0,
    cluster_key: str | None = None,
    score_column: str | None = None,
    risk_column: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    summary_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    columns = set(audit_rows[0].keys()) if audit_rows else set()
    for slice_variable in slice_variables:
        if slice_variable not in columns:
            warnings.append(f"Skipping missing slice variable: {slice_variable}")
            continue
        for balance_variable in balance_variables:
            if balance_variable == slice_variable:
                warnings.append(f"Skipping invalid BWER({slice_variable} | {balance_variable}): balance variable is identical to slice variable.")
                continue
            if balance_variable and balance_variable not in columns:
                warnings.append(f"Skipping {slice_variable}|{balance_variable}: missing balance variable.")
                continue
            invalid_balance, invalid_reason = is_invalid_balance_variable(audit_rows, slice_variable, balance_variable)
            if invalid_balance:
                warnings.append(f"Skipping invalid BWER({slice_variable} | {balance_variable}): {invalid_reason}.")
                continue
            result = compute_bwer(audit_rows, config, slice_variable, balance_variable, score_column, risk_column)
            summary = dict(result.summary)
            if n_bootstrap:
                ci = bootstrap_bwer(
                    audit_rows,
                    config,
                    slice_variable,
                    balance_variable,
                    n_bootstrap=n_bootstrap,
                    cluster_key=cluster_key,
                    seed=config.seed,
                    score_column=score_column,
                    risk_column=risk_column,
                )
                summary.update({key: ci[key] for key in ["ci_low", "ci_high", "bootstrap_n", "bootstrap_method"]})
                ci_row = dict(summary)
                ci_row["bootstrap_warnings"] = ci.get("warnings", "")
                ci_rows.append(ci_row)
                if ci.get("warnings"):
                    warnings.append(f"Bootstrap warning for {slice_variable}|{balance_variable or 'raw'}: {ci['warnings']}")
            summary_rows.append(summary)
            slice_rows.extend(result.by_slice)
            support_rows.extend(result.support_diagnostics)
            warnings.extend(result.warnings)
    return summary_rows, slice_rows, support_rows, ci_rows, sorted(set(warnings))
