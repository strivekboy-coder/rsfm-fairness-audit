"""Post-hoc analyses requested for the paper-facing GeoBWER synthesis.

The functions in this module operate only on frozen predictions/audit tables.
They deliberately keep data extraction, statistical summaries, and evidence
status separate so that unavailable metadata is reported rather than imputed.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class SupplementaryAnalysisError(RuntimeError):
    """Raised when a frozen-analysis contract cannot be verified."""


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def compute_fractional_mtd(
    risks: Mapping[str, float],
    *,
    beta: float = 0.10,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute deployment mean, exact fractional beta tail, and D=T-M."""
    if not 0 < beta <= 1:
        raise ValueError("beta must lie in (0, 1].")
    clean = {str(k): float(v) for k, v in risks.items() if math.isfinite(float(v))}
    if not clean:
        raise ValueError("No finite slice risks supplied.")
    if weights is None:
        mass = {k: 1.0 / len(clean) for k in clean}
    else:
        if set(weights) != set(clean):
            raise ValueError("weights and risks must have identical slice keys.")
        total = sum(float(weights[k]) for k in clean)
        if total <= 0:
            raise ValueError("deployment weights must have positive mass.")
        mass = {k: float(weights[k]) / total for k in clean}
        if any(value < 0 for value in mass.values()):
            raise ValueError("deployment weights cannot be negative.")
    mean = sum(mass[k] * clean[k] for k in clean)
    remaining = beta
    tail_numerator = 0.0
    allocations: list[dict[str, float | str]] = []
    for key, risk in sorted(clean.items(), key=lambda item: (-item[1], item[0])):
        selected = min(mass[key], remaining)
        if selected > 0:
            allocations.append({"slice": key, "risk": risk, "selected_mass": selected})
            tail_numerator += selected * risk
            remaining -= selected
        if remaining <= 1e-12:
            break
    if remaining > 1e-9:
        raise ValueError("deployment mass was insufficient for the requested tail.")
    tail = tail_numerator / beta
    return {"M": mean, "T": tail, "D": tail - mean, "beta": beta, "tail_allocation": allocations}


def build_sen1_decision_scenario(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_family: str = "supervised_resnet34_unet",
    mean_choice_mode: str = "S1+S2",
    tail_choice_mode: str = "S2",
    beta: float = 0.10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the retrospective event comparison used to illustrate M-vs-T choice."""
    # Accept both the historical optimization atlas and the canonical
    # Sen1Floods11 event-level artifact.  The latter is the portable source used
    # by Colab; the atlas is a local, ignored derivative and is retained only
    # for backwards-compatible local calls.
    canonical = bool(rows) and "event_id" in rows[0] and "mean_chip_iou_risk" in rows[0]
    if canonical:
        selected = [
            row for row in rows
            if str(row.get("family")) == model_family
            and str(row.get("mode")) in {mean_choice_mode, tail_choice_mode}
            and str(row.get("split")) == "combined_held_out"
            and str(row.get("comparison_role")) == "same_grid_primary_panel"
        ]
    else:
        selected = [
            row for row in rows
            if str(row.get("dataset")) == "Sen1Floods11"
            and str(row.get("slice_axis")) == "event"
            and str(row.get("model_family")) == model_family
            and str(row.get("mode")) in {mean_choice_mode, tail_choice_mode}
            and _truthy(row.get("eligible_for_primary_metric", True))
        ]
    by_mode_event: dict[tuple[str, str], list[float]] = defaultdict(list)
    supports: dict[tuple[str, str], float] = {}
    for row in selected:
        risk = _float(row.get("mean_chip_iou_risk") if canonical else row.get("risk"))
        if not math.isfinite(risk):
            continue
        event = str(row["event_id"] if canonical else row["slice_value"])
        key = (str(row["mode"]), event)
        by_mode_event[key].append(risk)
        supports[key] = _float(row.get("auditable_sample_count") if canonical else row.get("support"))
    events = sorted({event for mode, event in by_mode_event if mode in {mean_choice_mode, tail_choice_mode}})
    if not events:
        raise SupplementaryAnalysisError("No matching Sen1 event rows were found.")
    detail: list[dict[str, Any]] = []
    mode_risks: dict[str, dict[str, float]] = {mean_choice_mode: {}, tail_choice_mode: {}}
    for event in events:
        values: dict[str, float] = {}
        for mode in (mean_choice_mode, tail_choice_mode):
            data = by_mode_event.get((mode, event), [])
            if not data:
                raise SupplementaryAnalysisError(f"Missing {mode} risk for event {event}.")
            values[mode] = float(np.mean(data))
            mode_risks[mode][event] = values[mode]
        delta = values[mean_choice_mode] - values[tail_choice_mode]
        detail.append({
            "event": event,
            "mean_selected_mode": mean_choice_mode,
            "tail_selected_mode": tail_choice_mode,
            "mean_selected_risk": values[mean_choice_mode],
            "tail_selected_risk": values[tail_choice_mode],
            "risk_delta_mean_minus_tail_choice": delta,
            "event_winner": mean_choice_mode if delta < 0 else tail_choice_mode if delta > 0 else "tie",
            "seed_count_mean_choice": len(by_mode_event[(mean_choice_mode, event)]),
            "seed_count_tail_choice": len(by_mode_event[(tail_choice_mode, event)]),
            "support_mean_choice": supports.get((mean_choice_mode, event), float("nan")),
            "support_tail_choice": supports.get((tail_choice_mode, event), float("nan")),
        })
    summary: list[dict[str, Any]] = []
    for role, mode in (("mean_selected", mean_choice_mode), ("tail_selected", tail_choice_mode)):
        card = compute_fractional_mtd(mode_risks[mode], beta=beta)
        summary.append({
            "selection_role": role,
            "mode": mode,
            "event_count": len(events),
            "M": card["M"], "T": card["T"], "D": card["D"], "beta": beta,
            "events_with_lower_risk": sum(row["event_winner"] == mode for row in detail),
            "interpretation": "retrospective observed comparison; not a prospective utility policy",
        })
    return detail, summary


def build_sen1_consensus_event_risk(
    rows: Sequence[Mapping[str, Any]],
    *,
    families: Sequence[str] = ("supervised_resnet34_unet", "terramind_v1_base"),
    modes: Sequence[str] = ("S1", "S2", "S1+S2"),
) -> list[dict[str, Any]]:
    """Average canonical held-out event risks over the frozen 6-cell panel.

    Every event must contain three seeds for each family-by-modality cell.  The
    strict check prevents incomplete Drive copies from silently changing the
    consensus risk used by the descriptive event-association analysis.
    """
    selected = [
        row for row in rows
        if str(row.get("family")) in set(families)
        and str(row.get("mode")) in set(modes)
        and str(row.get("split")) == "combined_held_out"
        and str(row.get("comparison_role")) == "same_grid_primary_panel"
    ]
    expected_cells = {(family, mode) for family in families for mode in modes}
    by_event_cell: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    supports: dict[str, set[int]] = defaultdict(set)
    for row in selected:
        event = str(row.get("event_id", ""))
        family = str(row.get("family", ""))
        mode = str(row.get("mode", ""))
        seed = str(row.get("seed", ""))
        risk = _float(row.get("mean_chip_iou_risk"))
        support = _float(row.get("auditable_sample_count"))
        if event and seed and math.isfinite(risk):
            by_event_cell[(event, family, mode)][seed] = risk
        if event and math.isfinite(support):
            supports[event].add(int(support))
    events = sorted({key[0] for key in by_event_cell})
    if len(events) != 11:
        raise SupplementaryAnalysisError(f"Expected 11 canonical Sen1 events; found {len(events)}.")
    output: list[dict[str, Any]] = []
    for event in events:
        cell_values = []
        for family, mode in sorted(expected_cells):
            seed_values = by_event_cell.get((event, family, mode), {})
            if set(seed_values) != {"42", "73", "101"}:
                raise SupplementaryAnalysisError(
                    f"Incomplete canonical Sen1 panel for {event}/{family}/{mode}: "
                    f"seeds={sorted(seed_values)}"
                )
            cell_values.extend(seed_values.values())
        if len(supports[event]) != 1:
            raise SupplementaryAnalysisError(
                f"Inconsistent support for canonical Sen1 event {event}: {sorted(supports[event])}"
            )
        output.append({
            "event_id": event,
            "consensus_event_risk": float(np.mean(cell_values)),
            "panel_cell_count": len(expected_cells),
            "replicate_count": len(cell_values),
            "auditable_sample_count": next(iter(supports[event])),
        })
    return output


def summarize_sen1_validation_locked_mtd(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_family: str = "supervised_resnet34_unet",
    modes: Sequence[str] = ("s1_plus_s2", "s2"),
) -> dict[str, dict[str, float]]:
    """Summarize canonical validation-locked Sen1 M/T/D over three seeds."""
    selected = [
        row for row in rows
        if str(row.get("family")) == model_family
        and str(row.get("mode")).lower() in set(modes)
        and str(row.get("split")) == "combined_held_out"
        and _truthy(row.get("is_validation_selected_operating_point"))
    ]
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_mode[str(row.get("mode")).lower()].append(row)
    output: dict[str, dict[str, float]] = {}
    metric_columns = {"M": "event_mean_risk", "T": "event_tail_risk", "D": "event_geobwer"}
    for mode in modes:
        mode_rows = by_mode.get(mode, [])
        seeds = {str(row.get("seed")) for row in mode_rows}
        if seeds != {"42", "73", "101"} or len(mode_rows) != 3:
            raise SupplementaryAnalysisError(
                f"Expected one validation-locked row for each seed in Sen1 mode {mode}; "
                f"found rows={len(mode_rows)}, seeds={sorted(seeds)}."
            )
        card: dict[str, float] = {}
        for metric, column in metric_columns.items():
            values = np.asarray([_float(row.get(column)) for row in mode_rows], dtype=float)
            if not np.isfinite(values).all():
                raise SupplementaryAnalysisError(f"Non-finite {column} in Sen1 mode {mode}.")
            card[metric] = float(np.mean(values))
            card[f"{metric}_seed_sd"] = float(np.std(values, ddof=1))
        output[mode] = card
    return output


def _rates(truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    tp = np.sum(truth & prediction, axis=0)
    fn = np.sum(truth & ~prediction, axis=0)
    fp = np.sum(~truth & prediction, axis=0)
    tn = np.sum(~truth & ~prediction, axis=0)
    return tp, fn, fp, tn


def summarize_paired_multilabel_arrays(
    truth: np.ndarray,
    id_prediction: np.ndarray,
    shifted_prediction: np.ndarray,
    label_names: Sequence[str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Summarize paired multi-label transitions without implying exclusive classes.

    The two returned matrices contain counts of target-label false negatives
    co-occurring with false positives in another label, before and after shift.
    """
    truth = np.asarray(truth, dtype=np.int8)
    id_prediction = np.asarray(id_prediction, dtype=np.int8)
    shifted_prediction = np.asarray(shifted_prediction, dtype=np.int8)
    if truth.shape != id_prediction.shape or truth.shape != shifted_prediction.shape:
        raise ValueError("truth and paired predictions must have identical shapes.")
    if truth.ndim != 2 or truth.shape[1] != len(label_names):
        raise ValueError("Expected [samples, labels] arrays matching label_names.")
    id_tp, id_fn, id_fp, id_tn = _rates(truth, id_prediction)
    sh_tp, sh_fn, sh_fp, sh_tn = _rates(truth, shifted_prediction)
    id_error = id_prediction != truth
    sh_error = shifted_prediction != truth
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(label_names):
        positive = id_tp[index] + id_fn[index]
        negative = id_fp[index] + id_tn[index]
        rows.append({
            "class_index": index, "class_label": name, "sample_count": truth.shape[0],
            "positive_support": int(positive), "negative_support": int(negative),
            "id_fnr": float(id_fn[index] / positive) if positive else float("nan"),
            "shift_fnr": float(sh_fn[index] / positive) if positive else float("nan"),
            "delta_fnr": float((sh_fn[index] - id_fn[index]) / positive) if positive else float("nan"),
            "id_fpr": float(id_fp[index] / negative) if negative else float("nan"),
            "shift_fpr": float(sh_fp[index] / negative) if negative else float("nan"),
            "delta_fpr": float((sh_fp[index] - id_fp[index]) / negative) if negative else float("nan"),
            "persistent_error_rate": float(np.mean(id_error[:, index] & sh_error[:, index])),
            "new_error_rate": float(np.mean(~id_error[:, index] & sh_error[:, index])),
            "repaired_error_rate": float(np.mean(id_error[:, index] & ~sh_error[:, index])),
        })
    id_fn_mask = (truth == 1) & (id_prediction == 0)
    sh_fn_mask = (truth == 1) & (shifted_prediction == 0)
    id_fp_mask = (truth == 0) & (id_prediction == 1)
    sh_fp_mask = (truth == 0) & (shifted_prediction == 1)
    return (
        rows,
        id_fn_mask.astype(np.int64).T @ id_fp_mask.astype(np.int64),
        sh_fn_mask.astype(np.int64).T @ sh_fp_mask.astype(np.int64),
    )


@dataclass
class ClusterSliceSufficient:
    cluster: str
    slice_value: str
    risk_sum: float
    count: int
    balance_value: str = ""


def cluster_preflight(records: Sequence[ClusterSliceSufficient], *, min_clusters_per_slice: int = 2) -> dict[str, Any]:
    clusters: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.count > 0 and math.isfinite(record.risk_sum):
            clusters[record.slice_value].add(record.cluster)
    counts = {key: len(value) for key, value in clusters.items()}
    insufficient = sorted(key for key, value in counts.items() if value < min_clusters_per_slice)
    return {
        "slice_count": len(counts),
        "cluster_count": len({r.cluster for r in records}),
        "clusters_per_slice": counts,
        "minimum_clusters_per_slice": min(counts.values()) if counts else 0,
        "insufficient_slices": insufficient,
        "status": "eligible" if counts and not insufficient else "insufficient_independent_clusters",
    }


def cluster_bootstrap_mtd(
    records: Sequence[ClusterSliceSufficient],
    *,
    beta: float = 0.10,
    n_boot: int = 2000,
    seed: int = 20260828,
    min_clusters_per_slice: int = 2,
) -> dict[str, Any]:
    """Cluster bootstrap of a complete group-risk vector and its M/T/D image."""
    preflight = cluster_preflight(records, min_clusters_per_slice=min_clusters_per_slice)
    if preflight["status"] != "eligible":
        return {**preflight, "valid_bootstrap_replicates": 0, "requested_bootstrap_replicates": n_boot}
    clusters = sorted({record.cluster for record in records})
    slices = sorted({record.slice_value for record in records})
    ci = {cluster: index for index, cluster in enumerate(clusters)}
    si = {value: index for index, value in enumerate(slices)}
    sums = np.zeros((len(clusters), len(slices)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for record in records:
        sums[ci[record.cluster], si[record.slice_value]] += record.risk_sum
        counts[ci[record.cluster], si[record.slice_value]] += record.count
    point_counts = counts.sum(axis=0)
    point_risks = sums.sum(axis=0) / point_counts
    point = compute_fractional_mtd(dict(zip(slices, point_risks)), beta=beta)
    rng = np.random.default_rng(seed)
    draws: list[tuple[float, float, float]] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        boot_counts = counts[chosen].sum(axis=0)
        if np.any(boot_counts <= 0):
            continue
        boot_risks = sums[chosen].sum(axis=0) / boot_counts
        card = compute_fractional_mtd(dict(zip(slices, boot_risks)), beta=beta)
        draws.append((float(card["M"]), float(card["T"]), float(card["D"])))
    result: dict[str, Any] = {
        **preflight,
        "status": "computed" if len(draws) >= max(100, int(0.8 * n_boot)) else "unstable_bootstrap_support",
        "requested_bootstrap_replicates": n_boot,
        "valid_bootstrap_replicates": len(draws),
        "beta": beta,
        "M": point["M"], "T": point["T"], "D": point["D"],
        "interval_semantics": "independent-cluster nonparametric bootstrap; training-seed variation excluded",
    }
    if draws:
        array = np.asarray(draws)
        for column, metric in enumerate(("M", "T", "D")):
            result[f"{metric}_ci_low"] = float(np.quantile(array[:, column], 0.025))
            result[f"{metric}_ci_high"] = float(np.quantile(array[:, column], 0.975))
    return result


def cluster_bootstrap_standardized_mtd(
    records: Sequence[ClusterSliceSufficient],
    *,
    beta: float = 0.10,
    n_boot: int = 2000,
    seed: int = 20260828,
    min_clusters_per_slice: int = 2,
) -> dict[str, Any]:
    """Cluster bootstrap with an equal-weight balance level within every slice.

    This implements the class-standardised country estimand used by fMoW. A
    replicate is retained only when every frozen slice-by-balance cell remains
    represented; it never silently changes the estimand through renormalisation.
    """
    preflight = cluster_preflight(records, min_clusters_per_slice=min_clusters_per_slice)
    clusters = sorted({record.cluster for record in records})
    slices = sorted({record.slice_value for record in records})
    levels = sorted({record.balance_value for record in records if record.balance_value})
    if preflight["status"] != "eligible" or not levels:
        return {
            **preflight,
            "status": "insufficient_independent_clusters" if preflight["status"] != "eligible" else "missing_balance_levels",
            "valid_bootstrap_replicates": 0,
            "requested_bootstrap_replicates": n_boot,
        }
    ci = {cluster: index for index, cluster in enumerate(clusters)}
    si = {value: index for index, value in enumerate(slices)}
    li = {value: index for index, value in enumerate(levels)}
    sums = np.zeros((len(clusters), len(slices), len(levels)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for record in records:
        if not record.balance_value:
            continue
        sums[ci[record.cluster], si[record.slice_value], li[record.balance_value]] += record.risk_sum
        counts[ci[record.cluster], si[record.slice_value], li[record.balance_value]] += record.count
    point_counts = counts.sum(axis=0)
    missing_cells = int(np.sum(point_counts <= 0))
    if missing_cells:
        return {
            **preflight,
            "status": "incomplete_slice_by_balance_support",
            "balance_level_count": len(levels),
            "missing_slice_by_balance_cells": missing_cells,
            "valid_bootstrap_replicates": 0,
            "requested_bootstrap_replicates": n_boot,
        }
    point_cell_risks = sums.sum(axis=0) / point_counts
    point_slice_risks = point_cell_risks.mean(axis=1)
    point = compute_fractional_mtd(dict(zip(slices, point_slice_risks)), beta=beta)
    rng = np.random.default_rng(seed)
    draws: list[tuple[float, float, float]] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        boot_counts = counts[chosen].sum(axis=0)
        if np.any(boot_counts <= 0):
            continue
        boot_cells = sums[chosen].sum(axis=0) / boot_counts
        boot_slices = boot_cells.mean(axis=1)
        card = compute_fractional_mtd(dict(zip(slices, boot_slices)), beta=beta)
        draws.append((float(card["M"]), float(card["T"]), float(card["D"])))
    result: dict[str, Any] = {
        **preflight,
        "status": "computed" if len(draws) >= max(100, int(0.8 * n_boot)) else "unstable_bootstrap_support",
        "requested_bootstrap_replicates": n_boot,
        "valid_bootstrap_replicates": len(draws),
        "balance_level_count": len(levels),
        "missing_slice_by_balance_cells": 0,
        "beta": beta,
        "M": point["M"], "T": point["T"], "D": point["D"],
        "interval_semantics": "independent-cluster bootstrap; equal balance-level weighting within slice; strict complete-cell replicates",
    }
    if draws:
        array = np.asarray(draws)
        for column, metric in enumerate(("M", "T", "D")):
            result[f"{metric}_ci_low"] = float(np.quantile(array[:, column], 0.025))
            result[f"{metric}_ci_high"] = float(np.quantile(array[:, column], 0.975))
    return result


def infer_verified_datetime(tags: Mapping[str, Any], filename: str = "") -> tuple[str, str]:
    """Return a verified ISO date when present in metadata; never infer event dates."""
    candidates = []
    for key, value in tags.items():
        if any(token in str(key).lower() for token in ("date", "time", "acquisition", "sensing")):
            candidates.append(str(value))
    candidates.append(filename)
    patterns = (r"(?<!\d)(20\d{2})[-_]?(\d{2})[-_]?(\d{2})(?!\d)",)
    for candidate in candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate)
            if not match:
                continue
            try:
                value = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                continue
            return value.date().isoformat(), "verified_from_file_metadata_or_name"
    return "", "unavailable_not_imputed"


def robust_water_land_contrast(image: np.ndarray, flood_mask: np.ndarray, valid_mask: np.ndarray) -> float:
    """Scale-resistant multiband median contrast between mapped water and land."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or flood_mask.shape != array.shape[1:] or valid_mask.shape != flood_mask.shape:
        raise ValueError("Expected image [bands,y,x] and matching two-dimensional masks.")
    if np.sum(flood_mask & valid_mask) == 0 or np.sum((~flood_mask) & valid_mask) == 0:
        return float("nan")
    contrasts = []
    for band in array:
        valid = valid_mask & np.isfinite(band)
        water = band[valid & flood_mask]
        land = band[valid & ~flood_mask]
        pooled = band[valid]
        if len(water) == 0 or len(land) == 0:
            continue
        scale = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
        if scale <= 0:
            continue
        contrasts.append(abs(float(np.median(water) - np.median(land))) / scale)
    return float(np.mean(contrasts)) if contrasts else float("nan")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SupplementaryAnalysisError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path
