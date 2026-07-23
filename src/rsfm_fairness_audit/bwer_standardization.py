from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_core import BWERPointEstimate, compute_geobwer, fractional_tail_allocation, normalize_deployment_weights
from rsfm_fairness_audit.bwer_protocol import Validity


def _missing(value: Any) -> bool:
    return value is None or value == "" or str(value).lower() in {"nan", "none", "null"}


@dataclass(frozen=True)
class StandardizationResult:
    validity: Validity
    group_risks: tuple[tuple[str, float], ...]
    risk_lower: tuple[tuple[str, float], ...]
    risk_upper: tuple[tuple[str, float], ...]
    target_weights: tuple[tuple[str, float], ...]
    support: tuple[tuple[str, str, int], ...]
    missing_cells: tuple[tuple[str, str], ...]
    used_balance_levels: tuple[str, ...]
    message: str = ""

    def group_risk_dict(self) -> dict[str, float]:
        return dict(self.group_risks)

    def lower_dict(self) -> dict[str, float]:
        return dict(self.risk_lower)

    def upper_dict(self) -> dict[str, float]:
        return dict(self.risk_upper)


@dataclass(frozen=True)
class CommonSupportResult:
    validity: Validity
    groups: tuple[str, ...]
    aligned_risks: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    deployment_weights: tuple[tuple[str, float], ...]
    support_hash: str
    dropped_by_model: tuple[tuple[str, tuple[str, ...]], ...]
    message: str = ""

    def risks_for(self, model: str) -> dict[str, float]:
        for name, values in self.aligned_risks:
            if name == model:
                return dict(values)
        raise KeyError(model)


@dataclass(frozen=True)
class PartialBWERBounds:
    lower: float
    upper: float
    point_if_identified: float | None
    validity: Validity


def _normalize_target_weights(levels: Sequence[str], target_weights: Mapping[Any, float] | None) -> dict[str, float]:
    unique = tuple(sorted(set(str(level) for level in levels)))
    if not unique:
        raise ValueError("No balance levels are available.")
    if target_weights is None:
        return {level: 1.0 / len(unique) for level in unique}
    raw_all = {str(key): float(value) for key, value in target_weights.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in raw_all.values()):
        raise ValueError("target_weights must be finite and non-negative.")
    raw_by_string = {key: value for key, value in raw_all.items() if value > 0.0}
    # Observed levels omitted from a pre-registered target have q_z=0 and do
    # not belong to the standardised estimand.  Conversely, a positive target
    # level absent from the audit table cannot be identified at all.
    absent = sorted(set(raw_by_string) - set(unique))
    if absent:
        raise ValueError(f"Target balance levels are absent from the audit table: {absent}.")
    total = float(sum(raw_by_string.values()))
    if total <= 0.0:
        raise ValueError("target_weights must have positive total mass.")
    return {level: raw_by_string[level] / total for level in sorted(raw_by_string)}


def standardize_group_risks(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_column: str,
    balance_column: str,
    loss_column: str = "risk",
    target_weights: Mapping[Any, float] | None = None,
    missingness_rule: str = "strict",
    min_cell_support: int = 1,
    risk_bounds: tuple[float, float] = (0.0, 1.0),
) -> StandardizationResult:
    """Standardize every group to one common balance distribution.

    The formal ``strict`` rule never renormalizes a different target population
    inside each group. ``overlap`` is an explicit sensitivity estimand and
    ``partial_bounds`` keeps the target distribution while bounding missing cells.
    """

    if missingness_rule not in {"strict", "overlap", "partial_bounds"}:
        raise ValueError("missingness_rule must be strict, overlap, or partial_bounds.")
    if min_cell_support < 1:
        raise ValueError("min_cell_support must be positive.")
    lower_bound, upper_bound = map(float, risk_bounds)
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound) or lower_bound > upper_bound:
        raise ValueError("risk_bounds must be finite and ordered.")
    cells: dict[tuple[str, str], list[float]] = {}
    groups: set[str] = set()
    levels: set[str] = set()
    for row in rows:
        if _missing(row.get(group_column)) or _missing(row.get(balance_column)) or _missing(row.get(loss_column)):
            continue
        risk = float(row[loss_column])
        if not math.isfinite(risk):
            continue
        group = str(row[group_column])
        level = str(row[balance_column])
        groups.add(group)
        levels.add(level)
        cells.setdefault((group, level), []).append(risk)
    if len(groups) < 2 or not levels:
        return StandardizationResult(
            validity=Validity.INSUFFICIENT_SLICES,
            group_risks=(),
            risk_lower=(),
            risk_upper=(),
            target_weights=(),
            support=(),
            missing_cells=(),
            used_balance_levels=(),
            message="At least two groups and one balance level are required.",
        )
    weights = _normalize_target_weights(tuple(levels), target_weights)
    cell_means = {key: float(np.mean(values)) for key, values in cells.items() if len(values) >= min_cell_support}
    support = tuple(sorted((group, level, len(cells.get((group, level), ()))) for group in groups for level in weights))
    missing = tuple(sorted((group, level) for group in groups for level in weights if (group, level) not in cell_means))
    used_levels = tuple(sorted(weights))
    if missingness_rule == "overlap":
        shared = tuple(sorted(level for level in weights if all((group, level) in cell_means for group in groups)))
        if not shared:
            return StandardizationResult(
                validity=Validity.NOT_IDENTIFIED,
                group_risks=(),
                risk_lower=(),
                risk_upper=(),
                target_weights=tuple(sorted(weights.items())),
                support=support,
                missing_cells=missing,
                used_balance_levels=(),
                message="No balance level has common support across all groups.",
            )
        overlap_total = sum(weights[level] for level in shared)
        active_weights = {level: weights[level] / overlap_total for level in shared}
        risks = {
            group: float(sum(active_weights[level] * cell_means[(group, level)] for level in shared))
            for group in sorted(groups)
        }
        return StandardizationResult(
            validity=Validity.VALID,
            group_risks=tuple(risks.items()),
            risk_lower=tuple(risks.items()),
            risk_upper=tuple(risks.items()),
            target_weights=tuple(sorted(active_weights.items())),
            support=support,
            missing_cells=missing,
            used_balance_levels=shared,
            message="Overlap sensitivity estimand; target weights were restricted once globally, not per group.",
        )
    point: dict[str, float] = {}
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for group in sorted(groups):
        observed = sum(weights[level] * cell_means[(group, level)] for level in weights if (group, level) in cell_means)
        missing_mass = sum(weights[level] for level in weights if (group, level) not in cell_means)
        lower[group] = float(observed + missing_mass * lower_bound)
        upper[group] = float(observed + missing_mass * upper_bound)
        if missing_mass == 0.0:
            point[group] = float(observed)
    if missingness_rule == "strict" and missing:
        return StandardizationResult(
            validity=Validity.NOT_IDENTIFIED,
            group_risks=tuple(point.items()),
            risk_lower=tuple(lower.items()),
            risk_upper=tuple(upper.items()),
            target_weights=tuple(sorted(weights.items())),
            support=support,
            missing_cells=missing,
            used_balance_levels=used_levels,
            message="Strict standardization failed because at least one target group×balance cell lacks support.",
        )
    validity = Validity.VALID if not missing else Validity.NOT_IDENTIFIED
    return StandardizationResult(
        validity=validity,
        group_risks=tuple(point.items()),
        risk_lower=tuple(lower.items()),
        risk_upper=tuple(upper.items()),
        target_weights=tuple(sorted(weights.items())),
        support=support,
        missing_cells=missing,
        used_balance_levels=used_levels,
        message="" if not missing else "Point risks are partially identified; use the reported bounds.",
    )


def partial_bwer_bounds(
    risk_lower: Mapping[Any, float],
    risk_upper: Mapping[Any, float],
    *,
    beta: float,
    deployment_weights: Mapping[Any, float] | None = None,
) -> PartialBWERBounds:
    groups = sorted(set(str(key) for key in risk_lower) & set(str(key) for key in risk_upper))
    if len(groups) < 2:
        return PartialBWERBounds(float("nan"), float("nan"), None, Validity.INSUFFICIENT_SLICES)
    lower_map = {str(key): float(value) for key, value in risk_lower.items() if str(key) in groups}
    upper_map = {str(key): float(value) for key, value in risk_upper.items() if str(key) in groups}
    if any(lower_map[group] > upper_map[group] for group in groups):
        raise ValueError("Every group risk lower bound must not exceed its upper bound.")
    weights = normalize_deployment_weights(groups, deployment_weights)
    tail_lower, _ = fractional_tail_allocation(lower_map, beta, weights)
    tail_upper, _ = fractional_tail_allocation(upper_map, beta, weights)
    mean_lower = sum(weights[group] * lower_map[group] for group in groups)
    mean_upper = sum(weights[group] * upper_map[group] for group in groups)
    lower = max(0.0, float(tail_lower - mean_upper))
    upper = max(lower, float(tail_upper - mean_lower))
    point: float | None = None
    if all(math.isclose(lower_map[group], upper_map[group], abs_tol=1e-15) for group in groups):
        point = compute_geobwer(lower_map, beta, weights).bwer
    return PartialBWERBounds(lower, upper, point, Validity.VALID if point is not None else Validity.NOT_IDENTIFIED)


def common_group_support(
    model_group_risks: Mapping[str, Mapping[Any, float]],
    *,
    deployment_weights: Mapping[Any, float] | None = None,
    min_groups: int = 2,
) -> CommonSupportResult:
    if len(model_group_risks) < 2:
        raise ValueError("At least two models are required for common-support comparison.")
    finite: dict[str, dict[str, float]] = {}
    for model, risks in model_group_risks.items():
        finite[str(model)] = {str(group): float(value) for group, value in risks.items() if math.isfinite(float(value))}
    common = set.intersection(*(set(risks) for risks in finite.values())) if finite else set()
    groups = tuple(sorted(common))
    dropped = tuple(
        (model, tuple(sorted(set(risks) - common)))
        for model, risks in sorted(finite.items())
    )
    if len(groups) < min_groups:
        return CommonSupportResult(
            validity=Validity.NO_COMMON_SUPPORT,
            groups=groups,
            aligned_risks=(),
            deployment_weights=(),
            support_hash="",
            dropped_by_model=dropped,
            message=f"Only {len(groups)} common groups; min_groups={min_groups}.",
        )
    weights = normalize_deployment_weights(groups, deployment_weights)
    aligned = tuple(
        (model, tuple((group, finite[model][group]) for group in groups))
        for model in sorted(finite)
    )
    signature_payload = json.dumps({"groups": groups, "weights": weights}, sort_keys=True, separators=(",", ":"))
    support_hash = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    return CommonSupportResult(
        validity=Validity.VALID,
        groups=groups,
        aligned_risks=aligned,
        deployment_weights=tuple((group, weights[group]) for group in groups),
        support_hash=support_hash,
        dropped_by_model=dropped,
    )
