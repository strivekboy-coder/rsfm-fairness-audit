from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_core import compute_geobwer, fractional_tail_allocation, normalize_deployment_weights


CERTIFICATION_VERSION = "geobwer_certification_1.2"


class NoHarmDecision(str, Enum):
    CERTIFIED_NO_HARM_IMPROVEMENT = "certified_no_harm_improvement"
    DISPARITY_REDUCTION_WITH_TRADEOFF = "disparity_reduction_with_tradeoff"
    NO_CERTIFIED_DISPARITY_REDUCTION = "no_certified_disparity_reduction"


@dataclass(frozen=True)
class FunctionalInterval:
    point: float
    lower: float
    upper: float


@dataclass(frozen=True)
class SharpBWERIdentification:
    mean: FunctionalInterval
    tail: FunctionalInterval
    bwer: FunctionalInterval
    exact_lower: bool
    exact_upper: bool
    lower_method: str
    upper_method: str
    certification_version: str = CERTIFICATION_VERSION


@dataclass(frozen=True)
class PairedRiskTriple:
    delta_mean: FunctionalInterval
    delta_tail: FunctionalInterval
    delta_bwer: FunctionalInterval
    no_harm_decision: NoHarmDecision
    mean_harm_tolerance: float
    tail_harm_tolerance: float
    certification_version: str = CERTIFICATION_VERSION


def _aligned_box(
    risk_lower: Mapping[Any, float],
    risk_upper: Mapping[Any, float],
) -> tuple[tuple[str, ...], dict[str, float], dict[str, float]]:
    lower = {str(key): float(value) for key, value in risk_lower.items()}
    upper = {str(key): float(value) for key, value in risk_upper.items()}
    if set(lower) != set(upper) or not lower:
        raise ValueError("Risk lower/upper boxes must contain the same non-empty group universe.")
    groups = tuple(sorted(lower))
    for group in groups:
        if not math.isfinite(lower[group]) or not math.isfinite(upper[group]) or lower[group] > upper[group]:
            raise ValueError(f"Invalid risk interval for group={group!r}.")
    return groups, lower, upper


def _sharp_lower(
    lower: Mapping[str, float],
    upper: Mapping[str, float],
    beta: float,
    weights: Mapping[str, float],
) -> float:
    """Exact minimum of CVaR_beta(r)-E_mu[r] over an axis-aligned box.

    For fixed CVaR threshold t, every coordinate is minimized at
    r_g=clip(t,[l_g,u_g]).  The remaining objective is piecewise linear, so a
    minimizer occurs at a box breakpoint.
    """

    if math.isclose(beta, 1.0, abs_tol=1e-15):
        return 0.0
    breakpoints = sorted(set(lower.values()) | set(upper.values()))
    candidates: list[float] = []
    for threshold in breakpoints:
        value = float(threshold)
        for group, weight in weights.items():
            risk = min(max(threshold, lower[group]), upper[group])
            value += weight * (max(risk - threshold, 0.0) / beta - risk)
        candidates.append(float(value))
    return max(0.0, min(candidates))


def _sharp_upper_equal_weights(
    lower: Mapping[str, float],
    upper: Mapping[str, float],
    beta: float,
) -> float:
    """Exact box maximum for the equal-slice deployment measure.

    Extreme points of the capped simplex have floor(beta*G) full tail atoms
    and at most one fractional boundary atom.  Enumerating only the boundary
    atom and choosing the best full atoms is O(G^2 log G), not O(2^G).
    """

    groups = tuple(lower)
    count = len(groups)
    if math.isclose(beta, 1.0, abs_tol=1e-15):
        return 0.0
    h = float(beta) * count
    full_count = int(math.floor(h + 1e-12))
    fraction = h - full_count
    if fraction < 1e-12:
        fraction = 0.0
    mu = 1.0 / count
    cap = 1.0 / h
    base = -mu * sum(lower.values())
    full_gain = {
        group: (cap - mu) * upper[group] + mu * lower[group]
        for group in groups
    }
    if fraction == 0.0:
        selected = sorted(full_gain.values(), reverse=True)[:full_count]
        return max(0.0, float(base + sum(selected)))
    boundary_q = fraction / h
    boundary_coefficient = boundary_q - mu
    best = -float("inf")
    for boundary in groups:
        boundary_endpoint = upper[boundary] if boundary_coefficient >= 0.0 else lower[boundary]
        boundary_gain = boundary_coefficient * boundary_endpoint + mu * lower[boundary]
        available = sorted(
            (gain for group, gain in full_gain.items() if group != boundary),
            reverse=True,
        )
        candidate = base + boundary_gain + sum(available[:full_count])
        best = max(best, float(candidate))
    return max(0.0, best)


def _sharp_upper_vertex_enumeration(
    lower: Mapping[str, float],
    upper: Mapping[str, float],
    beta: float,
    weights: Mapping[str, float],
) -> float:
    groups = tuple(lower)
    best = 0.0
    for choices in itertools.product((0, 1), repeat=len(groups)):
        risks = {
            group: upper[group] if choices[index] else lower[group]
            for index, group in enumerate(groups)
        }
        best = max(best, compute_geobwer(risks, beta, weights).bwer)
    return float(best)


def sharp_geobwer_identification(
    risk_lower: Mapping[Any, float],
    risk_upper: Mapping[Any, float],
    *,
    beta: float,
    deployment_weights: Mapping[Any, float] | None = None,
    point_risks: Mapping[Any, float] | None = None,
    max_general_weight_groups: int = 16,
) -> SharpBWERIdentification:
    """Map one simultaneous group-risk box to M/T/GeoBWER intervals.

    The lower GeoBWER endpoint is exact for arbitrary deployment weights.  The
    upper endpoint is exact for the primary equal-slice measure and, for small
    custom-weight problems, by exhaustive box-vertex enumeration.  Large
    custom-weight problems retain a transparent monotone endpoint fallback.
    """

    if not 0.0 < float(beta) <= 1.0:
        raise ValueError("beta must be in (0,1].")
    groups, lower, upper = _aligned_box(risk_lower, risk_upper)
    weights = normalize_deployment_weights(groups, deployment_weights)
    if point_risks is None:
        point_map = {group: 0.5 * (lower[group] + upper[group]) for group in groups}
    else:
        point_map = {str(key): float(value) for key, value in point_risks.items()}
        if set(point_map) != set(groups):
            raise ValueError("point_risks must match the risk-box group universe.")
        if any(point_map[group] < lower[group] - 1e-12 or point_map[group] > upper[group] + 1e-12 for group in groups):
            raise ValueError("point_risks must lie inside the simultaneous risk box.")
    point = compute_geobwer(point_map, beta, weights)
    mean_low = float(sum(weights[group] * lower[group] for group in groups))
    mean_high = float(sum(weights[group] * upper[group] for group in groups))
    tail_low = fractional_tail_allocation(lower, beta, weights)[0]
    tail_high = fractional_tail_allocation(upper, beta, weights)[0]
    bwer_low = _sharp_lower(lower, upper, beta, weights)
    if abs(bwer_low) <= 1e-12:
        bwer_low = 0.0
    equal_mass = max(weights.values()) - min(weights.values()) <= 1e-12
    if equal_mass:
        bwer_high = _sharp_upper_equal_weights(lower, upper, beta)
        exact_upper = True
        upper_method = "equal_measure_capped_simplex_extreme_points"
    elif len(groups) <= int(max_general_weight_groups):
        bwer_high = _sharp_upper_vertex_enumeration(lower, upper, beta, weights)
        exact_upper = True
        upper_method = "general_measure_box_vertex_enumeration"
    else:
        mean_lower = mean_low
        bwer_high = max(0.0, float(tail_high - mean_lower))
        exact_upper = False
        upper_method = "monotone_endpoint_fallback"
    bwer_high = max(bwer_low, bwer_high)
    if abs(bwer_high) <= 1e-12:
        bwer_high = 0.0
    return SharpBWERIdentification(
        mean=FunctionalInterval(point.mean_risk, mean_low, mean_high),
        tail=FunctionalInterval(point.tail_risk, tail_low, tail_high),
        bwer=FunctionalInterval(point.bwer, bwer_low, bwer_high),
        exact_lower=True,
        exact_upper=exact_upper,
        lower_method="cvar_epigraph_breakpoint_minimization",
        upper_method=upper_method,
    )


def paired_risk_triple_from_boxes(
    lower_a: Mapping[Any, float],
    upper_a: Mapping[Any, float],
    lower_b: Mapping[Any, float],
    upper_b: Mapping[Any, float],
    *,
    beta: float,
    deployment_weights: Mapping[Any, float] | None = None,
    point_a: Mapping[Any, float] | None = None,
    point_b: Mapping[Any, float] | None = None,
    mean_harm_tolerance: float = 0.0,
    tail_harm_tolerance: float = 0.0,
) -> PairedRiskTriple:
    identified_a = sharp_geobwer_identification(
        lower_a,
        upper_a,
        beta=beta,
        deployment_weights=deployment_weights,
        point_risks=point_a,
    )
    identified_b = sharp_geobwer_identification(
        lower_b,
        upper_b,
        beta=beta,
        deployment_weights=deployment_weights,
        point_risks=point_b,
    )

    def difference(left: FunctionalInterval, right: FunctionalInterval) -> FunctionalInterval:
        return FunctionalInterval(
            point=float(left.point - right.point),
            lower=float(left.lower - right.upper),
            upper=float(left.upper - right.lower),
        )

    delta_mean = difference(identified_a.mean, identified_b.mean)
    delta_tail = difference(identified_a.tail, identified_b.tail)
    delta_bwer = difference(identified_a.bwer, identified_b.bwer)
    if delta_bwer.upper < 0.0:
        if delta_mean.upper <= mean_harm_tolerance and delta_tail.upper <= tail_harm_tolerance:
            decision = NoHarmDecision.CERTIFIED_NO_HARM_IMPROVEMENT
        else:
            decision = NoHarmDecision.DISPARITY_REDUCTION_WITH_TRADEOFF
    else:
        decision = NoHarmDecision.NO_CERTIFIED_DISPARITY_REDUCTION
    return PairedRiskTriple(
        delta_mean=delta_mean,
        delta_tail=delta_tail,
        delta_bwer=delta_bwer,
        no_harm_decision=decision,
        mean_harm_tolerance=float(mean_harm_tolerance),
        tail_harm_tolerance=float(tail_harm_tolerance),
    )
