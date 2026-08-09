from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_protocol import METRIC_VERSION


@dataclass(frozen=True)
class TailAllocation:
    beta: float
    deployment_weights: tuple[tuple[str, float], ...]
    selected_mass: tuple[tuple[str, float], ...]
    conditional_tail_weights: tuple[tuple[str, float], ...]
    boundary_risk: float
    boundary_tie_groups: tuple[str, ...]
    tail_effective_groups: float
    max_tail_atom_share: float
    tail_regime: str
    tail_capacity_ratio: float

    def selected_mass_dict(self) -> dict[str, float]:
        return dict(self.selected_mass)


@dataclass(frozen=True)
class BWERPointEstimate:
    beta: float
    mean_risk: float
    tail_risk: float
    bwer: float
    worst_group_risk: float
    worst_group_gap: float
    deployment_effective_groups: float
    allocation: TailAllocation
    metric_version: str = METRIC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_version": self.metric_version,
            "beta": self.beta,
            "mean_risk": self.mean_risk,
            "tail_risk": self.tail_risk,
            "bwer": self.bwer,
            "worst_group_risk": self.worst_group_risk,
            "worst_group_gap": self.worst_group_gap,
            "deployment_effective_groups": self.deployment_effective_groups,
            "tail_effective_groups": self.allocation.tail_effective_groups,
            "max_tail_atom_share": self.allocation.max_tail_atom_share,
            "tail_regime": self.allocation.tail_regime,
            "tail_capacity_ratio": self.allocation.tail_capacity_ratio,
            "boundary_risk": self.allocation.boundary_risk,
            "boundary_tie_groups": list(self.allocation.boundary_tie_groups),
            "deployment_weights": dict(self.allocation.deployment_weights),
            "selected_mass": dict(self.allocation.selected_mass),
            "conditional_tail_weights": dict(self.allocation.conditional_tail_weights),
        }


def _finite_risks(group_risks: Mapping[Any, float]) -> dict[str, float]:
    output: dict[str, float] = {}
    for group, raw in group_risks.items():
        risk = float(raw)
        if not math.isfinite(risk):
            continue
        output[str(group)] = risk
    if len(output) < 1:
        raise ValueError("At least one finite group risk is required.")
    return output


def normalize_deployment_weights(
    groups: Sequence[str],
    deployment_weights: Mapping[Any, float] | None = None,
) -> dict[str, float]:
    unique = tuple(dict.fromkeys(str(group) for group in groups))
    if not unique:
        raise ValueError("At least one group is required.")
    if deployment_weights is None:
        return {group: 1.0 / len(unique) for group in unique}
    raw: dict[str, float] = {}
    for group in unique:
        if group not in {str(key) for key in deployment_weights}:
            raise ValueError(f"Missing deployment weight for group={group!r}.")
    by_string = {str(key): float(value) for key, value in deployment_weights.items()}
    for group in unique:
        value = by_string[group]
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Deployment weights must be finite and non-negative.")
        raw[group] = value
    total = float(sum(raw.values()))
    if total <= 0.0:
        raise ValueError("Deployment weights must have positive total mass.")
    return {group: value / total for group, value in raw.items()}


def fractional_tail_allocation(
    group_risks: Mapping[Any, float],
    beta: float,
    deployment_weights: Mapping[Any, float] | None = None,
    *,
    tie_tolerance: float = 1e-12,
) -> tuple[float, TailAllocation]:
    """Return exact weighted upper-tail risk and its fractional mass allocation."""

    if not 0.0 < float(beta) <= 1.0:
        raise ValueError("beta must be in (0, 1].")
    risks = _finite_risks(group_risks)
    weights = normalize_deployment_weights(tuple(risks), deployment_weights)
    ranked = sorted(risks, key=lambda group: (-risks[group], group))
    remaining = float(beta)
    selected = {group: 0.0 for group in ranked}
    cursor = 0
    while cursor < len(ranked):
        if remaining <= 1e-15:
            break
        boundary = risks[ranked[cursor]]
        tied: list[str] = []
        while cursor < len(ranked) and abs(risks[ranked[cursor]] - boundary) <= tie_tolerance:
            tied.append(ranked[cursor])
            cursor += 1
        tied_mass = float(sum(weights[group] for group in tied))
        if remaining >= tied_mass - 1e-15:
            for group in tied:
                selected[group] = weights[group]
            remaining -= tied_mass
        else:
            # The tail value is invariant to how a boundary tie is split.  A
            # proportional allocation makes attribution permutation-invariant.
            fraction = remaining / tied_mass
            for group in tied:
                selected[group] = weights[group] * fraction
            remaining = 0.0
    if remaining > 1e-10:
        raise RuntimeError("Fractional tail allocation did not reach beta mass.")
    selected_total = float(sum(selected.values()))
    if not math.isclose(selected_total, float(beta), rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError(f"Selected mass {selected_total} does not equal beta={beta}.")
    conditional = {group: selected[group] / float(beta) for group in ranked}
    tail_risk = float(sum(conditional[group] * risks[group] for group in ranked))
    selected_groups = [group for group in ranked if selected[group] > 0.0]
    boundary_risk = risks[selected_groups[-1]]
    ties = tuple(sorted(group for group, risk in risks.items() if abs(risk - boundary_risk) <= tie_tolerance))
    q = np.asarray([conditional[group] for group in ranked], dtype=float)
    tail_effective = float(1.0 / np.sum(q * q)) if np.any(q > 0.0) else 0.0
    max_tail_atom_share = float(np.max(q)) if len(q) else 0.0
    if tail_effective <= 1.0 + 1e-12 or max_tail_atom_share >= 1.0 - 1e-12:
        tail_regime = "worst_slice"
    elif tail_effective < 2.0 - 1e-12:
        tail_regime = "near_worst_slice"
    else:
        tail_regime = "multi_slice_tail"
    allocation = TailAllocation(
        beta=float(beta),
        deployment_weights=tuple((group, weights[group]) for group in ranked),
        selected_mass=tuple((group, selected[group]) for group in ranked),
        conditional_tail_weights=tuple((group, conditional[group]) for group in ranked),
        boundary_risk=float(boundary_risk),
        boundary_tie_groups=ties,
        tail_effective_groups=tail_effective,
        max_tail_atom_share=max_tail_atom_share,
        tail_regime=tail_regime,
        tail_capacity_ratio=float(beta) / max(weights.values()),
    )
    return tail_risk, allocation


def compute_geobwer(
    group_risks: Mapping[Any, float],
    beta: float = 0.10,
    deployment_weights: Mapping[Any, float] | None = None,
) -> BWERPointEstimate:
    risks = _finite_risks(group_risks)
    weights = normalize_deployment_weights(tuple(risks), deployment_weights)
    tail_risk, allocation = fractional_tail_allocation(risks, beta, weights)
    mean_risk = float(sum(weights[group] * risks[group] for group in risks))
    raw_bwer = tail_risk - mean_risk
    bwer = 0.0 if abs(raw_bwer) <= 1e-15 else float(raw_bwer)
    worst = float(max(risks.values()))
    mu = np.asarray(list(weights.values()), dtype=float)
    return BWERPointEstimate(
        beta=float(beta),
        mean_risk=mean_risk,
        tail_risk=tail_risk,
        bwer=bwer,
        worst_group_risk=worst,
        worst_group_gap=float(worst - mean_risk),
        deployment_effective_groups=float(1.0 / np.sum(mu * mu)),
        allocation=allocation,
    )


def compute_geobwer_profile(
    group_risks: Mapping[Any, float],
    betas: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    deployment_weights: Mapping[Any, float] | None = None,
) -> list[BWERPointEstimate]:
    if len(set(float(beta) for beta in betas)) != len(betas):
        raise ValueError("betas must not contain duplicates.")
    return [compute_geobwer(group_risks, float(beta), deployment_weights) for beta in betas]


def legacy_whole_slice_bwer(group_risks: Mapping[Any, float], beta: float = 0.10) -> float:
    """Historical BWER1 point definition retained for exact reproducibility."""

    risks = sorted(_finite_risks(group_risks).values(), reverse=True)
    tail_n = max(1, int(math.ceil(len(risks) * float(beta))))
    return float(np.mean(risks[:tail_n]) - np.mean(risks))


def bwer_from_arrays(
    risks: Sequence[float],
    groups: Sequence[Any],
    *,
    beta: float = 0.10,
    deployment_weighting: str = "equal",
    custom_weights: Mapping[Any, float] | None = None,
) -> tuple[BWERPointEstimate, dict[str, float], dict[str, int]]:
    if len(risks) != len(groups):
        raise ValueError("risks and groups must have the same length.")
    grouped: dict[str, list[float]] = {}
    for raw_risk, raw_group in zip(risks, groups):
        risk = float(raw_risk)
        if math.isfinite(risk):
            grouped.setdefault(str(raw_group), []).append(risk)
    group_risks = {group: float(np.mean(values)) for group, values in grouped.items() if values}
    supports = {group: len(values) for group, values in grouped.items() if values}
    if deployment_weighting == "equal":
        weights = None
    elif deployment_weighting == "empirical":
        weights = {group: float(count) for group, count in supports.items()}
    elif deployment_weighting == "custom":
        if custom_weights is None:
            raise ValueError("custom_weights are required for custom deployment weighting.")
        weights = custom_weights
    else:
        raise ValueError("deployment_weighting must be equal, empirical, or custom.")
    return compute_geobwer(group_risks, beta, weights), group_risks, supports
