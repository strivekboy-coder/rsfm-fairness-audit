from __future__ import annotations

import itertools

import numpy as np

from rsfm_fairness_audit.bwer_core import compute_geobwer, fractional_tail_allocation
from rsfm_fairness_audit.bwer_standardization import partial_bwer_bounds
from rsfm_fairness_audit.geobwer_certification import (
    NoHarmDecision,
    paired_risk_triple_from_boxes,
    sharp_geobwer_identification,
)


def _vertex_extrema(lower: dict[str, float], upper: dict[str, float], beta: float) -> tuple[float, float]:
    values = []
    groups = tuple(lower)
    for choices in itertools.product((0, 1), repeat=len(groups)):
        risks = {group: (upper[group] if choices[index] else lower[group]) for index, group in enumerate(groups)}
        values.append(compute_geobwer(risks, beta).bwer)
    return min(values), max(values)


def test_sharp_equal_measure_upper_matches_exhaustive_vertices() -> None:
    rng = np.random.default_rng(7)
    for group_count in range(2, 7):
        for beta in (0.1, 0.25, 0.5, 0.8, 1.0):
            low_values = rng.uniform(0.0, 0.6, size=group_count)
            high_values = low_values + rng.uniform(0.0, 1.0 - low_values)
            lower = {f"g{i}": float(low_values[i]) for i in range(group_count)}
            upper = {f"g{i}": float(high_values[i]) for i in range(group_count)}
            result = sharp_geobwer_identification(lower, upper, beta=beta)
            _, brute_upper = _vertex_extrema(lower, upper, beta)
            assert result.exact_upper
            assert np.isclose(result.bwer.upper, brute_upper, atol=1e-12)


def test_sharp_lower_matches_dense_grid_for_two_groups() -> None:
    lower = {"a": 0.1, "b": 0.35}
    upper = {"a": 0.8, "b": 0.9}
    for beta in (0.2, 0.5, 0.75):
        result = sharp_geobwer_identification(lower, upper, beta=beta)
        grid = np.linspace(0.0, 1.0, 501)
        values = [
            compute_geobwer({"a": a, "b": b}, beta).bwer
            for a in grid[(grid >= lower["a"]) & (grid <= upper["a"])]
            for b in grid[(grid >= lower["b"]) & (grid <= upper["b"])]
        ]
        assert result.bwer.lower <= min(values) + 2e-3
        assert result.bwer.lower >= min(values) - 2e-3


def test_partial_identification_uses_sharp_box_and_collapses_on_complete_support() -> None:
    exact = partial_bwer_bounds({"a": 0.2, "b": 0.7}, {"a": 0.2, "b": 0.7}, beta=0.5)
    assert exact.point_if_identified == exact.lower == exact.upper
    assert exact.exact_lower and exact.exact_upper

    partial = partial_bwer_bounds({"a": 0.0, "b": 0.4}, {"a": 0.6, "b": 1.0}, beta=0.5)
    legacy_upper = compute_geobwer({"a": 0.6, "b": 1.0}, 0.5).tail_risk
    assert 0.0 <= partial.lower <= partial.upper <= legacy_upper


def test_paired_triple_flags_no_harm_and_tradeoff() -> None:
    improved = paired_risk_triple_from_boxes(
        {"a": 0.10, "b": 0.20},
        {"a": 0.11, "b": 0.21},
        {"a": 0.10, "b": 0.60},
        {"a": 0.11, "b": 0.61},
        beta=0.5,
    )
    assert improved.delta_bwer.upper < 0.0
    assert improved.no_harm_decision == NoHarmDecision.CERTIFIED_NO_HARM_IMPROVEMENT

    tradeoff = paired_risk_triple_from_boxes(
        {"a": 0.60, "b": 0.65},
        {"a": 0.61, "b": 0.66},
        {"a": 0.10, "b": 0.60},
        {"a": 0.11, "b": 0.61},
        beta=0.5,
    )
    assert tradeoff.delta_bwer.upper < 0.0
    assert tradeoff.delta_mean.upper > 0.0
    assert tradeoff.no_harm_decision == NoHarmDecision.DISPARITY_REDUCTION_WITH_TRADEOFF


def test_boundary_tie_allocation_is_permutation_invariant() -> None:
    risks_a = {"x": 0.9, "y": 0.9, "z": 0.1}
    risks_b = {"renamed_y": 0.9, "renamed_x": 0.9, "z": 0.1}
    _, allocation_a = fractional_tail_allocation(risks_a, beta=0.5)
    _, allocation_b = fractional_tail_allocation(risks_b, beta=0.5)
    selected_a = sorted(dict(allocation_a.selected_mass).values())
    selected_b = sorted(dict(allocation_b.selected_mass).values())
    assert np.allclose(selected_a, selected_b)
    assert np.isclose(dict(allocation_a.selected_mass)["x"], dict(allocation_a.selected_mass)["y"])


def test_sharp_band_is_never_wider_than_legacy_lipschitz_fallbacks() -> None:
    risks = {"a": 0.15, "b": 0.35, "c": 0.80, "d": 0.90}
    lower = {group: max(0.0, value - 0.08) for group, value in risks.items()}
    upper = {group: min(1.0, value + 0.08) for group, value in risks.items()}
    result = sharp_geobwer_identification(lower, upper, beta=0.25, point_risks=risks)
    errors = {group: 0.08 for group in risks}
    tail_error, _ = fractional_tail_allocation(errors, beta=0.25)
    weighted_radius = tail_error + np.mean(tuple(errors.values()))
    tv_radius = 2.0 * (1.0 - 0.25) * max(errors.values())
    legacy_radius = min(weighted_radius, tv_radius)
    point = compute_geobwer(risks, beta=0.25).bwer
    legacy_low = max(0.0, point - legacy_radius)
    legacy_high = min(0.75, point + legacy_radius)
    assert result.bwer.lower >= legacy_low - 1e-12
    assert result.bwer.upper <= legacy_high + 1e-12


def test_tail_regime_discloses_single_slice_dominance() -> None:
    single = compute_geobwer({"a": 0.9, "b": 0.2, "c": 0.1}, beta=0.1)
    broad = compute_geobwer({"a": 0.9, "b": 0.8, "c": 0.1}, beta=0.8)
    assert single.allocation.tail_regime == "worst_slice"
    assert broad.allocation.tail_regime == "multi_slice_tail"
