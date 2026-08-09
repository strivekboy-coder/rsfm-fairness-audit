from __future__ import annotations

import numpy as np
import pytest

from rsfm_fairness_audit.bwer_inference import (
    _multiplier_draws,
    certified_geobwer,
    certify_geobwer_from_band,
    equal_area_block_ids,
    honest_confirmed_bwer,
    one_sided_calibration_gate,
    paired_bwer_comparison,
    simultaneous_group_risk_band,
    simultaneous_standardized_risk_band,
)
from rsfm_fairness_audit.bwer_protocol import Validity


def _clustered_rows(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(["a", "b", "c"], 40)
    clusters = np.asarray([f"{group}_{index // 2}" for group in ["a", "b", "c"] for index in range(40)])
    losses = np.clip(0.2 + 0.1 * (groups == "c") + rng.normal(0.0, 0.05, len(groups)), 0.0, 1.0)
    return losses, groups, clusters


def test_sparse_multiplier_path_matches_exact_dense_product() -> None:
    influence = np.asarray(
        [
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, -0.3],
        ]
    )
    observed, standard_errors = _multiplier_draws(
        influence,
        n_bootstrap=100,
        seed=12,
        multiplier="rademacher",
    )
    expected_weights = np.random.default_rng(12).choice(
        np.asarray([-1.0, 1.0]), size=(100, influence.shape[0])
    )
    assert np.allclose(observed, expected_weights @ influence)
    assert np.allclose(standard_errors, np.sqrt(np.sum(influence * influence, axis=0)))


def test_simultaneous_band_and_certified_bwer_are_well_formed() -> None:
    losses, groups, clusters = _clustered_rows()
    band = simultaneous_group_risk_band(losses, groups, clusters, n_bootstrap=300, seed=3)
    assert band.validity == Validity.VALID
    assert band.cluster_count == 60
    assert set(dict(band.lower)) == {"a", "b", "c"}
    certified = certified_geobwer(losses, groups, clusters, beta=0.2, n_bootstrap=300, seed=3)
    assert 0.0 <= certified.ci_low <= certified.point.bwer <= certified.ci_high <= 1.0
    assert certified.radius <= certified.weighted_sum_radius
    assert certified.radius <= certified.total_variation_radius
    assert certified.parameter_upper_bound == pytest.approx(0.8)


def test_sharpened_radius_is_exactly_zero_at_beta_one() -> None:
    losses, groups, clusters = _clustered_rows()
    band = simultaneous_group_risk_band(losses, groups, clusters, n_bootstrap=300, seed=3)
    certified = certify_geobwer_from_band(band, beta=1.0)
    assert certified.point.bwer == pytest.approx(0.0)
    assert certified.radius == pytest.approx(0.0)
    assert certified.ci_low == pytest.approx(0.0)
    assert certified.ci_high == pytest.approx(0.0)


def test_standardized_band_uses_common_target_composition() -> None:
    losses, groups, clusters = _clustered_rows()
    balance = np.tile(np.repeat(["x", "y"], 20), 3)
    band = simultaneous_standardized_risk_band(
        losses,
        groups,
        balance,
        clusters,
        target_weights={"x": 0.5, "y": 0.5},
        n_bootstrap=300,
    )
    assert band.validity == Validity.VALID
    assert len(band.estimates) == 3


def test_standardized_band_fails_when_a_target_cell_is_missing() -> None:
    losses, groups, clusters = _clustered_rows()
    balance = np.tile(np.repeat(["x", "y"], 20), 3)
    keep = ~((groups == "c") & (balance == "y"))
    band = simultaneous_standardized_risk_band(
        losses[keep],
        groups[keep],
        balance[keep],
        clusters[keep],
        target_weights={"x": 0.5, "y": 0.5},
        n_bootstrap=300,
    )
    assert band.validity == Validity.NOT_IDENTIFIED


def test_paired_comparison_preserves_pairing() -> None:
    losses, groups, clusters = _clustered_rows()
    result = paired_bwer_comparison(losses, losses.copy(), groups, clusters, beta=0.2, n_bootstrap=300)
    assert result.validity == Validity.VALID
    assert result.delta_bwer == pytest.approx(0.0)
    assert result.ci_low <= 0.0 <= result.ci_high
    assert result.direct_multiplier_ci_low <= 0.0 <= result.direct_multiplier_ci_high


def test_equal_area_blocks_wrap_antimeridian_and_change_with_latitude() -> None:
    ids = equal_area_block_ids([0.0, 0.0, 80.0], [179.9, -179.9, 0.0], cell_km=100.0)
    assert len(ids) == 3
    assert ids[0] != ids[1]
    assert ids[2] != ids[0]


def test_too_few_clusters_returns_invalid_state() -> None:
    losses = [0.1, 0.2, 0.3, 0.4]
    groups = ["a", "a", "b", "b"]
    clusters = ["one", "one", "two", "two"]
    band = simultaneous_group_risk_band(losses, groups, clusters, n_bootstrap=100, min_clusters_per_group=2)
    assert band.validity == Validity.INSUFFICIENT_INDEPENDENT_UNITS


def test_honest_confirmation_recovers_persistent_bad_group() -> None:
    losses: list[float] = []
    groups: list[str] = []
    clusters: list[str] = []
    for group, risk in (("a", 0.10), ("b", 0.15), ("c", 0.65), ("d", 0.20)):
        for index in range(20):
            losses.append(risk)
            groups.append(group)
            clusters.append(f"{group}_{index}")
    result = honest_confirmed_bwer(losses, groups, clusters, beta=0.25, seed=7)
    assert result.validity == Validity.VALID
    assert result.both_directions_positive
    assert result.cross_fitted_contrast > 0.30
    assert len(result.folds) == 2


def test_honest_confirmation_rejects_group_equal_to_cluster_design() -> None:
    result = honest_confirmed_bwer(
        [0.1, 0.2, 0.3, 0.4],
        ["event1", "event2", "event3", "event4"],
        ["event1", "event2", "event3", "event4"],
        beta=0.25,
        seed=1,
    )
    assert result.validity == Validity.NOT_IDENTIFIED
    assert not result.folds


def test_one_sided_gate_rejects_cases_that_old_endpoint_logic_would_pass() -> None:
    # The old gate accepted this because coverage upper >= .95 and FPR lower
    # <= .05.  Neither statement certifies adequate coverage/FPR control.
    assert not one_sided_calibration_gate(
        coverage_ci=(0.82, 0.97),
        false_positive_ci=(0.01, 0.12),
        confidence_level=0.95,
        alpha=0.05,
        coverage_tolerance=0.02,
        false_positive_tolerance=0.01,
    )
    assert one_sided_calibration_gate(
        coverage_ci=(0.94, 0.98),
        false_positive_ci=(0.00, 0.04),
        confidence_level=0.95,
        alpha=0.05,
        coverage_tolerance=0.02,
        false_positive_tolerance=0.01,
    )
