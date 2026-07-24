from __future__ import annotations

import numpy as np

from rsfm_fairness_audit.spatial_conformal import (
    SpatialConformalConfig,
    _haversine_matrix_km,
    fit_spatial_multiclass_conformal,
    spatial_localization_preflight,
)


def _config() -> SpatialConformalConfig:
    return SpatialConformalConfig(
        candidate_bandwidth_km=(50.0, 150.0, 500.0, 2000.0),
        minimum_calibration_samples=40,
        minimum_effective_sample_size=15.0,
        effective_sample_size_quantile=0.10,
        calibration_anchor_limit=120,
        maximum_neighbors=120,
    )


def test_haversine_distance_wraps_the_antimeridian() -> None:
    distance = _haversine_matrix_km(
        np.asarray([[0.0, 179.0]]),
        np.asarray([[0.0, -179.0]]),
    )
    assert 200.0 < float(distance[0, 0]) < 230.0


def test_spatial_multiclass_thresholds_adapt_to_geographic_error_regimes() -> None:
    rng = np.random.default_rng(12)
    north = np.column_stack(
        (rng.normal(45.0, 0.25, 60), rng.normal(8.0, 0.25, 60))
    )
    south = np.column_stack(
        (rng.normal(-25.0, 0.25, 60), rng.normal(28.0, 0.25, 60))
    )
    coordinates = np.vstack((north, south))
    targets = np.zeros(120, dtype=int)
    probabilities = np.vstack(
        (
            np.tile([0.92, 0.08], (60, 1)),
            np.tile([0.55, 0.45], (60, 1)),
        )
    )
    result = fit_spatial_multiclass_conformal(
        probabilities,
        targets,
        coordinates,
        np.asarray([[45.0, 8.0], [-25.0, 28.0]]),
        alpha=0.10,
        method="lac",
        config=_config(),
    )
    assert result.preflight["status"] == "ready_empirical_comparator"
    assert np.all(result.identified)
    assert result.thresholds[0] < result.thresholds[1]
    assert result.nearest_calibration_distance_km.max() < 100.0


def test_all_task_gate_does_not_force_localized_crc() -> None:
    coordinates = np.column_stack(
        (np.linspace(40.0, 45.0, 120), np.linspace(5.0, 10.0, 120))
    )
    report = spatial_localization_preflight(
        coordinates,
        coordinates[:40],
        task_geometry="multilabel",
        config=_config(),
    )
    assert report["status"] == "screened_not_run_task_geometry"
    assert report["run_local_method"] is False
    assert report["formal_anchor"] == "conformal_risk_control"


def test_missing_coordinates_fail_closed_without_blocking_global_anchor() -> None:
    report = spatial_localization_preflight(
        None,
        None,
        task_geometry="segmentation",
        config=_config(),
    )
    assert report["status"] == "not_identified_missing_coordinates"
    assert report["run_local_method"] is False
    assert report["formal_anchor"] == "conformal_risk_control"
