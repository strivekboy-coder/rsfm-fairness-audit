from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.geobwer_uncertainty import (
    UncertaintyProtocolError,
    multiclass_nonconformity_scores,
)


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class SpatialConformalConfig:
    """Frozen design for the geographic-localization comparator.

    This is deliberately an empirical localization layer rather than a claim
    of unconditional pointwise conformal validity.  Exact marginal validity
    remains anchored by the ordinary split-conformal/CRC result.
    """

    candidate_bandwidth_km: tuple[float, ...] = (
        25.0,
        50.0,
        100.0,
        200.0,
        400.0,
        800.0,
        1600.0,
        3200.0,
        6400.0,
    )
    minimum_calibration_samples: int = 60
    minimum_effective_sample_size: float = 50.0
    effective_sample_size_quantile: float = 0.10
    calibration_anchor_limit: int = 1024
    maximum_neighbors: int = 2048
    distance_metric: str = "haversine"
    kernel: str = "gaussian"
    test_atom_weight: float = 1.0

    def __post_init__(self) -> None:
        candidates = tuple(float(value) for value in self.candidate_bandwidth_km)
        if not candidates or any(value <= 0.0 for value in candidates):
            raise ValueError("candidate_bandwidth_km must contain positive values.")
        if tuple(sorted(set(candidates))) != candidates:
            raise ValueError("candidate_bandwidth_km must be unique and increasing.")
        if self.minimum_calibration_samples < 2:
            raise ValueError("minimum_calibration_samples must be at least two.")
        if self.minimum_effective_sample_size <= 1.0:
            raise ValueError("minimum_effective_sample_size must exceed one.")
        if not 0.0 < self.effective_sample_size_quantile <= 0.5:
            raise ValueError("effective_sample_size_quantile must be in (0, 0.5].")
        if self.calibration_anchor_limit < 2 or self.maximum_neighbors < 2:
            raise ValueError("anchor and neighbor limits must be at least two.")
        if self.distance_metric != "haversine":
            raise ValueError("Global GeoFM localization is frozen to haversine distance.")
        if self.kernel != "gaussian":
            raise ValueError("Only the preregistered Gaussian geographic kernel is supported.")
        if self.test_atom_weight <= 0.0:
            raise ValueError("test_atom_weight must be positive.")


@dataclass(frozen=True)
class SpatialConformalResult:
    method: str
    thresholds: np.ndarray
    effective_sample_size: np.ndarray
    nearest_calibration_distance_km: np.ndarray
    identified: np.ndarray
    bandwidth_km: float
    neighbor_count: int
    calibration_samples: int
    preflight: Mapping[str, Any]


def _coordinates(values: Sequence[Sequence[float]], *, label: str) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise UncertaintyProtocolError(f"{label} coordinates must have shape [N,2].")
    if not np.all(np.isfinite(coordinates)):
        raise UncertaintyProtocolError(f"{label} coordinates contain missing/nonfinite values.")
    if np.any(np.abs(coordinates[:, 0]) > 90.0) or np.any(np.abs(coordinates[:, 1]) > 180.0):
        raise UncertaintyProtocolError(f"{label} coordinates fall outside latitude/longitude bounds.")
    return coordinates


def _haversine_matrix_km(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_rad = np.radians(left)
    right_rad = np.radians(right)
    lat_left = left_rad[:, 0][:, None]
    lon_left = left_rad[:, 1][:, None]
    lat_right = right_rad[:, 0][None, :]
    lon_right = right_rad[:, 1][None, :]
    delta_lat = lat_right - lat_left
    delta_lon = lon_right - lon_left
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_left) * np.cos(lat_right) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _effective_sample_size(weights: np.ndarray) -> np.ndarray:
    numerator = np.sum(weights, axis=1) ** 2
    denominator = np.sum(weights**2, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def _nearest_neighbors(
    calibration_coordinates: np.ndarray,
    query_coordinates: np.ndarray,
    *,
    maximum_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    count = min(int(maximum_neighbors), len(calibration_coordinates))
    try:
        from sklearn.neighbors import BallTree

        tree = BallTree(np.radians(calibration_coordinates), metric="haversine")
        distances, indexes = tree.query(np.radians(query_coordinates), k=count)
        return distances * EARTH_RADIUS_KM, indexes.astype(np.int64), "sklearn_balltree"
    except ImportError:
        # Exact fallback keeps the package's minimal dependency set. Formal
        # campaigns already install scikit-learn for the supervised probes.
        distances = _haversine_matrix_km(query_coordinates, calibration_coordinates)
        if count == len(calibration_coordinates):
            indexes = np.argsort(distances, axis=1, kind="stable")
        else:
            candidates = np.argpartition(distances, count - 1, axis=1)[:, :count]
            candidate_distances = np.take_along_axis(distances, candidates, axis=1)
            order = np.argsort(candidate_distances, axis=1, kind="stable")
            indexes = np.take_along_axis(candidates, order, axis=1)
        selected = np.take_along_axis(distances, indexes, axis=1)
        return selected, indexes.astype(np.int64), "numpy_exact_fallback"


def _select_bandwidth(
    coordinates: np.ndarray,
    config: SpatialConformalConfig,
) -> tuple[float | None, list[dict[str, float]], str]:
    anchor_count = min(len(coordinates), int(config.calibration_anchor_limit))
    # Evenly spaced deterministic anchors avoid geography-dependent random
    # seeds and make the selected bandwidth reproducible from the coordinates.
    anchors = np.unique(np.linspace(0, len(coordinates) - 1, anchor_count, dtype=int))
    maximum_neighbors = min(int(config.maximum_neighbors) + 1, len(coordinates))
    distances, indexes, backend = _nearest_neighbors(
        coordinates,
        coordinates[anchors],
        maximum_neighbors=maximum_neighbors,
    )
    # Remove each anchor's own calibration atom. Other zero-distance samples
    # remain legitimate neighbours.
    keep_count = min(int(config.maximum_neighbors), len(coordinates) - 1)
    neighbor_distances = np.empty((len(anchors), keep_count), dtype=float)
    for row_index, anchor in enumerate(anchors):
        keep = indexes[row_index] != anchor
        retained = distances[row_index, keep]
        if len(retained) < keep_count:
            raise UncertaintyProtocolError("Could not construct leave-one-out spatial neighbours.")
        neighbor_distances[row_index] = retained[:keep_count]

    diagnostics: list[dict[str, float]] = []
    selected: float | None = None
    for bandwidth in config.candidate_bandwidth_km:
        weights = np.exp(-0.5 * (neighbor_distances / float(bandwidth)) ** 2)
        ess = _effective_sample_size(weights)
        ess_gate = float(np.quantile(ess, config.effective_sample_size_quantile))
        diagnostics.append(
            {
                "bandwidth_km": float(bandwidth),
                "ess_gate_quantile": ess_gate,
                "median_ess": float(np.median(ess)),
                "median_nearest_distance_km": float(np.median(neighbor_distances[:, 0])),
                "median_weighted_radius_km": float(
                    np.median(
                        np.divide(
                            np.sum(weights * neighbor_distances, axis=1),
                            np.sum(weights, axis=1),
                            out=np.full(len(weights), np.inf),
                            where=np.sum(weights, axis=1) > 0.0,
                        )
                    )
                ),
            }
        )
        if selected is None and ess_gate >= config.minimum_effective_sample_size:
            selected = float(bandwidth)
    return selected, diagnostics, backend


def spatial_localization_preflight(
    calibration_coordinates: Sequence[Sequence[float]] | None,
    test_coordinates: Sequence[Sequence[float]] | None,
    *,
    task_geometry: str,
    config: SpatialConformalConfig,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "geobwer.spatial_localization_preflight.v1",
        "task_geometry": str(task_geometry),
        "config": asdict(config),
        "formal_anchor": (
            "ordinary_split_conformal"
            if task_geometry == "multiclass"
            else "conformal_risk_control"
        ),
        "local_method_validity_scope": (
            "empirical_geographic_localization_comparator; the test-centred geographic "
            "kernel is not asserted to be a density-ratio weight or an unconditional "
            "finite-sample pointwise coverage guarantee"
        ),
    }
    if calibration_coordinates is None or test_coordinates is None:
        report.update(
            {
                "status": "not_identified_missing_coordinates",
                "run_local_method": False,
                "reason": "Both calibration and test coordinates are required.",
            }
        )
        return report
    try:
        calibration = _coordinates(calibration_coordinates, label="calibration")
        test = _coordinates(test_coordinates, label="test")
    except UncertaintyProtocolError as exc:
        report.update(
            {
                "status": "not_identified_invalid_coordinates",
                "run_local_method": False,
                "reason": str(exc),
            }
        )
        return report
    report.update(
        {
            "calibration_samples": len(calibration),
            "test_samples": len(test),
            "coordinate_system": "EPSG:4326 latitude/longitude",
            "distance_metric": "great_circle_haversine_km",
        }
    )
    if len(calibration) < config.minimum_calibration_samples:
        report.update(
            {
                "status": "not_identified_insufficient_calibration_support",
                "run_local_method": False,
                "reason": (
                    f"Calibration support {len(calibration)} is below the frozen minimum "
                    f"{config.minimum_calibration_samples}."
                ),
            }
        )
        return report
    selected, candidates, backend = _select_bandwidth(calibration, config)
    report["bandwidth_diagnostics"] = candidates
    report["neighbor_backend"] = backend
    if selected is None:
        report.update(
            {
                "status": "not_identified_local_ess",
                "run_local_method": False,
                "reason": "No preregistered bandwidth passed the calibration-only local ESS gate.",
            }
        )
        return report
    report["selected_bandwidth_km"] = selected
    if task_geometry != "multiclass":
        report.update(
            {
                "status": "screened_not_run_task_geometry",
                "run_local_method": False,
                "reason": (
                    "A task-correct geographically localized CRC with the required guarantee "
                    "has not been established; retain the formal global CRC and audit its "
                    "spatial/event slices with GeoBWER."
                ),
            }
        )
        return report
    report.update(
        {
            "status": "ready_empirical_comparator",
            "run_local_method": True,
            "bandwidth_selection": (
                "smallest preregistered bandwidth whose calibration-only leave-one-out "
                f"ESS q={config.effective_sample_size_quantile:g} reaches "
                f"{config.minimum_effective_sample_size:g}"
            ),
        }
    )
    return report


def _weighted_thresholds(
    scores: np.ndarray,
    neighbor_indexes: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    test_atom_weight: float,
) -> np.ndarray:
    local_scores = scores[neighbor_indexes]
    order = np.argsort(local_scores, axis=1, kind="stable")
    ordered_scores = np.take_along_axis(local_scores, order, axis=1)
    ordered_weights = np.take_along_axis(weights, order, axis=1)
    target = (1.0 - float(alpha)) * (
        np.sum(ordered_weights, axis=1) + float(test_atom_weight)
    )
    cumulative = np.cumsum(ordered_weights, axis=1)
    reached = cumulative >= target[:, None]
    any_reached = np.any(reached, axis=1)
    indexes = np.argmax(reached, axis=1)
    thresholds = ordered_scores[np.arange(len(ordered_scores)), indexes].astype(float)
    thresholds[~any_reached] = np.inf
    return thresholds


def fit_spatial_multiclass_conformal(
    calibration_probabilities: Sequence[Sequence[float]],
    calibration_targets: Sequence[int],
    calibration_coordinates: Sequence[Sequence[float]],
    test_coordinates: Sequence[Sequence[float]],
    *,
    alpha: float,
    method: str,
    config: SpatialConformalConfig,
    raps_lambda: float = 0.01,
    raps_k_reg: int = 5,
) -> SpatialConformalResult:
    calibration = _coordinates(calibration_coordinates, label="calibration")
    test = _coordinates(test_coordinates, label="test")
    preflight = spatial_localization_preflight(
        calibration,
        test,
        task_geometry="multiclass",
        config=config,
    )
    if not preflight.get("run_local_method", False):
        raise UncertaintyProtocolError(
            f"Spatial conformal comparator failed preflight: {preflight.get('reason', preflight['status'])}"
        )
    scores = multiclass_nonconformity_scores(
        calibration_probabilities,
        calibration_targets,
        method=method,
        raps_lambda=raps_lambda,
        raps_k_reg=raps_k_reg,
    )
    if len(scores) != len(calibration):
        raise UncertaintyProtocolError("Calibration probabilities and coordinates must align.")
    bandwidth = float(preflight["selected_bandwidth_km"])
    distances, indexes, backend = _nearest_neighbors(
        calibration,
        test,
        maximum_neighbors=config.maximum_neighbors,
    )
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    ess = _effective_sample_size(weights)
    identified = ess >= float(config.minimum_effective_sample_size)
    thresholds = _weighted_thresholds(
        scores,
        indexes,
        weights,
        alpha=alpha,
        test_atom_weight=config.test_atom_weight,
    )
    # A low-support location receives the maximally conservative prediction
    # set, but remains explicitly marked as unsupported so perfect coverage
    # cannot be mistaken for useful uncertainty quantification.
    thresholds[~identified] = np.inf
    result_preflight = dict(preflight)
    result_preflight.update(
        {
            "neighbor_backend": backend,
            "test_identified_fraction": float(np.mean(identified)),
            "minimum_test_ess": float(np.min(ess)),
            "median_test_ess": float(np.median(ess)),
            "infinite_threshold_fraction": float(np.mean(~np.isfinite(thresholds))),
        }
    )
    return SpatialConformalResult(
        method=method,
        thresholds=thresholds,
        effective_sample_size=ess,
        nearest_calibration_distance_km=distances[:, 0],
        identified=identified,
        bandwidth_km=bandwidth,
        neighbor_count=indexes.shape[1],
        calibration_samples=len(calibration),
        preflight=result_preflight,
    )


__all__ = [
    "EARTH_RADIUS_KM",
    "SpatialConformalConfig",
    "SpatialConformalResult",
    "fit_spatial_multiclass_conformal",
    "spatial_localization_preflight",
]
