from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.geobwer_uncertainty import (
    apply_selective_threshold,
    crc_audit_rows,
    fit_false_negative_crc,
    fit_multiclass_conformal,
    fit_selective_threshold,
    multiclass_conformal_audit_rows,
    multiclass_prediction_sets,
)
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.spatial_conformal import (
    SpatialConformalConfig,
    fit_spatial_multiclass_conformal,
    spatial_localization_preflight,
)


class ExtensionAuditError(RuntimeError):
    """Raised when a calibrated extension cannot be traced to compatible artifacts."""


def _protocol(path_or_protocol: str | Path | BWERProtocol) -> BWERProtocol:
    if isinstance(path_or_protocol, BWERProtocol):
        return path_or_protocol
    return BWERProtocol.from_mapping(load_yaml(path_or_protocol))


def _derived_protocol(base: BWERProtocol, *, loss_name: str, extension: str, details: Mapping[str, Any]) -> BWERProtocol:
    metadata = dict(base.metadata)
    metadata.update({"uncertainty_extension": extension, **{str(key): str(value) for key, value in details.items()}})
    return replace(base, loss_name=loss_name, metadata=tuple(sorted(metadata.items())))


def _retag(rows: Sequence[Mapping[str, Any]], protocol: BWERProtocol) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["protocol_hash"] = protocol.signature
        row["metric_version"] = protocol.metric_version
        output.append(row)
    return output


def _load_npz_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        probability_key = "probabilities" if "probabilities" in artifact else "y_prob"
        target_key = "targets" if "targets" in artifact else "labels" if "labels" in artifact else "y_true"
        if probability_key not in artifact or target_key not in artifact:
            raise ExtensionAuditError(f"{path} must contain probabilities and targets/labels.")
        return np.asarray(artifact[probability_key]), np.asarray(artifact[target_key])


def _npz_scalar_text(artifact: np.lib.npyio.NpzFile, key: str) -> str:
    value = np.asarray(artifact[key])
    if value.size != 1:
        raise ExtensionAuditError(f"Calibration field {key!r} must be scalar.")
    return str(value.reshape(-1)[0])


def _load_calibration_arrays(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str, tuple[str, ...]]:
    calibration_path = Path(path)
    probabilities, targets = _load_npz_arrays(calibration_path)
    with np.load(calibration_path, allow_pickle=False) as artifact:
        if "split_role" not in artifact or "test_rows_used" not in artifact:
            raise ExtensionAuditError(
                "Formal calibration NPZ must declare split_role and test_rows_used; "
                "an unlabelled probability file cannot establish calibration/test separation."
            )
        split_role = _npz_scalar_text(artifact, "split_role").lower()
        test_rows_used = _npz_scalar_text(artifact, "test_rows_used").lower()
        if split_role not in {"calibration", "validation"}:
            raise ExtensionAuditError(f"Invalid calibration split_role={split_role!r}.")
        if test_rows_used not in {"false", "0"}:
            raise ExtensionAuditError("Calibration artifact declares test_rows_used=true.")
        if "sample_id" not in artifact:
            raise ExtensionAuditError("Formal calibration NPZ must include unique sample_id values.")
        sample_ids = np.asarray(artifact["sample_id"]).astype(str)
        if (
            sample_ids.shape != (len(probabilities),)
            or any(not value.strip() for value in sample_ids.tolist())
            or len(set(sample_ids.tolist())) != len(sample_ids)
        ):
            raise ExtensionAuditError("Calibration sample_id must be non-empty, unique, and align with probability rows.")
    digest = file_sha256(calibration_path)
    manifest_digest = ""
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest_test_rows = str(manifest.get("test_rows_used", True)).lower()
        if manifest_test_rows not in {"false", "0"}:
            raise ExtensionAuditError("Calibration manifest must explicitly declare test_rows_used=false.")
        recorded = str(manifest.get("probabilities_sha256", ""))
        if recorded != digest:
            raise ExtensionAuditError("Calibration manifest probability hash does not match the NPZ artifact.")
        manifest_digest = file_sha256(manifest_path)
    return probabilities, targets, digest, manifest_digest, tuple(sample_ids.tolist())


def _coordinates_from_npz(path: str | Path) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as artifact:
        if "latitude" not in artifact or "longitude" not in artifact:
            return None
        latitude = np.asarray(artifact["latitude"], dtype=float)
        longitude = np.asarray(artifact["longitude"], dtype=float)
    if latitude.ndim != 1 or longitude.ndim != 1 or latitude.shape != longitude.shape:
        return None
    return np.column_stack((latitude, longitude))


def _coordinates_from_rows(rows: Sequence[Mapping[str, Any]]) -> np.ndarray | None:
    values: list[tuple[float, float]] = []
    for row in rows:
        try:
            latitude = float(row.get("latitude", row.get("lat", "")))
            longitude = float(
                row.get("longitude", row.get("lon", row.get("lng", "")))
            )
        except (TypeError, ValueError):
            return None
        values.append((latitude, longitude))
    return np.asarray(values, dtype=float) if values else None


def _write_spatial_preflight(path: Path, report: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _prediction_set_efficiency_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["miscoverage_loss"] = row["risk"]
        row["risk"] = float(row["set_size_fraction"])
        output.append(row)
    return output


def _spatial_multiclass_rows(
    *,
    result: Any,
    probabilities: np.ndarray,
    targets: np.ndarray,
    sample_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sets = multiclass_prediction_sets(
        probabilities,
        result.thresholds,
        method=result.method,
    )
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sample_rows):
        covered = bool(sets[index, targets[index]])
        row = dict(source)
        row.update(
            {
                "risk": float(not covered),
                "miscoverage_loss": float(not covered),
                "covered": covered,
                "set_size": int(np.sum(sets[index])),
                "set_size_fraction": float(np.mean(sets[index])),
                "conformal_method": result.method,
                "conformal_threshold": float(result.thresholds[index]),
                "spatial_bandwidth_km": result.bandwidth_km,
                "spatial_effective_sample_size": float(
                    result.effective_sample_size[index]
                ),
                "nearest_calibration_distance_km": float(
                    result.nearest_calibration_distance_km[index]
                ),
                "spatial_support_identified": bool(result.identified[index]),
                "localization_method": "test_centered_gaussian_geographic_kernel",
            }
        )
        rows.append(row)
    return rows


def _assert_calibration_test_disjoint(
    calibration_sample_ids: Sequence[str],
    test_rows: Sequence[Mapping[str, Any]],
) -> None:
    test_ids = [str(row.get("sample_id", "")).strip() for row in test_rows]
    if any(not value for value in test_ids) or len(set(test_ids)) != len(test_ids):
        raise ExtensionAuditError("Test sample_id values must be non-empty and unique.")
    overlap = set(calibration_sample_ids) & set(test_ids)
    if overlap:
        raise ExtensionAuditError(
            f"Calibration/test sample leakage detected for {len(overlap)} sample IDs."
        )


def _array_bundle_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _audit_and_write(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    protocol: BWERProtocol,
    group_columns: Sequence[str],
    cluster_column: str | None,
    n_bootstrap: int,
    seed: int,
    required_group_universe: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Path]:
    tagged = _retag(rows, protocol)
    table = output_dir / "derived_audit_table.csv"
    write_csv(table, tagged)
    audit = audit_rows(
        tagged,
        group_columns=group_columns,
        protocol=protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column=cluster_column,
        formal=True,
        require_probabilities=False,
        required_group_universe=required_group_universe,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    artifacts = audit.to_report(output_dir / "geobwer")
    artifacts["derived_audit_table"] = table
    return artifacts


def _group_universe(
    rows: Sequence[Mapping[str, Any]],
    group_columns: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    return {
        str(column): tuple(sorted({str(row[column]) for row in rows}))
        for column in group_columns
    }


def _accepted(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_selective_group_coverage(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    group_columns: Sequence[str],
) -> tuple[Path, int, float]:
    diagnostics: list[dict[str, Any]] = []
    zero_accepted = 0
    minimum_coverage = 1.0
    for axis in group_columns:
        values = sorted({str(row[axis]) for row in rows})
        for group in values:
            subset = [row for row in rows if str(row[axis]) == group]
            accepted = [row for row in subset if _accepted(row.get("accepted", False))]
            coverage = len(accepted) / len(subset)
            minimum_coverage = min(minimum_coverage, coverage)
            zero_accepted += int(not accepted)
            diagnostics.append(
                {
                    "axis": axis,
                    "group": group,
                    "source_support": len(subset),
                    "accepted_support": len(accepted),
                    "coverage": coverage,
                    "conditional_risk": (
                        float(np.mean([float(row["risk"]) for row in accepted]))
                        if accepted
                        else ""
                    ),
                    "selective_risk_identified": bool(accepted),
                }
            )
    path = output_dir / "selective_group_coverage.csv"
    write_csv(path, diagnostics)
    return path, zero_accepted, minimum_coverage


def _tail_target_diagnostics(
    audit_artifacts: Mapping[str, Path],
    *,
    risk_target: float,
) -> dict[str, float]:
    rows = read_csv_rows(audit_artifacts["summary"])
    if not rows or rows[0].get("mean_risk", "") in {"", None}:
        return {
            "mean_target_violation": float("nan"),
            "tail_target_violation": float("nan"),
            "positive_mean_target_violation": float("nan"),
            "positive_tail_target_violation": float("nan"),
        }
    mean_risk = float(rows[0]["mean_risk"])
    tail_risk = float(rows[0]["tail_risk"])
    return {
        "mean_target_violation": mean_risk - float(risk_target),
        "tail_target_violation": tail_risk - float(risk_target),
        "positive_mean_target_violation": max(0.0, mean_risk - float(risk_target)),
        "positive_tail_target_violation": max(0.0, tail_risk - float(risk_target)),
    }


def run_multiclass_uncertainty_suite(
    calibration_probabilities: str | Path,
    test_formal_dir: str | Path,
    output_dir: str | Path,
    *,
    protocol: str | Path | BWERProtocol,
    group_columns: Sequence[str],
    calibration_manifest: str | Path | None = None,
    conformal_methods: Sequence[str] = ("lac", "aps", "raps"),
    selective_coverages: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    alpha: float = 0.10,
    n_bootstrap: int = 2000,
    seed: int = 42,
    spatial_conformal_config: SpatialConformalConfig | None = None,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    base = _protocol(protocol)
    if base.audit_measure != "balanced":
        raise ExtensionAuditError(
            "Selective-GeoBWER currently requires audit_measure=balanced so the source-population "
            "group measure is unchanged after abstention."
        )
    (
        calibration_probs,
        calibration_targets,
        calibration_sha256,
        calibration_manifest_sha256,
        calibration_sample_ids,
    ) = _load_calibration_arrays(calibration_probabilities, manifest_path=calibration_manifest)
    test_probs, test_targets = _load_npz_arrays(Path(test_formal_dir) / "probabilities.npz")
    test_rows = read_csv_rows(Path(test_formal_dir) / "formal_audit_table.csv")
    if len(test_rows) != len(test_probs):
        raise ExtensionAuditError("Test probability rows and formal audit table do not align.")
    _assert_calibration_test_disjoint(calibration_sample_ids, test_rows)
    cluster = base.spatial_block_column if base.inference_method == "spatial_maxt" else base.cluster_column
    artifacts: dict[str, Path] = {}
    summary_rows: list[dict[str, Any]] = []
    calibration_coordinates = (
        _coordinates_from_npz(calibration_probabilities)
        if spatial_conformal_config is not None
        else None
    )
    test_coordinates = (
        _coordinates_from_rows(test_rows)
        if spatial_conformal_config is not None
        else None
    )
    spatial_preflight: dict[str, Any] | None = None
    if spatial_conformal_config is not None:
        spatial_preflight = spatial_localization_preflight(
            calibration_coordinates,
            test_coordinates,
            task_geometry="multiclass",
            config=spatial_conformal_config,
        )
        preflight_path = _write_spatial_preflight(
            output / "spatial_localization_preflight.json",
            spatial_preflight,
        )
        artifacts["spatial_localization_preflight"] = preflight_path
    for method in conformal_methods:
        model = fit_multiclass_conformal(
            calibration_probs, calibration_targets, alpha=alpha, method=method
        )
        rows = multiclass_conformal_audit_rows(
            model, test_probs, test_targets, sample_rows=test_rows
        )
        extension_protocol = _derived_protocol(
            base,
            loss_name="miscoverage_loss",
            extension=f"split_conformal_{method}",
            details={
                "alpha": alpha,
                "calibration_sha256": calibration_sha256,
                "calibration_manifest_sha256": calibration_manifest_sha256,
                "boundary_rule": "deterministic_nonconformity_no_randomized_boundary",
                "raps_lambda": model.raps_lambda,
                "raps_k_reg": model.raps_k_reg,
            },
        )
        run_output = ensure_dir(output / f"conformal_{method}")
        run_artifacts = _audit_and_write(
            rows,
            run_output,
            protocol=extension_protocol,
            group_columns=group_columns,
            cluster_column=cluster,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for name, path in run_artifacts.items():
            artifacts[f"conformal_{method}_{name}"] = path
        efficiency_protocol = _derived_protocol(
            base,
            loss_name="prediction_set_fraction",
            extension=f"split_conformal_{method}_efficiency",
            details={
                "alpha": alpha,
                "calibration_sha256": calibration_sha256,
                "efficiency_loss": "set_size_divided_by_number_of_classes",
            },
        )
        efficiency_artifacts = _audit_and_write(
            _prediction_set_efficiency_rows(rows),
            ensure_dir(output / f"conformal_{method}_efficiency"),
            protocol=efficiency_protocol,
            group_columns=group_columns,
            cluster_column=cluster,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for name, path in efficiency_artifacts.items():
            artifacts[f"conformal_{method}_efficiency_{name}"] = path
        target_diagnostics = _tail_target_diagnostics(run_artifacts, risk_target=alpha)
        summary_rows.append(
            {
                "extension": f"conformal_{method}",
                "target_coverage": 1.0 - alpha,
                "test_coverage": 1.0 - float(np.mean([row["risk"] for row in rows])),
                "mean_set_size": float(np.mean([row["set_size"] for row in rows])),
                "mean_set_size_fraction": float(np.mean([row["set_size_fraction"] for row in rows])),
                "calibration_threshold": model.global_threshold,
                "protocol_hash": extension_protocol.signature,
                **target_diagnostics,
            }
        )
        if spatial_preflight is not None and spatial_preflight.get(
            "run_local_method", False
        ):
            assert calibration_coordinates is not None
            assert test_coordinates is not None
            spatial_result = fit_spatial_multiclass_conformal(
                calibration_probs,
                calibration_targets,
                calibration_coordinates,
                test_coordinates,
                alpha=alpha,
                method=method,
                config=spatial_conformal_config,
            )
            spatial_rows = _spatial_multiclass_rows(
                result=spatial_result,
                probabilities=test_probs,
                targets=test_targets,
                sample_rows=test_rows,
            )
            spatial_protocol = _derived_protocol(
                base,
                loss_name="miscoverage_loss",
                extension=f"geo_kernel_conformal_{method}",
                details={
                    "alpha": alpha,
                    "calibration_sha256": calibration_sha256,
                    "bandwidth_km": spatial_result.bandwidth_km,
                    "bandwidth_source": "calibration_only_leave_one_out_ess_gate",
                    "distance_metric": "great_circle_haversine_km",
                    "kernel": "gaussian",
                    "test_atom_weight": spatial_conformal_config.test_atom_weight,
                    "validity_scope": spatial_result.preflight[
                        "local_method_validity_scope"
                    ],
                },
            )
            spatial_output = ensure_dir(output / f"geo_kernel_conformal_{method}")
            spatial_artifacts = _audit_and_write(
                spatial_rows,
                spatial_output,
                protocol=spatial_protocol,
                group_columns=group_columns,
                cluster_column=cluster,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            for name, path in spatial_artifacts.items():
                artifacts[f"geo_kernel_conformal_{method}_{name}"] = path
            method_preflight_path = _write_spatial_preflight(
                spatial_output / "spatial_method_diagnostics.json",
                spatial_result.preflight,
            )
            artifacts[
                f"geo_kernel_conformal_{method}_diagnostics"
            ] = method_preflight_path
            spatial_efficiency_protocol = _derived_protocol(
                base,
                loss_name="prediction_set_fraction",
                extension=f"geo_kernel_conformal_{method}_efficiency",
                details={
                    "alpha": alpha,
                    "bandwidth_km": spatial_result.bandwidth_km,
                    "efficiency_loss": "set_size_divided_by_number_of_classes",
                    "unsupported_location_policy": "all_classes_plus_explicit_flag",
                },
            )
            spatial_efficiency_artifacts = _audit_and_write(
                _prediction_set_efficiency_rows(spatial_rows),
                ensure_dir(output / f"geo_kernel_conformal_{method}_efficiency"),
                protocol=spatial_efficiency_protocol,
                group_columns=group_columns,
                cluster_column=cluster,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            for name, path in spatial_efficiency_artifacts.items():
                artifacts[
                    f"geo_kernel_conformal_{method}_efficiency_{name}"
                ] = path
            spatial_target_diagnostics = _tail_target_diagnostics(
                spatial_artifacts, risk_target=alpha
            )
            summary_rows.append(
                {
                    "extension": f"geo_kernel_conformal_{method}",
                    "role": "empirical_spatial_localization_comparator",
                    "formal_marginal_anchor": f"conformal_{method}",
                    "target_coverage": 1.0 - alpha,
                    "test_coverage": 1.0
                    - float(np.mean([row["risk"] for row in spatial_rows])),
                    "mean_set_size": float(
                        np.mean([row["set_size"] for row in spatial_rows])
                    ),
                    "mean_set_size_fraction": float(
                        np.mean([row["set_size_fraction"] for row in spatial_rows])
                    ),
                    "spatial_bandwidth_km": spatial_result.bandwidth_km,
                    "minimum_test_ess": float(
                        np.min(spatial_result.effective_sample_size)
                    ),
                    "median_test_ess": float(
                        np.median(spatial_result.effective_sample_size)
                    ),
                    "spatial_support_identified_fraction": float(
                        np.mean(spatial_result.identified)
                    ),
                    "protocol_hash": spatial_protocol.signature,
                    **spatial_target_diagnostics,
                }
            )
    if spatial_preflight is not None and not spatial_preflight.get(
        "run_local_method", False
    ):
        summary_rows.append(
            {
                "extension": "geo_kernel_conformal_preflight",
                "role": "screened_not_run",
                "spatial_localization_status": spatial_preflight["status"],
                "reason": spatial_preflight.get("reason", ""),
            }
        )
    calibration_confidence = np.max(calibration_probs, axis=1)
    test_confidence = np.max(test_probs, axis=1)
    test_risk = (np.argmax(test_probs, axis=1) != test_targets).astype(float)
    for coverage in selective_coverages:
        selective_model = fit_selective_threshold(calibration_confidence, target_coverage=float(coverage))
        applied = apply_selective_threshold(
            selective_model, test_risk, test_confidence, sample_rows=test_rows
        )
        retained = [row for row in applied if row["accepted"]]
        extension_protocol = _derived_protocol(
            base,
            loss_name=base.loss_name,
            extension="selective_prediction",
            details={
                "target_coverage": coverage,
                "confidence_threshold": selective_model.confidence_threshold,
                "threshold_source": "calibration_only",
                "calibration_sha256": calibration_sha256,
            },
        )
        slug = f"selective_{int(round(100 * coverage)):03d}"
        run_output = ensure_dir(output / slug)
        coverage_path, zero_groups, minimum_group_coverage = _write_selective_group_coverage(
            applied,
            run_output,
            group_columns=group_columns,
        )
        run_artifacts = _audit_and_write(
            retained,
            run_output,
            protocol=extension_protocol,
            group_columns=group_columns,
            cluster_column=cluster,
            n_bootstrap=n_bootstrap,
            seed=seed,
            required_group_universe=_group_universe(test_rows, group_columns),
        )
        for name, path in run_artifacts.items():
            artifacts[f"{slug}_{name}"] = path
        artifacts[f"{slug}_group_coverage"] = coverage_path
        summary_rows.append(
            {
                "extension": slug,
                "target_coverage": coverage,
                "test_coverage": len(retained) / len(applied),
                "selective_risk": float(np.mean([row["risk"] for row in retained])),
                "minimum_group_coverage": minimum_group_coverage,
                "zero_accepted_groups": zero_groups,
                "selective_geobwer_identified": zero_groups == 0,
                "confidence_threshold": selective_model.confidence_threshold,
                "protocol_hash": extension_protocol.signature,
            }
        )
    summary_path = output / "uncertainty_summary.csv"
    write_csv(summary_path, summary_rows)
    artifacts["summary"] = summary_path
    return artifacts


def run_multilabel_uncertainty_suite(
    calibration_probabilities: str | Path,
    test_formal_dir: str | Path,
    output_dir: str | Path,
    *,
    protocol: str | Path | BWERProtocol,
    group_columns: Sequence[str] = ("country",),
    calibration_manifest: str | Path | None = None,
    selective_coverages: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    crc_alpha: float = 0.10,
    n_bootstrap: int = 2000,
    seed: int = 42,
    spatial_localization_config: SpatialConformalConfig | None = None,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    base = _protocol(protocol)
    if base.audit_measure != "balanced":
        raise ExtensionAuditError(
            "Selective-GeoBWER currently requires audit_measure=balanced so the source-population "
            "group measure is unchanged after abstention."
        )
    (
        calibration_probs,
        calibration_targets,
        calibration_sha256,
        calibration_manifest_sha256,
        calibration_sample_ids,
    ) = _load_calibration_arrays(calibration_probabilities, manifest_path=calibration_manifest)
    test_npz = Path(test_formal_dir) / "probabilities.npz"
    test_probs, test_targets = _load_npz_arrays(test_npz)
    with np.load(test_npz, allow_pickle=False) as artifact:
        thresholds = np.asarray(artifact["thresholds"] if "thresholds" in artifact else 0.5, dtype=float)
    test_rows = read_csv_rows(Path(test_formal_dir) / "formal_audit_table.csv")
    if len(test_rows) != len(test_probs):
        raise ExtensionAuditError("Test probability rows and formal audit table do not align.")
    _assert_calibration_test_disjoint(calibration_sample_ids, test_rows)
    cluster = base.spatial_block_column if base.inference_method == "spatial_maxt" else base.cluster_column
    artifacts: dict[str, Path] = {}
    summary_rows: list[dict[str, Any]] = []
    if spatial_localization_config is not None:
        spatial_preflight = spatial_localization_preflight(
            _coordinates_from_npz(calibration_probabilities),
            _coordinates_from_rows(test_rows),
            task_geometry="multilabel",
            config=spatial_localization_config,
        )
        artifacts["spatial_localization_preflight"] = _write_spatial_preflight(
            output / "spatial_localization_preflight.json",
            spatial_preflight,
        )
        summary_rows.append(
            {
                "extension": "geo_kernel_crc_preflight",
                "role": "screened_not_run",
                "formal_uncertainty_method_complete": True,
                "formal_anchor": "conformal_risk_control",
                "localized_geo_method_applicability": (
                    "not_formally_established_for_multilabel_under_frozen_protocol"
                ),
                "spatial_localization_status": spatial_preflight["status"],
                "reason": spatial_preflight.get("reason", ""),
            }
        )
    crc_model = fit_false_negative_crc(
        calibration_probs, calibration_targets, alpha=crc_alpha, risk_name="false_negative_rate"
    )
    crc_rows = crc_audit_rows(crc_model, test_probs, test_targets, sample_rows=test_rows)
    crc_protocol = _derived_protocol(
        base,
        loss_name="false_negative_rate",
        extension="conformal_risk_control",
        details={
            "alpha": crc_alpha,
            "threshold_source": "validation_only",
            "calibration_sha256": calibration_sha256,
            "calibration_manifest_sha256": calibration_manifest_sha256,
        },
    )
    crc_artifacts = _audit_and_write(
        crc_rows,
        ensure_dir(output / "conformal_risk_control"),
        protocol=crc_protocol,
        group_columns=group_columns,
        cluster_column=cluster,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    artifacts.update({f"crc_{key}": value for key, value in crc_artifacts.items()})
    crc_target_diagnostics = _tail_target_diagnostics(crc_artifacts, risk_target=crc_alpha)
    summary_rows.append(
        {
            "extension": "conformal_risk_control",
            "formal_uncertainty_method_complete": True,
            "formal_anchor": "conformal_risk_control",
            "risk_target": crc_alpha,
            "test_mean_risk": float(np.mean([row["risk"] for row in crc_rows])),
            "mean_prediction_set_fraction": float(np.mean([row["prediction_set_fraction"] for row in crc_rows])),
            "probability_threshold": crc_model.probability_threshold,
            "protocol_hash": crc_protocol.signature,
            **crc_target_diagnostics,
        }
    )
    threshold_array = np.full(test_probs.shape[1], float(thresholds)) if thresholds.ndim == 0 else thresholds
    test_prediction = test_probs >= threshold_array[None, :]
    test_risk = np.mean(test_prediction != test_targets, axis=1)
    calibration_confidence = np.mean(np.maximum(calibration_probs, 1.0 - calibration_probs), axis=1)
    test_confidence = np.mean(np.maximum(test_probs, 1.0 - test_probs), axis=1)
    for coverage in selective_coverages:
        selective_model = fit_selective_threshold(calibration_confidence, target_coverage=float(coverage))
        applied = apply_selective_threshold(
            selective_model, test_risk, test_confidence, sample_rows=test_rows
        )
        retained = [row for row in applied if row["accepted"]]
        selective_protocol = _derived_protocol(
            base,
            loss_name="hamming_loss",
            extension="selective_prediction",
            details={
                "target_coverage": coverage,
                "threshold_source": "validation_only",
                "calibration_sha256": calibration_sha256,
            },
        )
        slug = f"selective_{int(round(100 * coverage)):03d}"
        run_output = ensure_dir(output / slug)
        coverage_path, zero_groups, minimum_group_coverage = _write_selective_group_coverage(
            applied,
            run_output,
            group_columns=group_columns,
        )
        run_artifacts = _audit_and_write(
            retained,
            run_output,
            protocol=selective_protocol,
            group_columns=group_columns,
            cluster_column=cluster,
            n_bootstrap=n_bootstrap,
            seed=seed,
            required_group_universe=_group_universe(test_rows, group_columns),
        )
        artifacts.update({f"{slug}_{key}": value for key, value in run_artifacts.items()})
        artifacts[f"{slug}_group_coverage"] = coverage_path
        summary_rows.append(
            {
                "extension": slug,
                "target_coverage": coverage,
                "test_coverage": len(retained) / len(test_rows),
                "selective_risk": float(np.mean([row["risk"] for row in retained])),
                "minimum_group_coverage": minimum_group_coverage,
                "zero_accepted_groups": zero_groups,
                "selective_geobwer_identified": zero_groups == 0,
                "protocol_hash": selective_protocol.signature,
            }
        )
    summary_path = output / "uncertainty_summary.csv"
    write_csv(summary_path, summary_rows)
    artifacts["summary"] = summary_path
    return artifacts


def run_segmentation_uncertainty_suite(
    calibration_probabilities: Sequence[np.ndarray],
    calibration_targets: Sequence[np.ndarray],
    test_formal_dir: str | Path,
    output_dir: str | Path,
    *,
    protocol: str | Path | BWERProtocol,
    group_columns: Sequence[str] = ("event_id",),
    calibration_valid_masks: Sequence[np.ndarray] | None = None,
    calibration_sample_ids: Sequence[str] | None = None,
    calibration_sample_rows: Sequence[Mapping[str, Any]] | None = None,
    selective_coverages: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    crc_alpha: float = 0.10,
    n_bootstrap: int = 2000,
    seed: int = 42,
    spatial_localization_config: SpatialConformalConfig | None = None,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    base = _protocol(protocol)
    if base.audit_measure != "balanced":
        raise ExtensionAuditError(
            "Selective-GeoBWER currently requires audit_measure=balanced so the source-population "
            "group measure is unchanged after abstention."
        )
    test_rows = read_csv_rows(Path(test_formal_dir) / "formal_audit_table.csv")
    if calibration_sample_ids is None:
        raise ExtensionAuditError(
            "Formal segmentation uncertainty audit requires calibration_sample_ids to certify split separation."
        )
    if len(calibration_sample_ids) != len(calibration_probabilities):
        raise ExtensionAuditError("calibration_sample_ids must align with calibration probability maps.")
    if len(set(str(value) for value in calibration_sample_ids)) != len(calibration_sample_ids):
        raise ExtensionAuditError("calibration_sample_ids must be unique.")
    _assert_calibration_test_disjoint([str(value) for value in calibration_sample_ids], test_rows)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    for row in test_rows:
        with np.load(Path(test_formal_dir) / row["probability_map_path"], allow_pickle=False) as artifact:
            probabilities.append(np.asarray(artifact["positive_probability"]))
            targets.append(np.asarray(artifact["target"]))
            if "valid" not in artifact:
                raise ExtensionAuditError(
                    f"{row['probability_map_path']} is missing the formal segmentation valid mask."
                )
            valid_masks.append(np.asarray(artifact["valid"], dtype=bool))
    calibration_probs = np.stack(calibration_probabilities)
    calibration_y = np.stack(calibration_targets)
    calibration_valid = (
        np.isin(calibration_y, (0, 1))
        if calibration_valid_masks is None
        else np.stack(calibration_valid_masks).astype(bool)
    )
    test_probs = np.stack(probabilities)
    test_y = np.stack(targets)
    test_valid = np.stack(valid_masks)
    calibration_sha256 = _array_bundle_sha256(calibration_probs, calibration_y, calibration_valid)
    crc_model = fit_false_negative_crc(
        calibration_probs,
        calibration_y,
        alpha=crc_alpha,
        risk_name="pixel_false_negative_rate",
        valid_masks=calibration_valid,
    )
    rows = crc_audit_rows(
        crc_model,
        test_probs,
        test_y,
        sample_rows=test_rows,
        valid_masks=test_valid,
    )
    extension_protocol = _derived_protocol(
        base,
        loss_name="pixel_false_negative_rate",
        extension="conformal_risk_control",
        details={
            "alpha": crc_alpha,
            "threshold_source": "validation_only",
            "calibration_sha256": calibration_sha256,
        },
    )
    cluster = base.spatial_block_column if base.inference_method == "spatial_maxt" else base.cluster_column
    artifacts = _audit_and_write(
        rows,
        ensure_dir(output / "conformal_risk_control"),
        protocol=extension_protocol,
        group_columns=group_columns,
        cluster_column=cluster,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    crc_target_diagnostics = _tail_target_diagnostics(artifacts, risk_target=crc_alpha)
    summary_rows: list[dict[str, Any]] = [
        {
            "extension": "conformal_risk_control",
            "formal_uncertainty_method_complete": True,
            "formal_anchor": "conformal_risk_control",
            "risk_target": crc_alpha,
            "test_mean_risk": float(np.mean([row["risk"] for row in rows])),
            "mean_prediction_set_fraction": float(np.mean([row["prediction_set_fraction"] for row in rows])),
            "probability_threshold": crc_model.probability_threshold,
            "protocol_hash": extension_protocol.signature,
            **crc_target_diagnostics,
        }
    ]
    if spatial_localization_config is not None:
        if calibration_sample_rows is not None and len(calibration_sample_rows) != len(
            calibration_probabilities
        ):
            raise ExtensionAuditError(
                "calibration_sample_rows must align with calibration probability maps."
            )
        spatial_preflight = spatial_localization_preflight(
            (
                _coordinates_from_rows(calibration_sample_rows)
                if calibration_sample_rows is not None
                else None
            ),
            _coordinates_from_rows(test_rows),
            task_geometry="segmentation",
            config=spatial_localization_config,
        )
        artifacts["spatial_localization_preflight"] = _write_spatial_preflight(
            output / "spatial_localization_preflight.json",
            spatial_preflight,
        )
        summary_rows.append(
            {
                "extension": "geo_kernel_crc_preflight",
                "role": "screened_not_run",
                "formal_uncertainty_method_complete": True,
                "formal_anchor": "conformal_risk_control",
                "localized_geo_method_applicability": (
                    "not_formally_established_for_segmentation_under_frozen_protocol"
                ),
                "spatial_localization_status": spatial_preflight["status"],
                "reason": spatial_preflight.get("reason", ""),
            }
        )
    calibration_confidence = np.asarray(
        [
            float(np.mean(np.maximum(probability[valid], 1.0 - probability[valid])))
            for probability, valid in zip(calibration_probs, calibration_valid)
        ]
    )
    test_confidence = np.asarray(
        [
            float(np.mean(np.maximum(probability[valid], 1.0 - probability[valid])))
            for probability, valid in zip(test_probs, test_valid)
        ]
    )
    test_risk = np.asarray([float(row["risk"]) for row in test_rows])
    for coverage in selective_coverages:
        selective_model = fit_selective_threshold(calibration_confidence, target_coverage=float(coverage))
        applied = apply_selective_threshold(
            selective_model, test_risk, test_confidence, sample_rows=test_rows
        )
        retained = [row for row in applied if row["accepted"]]
        selective_protocol = _derived_protocol(
            base,
            loss_name=base.loss_name,
            extension="selective_prediction",
            details={
                "target_coverage": coverage,
                "confidence_definition": "mean_valid_pixel_binary_certainty",
                "threshold_source": "validation_only",
                "calibration_sha256": calibration_sha256,
            },
        )
        slug = f"selective_{int(round(100 * coverage)):03d}"
        run_output = ensure_dir(output / slug)
        coverage_path, zero_groups, minimum_group_coverage = _write_selective_group_coverage(
            applied,
            run_output,
            group_columns=group_columns,
        )
        selective_artifacts = _audit_and_write(
            retained,
            run_output,
            protocol=selective_protocol,
            group_columns=group_columns,
            cluster_column=cluster,
            n_bootstrap=n_bootstrap,
            seed=seed,
            required_group_universe=_group_universe(test_rows, group_columns),
        )
        artifacts.update({f"{slug}_{key}": value for key, value in selective_artifacts.items()})
        artifacts[f"{slug}_group_coverage"] = coverage_path
        summary_rows.append(
            {
                "extension": slug,
                "target_coverage": coverage,
                "test_coverage": len(retained) / len(test_rows),
                "selective_risk": float(np.mean([row["risk"] for row in retained])),
                "minimum_group_coverage": minimum_group_coverage,
                "zero_accepted_groups": zero_groups,
                "selective_geobwer_identified": zero_groups == 0,
                "confidence_threshold": selective_model.confidence_threshold,
                "protocol_hash": selective_protocol.signature,
            }
        )
    summary = output / "uncertainty_summary.csv"
    write_csv(summary, summary_rows)
    artifacts["summary"] = summary
    return artifacts


def run_multiclass_spatial_upgrade(
    calibration_probabilities: str | Path,
    calibration_metadata_csv: str | Path,
    test_formal_dir: str | Path,
    output_dir: str | Path,
    *,
    protocol: str | Path | BWERProtocol,
    group_columns: Sequence[str],
    source_calibration_manifest: str | Path | None = None,
    conformal_methods: Sequence[str] = ("lac", "aps", "raps"),
    alpha: float = 0.10,
    n_bootstrap: int = 2000,
    seed: int = 42,
    spatial_conformal_config: SpatialConformalConfig | None = None,
) -> dict[str, Path]:
    """Upgrade existing full probabilities without rerunning a GeoFM.

    Coordinates are joined by calibration sample_id, written to a new
    calibration artifact, and the complete global-versus-local conformal audit
    is recomputed downstream. The source probability file is never modified.
    """

    output = ensure_dir(output_dir)
    source = Path(calibration_probabilities)
    metadata_rows = read_csv_rows(calibration_metadata_csv)
    coordinate_by_id: dict[str, tuple[float, float]] = {}
    for row in metadata_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        coordinates = _coordinates_from_rows([row])
        if not sample_id or coordinates is None or not np.all(np.isfinite(coordinates)):
            continue
        value = (float(coordinates[0, 0]), float(coordinates[0, 1]))
        previous = coordinate_by_id.get(sample_id)
        if previous is not None and previous != value:
            raise ExtensionAuditError(
                f"Conflicting calibration coordinates for sample_id={sample_id!r}."
            )
        coordinate_by_id[sample_id] = value
    with np.load(source, allow_pickle=False) as artifact:
        arrays = {name: np.asarray(artifact[name]) for name in artifact.files}
    required = {"sample_id", "split_role", "test_rows_used"}
    missing_fields = sorted(required - set(arrays))
    if missing_fields:
        raise ExtensionAuditError(
            "Source calibration NPZ is missing required fields: "
            + ", ".join(missing_fields)
        )
    if str(np.asarray(arrays["test_rows_used"]).reshape(-1)[0]).lower() not in {
        "false",
        "0",
    }:
        raise ExtensionAuditError("Source calibration NPZ declares test_rows_used=true.")
    sample_ids = arrays["sample_id"].astype(str)
    missing = [sample_id for sample_id in sample_ids.tolist() if sample_id not in coordinate_by_id]
    if missing:
        raise ExtensionAuditError(
            f"Calibration metadata is missing verified coordinates for {len(missing)} sample IDs."
        )
    arrays["latitude"] = np.asarray(
        [coordinate_by_id[sample_id][0] for sample_id in sample_ids.tolist()],
        dtype=np.float64,
    )
    arrays["longitude"] = np.asarray(
        [coordinate_by_id[sample_id][1] for sample_id in sample_ids.tolist()],
        dtype=np.float64,
    )
    enriched = output / "calibration_probabilities_with_coordinates.npz"
    np.savez_compressed(enriched, **arrays)
    manifest = output / "calibration_manifest_with_coordinates.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.multiclass_spatial_calibration_upgrade.v1",
                "split_role": str(np.asarray(arrays["split_role"]).reshape(-1)[0]),
                "test_rows_used": False,
                "probabilities_sha256": file_sha256(enriched),
                "source_calibration_sha256": file_sha256(source),
                "source_calibration_manifest_sha256": (
                    file_sha256(source_calibration_manifest)
                    if source_calibration_manifest is not None
                    else ""
                ),
                "coordinate_metadata_sha256": file_sha256(calibration_metadata_csv),
                "coordinate_join": "exact_sample_id",
                "sample_count": len(sample_ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    config = spatial_conformal_config or SpatialConformalConfig()
    artifacts = run_multiclass_uncertainty_suite(
        enriched,
        test_formal_dir,
        output / "uncertainty_extensions",
        protocol=protocol,
        group_columns=group_columns,
        calibration_manifest=manifest,
        conformal_methods=conformal_methods,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
        spatial_conformal_config=config,
    )
    return {
        "calibration_probabilities": enriched,
        "calibration_manifest": manifest,
        **artifacts,
    }


__all__ = [
    "ExtensionAuditError",
    "run_multiclass_uncertainty_suite",
    "run_multiclass_spatial_upgrade",
    "run_multilabel_uncertainty_suite",
    "run_segmentation_uncertainty_suite",
]
