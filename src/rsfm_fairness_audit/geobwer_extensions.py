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
)
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


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
    selective_coverages: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    crc_alpha: float = 0.10,
    n_bootstrap: int = 2000,
    seed: int = 42,
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
            "risk_target": crc_alpha,
            "test_mean_risk": float(np.mean([row["risk"] for row in rows])),
            "mean_prediction_set_fraction": float(np.mean([row["prediction_set_fraction"] for row in rows])),
            "probability_threshold": crc_model.probability_threshold,
            "protocol_hash": extension_protocol.signature,
            **crc_target_diagnostics,
        }
    ]
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


__all__ = [
    "ExtensionAuditError",
    "run_multiclass_uncertainty_suite",
    "run_multilabel_uncertainty_suite",
    "run_segmentation_uncertainty_suite",
]
