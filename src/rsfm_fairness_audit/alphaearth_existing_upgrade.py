from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.formal_outputs import file_sha256, write_multiclass_bundle
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.geobwer_extensions import run_multiclass_uncertainty_suite
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.persistent_cache import (
    hydrate_output,
    persist_output,
    validate_storage_contract,
)
from rsfm_fairness_audit.spatial_conformal import SpatialConformalConfig


AXES = (
    "country_iso3",
    "region",
    "worldcover_class_name",
    "country_class",
    "region_class",
    "income_group",
    "biome_or_ecoregion",
    "urban_rural_or_built_proxy",
)
STANDARDIZED_AXES = (
    "country_iso3",
    "region",
    "income_group",
    "biome_or_ecoregion",
    "urban_rural_or_built_proxy",
)
FORMAL_DW_SCOPES = {"test_only", "eval_calibration_test"}
DESCRIPTIVE_DW_SCOPES = {"all_split_descriptive"}


class AlphaEarthExistingUpgradeError(RuntimeError):
    """Raised when frozen AlphaEarth evidence cannot support a safe upgrade."""


@dataclass(frozen=True)
class AlphaEarthExistingUpgradeConfig:
    source_root: Path
    output_dir: Path
    persistent_output_dir: Path | None = None
    protocol_path: Path = Path("configs/geobwer/alphaearth.yaml")
    model_name: str = "alphaearth_hist_gradient_boosting"
    audit_bootstrap: int = 2000
    conformal_alpha: float = 0.10
    seed: int = 42
    spatial_conformal_config: SpatialConformalConfig = SpatialConformalConfig()


def _text(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value and value.lower() not in {"nan", "none", "null"}:
            return value
    return ""


def _float(row: Mapping[str, Any], *names: str) -> float:
    value = _text(row, *names)
    try:
        return float(value)
    except ValueError as exc:
        raise AlphaEarthExistingUpgradeError(
            f"Missing or invalid numeric field {names}: {value!r}"
        ) from exc


def _hash_records(records: Sequence[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                [str(value) for value in record],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AlphaEarthExistingUpgradeError(
            "A formal upgrade must run from a Git checkout so the code commit can be frozen."
        ) from exc


def _assert_distinct_roots(source: Path, live: Path, persistent: Path | None) -> None:
    roots = [source.resolve(), live.resolve()]
    if persistent is not None:
        roots.append(persistent.resolve())
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise AlphaEarthExistingUpgradeError(
                    f"Frozen source and output roots must be disjoint: {left} vs {right}"
                )


def _source_artifacts(source_root: Path) -> dict[str, Path]:
    source = source_root.resolve()
    if not source.is_dir():
        raise AlphaEarthExistingUpgradeError(
            f"AlphaEarth frozen source directory does not exist: {source}"
        )
    required = {
        "eval_predictions": source / "alphaearth_full_eval_predictions.csv",
        "metrics": source / "alphaearth_full_metrics.csv",
        "legacy_bwer": source / "alphaearth_full_bwer_summary.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise AlphaEarthExistingUpgradeError(
            "The selected directory is not a formal AlphaEarth result root; "
            f"missing content evidence: {', '.join(missing)}"
        )
    optional = {
        "all_split_predictions": source / "alphaearth_full_all_split_predictions.csv",
        "dynamic_world_aligned": source / "alphaearth_full_dw_aligned.csv",
        "dynamic_world_diagnostic": source / "alphaearth_dynamic_world_agreement.csv",
        "report": source / "alphaearth_full_audit_report.md",
    }
    return {**required, **{key: path for key, path in optional.items() if path.is_file()}}


def _read_eval_predictions(
    path: Path,
) -> tuple[list[dict[str, str]], tuple[str, ...], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        probability_columns = tuple(field for field in fields if field.startswith("prob_"))
        rows = [dict(row) for row in reader]
    if len(probability_columns) != 11:
        raise AlphaEarthExistingUpgradeError(
            f"Expected the frozen 11-class probability vector, found {len(probability_columns)} columns."
        )
    classes = tuple(column[5:] for column in probability_columns)
    if len(set(classes)) != 11:
        raise AlphaEarthExistingUpgradeError("Probability class names are not unique.")
    if not rows:
        raise AlphaEarthExistingUpgradeError("Formal eval prediction table is empty.")
    ids = [_text(row, "sample_id") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise AlphaEarthExistingUpgradeError(
            "Formal eval sample_id values must be non-empty and unique."
        )
    probabilities = np.asarray(
        [[float(row[column]) for column in probability_columns] for row in rows],
        dtype=np.float32,
    )
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4)
    ):
        raise AlphaEarthExistingUpgradeError(
            "Frozen probabilities must be finite, in [0,1], and sum to one."
        )
    labels = [_text(row, "label", "worldcover_label") for row in rows]
    class_index = {name: index for index, name in enumerate(classes)}
    missing_labels = sorted(set(labels) - set(classes))
    if missing_labels:
        raise AlphaEarthExistingUpgradeError(
            f"WorldCover labels are absent from the prob_* class mapping: {missing_labels}"
        )
    targets = np.asarray([class_index[label] for label in labels], dtype=np.int64)
    argmax_labels = [classes[index] for index in np.argmax(probabilities, axis=1)]
    recorded_predictions = [_text(row, "prediction") for row in rows]
    mismatches = sum(
        bool(recorded) and recorded != inferred
        for recorded, inferred in zip(recorded_predictions, argmax_labels)
    )
    if mismatches:
        raise AlphaEarthExistingUpgradeError(
            f"Probability-column order conflicts with {mismatches} recorded predictions."
        )
    for index, row in enumerate(rows):
        inferred_risk = float(argmax_labels[index] != labels[index])
        recorded_risk = _text(row, "risk")
        recorded_correct = _text(row, "correct")
        if recorded_risk and not np.isclose(float(recorded_risk), inferred_risk):
            raise AlphaEarthExistingUpgradeError(
                f"Recorded risk conflicts with probabilities at sample_id={ids[index]!r}."
            )
        if recorded_correct and not np.isclose(
            float(recorded_correct), 1.0 - inferred_risk
        ):
            raise AlphaEarthExistingUpgradeError(
                f"Recorded correctness conflicts with probabilities at sample_id={ids[index]!r}."
            )
    return rows, classes, probabilities, targets


def _validate_result_lineage(
    rows: Sequence[Mapping[str, Any]],
    metrics_path: Path,
    *,
    calibration_count: int,
    test_count: int,
) -> dict[str, Any]:
    metrics = read_csv_rows(metrics_path)
    if len(metrics) != 1:
        raise AlphaEarthExistingUpgradeError(
            "Formal AlphaEarth metrics must contain exactly one frozen model result."
        )
    metric = metrics[0]
    for field in ("model", "dataset", "split_protocol"):
        prediction_values = sorted(
            {_text(row, field) for row in rows if _text(row, field)}
        )
        metric_value = _text(metric, field)
        if len(prediction_values) > 1:
            raise AlphaEarthExistingUpgradeError(
                f"Frozen eval predictions mix multiple {field} values: {prediction_values}"
            )
        if prediction_values and metric_value and prediction_values[0] != metric_value:
            raise AlphaEarthExistingUpgradeError(
                f"Frozen metrics and eval predictions disagree on {field}."
            )
    for field, expected in (
        ("n_calibration", calibration_count),
        ("n_test", test_count),
    ):
        value = _text(metric, field)
        if value and int(float(value)) != expected:
            raise AlphaEarthExistingUpgradeError(
                f"Frozen metrics {field}={value} disagrees with eval rows={expected}."
            )
    protocols = sorted(
        {_text(row, "split_protocol") for row in rows if _text(row, "split_protocol")}
    )
    if protocols and not all("spatial" in value.lower() for value in protocols):
        raise AlphaEarthExistingUpgradeError(
            f"AlphaEarth formal lineage is not a spatial split: {protocols}"
        )
    return {
        "metrics_sha256": file_sha256(metrics_path),
        "model": _text(metric, "model")
        or next((_text(row, "model") for row in rows if _text(row, "model")), ""),
        "dataset": _text(metric, "dataset")
        or next((_text(row, "dataset") for row in rows if _text(row, "dataset")), ""),
        "split_protocol": _text(metric, "split_protocol")
        or (protocols[0] if protocols else "strict_spatial_block_hash"),
        "calibration_count_matches_metrics": True,
        "test_count_matches_metrics": True,
        "lineage_connection": (
            "eval rows are the frozen runner's calibration+test prediction artifact; "
            "probabilities reproduce prediction/risk and counts agree with frozen metrics"
        ),
    }


def _validate_splits(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    roles = np.asarray([_text(row, "split").lower() for row in rows], dtype=str)
    calibration_mask = np.isin(roles, ("calibration", "validation", "val"))
    test_mask = roles == "test"
    if not np.any(calibration_mask) or not np.any(test_mask):
        raise AlphaEarthExistingUpgradeError(
            "Formal eval predictions must contain calibration and test rows."
        )
    if np.any(~(calibration_mask | test_mask)):
        unexpected = sorted(set(roles[~(calibration_mask | test_mask)].tolist()))
        raise AlphaEarthExistingUpgradeError(
            f"Formal eval predictions contain non-eval splits: {unexpected}"
        )
    calibration_ids = {
        _text(rows[index], "sample_id") for index in np.flatnonzero(calibration_mask)
    }
    test_ids = {_text(rows[index], "sample_id") for index in np.flatnonzero(test_mask)}
    if calibration_ids & test_ids:
        raise AlphaEarthExistingUpgradeError("Calibration/test sample leakage detected.")
    calibration_blocks = {
        _text(rows[index], "spatial_block_id")
        for index in np.flatnonzero(calibration_mask)
    }
    test_blocks = {
        _text(rows[index], "spatial_block_id") for index in np.flatnonzero(test_mask)
    }
    if "" in calibration_blocks or "" in test_blocks:
        raise AlphaEarthExistingUpgradeError(
            "Frozen strict spatial_block_id is required on every eval row."
        )
    overlap = calibration_blocks & test_blocks
    if overlap:
        raise AlphaEarthExistingUpgradeError(
            f"Spatial-block leakage detected across calibration/test: {len(overlap)} blocks."
        )
    split_hash = _hash_records(
        sorted(
            (
                _text(row, "sample_id"),
                "calibration" if calibration_mask[index] else "test",
                _text(row, "spatial_block_id"),
            )
            for index, row in enumerate(rows)
        )
    )
    return calibration_mask, test_mask, {
        "calibration_rows": int(np.sum(calibration_mask)),
        "test_rows": int(np.sum(test_mask)),
        "calibration_test_sample_overlap": 0,
        "calibration_test_spatial_block_overlap": 0,
        "split_assignment_hash": split_hash,
    }


def _enrich_axes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        country = _text(row, "country_iso3", "country")
        region = _text(row, "region")
        class_name = _text(
            row, "worldcover_class_name", "class_label", "label", "worldcover_label"
        )
        row.update(
            {
                "country_iso3": country,
                "country": country,
                "region": region,
                "worldcover_class_name": class_name,
                "country_class": _text(row, "country_class")
                or (f"{country}|{class_name}" if country and class_name else ""),
                "region_class": _text(row, "region_class")
                or (f"{region}|{class_name}" if region and class_name else ""),
                "independent_unit_id": _text(row, "sample_id"),
                "latitude": _float(row, "lat", "latitude"),
                "longitude": _float(row, "lon", "longitude"),
            }
        )
        output.append(row)
    return output


def _axis_support(
    rows: Sequence[Mapping[str, Any]], protocol: BWERProtocol
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    output: list[dict[str, Any]] = []
    complete_axes: list[str] = []
    total = len(rows)
    for axis in AXES:
        valid_rows = [row for row in rows if _text(row, axis)]
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in valid_rows:
            grouped.setdefault(_text(row, axis), []).append(row)
        supported = {
            group
            for group, items in grouped.items()
            if len(items) >= protocol.min_units_per_slice
            and len({_text(item, "spatial_block_id") for item in items})
            >= protocol.min_clusters_per_slice
        }
        sample_coverage = len(valid_rows) / total if total else 0.0
        deployment_coverage = len(supported) / len(grouped) if grouped else 0.0
        output.append(
            {
                "axis": axis,
                "fixed_universe_groups": len(grouped),
                "supported_universe_groups": len(supported),
                "sample_metadata_coverage": sample_coverage,
                "equal_slice_deployment_mass_coverage": deployment_coverage,
                "excluded_deployment_mass": 1.0 - deployment_coverage,
                "missing_sample_count": total - len(valid_rows),
                "min_units_per_slice": protocol.min_units_per_slice,
                "min_clusters_per_slice": protocol.min_clusters_per_slice,
                "axis_status": (
                    "formal_complete"
                    if len(valid_rows) == total
                    else "not_identified_missing_group_metadata"
                ),
            }
        )
        if len(valid_rows) == total:
            complete_axes.append(axis)
    return output, tuple(complete_axes)


def _retag(rows: Sequence[Mapping[str, Any]], protocol: BWERProtocol) -> list[dict[str, Any]]:
    return [
        {**dict(row), "protocol_hash": protocol.signature, "metric_version": protocol.metric_version}
        for row in rows
    ]


def _joint_risk_card(
    raw_rows: Sequence[Mapping[str, Any]],
    standardized_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    support = {str(row["axis"]): row for row in support_rows}
    output: list[dict[str, Any]] = []
    for family, rows in (("raw", raw_rows), ("class_standardized", standardized_rows)):
        for source in rows:
            row = dict(source)
            axis = str(row.get("axis", ""))
            info = support.get(axis, {})
            beta = float(row.get("beta", 0.10) or 0.10)
            tail_effective = float(row.get("tail_effective_groups", 0.0) or 0.0)
            row.update(
                {
                    "risk_family": family,
                    "beta_effective_tail_slices": tail_effective,
                    "tail_saturation": bool(
                        tail_effective <= 1.0 + 1e-12
                        or float(row.get("max_tail_atom_share", 0.0) or 0.0)
                        >= 1.0 - 1e-12
                    ),
                    "fixed_universe_groups": info.get("fixed_universe_groups", ""),
                    "supported_universe_groups": info.get(
                        "supported_universe_groups", ""
                    ),
                    "sample_metadata_coverage": info.get(
                        "sample_metadata_coverage", ""
                    ),
                    "equal_slice_deployment_mass_coverage": info.get(
                        "equal_slice_deployment_mass_coverage", ""
                    ),
                    "partial_identification_lower": (
                        row.get("bwer", "")
                        if float(info.get("excluded_deployment_mass", 1.0) or 0.0)
                        == 0.0
                        else 0.0
                    ),
                    "partial_identification_upper": (
                        row.get("bwer", "")
                        if float(info.get("excluded_deployment_mass", 1.0) or 0.0)
                        == 0.0
                        else 1.0 - beta
                    ),
                    "partial_identification_scope": (
                        "point_identified_fixed_universe"
                        if float(info.get("excluded_deployment_mass", 1.0) or 0.0)
                        == 0.0
                        else "conservative_fixed_universe_bound; supported point estimate is conditional"
                    ),
                }
            )
            output.append(row)
    audited = {(str(row.get("axis")), str(row.get("risk_family"))) for row in output}
    for axis, info in support.items():
        if (axis, "raw") not in audited:
            output.append(
                {
                    "axis": axis,
                    "risk_family": "raw",
                    "validity": info.get("axis_status"),
                    "fixed_universe_groups": info.get("fixed_universe_groups"),
                    "supported_universe_groups": info.get("supported_universe_groups"),
                    "sample_metadata_coverage": info.get("sample_metadata_coverage"),
                    "equal_slice_deployment_mass_coverage": info.get(
                        "equal_slice_deployment_mass_coverage"
                    ),
                    "message": "Axis was not formally audited because group metadata are incomplete.",
                }
            )
    return output


def _validate_dynamic_world_scopes(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        scope = _text(row, "scope")
        claim_role = _text(row, "claim_role", "claim_kind")
        if scope in DESCRIPTIVE_DW_SCOPES and claim_role not in {
            "",
            "descriptive_background_only",
        }:
            raise AlphaEarthExistingUpgradeError(
                "all_split_descriptive Dynamic World rows cannot be promoted to formal claims."
            )
        if scope and scope not in FORMAL_DW_SCOPES | DESCRIPTIVE_DW_SCOPES:
            raise AlphaEarthExistingUpgradeError(
                f"Unregistered Dynamic World diagnostic scope: {scope}"
            )


def _dynamic_world_rows(source_root: Path) -> list[dict[str, Any]]:
    try:
        from scripts.analysis.build_alphaearth_final_evidence_hardening_v2 import (
            build_dynamic_world_diagnostic,
        )
    except ImportError as exc:
        raise AlphaEarthExistingUpgradeError(
            "Dynamic World diagnostic builder is unavailable in this checkout."
        ) from exc
    rows, _ = build_dynamic_world_diagnostic(source_root)
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        scope = _text(row, "scope")
        if scope in FORMAL_DW_SCOPES:
            row["claim_role"] = "map_product_agreement_or_ambiguity_diagnostic"
            row["paper_eligible"] = True
        elif scope in DESCRIPTIVE_DW_SCOPES:
            row["claim_role"] = "descriptive_background_only"
            row["paper_eligible"] = False
        else:
            row["claim_role"] = "availability_or_validation_record"
            row["paper_eligible"] = False
        row["reference_semantics"] = (
            "WorldCover is a reference map product, not perfect human ground truth."
        )
        output.append(row)
    _validate_dynamic_world_scopes(output)
    return output


def _completion_signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_completion(root: Path, signature: str) -> bool:
    if not root.exists():
        return False
    contract_path = root / "completion_contract.json"
    if not contract_path.is_file():
        if any(root.iterdir()):
            raise AlphaEarthExistingUpgradeError(
                f"Non-empty partial output lacks completion contract: {root}"
            )
        return False
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("completion_signature") != signature:
        raise AlphaEarthExistingUpgradeError(
            f"AlphaEarth completion signature drift: {root}"
        )
    for relative, expected in dict(payload.get("artifacts", {})).items():
        path = root / str(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise AlphaEarthExistingUpgradeError(
                f"AlphaEarth completion artifact mismatch: {path}"
            )
    return True


def _write_completion(
    root: Path, signature_payload: Mapping[str, Any]
) -> Path:
    artifacts = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "completion_contract.json"
    }
    path = root / "completion_contract.json"
    path.write_text(
        json.dumps(
            {
                "schema": "geobwer.alphaearth_existing_upgrade_completion.v1",
                "formal_evidence": True,
                "completion_signature": _completion_signature(signature_payload),
                "signature_payload": dict(signature_payload),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def run_alphaearth_existing_upgrade(
    config: AlphaEarthExistingUpgradeConfig,
) -> dict[str, Path]:
    validate_storage_contract(config.output_dir, config.persistent_output_dir)
    _assert_distinct_roots(
        config.source_root, config.output_dir, config.persistent_output_dir
    )
    sources = _source_artifacts(config.source_root)
    base_protocol = BWERProtocol.from_mapping(load_yaml(config.protocol_path))
    metadata = dict(base_protocol.metadata)
    metadata.update(
        {
            "upgrade_source": "frozen_formal_eval_predictions",
            "spatial_block_source": "frozen_strict_spatial_block_hash",
            "reference_semantics": "reference_map_agreement_not_perfect_ground_truth",
        }
    )
    protocol = replace(base_protocol, metadata=tuple(sorted(metadata.items())))
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    code_commit = _git_commit()
    signature_payload = {
        "schema": "geobwer.alphaearth_existing_upgrade_signature.v1",
        "code_commit": code_commit,
        "protocol_hash": protocol.signature,
        "source_hashes": source_hashes,
        "audit_bootstrap": config.audit_bootstrap,
        "conformal_alpha": config.conformal_alpha,
        "seed": config.seed,
        "spatial_conformal_config": asdict(config.spatial_conformal_config),
        "axes": list(AXES),
    }
    signature = _completion_signature(signature_payload)
    hydrate_output(config.output_dir, config.persistent_output_dir)
    output = config.output_dir
    if _validate_completion(output, signature):
        print("[alphaearth:existing-upgrade] verified completion contract; skipping", flush=True)
        return {
            "completion_contract": output / "completion_contract.json",
            "postprocess_manifest": output / "postprocess_manifest.json",
        }
    ensure_dir(output)

    print("[alphaearth:existing-upgrade] validating frozen eval probabilities", flush=True)
    rows, classes, probabilities, targets = _read_eval_predictions(
        sources["eval_predictions"]
    )
    calibration_mask, test_mask, split_evidence = _validate_splits(rows)
    calibration_rows = _enrich_axes(
        [rows[index] for index in np.flatnonzero(calibration_mask)]
    )
    test_rows = _enrich_axes([rows[index] for index in np.flatnonzero(test_mask)])
    calibration_probabilities = probabilities[calibration_mask]
    calibration_targets = targets[calibration_mask]
    test_probabilities = probabilities[test_mask]
    test_targets = targets[test_mask]
    result_lineage = _validate_result_lineage(
        rows,
        sources["metrics"],
        calibration_count=len(calibration_rows),
        test_count=len(test_rows),
    )

    if any(output.iterdir()):
        raise AlphaEarthExistingUpgradeError(
            f"Output became non-empty before formal execution: {output}"
        )

    formal_bundle = write_multiclass_bundle(
        output / "formal_outputs",
        sample_rows=test_rows,
        probabilities=test_probabilities,
        targets=test_targets,
        class_names=classes,
        dataset="alphaearth_worldcover_full",
        model=config.model_name,
        split="test",
        protocol=protocol,
        model_lineage={
            "model": "HistGradientBoostingClassifier",
            "representation": "Google_Satellite_Embedding_V1_Annual_AlphaEarth",
            "adaptation_protocol": "frozen_existing_probabilities_no_refit",
            "prediction_source_sha256": source_hashes["eval_predictions"],
        },
        dataset_lineage={
            "dataset": "AlphaEarth_embeddings_x_ESA_WorldCover",
            "split": "strict_spatial_block_hash",
            "split_assignment_hash": split_evidence["split_assignment_hash"],
            "reference_semantics": "map_product_agreement_not_human_ground_truth",
        },
        split_role="evaluation",
    )
    calibration_path = output / "calibration_probabilities.npz"
    np.savez_compressed(
        calibration_path,
        probabilities=calibration_probabilities,
        targets=calibration_targets,
        class_names=np.asarray(classes, dtype=str),
        sample_id=np.asarray([_text(row, "sample_id") for row in calibration_rows]),
        latitude=np.asarray([_float(row, "latitude") for row in calibration_rows]),
        longitude=np.asarray([_float(row, "longitude") for row in calibration_rows]),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    calibration_manifest = output / "calibration_manifest.json"
    calibration_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.alphaearth_existing_calibration.v1",
                "split_role": "calibration",
                "test_rows_used": False,
                "sample_count": len(calibration_rows),
                "probabilities_sha256": file_sha256(calibration_path),
                "source_eval_predictions_sha256": source_hashes["eval_predictions"],
                "assignment_hash": _hash_records(
                    [
                        (
                            _text(row, "sample_id"),
                            int(target),
                            _text(row, "spatial_block_id"),
                        )
                        for row, target in zip(calibration_rows, calibration_targets)
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    formal_rows = read_csv_rows(formal_bundle.audit_table)
    support_rows, complete_axes = _axis_support(formal_rows, protocol)
    write_csv(output / "support_coverage.csv", support_rows)
    if len(complete_axes) < 1:
        raise AlphaEarthExistingUpgradeError(
            "No preregistered AlphaEarth axis has complete group metadata."
        )
    print(
        f"[alphaearth:existing-upgrade] GeoBWER axes={','.join(complete_axes)}",
        flush=True,
    )
    raw = audit_rows(
        formal_rows,
        group_columns=complete_axes,
        protocol=protocol,
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seed,
    )
    raw_artifacts = raw.to_report(output / "geobwer_raw")
    standardized_axes = tuple(axis for axis in STANDARDIZED_AXES if axis in complete_axes)
    standardized_rows: list[dict[str, Any]] = []
    standardized_artifacts: dict[str, Path] = {}
    if standardized_axes:
        standardized = audit_rows(
            formal_rows,
            group_columns=standardized_axes,
            protocol=protocol,
            balance_column="worldcover_class_name",
            formal=True,
            require_probabilities=True,
            n_bootstrap=config.audit_bootstrap,
            seed=config.seed,
        )
        standardized_artifacts = standardized.to_report(
            output / "geobwer_class_standardized"
        )
        standardized_rows = read_csv_rows(standardized_artifacts["summary"])
    joint_rows = _joint_risk_card(
        read_csv_rows(raw_artifacts["summary"]),
        standardized_rows,
        support_rows,
    )
    write_csv(output / "joint_risk_card.csv", joint_rows)

    print("[alphaearth:existing-upgrade] running uncertainty suite", flush=True)
    uncertainty = run_multiclass_uncertainty_suite(
        calibration_path,
        formal_bundle.output_dir,
        output / "uncertainty_extensions",
        protocol=protocol,
        group_columns=complete_axes,
        calibration_manifest=calibration_manifest,
        alpha=config.conformal_alpha,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seed,
        spatial_conformal_config=config.spatial_conformal_config,
    )
    uncertainty_summary = read_csv_rows(uncertainty["summary"])
    conformal_rows = [
        row
        for row in uncertainty_summary
        if str(row.get("extension", "")).startswith(
            ("conformal_", "geo_kernel_conformal_")
        )
    ]
    write_csv(output / "conformal_global_vs_spatial.csv", conformal_rows)
    selective_rows = [
        {
            **row,
            "identifiability_interpretation": (
                "identified_on_all_preregistered_groups"
                if str(row.get("selective_geobwer_identified", "")).lower()
                in {"true", "1"}
                else "not_identified_due_to_zero_accepted_slice"
            ),
        }
        for row in uncertainty_summary
        if str(row.get("extension", "")).startswith("selective_")
    ]
    write_csv(output / "selective_identifiability.csv", selective_rows)

    dynamic_rows = _dynamic_world_rows(config.source_root)
    write_csv(output / "dynamic_world_scoped_diagnostics.csv", dynamic_rows)
    assignment_hash = _hash_records(
        sorted(
            (
                _text(row, "sample_id"),
                _text(row, "split"),
                _text(row, "spatial_block_id"),
                _text(row, "label", "worldcover_label"),
            )
            for row in rows
        )
    )
    probability_bundle_hash = _array_hash(
        probabilities, targets, np.asarray(classes, dtype="U")
    )
    interpretation = output / "scientific_interpretation_report.md"
    spatial_rows = [
        row
        for row in conformal_rows
        if str(row.get("extension", "")).startswith("geo_kernel_conformal_")
    ]
    tradeoff_lines: list[str] = []
    by_extension = {
        str(row.get("extension", "")): row for row in conformal_rows
    }
    for method in ("lac", "aps", "raps"):
        global_row = by_extension.get(f"conformal_{method}")
        spatial_row = by_extension.get(f"geo_kernel_conformal_{method}")
        if global_row is None or spatial_row is None:
            continue
        coverage_delta = float(spatial_row["test_coverage"]) - float(
            global_row["test_coverage"]
        )
        set_delta = float(spatial_row["mean_set_size"]) - float(
            global_row["mean_set_size"]
        )
        tradeoff_lines.append(
            f"- {method.upper()}: spatial-minus-global coverage={coverage_delta:+.6f}; "
            f"mean-set-size={set_delta:+.6f}. "
            + (
                "Coverage improvement is accompanied by larger sets."
                if coverage_delta > 0.0 and set_delta > 0.0
                else "No simple coverage-for-larger-sets pattern was observed."
            )
        )
    if not tradeoff_lines:
        tradeoff_lines.append(
            "- The spatial comparator did not pass its frozen support preflight; only the ordinary marginal anchor is interpretable."
        )
    interpretation.write_text(
        "\n".join(
            [
                "# AlphaEarth GeoBWER 1.1 existing-output upgrade",
                "",
                "This audit measures agreement with ESA WorldCover. WorldCover is a reference-map product, not perfect ground truth.",
                "",
                "Ordinary split conformal LAC/APS/RAPS is the formal marginal-coverage anchor. The geographic-kernel result is an empirical localization comparator, not a finite-sample pointwise coverage guarantee.",
                "",
                f"- Formal test rows: {len(test_rows)}",
                f"- Calibration rows: {len(calibration_rows)}; test rows were not used for calibration.",
                f"- Formally complete axes: {', '.join(complete_axes)}",
                f"- Spatial comparator rows reported: {len(spatial_rows)}",
                "- Compare coverage jointly with mean set size and set-size fraction; increased coverage obtained only by larger sets is an efficiency trade-off, not free improvement.",
                "- Dynamic World is used only for map-product agreement and ambiguity diagnostics. all_split_descriptive rows are background-only.",
                "- Axes with incomplete metadata or inadequate support remain exploratory or partially identified; the runner does not lower thresholds after inspecting test outcomes.",
                "",
                "## Global-versus-spatial efficiency trade-off",
                "",
                *tradeoff_lines,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_inventory = output / "source_inventory.json"
    source_inventory.write_text(
        json.dumps(
            {
                "schema": "geobwer.alphaearth_source_inventory.v1",
                "selection_basis": (
                    "formal eval probability table + formal metrics + legacy BWER evidence; "
                    "raw GEE shard roots are not accepted as result roots"
                ),
                "source_root": str(config.source_root.resolve()),
                "artifacts": {
                    name: {"path": str(path.resolve()), "sha256": source_hashes[name]}
                    for name, path in sources.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = output / "postprocess_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.alphaearth_existing_upgrade.v1",
                "formal_evidence": True,
                "code_commit": code_commit,
                "metric_version": protocol.metric_version,
                "protocol": protocol.to_dict(),
                "protocol_hash": protocol.signature,
                "source_hashes": source_hashes,
                "split_evidence": split_evidence,
                "result_lineage": result_lineage,
                "assignment_hash": assignment_hash,
                "probability_bundle_hash": probability_bundle_hash,
                "class_names": list(classes),
                "test_rows_used_for_calibration": False,
                "audit_bootstrap": config.audit_bootstrap,
                "spatial_conformal_config": asdict(config.spatial_conformal_config),
                "complete_axes": list(complete_axes),
                "source_artifacts_modified": False,
                "outputs": {
                    "joint_risk_card": "joint_risk_card.csv",
                    "support_coverage": "support_coverage.csv",
                    "conformal_global_vs_spatial": "conformal_global_vs_spatial.csv",
                    "selective_identifiability": "selective_identifiability.csv",
                    "dynamic_world_scoped_diagnostics": "dynamic_world_scoped_diagnostics.csv",
                    "scientific_interpretation_report": "scientific_interpretation_report.md",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_completion(output, signature_payload)
    persist_output(
        output,
        config.persistent_output_dir,
        label="alphaearth-existing-geobwer-upgrade-complete",
    )
    return {
        "joint_risk_card": output / "joint_risk_card.csv",
        "support_coverage": output / "support_coverage.csv",
        "conformal_global_vs_spatial": output / "conformal_global_vs_spatial.csv",
        "selective_identifiability": output / "selective_identifiability.csv",
        "dynamic_world_scoped_diagnostics": output
        / "dynamic_world_scoped_diagnostics.csv",
        "scientific_interpretation_report": interpretation,
        "postprocess_manifest": manifest,
        "completion_contract": output / "completion_contract.json",
    }


__all__ = [
    "AXES",
    "AlphaEarthExistingUpgradeConfig",
    "AlphaEarthExistingUpgradeError",
    "run_alphaearth_existing_upgrade",
]
