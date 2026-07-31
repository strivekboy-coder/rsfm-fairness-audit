from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.sen1_prithvi_artifact_audit import (
    _artifact,
    _audit_export,
    _json,
    _sequence_hash,
)


class Sen1PrithviV0432ArtifactAuditError(RuntimeError):
    """Raised when the frozen v0.4.32 Prithvi evidence is not auditable."""


@dataclass(frozen=True)
class _Expectation:
    version: str
    commit: str
    train_count: int
    validation_count: int
    test_count: int
    bolivia_count: int
    non_bolivia_event_count: int


FORMAL_EXPECTATION = _Expectation(
    version="0.4.32",
    commit="f0d2e9e40cf4d392733a656021eccd5fa1d848fe",
    train_count=252,
    validation_count=89,
    test_count=90,
    bolivia_count=15,
    non_bolivia_event_count=10,
)


def _fail(message: str) -> None:
    raise Sen1PrithviV0432ArtifactAuditError(message)


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _path_suffix_matches(recorded: Any, expected: Path, root: Path) -> bool:
    recorded_parts = tuple(
        part.lower()
        for part in str(recorded).replace("\\", "/").split("/")
        if part
    )
    expected_parts = tuple(
        part.lower() for part in expected.relative_to(root).parts
    )
    return len(recorded_parts) >= len(expected_parts) and (
        recorded_parts[-len(expected_parts) :] == expected_parts
    )


def _validate_gate(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    expectation: _Expectation,
) -> tuple[dict[str, Any], Path, set[str], set[str]]:
    gate_path = root / "pre_model_prepared_mask_gate.json"
    gate = _json(gate_path)
    binding = manifest.get("pre_model_prepared_mask_gate", {})
    if (
        not _path_suffix_matches(binding.get("path"), gate_path, root)
        or binding.get("sha256") != file_sha256(gate_path)
        or binding.get("schema")
        != "geobwer.sen1floods11.prithvi_prepared_mask_gate.v1"
        or binding.get("status") != "pass"
        or binding.get("model_loaded_by_gate") is not False
        or gate.get("schema")
        != "geobwer.sen1floods11.prithvi_prepared_mask_gate.v1"
        or gate.get("status") != "pass"
        or gate.get("model_loaded_by_gate") is not False
        or gate.get("blocking_errors") != []
        or int(gate.get("combined_sample_count", -1))
        != expectation.train_count
        + expectation.validation_count
        + expectation.test_count
        + expectation.bolivia_count
        or int(gate.get("split_overlap_count", -1)) != 0
    ):
        _fail("The pre-model prepared-mask gate or its manifest binding is invalid.")
    contract = gate.get("mask_contract", {})
    if (
        contract.get("shape") != [224, 224]
        or contract.get("finite") is not True
        or contract.get("integer") is not True
        or contract.get("allowed_values") != [-1, 0, 1]
        or contract.get("npz_required_key") != "mask"
    ):
        _fail("The prepared-mask gate does not bind the frozen mask semantics.")
    core = gate.get("core", {})
    bolivia = gate.get("bolivia", {})
    core_records = core.get("records", [])
    bolivia_records = bolivia.get("records", [])
    expected_core = (
        expectation.train_count
        + expectation.validation_count
        + expectation.test_count
    )
    if (
        int(core.get("sample_count", -1)) != expected_core
        or len(core_records) != expected_core
        or int(bolivia.get("sample_count", -1)) != expectation.bolivia_count
        or len(bolivia_records) != expectation.bolivia_count
        or bolivia.get("events") != ["Bolivia"]
        or "Bolivia" in set(core.get("events", []))
        or len(set(core.get("events", []))) != expectation.non_bolivia_event_count
    ):
        _fail("The mask gate does not contain the frozen 431+15 event universe.")
    core_ids = {str(row.get("sample_id", "")).strip() for row in core_records}
    bolivia_ids = {
        str(row.get("sample_id", "")).strip() for row in bolivia_records
    }
    if (
        "" in core_ids
        or "" in bolivia_ids
        or len(core_ids) != expected_core
        or len(bolivia_ids) != expectation.bolivia_count
        or core_ids & bolivia_ids
    ):
        _fail("The mask-gate physical sample identities are invalid or overlapping.")
    return gate, gate_path, core_ids, bolivia_ids


def _validate_manifest(
    root: Path,
    *,
    expectation: _Expectation,
) -> tuple[dict[str, Any], Path, Path, dict[str, str]]:
    manifest_path = root / "campaign_manifest.json"
    manifest = _json(manifest_path)
    if (
        manifest.get("schema")
        != "geobwer.sen1floods11.prithvi_tl_probability_migration.v3"
        or manifest.get("formal_evidence") is not True
        or manifest.get("split_protocol")
        != "official_252_89_90_plus_15_bolivia_holdout"
        or int(manifest.get("train_count", -1)) != expectation.train_count
        or int(manifest.get("validation_count", -1))
        != expectation.validation_count
        or int(manifest.get("test_count", -1)) != expectation.test_count
        or int(manifest.get("bolivia_holdout_count", -1))
        != expectation.bolivia_count
        or int(manifest.get("combined_evaluation_count", -1))
        != expectation.test_count + expectation.bolivia_count
        or manifest.get("no_training_or_calibration_leakage") is not True
        or manifest.get("bolivia_holdout_used_for_training_or_calibration")
        is not False
    ):
        _fail("The Prithvi campaign manifest violates the frozen formal contract.")
    config = manifest.get("config", {})
    if (
        str(config.get("diagnostic_max_samples")) not in {"None", ""}
        or str(config.get("device", "")).lower() != "cuda"
    ):
        _fail("The Prithvi output is not a formal CUDA campaign.")
    device = manifest.get("device_contract", {})
    cuda_fields = {
        "resolved_device": str(device.get("resolved_device", "")),
        "model_parameter_device": str(device.get("model_parameter_device", "")),
        "model_input_device": str(device.get("model_input_device", "")),
    }
    if (
        device.get("status") != "pass"
        or device.get("strict_no_cpu_fallback") is not True
        or not all(value.startswith("cuda") for value in cuda_fields.values())
    ):
        _fail(f"CUDA device evidence is incomplete: {cuda_fields}")
    checkpoint = Path(str(manifest.get("checkpoint_path", "")))
    if (
        not checkpoint.is_file()
        or not manifest.get("checkpoint_sha256")
        or file_sha256(checkpoint) != manifest.get("checkpoint_sha256")
    ):
        _fail("The checkpoint is missing or its SHA-256 does not match the manifest.")
    return manifest, manifest_path, checkpoint, cuda_fields


def _audit_engine(root: Path, *, expectation: _Expectation) -> dict[str, Any]:
    manifest, manifest_path, checkpoint, cuda_fields = _validate_manifest(
        root, expectation=expectation
    )
    gate, gate_path, core_ids, bolivia_ids = _validate_gate(
        root, manifest, expectation=expectation
    )
    counts = {
        "validation": expectation.validation_count,
        "test": expectation.test_count,
        "bolivia_holdout": expectation.bolivia_count,
    }
    reports: dict[str, Any] = {}
    artifacts = [
        _artifact(manifest_path, base=root),
        _artifact(gate_path, base=root),
        _artifact(checkpoint),
    ]
    for split, expected_count in counts.items():
        export = root / "probabilities" / split
        recorded = manifest.get("probability_exports", {}).get(split)
        if not _path_suffix_matches(recorded, export, root):
            _fail(f"Manifest does not bind the canonical {split} export.")
        report = _audit_export(
            export,
            expected_count=expected_count,
            split=split,
            artifact_base=root,
        )
        runtime = manifest.get("split_runtime_validation", {}).get(split, {})
        if (
            int(runtime.get("row_count", -1)) != expected_count
            or runtime.get("full_probability_layout") != "[B,2,H,W]"
            or runtime.get("probabilities_finite") is not True
            or runtime.get("probabilities_in_unit_interval") is not True
            or runtime.get("target_shape_matches_probability_shape") is not True
            or float(runtime.get("maximum_probability_sum_error", 1.0)) > 1e-5
        ):
            _fail(f"Runtime validation is invalid for {split}.")
        artifacts.extend(report.pop("artifacts"))
        report.pop("target_rows")
        reports[split] = report

    validation_ids = set(reports["validation"]["canonical_sample_ids"])
    test_ids = set(reports["test"]["canonical_sample_ids"])
    observed_bolivia_ids = set(
        reports["bolivia_holdout"]["canonical_sample_ids"]
    )
    train_ids = core_ids - validation_ids - test_ids
    if (
        not validation_ids.issubset(core_ids)
        or not test_ids.issubset(core_ids)
        or validation_ids & test_ids
        or observed_bolivia_ids != bolivia_ids
        or (validation_ids | test_ids) & bolivia_ids
        or len(train_ids) != expectation.train_count
        or train_ids | validation_ids | test_ids != core_ids
        or len(core_ids | bolivia_ids)
        != expectation.train_count
        + expectation.validation_count
        + expectation.test_count
        + expectation.bolivia_count
    ):
        _fail("Probability exports do not preserve the exact 252/89/90/15 partition.")
    event = lambda sample_id: sample_id.split("_", 1)[0]
    test_events = sorted({event(value) for value in test_ids})
    core_events = sorted({event(value) for value in core_ids})
    bolivia_events = sorted({event(value) for value in observed_bolivia_ids})
    if (
        set(test_events) != set(core_events)
        or len(test_events) != expectation.non_bolivia_event_count
        or "Bolivia" in test_events
        or bolivia_events != ["Bolivia"]
    ):
        _fail("The probability exports do not preserve the frozen 11-event coverage.")

    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            inventory.append(_artifact(path, base=root))
    collection = hashlib.sha256()
    for item in inventory:
        collection.update(str(item["path"]).encode("utf-8"))
        collection.update(b"\0")
        collection.update(str(item["sha256"]).encode("ascii"))
        collection.update(b"\0")
    for report in reports.values():
        report.pop("writer_sample_ids", None)
        report.pop("canonical_sample_ids", None)
    return {
        "schema": "geobwer.sen1floods11.prithvi_v0432_artifact_audit.v1",
        "status": "pass",
        "formal_evidence": True,
        "audit_mode": "external_read_only_no_inference",
        "target": {
            "package_version": expectation.version,
            "externally_frozen_code_commit": expectation.commit,
            "commit_identity_source": (
                "operator-supplied frozen run identity; migration manifest v3 "
                "does not embed package_version/code_commit"
            ),
            "source_root": str(root),
            "campaign_manifest_sha256": file_sha256(manifest_path),
            "mask_gate_sha256": file_sha256(gate_path),
        },
        "counts": {
            "train": expectation.train_count,
            **counts,
            "combined_evaluation": expectation.test_count
            + expectation.bolivia_count,
            "probability_units_total": sum(
                report["row_count"] for report in reports.values()
            ),
            "total_hand_labeled_universe": (
                expectation.train_count
                + expectation.validation_count
                + expectation.test_count
                + expectation.bolivia_count
            ),
        },
        "device_contract": {**cuda_fields, "status": "pass"},
        "checkpoint": _artifact(checkpoint),
        "mask_gate": {
            "schema": gate["schema"],
            "status": gate["status"],
            "sha256": file_sha256(gate_path),
            "combined_sample_count": gate["combined_sample_count"],
            "model_loaded_by_gate": gate["model_loaded_by_gate"],
        },
        "split_identity": {
            "identity_basis": "canonical physical chip IDs bound by pre-model mask gate",
            "four_way_partition": "exact_zero_overlap",
            "train_sample_set_sha256": _sequence_hash(sorted(train_ids)),
            "validation_sample_set_sha256": reports["validation"][
                "canonical_sample_set_sha256"
            ],
            "test_sample_set_sha256": reports["test"][
                "canonical_sample_set_sha256"
            ],
            "bolivia_sample_set_sha256": reports["bolivia_holdout"][
                "canonical_sample_set_sha256"
            ],
            "standard_test_events": test_events,
            "bolivia_events": bolivia_events,
        },
        "probability_exports": reports,
        "lineage": {
            "no_training_or_calibration_leakage": True,
            "bolivia_holdout_used_for_training_or_calibration": False,
        },
        "artifact_inventory": {
            "artifact_count": len(inventory),
            "collection_sha256": collection.hexdigest(),
            "files": inventory,
        },
        "source_artifacts_modified": False,
        "gpu_inference_executed_by_audit": False,
        "blocking_errors": [],
    }


def audit_sen1_prithvi_v0432_artifacts(
    source_root: str | Path,
    *,
    output_json: str | Path,
    expectation: _Expectation = FORMAL_EXPECTATION,
) -> dict[str, Any]:
    """Audit immutable v0.4.32 Prithvi outputs and write only external evidence."""

    source = Path(source_root).resolve()
    destination = Path(output_json).resolve()
    if not source.is_dir():
        _fail(f"Frozen Prithvi source root is missing: {source}")
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        _fail("Audit evidence must be written outside the frozen source root.")
    if destination.exists():
        _fail(f"Refusing to overwrite existing audit evidence: {destination}")
    before = _snapshot(source)
    report = _audit_engine(source, expectation=expectation)
    after = _snapshot(source)
    if before != after:
        _fail("Frozen Prithvi source artifacts changed during the read-only audit.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


__all__ = [
    "Sen1PrithviV0432ArtifactAuditError",
    "audit_sen1_prithvi_v0432_artifacts",
]
