from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.formal_outputs import file_sha256


class Sen1PrithviArtifactAuditError(RuntimeError):
    """Raised when immutable Prithvi probability evidence violates its contract."""


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
    version="0.4.29",
    commit="13bc0a38b76be449ac091b9a5cf48dd8a0e2943c",
    train_count=252,
    validation_count=89,
    test_count=90,
    bolivia_count=15,
    non_bolivia_event_count=10,
)


def _fail(message: str) -> None:
    raise Sen1PrithviArtifactAuditError(message)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"Required JSON is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Sen1PrithviArtifactAuditError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        _fail(f"Expected a JSON object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        _fail(f"Required metadata CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _canonical(value: Any) -> str:
    name = Path(str(value or "")).stem
    for suffix in ("_S2Hand", "_S1Hand", "_LabelHand", "_S2", "_S1", "_Label"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _row_prefix(row: Mapping[str, Any]) -> str:
    for key in (
        "sample_id", "chip_id", "source_s2_path", "source_s1_path",
        "source_image_path", "image_path", "chip_path",
    ):
        value = _canonical(row.get(key))
        if value:
            return value
    return ""


def _filename_prefix(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            decoded = value
        value = decoded
    if isinstance(value, Mapping):
        for key in ("S2L1C", "S1GRD", "image", "mask"):
            prefix = _canonical(value.get(key))
            if prefix:
                return prefix
        for item in value.values():
            prefix = _canonical(item)
            if prefix:
                return prefix
    return _canonical(value)


def _read_index(export: Path) -> list[dict[str, Any]]:
    parts = sorted((export / "index_parts").glob("*.jsonl"))
    if not parts:
        _fail(f"No probability index parts found: {export}")
    rows: list[dict[str, Any]] = []
    for part in parts:
        for line_number, line in enumerate(
            part.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise Sen1PrithviArtifactAuditError(
                    f"Invalid JSONL row {line_number}: {part}"
                ) from exc
            if not isinstance(row, dict):
                _fail(f"Index row is not an object: {part}:{line_number}")
            rows.append(row)
    return rows


def _sequence_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _target_hash(rows: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for prefix, target in rows:
        # Hash label semantics rather than writer-specific integer width. Both
        # exporters are required to contain only {-1,0,1}; int16 versus int64
        # is serialization detail, not a target difference.
        array = np.ascontiguousarray(target, dtype=np.int8)
        digest.update(prefix.encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"canonical_int8")
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _artifact(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    return {
        "path": (
            path.relative_to(base).as_posix()
            if base is not None and path.is_relative_to(base)
            else str(path)
        ),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _audit_export(
    export: Path,
    *,
    expected_count: int,
    split: str,
    artifact_base: Path,
) -> dict[str, Any]:
    rows = _read_index(export)
    if len(rows) != expected_count:
        _fail(f"{split} has {len(rows)} probability units; expected {expected_count}.")
    writer_ids: list[str] = []
    prefixes: list[str] = []
    target_rows: list[tuple[str, np.ndarray]] = []
    artifacts: list[dict[str, Any]] = []
    probability_paths: set[str] = set()
    max_sum_error = 0.0
    observed_targets: set[int] = set()
    for row in rows:
        writer_id = str(row.get("sample_id", "")).strip()
        relative = str(row.get("probability_path", "")).strip()
        if (
            not writer_id
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in probability_paths
        ):
            _fail(f"Unsafe/empty/duplicate {split} index row.")
        probability_paths.add(relative)
        path = export / relative
        if not path.is_file():
            _fail(f"Probability NPZ is missing: {path}")
        try:
            with np.load(path, allow_pickle=False) as bundle:
                probabilities = np.asarray(bundle["probabilities"])
                target = np.asarray(bundle["target"])
                filename = bundle["filename"] if "filename" in bundle else row.get("filename")
        except Exception as exc:
            raise Sen1PrithviArtifactAuditError(f"Cannot read probability NPZ: {path}") from exc
        prefix = _filename_prefix(row.get("filename"))
        npz_prefix = _filename_prefix(filename)
        if not prefix:
            prefix = npz_prefix
        if not prefix or (npz_prefix and npz_prefix != prefix):
            _fail(f"Index/NPZ physical sample identity mismatch: {path}")
        if probabilities.ndim != 3 or probabilities.shape[0] != 2:
            _fail(f"Expected [2,H,W] probabilities, got {probabilities.shape}: {path}")
        if target.shape != probabilities.shape[1:]:
            _fail(f"Target/probability shape mismatch: {path}")
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < -1e-6)
            or np.any(probabilities > 1.0 + 1e-6)
        ):
            _fail(f"Invalid probability values: {path}")
        error = float(np.max(np.abs(probabilities.sum(axis=0) - 1.0)))
        if error > 1e-5:
            _fail(f"Class probabilities do not sum to one ({error}): {path}")
        if not np.issubdtype(target.dtype, np.integer):
            if not np.all(np.isfinite(target)) or not np.all(target == np.round(target)):
                _fail(f"Target is not integer-valued: {path}")
        values = {int(value) for value in np.unique(target)}
        if not values.issubset({-1, 0, 1}):
            _fail(f"Unexpected target values {sorted(values)}: {path}")
        writer_ids.append(writer_id)
        prefixes.append(prefix)
        target_rows.append((prefix, target))
        artifacts.append(_artifact(path, base=artifact_base))
        max_sum_error = max(max_sum_error, error)
        observed_targets.update(values)
    if len(set(writer_ids)) != expected_count or len(set(prefixes)) != expected_count:
        _fail(f"{split} contains duplicate writer IDs or physical chip IDs.")
    for path in sorted((export / "index_parts").glob("*.jsonl")):
        artifacts.append(_artifact(path, base=artifact_base))
    writer_manifest = export / "writer_manifest_rank_0.json"
    if writer_manifest.is_file():
        artifacts.append(_artifact(writer_manifest, base=artifact_base))
    return {
        "row_count": len(rows),
        "writer_sample_ids": writer_ids,
        "canonical_sample_ids": prefixes,
        "writer_sample_order_sha256": _sequence_hash(writer_ids),
        "canonical_sample_order_sha256": _sequence_hash(prefixes),
        "canonical_sample_set_sha256": _sequence_hash(sorted(prefixes)),
        "target_sha256": _target_hash(target_rows),
        "target_rows": target_rows,
        "observed_target_values": sorted(observed_targets),
        "maximum_probability_sum_error": max_sum_error,
        "artifacts": artifacts,
    }


def _unet_reference(
    campaign_path: Path,
    audit_path: Path,
    *,
    split_counts: Mapping[str, int],
    artifact_base: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    campaign = _json(campaign_path)
    audit = _json(audit_path)
    if (
        campaign.get("schema") != "geobwer.sen1floods11.supervised_panel.v6"
        or campaign.get("package_version") != "0.4.28"
        or campaign.get("code_commit") != "60cff004057c99799ae3c9523a0eab5de4070f59"
        or campaign.get("formal_evidence") is not True
        or audit.get("schema") != "geobwer.sen1floods11.unet_artifact_audit.v1"
        or audit.get("status") != "pass"
        or audit.get("formal_evidence") is not True
        or audit.get("blocking_errors") != []
        or audit.get("target", {}).get("campaign_manifest_sha256")
        != file_sha256(campaign_path)
    ):
        _fail("U-Net campaign or PASS audit is not the frozen audited reference.")
    root = campaign_path.parent
    run = root / "s2" / "seed_42"
    references: dict[str, Any] = {}
    artifacts = [_artifact(campaign_path), _artifact(audit_path)]
    for split in ("validation", "test"):
        report = _audit_export(
            run / "probabilities" / split,
            expected_count=int(split_counts[split]),
            split=f"unet_reference_{split}",
            artifact_base=artifact_base,
        )
        references[split] = report
        artifacts.extend(report.pop("artifacts"))
        report.pop("target_rows")
    return references, artifacts


def _audit_engine(
    source_root: Path,
    *,
    core_metadata: Path,
    bolivia_metadata: Path,
    unet_campaign: Path,
    unet_audit: Path,
    expectation: _Expectation,
) -> dict[str, Any]:
    root = source_root.resolve()
    manifest_path = root / "campaign_manifest.json"
    manifest = _json(manifest_path)
    expected_counts = {
        "train": expectation.train_count,
        "validation": expectation.validation_count,
        "test": expectation.test_count,
        "bolivia_holdout": expectation.bolivia_count,
    }
    if (
        manifest.get("schema")
        != "geobwer.sen1floods11.prithvi_tl_probability_migration.v3"
        or manifest.get("formal_evidence") is not True
        or manifest.get("split_protocol")
        != "official_252_89_90_plus_15_bolivia_holdout"
        or manifest.get("bolivia_holdout_used_for_training_or_calibration") is not False
        or manifest.get("no_training_or_calibration_leakage") is not True
        or int(manifest.get("train_count", -1)) != expectation.train_count
        or int(manifest.get("validation_count", -1)) != expectation.validation_count
        or int(manifest.get("test_count", -1)) != expectation.test_count
        or int(manifest.get("bolivia_holdout_count", -1)) != expectation.bolivia_count
        or int(manifest.get("combined_evaluation_count", -1))
        != expectation.test_count + expectation.bolivia_count
    ):
        _fail("Prithvi campaign manifest violates the frozen formal contract.")
    config = manifest.get("config", {})
    if (
        str(config.get("diagnostic_max_samples")) not in {"None", ""}
        or str(config.get("device", "")).lower() != "cuda"
    ):
        _fail("Prithvi campaign was not a formal CUDA run.")
    device = manifest.get("device_contract", {})
    cuda_fields = {
        "resolved_device": str(device.get("resolved_device", "")),
        "model_parameter_device": str(device.get("model_parameter_device", "")),
        "model_input_device": str(device.get("model_input_device", "")),
    }
    if (
        device.get("status") != "pass"
        or not device.get("strict_no_cpu_fallback")
        or not all(value.startswith("cuda") for value in cuda_fields.values())
    ):
        _fail(f"Prithvi CUDA device contract is not certified: {cuda_fields}")
    checkpoint = Path(str(manifest.get("checkpoint_path", "")))
    if (
        not checkpoint.is_file()
        or not manifest.get("checkpoint_sha256")
        or file_sha256(checkpoint) != manifest.get("checkpoint_sha256")
    ):
        _fail("Prithvi checkpoint is missing or its SHA-256 disagrees with the manifest.")

    reports: dict[str, Any] = {}
    artifacts = [_artifact(manifest_path, base=root), _artifact(checkpoint)]
    for split in ("validation", "test", "bolivia_holdout"):
        export = root / "probabilities" / split
        recorded = str(manifest.get("probability_exports", {}).get(split, "")).replace("\\", "/")
        if not recorded.endswith(f"/probabilities/{split}"):
            _fail(f"Manifest does not bind the canonical {split} probability export.")
        report = _audit_export(
            export,
            expected_count=expected_counts[split],
            split=split,
            artifact_base=root,
        )
        artifacts.extend(report.pop("artifacts"))
        report.pop("target_rows")
        reports[split] = report

    unet_refs, unet_artifacts = _unet_reference(
        unet_campaign,
        unet_audit,
        split_counts=expected_counts,
        artifact_base=unet_campaign.parent,
    )
    artifacts.extend(unet_artifacts)
    for split in ("validation", "test"):
        if set(reports[split]["canonical_sample_ids"]) != set(
            unet_refs[split]["canonical_sample_ids"]
        ):
            _fail(f"Prithvi and audited U-Net {split} physical sample sets differ.")
        if reports[split]["target_sha256"] != unet_refs[split]["target_sha256"]:
            _fail(f"Prithvi and audited U-Net {split} target SHA-256 values differ.")

    core_rows = _csv(core_metadata)
    bolivia_rows = _csv(bolivia_metadata)
    expected_core_count = (
        expectation.train_count + expectation.validation_count + expectation.test_count
    )
    expected_universe_count = expected_core_count + expectation.bolivia_count
    if (
        len(core_rows) != expected_core_count
        or len(bolivia_rows) != expectation.bolivia_count
    ):
        _fail(
            "Core/Bolivia metadata row counts disagree with the frozen "
            f"{expected_core_count}/{expectation.bolivia_count} contract."
        )
    core_ids = [_row_prefix(row) for row in core_rows]
    bolivia_ids = [_row_prefix(row) for row in bolivia_rows]
    if (
        any(not value for value in core_ids + bolivia_ids)
        or len(set(core_ids)) != expected_core_count
        or len(set(bolivia_ids)) != expectation.bolivia_count
        or set(core_ids) & set(bolivia_ids)
    ):
        _fail("Metadata contains empty, duplicate, or overlapping physical sample IDs.")
    if set(reports["bolivia_holdout"]["canonical_sample_ids"]) != set(bolivia_ids):
        _fail("Prithvi Bolivia probability set differs from the independent 15-chip metadata.")
    validation = set(reports["validation"]["canonical_sample_ids"])
    test = set(reports["test"]["canonical_sample_ids"])
    bolivia = set(reports["bolivia_holdout"]["canonical_sample_ids"])
    train = set(core_ids) - validation - test
    if (
        len(train) != expectation.train_count
        or validation & test
        or validation & bolivia
        or test & bolivia
        or train & (validation | test | bolivia)
        or train | validation | test != set(core_ids)
        or len(train | validation | test | bolivia) != expected_universe_count
    ):
        _fail("The train/validation/test/Bolivia partition is not an exact 446-chip partition.")
    event = lambda prefix: prefix.split("_", 1)[0]
    test_events = sorted({event(value) for value in test})
    bolivia_events = sorted({event(value) for value in bolivia})
    core_events = sorted({event(value) for value in core_ids})
    if (
        len(test_events) != expectation.non_bolivia_event_count
        or set(test_events) != set(core_events)
        or "Bolivia" in test_events
        or bolivia_events != ["Bolivia"]
    ):
        _fail("Standard test/Bolivia event coverage does not certify the 11-event universe.")
    artifacts.extend(
        [_artifact(core_metadata), _artifact(bolivia_metadata)]
    )
    artifact_map = {
        (item["path"], item["sha256"]): item for item in artifacts
    }
    ordered_artifacts = sorted(
        artifact_map.values(), key=lambda item: (str(item["path"]), str(item["sha256"]))
    )
    collection = hashlib.sha256()
    for item in ordered_artifacts:
        collection.update(str(item["path"]).encode("utf-8"))
        collection.update(b"\0")
        collection.update(str(item["sha256"]).encode("ascii"))
        collection.update(b"\0")
    for report in reports.values():
        report.pop("writer_sample_ids", None)
        report.pop("canonical_sample_ids", None)
    for report in unet_refs.values():
        report.pop("writer_sample_ids", None)
        report.pop("canonical_sample_ids", None)
    return {
        "schema": "geobwer.sen1floods11.prithvi_artifact_audit.v1",
        "status": "pass",
        "formal_evidence": True,
        "audit_mode": "external_read_only_no_inference",
        "target": {
            "migration_schema": manifest["schema"],
            "package_version": expectation.version,
            "code_commit": expectation.commit,
            "source_root": str(root),
            "campaign_manifest_sha256": file_sha256(manifest_path),
        },
        "counts": {
            **expected_counts,
            "combined_evaluation": expectation.test_count + expectation.bolivia_count,
            "probability_units_total": sum(
                reports[key]["row_count"] for key in reports
            ),
            "total_hand_labeled_universe": expected_universe_count,
        },
        "device_contract": {**cuda_fields, "status": "pass"},
        "checkpoint": _artifact(checkpoint),
        "split_identity": {
            "validation_and_test_match_audited_unet": "exact_canonical_set_and_target",
            "bolivia_matches_independent_metadata": "exact_canonical_set",
            "four_way_partition": "exact_zero_overlap",
            "standard_test_events": test_events,
            "bolivia_events": bolivia_events,
            "train_sample_set_sha256": _sequence_hash(sorted(train)),
            "core_metadata_sample_order_sha256": _sequence_hash(core_ids),
            "bolivia_metadata_sample_order_sha256": _sequence_hash(bolivia_ids),
        },
        "probability_exports": reports,
        "audited_unet_reference": {
            "campaign_manifest_sha256": file_sha256(unet_campaign),
            "pass_audit_sha256": file_sha256(unet_audit),
            "reference_run": "resnet34_unet_s2_seed_42",
            "exports": unet_refs,
        },
        "lineage": {
            "core_metadata_sha256": file_sha256(core_metadata),
            "bolivia_metadata_sha256": file_sha256(bolivia_metadata),
            "no_training_or_calibration_leakage": True,
            "bolivia_holdout_used_for_training_or_calibration": False,
        },
        "artifact_inventory": {
            "artifact_count": len(ordered_artifacts),
            "collection_sha256": collection.hexdigest(),
            "files": ordered_artifacts,
        },
        "source_artifacts_modified": False,
        "gpu_inference_executed_by_audit": False,
        "terramind_started_by_audit": False,
        "blocking_errors": [],
    }


def audit_sen1_prithvi_v0429_artifacts(
    source_root: str | Path,
    *,
    core_metadata: str | Path,
    bolivia_metadata: str | Path,
    unet_campaign: str | Path,
    unet_audit: str | Path,
    output_json: str | Path,
    expectation: _Expectation = FORMAL_EXPECTATION,
) -> dict[str, Any]:
    """Audit frozen Prithvi outputs without mutating or executing the model."""

    source = Path(source_root).resolve()
    destination = Path(output_json).resolve()
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        _fail("Audit evidence must be outside the frozen Prithvi source root.")
    if destination.exists():
        _fail(f"Refusing to overwrite existing audit evidence: {destination}")
    source_before = {
        path.relative_to(source).as_posix(): (
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    report = _audit_engine(
        source,
        core_metadata=Path(core_metadata).resolve(),
        bolivia_metadata=Path(bolivia_metadata).resolve(),
        unet_campaign=Path(unet_campaign).resolve(),
        unet_audit=Path(unet_audit).resolve(),
        expectation=expectation,
    )
    source_after = {
        path.relative_to(source).as_posix(): (
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    if source_before != source_after:
        _fail("Frozen Prithvi source artifacts changed during the read-only audit.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


__all__ = [
    "Sen1PrithviArtifactAuditError",
    "audit_sen1_prithvi_v0429_artifacts",
]
