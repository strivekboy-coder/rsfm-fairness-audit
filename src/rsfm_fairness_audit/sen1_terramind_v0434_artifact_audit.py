from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from rsfm_fairness_audit.formal_outputs import file_sha256


class Sen1TerraMindV0434ArtifactAuditError(RuntimeError):
    """Raised when frozen TerraMind v0.4.34 evidence is incomplete."""


TARGET_COMMIT = "8122085ad69e660957a8515d62f78cc1f337a787"
TARGET_PACKAGE_VERSION = "0.4.34"
MODES = ("S1", "S2", "S1+S2")
SEEDS = (42, 73, 101)
PANEL_SCOPE = "all_19_models_unet9_terramind9_prithvi1"
SPLIT_COUNTS = {"validation": 89, "test": 90, "bolivia_holdout": 15}


def _fail(message: str) -> None:
    raise Sen1TerraMindV0434ArtifactAuditError(message)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"Missing required JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Invalid JSON at {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"Expected a JSON object at {path}.")
    return value


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _snapshot_signature(snapshot: Mapping[str, tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for relative, (size, mtime) in sorted(snapshot.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(mtime).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_record_path(root: Path, record: Mapping[str, Any]) -> Path:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or not relative.parts:
        _fail(f"Completion artifact path is not a safe relative path: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        _fail(f"Completion artifact escapes the frozen root: {relative}")
    return path


def _verify_artifact_record(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    path = _safe_record_path(root, record)
    if not path.is_file():
        _fail(f"Missing completion artifact for {label}: {path}")
    size = int(path.stat().st_size)
    if size != int(record.get("size_bytes", -1)):
        _fail(f"Completion artifact size drift for {label}: {path}")
    observed_sha = file_sha256(path)
    if observed_sha != str(record.get("sha256", "")):
        _fail(f"Completion artifact SHA-256 drift for {label}: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": size,
        "sha256": observed_sha,
    }


def _probability_path(export: Path, value: Any) -> Path:
    raw = Path(str(value))
    candidate = raw if raw.is_absolute() else export / raw
    if raw.is_absolute() and not candidate.is_file():
        candidate = export / "samples" / raw.name
    candidate = candidate.resolve()
    try:
        candidate.relative_to(export.resolve())
    except ValueError:
        _fail(f"Probability path escapes its export root: {value!r}")
    return candidate


def _audit_probability_export(
    export: Path,
    *,
    expected_count: int,
    checkpoint_sha256: str,
    split: str,
) -> dict[str, Any]:
    completion_path = export / "prediction_completion_contract.json"
    completion = _json(completion_path)
    if (
        completion.get("schema")
        != "geobwer.sen1floods11.terramind_prediction_protocol.v1"
        or int(completion.get("expected_row_count", -1)) != expected_count
        or completion.get("checkpoint_sha256") != checkpoint_sha256
    ):
        _fail(f"Prediction completion contract is incompatible for {split}: {export}")
    index_parts = sorted((export / "index_parts").glob("*.jsonl"))
    writer_manifests = sorted(export.glob("writer_manifest_rank_*.json"))
    if not index_parts or not writer_manifests:
        _fail(f"Incomplete probability-export envelope for {split}: {export}")
    rows: list[dict[str, Any]] = []
    index_hashes: dict[str, str] = {}
    for index_path in index_parts:
        index_hashes[index_path.name] = file_sha256(index_path)
        for line_number, line in enumerate(
            index_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(f"Invalid JSONL at {index_path}:{line_number}: {exc}")
            if not isinstance(item, dict):
                _fail(f"Non-object JSONL row at {index_path}:{line_number}.")
            rows.append(item)
    sample_ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if (
        len(rows) != expected_count
        or "" in sample_ids
        or len(set(sample_ids)) != expected_count
    ):
        _fail(
            f"{split} must contain {expected_count} unique sample IDs; "
            f"observed rows={len(rows)}, unique={len(set(sample_ids))}."
        )
    inventory_digest = hashlib.sha256()
    total_size = 0
    for row in rows:
        artifact = _probability_path(export, row.get("probability_path"))
        if not artifact.is_file():
            _fail(f"Missing referenced probability file for {split}: {artifact}")
        relative = artifact.relative_to(export).as_posix()
        size = int(artifact.stat().st_size)
        total_size += size
        inventory_digest.update(relative.encode("utf-8"))
        inventory_digest.update(b"\0")
        inventory_digest.update(str(size).encode("ascii"))
        inventory_digest.update(b"\0")
    return {
        "row_count": len(rows),
        "unique_sample_count": len(set(sample_ids)),
        "referenced_probability_file_count": len(rows),
        "referenced_probability_total_size_bytes": total_size,
        "probability_inventory_path_size_sha256": inventory_digest.hexdigest(),
        "prediction_completion_contract_sha256": file_sha256(completion_path),
        "index_part_count": len(index_parts),
        "index_part_sha256": index_hashes,
        "writer_manifest_count": len(writer_manifests),
        "writer_manifest_sha256": {
            path.name: file_sha256(path) for path in writer_manifests
        },
    }


def _expected_run_names() -> dict[str, tuple[str, int, str]]:
    result: dict[str, tuple[str, int, str]] = {}
    for mode in MODES:
        slug = mode.lower().replace("+", "_plus_")
        for seed in SEEDS:
            result[f"terramind_v1_base_{slug}_seed_{seed}"] = (
                mode,
                seed,
                slug,
            )
    return result


def _audit_calibration_failure(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "calibration_failure_report.json"
    payload = _json(path)
    expected_models = {
        f"{prefix}_{mode.lower().replace('+', '_plus_')}_seed_{seed}"
        for prefix in ("resnet34_unet", "terramind_v1_base")
        for mode in MODES
        for seed in SEEDS
    }
    expected_models.add("prithvi_eo_v2_300_tl_s2")
    models = payload.get("models", {})
    candidates = [float(value) for value in payload.get("candidate_cell_km", [])]
    if (
        payload.get("schema")
        != "geobwer.sen1floods11.common_spatial_block_calibration_failure.v1"
        or payload.get("status") != "calibration_invalid"
        or payload.get("validation_only") is not True
        or payload.get("calibration_panel_scope") != PANEL_SCOPE
        or int(payload.get("model_count", -1)) != 19
        or set(map(str, payload.get("model_names", []))) != expected_models
        or not isinstance(models, dict)
        or set(map(str, models)) != expected_models
        or payload.get("common_passing_cells") != []
        or not candidates
    ):
        _fail("The 19-model validation-only calibration-failure contract is invalid.")
    required_candidate_fields = {
        "cell_km",
        "passes",
        "null_coverage",
        "null_coverage_ci_low",
        "null_coverage_ci_high",
        "false_positive_rate",
        "false_positive_ci_low",
        "false_positive_ci_high",
        "moderate_tail_power",
        "moderate_tail_power_ci_low",
        "moderate_tail_power_ci_high",
    }
    matrix_count = 0
    for model_name in sorted(expected_models):
        records = models[model_name].get("candidates", [])
        if len(records) != len(candidates):
            _fail(f"Candidate count drift for calibration model={model_name}.")
        observed_cells = []
        for record in records:
            if not required_candidate_fields.issubset(record):
                _fail(f"Incomplete calibration matrix row for model={model_name}.")
            observed_cells.append(float(record["cell_km"]))
            matrix_count += 1
        if observed_cells != candidates:
            _fail(f"Candidate order/identity drift for model={model_name}.")
    failures = payload.get("failures_by_cell", {})
    if set(map(float, failures)) != set(candidates):
        _fail("Per-cell calibration failure matrix is incomplete.")
    for cell in candidates:
        record = failures[str(cell)]
        failed_models = set(map(str, record.get("failed_models", [])))
        reasons = record.get("failure_reasons_by_model", {})
        if not failed_models or set(map(str, reasons)) != failed_models:
            _fail(f"Candidate cell={cell} lacks explicit failed models/reasons.")
    return {
        "sha256": file_sha256(path),
        "calibration_signature": payload.get("calibration_signature"),
        "calibration_panel_scope": payload["calibration_panel_scope"],
        "model_count": len(models),
        "candidate_cell_km": candidates,
        "model_candidate_matrix_row_count": matrix_count,
        "common_passing_cells": [],
        "validation_only": True,
        "status": "calibration_invalid",
    }, path


def _audit_engine(root: Path, old_root: Path) -> dict[str, Any]:
    completion_path = root / "descriptive_only_completion_contract.json"
    completion = _json(completion_path)
    expected_runs = _expected_run_names()
    if (
        completion.get("schema")
        != "geobwer.sen1floods11.terramind_descriptive_only_panel.v1"
        or completion.get("status") != "descriptive_only_complete"
        or completion.get("formal_evidence") is not False
        or completion.get("package_version") != TARGET_PACKAGE_VERSION
        or completion.get("code_commit") != TARGET_COMMIT
        or completion.get("calibration_panel_scope") != PANEL_SCOPE
        or int(completion.get("run_count", -1)) != 9
        or set(completion.get("runs", {})) != set(expected_runs)
        or completion.get("validation_only_calibration") is not True
        or completion.get("test_or_bolivia_used_for_calibration") is not False
        or set(completion.get("inference_disabled", {}).values()) != {True}
    ):
        _fail("The descriptive-only campaign completion contract is invalid.")
    forbidden_paths = [
        root / "campaign_completion_contract.json",
        root / "model_panel",
    ]
    if any(path.exists() for path in forbidden_paths):
        _fail(f"Formal completion/model-panel evidence exists unexpectedly: {forbidden_paths}")
    forbidden_files = sorted(root.rglob("geobwer_summary.csv"))
    formal_output_files = [
        path for path in root.rglob("formal_outputs") if path.is_dir() and any(path.rglob("*"))
    ]
    if forbidden_files or formal_output_files:
        _fail(
            "Scale-dependent formal GeoBWER artifacts exist despite invalid "
            f"calibration: files={forbidden_files}, dirs={formal_output_files}."
        )

    calibration_report, calibration_path = _audit_calibration_failure(root)
    invalid_path = root / "calibration_invalid_contract.json"
    invalid = _json(invalid_path)
    if (
        invalid.get("status") != "calibration_invalid"
        or invalid.get("validation_only") is not True
        or invalid.get("calibration_panel_scope") != PANEL_SCOPE
        or int(invalid.get("model_count", -1)) != 19
    ):
        _fail("The calibration-invalid transition contract is invalid.")

    completion_records = completion["runs"]
    runs: dict[str, Any] = {}
    verified_completion_artifacts = 0
    old_reuse_matches: dict[str, Any] = {}
    for run_name, (mode, seed, slug) in expected_runs.items():
        run_dir = root / slug / f"seed_{seed}"
        source_run_dir = old_root / slug / f"seed_{seed}"
        if not run_dir.is_dir() or not source_run_dir.is_dir():
            _fail(f"Missing current/source run directory for {run_name}.")
        records = completion_records[run_name]
        required_record_keys = {
            "checkpoint",
            "fit_protocol",
            "fit_completion",
            "validation_prediction_contract",
            "test_prediction_contract",
            "bolivia_prediction_contract",
            "descriptive_split_report",
            "descriptive_validation",
            "descriptive_standard_test",
            "descriptive_bolivia_holdout",
            "descriptive_combined_held_out",
        }
        if set(records) != required_record_keys:
            _fail(f"Completion artifact-key drift for {run_name}: {sorted(records)}")
        verified: dict[str, Any] = {}
        for key in sorted(required_record_keys):
            verified[key] = _verify_artifact_record(
                root, records[key], label=f"{run_name}/{key}"
            )
            verified_completion_artifacts += 1

        checkpoint = _safe_record_path(root, records["checkpoint"])
        best = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
        if len(best) != 1 or best[0].resolve() != checkpoint:
            _fail(f"Expected exactly one bound best checkpoint for {run_name}.")
        fit_complete_path = run_dir / "fit_complete.json"
        fit_complete = _json(fit_complete_path)
        if (
            fit_complete.get("checkpoint_sha256") != verified["checkpoint"]["sha256"]
            or fit_complete.get("fit_protocol_sha256")
            != verified["fit_protocol"]["sha256"]
        ):
            _fail(f"Fit completion SHA binding is invalid for {run_name}.")

        exports = {
            split: _audit_probability_export(
                run_dir / "probabilities" / split,
                expected_count=count,
                checkpoint_sha256=verified["checkpoint"]["sha256"],
                split=split,
            )
            for split, count in SPLIT_COUNTS.items()
        }
        descriptive = _json(
            _safe_record_path(root, records["descriptive_split_report"])
        )
        if (
            descriptive.get("status") != "descriptive_only"
            or descriptive.get("formal_evidence") is not False
            or descriptive.get("inferential_geobwer_run") is not False
            or descriptive.get("bootstrap_run") is not False
            or descriptive.get("model_panel_inference_run") is not False
        ):
            _fail(f"Descriptive-only safeguards are invalid for {run_name}.")

        reused_relative = {
            "fit_complete": Path("fit_complete.json"),
            "fit_protocol": Path("fit_protocol.json"),
            "checkpoint": checkpoint.relative_to(run_dir),
            "validation_prediction_contract": Path(
                "probabilities/validation/prediction_completion_contract.json"
            ),
        }
        reuse_details: dict[str, Any] = {}
        for label, relative in reused_relative.items():
            current = run_dir / relative
            old = source_run_dir / relative
            if not old.is_file():
                _fail(f"Old v0.4.30 recovery artifact is missing: {old}")
            current_sha = file_sha256(current)
            old_sha = file_sha256(old)
            if current_sha != old_sha:
                _fail(f"Recovered v0.4.30 artifact SHA mismatch for {run_name}/{label}.")
            reuse_details[label] = {"sha256": current_sha, "exact_match": True}
        old_reuse_matches[run_name] = reuse_details
        runs[run_name] = {
            "mode": mode,
            "seed": seed,
            "checkpoint": verified["checkpoint"],
            "fit_complete_sha256": verified["fit_completion"]["sha256"],
            "probability_exports": exports,
            "descriptive_split_report_sha256": verified[
                "descriptive_split_report"
            ]["sha256"],
        }

    for path in root.rglob("*.json"):
        payload = _json(path)
        if (
            payload.get("schema")
            == "geobwer.sen1floods11.terramind_formal_panel.v1"
            or payload.get("formal_evidence") is True
            and payload.get("status") in {"complete", "pass"}
        ):
            _fail(f"Forbidden formal GeoBWER completion found at {path}.")
        for key in (
            "inferential_geobwer_run",
            "bootstrap_run",
            "model_panel_inference_run",
        ):
            if payload.get(key) is True:
                _fail(f"Forbidden {key}=true found at {path}.")

    return {
        "schema": "geobwer.sen1floods11.terramind_v0434_descriptive_artifact_audit.v1",
        "status": "pass",
        "formal_evidence": False,
        "audit_mode": "read_only_file_json_count_sha_no_model_or_numeric_recompute",
        "target": {
            "package_version": TARGET_PACKAGE_VERSION,
            "code_commit": TARGET_COMMIT,
            "source_root": str(root),
            "old_read_only_resume_root": str(old_root),
            "descriptive_completion_contract_sha256": file_sha256(completion_path),
            "calibration_failure_report_sha256": file_sha256(calibration_path),
            "calibration_invalid_contract_sha256": file_sha256(invalid_path),
        },
        "counts": {
            "mode_count": 3,
            "seed_count": 3,
            "run_count": len(runs),
            "best_checkpoint_count": len(runs),
            "validation_probability_units": 9 * SPLIT_COUNTS["validation"],
            "test_probability_units": 9 * SPLIT_COUNTS["test"],
            "bolivia_probability_units": 9 * SPLIT_COUNTS["bolivia_holdout"],
            "verified_completion_artifact_count": verified_completion_artifacts,
            "target_file_count": len(_snapshot(root)),
            "old_source_file_count": len(_snapshot(old_root)),
        },
        "calibration_failure": calibration_report,
        "runs": runs,
        "old_v0430_read_only_recovery": {
            "critical_artifact_exact_sha_match_for_all_runs": True,
            "runs": old_reuse_matches,
        },
        "limitations": {
            "spatial_inference_valid": False,
            "descriptive_outputs_complete": True,
            "inferential_geobwer_available": False,
            "bootstrap_significance_available": False,
            "model_panel_inference_available": False,
            "probability_arrays_deserialized_by_audit": False,
            "model_or_checkpoint_deserialized_by_audit": False,
            "gpu_used_by_audit": False,
        },
        "blocking_errors": [],
    }


def audit_sen1_terramind_v0434_artifacts(
    source_root: str | Path,
    *,
    old_resume_root: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    """Audit frozen v0.4.34 descriptive evidence without modifying either root."""

    source = Path(source_root).resolve()
    old_source = Path(old_resume_root).resolve()
    destination = Path(output_json).resolve()
    if not source.is_dir():
        _fail(f"Frozen TerraMind v0.4.34 root is missing: {source}")
    if not old_source.is_dir():
        _fail(f"Frozen TerraMind v0.4.30 resume root is missing: {old_source}")
    if source == old_source:
        _fail("v0.4.34 output and v0.4.30 read-only resume source must differ.")
    for frozen_root in (source, old_source):
        try:
            destination.relative_to(frozen_root)
        except ValueError:
            pass
        else:
            _fail("Audit JSON must be written outside both frozen source roots.")
    if destination.exists():
        _fail(f"Refusing to overwrite existing audit evidence: {destination}")
    source_before = _snapshot(source)
    old_before = _snapshot(old_source)
    report = _audit_engine(source, old_source)
    source_after = _snapshot(source)
    old_after = _snapshot(old_source)
    if source_before != source_after:
        _fail("v0.4.34 source artifacts changed during the read-only audit.")
    if old_before != old_after:
        _fail("v0.4.30 resume-source artifacts changed during the read-only audit.")
    report["read_only_snapshot"] = {
        "v0434_unchanged": True,
        "v0434_snapshot_sha256": _snapshot_signature(source_before),
        "v0430_unchanged": True,
        "v0430_snapshot_sha256": _snapshot_signature(old_before),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


__all__ = [
    "Sen1TerraMindV0434ArtifactAuditError",
    "audit_sen1_terramind_v0434_artifacts",
]
