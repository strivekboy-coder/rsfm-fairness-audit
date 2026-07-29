from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.sen1_unet_artifact_audit import (
    AMP_SCHEMA,
    IMPUTATION_POLICY,
    SOURCE_COMMIT,
    SOURCE_VERSION,
    Sen1UNetArtifactAuditError,
    _Expectation,
    _artifact_tree_binding,
    _audit_engine,
)


TEST_EXPECTATION = _Expectation(
    version="0.4.28",
    commit="60cff004057c99799ae3c9523a0eab5de4070f59",
    modes=("S1", "S2", "S1+S2"),
    seeds=(42, 73, 101),
    split_counts={"validation": 2, "test": 2, "bolivia_holdout": 1},
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _name(mode: str, seed: int) -> str:
    return f"resnet34_unet_{mode.lower().replace('+', '_plus_')}_seed_{seed}"


def _make_export(
    export: Path,
    *,
    split: str,
    mode: str,
    count: int,
    quality_path: Path,
) -> dict[str, object]:
    rows = []
    valid_counts = []
    for index in range(count):
        sample_id = f"{split}_{index}"
        artifact = export / "samples" / f"{sample_id}.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        probabilities = np.asarray(
            [[[0.8, 0.2], [0.3, 0.7]], [[0.2, 0.8], [0.7, 0.3]]],
            dtype=np.float32,
        )
        target = np.asarray([[-1, 0], [1, 0]], dtype=np.int64)
        np.savez_compressed(
            artifact,
            probabilities=probabilities,
            target=target,
            filename=np.asarray(sample_id),
        )
        rows.append(
            {
                "sample_id": sample_id,
                "probability_path": f"samples/{sample_id}.npz",
                "probability_shape": "[2, 2, 2]",
                "target_shape": "[2, 2]",
            }
        )
        valid_counts.append(3)
    index_path = export / "index_parts" / "part-000000.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    binding_path = export / "input_quality_binding.json"
    _write_json(
        binding_path,
        {
            "schema": "geobwer.sen1floods11.input_quality_binding.v1",
            "split": split,
            "split_role": "standard_test" if split == "test" else split,
            "sensor_mode": mode,
            "imputation_policy": IMPUTATION_POLICY,
            "input_quality_contract": str(quality_path),
            "input_quality_contract_sha256": file_sha256(quality_path),
            "prefix_sha256": f"{split}-prefix",
            "summary": {},
            "fully_missing_modality_records": [],
        },
    )
    support_path = export / "support_contract.json"
    support = {
        "schema": "geobwer.sen1floods11.probability_support.v1",
        "split": split,
        "sensor_mode": mode,
        "input_quality_binding": str(binding_path),
        "input_quality_binding_sha256": file_sha256(binding_path),
        "fully_missing_modality_count": 0,
        "fully_missing_sample_ids": [],
        "row_count": count,
        "all_ignore_row_count": 0,
        "valid_row_count": count,
        "aggregate_valid_pixel_count": count * 3,
        "observed_target_values": [-1, 0, 1],
        "valid_pixel_counts": valid_counts,
    }
    _write_json(support_path, support)
    return {
        "row_count": count,
        "all_ignore_row_count": 0,
        "valid_row_count": count,
        "aggregate_valid_pixel_count": count * 3,
        "observed_target_values": [-1, 0, 1],
        "valid_pixel_counts": valid_counts,
        "support_contract": str(support_path),
        "support_contract_sha256": file_sha256(support_path),
        "input_quality_binding": str(binding_path),
        "input_quality_binding_sha256": file_sha256(binding_path),
        "input_quality_summary": {},
    }


def _build_panel(root: Path, *, overflow: bool = True) -> Path:
    normalization_contracts = {}
    quality_contracts = {}
    mode_artifacts = {}
    for mode in TEST_EXPECTATION.modes:
        slug = mode.lower().replace("+", "_plus_")
        normalization_path = root / "normalization" / f"{slug}.json"
        _write_json(
            normalization_path,
            {
                "schema": "geobwer.sen1floods11.train_normalization.v4",
                "sensor_mode": mode,
                "selection_split": "official_train",
                "test_rows_used": False,
                "imputation_policy": IMPUTATION_POLICY,
                "normalization_sample_count": 252,
                "sample_prefixes": [f"train_{i}" for i in range(252)],
                "sample_prefix_sha256": "train-prefix",
                "mean": [0.0],
                "std": [1.0],
            },
        )
        quality_path = root / "input_quality" / f"{slug}.json"
        _write_json(
            quality_path,
            {
                "schema": "geobwer.sen1floods11.input_quality.v2",
                "sensor_mode": mode,
                "imputation_policy": IMPUTATION_POLICY,
                "normalization_sha256": file_sha256(normalization_path),
                "summary": {},
                "splits": {
                    split: {
                        "prefix_sha256": f"{split}-prefix",
                        "records": [{"sample_id": f"{split}_{i}"} for i in range(count)],
                    }
                    for split, count in TEST_EXPECTATION.split_counts.items()
                },
            },
        )
        normalization_contracts[mode] = {
            "path": str(normalization_path),
            "sha256": file_sha256(normalization_path),
            "imputation_policy": IMPUTATION_POLICY,
            "normalization_sample_count": 252,
            "sample_prefix_sha256": "train-prefix",
        }
        quality_contracts[mode] = {
            "path": str(quality_path),
            "sha256": file_sha256(quality_path),
            "imputation_policy": IMPUTATION_POLICY,
            "summary": {},
        }
        mode_artifacts[mode] = (normalization_path, quality_path)

    campaign_runs = {}
    for mode in TEST_EXPECTATION.modes:
        slug = mode.lower().replace("+", "_plus_")
        normalization_path, quality_path = mode_artifacts[mode]
        for seed in TEST_EXPECTATION.seeds:
            name = _name(mode, seed)
            run = root / slug / f"seed_{seed}"
            checkpoint = run / "best_resnet34_unet.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{mode}-{seed}-checkpoint".encode())
            split_support = {}
            exports = {}
            for split, count in TEST_EXPECTATION.split_counts.items():
                export = run / "probabilities" / split
                split_support[split] = _make_export(
                    export,
                    split=split,
                    mode=mode,
                    count=count,
                    quality_path=quality_path,
                )
                exports[split] = str(export)
            records = []
            if overflow and mode == "S1" and seed == 101:
                records = [
                    {
                        "schema": AMP_SCHEMA,
                        "mode": mode,
                        "seed": seed,
                        "training_stage": "model_selection",
                        "epoch": 5,
                        "batch_index": 10,
                        "sample_ids": ["Ghana_134751"],
                        "scale_before": 65536.0,
                        "scale_after": 32768.0,
                        "overflow_parameter_names": ["head.3.weight"],
                        "optimizer_step_skipped": True,
                    }
                ]
                _write_json(
                    run / "amp_overflow_journal.json",
                    {
                        "schema": AMP_SCHEMA,
                        "max_consecutive_overflows": 3,
                        "max_total_overflows": 20,
                        "amp_overflow_count": 1,
                        "consecutive_amp_overflow_count": 1,
                        "maximum_consecutive_amp_overflow_count": 1,
                        "skipped_optimizer_step_count": 1,
                        "amp_overflow_records": records,
                    },
                )
            journal = run / "amp_overflow_journal.json"
            manifest = {
                "schema": "geobwer.sen1floods11.supervised_resnet34_unet.v6",
                "package_version": TEST_EXPECTATION.version,
                "code_commit": TEST_EXPECTATION.commit,
                "formal_evidence": True,
                "architecture": "resnet34_unet",
                "adaptation_protocol": "supervised_from_scratch",
                "sensor_mode": mode,
                "input_channels": {"S1": 2, "S2": 13, "S1+S2": 15}[mode],
                "seed": seed,
                "best_epoch": 1,
                "best_validation_iou": 0.4,
                "best_inner_selection_iou": 0.4,
                "model_selection": "official_train_inner_event_disjoint",
                "outer_validation_used_for_model_selection": False,
                "bolivia_holdout_used_for_training_or_selection": False,
                "split_metrics": {"validation": 0.4, "test": 0.3, "bolivia_holdout": 0.2},
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "normalization": {"mean": [0.0], "std": [1.0]},
                "normalization_sha256": file_sha256(normalization_path),
                "imputation_policy": IMPUTATION_POLICY,
                "input_quality_contract": {
                    "path": str(quality_path),
                    "sha256": file_sha256(quality_path),
                    "summary": {},
                },
                "probability_exports": exports,
                "split_support": split_support,
                "skipped_all_ignore_batch_count": 0,
                "refit_skipped_all_ignore_batch_count": 0,
                "amp_overflow_policy_schema": AMP_SCHEMA,
                "amp_max_consecutive_overflows": 3,
                "amp_max_total_overflows": 20,
                "amp_overflow_count": len(records),
                "amp_overflow_records": records,
                "skipped_optimizer_step_count": len(records),
                "maximum_consecutive_amp_overflow_count": len(records),
                "amp_overflow_journal": str(journal) if records else None,
                "amp_overflow_journal_sha256": file_sha256(journal) if records else None,
                "history": [{"epoch": 1, "train_loss": 0.7}],
                "refit_history": [{"epoch": 1, "train_loss": 0.6}],
            }
            manifest_path = run / "run_manifest.json"
            _write_json(manifest_path, manifest)
            campaign_runs[name] = {
                "checkpoint": str(checkpoint),
                "manifest": str(manifest_path),
                "validation_export": exports["validation"],
                "test_export": exports["test"],
                "bolivia_holdout_export": exports["bolivia_holdout"],
                "validation_iou": 0.4,
                "test_iou": 0.3,
                "bolivia_holdout_iou": 0.2,
            }
    _write_json(
        root / "campaign_manifest.json",
        {
            "schema": "geobwer.sen1floods11.supervised_panel.v6",
            "package_version": TEST_EXPECTATION.version,
            "code_commit": TEST_EXPECTATION.commit,
            "formal_evidence": True,
            "design": "resnet34_unet_x_sensor_mode_x_seed",
            "split_protocol": "official_252_89_90_plus_15_bolivia_holdout",
            "evaluation_sample_count": 3,
            "standard_test_count": 2,
            "bolivia_holdout_count": 1,
            "no_training_or_calibration_leakage": True,
            "sensor_modes": list(TEST_EXPECTATION.modes),
            "seeds": list(TEST_EXPECTATION.seeds),
            "config": {"diagnostic_max_samples": None},
            "normalization_contracts": normalization_contracts,
            "input_quality_contracts": quality_contracts,
            "carry_forward": None,
            "runs": campaign_runs,
        },
    )
    return root


def test_audits_complete_nine_run_panel_and_amp_journal(tmp_path: Path) -> None:
    root = _build_panel(tmp_path / "source")
    report = _audit_engine(root, expectation=TEST_EXPECTATION, repository_root=None)
    assert report["status"] == "pass"
    assert report["model_count"] == 9
    assert report["total_amp_overflow_count"] == 1
    assert report["cross_model_sample_and_target_identity"] == "exact"
    assert report["runs"]["resnet34_unet_s1_seed_101"]["amp"]["amp_overflow_count"] == 1


def test_rejects_checkpoint_sha_mismatch(tmp_path: Path) -> None:
    root = _build_panel(tmp_path / "source", overflow=False)
    (root / "s1" / "seed_42" / "best_resnet34_unet.pt").write_bytes(b"tampered")
    with pytest.raises(Sen1UNetArtifactAuditError, match="Checkpoint"):
        _audit_engine(root, expectation=TEST_EXPECTATION, repository_root=None)


def test_rejects_invalid_probability_bundle(tmp_path: Path) -> None:
    root = _build_panel(tmp_path / "source", overflow=False)
    artifact = (
        root
        / "s1"
        / "seed_42"
        / "probabilities"
        / "test"
        / "samples"
        / "test_0.npz"
    )
    np.savez_compressed(
        artifact,
        probabilities=np.full((2, 2, 2), np.nan, dtype=np.float32),
        target=np.zeros((2, 2), dtype=np.int64),
    )
    with pytest.raises(Sen1UNetArtifactAuditError, match="Invalid probability"):
        _audit_engine(root, expectation=TEST_EXPECTATION, repository_root=None)


def test_rejects_cross_model_target_mismatch(tmp_path: Path) -> None:
    root = _build_panel(tmp_path / "source", overflow=False)
    artifact = (
        root
        / "s2"
        / "seed_73"
        / "probabilities"
        / "validation"
        / "samples"
        / "validation_0.npz"
    )
    with np.load(artifact, allow_pickle=False) as old:
        probabilities = old["probabilities"]
    np.savez_compressed(
        artifact,
        probabilities=probabilities,
        target=np.asarray([[-1, 1], [1, 0]], dtype=np.int64),
    )
    with pytest.raises(Sen1UNetArtifactAuditError, match="Cross-model"):
        _audit_engine(root, expectation=TEST_EXPECTATION, repository_root=None)


def test_rejects_amp_journal_sha_mismatch(tmp_path: Path) -> None:
    root = _build_panel(tmp_path / "source", overflow=True)
    journal = root / "s1" / "seed_101" / "amp_overflow_journal.json"
    journal.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Sen1UNetArtifactAuditError, match="journal"):
        _audit_engine(root, expectation=TEST_EXPECTATION, repository_root=None)


def test_accepts_only_cryptographically_bound_carry_forward(tmp_path: Path) -> None:
    root = _build_panel(tmp_path / "source", overflow=False)
    mode, seed = "S1", 42
    name = _name(mode, seed)
    run = root / "s1" / "seed_42"
    run_manifest_path = run / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest.update(
        {
            "schema": "geobwer.sen1floods11.supervised_resnet34_unet.v5",
            "package_version": SOURCE_VERSION,
            "code_commit": SOURCE_COMMIT,
        }
    )
    _write_json(run_manifest_path, run_manifest)
    carry_path = tmp_path / "migration" / "carry_forward.json"
    carry = {
        "schema": "geobwer.sen1floods11.amp_carry_forward.v1",
        "status": "validated",
        "source_version": SOURCE_VERSION,
        "source_commit": SOURCE_COMMIT,
        "target_version": TEST_EXPECTATION.version,
        "target_commit": TEST_EXPECTATION.commit,
        "source_overflow_hard_fail_proof": {
            "source_blob_sha256": "a" * 64,
            "proof": "A completed source could not have observed an overflow.",
        },
        "no_overflow_numerical_semantics": {"status": "preserved"},
        "entries": [
            {
                "mode": mode,
                "seed": seed,
                "run_dir": str(run),
                "normalization_sha256": run_manifest["normalization_sha256"],
                "input_quality_contract_sha256": run_manifest[
                    "input_quality_contract"
                ]["sha256"],
                "best_validation_iou": 0.4,
                "test_iou": 0.3,
                "bolivia_holdout_iou": 0.2,
                "artifact_binding": _artifact_tree_binding(run),
                "overflow_observation": {
                    "observed_amp_overflow_count": 0,
                    "proof_basis": "formal hard-fail source",
                },
            }
        ],
    }
    _write_json(carry_path, carry)
    campaign_path = root / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["carry_forward"] = {
        "manifest": str(carry_path),
        "manifest_sha256": file_sha256(carry_path),
        "source_commit": SOURCE_COMMIT,
        "source_version": SOURCE_VERSION,
        "target_version": TEST_EXPECTATION.version,
        "target_commit": TEST_EXPECTATION.commit,
    }
    campaign["runs"][name].update(
        {
            "carry_forward": True,
            "carry_forward_manifest": str(carry_path),
            "carry_forward_manifest_sha256": file_sha256(carry_path),
        }
    )
    _write_json(campaign_path, campaign)

    report = _audit_engine(root, expectation=TEST_EXPECTATION, repository_root=None)
    assert report["runs"][name]["carry_forward"] is True
    assert (
        report["runs"][name]["amp"]["evidence"]
        == "validated_v0.4.27_hard_fail_carry_forward"
    )
