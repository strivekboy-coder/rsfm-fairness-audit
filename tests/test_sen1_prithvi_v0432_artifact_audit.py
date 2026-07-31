from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.sen1_prithvi_v0432_artifact_audit import (
    Sen1PrithviV0432ArtifactAuditError,
    _Expectation,
    audit_sen1_prithvi_v0432_artifacts,
)


EXPECTATION = _Expectation(
    version="0.4.32",
    commit="frozen",
    train_count=2,
    validation_count=2,
    test_count=2,
    bolivia_count=1,
    non_bolivia_event_count=2,
)


def _export(root: Path, prefixes: list[str], *, corrupt: bool = False) -> None:
    (root / "samples").mkdir(parents=True)
    (root / "index_parts").mkdir()
    rows = []
    for index, prefix in enumerate(prefixes):
        probabilities = np.zeros((2, 3, 4), dtype=np.float32)
        probabilities[0] = 0.7
        probabilities[1] = 0.3
        if corrupt and index == 0:
            probabilities[1, 0, 0] = 0.8
        target = np.asarray(
            [[-1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int16
        )
        filename = json.dumps({"S2L1C": f"/data/{prefix}_S2Hand.tif"})
        relative = f"samples/{index:03d}.npz"
        np.savez_compressed(
            root / relative,
            probabilities=probabilities,
            target=target,
            filename=np.asarray(filename),
        )
        rows.append(
            {
                "sample_id": hashlib.sha256(
                    f"{root}:{prefix}".encode("utf-8")
                ).hexdigest(),
                "probability_path": relative,
                "filename": {"S2L1C": f"/data/{prefix}_S2Hand.tif"},
            }
        )
    (root / "index_parts/part-000000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (root / "writer_manifest_rank_0.json").write_text(
        json.dumps({"row_count": len(rows)}), encoding="utf-8"
    )


def _fixture(tmp_path: Path, *, corrupt: bool = False) -> dict[str, Path]:
    root = tmp_path / "prithvi"
    groups = {
        "train": ["EventA_1", "EventB_1"],
        "validation": ["EventA_2", "EventB_2"],
        "test": ["EventA_3", "EventB_3"],
        "bolivia_holdout": ["Bolivia_1"],
    }
    for split in ("validation", "test", "bolivia_holdout"):
        _export(
            root / "probabilities" / split,
            groups[split],
            corrupt=corrupt and split == "test",
        )
    core = groups["train"] + groups["validation"] + groups["test"]
    gate = {
        "schema": "geobwer.sen1floods11.prithvi_prepared_mask_gate.v1",
        "status": "pass",
        "mask_contract": {
            "shape": [224, 224],
            "finite": True,
            "integer": True,
            "allowed_values": [-1, 0, 1],
            "npz_required_key": "mask",
        },
        "core": {
            "sample_count": len(core),
            "events": ["EventA", "EventB"],
            "records": [
                {"sample_id": value, "event": value.split("_", 1)[0]}
                for value in core
            ],
        },
        "bolivia": {
            "sample_count": 1,
            "events": ["Bolivia"],
            "records": [{"sample_id": "Bolivia_1", "event": "Bolivia"}],
        },
        "combined_sample_count": 7,
        "split_overlap_count": 0,
        "model_loaded_by_gate": False,
        "blocking_errors": [],
    }
    gate_path = root / "pre_model_prepared_mask_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    runtime = {
        split: {
            "row_count": len(groups[split]),
            "full_probability_layout": "[B,2,H,W]",
            "probabilities_finite": True,
            "probabilities_in_unit_interval": True,
            "maximum_probability_sum_error": 1e-7,
            "target_shape_matches_probability_shape": True,
        }
        for split in ("validation", "test", "bolivia_holdout")
    }
    manifest = {
        "schema": "geobwer.sen1floods11.prithvi_tl_probability_migration.v3",
        "formal_evidence": True,
        "split_protocol": "official_252_89_90_plus_15_bolivia_holdout",
        "train_count": 2,
        "validation_count": 2,
        "test_count": 2,
        "bolivia_holdout_count": 1,
        "combined_evaluation_count": 3,
        "no_training_or_calibration_leakage": True,
        "bolivia_holdout_used_for_training_or_calibration": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "device_contract": {
            "status": "pass",
            "strict_no_cpu_fallback": True,
            "resolved_device": "cuda:0",
            "model_parameter_device": "cuda:0",
            "model_input_device": "cuda:0",
        },
        "pre_model_prepared_mask_gate": {
            "path": "/content/output/pre_model_prepared_mask_gate.json",
            "sha256": file_sha256(gate_path),
            "schema": gate["schema"],
            "status": "pass",
            "model_loaded_by_gate": False,
        },
        "probability_exports": {
            split: f"/content/output/probabilities/{split}"
            for split in ("validation", "test", "bolivia_holdout")
        },
        "split_runtime_validation": runtime,
        "config": {"diagnostic_max_samples": None, "device": "cuda"},
    }
    (root / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return {"root": root, "output": tmp_path / "evidence/audit.json"}


def _run(paths: dict[str, Path]):
    return audit_sen1_prithvi_v0432_artifacts(
        paths["root"], output_json=paths["output"], expectation=EXPECTATION
    )


def test_v0432_read_only_artifact_audit_passes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    before = {
        path.relative_to(paths["root"]): (path.stat().st_size, file_sha256(path))
        for path in paths["root"].rglob("*")
        if path.is_file()
    }
    report = _run(paths)
    after = {
        path.relative_to(paths["root"]): (path.stat().st_size, file_sha256(path))
        for path in paths["root"].rglob("*")
        if path.is_file()
    }
    assert report["status"] == "pass"
    assert report["blocking_errors"] == []
    assert report["counts"]["probability_units_total"] == 5
    assert report["split_identity"]["four_way_partition"] == "exact_zero_overlap"
    assert report["mask_gate"]["status"] == "pass"
    assert before == after


def test_v0432_audit_rejects_gate_sha_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    gate = paths["root"] / "pre_model_prepared_mask_gate.json"
    gate.write_text(gate.read_text() + " ", encoding="utf-8")
    with pytest.raises(Sen1PrithviV0432ArtifactAuditError, match="mask gate"):
        _run(paths)


def test_v0432_audit_rejects_corrupt_probability(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, corrupt=True)
    with pytest.raises(RuntimeError, match="sum to one"):
        _run(paths)
    assert not paths["output"].exists()


def test_v0432_audit_refuses_overwrite(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["output"].parent.mkdir()
    paths["output"].write_text("existing", encoding="utf-8")
    with pytest.raises(Sen1PrithviV0432ArtifactAuditError, match="overwrite"):
        _run(paths)
