from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest

import rsfm_fairness_audit.fmow_spatial_upgrade as spatial_upgrade
from rsfm_fairness_audit.fmow_spatial_upgrade import (
    FmowSpatialUpgradeError,
    completion_signature,
    derive_legacy_dofa_calibration,
    validate_completion_contract,
    write_completion_contract,
)
from rsfm_fairness_audit.formal_outputs import file_sha256


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _test_root(name: str) -> Path:
    root = Path("work") / "test_runs" / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _row_hash(rows: list[dict[str, str]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["sample_id"].encode())
        digest.update(b"\0")
        digest.update(row["image_id"].encode())
        digest.update(b"\0")
        digest.update(row["image_path"].encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _legacy_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, np.ndarray, np.ndarray, tuple[str, ...]]:
    source = tmp_path / "source"
    seed_root = source / "probe_seeds" / "seed_42"
    cache_root = source / "embedding_cache"
    formal_root = seed_root / "formal_outputs"
    seed_root.mkdir(parents=True)
    cache_root.mkdir()
    formal_root.mkdir()
    classes = ("a", "b", "c")
    calibration_rows = [
        {
            "sample_id": f"cal-{index}",
            "image_id": f"image-{index}",
            "image_path": f"chips/cal-{index}.tif",
            "split": "calibration",
            "category": classes[index % 3],
            "latitude": str(10.0 + index),
            "longitude": str(20.0 + index),
        }
        for index in range(4)
    ]
    metadata_rows = calibration_rows + [
        {
            "sample_id": "test-0",
            "image_id": "test-image-0",
            "image_path": "chips/test-0.tif",
            "split": "test",
            "category": "a",
            "latitude": "30",
            "longitude": "40",
        }
    ]
    metadata = tmp_path / "metadata.csv"
    _write_csv(metadata, metadata_rows)
    _write_csv(
        formal_root / "formal_audit_table.csv",
        [
            {
                "sample_id": "test-0",
                "risk": "0",
            }
        ],
    )
    embeddings = np.arange(20, dtype=np.float32).reshape(4, 5) / 10.0
    cache_path = cache_root / "dofa_calibration_fixture.npz"
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        labels=np.asarray([row["category"] for row in calibration_rows]),
        sample_ids=np.asarray([row["sample_id"] for row in calibration_rows]),
    )
    rng = np.random.default_rng(42)
    weight = rng.normal(size=(3, 5)).astype(np.float32)
    bias = rng.normal(size=(3,)).astype(np.float32)
    mean = embeddings.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(embeddings.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    checkpoint = seed_root / "linear_probe.pt"
    checkpoint.write_bytes(b"frozen-linear-probe-fixture")
    normalized = (embeddings - mean) / std
    logits = normalized @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    np.savez_compressed(
        seed_root / "calibration_predictions.npz",
        logits=logits.astype(np.float32),
        probabilities=probabilities.astype(np.float32),
        class_names=np.asarray(classes),
    )
    selection = seed_root / "probe_selection_manifest.json"
    selection.write_text(json.dumps({"schema": "fixture"}), encoding="utf-8")
    panel = source / "probe_panel_manifest.json"
    panel.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.dofav2_probe_panel.v2",
                "components": {
                    "42": {
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "selection_manifest_sha256": file_sha256(selection),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "run_manifest.json").write_text(
        json.dumps(
            {
                "embedding_caches": {
                    "calibration": {"path": f"/frozen/{cache_path.name}"}
                },
                "dataset_lineage": {
                    "calibration_row_hash": _row_hash(calibration_rows)
                },
            }
        ),
        encoding="utf-8",
    )
    return (
        source,
        metadata,
        formal_root,
        logits.astype(np.float32),
        probabilities.astype(np.float32),
        classes,
    )


def test_legacy_dofa_calibration_is_replayed_and_identity_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _test_root("legacy_dofa_spatial")
    source, metadata, formal_root, logits, probabilities, classes = _legacy_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        spatial_upgrade,
        "_replay_linear_probe",
        lambda _checkpoint, _embeddings: (logits, probabilities, classes),
    )
    derived, manifest_path = derive_legacy_dofa_calibration(
        source,
        metadata,
        tmp_path / "derived",
        seed=42,
        test_formal_dir=formal_root,
        expected_count=4,
        expected_class_count=3,
    )
    with np.load(derived, allow_pickle=False) as artifact:
        assert artifact["sample_id"].tolist() == ["cal-0", "cal-1", "cal-2", "cal-3"]
        assert artifact["targets"].tolist() == [0, 1, 2, 0]
        assert str(artifact["split_role"]) == "calibration"
        assert not bool(artifact["test_rows_used"])
        assert np.allclose(artifact["probabilities"].sum(axis=1), 1.0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["test_rows_used"] is False
    assert manifest["cpu_replay_logits_allclose"] is True
    assert len(manifest["exact_sample_id_assignment_hash"]) == 64
    assert manifest["probabilities_sha256"] == file_sha256(derived)
    assert manifest["source_artifacts_modified"] is False
    shutil.rmtree(tmp_path)


def test_legacy_dofa_calibration_rejects_metadata_order_drift() -> None:
    tmp_path = _test_root("legacy_dofa_order")
    source, metadata, formal_root, _logits, _probabilities, _classes = _legacy_fixture(
        tmp_path
    )
    rows = list(csv.DictReader(metadata.open(newline="", encoding="utf-8")))
    rows[0], rows[1] = rows[1], rows[0]
    _write_csv(metadata, rows)
    with pytest.raises(FmowSpatialUpgradeError, match="order/hash"):
        derive_legacy_dofa_calibration(
            source,
            metadata,
            tmp_path / "derived",
            seed=42,
            test_formal_dir=formal_root,
            expected_count=4,
            expected_class_count=3,
        )
    shutil.rmtree(tmp_path)


def test_spatial_upgrade_completion_contract_is_resumable_and_strict() -> None:
    tmp_path = _test_root("legacy_dofa_completion")
    root = tmp_path / "seed_42"
    root.mkdir()
    artifact = root / "result.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    payload = {"seed": 42, "source": "abc"}
    write_completion_contract(root, seed=42, signature_payload=payload)
    signature = completion_signature(payload)
    assert validate_completion_contract(root, signature)
    with pytest.raises(FmowSpatialUpgradeError, match="signature mismatch"):
        validate_completion_contract(root, completion_signature({"seed": 73}))
    artifact.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(FmowSpatialUpgradeError, match="artifact mismatch"):
        validate_completion_contract(root, signature)
    shutil.rmtree(tmp_path)
