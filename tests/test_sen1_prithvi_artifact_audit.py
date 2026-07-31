from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.sen1_prithvi_artifact_audit import (
    Sen1PrithviArtifactAuditError,
    _Expectation,
    audit_sen1_prithvi_v0429_artifacts,
)


EXPECTATION = _Expectation(
    version="0.4.29",
    commit="frozen",
    train_count=2,
    validation_count=2,
    test_count=2,
    bolivia_count=1,
    non_bolivia_event_count=2,
)


def _write_export(root: Path, prefixes: list[str], *, bad_probability: bool = False) -> None:
    samples = root / "samples"
    parts = root / "index_parts"
    samples.mkdir(parents=True)
    parts.mkdir()
    rows = []
    for index, prefix in enumerate(prefixes):
        probabilities = np.zeros((2, 3, 4), dtype=np.float32)
        probabilities[0] = 0.75
        probabilities[1] = 0.25
        if bad_probability and index == 0:
            probabilities[1, 0, 0] = 0.5
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
    (parts / "part-000000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (root / "writer_manifest_rank_0.json").write_text(
        json.dumps({"row_count": len(rows)}), encoding="utf-8"
    )


def _write_metadata(path: Path, prefixes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_s2_path", "event_id"])
        writer.writeheader()
        for prefix in prefixes:
            writer.writerow(
                {
                    "source_s2_path": f"/prepared/{prefix}_S2Hand.tif",
                    "event_id": prefix.split("_", 1)[0],
                }
            )


def _fixture(tmp_path: Path, *, bad_probability: bool = False) -> dict[str, Path]:
    source = tmp_path / "prithvi"
    validation = ["EventA_3", "EventB_3"]
    test = ["EventA_4", "EventB_4"]
    train = ["EventA_1", "EventB_1"]
    bolivia = ["Bolivia_1"]
    _write_export(source / "probabilities" / "validation", validation)
    _write_export(
        source / "probabilities" / "test",
        test,
        bad_probability=bad_probability,
    )
    _write_export(source / "probabilities" / "bolivia_holdout", bolivia)
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"official checkpoint")
    manifest = {
        "schema": "geobwer.sen1floods11.prithvi_tl_probability_migration.v3",
        "formal_evidence": True,
        "split_protocol": "official_252_89_90_plus_15_bolivia_holdout",
        "train_count": 2,
        "validation_count": 2,
        "test_count": 2,
        "bolivia_holdout_count": 1,
        "combined_evaluation_count": 3,
        "bolivia_holdout_used_for_training_or_calibration": False,
        "no_training_or_calibration_leakage": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "device_contract": {
            "resolved_device": "cuda:0",
            "model_parameter_device": "cuda:0",
            "model_input_device": "cuda:0",
            "strict_no_cpu_fallback": True,
            "status": "pass",
        },
        "probability_exports": {
            split: f"/content/output/probabilities/{split}"
            for split in ("validation", "test", "bolivia_holdout")
        },
        "config": {"diagnostic_max_samples": None, "device": "cuda"},
    }
    (source / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    core_metadata = tmp_path / "core.csv"
    bolivia_metadata = tmp_path / "bolivia.csv"
    _write_metadata(core_metadata, train + validation + test)
    _write_metadata(bolivia_metadata, bolivia)

    unet_root = tmp_path / "unet"
    _write_export(unet_root / "s2" / "seed_42" / "probabilities" / "validation", validation)
    _write_export(unet_root / "s2" / "seed_42" / "probabilities" / "test", test)
    unet_campaign = unet_root / "campaign_manifest.json"
    unet_campaign.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.supervised_panel.v6",
                "package_version": "0.4.28",
                "code_commit": "60cff004057c99799ae3c9523a0eab5de4070f59",
                "formal_evidence": True,
            }
        ),
        encoding="utf-8",
    )
    unet_audit = tmp_path / "unet_audit.json"
    unet_audit.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.unet_artifact_audit.v1",
                "status": "pass",
                "formal_evidence": True,
                "blocking_errors": [],
                "target": {"campaign_manifest_sha256": file_sha256(unet_campaign)},
            }
        ),
        encoding="utf-8",
    )
    return {
        "source": source,
        "core": core_metadata,
        "bolivia": bolivia_metadata,
        "unet_campaign": unet_campaign,
        "unet_audit": unet_audit,
        "output": tmp_path / "evidence" / "audit.json",
    }


def _run(paths: dict[str, Path]) -> dict:
    return audit_sen1_prithvi_v0429_artifacts(
        paths["source"],
        core_metadata=paths["core"],
        bolivia_metadata=paths["bolivia"],
        unet_campaign=paths["unet_campaign"],
        unet_audit=paths["unet_audit"],
        output_json=paths["output"],
        expectation=EXPECTATION,
    )


def test_read_only_prithvi_external_audit_passes_and_records_hashes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    before = {
        path.relative_to(paths["source"]): file_sha256(path)
        for path in paths["source"].rglob("*")
        if path.is_file()
    }
    report = _run(paths)
    after = {
        path.relative_to(paths["source"]): file_sha256(path)
        for path in paths["source"].rglob("*")
        if path.is_file()
    }
    assert report["status"] == "pass"
    assert report["blocking_errors"] == []
    assert report["counts"]["probability_units_total"] == 5
    assert report["split_identity"]["four_way_partition"] == "exact_zero_overlap"
    assert report["device_contract"]["model_input_device"] == "cuda:0"
    assert report["probability_exports"]["test"]["target_sha256"]
    assert report["artifact_inventory"]["artifact_count"] >= 15
    assert before == after
    assert json.loads(paths["output"].read_text())["status"] == "pass"


def test_prithvi_audit_refuses_overwrite(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["output"].parent.mkdir()
    paths["output"].write_text("existing", encoding="utf-8")
    with pytest.raises(Sen1PrithviArtifactAuditError, match="overwrite"):
        _run(paths)


def test_prithvi_audit_rejects_invalid_probability(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, bad_probability=True)
    with pytest.raises(Sen1PrithviArtifactAuditError, match="sum to one"):
        _run(paths)
    assert not paths["output"].exists()


def test_prithvi_audit_rejects_unet_sample_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    index = (
        paths["unet_campaign"].parent
        / "s2/seed_42/probabilities/test/index_parts/part-000000.jsonl"
    )
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rows[0]["filename"] = {"S2L1C": "/data/EventA_WRONG_S2Hand.tif"}
    index.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    # Update the separately frozen U-Net audit binding so the test reaches
    # physical-set validation instead of failing only on campaign SHA.
    audit = json.loads(paths["unet_audit"].read_text())
    audit["target"]["campaign_manifest_sha256"] = file_sha256(paths["unet_campaign"])
    paths["unet_audit"].write_text(json.dumps(audit))
    with pytest.raises(Sen1PrithviArtifactAuditError, match="physical sample"):
        _run(paths)


def test_prithvi_audit_rejects_non_cuda_contract(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest_path = paths["source"] / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["device_contract"]["model_input_device"] = "cpu"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(Sen1PrithviArtifactAuditError, match="CUDA"):
        _run(paths)
