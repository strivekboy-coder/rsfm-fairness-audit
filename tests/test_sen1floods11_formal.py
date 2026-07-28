from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.sen1floods11_formal import finalize_sen1floods11_segmentation
from rsfm_fairness_audit.terratorch_exports import write_probability_batch


WORK = Path("work/test_sen1_formal")


@pytest.fixture(autouse=True)
def _clean_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    yield
    if WORK.exists():
        shutil.rmtree(WORK)


def test_finalizes_labeled_probability_exports_without_test_tuning():
    export = (WORK / "export").resolve()
    rows = write_probability_batch(
        export,
        outputs={
            "probabilities": np.asarray(
                [
                    [[[0.9, 0.2], [0.6, 0.4]], [[0.1, 0.8], [0.4, 0.6]]],
                    [[[0.2, 0.3], [0.8, 0.7]], [[0.8, 0.7], [0.2, 0.3]]],
                ],
                dtype=np.float32,
            ),
            "filename": ["Bolivia_001_S2Hand.tif", "Pakistan_001_S2Hand.tif"],
        },
        batch={"mask": np.asarray([[[0, 1], [0, 1]], [[1, 1], [0, -1]]], dtype=np.int64)},
        batch_idx=0,
    )
    index_dir = export / "index_parts"
    index_dir.mkdir(parents=True)
    (index_dir / "batch.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    # Simulate the production /content -> Google Drive mirror.  The index must
    # remain usable after the export directory changes location.
    moved_export = (WORK / "drive_mirror" / "export").resolve()
    moved_export.parent.mkdir(parents=True)
    shutil.move(str(export), str(moved_export))
    export = moved_export
    metadata = WORK / "metadata.csv"
    write_csv(
        metadata,
        [
            {"sample_id": "Bolivia_001", "event_id": "Bolivia", "latitude": -16.5, "longitude": -68.1},
            {"sample_id": "Pakistan_001", "event_id": "Pakistan", "latitude": 30.2, "longitude": 71.5},
        ],
    )
    calibration = WORK / "block_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.common_spatial_block_calibration.v2",
                "validity": "valid",
                "all_models_passed": True,
                "selection_data": "validation_only",
                "selected_cell_km": 50.0,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = WORK / "model.ckpt"
    checkpoint.write_bytes(b"verified-test-checkpoint")
    pretraining_checkpoint = WORK / "pretraining.pt"
    pretraining_checkpoint.write_bytes(b"verified-pretraining-checkpoint")

    bundle = finalize_sen1floods11_segmentation(
        export,
        WORK / "formal",
        model_name="terramind_s2",
        checkpoint_path=checkpoint,
        pretraining_checkpoint_path=pretraining_checkpoint,
        pretraining_checkpoint_sha256=file_sha256(pretraining_checkpoint),
        protocol_path="configs/geobwer/sen1floods11.yaml",
        block_calibration_path=calibration,
        metadata_csv=metadata,
        sensor_mode="S2",
        terratorch_version="1.2.5-test",
    )
    audit = read_csv_rows(bundle.audit_table)
    assert len(audit) == 2
    assert {row["event_id"] for row in audit} == {"Bolivia", "Pakistan"}
    assert all(row["spatial_block_id"].startswith("cea_50km_") for row in audit)
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    assert manifest["protocol"]["metadata"]["spatial_block_calibrated"] == "true"
    assert manifest["protocol"]["metadata"]["small_cluster_calibrated"] == "true"
def test_supervised_smoke_subset_retains_multiple_event_groups():
    from rsfm_fairness_audit.sen1_supervised_campaign import (
        _diagnostic_prefix_subset,
    )

    ordered = [
        *(f"Bolivia_{index:03d}" for index in range(8)),
        *(f"Pakistan_{index:03d}" for index in range(8)),
        *(f"Somalia_{index:03d}" for index in range(8)),
    ]
    selected = _diagnostic_prefix_subset(
        ordered,
        8,
        require_multiple_groups=True,
    )
    assert len(selected) == 8
    assert len({value.split("_", 1)[0] for value in selected}) == 3


def test_supervised_smoke_subset_rejects_one_event_training_data():
    from rsfm_fairness_audit.sen1_supervised_campaign import (
        Sen1SupervisedCampaignError,
        _diagnostic_prefix_subset,
    )

    with pytest.raises(Sen1SupervisedCampaignError, match="at least two event groups"):
        _diagnostic_prefix_subset(
            [f"Bolivia_{index:03d}" for index in range(8)],
            8,
            require_multiple_groups=True,
        )
