from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.sen1floods11_formal import (
    Sen1FormalizationError,
    finalize_sen1_probability_export,
    finalize_sen1floods11_segmentation,
    load_sen1_probability_units,
    write_sen1_probability_export,
)
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


def _all_ignore_fixture(root: Path, *, both_all_ignore: bool = False) -> tuple[Path, Path]:
    export = write_sen1_probability_export(
        root / "mixed_export",
        probabilities=[
            np.asarray([[0.1, 0.8], [0.7, 0.2]], dtype=np.float32),
            np.asarray([[0.3, 0.6], [0.4, 0.9]], dtype=np.float32),
        ],
        targets=[
            (
                np.full((2, 2), -1, dtype=np.int16)
                if both_all_ignore
                else np.asarray([[0, 1], [1, -1]], dtype=np.int16)
            ),
            np.full((2, 2), -1, dtype=np.int16),
        ],
        filenames=[
            "Ghana_142312_S2Hand.tif",
            "Ghana_5079_S2Hand.tif",
        ],
    )
    metadata = root / "complete_test_metadata.csv"
    write_csv(
        metadata,
        [
            {
                "sample_id": "Ghana_142312",
                "event_id": "Ghana",
                "latitude": 6.31,
                "longitude": -1.72,
                "country": "Ghana",
            },
            {
                "sample_id": "Ghana_5079",
                "event_id": "Ghana",
                "latitude": 6.32,
                "longitude": -1.71,
                "country": "Ghana",
            },
        ],
    )
    return export, metadata


def test_formal_loader_preserves_mixed_valid_and_ghana_style_all_ignore():
    export, metadata = _all_ignore_fixture(WORK)

    rows, probabilities, targets, valid = load_sen1_probability_units(
        export,
        metadata_csv=metadata,
    )

    assert [row["sample_id"] for row in rows] == ["Ghana_142312", "Ghana_5079"]
    assert [row["valid_pixel_count"] for row in rows] == [3, 0]
    assert [row["label_support_status"] for row in rows] == [
        "identified",
        "all_ignore",
    ]
    assert [int(mask.sum()) for mask in valid] == [3, 0]
    assert np.array_equal(targets[1], np.full((2, 2), -1, dtype=np.int16))
    assert len(probabilities) == 2


def test_formal_loader_rejects_split_when_every_chip_is_all_ignore():
    export, metadata = _all_ignore_fixture(WORK, both_all_ignore=True)

    with pytest.raises(
        Sen1FormalizationError,
        match="no valid hand-labeled pixels across the complete split",
    ):
        load_sen1_probability_units(export, metadata_csv=metadata)


def test_formal_loader_coordinate_failure_is_independent_of_all_ignore():
    export, metadata = _all_ignore_fixture(WORK)
    rows = read_csv_rows(metadata)
    write_csv(metadata, [rows[0]])

    with pytest.raises(
        Sen1FormalizationError,
        match="Cannot recover coordinates for chip=Ghana_5079",
    ):
        load_sen1_probability_units(export, metadata_csv=metadata)


def test_formal_loader_rejects_values_outside_frozen_label_contract():
    export, metadata = _all_ignore_fixture(WORK)
    index_row = json.loads(
        next((export / "index_parts").glob("*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    artifact_path = export / index_row["probability_path"]
    with np.load(artifact_path, allow_pickle=False) as artifact:
        payload = {key: np.asarray(artifact[key]) for key in artifact.files}
    payload["target"] = np.asarray([[0, 1], [2, -1]], dtype=np.int16)
    np.savez_compressed(artifact_path, **payload)

    with pytest.raises(
        Sen1FormalizationError,
        match="outside the frozen Sen1Floods11 label contract",
    ):
        load_sen1_probability_units(export, metadata_csv=metadata)


def test_formalization_excludes_all_ignore_only_from_risk_estimand():
    export, metadata = _all_ignore_fixture(WORK)
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

    bundle = finalize_sen1_probability_export(
        export,
        WORK / "formal",
        model_name="fixture_model",
        protocol_path="configs/geobwer/sen1floods11.yaml",
        block_calibration_path=calibration,
        model_lineage={"model": "fixture_model"},
        dataset_lineage={
            "dataset": "Sen1Floods11-v1.1-HandLabeled",
            "split": "test",
        },
        metadata_csv=metadata,
    )

    audit = read_csv_rows(bundle.audit_table)
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    assert [row["sample_id"] for row in audit] == ["Ghana_142312"]
    assert manifest["dataset_lineage"]["sample_ids"] == [
        "Ghana_142312",
        "Ghana_5079",
    ]
    assert manifest["dataset_lineage"]["formal_audit_sample_ids"] == [
        "Ghana_142312"
    ]
    assert manifest["dataset_lineage"]["source_split_sample_count"] == 2
    assert manifest["dataset_lineage"]["auditable_sample_count"] == 1
    assert manifest["dataset_lineage"]["all_ignore_sample_count"] == 1
    assert manifest["dataset_lineage"]["all_ignore_sample_ids"] == ["Ghana_5079"]


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
