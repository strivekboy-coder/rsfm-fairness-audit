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
    calibrate_common_sen1_spatial_blocks,
    combine_sen1_evaluation_exports,
    finalize_sen1_probability_export,
    finalize_sen1floods11_segmentation,
    load_sen1_probability_units,
    write_sen1_descriptive_probability_report,
    write_sen1_evaluation_split_report,
    write_sen1_probability_export,
)
from rsfm_fairness_audit.bwer_inference import (
    BlockCandidateCalibration,
    SpatialBlockCalibration,
    SpatialRangeEstimate,
)
from rsfm_fairness_audit.bwer_protocol import Validity
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


def test_paraguay_complete_s1_missing_but_valid_label_remains_in_risk_table():
    export = write_sen1_probability_export(
        WORK / "paraguay_missing_s1_export",
        probabilities=[
            np.asarray([[0.8, 0.9], [0.7, 0.6]], dtype=np.float32),
        ],
        targets=[
            np.asarray([[1, 1], [-1, 1]], dtype=np.int16),
        ],
        filenames=["Paraguay_34417_S1Hand.tif"],
    )
    metadata = WORK / "paraguay_metadata.csv"
    write_csv(
        metadata,
        [
            {
                "sample_id": "Paraguay_34417",
                "event_id": "Paraguay",
                "latitude": -23.4,
                "longitude": -58.4,
                "country": "Paraguay",
            }
        ],
    )
    calibration = WORK / "paraguay_block_calibration.json"
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
        WORK / "paraguay_formal",
        model_name="s1_fixture",
        protocol_path="configs/geobwer/sen1floods11.yaml",
        block_calibration_path=calibration,
        model_lineage={
            "imputation_policy": "official_train_band_mean_normalized_zero",
            "fully_missing_modalities": {"Paraguay_34417": ["S1"]},
        },
        dataset_lineage={
            "dataset": "Sen1Floods11-v1.1-HandLabeled",
            "split": "standard_test",
        },
        metadata_csv=metadata,
    )
    audit = read_csv_rows(bundle.audit_table)
    assert len(audit) == 1
    assert audit[0]["sample_id"] == "Paraguay_34417"
    assert np.isfinite(float(audit[0]["risk"]))


def _official_evaluation_exports(root: Path) -> tuple[Path, Path, Path]:
    standard_names = [
        f"Event{event:02d}_test_{index:03d}_S2Hand.tif"
        for event in range(10)
        for index in range(9)
    ]
    bolivia_names = [
        f"Bolivia_holdout_{index:03d}_S2Hand.tif" for index in range(15)
    ]

    def _write(path: Path, names: list[str]) -> Path:
        return write_sen1_probability_export(
            path,
            probabilities=[
                np.asarray([[0.2, 0.8], [0.7, 0.1]], dtype=np.float32)
                for _ in names
            ],
            targets=[
                np.asarray([[0, 1], [1, -1]], dtype=np.int16)
                for _ in names
            ],
            filenames=names,
        )

    metadata_rows = []
    for offset, name in enumerate([*standard_names, *bolivia_names]):
        sample_id = name.removesuffix("_S2Hand.tif")
        event_id = sample_id.split("_", 1)[0]
        metadata_rows.append(
            {
                "sample_id": sample_id,
                "event_id": event_id,
                "latitude": -20.0 + (offset % 40),
                "longitude": -120.0 + offset,
                "country": event_id,
            }
        )
    metadata = root / "evaluation_metadata.csv"
    write_csv(metadata, metadata_rows)
    return (
        _write(root / "standard_export", standard_names),
        _write(root / "bolivia_export", bolivia_names),
        metadata,
    )


def test_combined_evaluation_preserves_90_plus_15_and_eleven_event_contract():
    standard, bolivia, metadata = _official_evaluation_exports(WORK)
    combined = combine_sen1_evaluation_exports(
        standard,
        bolivia,
        WORK / "combined_export",
    )
    manifest = json.loads(
        (combined / "combined_export_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evaluation_sample_count"] == 105
    assert manifest["standard_test_count"] == 90
    assert manifest["bolivia_holdout_count"] == 15
    assert manifest["combined_event_count"] == 11
    assert manifest["sample_overlap"] == 0
    assert manifest["no_training_or_calibration_leakage"] is True

    units, _probabilities, _targets, _valid = load_sen1_probability_units(
        combined,
        metadata_csv=metadata,
    )
    roles = [row["evaluation_split_role"] for row in units]
    assert roles.count("standard_test") == 90
    assert roles.count("bolivia_holdout") == 15
    assert len({row["event_id"] for row in units}) == 11

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
    standard_bundle = finalize_sen1_probability_export(
        standard,
        WORK / "standard_formal",
        model_name="fixture_model",
        protocol_path="configs/geobwer/sen1floods11.yaml",
        block_calibration_path=calibration,
        model_lineage={"model": "fixture_model"},
        dataset_lineage={
            "dataset": "Sen1Floods11-v1.1-HandLabeled",
            "split_protocol": "official_252_89_90_plus_15_bolivia_holdout",
            "no_training_or_calibration_leakage": True,
        },
        metadata_csv=metadata,
        split="standard_test",
        evaluation_split_role="standard_test",
    )
    bolivia_bundle = finalize_sen1_probability_export(
        bolivia,
        WORK / "bolivia_formal",
        model_name="fixture_model",
        protocol_path="configs/geobwer/sen1floods11.yaml",
        block_calibration_path=calibration,
        model_lineage={"model": "fixture_model"},
        dataset_lineage={
            "dataset": "Sen1Floods11-v1.1-HandLabeled",
            "split_protocol": "official_252_89_90_plus_15_bolivia_holdout",
            "no_training_or_calibration_leakage": True,
        },
        metadata_csv=metadata,
        split="bolivia_holdout",
        evaluation_split_role="bolivia_holdout",
    )
    bundle = finalize_sen1_probability_export(
        combined,
        WORK / "combined_formal",
        model_name="fixture_model",
        protocol_path="configs/geobwer/sen1floods11.yaml",
        block_calibration_path=calibration,
        model_lineage={"model": "fixture_model"},
        dataset_lineage={
            "dataset": "Sen1Floods11-v1.1-HandLabeled",
            "split_protocol": "official_252_89_90_plus_15_bolivia_holdout",
            "no_training_or_calibration_leakage": True,
        },
        metadata_csv=metadata,
        evaluation_split_role="combined_held_out",
    )
    formal_manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    lineage = formal_manifest["dataset_lineage"]
    assert lineage["evaluation_sample_count"] == 105
    assert lineage["standard_test_count"] == 90
    assert lineage["bolivia_holdout_count"] == 15
    assert lineage["no_training_or_calibration_leakage"] is True
    assert set(lineage["evaluation_event_sets"]["bolivia_holdout"]) == {
        "Bolivia"
    }
    report_path = write_sen1_evaluation_split_report(
        WORK / "evaluation_split_report.json",
        standard_test_bundle=standard_bundle,
        bolivia_holdout_bundle=bolivia_bundle,
        combined_held_out_bundle=bundle,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["views"]["standard_test"]["source_sample_count"] == 90
    assert report["views"]["bolivia_holdout"]["source_sample_count"] == 15
    assert report["views"]["combined_held_out"]["source_sample_count"] == 105
    assert "not identified from one event" in report["interpretation"][
        "bolivia_holdout"
    ]


def test_combined_evaluation_rejects_standard_test_without_ten_events():
    standard, bolivia, _metadata = _official_evaluation_exports(WORK)
    index_path = standard / "index_parts" / "part-000000.jsonl"
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows):
        row["filename"] = f"Event00_relabelled_{index:03d}_S2Hand.tif"
    index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(
        Sen1FormalizationError,
        match="exactly 10 non-Bolivia events",
    ):
        combine_sen1_evaluation_exports(
            standard,
            bolivia,
            WORK / "invalid_combined_export",
        )


def test_combined_evaluation_resume_rejects_source_signature_drift():
    standard, bolivia, _metadata = _official_evaluation_exports(WORK)
    combined = combine_sen1_evaluation_exports(
        standard,
        bolivia,
        WORK / "combined_export",
    )
    assert combined.is_dir()
    first_row = json.loads(
        (standard / "index_parts" / "part-000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    artifact = standard / first_row["probability_path"]
    with np.load(artifact, allow_pickle=False) as payload:
        target = np.asarray(payload["target"])
    probabilities = np.full((2, *target.shape), 0.5, dtype=np.float32)
    np.savez_compressed(
        artifact,
        probabilities=probabilities,
        target=target,
    )
    with pytest.raises(
        Sen1FormalizationError,
        match="contract is incompatible",
    ):
        combine_sen1_evaluation_exports(
            standard,
            bolivia,
            combined,
        )


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


def _synthetic_calibration_export(root: Path, model: str) -> tuple[Path, Path]:
    export = write_sen1_probability_export(
        root / model,
        probabilities=[
            np.full((2, 2), 0.2, dtype=np.float32),
            np.full((2, 2), 0.8, dtype=np.float32),
            np.full((2, 2), 0.6, dtype=np.float32),
        ],
        targets=[
            np.asarray([[0, 0], [1, 1]], dtype=np.int16),
            np.asarray([[1, 1], [0, 0]], dtype=np.int16),
            np.asarray([[1, 0], [1, 0]], dtype=np.int16),
        ],
        filenames=[
            "Ghana_001_S2Hand.tif",
            "Pakistan_001_S2Hand.tif",
            "Pakistan_002_S2Hand.tif",
        ],
    )
    metadata = root / "calibration_metadata.csv"
    if not metadata.exists():
        write_csv(
            metadata,
            [
                {"sample_id": "Ghana_001", "event_id": "Ghana", "latitude": 6.0, "longitude": -1.0},
                {"sample_id": "Pakistan_001", "event_id": "Pakistan", "latitude": 30.0, "longitude": 70.0},
                {"sample_id": "Pakistan_002", "event_id": "Pakistan", "latitude": 30.2, "longitude": 70.2},
            ],
        )
    return export, metadata


def _fake_spatial_calibration(*_args, pass_cell: float | None, **kwargs):
    candidates = []
    for cell in kwargs["candidate_cell_km"]:
        passed = pass_cell is not None and float(cell) == float(pass_cell)
        candidates.append(
            BlockCandidateCalibration(
                cell_km=float(cell),
                cluster_count=6,
                range_adequate=passed,
                valid_simulations=int(kwargs["n_simulations"]),
                null_coverage=0.96 if passed else 0.70,
                null_coverage_ci_low=0.91 if passed else 0.60,
                null_coverage_ci_high=0.99 if passed else 0.79,
                false_positive_rate=0.01 if passed else 0.20,
                false_positive_ci_low=0.0 if passed else 0.12,
                false_positive_ci_high=0.04 if passed else 0.30,
                moderate_tail_power=0.85,
                moderate_tail_power_ci_low=0.80,
                moderate_tail_power_ci_high=0.90,
                power_target_met=True,
                coverage_gate=0.93,
                false_positive_gate=0.06,
                passes=passed,
            )
        )
    return SpatialBlockCalibration(
        validity=(Validity.VALID if pass_cell is not None else Validity.SPATIAL_BLOCK_NOT_CALIBRATED),
        selected_cell_km=pass_cell,
        range_estimate=SpatialRangeEstimate(40.0, 0.2, 3, (), ()),
        candidates=tuple(candidates),
        simulation_repetitions=int(kwargs["n_simulations"]),
    )


def test_common_spatial_calibration_records_explicit_panel_scope(monkeypatch):
    first, metadata = _synthetic_calibration_export(WORK, "model_a")
    second, _ = _synthetic_calibration_export(WORK, "model_b")
    monkeypatch.setattr(
        "rsfm_fairness_audit.sen1floods11_formal.calibrate_spatial_block_scale",
        lambda *args, **kwargs: _fake_spatial_calibration(
            *args, pass_cell=50.0, **kwargs
        ),
    )
    result = calibrate_common_sen1_spatial_blocks(
        {"model_a": first, "model_b": second},
        WORK / "common.json",
        metadata_csv=metadata,
        candidate_cell_km=(25.0, 50.0),
        n_simulations=100,
        n_bootstrap=10,
        calibration_panel_scope="synthetic_two_model_panel",
        expected_model_names=("model_a", "model_b"),
    )
    assert result["validity"] == "valid"
    assert result["selected_cell_km"] == 50.0
    assert result["calibration_panel_scope"] == "synthetic_two_model_panel"
    assert result["model_count"] == 2


def test_common_spatial_calibration_writes_auditable_invalid_branch(monkeypatch):
    first, metadata = _synthetic_calibration_export(WORK, "model_a")
    second, _ = _synthetic_calibration_export(WORK, "model_b")
    monkeypatch.setattr(
        "rsfm_fairness_audit.sen1floods11_formal.calibrate_spatial_block_scale",
        lambda *args, **kwargs: _fake_spatial_calibration(
            *args, pass_cell=None, **kwargs
        ),
    )
    failure = WORK / "calibration_failure_report.json"
    result = calibrate_common_sen1_spatial_blocks(
        {"model_a": first, "model_b": second},
        WORK / "common.json",
        metadata_csv=metadata,
        candidate_cell_km=(25.0, 50.0),
        n_simulations=100,
        n_bootstrap=10,
        calibration_panel_scope="synthetic_two_model_panel",
        expected_model_names=("model_a", "model_b"),
        failure_output_json=failure,
        return_invalid=True,
    )
    assert not (WORK / "common.json").exists()
    assert result["status"] == "calibration_invalid"
    assert result["validation_only"] is True
    assert result["common_passing_cells"] == []
    assert result["inferential_geobwer_permitted"] is False
    assert set(result["failures_by_cell"]["25.0"]["failed_models"]) == {
        "model_a",
        "model_b",
    }
    assert failure.is_file()
    descriptive = write_sen1_descriptive_probability_report(
        first,
        WORK / "descriptive.json",
        model_name="model_a",
        split_role="standard_test",
        metadata_csv=metadata,
    )
    descriptive_payload = json.loads(descriptive.read_text(encoding="utf-8"))
    assert descriptive_payload["status"] == "descriptive_only"
    assert descriptive_payload["inferential_geobwer_run"] is False
    assert descriptive_payload["bootstrap_run"] is False


def test_common_spatial_calibration_rejects_wrong_panel_before_simulation(monkeypatch):
    first, metadata = _synthetic_calibration_export(WORK, "model_a")
    called = False

    def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("calibration should not run")

    monkeypatch.setattr(
        "rsfm_fairness_audit.sen1floods11_formal.calibrate_spatial_block_scale",
        should_not_run,
    )
    with pytest.raises(Sen1FormalizationError, match="frozen scope"):
        calibrate_common_sen1_spatial_blocks(
            {"model_a": first},
            WORK / "common.json",
            metadata_csv=metadata,
            expected_model_names=("model_a", "model_b"),
        )
    assert called is False
