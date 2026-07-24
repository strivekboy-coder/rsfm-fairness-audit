from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.formal_outputs import (
    write_multiclass_bundle,
    write_multilabel_bundle,
    write_segmentation_bundle,
)
from rsfm_fairness_audit.geobwer_extensions import (
    ExtensionAuditError,
    run_multiclass_spatial_upgrade,
    run_multiclass_uncertainty_suite,
    run_multilabel_uncertainty_suite,
    run_segmentation_uncertainty_suite,
)
from rsfm_fairness_audit.io import read_csv_rows
from rsfm_fairness_audit.spatial_conformal import SpatialConformalConfig


def _protocol(task: str, loss: str) -> BWERProtocol:
    return BWERProtocol(
        beta=0.25,
        beta_profile=(0.25, 0.5),
        inference_method="none",
        group_variable="country",
        loss_name=loss,
        task_adapter=task,
    )


def _rows(count: int) -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"sample-{index:03d}",
            "independent_unit_id": f"sample-{index:03d}",
            "country": "A" if index % 2 == 0 else "B",
        }
        for index in range(count)
    ]


def _test_root(name: str) -> Path:
    path = Path("work") / "test_runs" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_multiclass_uncertainty_suite_runs_from_calibration_to_geobwer() -> None:
    tmp_path = _test_root("multiclass_extensions")
    rng = np.random.default_rng(7)
    protocol = _protocol("multiclass", "zero_one_loss")
    calibration_probabilities = rng.dirichlet([3.0, 2.0, 1.0], size=40)
    calibration_targets = np.argmax(calibration_probabilities, axis=1)
    calibration = tmp_path / "calibration.npz"
    np.savez_compressed(
        calibration,
        probabilities=calibration_probabilities,
        targets=calibration_targets,
        sample_id=np.asarray([f"calibration-{index:03d}" for index in range(40)]),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    test_probabilities = rng.dirichlet([2.0, 2.0, 2.0], size=40)
    test_targets = np.argmax(test_probabilities, axis=1)
    formal = write_multiclass_bundle(
        tmp_path / "formal",
        sample_rows=_rows(40),
        probabilities=test_probabilities,
        targets=test_targets,
        class_names=("a", "b", "c"),
        dataset="demo",
        model="demo-model",
        split="test",
        protocol=protocol,
        model_lineage={"checkpoint_sha256": "demo"},
        dataset_lineage={"manifest_sha256": "demo"},
    )
    artifacts = run_multiclass_uncertainty_suite(
        calibration,
        formal.output_dir,
        tmp_path / "extensions",
        protocol=protocol,
        group_columns=("country",),
        conformal_methods=("lac", "aps", "raps"),
        selective_coverages=(0.8,),
        n_bootstrap=20,
    )
    assert artifacts["summary"].exists()
    assert artifacts["conformal_lac_summary"].exists()
    assert artifacts["selective_080_summary"].exists()
    assert artifacts["selective_080_group_coverage"].exists()
    summary_rows = read_csv_rows(artifacts["summary"])
    conformal_row = next(row for row in summary_rows if row["extension"] == "conformal_lac")
    assert "tail_target_violation" in conformal_row
    selective_row = next(row for row in summary_rows if row["extension"] == "selective_080")
    assert "minimum_group_coverage" in selective_row
    assert "selective_geobwer_identified" in selective_row
    shutil.rmtree(tmp_path)


def test_multiclass_uncertainty_suite_runs_spatial_comparator_and_efficiency_audit() -> None:
    tmp_path = _test_root("multiclass_spatial_extensions")
    rng = np.random.default_rng(19)
    protocol = _protocol("multiclass", "zero_one_loss")
    calibration_probabilities = rng.dirichlet([3.0, 2.0, 1.0], size=60)
    calibration_targets = np.argmax(calibration_probabilities, axis=1)
    latitude = np.concatenate((rng.normal(45.0, 0.2, 30), rng.normal(-20.0, 0.2, 30)))
    longitude = np.concatenate((rng.normal(8.0, 0.2, 30), rng.normal(30.0, 0.2, 30)))
    calibration = tmp_path / "calibration.npz"
    np.savez_compressed(
        calibration,
        probabilities=calibration_probabilities,
        targets=calibration_targets,
        latitude=latitude,
        longitude=longitude,
        sample_id=np.asarray([f"calibration-{index:03d}" for index in range(60)]),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    test_probabilities = rng.dirichlet([2.0, 2.0, 2.0], size=40)
    test_targets = np.argmax(test_probabilities, axis=1)
    rows = _rows(40)
    for index, row in enumerate(rows):
        row["latitude"] = 45.0 + 0.01 * index if index < 20 else -20.0 + 0.01 * index
        row["longitude"] = 8.0 if index < 20 else 30.0
    formal = write_multiclass_bundle(
        tmp_path / "formal",
        sample_rows=rows,
        probabilities=test_probabilities,
        targets=test_targets,
        class_names=("a", "b", "c"),
        dataset="demo",
        model="demo-model",
        split="test",
        protocol=protocol,
        model_lineage={"checkpoint_sha256": "demo"},
        dataset_lineage={"manifest_sha256": "demo"},
    )
    spatial_config = SpatialConformalConfig(
        candidate_bandwidth_km=(50.0, 200.0, 1000.0),
        minimum_calibration_samples=40,
        minimum_effective_sample_size=10.0,
        calibration_anchor_limit=60,
        maximum_neighbors=60,
    )
    artifacts = run_multiclass_uncertainty_suite(
        calibration,
        formal.output_dir,
        tmp_path / "extensions",
        protocol=protocol,
        group_columns=("country",),
        conformal_methods=("lac",),
        selective_coverages=(),
        spatial_conformal_config=spatial_config,
        n_bootstrap=20,
    )
    assert artifacts["spatial_localization_preflight"].exists()
    assert artifacts["conformal_lac_efficiency_summary"].exists()
    assert artifacts["geo_kernel_conformal_lac_summary"].exists()
    assert artifacts["geo_kernel_conformal_lac_efficiency_summary"].exists()
    summary = read_csv_rows(artifacts["summary"])
    spatial = next(row for row in summary if row["extension"] == "geo_kernel_conformal_lac")
    assert spatial["role"] == "empirical_spatial_localization_comparator"
    shutil.rmtree(tmp_path)


def test_spatial_upgrade_reuses_probabilities_and_joins_coordinates_by_sample_id() -> None:
    tmp_path = _test_root("multiclass_spatial_upgrade")
    rng = np.random.default_rng(23)
    protocol = _protocol("multiclass", "zero_one_loss")
    calibration_probabilities = rng.dirichlet([3.0, 2.0], size=60)
    calibration_targets = np.argmax(calibration_probabilities, axis=1)
    calibration = tmp_path / "calibration.npz"
    sample_ids = np.asarray([f"calibration-{index:03d}" for index in range(60)])
    np.savez_compressed(
        calibration,
        probabilities=calibration_probabilities,
        targets=calibration_targets,
        sample_id=sample_ids,
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "sample_id,latitude,longitude\n"
        + "\n".join(
            f"{sample_id},{45.0 + (index % 2) * -65.0},{8.0 + (index % 2) * 22.0}"
            for index, sample_id in enumerate(sample_ids.tolist())
        )
        + "\n",
        encoding="utf-8",
    )
    test_probabilities = rng.dirichlet([2.0, 2.0], size=40)
    test_targets = np.argmax(test_probabilities, axis=1)
    rows = _rows(40)
    for index, row in enumerate(rows):
        row["latitude"] = 45.0 if index % 2 == 0 else -20.0
        row["longitude"] = 8.0 if index % 2 == 0 else 30.0
    formal = write_multiclass_bundle(
        tmp_path / "formal",
        sample_rows=rows,
        probabilities=test_probabilities,
        targets=test_targets,
        class_names=("a", "b"),
        dataset="demo",
        model="demo-model",
        split="test",
        protocol=protocol,
        model_lineage={"checkpoint_sha256": "demo"},
        dataset_lineage={"manifest_sha256": "demo"},
    )
    config = SpatialConformalConfig(
        candidate_bandwidth_km=(50.0, 200.0, 1000.0),
        minimum_calibration_samples=40,
        minimum_effective_sample_size=10.0,
        calibration_anchor_limit=60,
        maximum_neighbors=60,
    )
    artifacts = run_multiclass_spatial_upgrade(
        calibration,
        metadata,
        formal.output_dir,
        tmp_path / "upgrade",
        protocol=protocol,
        group_columns=("country",),
        conformal_methods=("lac",),
        spatial_conformal_config=config,
        n_bootstrap=20,
    )
    with np.load(artifacts["calibration_probabilities"], allow_pickle=False) as enriched:
        assert np.array_equal(enriched["probabilities"], calibration_probabilities)
        assert enriched["latitude"].shape == (60,)
        assert enriched["longitude"].shape == (60,)
    assert artifacts["geo_kernel_conformal_lac_summary"].exists()
    shutil.rmtree(tmp_path)


def test_multilabel_and_segmentation_crc_use_calibration_only() -> None:
    tmp_path = _test_root("structured_extensions")
    rng = np.random.default_rng(9)
    multilabel_protocol = _protocol("multilabel", "hamming_loss")
    calibration_probs = rng.uniform(0.05, 0.95, size=(40, 4))
    calibration_targets = (calibration_probs >= 0.35).astype(np.int8)
    calibration = tmp_path / "multilabel_calibration.npz"
    np.savez_compressed(
        calibration,
        probabilities=calibration_probs,
        targets=calibration_targets,
        sample_id=np.asarray([f"calibration-{index:03d}" for index in range(40)]),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    test_probs = rng.uniform(0.05, 0.95, size=(40, 4))
    test_targets = (test_probs >= 0.4).astype(np.int8)
    formal = write_multilabel_bundle(
        tmp_path / "multilabel_formal",
        sample_rows=_rows(40),
        probabilities=test_probs,
        targets=test_targets,
        class_names=("a", "b", "c", "d"),
        dataset="demo",
        model="demo-model",
        split="test",
        protocol=multilabel_protocol,
        model_lineage={"checkpoint_sha256": "demo"},
        dataset_lineage={"manifest_sha256": "demo"},
        threshold=(0.4, 0.4, 0.4, 0.4),
    )
    multilabel_artifacts = run_multilabel_uncertainty_suite(
        calibration,
        formal.output_dir,
        tmp_path / "multilabel_extensions",
        protocol=multilabel_protocol,
        selective_coverages=(0.8,),
        crc_alpha=0.2,
        n_bootstrap=20,
    )
    assert multilabel_artifacts["crc_summary"].exists()

    segmentation_protocol = _protocol("segmentation", "one_minus_iou")
    maps = [rng.uniform(0.05, 0.95, size=(4, 4)).astype(np.float32) for _ in range(40)]
    masks = [(probability >= 0.3).astype(np.int16) for probability in maps]
    valid = [np.ones((4, 4), dtype=bool) for _ in maps]
    # Confirm ignore pixels are carried by the formal output and excluded from CRC.
    masks[0][0, 0] = 255
    valid[0][0, 0] = False
    seg_formal = write_segmentation_bundle(
        tmp_path / "segmentation_formal",
        sample_rows=_rows(40),
        positive_probability_maps=maps,
        target_masks=masks,
        valid_masks=valid,
        dataset="demo",
        model="demo-model",
        split="test",
        protocol=segmentation_protocol,
        model_lineage={"checkpoint_sha256": "demo"},
        dataset_lineage={"manifest_sha256": "demo"},
    )
    segmentation_artifacts = run_segmentation_uncertainty_suite(
        maps,
        [(probability >= 0.25).astype(np.int8) for probability in maps],
        seg_formal.output_dir,
        tmp_path / "segmentation_extensions",
        protocol=segmentation_protocol,
        group_columns=("country",),
        calibration_valid_masks=valid,
        calibration_sample_ids=[f"calibration-{index:03d}" for index in range(40)],
        crc_alpha=0.2,
        n_bootstrap=20,
    )
    assert segmentation_artifacts["summary"].exists()
    shutil.rmtree(tmp_path)


def test_formal_extension_rejects_unlabelled_calibration_artifact() -> None:
    tmp_path = _test_root("bad_calibration")
    probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, probabilities=probabilities, targets=np.asarray([0, 1]))
    protocol = _protocol("multiclass", "zero_one_loss")
    formal = write_multiclass_bundle(
        tmp_path / "formal",
        sample_rows=_rows(2),
        probabilities=probabilities,
        targets=(0, 1),
        class_names=("a", "b"),
        dataset="demo",
        model="demo-model",
        split="test",
        protocol=protocol,
        model_lineage={"checkpoint_sha256": "demo"},
        dataset_lineage={"manifest_sha256": "demo"},
    )
    with pytest.raises(ExtensionAuditError, match="split_role"):
        run_multiclass_uncertainty_suite(
            bad,
            formal.output_dir,
            tmp_path / "extensions",
            protocol=protocol,
            group_columns=("country",),
            conformal_methods=("lac",),
            selective_coverages=(0.5,),
            n_bootstrap=20,
        )
    shutil.rmtree(tmp_path)


def test_formal_extension_rejects_calibration_test_sample_overlap() -> None:
    tmp_path = _test_root("calibration_leakage")
    probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    calibration = tmp_path / "calibration.npz"
    np.savez_compressed(
        calibration,
        probabilities=probabilities,
        targets=np.asarray([0, 1]),
        sample_id=np.asarray(["sample-000", "calibration-only"]),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    protocol = _protocol("multiclass", "zero_one_loss")
    formal = write_multiclass_bundle(
        tmp_path / "formal",
        sample_rows=_rows(2),
        probabilities=probabilities,
        targets=(0, 1),
        class_names=("a", "b"),
        dataset="demo",
        model="demo-model",
        split="test",
        protocol=protocol,
        model_lineage={"checkpoint_sha256": "demo"},
        dataset_lineage={"manifest_sha256": "demo"},
    )
    with pytest.raises(ExtensionAuditError, match="leakage"):
        run_multiclass_uncertainty_suite(
            calibration,
            formal.output_dir,
            tmp_path / "extensions",
            protocol=protocol,
            group_columns=("country",),
            conformal_methods=("lac",),
            selective_coverages=(0.5,),
            n_bootstrap=20,
        )
    shutil.rmtree(tmp_path)
