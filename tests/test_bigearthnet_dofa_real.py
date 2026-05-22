from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.adapters.bigearthnet import BigEarthNetDatasetAdapter, BigEarthNetDatasetError
from rsfm_fairness_audit.adapters.dofa import DOFAAdapter, DOFAConfigurationError
from rsfm_fairness_audit.cli import build_parser
from rsfm_fairness_audit.io import read_csv_rows
from rsfm_fairness_audit.pipeline import run_dummy_pipeline, run_real_pipeline
from rsfm_fairness_audit.preflight import run_real_preflight


class MockDOFAModel:
    def extract_embeddings(self, images: np.ndarray, wavelengths: list[float]) -> np.ndarray:
        features = np.concatenate(
            [
                images.mean(axis=(2, 3)),
                images.std(axis=(2, 3)),
            ],
            axis=1,
        )
        return features.astype(np.float32)


def _write_fixture(root: Path, count: int = 8, include_coordinates: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        image = np.full((9, 8, 8), fill_value=float(index % 2), dtype=np.float32)
        image += np.arange(9, dtype=np.float32)[:, None, None] * 0.01
        path = root / f"sample_{index:03d}_s2.npy"
        np.save(path, image)
        row = {
            "sample_id": f"BEN-{index:03d}",
            "label": index % 2,
            "label_vector": "[1, 0]" if index % 2 == 0 else "[0, 1]",
            "label_names": '["forest"]' if index % 2 == 0 else '["urban"]',
            "country": "germany" if index < count // 2 else "spain",
            "region": "germany" if index < count // 2 else "spain",
            "sensor": "S2",
            "split": "train" if index < count - 2 else "val",
            "s2_path": path.name,
        }
        if include_coordinates:
            row["latitude"] = 50.0 + index
            row["longitude"] = 8.0 + index
        rows.append(row)
    metadata_path = root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def test_bigearthnet_adapter_instantiates_without_dataset() -> None:
    adapter = BigEarthNetDatasetAdapter("not-present", subset_size=32)
    assert adapter.subset_size == 32


def test_bigearthnet_missing_path_error_is_clear() -> None:
    adapter = BigEarthNetDatasetAdapter("definitely-not-present")
    with pytest.raises(BigEarthNetDatasetError, match="data_root does not exist"):
        adapter.load_metadata()


def test_bigearthnet_subset_mode_and_deterministic_metadata() -> None:
    root = Path("outputs/test_bigearthnet_fixture")
    metadata_path = _write_fixture(root, count=8)
    adapter = BigEarthNetDatasetAdapter(root, metadata_path=metadata_path, subset_size=4, split="all", sensor_mode="S2")

    first = adapter.load_metadata()
    second = adapter.load_metadata()

    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len(first) == 4
    sample = adapter.load_sample(0)
    assert sample["image"].shape == (9, 8, 8)
    assert adapter.get_region(0) == "germany"
    assert adapter.get_sensor(0) == "S2"
    assert adapter.get_group_keys(0)["region_class"] == "germany::class_0"


def test_bigearthnet_subset_manifest_filters_metadata() -> None:
    root = Path("outputs/test_bigearthnet_manifest_fixture")
    metadata_path = _write_fixture(root, count=6)
    manifest_path = root / "subset.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        writer.writerows([{"sample_id": "BEN-001"}, {"sample_id": "BEN-004"}])

    adapter = BigEarthNetDatasetAdapter(root, metadata_path=metadata_path, subset_manifest_path=manifest_path)
    assert [row["sample_id"] for row in adapter.load_metadata()] == ["BEN-001", "BEN-004"]


def test_dofa_adapter_fails_without_configuration() -> None:
    adapter = DOFAAdapter(sensor_mode="S2")
    with pytest.raises(DOFAConfigurationError, match="DOFA is not configured"):
        adapter.load_model()


def test_dofa_config_validation_checks_wavelength_count() -> None:
    adapter = DOFAAdapter(sensor_mode="S2", expected_bands=9, wavelengths=[0.665, 0.56], model=MockDOFAModel())
    batch = {"samples": [{"image": np.zeros((9, 8, 8), dtype=np.float32)}], "metadata": [{"sample_id": "x"}]}
    with pytest.raises(DOFAConfigurationError, match="wavelength count"):
        adapter.preprocess(batch)


def test_dofa_missing_checkpoint_with_repo_path_is_clear() -> None:
    adapter = DOFAAdapter(repo_path=".", sensor_mode="S2")
    with pytest.raises(DOFAConfigurationError, match="requires checkpoint_path"):
        adapter.load_model()


def test_dofa_missing_repo_path_is_clear() -> None:
    adapter = DOFAAdapter(repo_path="definitely-not-a-dofa-repo", checkpoint_path="missing.pth")
    with pytest.raises(DOFAConfigurationError, match="repo_path does not exist"):
        adapter.load_model()


def test_dofa_checkpoint_requires_repo_path() -> None:
    checkpoint = Path("outputs/test_dofa_config/DOFA_ViT_base_e100.pth")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    adapter = DOFAAdapter(checkpoint_path=checkpoint)
    with pytest.raises(DOFAConfigurationError, match="requires repo_path"):
        adapter.load_model()


def test_dofa_from_config_file_keeps_download_disabled() -> None:
    adapter = DOFAAdapter.from_config_file("configs/models/dofa.yaml")
    assert adapter.model_variant == "vit_base_dofa"
    assert adapter.allow_torch_hub_download is True
    assert adapter.band_profile == "sentinel2_12_lccol"
    assert adapter.expected_bands == 12
    assert adapter.image_size == 224


def test_dofa_band_profile_validates_12_channel_inputs() -> None:
    adapter = DOFAAdapter.from_config_file("configs/models/dofa.yaml", model=MockDOFAModel())
    batch = {"samples": [{"image": np.zeros((12, 224, 224), dtype=np.float32)}], "metadata": [{"sample_id": "x"}]}
    processed = adapter.preprocess(batch)
    assert processed["images"].shape[1] == 12
    assert len(processed["wavelengths"]) == 12


def test_dofa_config_detects_normalization_length_mismatch() -> None:
    adapter = DOFAAdapter(
        sensor_mode="S2",
        expected_bands=12,
        wavelengths=[0.443] * 12,
        normalization_mean=[0.0] * 11,
        normalization_std=[1.0] * 12,
        model=MockDOFAModel(),
    )
    batch = {"samples": [{"image": np.zeros((12, 8, 8), dtype=np.float32)}], "metadata": [{"sample_id": "x"}]}
    with pytest.raises(DOFAConfigurationError, match="normalization_mean length"):
        adapter.preprocess(batch)


def test_dofa_input_scale_is_applied_before_normalization() -> None:
    adapter = DOFAAdapter(
        sensor_mode="S2",
        expected_bands=13,
        wavelengths=[0.443, 0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 0.945, 1.373, 1.61, 2.19],
        normalization_mean=[0.0] * 13,
        normalization_std=[1.0] * 13,
        input_scale=10000,
        model=MockDOFAModel(),
    )
    image = np.full((13, 4, 4), 3000.0, dtype=np.float32)
    processed = adapter.preprocess({"samples": [{"image": image}], "metadata": [{"sample_id": "x"}]})
    assert np.isclose(processed["images"].max(), 0.3)


def test_run_real_with_mock_dofa_writes_expected_artifacts() -> None:
    root = Path("outputs/test_real_pipeline_fixture")
    metadata_path = _write_fixture(root, count=8)
    dataset = BigEarthNetDatasetAdapter(root, metadata_path=metadata_path, subset_size=6, sensor_mode="S2")
    model = DOFAAdapter(sensor_mode="S2", model=MockDOFAModel())

    artifacts = run_real_pipeline(dataset, model, "outputs/test_real_pipeline", "bigearthnet", "dofa")

    assert artifacts["embeddings"].exists()
    assert artifacts["predictions"].exists()
    assert artifacts["region_matrix"].exists()
    assert artifacts["gap_table"].exists()
    assert artifacts["average_vs_worst"].exists()
    assert artifacts["sensor_heatmap"].exists()
    assert artifacts["report"].exists()
    assert "fairness_map" in artifacts
    assert artifacts["fairness_map"].exists()
    assert len(read_csv_rows(artifacts["predictions"])) == 6


def test_run_real_streaming_embeddings_writes_chunks() -> None:
    root = Path("outputs/test_real_pipeline_streaming_fixture")
    metadata_path = _write_fixture(root, count=8)
    dataset = BigEarthNetDatasetAdapter(root, metadata_path=metadata_path, subset_size=6, sensor_mode="S2")
    model = DOFAAdapter(sensor_mode="S2", model=MockDOFAModel())

    artifacts = run_real_pipeline(
        dataset,
        model,
        "outputs/test_real_pipeline_streaming",
        "bigearthnet",
        "dofa",
        chunk_size=2,
        streaming_embeddings=True,
    )

    chunks = list(Path("outputs/test_real_pipeline_streaming/embedding_chunks").glob("chunk_*.npz"))
    assert len(chunks) == 3
    assert artifacts["embeddings"].exists()
    assert artifacts["predictions"].exists()


def test_run_real_writes_fallback_map_when_coordinates_are_unverified() -> None:
    root = Path("outputs/test_real_pipeline_no_coords_fixture")
    metadata_path = _write_fixture(root, count=8, include_coordinates=False)
    dataset = BigEarthNetDatasetAdapter(root, metadata_path=metadata_path, subset_size=6, sensor_mode="S2")
    model = DOFAAdapter(sensor_mode="S2", model=MockDOFAModel())

    artifacts = run_real_pipeline(dataset, model, "outputs/test_real_pipeline_no_coords", "bigearthnet", "dofa")

    assert "fairness_map" in artifacts
    assert artifacts["fairness_map"].exists()
    assert artifacts["figures_fairness_map"].exists()
    assert artifacts["tables_fairness_matrix"].exists()


def test_run_real_command_exists() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-real",
            "--dataset",
            "bigearthnet",
            "--model",
            "dofa",
            "--data-root",
            "somewhere",
            "--subset-size",
            "32",
            "--model-config",
            "configs/models/dofa.yaml",
        ]
    )
    assert args.command == "run-real"
    assert str(args.model_config).endswith("dofa.yaml")


def test_check_real_command_exists() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "check-real",
            "--dataset",
            "bigearthnet",
            "--model",
            "dofa",
            "--model-config",
            "configs/models/dofa.yaml",
            "--data-root",
            "somewhere",
        ]
    )
    assert args.command == "check-real"


def test_preflight_reports_missing_dofa_configuration() -> None:
    root = Path("outputs/test_preflight_missing_config")
    _write_fixture(root, count=4)
    checks = run_real_preflight(
        model="dofa",
        dataset="bigearthnet",
        model_config="configs/models/dofa.yaml",
        data_root=root,
    )
    statuses = {check.name: check.status for check in checks}
    assert statuses["model_config"] == "pass"
    assert statuses["dofa_loading"] == "warn"
    assert statuses["data_root"] == "pass"
    assert statuses["bigearthnet_bands"] == "fail"


def test_preflight_accepts_torch_hub_mode_without_downloading() -> None:
    root = Path("outputs/test_preflight_hub")
    _write_fixture(root, count=4)
    config_path = root / "dofa_hub.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_variant: vit_base_dofa",
                "repo_path: null",
                "checkpoint_path: null",
                "torch_hub_repo: zhu-xlab/DOFA",
                "allow_torch_hub_download: true",
                "device: auto",
                "batch_size: 4",
                "input_modality: S2",
                "expected_bands: 9",
                "wavelength_list: [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]",
                "image_size: 224",
                "embedding_layer: forward_features",
            ]
        ),
        encoding="utf-8",
    )

    checks = run_real_preflight(
        model="dofa",
        dataset="bigearthnet",
        model_config=config_path,
        data_root=root,
    )
    statuses = {check.name: check.status for check in checks}
    assert statuses["dofa_loading"] == "warn"
    assert statuses["bigearthnet_sample"] == "pass"
    assert "dofa_checkpoint" not in statuses


def test_dummy_pipeline_still_runs_after_real_integration() -> None:
    artifacts = run_dummy_pipeline("outputs/test_dummy_after_real", num_samples=96, seed=5)
    assert artifacts["region_matrix"].exists()
