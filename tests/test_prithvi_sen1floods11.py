from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import scripts.prepare_sen1floods11_subset as prep
from rsfm_fairness_audit.adapters.prithvi import PrithviAdapter, PrithviConfigurationError
from rsfm_fairness_audit.adapters.sen1floods11 import Sen1Floods11DatasetAdapter
from rsfm_fairness_audit.cli import build_parser
from rsfm_fairness_audit.io import read_csv_rows
from rsfm_fairness_audit.pipeline import run_real_pipeline
from rsfm_fairness_audit.segmentation import run_segmentation_smoke


class MockPrithviModel:
    def extract_embeddings(self, images: np.ndarray) -> np.ndarray:
        return images.mean(axis=(1, 3, 4)).astype(np.float32)

    def extract_patch_features(self, images: np.ndarray) -> np.ndarray:
        return images.mean(axis=1).astype(np.float32)


def _write_prepared_fixture(root: Path, count: int = 6) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "chips").mkdir(exist_ok=True)
    (root / "masks").mkdir(exist_ok=True)
    rows = []
    for index in range(count):
        image = np.full((4, 6, 224, 224), index + 1, dtype=np.float32)
        mask = np.zeros((224, 224), dtype=np.int16)
        if index % 2:
            mask[:40, :40] = 1
        mask[-4:, :] = -1
        chip_path = root / "chips" / f"sample_{index:03d}.npz"
        mask_path = root / "masks" / f"sample_{index:03d}.npz"
        np.savez_compressed(chip_path, image=image)
        np.savez_compressed(mask_path, mask=mask)
        rows.append(
            {
                "sample_id": f"SEN-{index:03d}",
                "chip_path": str(chip_path.relative_to(root)),
                "mask_path": str(mask_path.relative_to(root)),
                "label": index % 2,
                "region": "Bolivia" if index < count // 2 else "India",
                "event": "Bolivia" if index < count // 2 else "India",
                "sensor": "S2",
                "split": "all",
            }
        )
    metadata_path = root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def test_prithvi_config_validation_rejects_wrong_shape() -> None:
    adapter = PrithviAdapter(expected_frames=1, model=MockPrithviModel())
    with pytest.raises(PrithviConfigurationError, match="expects 4 frames"):
        adapter.load_model()


def test_prithvi_repeats_single_timestamp_to_four_frames() -> None:
    adapter = PrithviAdapter(model=MockPrithviModel())
    batch = {
        "samples": [{"image": np.zeros((6, 224, 224), dtype=np.float32)}],
        "metadata": [{"sample_id": "x"}],
    }
    processed = adapter.preprocess(batch)
    assert processed["images"].shape == (1, 4, 6, 224, 224)


def test_sen1floods11_adapter_loads_prepared_samples() -> None:
    root = Path("outputs/test_sen1_adapter")
    _write_prepared_fixture(root)
    adapter = Sen1Floods11DatasetAdapter(root, subset_size=2)
    sample = adapter.load_sample(0)
    assert sample["image"].shape == (4, 6, 224, 224)
    assert sample["mask"].shape == (224, 224)
    assert adapter.get_region(0) == "Bolivia"


def test_run_real_with_mock_prithvi_writes_artifacts() -> None:
    root = Path("outputs/test_prithvi_real")
    _write_prepared_fixture(root)
    dataset = Sen1Floods11DatasetAdapter(root, subset_size=6)
    model = PrithviAdapter(model=MockPrithviModel())
    artifacts = run_real_pipeline(dataset, model, "outputs/test_prithvi_real_run", "sen1floods11", "prithvi")
    assert artifacts["tables_fairness_matrix"].exists()
    assert artifacts["figures_average_vs_worst_group"].exists()


def test_run_segmentation_smoke_writes_artifacts() -> None:
    root = Path("outputs/test_prithvi_seg")
    _write_prepared_fixture(root)
    dataset = Sen1Floods11DatasetAdapter(root, subset_size=4)
    model = PrithviAdapter(model=MockPrithviModel())
    artifacts = run_segmentation_smoke(dataset, model, "outputs/test_prithvi_seg_run")
    assert artifacts["segmentation_metrics"].exists()
    assert artifacts["segmentation_fairness_matrix_region"].exists()
    assert artifacts["iou_by_group"].exists()
    rows = read_csv_rows(artifacts["segmentation_metrics"])
    assert len(rows) == 4


def test_prepare_sen1floods11_maps_13_band_s2_to_prithvi_shape(monkeypatch) -> None:
    source = Path("outputs/test_sen1_prepare_source")
    source.mkdir(parents=True, exist_ok=True)
    s2 = source / "Bolivia_000001_S2Hand.tif"
    qc = source / "Bolivia_000001_QC.tif"
    s2.write_text("mock", encoding="utf-8")
    qc.write_text("mock", encoding="utf-8")

    def fake_read_tif(path: Path) -> np.ndarray:
        if "QC" in path.name:
            mask = np.zeros((5, 7), dtype=np.float32)
            mask[0, 0] = -1
            mask[1:3, 1:3] = 1
            return mask[None, :, :]
        return np.ones((13, 5, 7), dtype=np.float32)

    monkeypatch.setattr(prep, "_read_tif", fake_read_tif)
    metadata_path = prep.prepare_sen1floods11_subset(
        output_dir=Path("outputs/test_sen1_prepare_output"),
        source_root=source,
        max_samples=1,
        target_size=16,
    )
    rows = read_csv_rows(metadata_path)
    image = np.load(Path("outputs/test_sen1_prepare_output", rows[0]["chip_path"]))["image"]
    mask = np.load(Path("outputs/test_sen1_prepare_output", rows[0]["mask_path"]))["mask"]
    assert image.shape == (4, 6, 16, 16)
    assert mask.shape == (16, 16)
    assert rows[0]["source_dataset"] == "sen1floods11"


def test_prithvi_cli_commands_exist() -> None:
    parser = build_parser()
    run_args = parser.parse_args(
        [
            "run-real",
            "--dataset",
            "sen1floods11",
            "--model",
            "prithvi",
            "--data-root",
            "data/sen1",
            "--model-config",
            "configs/models/prithvi.yaml",
        ]
    )
    seg_args = parser.parse_args(
        [
            "run-segmentation-real",
            "--dataset",
            "sen1floods11",
            "--model",
            "prithvi",
            "--data-root",
            "data/sen1",
            "--model-config",
            "configs/models/prithvi.yaml",
        ]
    )
    assert run_args.model == "prithvi"
    assert seg_args.command == "run-segmentation-real"
