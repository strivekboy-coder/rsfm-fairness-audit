from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import scripts.prepare_sen1floods11_subset as prep
from rsfm_fairness_audit.adapters.prithvi import PrithviAdapter, PrithviConfigurationError, PrithviSen1Floods11TLAdapter
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


class MockPrithviTLModel:
    def eval(self) -> None:
        return None

    def predict_segmentation(self, batch):
        images = batch["images"]
        # Make water probability high in the same top-left area
        # used by odd fixture masks so tests exercise the official-prediction path.
        batch_size, _, _, height, width = images.shape
        score_maps = np.full((batch_size, height, width), 0.1, dtype=np.float32)
        score_maps[:, :40, :40] = 0.9
        predictions = (score_maps >= 0.5).astype(np.int16)
        probabilities = np.stack([1.0 - score_maps, score_maps], axis=1)
        return {
            "predictions": predictions,
            "score_maps": score_maps,
            "confidence": np.maximum(score_maps, 1.0 - score_maps),
            "probabilities": probabilities.astype(np.float32),
        }


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


def test_prithvi_accepts_current_terratorch_registry_name() -> None:
    adapter = PrithviAdapter(
        terratorch_model_name="terratorch_prithvi_eo_v2_300",
        model=MockPrithviModel(),
    )
    adapter.load_model()
    assert adapter.terratorch_model_name == "terratorch_prithvi_eo_v2_300"


def test_prithvi_config_validation_rejects_unverified_registry_name() -> None:
    adapter = PrithviAdapter(terratorch_model_name="some_other_model", model=MockPrithviModel())
    with pytest.raises(PrithviConfigurationError, match="TerraTorch model name"):
        adapter.load_model()


def test_prithvi_repeats_single_timestamp_to_four_frames() -> None:
    adapter = PrithviAdapter(model=MockPrithviModel())
    batch = {
        "samples": [{"image": np.zeros((6, 224, 224), dtype=np.float32)}],
        "metadata": [{"sample_id": "x"}],
    }
    processed = adapter.preprocess(batch)
    assert processed["images"].shape == (1, 4, 6, 224, 224)


def test_prithvi_pools_token_outputs_instead_of_flattening() -> None:
    adapter = PrithviAdapter(model=MockPrithviModel())
    embeddings = adapter._pool_output(np.ones((1, 197, 768), dtype=np.float32))
    assert embeddings.shape == (1, 768)


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


def test_streaming_run_real_with_mock_prithvi_writes_artifacts() -> None:
    root = Path("outputs/test_prithvi_streaming_real")
    _write_prepared_fixture(root, count=4)
    dataset = Sen1Floods11DatasetAdapter(root, subset_size=4)
    model = PrithviAdapter(model=MockPrithviModel(), batch_size=1)
    artifacts = run_real_pipeline(
        dataset,
        model,
        "outputs/test_prithvi_streaming_real_run",
        "sen1floods11",
        "prithvi",
        chunk_size=2,
        streaming_embeddings=True,
    )
    assert artifacts["predictions"].exists()
    assert (Path("outputs/test_prithvi_streaming_real_run") / "embedding_chunks" / "chunk_00000.npz").exists()


def test_run_segmentation_smoke_writes_artifacts() -> None:
    root = Path("outputs/test_prithvi_seg")
    _write_prepared_fixture(root)
    dataset = Sen1Floods11DatasetAdapter(root, subset_size=4)
    model = PrithviAdapter(model=MockPrithviModel())
    artifacts = run_segmentation_smoke(dataset, model, "outputs/test_prithvi_seg_run")
    assert artifacts["segmentation_metrics"].exists()
    assert artifacts["diagnostic_baseline_comparison"].exists()
    assert artifacts["diagnostic_baseline_per_chip"].exists()
    assert artifacts["segmentation_fairness_matrix_region"].exists()
    assert artifacts["iou_by_group"].exists()
    rows = read_csv_rows(artifacts["segmentation_metrics"])
    assert len(rows) == 4
    assert "predicted_positive_pixel_ratio" in rows[0]
    assert "ground_truth_positive_pixel_ratio" in rows[0]
    assert "label_values_distribution" in rows[0]
    assert "prediction_unique_values" in rows[0]
    assert rows[0]["input_band_order"] == "B02,B03,B04,B05,B06,B07"
    assert rows[0]["mask_resize_alignment"] == "image=bilinear_224x224;mask=nearest_224x224;source=LabelHand"
    comparison = read_csv_rows(artifacts["diagnostic_baseline_comparison"])
    baselines = {row["baseline_name"] for row in comparison}
    assert {
        "mean_threshold_high_positive",
        "mean_threshold_low_positive",
        "ndwi_like_b03_b06_positive",
        "ndwi_like_b03_b07_positive",
    } <= baselines
    assert any(row["event_id"] == "__overall__" for row in comparison)


def test_run_segmentation_with_official_tl_adapter_preserves_protocol_metadata() -> None:
    root = Path("outputs/test_prithvi_tl_seg")
    _write_prepared_fixture(root, count=4)
    dataset = Sen1Floods11DatasetAdapter(root, subset_size=4)
    model = PrithviSen1Floods11TLAdapter(model=MockPrithviTLModel())
    artifacts = run_segmentation_smoke(dataset, model, "outputs/test_prithvi_tl_seg_run", debug_samples=2)
    rows = read_csv_rows(artifacts["segmentation_metrics"])
    event_rows = read_csv_rows(artifacts["event_segmentation_metrics"])
    assert artifacts["model_debug"].exists()
    assert artifacts["raw_model_output_debug"].exists()
    assert rows[0]["model"] == "prithvi_tl_sen1floods11"
    assert rows[0]["model_family"] == "Prithvi"
    assert rows[0]["adaptation_protocol"] == "task_adapted_decoder"
    assert rows[0]["training_budget"] == "official_sen1floods11_finetune"
    assert rows[0]["confidence_source"] == "max_softmax_probability"
    assert rows[0]["background_prob_mean"] != ""
    assert rows[0]["water_prob_mean"] != ""
    assert rows[0]["positive_class_definition"] == "class_1_water_flood"
    assert event_rows[0]["adaptation_protocol"] == "task_adapted_decoder"


def test_prepare_sen1floods11_maps_13_band_s2_to_prithvi_shape(monkeypatch) -> None:
    source = Path("outputs/test_sen1_prepare_source")
    source.mkdir(parents=True, exist_ok=True)
    s2 = source / "Bolivia_000001_S2Hand.tif"
    qc = source / "Bolivia_000001_LabelHand.tif"
    s2.write_text("mock", encoding="utf-8")
    qc.write_text("mock", encoding="utf-8")

    def fake_read_tif(path: Path) -> np.ndarray:
        if "QC" in path.name or "LabelHand" in path.name:
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


def test_prepare_sen1floods11_tl_band_profile_uses_official_indices(monkeypatch) -> None:
    source = Path("outputs/test_sen1_prepare_tl_source")
    source.mkdir(parents=True, exist_ok=True)
    s2 = source / "Bolivia_000002_S2Hand.tif"
    qc = source / "Bolivia_000002_LabelHand.tif"
    s2.write_text("mock", encoding="utf-8")
    qc.write_text("mock", encoding="utf-8")

    def fake_read_tif(path: Path) -> np.ndarray:
        if "LabelHand" in path.name:
            mask = np.zeros((5, 7), dtype=np.float32)
            mask[1:3, 1:3] = 1
            return mask[None, :, :]
        return np.stack([np.full((5, 7), index, dtype=np.float32) for index in range(13)])

    monkeypatch.setattr(prep, "_read_tif", fake_read_tif)
    metadata_path = prep.prepare_sen1floods11_subset(
        output_dir=Path("outputs/test_sen1_prepare_tl_output"),
        source_root=source,
        max_samples=1,
        target_size=5,
        band_profile="prithvi_tl_sen1floods11",
    )
    rows = read_csv_rows(metadata_path)
    image = np.load(Path("outputs/test_sen1_prepare_tl_output", rows[0]["chip_path"]))["image"]
    assert image.shape == (1, 6, 5, 5)
    assert image[:, :, 0, 0].tolist() == [[1.0, 2.0, 3.0, 8.0, 11.0, 12.0]]
    assert rows[0]["band_profile"] == "prithvi_tl_sen1floods11"


def test_prepare_sen1floods11_max_samples_zero_uses_all_local_pairs(monkeypatch) -> None:
    source = Path("outputs/test_sen1_prepare_all_source")
    source.mkdir(parents=True, exist_ok=True)
    for name in ["Bolivia_001", "Bolivia_002"]:
        (source / f"{name}_S2Hand.tif").write_text("mock", encoding="utf-8")
        (source / f"{name}_LabelHand.tif").write_text("mock", encoding="utf-8")

    def fake_read_tif(path: Path) -> np.ndarray:
        if "LabelHand" in path.name:
            mask = np.zeros((5, 7), dtype=np.float32)
            mask[1:3, 1:3] = 1
            return mask[None, :, :]
        return np.ones((13, 5, 7), dtype=np.float32)

    monkeypatch.setattr(prep, "_read_tif", fake_read_tif)
    metadata_path = prep.prepare_sen1floods11_subset(
        output_dir=Path("outputs/test_sen1_prepare_all_output"),
        source_root=source,
        max_samples=0,
        target_size=16,
    )
    rows = read_csv_rows(metadata_path)
    assert len(rows) == 2


def test_sen1floods11_gcs_label_candidates_prefer_official_labelhand() -> None:
    uri = "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/India_123_S2Hand.tif"
    candidates = prep._label_uri_candidates(uri)
    assert candidates[0] == "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/India_123_LabelHand.tif"
    assert any(candidate.endswith("India_123_QC.tif") for candidate in candidates)


def test_sen1floods11_batch_download_selection_uses_parallel_cp(monkeypatch) -> None:
    calls = []

    def fake_ls_exists(uri: str) -> bool:
        return "LabelHand" in uri

    def fake_download_many(uris: list[str], raw_dir: Path) -> None:
        calls.append((uris, raw_dir))
        raw_dir.mkdir(parents=True, exist_ok=True)
        for uri in uris:
            (raw_dir / Path(uri).name).write_text("mock", encoding="utf-8")

    monkeypatch.setattr(prep, "_gsutil_ls_exists", fake_ls_exists)
    monkeypatch.setattr(prep, "_download_many", fake_download_many)
    uris = [
        "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/India_001_S2Hand.tif",
        "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/India_002_S2Hand.tif",
    ]
    pairs, failures = prep._select_and_download_gcs_pairs(
        uris,
        cache_dir=Path("outputs/test_sen1_batch_cache"),
        max_samples=2,
        candidate_limit=2,
        parallel_download=True,
    )
    assert failures == 0
    assert len(pairs) == 2
    assert len(calls) == 1
    assert len(calls[0][0]) == 4


def test_sen1floods11_pair_manifest_uses_bulk_listings_and_cache(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_check_output(command: list[str], text: bool, stderr: object) -> str:
        calls.append(command)
        pattern = command[-1]
        if "S2Hand" in pattern:
            return "\n".join(
                [
                    "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/Ghana_001_S2Hand.tif",
                    "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/Bolivia_001_S2Hand.tif",
                ]
            )
        if "LabelHand" in pattern:
            return "\n".join(
                [
                    "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Ghana_001_LabelHand.tif",
                    "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Bolivia_001_LabelHand.tif",
                ]
            )
        return ""

    monkeypatch.setattr(prep.subprocess, "check_output", fake_check_output)
    cache_dir = Path("outputs/test_sen1_manifest_cache")
    rows = prep._build_gcs_pair_manifest("gs://sen1floods11", cache_dir, refresh=True)
    assert [row["sample_id"] for row in rows] == ["Bolivia_001", "Ghana_001"]
    assert all(command[:2] == ["gsutil", "ls"] for command in calls)
    assert all(len(command) <= 4 for command in calls)
    calls.clear()
    cached = prep._build_gcs_pair_manifest("gs://sen1floods11", cache_dir)
    assert cached == rows
    assert calls == []


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
    tl_seg_args = parser.parse_args(
        [
            "run-segmentation-real",
            "--dataset",
            "sen1floods11",
            "--model",
            "prithvi_tl_sen1floods11",
            "--data-root",
            "data/sen1",
            "--model-config",
            "configs/models/prithvi_tl_sen1floods11.yaml",
            "--debug-samples",
            "2",
        ]
    )
    assert run_args.model == "prithvi"
    assert seg_args.command == "run-segmentation-real"
    assert tl_seg_args.model == "prithvi_tl_sen1floods11"
    assert tl_seg_args.debug_samples == 2


def test_prithvi_requirements_pin_numpy_below_numba_limit() -> None:
    text = Path("requirements-prithvi.txt").read_text(encoding="utf-8")
    assert "numpy>=1.24,<2.1" in text


def test_prithvi_config_uses_current_terratorch_registry_name() -> None:
    text = Path("configs/models/prithvi.yaml").read_text(encoding="utf-8")
    assert "hf_model_id: ibm-nasa-geospatial/Prithvi-EO-2.0-300M" in text
    assert "terratorch_model_name: terratorch_prithvi_eo_v2_300" in text


def test_prithvi_tl_sen1floods11_config_uses_official_hf_model() -> None:
    text = Path("configs/models/prithvi_tl_sen1floods11.yaml").read_text(encoding="utf-8")
    assert "hf_model_id: ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11" in text
    assert "adaptation_protocol: task_adapted_decoder" in text
