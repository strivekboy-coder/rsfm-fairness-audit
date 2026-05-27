from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.adapters.bigearthnet import BigEarthNetDatasetAdapter
from rsfm_fairness_audit.adapters.croma import CROMAAdapter, CROMAConfigurationError
from rsfm_fairness_audit.cli import build_parser
from rsfm_fairness_audit.io import read_csv_rows
from rsfm_fairness_audit.pipeline import compare_model_runs, run_real_pipeline


class MockCROMAModel:
    def extract_embeddings(self, *inputs) -> np.ndarray:
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            arrays = [inputs[0]["SAR_images"], inputs[0]["optical_images"]]
        else:
            arrays = list(inputs)
        optical_images = np.concatenate(arrays, axis=1)
        return np.concatenate(
            [
                optical_images.mean(axis=(2, 3)),
                optical_images.std(axis=(2, 3)),
            ],
            axis=1,
        ).astype(np.float32)


def test_croma_torch_forward_uses_resolved_device() -> None:
    torch = pytest.importorskip("torch")

    class TorchCROMAModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.observed_input_device = None

        def forward(self, SAR_images):
            self.observed_input_device = str(SAR_images.device)
            return {"SAR_GAP": SAR_images.mean(dim=(2, 3))}

    model = TorchCROMAModel()
    adapter = CROMAAdapter(input_modality="SAR", embedding_key="SAR_GAP", model=model, image_size=16, device="cpu")
    adapter.load_model()
    processed = adapter.preprocess(
        {
            "samples": [{"image": np.zeros((2, 16, 16), dtype=np.float32)}],
            "metadata": [{"sample_id": "x"}],
        }
    )

    embeddings = adapter.extract_embeddings(processed)

    assert embeddings.shape == (1, 2)
    assert model.observed_input_device == str(next(model.parameters()).device)


def _write_s2_fixture(root: Path, count: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        image = np.full((12, 16, 16), fill_value=float(index % 3), dtype=np.float32)
        image += np.arange(12, dtype=np.float32)[:, None, None] * 0.01
        chip_path = root / f"croma_sample_{index:03d}.npz"
        np.savez_compressed(chip_path, image=image)
        rows.append(
            {
                "sample_id": f"BEN-CROMA-{index:03d}",
                "label": index % 2,
                "labels": "forest" if index % 2 == 0 else "urban",
                "region": "fallback_a" if index < count // 2 else "fallback_b",
                "fallback_group": "fallback_a" if index < count // 2 else "fallback_b",
                "sensor": "S2",
                "split": "train",
                "chip_path": chip_path.name,
                "source_dataset": "lc-col/bigearthnet",
                "band_profile": "sentinel2_12_lccol",
            }
        )
    metadata_path = root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def test_croma_config_validation_restricts_hf_repo() -> None:
    adapter = CROMAAdapter(hf_repo_id="someone/else", model=MockCROMAModel())
    with pytest.raises(CROMAConfigurationError, match="official repo"):
        adapter.load_model()


def test_croma_config_validation_restricts_checkpoint_filename() -> None:
    adapter = CROMAAdapter(hf_checkpoint_filename="not_official.pt", model=MockCROMAModel())
    with pytest.raises(CROMAConfigurationError, match="checkpoint filename"):
        adapter.load_model()


def test_croma_missing_source_implementation_is_clear() -> None:
    checkpoint = Path("outputs/test_croma_missing_source/CROMA_base.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    adapter = CROMAAdapter(checkpoint_path=checkpoint)
    with pytest.raises(CROMAConfigurationError, match="official implementation is not configured"):
        adapter.load_model()


def test_croma_optical_mock_accepts_12_channel_inputs() -> None:
    adapter = CROMAAdapter(model=MockCROMAModel(), image_size=16)
    batch = {
        "samples": [{"image": np.zeros((12, 16, 16), dtype=np.float32)}],
        "metadata": [{"sample_id": "x"}],
    }
    adapter.load_model()
    processed = adapter.preprocess(batch)
    embeddings = adapter.extract_embeddings(processed)
    assert processed["optical_images"].shape == (1, 12, 16, 16)
    assert embeddings.shape == (1, 24)


def test_croma_rejects_wrong_channel_count() -> None:
    adapter = CROMAAdapter(model=MockCROMAModel(), image_size=16)
    batch = {
        "samples": [{"image": np.zeros((9, 16, 16), dtype=np.float32)}],
        "metadata": [{"sample_id": "x"}],
    }
    with pytest.raises(CROMAConfigurationError, match="input has 9 channels"):
        adapter.preprocess(batch)


def test_croma_both_requires_paired_samples() -> None:
    adapter = CROMAAdapter(input_modality="both", model=MockCROMAModel())
    batch = {
        "samples": [{"image": np.zeros((12, 16, 16), dtype=np.float32)}],
        "metadata": [{"sample_id": "x"}],
    }
    with pytest.raises(CROMAConfigurationError, match="paired dict"):
        adapter.preprocess(batch)


def test_croma_sar_mock_accepts_2_channel_inputs() -> None:
    adapter = CROMAAdapter(input_modality="SAR", model=MockCROMAModel(), image_size=16)
    batch = {
        "samples": [{"image": np.zeros((2, 16, 16), dtype=np.float32)}],
        "metadata": [{"sample_id": "x"}],
    }
    adapter.load_model()
    processed = adapter.preprocess(batch)
    embeddings = adapter.extract_embeddings(processed)
    assert processed["SAR_images"].shape == (1, 2, 16, 16)
    assert embeddings.shape == (1, 4)


def test_croma_both_mock_accepts_paired_inputs() -> None:
    adapter = CROMAAdapter(input_modality="both", model=MockCROMAModel(), image_size=16)
    batch = {
        "samples": [
            {
                "image": {
                    "S1": np.zeros((2, 16, 16), dtype=np.float32),
                    "S2": np.zeros((12, 16, 16), dtype=np.float32),
                }
            }
        ],
        "metadata": [{"sample_id": "x"}],
    }
    adapter.load_model()
    processed = adapter.preprocess(batch)
    embeddings = adapter.extract_embeddings(processed)
    assert processed["SAR_images"].shape == (1, 2, 16, 16)
    assert processed["optical_images"].shape == (1, 12, 16, 16)
    assert embeddings.shape == (1, 28)


def test_run_real_with_mock_croma_writes_expected_artifacts() -> None:
    root = Path("outputs/test_croma_real_fixture")
    metadata_path = _write_s2_fixture(root, count=8)
    dataset = BigEarthNetDatasetAdapter(root, metadata_path=metadata_path, subset_size=6, sensor_mode="S2")
    model = CROMAAdapter(model=MockCROMAModel(), image_size=16)

    artifacts = run_real_pipeline(dataset, model, "outputs/test_croma_real_pipeline", "bigearthnet", "croma")

    assert artifacts["embeddings"].exists()
    assert artifacts["tables_fairness_matrix"].exists()
    assert artifacts["tables_raw_vs_balanced_gap"].exists()
    assert artifacts["figures_average_vs_worst_group"].exists()
    assert artifacts["figures_fairness_map"].exists()
    assert len(read_csv_rows(artifacts["predictions"])) == 6


def test_run_real_command_accepts_croma() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-real",
            "--dataset",
            "bigearthnet",
            "--model",
            "croma",
            "--data-root",
            "somewhere",
            "--subset-size",
            "64",
            "--model-config",
            "configs/models/croma.yaml",
        ]
    )
    assert args.command == "run-real"
    assert args.model == "croma"


def test_compare_model_runs_writes_table() -> None:
    dofa_root = Path("outputs/test_compare_dofa")
    croma_root = Path("outputs/test_compare_croma")
    dofa_root.mkdir(parents=True, exist_ok=True)
    croma_root.mkdir(parents=True, exist_ok=True)
    header = "gap_name,num_groups,average_performance,balanced_average_performance,worst_group,worst_region_performance,best_group,best_region_performance,best_worst_gap,group_standard_deviation,max_drop_from_global,fairness_risk_score\n"
    dofa_root.joinpath("fairness_summary.csv").write_text(
        header + "raw_region_gap,2,0.7,0.7,a,0.6,b,0.8,0.2,0.1,0.1,0.05\n",
        encoding="utf-8",
    )
    croma_root.joinpath("fairness_summary.csv").write_text(
        header + "raw_region_gap,2,0.75,0.75,a,0.65,b,0.85,0.2,0.1,0.1,0.05\n",
        encoding="utf-8",
    )
    dofa_root.joinpath("raw_vs_balanced_gap.csv").write_text(
        "slice_name,raw_fairness_gap,balanced_fairness_gap\nregion,0.2,0.15\n",
        encoding="utf-8",
    )
    croma_root.joinpath("raw_vs_balanced_gap.csv").write_text(
        "slice_name,raw_fairness_gap,balanced_fairness_gap\nregion,0.2,0.12\n",
        encoding="utf-8",
    )

    artifacts = compare_model_runs(
        {"dofa": dofa_root, "croma": croma_root},
        "outputs/test_model_comparison",
    )

    rows = read_csv_rows(artifacts["model_comparison"])
    assert [row["model"] for row in rows] == ["dofa", "croma"]
    assert artifacts["comparison_figure"].exists()
