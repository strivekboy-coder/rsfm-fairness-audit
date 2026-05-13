from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import scripts.prepare_ben_ge_800_subset as prep
from rsfm_fairness_audit.adapters.ben_ge import BenGEDatasetAdapter
from rsfm_fairness_audit.adapters.croma import CROMAAdapter
from rsfm_fairness_audit.cli import build_parser
from rsfm_fairness_audit.io import read_csv_rows
from rsfm_fairness_audit.pipeline import compare_sensor_mode_runs, run_real_pipeline


class MockCROMAModel:
    def extract_embeddings(self, *arrays):
        if len(arrays) == 1 and isinstance(arrays[0], dict):
            arrays = [arrays[0]["SAR_images"], arrays[0]["optical_images"]]
        image = np.concatenate(list(arrays), axis=1)
        return np.concatenate([image.mean(axis=(2, 3)), image.std(axis=(2, 3))], axis=1).astype(np.float32)


def _write_ben_ge_fixture(root: Path, count: int = 6) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        s1 = np.full((2, 16, 16), index % 2, dtype=np.float32)
        s2 = np.full((12, 16, 16), index % 3, dtype=np.float32)
        s1_path = root / f"sample_{index:03d}_s1.npz"
        s2_path = root / f"sample_{index:03d}_s2.npz"
        np.savez_compressed(s1_path, image=s1)
        np.savez_compressed(s2_path, image=s2)
        rows.append(
            {
                "sample_id": f"BEN-GE-{index:03d}",
                "s1_path": s1_path.name,
                "s2_path": s2_path.name,
                "label": index % 2,
                "label_vector": "[1, 0]" if index % 2 == 0 else "[0, 1]",
                "label_names": '["tree"]' if index % 2 == 0 else '["urban"]',
                "region": "climatezone_1" if index < count // 2 else "climatezone_2",
                "climatezone": 1 if index < count // 2 else 2,
                "latitude": 50.0 + index,
                "longitude": 8.0 + index,
                "sensor": "S1+S2",
                "split": "all",
            }
        )
    metadata_path = root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def test_ben_ge_adapter_loads_paired_samples() -> None:
    root = Path("outputs/test_ben_ge_adapter")
    _write_ben_ge_fixture(root)
    adapter = BenGEDatasetAdapter(root, subset_size=2, sensor_mode="S1+S2")
    sample = adapter.load_sample(0)
    assert sample["image"]["S1"].shape == (2, 16, 16)
    assert sample["image"]["S2"].shape == (12, 16, 16)
    assert adapter.get_sensor(0) == "S1+S2"


def test_run_real_with_mock_croma_both_on_ben_ge() -> None:
    root = Path("outputs/test_ben_ge_run_real")
    _write_ben_ge_fixture(root)
    dataset = BenGEDatasetAdapter(root, subset_size=6, sensor_mode="S1+S2")
    model = CROMAAdapter(input_modality="both", model=MockCROMAModel(), image_size=16)
    artifacts = run_real_pipeline(dataset, model, "outputs/test_ben_ge_croma_both", "ben_ge", "croma")
    assert artifacts["tables_fairness_matrix"].exists()
    assert artifacts["figures_fairness_map"].exists()


def test_compare_sensor_mode_runs_writes_outputs() -> None:
    roots = {}
    header = "gap_name,num_groups,average_performance,balanced_average_performance,worst_group,worst_region_performance,best_group,best_region_performance,best_worst_gap,group_standard_deviation,max_drop_from_global,fairness_risk_score\n"
    for mode, avg in [("sar", 0.5), ("optical", 0.6), ("both", 0.7)]:
        root = Path(f"outputs/test_sensor_compare_{mode}")
        root.mkdir(parents=True, exist_ok=True)
        root.joinpath("fairness_summary.csv").write_text(
            header + f"raw_region_gap,2,{avg},{avg},a,{avg - 0.1},b,{avg + 0.1},0.2,0.1,0.1,0.05\n",
            encoding="utf-8",
        )
        root.joinpath("raw_vs_balanced_gap.csv").write_text(
            "slice_name,raw_fairness_gap,balanced_fairness_gap\nregion,0.2,0.15\n",
            encoding="utf-8",
        )
        roots[mode] = root
    artifacts = compare_sensor_mode_runs(roots, "outputs/test_sensor_mode_comparison")
    assert artifacts["sensor_mode_comparison"].exists()
    assert artifacts["sensor_fairness_heatmap"].exists()
    assert artifacts["average_vs_worst_sensor_mode"].exists()
    assert artifacts["report"].exists()


def test_ben_ge_prepare_script_with_mocked_tifs(monkeypatch) -> None:
    source = Path("outputs/test_ben_ge_prepare_source")
    patch = "S2A_TEST_1_1"
    patch_s1 = "S1A_TEST_1_1"
    (source / "sentinel-1" / patch_s1).mkdir(parents=True, exist_ok=True)
    (source / "sentinel-2" / patch).mkdir(parents=True, exist_ok=True)
    (source / "ben-ge-800_meta.csv").write_text(
        "patch_id,patch_id_s1,lat,lon,climatezone\nS2A_TEST_1_1,S1A_TEST_1_1,50,8,4\n",
        encoding="utf-8",
    )
    for band in prep.S1_BANDS:
        (source / "sentinel-1" / patch_s1 / f"{patch_s1}_{band}.tif").write_text("mock", encoding="utf-8")
    for band in prep.S2_BANDS:
        (source / "sentinel-2" / patch / f"{patch}_{band}.tif").write_text("mock", encoding="utf-8")
    (source / "sentinel-2" / patch / f"{patch}_labels_metadata.json").write_text(
        json.dumps({"labels": ["Tree cover"]}),
        encoding="utf-8",
    )

    def fake_read_tif(path: Path) -> np.ndarray:
        if "_B01." in str(path) or "_B09." in str(path):
            return np.ones((2, 2), dtype=np.float32)
        if "_B11." in str(path) or "_B12." in str(path):
            return np.ones((3, 5), dtype=np.float32)
        return np.ones((4, 4), dtype=np.float32)

    monkeypatch.setattr(prep, "_read_tif", fake_read_tif)
    metadata_path = prep.prepare_ben_ge_800_subset(
        output_dir=Path("outputs/test_ben_ge_prepare_output"),
        max_samples=1,
        source_dir=source,
        target_size=8,
    )
    rows = read_csv_rows(metadata_path)
    assert rows[0]["source_dataset"] == "ben-ge-800"
    s1 = np.load(Path("outputs/test_ben_ge_prepare_output", rows[0]["s1_path"]))["image"]
    s2 = np.load(Path("outputs/test_ben_ge_prepare_output", rows[0]["s2_path"]))["image"]
    assert s1.shape == (2, 8, 8)
    assert s2.shape == (12, 8, 8)


def test_cli_accepts_ben_ge_and_compare_sensor_modes() -> None:
    parser = build_parser()
    run_args = parser.parse_args(
        [
            "run-real",
            "--dataset",
            "ben_ge",
            "--model",
            "croma",
            "--data-root",
            "data/ben_ge_800_subset",
            "--sensor-mode",
            "S1+S2",
            "--model-config",
            "configs/models/croma_both.yaml",
        ]
    )
    compare_args = parser.parse_args(
        [
            "compare-sensor-modes",
            "--run",
            "sar=outputs/sar",
            "--run",
            "optical=outputs/optical",
            "--run",
            "both=outputs/both",
        ]
    )
    assert run_args.dataset == "ben_ge"
    assert compare_args.command == "compare-sensor-modes"
