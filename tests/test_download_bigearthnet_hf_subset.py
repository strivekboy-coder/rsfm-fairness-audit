from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from rsfm_fairness_audit.adapters.bigearthnet import BigEarthNetDatasetAdapter
from rsfm_fairness_audit.cli import build_parser


def _load_download_module():
    script_path = Path("scripts/download_bigearthnet_hf_subset.py")
    spec = importlib.util.spec_from_file_location("download_bigearthnet_hf_subset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_hf_subset_falls_back_and_writes_adapter_manifest(monkeypatch) -> None:
    module = _load_download_module()

    def fake_load_dataset(path: str, split: str, streaming: bool, **kwargs):
        if path == "GFM-Bench/BigEarthNet":
            return [{"sample_id": f"meta-{idx}", "label": idx % 2} for idx in range(40)]
        return [
            {
                "sample_id": f"real-{idx}",
                "image": np.full((9, 6, 6), idx, dtype=np.float32),
                "labels": [1, 0] if idx % 2 == 0 else [0, 1],
                "country": "germany" if idx % 2 == 0 else "spain",
                "lat": 50 + idx,
                "lon": 8 + idx,
            }
            for idx in range(6)
        ]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    output = Path("outputs/test_hf_download_subset")
    metadata_path = module.download_subset(
        source="GFM-Bench/BigEarthNet",
        output_dir=output,
        max_samples=4,
        sensor_mode="S2",
        seed=42,
    )

    rows = list(csv.DictReader(metadata_path.open(newline="", encoding="utf-8")))
    assert len(rows) == 4
    assert rows[0]["source_dataset"] == "lc-col/bigearthnet"
    assert rows[0]["chip_path"].endswith(".npz")
    assert rows[0]["s2_path"].endswith(".npz")
    assert (output / rows[0]["s2_path"]).exists()

    adapter = BigEarthNetDatasetAdapter(output, subset_size=2, sensor_mode="S2")
    assert adapter.load_sample(0)["image"].shape == (9, 6, 6)


def test_run_real_accepts_requested_aliases() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-real",
            "--dataset",
            "bigearthnet",
            "--dataset-root",
            "data/bigearthnet_hf_subset",
            "--model",
            "dofa",
            "--config",
            "configs/models/dofa.yaml",
            "--output-dir",
            "outputs/dofa_bigearthnet_hf64",
            "--max-samples",
            "64",
        ]
    )
    assert args.command == "run-real"
    assert str(args.data_root).endswith("bigearthnet_hf_subset")
    assert str(args.model_config).endswith("dofa.yaml")
    assert args.subset_size == 64
