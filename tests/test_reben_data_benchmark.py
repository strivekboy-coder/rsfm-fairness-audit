from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

import rsfm_fairness_audit.reben_data_benchmark as benchmark
from rsfm_fairness_audit.reben_resnet50_campaign import RebenResNet50Config


class _MemoryAdapter:
    def __init__(self, count: int = 24) -> None:
        self.rows = [
            {
                "sample_id": f"sample-{index:04d}",
                "label_vector": [int(label == index % 19) for label in range(19)],
            }
            for index in range(count)
        ]

    def load_metadata(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows]

    def load_sample(self, index: int) -> Mapping[str, Any]:
        image = np.full((12, 8, 8), index / 100.0, dtype=np.float32)
        return {"image": image, "metadata": dict(self.rows[index])}


def test_worker_benchmark_preserves_order_inputs_labels_and_forward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.head = torch.nn.Linear(12, 19)

        def forward(self, value):
            return self.head(self.pool(value).flatten(1))

    monkeypatch.setattr(
        benchmark,
        "build_resnet50_multiband",
        lambda *_args, **_kwargs: _TinyModel(),
    )
    config = RebenResNet50Config(
        lmdb_root=tmp_path / "lmdb",
        metadata_parquet=tmp_path / "metadata.parquet",
        output_dir=tmp_path / "out",
        sensor_modes=("S2",),
        seeds=(42, 73, 101),
        batch_size=4,
        device="cpu",
        persistent_workers=False,
    )
    report = benchmark.benchmark_reben_loader_workers(
        _MemoryAdapter(),
        {"mean": [0.0] * 12, "std": [1.0] * 12},
        config,
        mode="S2",
        worker_counts=(0, 2, 4, 8),
        max_batches=5,
        checksum_batches=5,
        warmup_batches=0,
    )
    assert [row["num_workers"] for row in report["results"]] == [0, 2, 4, 8]
    assert all(
        all(
            row["correctness"][key]
            for key in (
                "sample_order_equal",
                "labels_equal",
                "inputs_equal",
                "forward_allclose",
            )
        )
        for row in report["results"]
    )
