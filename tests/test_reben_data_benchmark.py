from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
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


class _FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = np.asarray(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def contiguous(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value

    def to(self, *_args: Any, **_kwargs: Any) -> "_FakeTensor":
        return self

    def tolist(self) -> list[Any]:
        return self.value.tolist()


class _TinyModel:
    def to(self, _device: Any) -> "_TinyModel":
        return self

    def eval(self) -> "_TinyModel":
        return self

    def state_dict(self) -> dict[str, _FakeTensor]:
        return {"weight": _FakeTensor(np.asarray([1.0], dtype=np.float32))}

    def __call__(self, value: _FakeTensor) -> _FakeTensor:
        pooled = value.value.mean(axis=(1, 2, 3), keepdims=False)
        return _FakeTensor(
            np.repeat(pooled[:, np.newaxis], 19, axis=1).astype(np.float32)
        )


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def manual_seed_all(_seed: int) -> None:
        return None


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def manual_seed(_seed: int) -> None:
        return None

    @staticmethod
    def inference_mode():
        return nullcontext()


def test_worker_benchmark_preserves_order_inputs_labels_and_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _MemoryAdapter(count=20)
    loader_worker_calls: list[int] = []

    def _fake_loader(
        dataset: _MemoryAdapter,
        config: RebenResNet50Config,
        *,
        shuffle: bool,
        seed: int,
    ):
        assert shuffle is False
        assert seed == 42
        loader_worker_calls.append(config.num_workers)
        batches = []
        for start in range(0, len(dataset.rows), 4):
            indices = np.arange(start, start + 4, dtype=np.int64)
            images = np.stack(
                [
                    dataset.load_sample(int(index))["image"]
                    for index in indices
                ],
                axis=0,
            )
            labels = np.asarray(
                [dataset.rows[int(index)]["label_vector"] for index in indices],
                dtype=np.float32,
            )
            batches.append(
                (
                    _FakeTensor(images),
                    _FakeTensor(labels),
                    _FakeTensor(indices),
                )
            )
        return batches

    monkeypatch.setattr(benchmark, "_require_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        benchmark,
        "_device",
        lambda _requested: SimpleNamespace(type="cpu"),
    )
    monkeypatch.setattr(
        benchmark,
        "_TorchDataset",
        lambda source, _mode, _contract: source,
    )
    monkeypatch.setattr(benchmark, "_loader", _fake_loader)
    monkeypatch.setattr(
        benchmark,
        "build_resnet50_multiband",
        lambda *_args, **_kwargs: _TinyModel(),
    )
    config = RebenResNet50Config(
        lmdb_root=Path("unused-lmdb"),
        metadata_parquet=Path("unused-metadata.parquet"),
        output_dir=Path("unused-output"),
        sensor_modes=("S2",),
        seeds=(42, 73, 101),
        batch_size=4,
        device="cpu",
        persistent_workers=False,
    )
    report = benchmark.benchmark_reben_loader_workers(
        adapter,
        {"mean": [0.0] * 12, "std": [1.0] * 12},
        config,
        mode="S2",
        worker_counts=(0, 2),
        max_batches=5,
        checksum_batches=5,
        warmup_batches=0,
    )
    assert loader_worker_calls == [0, 0, 2, 2]
    assert [row["num_workers"] for row in report["results"]] == [0, 2]
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
    assert all(
        row["correctness"]["max_abs_forward_difference"] == pytest.approx(0.0)
        for row in report["results"]
    )
    assert all("first_logits" not in row for row in report["results"])
    json.dumps(report)
