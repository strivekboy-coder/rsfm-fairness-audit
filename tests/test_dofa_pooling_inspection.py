from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from scripts.inspect_fmow_dofa_pooling_ablation import inspect_pooling


def _write_run(root: Path, name: str, pooling: str, embeddings: np.ndarray) -> Path:
    run = root / name
    cache = root / "embedding_cache"
    run.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    train_path = cache / f"dofa_train_{name}.npz"
    eval_path = cache / f"dofa_eval_{name}.npz"
    sample_ids = np.asarray([f"s{i}" for i in range(embeddings.shape[0])])
    np.savez_compressed(train_path, embeddings=embeddings, sample_ids=sample_ids, labels=np.asarray(["a"] * embeddings.shape[0]))
    np.savez_compressed(eval_path, embeddings=embeddings, sample_ids=sample_ids, labels=np.asarray(["a"] * embeddings.shape[0]))
    (run / "run_metadata.json").write_text(
        json.dumps(
            {
                "embedding_pooling": pooling,
                "embedding_dim": embeddings.shape[1],
                "train_embedding_cache_path": str(train_path),
                "eval_embedding_cache_path": str(eval_path),
                "train_embedding_cache_key": f"train-{pooling}",
                "eval_embedding_cache_key": f"eval-{pooling}",
                "input_scale": 10000,
                "wavelength_list": [0.1] * 13,
            }
        ),
        encoding="utf-8",
    )
    return run


def test_pooling_inspection_identifies_pooled_identical_embeddings() -> None:
    root = Path("outputs") / f"test_dofa_pooling_inspect_{uuid4().hex}"
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    flatten = _write_run(root, "flatten", "flatten", embeddings)
    mean_tokens = _write_run(root, "mean_tokens", "mean_tokens", embeddings.copy())

    summary = inspect_pooling(root, flatten, mean_tokens)

    assert summary["all_identical"] is True
    assert summary["cache_keys_differ"] is True
    assert "already 2D" in summary["reason"]
    assert (root / "pooling_ablation_report.md").exists()
    assert (root / "pooling_embedding_diagnostics.csv").exists()

