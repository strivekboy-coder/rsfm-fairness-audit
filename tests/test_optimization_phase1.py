from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.optimization_phase1 import distribution_metrics, synthetic_counterexamples
from rsfm_fairness_audit.reben_phase1_runners import validate_cache_contract, validate_paired_cache_contract
from rsfm_fairness_audit.reben_terramind_campaign import train_streaming_multilabel_probe


def test_distribution_metrics_reports_variance_and_tail_separately() -> None:
    result = distribution_metrics({"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.4})
    assert result["weighted_std"] > 0
    assert result["weighted_variance"] == pytest.approx(result["weighted_std"] ** 2)
    assert result["geobwer_beta_0_10"] > 0
    assert len(result["profile"]) == 4


def test_counterexamples_include_levelling_down() -> None:
    rows = {row["scenario"]: row for row in synthetic_counterexamples()}
    before = rows["levelling_down_before"]
    after = rows["levelling_down_after"]
    assert after["mean_risk"] > before["mean_risk"]
    assert after["geobwer_beta_0_10"] < before["geobwer_beta_0_10"]


def _write_split(root: Path, split: str, ids: list[str]) -> None:
    target = root / split
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "embeddings.npy", np.zeros((len(ids), 4), dtype=np.float32))
    np.save(target / "labels.npy", np.zeros((len(ids), 19), dtype=np.int8))
    rows = [
        {
            "sample_id": sample_id,
            "country": "DEU",
            "source_tile_id": f"tile_{sample_id}",
            "independent_unit_id": f"unit_{sample_id}",
        }
        for sample_id in ids
    ]
    (target / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_embedding_cache_contract_rejects_sample_leakage() -> None:
    test_root = Path("work/test_optimization_phase1_contract")
    _write_split(test_root, "train", ["a", "b"])
    _write_split(test_root, "val", ["c"])
    _write_split(test_root, "test", ["d"])
    contract = validate_cache_contract(test_root)
    assert contract["sample_overlap_counts"] == {"train__val": 0, "train__test": 0, "val__test": 0}

    _write_split(test_root, "test", ["a"])
    try:
        validate_cache_contract(test_root)
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected leakage rejection")


def test_paired_cache_contract_accepts_aligned_test_samples() -> None:
    root = Path("work/test_optimization_phase1_paired")
    s2 = root / "s2"
    s1 = root / "s1"
    for cache_root, prefix in ((s2, "s2"), (s1, "s1")):
        _write_split(cache_root, "train", [f"{prefix}_train"])
        _write_split(cache_root, "val", [f"{prefix}_val"])
        _write_split(cache_root, "test", ["paired_a", "paired_b"])
    contract = validate_paired_cache_contract(s2, s1)
    assert contract["status"] == "ready"
    assert contract["paired_test_sample_count"] == 2


def test_indexed_probe_reuses_full_embedding_cache() -> None:
    pytest.importorskip("torch")
    root = Path("work/test_optimization_phase1_indexed_probe")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(12, 5)).astype(np.float32)
    train_y = (rng.random((12, 19)) > 0.75).astype(np.int8)
    val_x = rng.normal(size=(4, 5)).astype(np.float32)
    test_x = rng.normal(size=(5, 5)).astype(np.float32)
    np.save(root / "train_x.npy", train_x)
    np.save(root / "train_y.npy", train_y)
    np.save(root / "val_x.npy", val_x)
    np.save(root / "test_x.npy", test_x)
    probabilities, checkpoint = train_streaming_multilabel_probe(
        root / "train_x.npy",
        root / "train_y.npy",
        {"validation": root / "val_x.npy", "test": root / "test_x.npy"},
        root / "probe",
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=3,
        device="cpu",
        seed=7,
        cache_signature="indexed-test-v1",
        train_indices=np.asarray([0, 2, 4, 6, 8, 10]),
    )
    assert checkpoint.exists()
    assert np.load(probabilities["validation"]).shape == (4, 19)
    assert np.load(probabilities["test"]).shape == (5, 19)
