from __future__ import annotations

import pytest

from rsfm_fairness_audit.bwer import BWERConfig, compute_bwer


def test_raw_bwer_tail_and_max_are_computable_by_hand() -> None:
    rows = [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a1", "region": "A", "score": 0.9},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a2", "region": "A", "score": 0.9},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b1", "region": "B", "score": 0.5},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b2", "region": "B", "score": 0.5},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "c1", "region": "C", "score": 0.8},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "c2", "region": "C", "score": 0.8},
    ]
    config = BWERConfig(dataset="dummy", model="m", task="classification", tail_fraction=0.10, min_samples_per_slice=1)
    result = compute_bwer(rows, config, "region")
    assert result.summary["worst_slice"] == "B"
    assert result.summary["bwer"] == pytest.approx(0.5 - ((0.1 + 0.5 + 0.2) / 3))
    assert result.summary["max_bwer"] == pytest.approx(result.summary["bwer"])
    assert result.summary["tail_slices"] == "B"


def test_tail_fraction_can_include_bottom_k_tail() -> None:
    rows = []
    for region, score in [("A", 0.9), ("B", 0.5), ("C", 0.8)]:
        rows.extend(
            {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": f"{region}-{i}", "region": region, "score": score}
            for i in range(2)
        )
    config = BWERConfig(dataset="dummy", model="m", task="classification", tail_fraction=0.34, min_samples_per_slice=1)
    result = compute_bwer(rows, config, "region")
    assert result.summary["tail_slices"] == "B;C"
    assert result.summary["tail_risk"] == pytest.approx((0.5 + 0.2) / 2)
