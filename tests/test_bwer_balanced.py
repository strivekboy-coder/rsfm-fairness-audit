from __future__ import annotations

import pytest

from rsfm_fairness_audit.bwer import BWERConfig, compute_bwer


def test_class_balanced_bwer_uniform_weighting() -> None:
    rows = []
    for region, cls, scores in [
        ("A", "0", [1.0, 1.0]),
        ("A", "1", [0.5, 0.5]),
        ("B", "0", [0.5, 0.5]),
        ("B", "1", [0.0, 0.0]),
    ]:
        rows.extend(
            {
                "dataset": "dummy",
                "model": "m",
                "task": "classification",
                "split": "all",
                "unit_id": f"{region}-{cls}-{i}",
                "region": region,
                "class_label": cls,
                "score": score,
            }
            for i, score in enumerate(scores)
        )
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=1)
    result = compute_bwer(rows, config, "region", "class_label")
    risks = {row["slice_value"]: row["balanced_risk"] for row in result.by_slice}
    assert risks["A"] == pytest.approx(0.25)
    assert risks["B"] == pytest.approx(0.75)
    assert result.summary["bwer"] == pytest.approx(0.25)


def test_empirical_weighting_and_missing_combinations_warn() -> None:
    rows = [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a", "region": "A", "class_label": "0", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b", "region": "B", "class_label": "0", "score": 0.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "c", "region": "B", "class_label": "1", "score": 1.0},
    ]
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=1, weighting="empirical")
    result = compute_bwer(rows, config, "region", "class_label")
    assert any("missing class_label levels" in warning for warning in result.warnings)
    risks = {row["slice_value"]: row["balanced_risk"] for row in result.by_slice}
    assert risks["A"] == pytest.approx(0.0)
