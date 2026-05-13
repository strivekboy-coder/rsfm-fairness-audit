from __future__ import annotations

from rsfm_fairness_audit.bwer import BWERConfig, bootstrap_bwer


def _rows() -> list[dict[str, object]]:
    rows = []
    for event, region, score in [("e1", "A", 1.0), ("e2", "B", 0.0), ("e3", "C", 0.5)]:
        rows.extend(
            {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": f"{event}-{i}", "event_id": event, "region": region, "score": score}
            for i in range(4)
        )
    return rows


def test_ordinary_bootstrap_returns_deterministic_ci() -> None:
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=1)
    first = bootstrap_bwer(_rows(), config, "region", n_bootstrap=25, seed=7)
    second = bootstrap_bwer(_rows(), config, "region", n_bootstrap=25, seed=7)
    assert first["bootstrap_n"] == 25
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]


def test_cluster_bootstrap_returns_ci() -> None:
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=1)
    result = bootstrap_bwer(_rows(), config, "region", n_bootstrap=25, cluster_key="event_id", seed=9)
    assert result["bootstrap_method"] == "cluster"
    assert result["bootstrap_n"] > 0
