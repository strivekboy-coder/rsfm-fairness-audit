from __future__ import annotations

import pytest

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_table
from rsfm_fairness_audit.bwer import BWERConfig, compute_bwer
from rsfm_fairness_audit.io import read_csv_rows


def _rows() -> list[dict[str, object]]:
    return [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a0", "region": "A", "class_label": "0", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a1", "region": "A", "class_label": "1", "score": 0.5},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b0", "region": "B", "class_label": "0", "score": 0.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b0b", "region": "B", "class_label": "0", "score": 0.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "c0", "region": "C", "class_label": "0", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "c1", "region": "C", "class_label": "1", "score": 1.0},
    ]


def test_renormalize_preserves_available_level_behavior() -> None:
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=1)
    result = compute_bwer(_rows(), config, "region", "class_label")
    risks = {row["slice_value"]: row["balanced_risk"] for row in result.by_slice}
    assert risks["B"] == pytest.approx(1.0)
    assert {row["missing_balance_policy"] for row in result.support_diagnostics} == {"renormalize"}
    assert result.summary["missing_gz_count"] == 1


def test_invalidate_excludes_slices_with_missing_balance_levels() -> None:
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=1, missing_balance_policy="invalidate")
    result = compute_bwer(_rows(), config, "region", "class_label")
    by_slice = {row["slice_value"]: row for row in result.by_slice}
    assert by_slice["B"]["is_valid_slice"] is False
    assert by_slice["B"]["support_warning"] == "missing_balance_levels=1"
    assert result.summary["n_slices_valid"] == 2


def test_overlap_uses_only_shared_balance_levels() -> None:
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=1, missing_balance_policy="overlap")
    result = compute_bwer(_rows(), config, "region", "class_label")
    risks = {row["slice_value"]: row["balanced_risk"] for row in result.by_slice}
    assert result.summary["n_total_balance_levels"] == 2
    assert result.summary["n_used_balance_levels"] == 1
    assert risks["A"] == pytest.approx(0.0)
    assert risks["B"] == pytest.approx(1.0)
    assert risks["C"] == pytest.approx(0.0)
    used = [row for row in result.support_diagnostics if row["used_in_balanced_risk"]]
    assert {row["balance_level"] for row in used} == {"0"}


def test_overlap_ignores_invalid_tiny_slices_when_finding_shared_levels() -> None:
    rows = [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a0", "region": "A", "class_label": "0", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a1", "region": "A", "class_label": "1", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b0", "region": "B", "class_label": "0", "score": 0.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b1", "region": "B", "class_label": "1", "score": 0.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "tiny", "region": "C", "class_label": "0", "score": 1.0},
    ]
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=2, missing_balance_policy="overlap")
    result = compute_bwer(rows, config, "region", "class_label")
    by_slice = {row["slice_value"]: row for row in result.by_slice}
    assert result.summary["n_used_balance_levels"] == 2
    assert by_slice["C"]["is_valid_slice"] is False
    assert by_slice["A"]["balanced_risk"] == pytest.approx(0.0)
    assert by_slice["B"]["balanced_risk"] == pytest.approx(1.0)


def test_support_diagnostics_csv_is_created() -> None:
    output = "outputs/test_bwer_missing_policy"
    artifacts = evaluate_bwer_table(
        _rows(),
        "dummy",
        "m",
        "classification",
        output,
        slice_variable="region",
        balance_variable="class_label",
        bootstrap=0,
        missing_balance_policy="invalidate",
    )
    diagnostics = read_csv_rows(artifacts["support_diagnostics"])
    summary = read_csv_rows(artifacts["bwer_summary"])
    assert diagnostics
    assert {"slice_value", "balance_level", "has_support", "used_in_balanced_risk", "missing_balance_policy"} <= set(diagnostics[0])
    assert summary[0]["missing_balance_policy"] == "invalidate"
    assert summary[0]["missing_gz_count"] == "1"
