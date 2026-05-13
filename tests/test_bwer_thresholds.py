from __future__ import annotations

import pytest

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_table
from rsfm_fairness_audit.bwer import BWERConfig, compute_bwer
from rsfm_fairness_audit.io import read_csv_rows


def test_small_slices_are_excluded_but_written_to_by_slice() -> None:
    rows = [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a1", "region": "A", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b1", "region": "B", "score": 0.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b2", "region": "B", "score": 0.0},
    ]
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=2)
    result = compute_bwer(rows, config, "region")
    by_slice = {row["slice_value"]: row for row in result.by_slice}
    assert by_slice["A"]["is_valid_slice"] is False
    assert by_slice["B"]["is_valid_slice"] is True
    assert "min_slices_required" in result.summary["warnings"]


def test_min_positive_support_can_invalidate_slice() -> None:
    rows = [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a1", "region": "A", "class_label": "0", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b1", "region": "B", "class_label": "1", "score": 0.0},
    ]
    config = BWERConfig(dataset="dummy", model="m", task="classification", min_samples_per_slice=1, min_positive_support=2)
    result = compute_bwer(rows, config, "region")
    assert all(row["is_valid_slice"] is False for row in result.by_slice)


def test_pipeline_writes_warning_when_all_slices_are_invalid() -> None:
    rows = [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a1", "region": "A", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b1", "region": "B", "score": 0.0},
    ]
    artifacts = evaluate_bwer_table(rows, "dummy", "m", "classification", "outputs/test_bwer_all_invalid", slice_variable="region")
    summary = read_csv_rows(artifacts["bwer_summary"])
    assert summary
    assert summary[0]["n_slices_valid"] == "0"
    assert "Only 0 valid slices" in summary[0]["warnings"]


def test_missing_slice_override_fails_clearly_and_writes_warning() -> None:
    rows = [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a1", "region": "A", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b1", "region": "B", "score": 0.0},
    ]
    output = "outputs/test_bwer_missing_slice_override"
    with pytest.raises(ValueError, match="no valid BWER variants produced"):
        evaluate_bwer_table(rows, "dummy", "m", "classification", output, slice_variable="country")
    warnings_text = __import__("pathlib").Path(output, "warnings.json").read_text(encoding="utf-8")
    assert "no valid BWER variants produced" in warnings_text
