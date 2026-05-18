from __future__ import annotations

from pathlib import Path

import pytest

from rsfm_fairness_audit.audit_pipeline import _dataset_taxonomy, evaluate_bwer_table
from rsfm_fairness_audit.audit_table import build_audit_table_from_segmentation_metrics, validate_audit_table
from rsfm_fairness_audit.bwer import BWERConfig, compute_bwer, compute_bwer_family
from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.segmentation import aggregate_segmentation_metrics
from rsfm_fairness_audit.slice_support import evaluate_slice_support


def test_classification_support_remains_sample_based() -> None:
    rows = [
        {"dataset": "d", "model": "m", "task": "classification", "split": "all", "unit_id": "a", "region": "A", "score": 1.0, "valid_pixel_count": 10000},
        {"dataset": "d", "model": "m", "task": "classification", "split": "all", "unit_id": "b", "region": "B", "score": 0.0, "valid_pixel_count": 10000},
    ]
    result = compute_bwer(rows, BWERConfig(dataset="d", model="m", task="classification", min_units_required=1), "region")
    assert {row["slice_value"]: row["n_units"] for row in result.by_slice} == {"A": 1, "B": 1}


def test_segmentation_support_uses_effective_pixel_support() -> None:
    rows = [
        {"dataset": "d", "model": "m", "task": "segmentation", "split": "all", "unit_id": "e1", "event_id": "E1", "TP": 30, "FP": 10, "FN": 10, "TN": 50, "valid_pixel_count": 100},
        {"dataset": "d", "model": "m", "task": "segmentation", "split": "all", "unit_id": "e2", "event_id": "E2", "TP": 10, "FP": 30, "FN": 10, "TN": 50, "valid_pixel_count": 100},
    ]
    result = compute_bwer(rows, BWERConfig(dataset="d", model="m", task="segmentation", min_units_required=50), "event_id")
    by_slice = {row["slice_value"]: row for row in result.by_slice}
    assert by_slice["E1"]["n_units"] == 100
    assert by_slice["E1"]["n_positive"] == 40
    assert by_slice["E1"]["positive_pixel_support"] == 40
    assert by_slice["E1"]["risk_source"] == "1_minus_iou"


def test_sen1floods11_classification_and_segmentation_taxonomies_are_distinct() -> None:
    classification, _ = _dataset_taxonomy("configs/slice_taxonomy.yaml", "sen1floods11", "classification")
    segmentation, _ = _dataset_taxonomy("configs/slice_taxonomy.yaml", "sen1floods11", "segmentation")
    assert classification["min_positive_support"] is None
    assert segmentation["min_positive_support"] == 1000
    assert segmentation["min_valid_pixel_support"] == 1000
    assert classification["task_type"] == "classification"
    assert segmentation["task_type"] == "segmentation"


def test_invalid_balance_variable_is_marked_not_runnable() -> None:
    output = Path("outputs/test_invalid_balance_variable")
    rows = [
        {"dataset": "d", "model": "m", "task": "classification", "split": "all", "unit_id": "a", "event_id": "A", "event": "A", "score": 1.0},
        {"dataset": "d", "model": "m", "task": "classification", "split": "all", "unit_id": "b", "event_id": "B", "event": "B", "score": 0.0},
    ]
    artifacts = evaluate_slice_support(rows, "d", "m", "classification", output, candidates=["event_id|event"])
    recs = read_csv_rows(artifacts["recommendations"])
    assert recs[0]["formal_bwer_runnable"] == "False"
    assert "deterministic proxy" in recs[0]["reason"] or "identical" in recs[0]["reason"]
    summary, *_rest, warnings = compute_bwer_family(rows, BWERConfig(dataset="d", model="m", task="classification"), ["event_id"], ["event"])
    assert summary == []
    assert any("Skipping invalid BWER(event_id | event)" in warning for warning in warnings)


def test_segmentation_event_metrics_are_aggregated_from_counts() -> None:
    rows = [
        {"event_id": "E", "dataset": "sen1floods11", "model": "prithvi", "split": "all", "region": "R", "TP": 1, "FP": 1, "FN": 0, "TN": 2, "valid_pixel_count": 4, "positive_pixel_count": 1, "predicted_positive_pixel_count": 2},
        {"event_id": "E", "dataset": "sen1floods11", "model": "prithvi", "split": "all", "region": "R", "TP": 1, "FP": 0, "FN": 1, "TN": 2, "valid_pixel_count": 4, "positive_pixel_count": 2, "predicted_positive_pixel_count": 1},
    ]
    aggregated = aggregate_segmentation_metrics(rows, "event_id")
    assert aggregated[0]["TP"] == 2
    assert aggregated[0]["FP"] == 1
    assert aggregated[0]["FN"] == 1
    assert aggregated[0]["valid_pixel_count"] == 8
    assert aggregated[0]["micro_iou"] == pytest.approx(2 / 4)
    assert aggregated[0]["micro_dice"] == pytest.approx(4 / 6)


def test_segmentation_outputs_can_feed_bwer_and_preserve_protocol_metadata() -> None:
    output = Path("outputs/test_segmentation_bwer_v2")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "event_segmentation_metrics.csv"
    rows = [
        {"dataset": "sen1floods11", "model": "prithvi", "task": "segmentation", "split": "all", "unit_id": "E1", "sample_id": "E1", "event_id": "E1", "aggregation_level": "event", "TP": 900, "FP": 50, "FN": 100, "TN": 1000, "valid_pixel_count": 2050, "positive_pixel_count": 1000, "micro_iou": 900 / 1050, "risk": 1 - (900 / 1050), "input_mode": "S2", "adaptation_protocol": "frozen_encoder_lightweight_head", "training_budget": "unsupervised_threshold_head", "split_protocol": "event_held_out"},
        {"dataset": "sen1floods11", "model": "prithvi", "task": "segmentation", "split": "all", "unit_id": "E2", "sample_id": "E2", "event_id": "E2", "aggregation_level": "event", "TP": 600, "FP": 200, "FN": 400, "TN": 1000, "valid_pixel_count": 2200, "positive_pixel_count": 1000, "micro_iou": 600 / 1200, "risk": 1 - (600 / 1200), "input_mode": "S2", "adaptation_protocol": "frozen_encoder_lightweight_head", "training_budget": "unsupervised_threshold_head", "split_protocol": "event_held_out"},
    ]
    write_csv(metrics_path, rows)
    audit_rows = build_audit_table_from_segmentation_metrics(metrics_path, dataset="sen1floods11", model="prithvi", task="segmentation", score_column="micro_iou")
    validate_audit_table(audit_rows)
    artifacts = evaluate_bwer_table(audit_rows, "sen1floods11", "prithvi", "segmentation", output / "bwer", slice_variable="event_id", balance_variable="raw")
    summary = read_csv_rows(artifacts["bwer_summary"])
    assert summary[0]["task_type"] == "segmentation"
    assert summary[0]["adaptation_protocol"] == "frozen_encoder_lightweight_head"
    assert summary[0]["split_protocol"] == "event_held_out"
