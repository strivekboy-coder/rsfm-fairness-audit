from __future__ import annotations

import pytest

from rsfm_fairness_audit.audit_table import (
    AuditTableError,
    build_audit_table_from_predictions,
    build_audit_table_from_segmentation_metrics,
    validate_audit_table,
)
from rsfm_fairness_audit.io import write_csv


def test_predictions_can_be_converted_into_audit_table() -> None:
    root = __import__("pathlib").Path("outputs/test_audit_table_predictions")
    root.mkdir(parents=True, exist_ok=True)
    predictions = root / "predictions.csv"
    write_csv(predictions, [{"sample_id": "s1", "label": 1, "prediction": 1, "region": "A"}])
    rows = build_audit_table_from_predictions(predictions, dataset="d", model="m", task="classification")
    assert rows[0]["score"] == 1.0
    assert rows[0]["region"] == "A"


def test_missing_required_columns_produce_clear_error() -> None:
    with pytest.raises(AuditTableError, match="score/risk"):
        validate_audit_table([{"dataset": "d", "model": "m", "task": "t", "split": "all", "unit_id": "u"}])


def test_malformed_later_row_produces_row_specific_error() -> None:
    rows = [
        {"dataset": "d", "model": "m", "task": "t", "split": "all", "unit_id": "u1", "score": 0.5},
        {"dataset": "d", "model": "m", "task": "t", "split": "all", "unit_id": "u2", "score": "bad"},
    ]
    with pytest.raises(AuditTableError, match="row 2.*score"):
        validate_audit_table(rows)


def test_nan_score_row_is_rejected() -> None:
    rows = [{"dataset": "d", "model": "m", "task": "t", "split": "all", "unit_id": "u1", "score": "nan"}]
    with pytest.raises(AuditTableError, match="score/risk"):
        validate_audit_table(rows)


def test_infinite_risk_row_is_rejected() -> None:
    rows = [{"dataset": "d", "model": "m", "task": "t", "split": "all", "unit_id": "u1", "risk": "inf"}]
    with pytest.raises(AuditTableError, match="risk must be finite"):
        validate_audit_table(rows)


def test_random_chip_split_is_valid_split_protocol() -> None:
    validate_audit_table(
        [
            {
                "dataset": "sen1floods11",
                "model": "unet_sen1floods11_s2_512",
                "task": "segmentation",
                "split": "test",
                "unit_id": "Pakistan",
                "score": 0.64,
                "risk": 0.36,
                "adaptation_protocol": "supervised_baseline",
                "split_protocol": "random_chip_split",
            }
        ]
    )


def test_segmentation_metrics_convert_to_audit_table() -> None:
    root = __import__("pathlib").Path("outputs/test_audit_table_segmentation")
    root.mkdir(parents=True, exist_ok=True)
    metrics = root / "seg.csv"
    write_csv(metrics, [{"sample_id": "s1", "water_iou": 0.4, "event": "E"}])
    rows = build_audit_table_from_segmentation_metrics(metrics, dataset="sen1floods11", model="prithvi")
    assert rows[0]["score"] == "0.4"
    assert rows[0]["risk"] == 0.6
    assert rows[0]["event"] == "E"
