from __future__ import annotations

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_table
from rsfm_fairness_audit.bwer import create_interaction_slice
from rsfm_fairness_audit.config import load_yaml


def test_slice_taxonomy_config_loads() -> None:
    config = load_yaml("configs/slice_taxonomy.yaml")
    assert "dummy" in config["datasets"]
    assert "sen1floods11_classification" in config["datasets"]
    assert "sen1floods11_segmentation" in config["datasets"]
    assert config["datasets"]["sen1floods11_classification"]["min_positive_support"] is None
    assert config["datasets"]["sen1floods11_segmentation"]["min_positive_support"] == 1000


def test_missing_taxonomy_columns_are_skipped_with_warning() -> None:
    from pathlib import Path

    rows = [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a", "region": "A", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b", "region": "B", "score": 0.0},
    ]
    artifacts = evaluate_bwer_table(rows, "dummy", "m", "classification", Path("outputs/test_slice_taxonomy"), bootstrap=0)
    assert artifacts["warnings"].exists()
    assert artifacts["bwer_summary"].exists()


def test_interaction_slices_can_be_created() -> None:
    rows = [{"country": "A", "class_label": "water"}]
    output = create_interaction_slice(rows, ["country", "class_label"], "country__class_label")
    assert output[0]["country__class_label"] == "A__water"
