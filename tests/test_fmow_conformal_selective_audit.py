from __future__ import annotations

import math
import uuid
from pathlib import Path

from rsfm_fairness_audit.io import read_csv_rows
from scripts.analysis.build_fmow_conformal_selective_audit import (
    build_fmow_conformal_selective_audit,
    build_grouped_calibration_split,
    conformal_threshold_from_true_probability,
    rank_divergence_rows,
    selective_risk_rows,
    support_filtered_slice_summary,
    true_class_probability,
)


def test_grouped_calibration_split_keeps_locations_disjoint() -> None:
    rows = [
        {"sample_id": "a", "location_id": "loc1"},
        {"sample_id": "b", "location_id": "loc1"},
        {"sample_id": "c", "location_id": "loc2"},
        {"sample_id": "d", "location_id": "loc2"},
    ]

    split, report = build_grouped_calibration_split(rows, seed=7, calibration_fraction=0.5)

    calibration_groups = {row["location_id"] for row in split if row["calibration_split"] == "calibration"}
    test_groups = {row["location_id"] for row in split if row["calibration_split"] == "test"}
    assert calibration_groups.isdisjoint(test_groups)
    assert report["group_column"] == "location_id"


def test_selective_risk_tiny_table_reduces_average_risk() -> None:
    rows = [
        {"_confidence": 0.95, "_risk": 0.0, "country": "A", "region": "R1", "class_label": "x"},
        {"_confidence": 0.85, "_risk": 0.0, "country": "A", "region": "R1", "class_label": "y"},
        {"_confidence": 0.30, "_risk": 1.0, "country": "B", "region": "R2", "class_label": "x"},
        {"_confidence": 0.20, "_risk": 1.0, "country": "B", "region": "R2", "class_label": "y"},
    ]

    summary, retained, high_conf = selective_risk_rows(rows, run_id="tiny", coverages=[0.5], selector_name="confidence_topk_test")

    assert summary[0]["mean_risk"] < summary[0]["baseline_mean_risk"]
    assert retained
    assert high_conf


def test_true_class_probability_missing_vector_is_graceful() -> None:
    assert math.isnan(true_class_probability({"label": "1", "max_probability": "0.9"}))
    assert true_class_probability({"label": "1", "probabilities": "[0.2, 0.7, 0.1]"}) == 0.7


def test_conformal_threshold_from_true_probability() -> None:
    rows = [{"true_probability": "0.9"}, {"true_probability": "0.8"}, {"true_probability": "0.6"}]

    qhat = conformal_threshold_from_true_probability(rows, alpha=0.1)

    assert 0.0 <= qhat <= 0.4


def test_support_filtered_summary_excludes_low_support_slice() -> None:
    rows = [
        {"run_id": "a", "selector": "s", "coverage_target": 0.8, "slice_variable": "country", "slice_value": "tiny", "retained_count": 1, "total_count": 1, "mean_risk": 1.0},
        {"run_id": "a", "selector": "s", "coverage_target": 0.8, "slice_variable": "country", "slice_value": "big", "retained_count": 20, "total_count": 25, "mean_risk": 0.8},
    ]

    summary = support_filtered_slice_summary(rows, min_support=10)

    assert len(summary) == 1
    assert summary[0]["slice_value"] == "big"
    assert summary[0]["n_excluded_low_support_slices"] == 1


def test_rank_divergence_rows_compare_aggregate_and_bwer_best() -> None:
    aggregate = [
        {"run_id": "resnet50_13band", "selector": "baseline_all_test", "coverage_target": 1.0, "mean_risk": 0.7},
        {"run_id": "dofa_scaled10000", "selector": "baseline_all_test", "coverage_target": 1.0, "mean_risk": 0.8},
    ]
    bwer = [
        {"run_id": "resnet50_13band", "selector": "baseline_all_test", "coverage_target": 1.0, "analysis_type": "raw", "bwer": 0.2},
        {"run_id": "dofa_scaled10000", "selector": "baseline_all_test", "coverage_target": 1.0, "analysis_type": "raw", "bwer": 0.1},
    ]

    rows = rank_divergence_rows(aggregate, bwer, scenario_label="tiny")

    assert rows[0]["aggregate_best_run"] == "resnet50_13band"
    assert rows[0]["bwer_best_run"] == "dofa_scaled10000"
    assert rows[0]["rank_diverges"] is True


def test_fmow_builder_writes_outputs_with_confidence_diagnostic() -> None:
    root = Path("outputs") / f"test_fmow_conformal_selective_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    audit = root / "audit.csv"
    audit.write_text(
        "sample_id,location_id,country,region,class_label,risk,confidence,max_probability,correct\n"
        "a,l1,A,R1,x,0,0.95,0.95,True\n"
        "b,l1,A,R1,y,0,0.90,0.90,True\n"
        "c,l2,B,R2,x,1,0.40,0.40,False\n"
        "d,l2,B,R2,y,1,0.30,0.30,False\n"
        "e,l3,C,R3,x,0,0.80,0.80,True\n"
        "f,l3,C,R3,y,1,0.20,0.20,False\n",
        encoding="utf-8",
    )
    drive = root / "drive_real_audit_v1"
    drive.mkdir()
    for name in ["audit_contract_report.md", "audit_contract_coverage.csv", "missing_fields_by_experiment.csv", "rerun_requirements.csv"]:
        (drive / name).write_text("ok\n", encoding="utf-8")

    artifacts = build_fmow_conformal_selective_audit(
        output_dir=root / "out",
        drive_audit_dir=drive,
        audit_tables={"resnet50_13band": audit},
        coverages=[0.5],
        seed=1,
        min_samples_per_slice=1,
        unified_output_dir=root / "unified_v2",
    )

    assert artifacts["fmow_selective_risk_summary"].exists()
    assert artifacts["fmow_support_filtered_slice_summary"].exists()
    assert artifacts["rank_divergence_under_selective_audit"].exists()
    conformal = read_csv_rows(artifacts["fmow_conformal_coverage_summary"])
    assert conformal[0]["formal_label_coverage_claim"] == "no"
    rank = read_csv_rows(artifacts["rank_divergence_under_selective_audit"])
    assert {"baseline_all_test", "confidence_topk_test", "calibrated_confidence_threshold_diagnostic"}.issubset({row["selector"] for row in rank})
    assert artifacts["unified_v2_paper_ready_report"].exists()
    assert Path(artifacts["figure_coverage_vs_risk"]).exists()
