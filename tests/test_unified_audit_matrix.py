from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from rsfm_fairness_audit.io import read_csv_rows
from scripts.analysis.build_selective_risk_audit import build_selective_risk_audit
from scripts.analysis.build_unified_audit_matrix import build_unified_matrix


def _case_root(name: str) -> Path:
    root = Path("outputs") / f"test_unified_{name}_{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _write_registry(path: Path, prediction_table: Path | None = None) -> None:
    pred_block = ""
    if prediction_table is not None:
        pred_block = f"""
      prediction_table_candidates:
        toy_run:
          - {prediction_table.as_posix()}
"""
    path.write_text(
        f"""
version: test
defaults:
  output_unified: {path.parent.as_posix()}/unified
  output_selective: {path.parent.as_posix()}/selective
  coverages: [0.7, 0.9]
  tail_fraction: 0.1
  min_samples_per_slice: 1
experiments:
  - experiment_id: toy_event
    dataset: ToyEvent
    deployment_axis: event_disaster
    task_type: segmentation
    formal_status: formal
    result_level: formal_result
    protocol_summary: toy
    primary_metric_family: iou_risk
    aggregate_metric_name: aggregate_iou
    risk_metric_name: 1_minus_iou
    primary_bwer_slice: event_id
    standardised_balance: extent_bin
    formal_runs:
      - run_id: toy_model
        model_family: toy
        model_variant: toy
        split_protocol: test
        eval_scope: val
        aggregate_score: 0.8
        raw_bwer: 0.2
        standardised_bwer: 0.1
        worst_slice: hard_event
        data_source: fixture
    caveats:
      - toy caveat
    protocol_risks:
      - toy risk
    support_notes:
      - toy support
    claim_support:
      - claim: toy claim
        support: supported
        caveat: toy caveat
    selective_risk:
      availability: available_if_prediction_tables_present
      confidence_definition: max_probability
      confidence_column: confidence
      risk_column: risk
      slice_columns: [country]
{pred_block}
""",
        encoding="utf-8",
    )


def test_unified_registry_loading_and_metric_harmonization() -> None:
    pytest.importorskip("matplotlib")
    root = _case_root("matrix")
    registry = root / "registry.yaml"
    _write_registry(registry)

    artifacts = build_unified_matrix(registry, root / "unified")

    rows = read_csv_rows(artifacts["unified_main_results_table"])
    assert rows[0]["experiment_id"] == "toy_event"
    assert rows[0]["metric_family"] == "iou_risk"
    assert float(rows[0]["aggregate_risk"]) == pytest.approx(0.2)
    assert artifacts["figure_average_vs_bwer_cross_dataset"].exists()


def test_selective_missing_file_is_recorded_without_fabricating() -> None:
    pytest.importorskip("matplotlib")
    root = _case_root("missing")
    registry = root / "registry.yaml"
    _write_registry(registry, root / "does_not_exist.csv")

    artifacts = build_selective_risk_audit(registry, root / "selective")

    availability = read_csv_rows(artifacts["confidence_availability_audit"])
    assert availability[0]["status"] == "unavailable_missing_prediction_table"
    summary = read_csv_rows(artifacts["selective_risk_summary_cross_dataset"])
    assert summary[0]["status"] == "unavailable"
    assert summary[0]["slice_variable"] == "unavailable"


def test_selective_risk_from_tiny_prediction_table_and_figures() -> None:
    pytest.importorskip("matplotlib")
    root = _case_root("selective")
    predictions = root / "predictions.csv"
    predictions.write_text(
        "sample_id,country,confidence,risk\n"
        "a,DE,0.9,0\n"
        "b,DE,0.8,1\n"
        "c,FR,0.7,0\n"
        "d,FR,0.1,1\n",
        encoding="utf-8",
    )
    registry = root / "registry.yaml"
    _write_registry(registry, predictions)

    artifacts = build_selective_risk_audit(registry, root / "selective")

    summary = read_csv_rows(artifacts["selective_risk_summary_cross_dataset"])
    retained = read_csv_rows(artifacts["retained_coverage_by_slice"])
    assert summary
    assert retained
    assert artifacts["figure_selective_risk_curves_cross_dataset"].exists()


def test_reben_unified_reads_bwer_and_mean_bce_risk_from_completed_outputs() -> None:
    pytest.importorskip("matplotlib")
    root = _case_root("reben")
    comparison = root / "comparison"
    comparison.mkdir()
    (comparison / "aggregate_sensor_mode_comparison.csv").write_text(
        "sensor_mode,macro_ap,micro_ap,macro_f1,micro_f1,mean_bce_risk\n"
        "S1,0.4,0.5,0.3,0.4,0.6\n"
        "S2,0.5,0.6,0.4,0.5,0.5\n"
        "S1+S2,0.6,0.7,0.5,0.6,0.4\n",
        encoding="utf-8",
    )
    (comparison / "bce_bwer_sensor_mode_comparison.csv").write_text(
        "sensor_mode,risk_name,slice_variable,balance_variable,bwer,worst_slice,tail_slices\n"
        "S1,risk_bce,country,,0.2,FR,FR\n"
        "S1,risk_bce,country,class_label,0.21,FR,FR\n"
        "S2,risk_bce,country,,0.15,FR,FR\n"
        "S2,risk_bce,country,class_label,0.16,FR,FR\n"
        "S1+S2,risk_bce,country,,0.1,FR,FR\n"
        "S1+S2,risk_bce,country,class_label,0.11,FR,FR\n",
        encoding="utf-8",
    )
    registry = root / "registry.yaml"
    registry.write_text(
        f"""
version: test
defaults:
  output_unified: {root.as_posix()}/unified
experiments:
  - experiment_id: reben_croma_sensor_mode
    dataset: BigEarthNet v2 / reBEN
    deployment_axis: sensor_modality
    task_type: multi_label_classification
    formal_status: formal_partial
    result_level: formal_result
    protocol_summary: toy
    output_dir_candidates:
      - {comparison.as_posix()}
    primary_metric_family: bce_risk
    aggregate_metric_name: macro_ap
    risk_metric_name: labelwise_bce
    primary_bwer_slice: country
    standardised_balance: class_label
    formal_runs:
      - run_id: croma_s1
        sensor_mode: S1
        aggregate_score: 0.0
        mean_bce_risk: 9.0
      - run_id: croma_s2
        sensor_mode: S2
        aggregate_score: 0.0
        mean_bce_risk: 9.0
      - run_id: croma_s1_plus_s2
        sensor_mode: S1+S2
        aggregate_score: 0.0
        mean_bce_risk: 9.0
""",
        encoding="utf-8",
    )

    artifacts = build_unified_matrix(registry, root / "unified")
    rows = {row["sensor_mode"]: row for row in read_csv_rows(artifacts["unified_main_results_table"])}
    assert rows["S1+S2"]["aggregate_score"] == "0.6"
    assert rows["S1+S2"]["aggregate_risk"] == "0.4"
    assert rows["S1+S2"]["raw_bwer"] == "0.1"
    assert rows["S1+S2"]["standardised_bwer"] == "0.11"


def test_selective_reads_reben_per_run_summaries_when_comparison_missing() -> None:
    pytest.importorskip("matplotlib")
    root = _case_root("reben_selective")
    s1_outer = root / "reben_croma_sensor_mode_audit_croma_s1_full"
    s1_nested = s1_outer / "content" / "outputs" / "reben_croma_sensor_mode_audit_croma_s1_full"
    s1_nested.mkdir(parents=True)
    (s1_nested / "selective_risk_summary.csv").write_text(
        "coverage_target,slice_variable,slice_value,confidence_threshold,retained_count,total_count,retained_coverage,abstention_rate,mean_risk\n"
        "0.8,all,all,0.4,80,100,0.8,0.2,0.3\n"
        "0.8,country,DE,0.4,40,50,0.8,0.2,0.2\n"
        "0.8,country,FR,0.4,40,50,0.8,0.2,0.4\n",
        encoding="utf-8",
    )
    registry = root / "registry.yaml"
    registry.write_text(
        f"""
version: test
defaults:
  output_selective: {root.as_posix()}/selective
  coverages: [0.8]
experiments:
  - experiment_id: reben_croma_sensor_mode
    dataset: BigEarthNet v2 / reBEN
    deployment_axis: sensor_modality
    task_type: multi_label_classification
    selective_risk:
      availability: available
      confidence_definition: label_probability_confidence
      source_run_summary_candidates:
        croma_s1:
          sensor_mode: S1
          paths:
            - {s1_outer.as_posix()}/selective_risk_summary.csv
        croma_s2:
          sensor_mode: S2
          paths:
            - {root.as_posix()}/missing.csv
""",
        encoding="utf-8",
    )

    artifacts = build_selective_risk_audit(registry, root / "selective")
    summary = read_csv_rows(artifacts["selective_risk_summary_cross_dataset"])
    retained = read_csv_rows(artifacts["retained_coverage_by_slice"])
    bwer = read_csv_rows(artifacts["selective_bwer_summary"])
    assert any(row["run_id"] == "croma_s1" and row["slice_variable"] == "country" for row in summary)
    assert any(row["run_id"] == "croma_s2" and row["status"] == "unavailable" for row in summary)
    assert retained
    assert any(row["run_id"] == "croma_s1" and row["slice_variable"] == "country" for row in bwer)
