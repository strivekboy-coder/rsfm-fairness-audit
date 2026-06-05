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
    summary = artifacts["selective_risk_summary_cross_dataset"].read_text(encoding="utf-8")
    assert summary == ""


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
