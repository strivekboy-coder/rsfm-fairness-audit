from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from rsfm_fairness_audit.io import read_csv_rows
from scripts.analysis.build_bwer_robustness_v1 import build_bwer_robustness
from scripts.analysis.check_audit_contract import build_audit_contract_coverage


def _case_root(name: str) -> Path:
    root = Path("outputs") / f"test_audit_contract_{name}_{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def test_audit_contract_schema_loading_smoke() -> None:
    root = _case_root("schema")
    predictions = root / "predictions.csv"
    predictions.write_text(
        "sample_id,split,dataset,task_type,model,y_true,y_pred,confidence,risk,country,class_label\n"
        "a,val,Toy,single_label_classification,m,1,1,0.9,0,DE,built\n",
        encoding="utf-8",
    )
    registry = root / "registry.yaml"
    registry.write_text(
        f"""
version: test
experiments:
  - experiment_id: toy
    dataset: Toy
    task_type: single_label_classification
    standardised_balance: class_label
    formal_runs:
      - run_id: toy_run
    selective_risk:
      availability: available_if_prediction_tables_present
      prediction_table_candidates:
        toy_run:
          - {predictions.as_posix()}
""",
        encoding="utf-8",
    )
    artifacts = build_audit_contract_coverage(registry, root / "contract")
    rows = read_csv_rows(artifacts["audit_contract_coverage"])
    assert rows[0]["task_type"] == "single_label_classification"
    assert rows[0]["supports_raw_bwer"] == "True"
    assert rows[0]["supports_selective_risk"] == "True"


def test_missing_fields_detection_marks_selective_limited() -> None:
    root = _case_root("missing")
    predictions = root / "predictions.csv"
    predictions.write_text(
        "sample_id,split,dataset,task_type,model,y_true,y_pred,risk,country,class_label\n"
        "a,val,Toy,single_label_classification,m,1,0,1,DE,built\n",
        encoding="utf-8",
    )
    registry = root / "registry.yaml"
    registry.write_text(
        f"""
version: test
experiments:
  - experiment_id: toy
    dataset: Toy
    task_type: single_label_classification
    standardised_balance: class_label
    formal_runs:
      - run_id: toy_run
    selective_risk:
      availability: available_if_prediction_tables_present
      prediction_table_candidates:
        toy_run:
          - {predictions.as_posix()}
""",
        encoding="utf-8",
    )
    artifacts = build_audit_contract_coverage(registry, root / "contract")
    missing = read_csv_rows(artifacts["missing_fields_by_experiment"])
    rerun = read_csv_rows(artifacts["rerun_requirements"])
    assert any(row["category"] == "score" for row in missing)
    assert rerun[0]["requirement"] == "requires_probability_or_logit_export"


def test_bwer_robustness_metric_comparison_tiny_registry() -> None:
    root = _case_root("robust")
    registry = root / "registry.yaml"
    registry.write_text(
        """
version: test
defaults:
  alphas: [0.05, 0.1, 0.2]
  support_thresholds: [10, 20]
experiments:
  - experiment_id: toy
    dataset: Toy
    task_type: single_label_classification
    deployment_axis: geography
    primary_metric_family: classification_error
    primary_bwer_slice: country
    standardised_balance: class_label
    formal_runs:
      - run_id: model_a
        model_family: toy
        aggregate_score: 0.8
        raw_bwer: 0.12
        standardised_bwer: 0.10
        worst_slice: DE
        tail_slices: DE
""",
        encoding="utf-8",
    )
    artifacts = build_bwer_robustness(registry, root / "robustness")
    comparison = read_csv_rows(artifacts["bwer_metric_comparison"])
    alpha = read_csv_rows(artifacts["alpha_sensitivity_summary"])
    assert comparison[0]["aggregate_risk"] == "0.19999999999999996"
    assert comparison[0]["raw_bwer"] == "0.12"
    assert any(row["alpha"] == "0.1" and row["bwer"] == "0.12" for row in alpha)


def test_contract_missing_file_handling_is_graceful() -> None:
    root = _case_root("nofile")
    registry = root / "registry.yaml"
    registry.write_text(
        f"""
version: test
experiments:
  - experiment_id: toy
    dataset: Toy
    task_type: single_label_classification
    formal_runs:
      - run_id: toy_run
    selective_risk:
      availability: available_if_prediction_tables_present
      prediction_table_candidates:
        toy_run:
          - {root.as_posix()}/does_not_exist.csv
""",
        encoding="utf-8",
    )
    artifacts = build_audit_contract_coverage(registry, root / "contract")
    coverage = read_csv_rows(artifacts["audit_contract_coverage"])
    assert coverage[0]["artifact_status"] == "documented_record_only"
    assert coverage[0]["supports_raw_bwer"] == "False"
