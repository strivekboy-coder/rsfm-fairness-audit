from __future__ import annotations

from pathlib import Path

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_table
from rsfm_fairness_audit.cli import main
from rsfm_fairness_audit.io import read_csv_rows, write_csv


def test_evaluate_bwer_cli_writes_outputs(monkeypatch) -> None:
    root = Path("outputs/test_cli_bwer")
    root.mkdir(parents=True, exist_ok=True)
    audit_table = root / "audit.csv"
    write_csv(
        audit_table,
        [
            {"dataset": "dummy", "model": "dummy", "task": "classification", "split": "all", "unit_id": "a1", "region": "A", "class_label": "0", "score": 1.0},
            {"dataset": "dummy", "model": "dummy", "task": "classification", "split": "all", "unit_id": "a2", "region": "A", "class_label": "1", "score": 0.5},
            {"dataset": "dummy", "model": "dummy", "task": "classification", "split": "all", "unit_id": "b1", "region": "B", "class_label": "0", "score": 0.0},
            {"dataset": "dummy", "model": "dummy", "task": "classification", "split": "all", "unit_id": "b2", "region": "B", "class_label": "1", "score": 0.0},
        ],
    )
    output = root / "audit_out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "evaluate-bwer",
            "--audit-table",
            str(audit_table),
            "--dataset",
            "dummy",
            "--model",
            "dummy",
            "--task",
            "classification",
            "--slice-variable",
            "region",
            "--output-dir",
            str(output),
            "--bootstrap",
            "5",
        ],
    )
    main()
    assert (output / "bwer_summary.csv").exists()
    assert (output / "figures" / "average_vs_bwer.png").exists()
    assert read_csv_rows(output / "bwer_summary.csv")


def test_run_audit_cli_accepts_run_real_style_predictions(monkeypatch) -> None:
    root = Path("outputs/test_cli_run_audit_predictions")
    root.mkdir(parents=True, exist_ok=True)
    predictions = root / "predictions.csv"
    write_csv(
        predictions,
        [
            {"sample_id": "s1", "label": "forest", "prediction": "forest", "region": "A", "sensor": "S2", "correct": 1},
            {"sample_id": "s2", "label": "water", "prediction": "forest", "region": "B", "sensor": "S2", "correct": 0},
        ],
    )
    output = root / "audit_out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "run-audit",
            "--predictions",
            str(predictions),
            "--dataset",
            "bigearthnet",
            "--model",
            "dofa",
            "--task",
            "classification",
            "--slice-variable",
            "region",
            "--output-dir",
            str(output),
        ],
    )
    main()
    assert read_csv_rows(output / "audit_table.csv")[0]["class_label"] == "forest"
    assert read_csv_rows(output / "bwer_summary.csv")


def test_run_audit_cli_accepts_prithvi_segmentation_metrics(monkeypatch) -> None:
    root = Path("outputs/test_cli_run_audit_segmentation")
    root.mkdir(parents=True, exist_ok=True)
    metrics = root / "segmentation_metrics.csv"
    write_csv(
        metrics,
        [
            {"sample_id": "e1_0", "event_id": "e1", "water_iou": 0.8, "positive_pixels": 1200},
            {"sample_id": "e2_0", "event_id": "e2", "water_iou": 0.2, "positive_pixels": 1300},
        ],
    )
    output = root / "audit_out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "run-audit",
            "--segmentation-metrics",
            str(metrics),
            "--dataset",
            "sen1floods11",
            "--model",
            "prithvi",
            "--task",
            "segmentation",
            "--slice-variable",
            "event_id",
            "--output-dir",
            str(output),
        ],
    )
    main()
    assert read_csv_rows(output / "audit_table.csv")[0]["class_label"] == "water"
    assert read_csv_rows(output / "bwer_summary.csv")


def test_taxonomy_alias_warning_is_written() -> None:
    rows = [
        {"dataset": "ben_ge", "model": "croma", "task": "classification", "split": "all", "unit_id": "a1", "sensor_mode": "sar", "score": 1.0},
        {"dataset": "ben_ge", "model": "croma", "task": "classification", "split": "all", "unit_id": "b1", "sensor_mode": "both", "score": 0.0},
    ]
    artifacts = evaluate_bwer_table(rows, "ben_ge", "croma", "classification", "outputs/test_bwer_taxonomy_alias", slice_variable="sensor_mode")
    warning_text = artifacts["warnings"].read_text(encoding="utf-8")
    assert "dataset=ben_ge -&gt;" not in warning_text
    assert "dataset=ben_ge -> ben_ge_800" in warning_text


def test_missing_taxonomy_name_suggests_known_datasets() -> None:
    rows = [
        {"dataset": "unknown", "model": "m", "task": "classification", "split": "all", "unit_id": "a1", "region": "A", "score": 1.0},
        {"dataset": "unknown", "model": "m", "task": "classification", "split": "all", "unit_id": "b1", "region": "B", "score": 0.0},
    ]
    artifacts = evaluate_bwer_table(rows, "unknown", "m", "classification", "outputs/test_bwer_missing_taxonomy", slice_variable="region")
    assert "Known dataset taxonomy names" in artifacts["warnings"].read_text(encoding="utf-8")


def test_cluster_bootstrap_with_one_cluster_writes_warning() -> None:
    rows = [
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "a1", "event_id": "e1", "region": "A", "score": 1.0},
        {"dataset": "dummy", "model": "m", "task": "classification", "split": "all", "unit_id": "b1", "event_id": "e1", "region": "B", "score": 0.0},
    ]
    artifacts = evaluate_bwer_table(
        rows,
        "dummy",
        "m",
        "classification",
        "outputs/test_bwer_one_cluster",
        slice_variable="region",
        bootstrap=5,
        cluster_key="event_id",
    )
    assert "too_few_clusters" in artifacts["warnings"].read_text(encoding="utf-8")
