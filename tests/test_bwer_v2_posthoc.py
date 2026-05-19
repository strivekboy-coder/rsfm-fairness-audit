from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from rsfm_fairness_audit.bwer_v2 import run_bwer_v2_posthoc
from rsfm_fairness_audit.cli import main
from rsfm_fairness_audit.io import read_csv_rows, write_csv


def _write_completed_segmentation_run(root: Path) -> None:
    rows = [
        {
            "dataset": "sen1floods11",
            "model": "prithvi_tl_sen1floods11",
            "task": "segmentation",
            "event_id": "Pakistan",
            "event": "Pakistan",
            "country": "Pakistan",
            "TP": 655,
            "FP": 200,
            "FN": 145,
            "TN": 9000,
            "valid_pixel_count": 10000,
            "positive_pixel_count": 800,
            "predicted_positive_pixel_count": 855,
            "micro_iou": 0.655,
            "micro_dice": 0.7915407855,
            "precision": 655 / 855,
            "recall": 655 / 800,
            "risk": 0.345,
            "input_mode": "S2",
            "adaptation_protocol": "task_adapted_decoder",
            "training_budget": "official_sen1floods11_finetune",
            "split_protocol": "standard_split",
        },
        {
            "dataset": "sen1floods11",
            "model": "prithvi_tl_sen1floods11",
            "task": "segmentation",
            "event_id": "Bolivia",
            "event": "Bolivia",
            "country": "Bolivia",
            "TP": 677,
            "FP": 160,
            "FN": 163,
            "TN": 9000,
            "valid_pixel_count": 10000,
            "positive_pixel_count": 840,
            "predicted_positive_pixel_count": 837,
            "micro_iou": 0.677,
            "micro_dice": 0.8073985680,
            "precision": 677 / 837,
            "recall": 677 / 840,
            "risk": 0.323,
            "input_mode": "S2",
            "adaptation_protocol": "task_adapted_decoder",
            "training_budget": "official_sen1floods11_finetune",
            "split_protocol": "standard_split",
        },
        {
            "dataset": "sen1floods11",
            "model": "prithvi_tl_sen1floods11",
            "task": "segmentation",
            "event_id": "Mekong",
            "event": "Mekong",
            "country": "Mekong",
            "TP": 916,
            "FP": 50,
            "FN": 34,
            "TN": 9000,
            "valid_pixel_count": 10000,
            "positive_pixel_count": 950,
            "predicted_positive_pixel_count": 966,
            "micro_iou": 0.916,
            "micro_dice": 0.9561586639,
            "precision": 916 / 966,
            "recall": 916 / 950,
            "risk": 0.084,
            "input_mode": "S2",
            "adaptation_protocol": "task_adapted_decoder",
            "training_budget": "official_sen1floods11_finetune",
            "split_protocol": "standard_split",
        },
    ]
    write_csv(root / "event_segmentation_metrics.csv", rows)
    chip_rows = []
    positives = {
        "Pakistan": [20, 220, 820],
        "Bolivia": [30, 260, 780],
        "Mekong": [40, 300, 760],
    }
    for event_id, values in positives.items():
        for index, positive in enumerate(values):
            tp = int(positive * (0.70 if event_id != "Mekong" else 0.92))
            fn = positive - tp
            fp = 70 if event_id == "Pakistan" else 30 if event_id == "Bolivia" else 15
            tn = 1000 - positive - fp
            chip_rows.append(
                {
                    "sample_id": f"{event_id}_{index}",
                    "event_id": event_id,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "valid_pixel_count": 1000,
                    "positive_pixel_count": positive,
                    "predicted_positive_pixel_count": tp + fp,
                    "ground_truth_positive_pixel_ratio": positive / 1000,
                }
            )
    write_csv(root / "segmentation_metrics.csv", chip_rows)
    write_csv(root / "bwer_summary.csv", [{"slice_variable": "event_id", "bwer": 0.1}])
    (root / "model_debug.json").write_text(
        json.dumps(
            {
                "checkpoint_source": "official_huggingface",
                "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
                "inference_window_size": 512,
            }
        ),
        encoding="utf-8",
    )


def test_run_bwer_v2_posthoc_writes_full_output_set() -> None:
    run_dir = Path("outputs") / f"test_bwer_v2_posthoc_{uuid.uuid4().hex}" / "prithvi_tl_sen1floods11_official_full_512"
    if run_dir.parent.exists():
        shutil.rmtree(run_dir.parent)
    run_dir.mkdir(parents=True)
    _write_completed_segmentation_run(run_dir)
    out = run_dir / "bwer_v2"
    artifacts = run_bwer_v2_posthoc(run_dir, out, bootstrap=20, seed=7)

    expected = [
        "bwer_v2_summary",
        "derived_balance_variables",
        "standardised_bwer",
        "alpha_sensitivity",
        "support_sensitivity",
        "reference_weight_sensitivity",
        "missing_policy_sensitivity",
        "stabilised_bwer",
        "leave_one_slice_out",
        "bootstrap_ci",
        "event_failure_analysis",
        "event_ranking",
        "metric_primitives_report",
        "adaptation_protocol_report",
        "split_diagnostics_report",
        "bwer_audit_report",
    ]
    for name in expected:
        assert artifacts[name].exists(), name

    summary = read_csv_rows(out / "bwer_v2_summary.csv")[0]
    assert summary["dataset"] == "sen1floods11"
    assert summary["model"] == "prithvi_tl_sen1floods11"
    assert summary["model_family"] == "Prithvi"
    assert summary["slice_variable"] == "event_id"
    assert summary["resolution"] == "512"
    assert "valid_pixel_count" in summary["support_definition"]
    assert summary["bootstrap_method"] == "posthoc_event_bootstrap"
    summaries = read_csv_rows(out / "bwer_v2_summary.csv")
    assert {row["analysis_type"] for row in summaries} >= {"raw", "standardised"}
    assert any(row["balance_variable"] == "flood_extent_bin" for row in summaries)
    derived = read_csv_rows(out / "derived_balance_variables.csv")
    assert derived[0]["flood_extent_bin"] == "low_flood_extent"

    alpha = read_csv_rows(out / "alpha_sensitivity.csv")
    assert [row["alpha"] for row in alpha] == ["0.1", "0.2", "0.3", "0.4"]

    support = read_csv_rows(out / "support_sensitivity.csv")
    assert all(row["all_events_valid"] == "True" for row in support)

    reference = read_csv_rows(out / "reference_weight_sensitivity.csv")
    assert any(row["balance_variable"] == "flood_extent_bin" for row in reference)
    missing = read_csv_rows(out / "missing_policy_sensitivity.csv")
    assert any(row["balance_variable"] == "flood_extent_bin" for row in missing)

    failure = read_csv_rows(out / "event_failure_analysis.csv")
    pakistan = next(row for row in failure if row["event_id"] == "Pakistan")
    assert pakistan["tail_flag"] == "True"
    assert pakistan["IoU_rank"] == "3"
    adaptation_report = (out / "adaptation_protocol_report.md").read_text(encoding="utf-8")
    assert "model_family: Prithvi" in adaptation_report
    assert "official Sen1Floods11 task-adapted decoder route" in adaptation_report


def test_run_bwer_v2_cli(monkeypatch) -> None:
    run_dir = Path("outputs") / f"test_bwer_v2_cli_{uuid.uuid4().hex}" / "run"
    if run_dir.parent.exists():
        shutil.rmtree(run_dir.parent)
    run_dir.mkdir(parents=True)
    _write_completed_segmentation_run(run_dir)
    out = run_dir / "bwer_v2"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "run-bwer-v2",
            "--input-dir",
            str(run_dir),
            "--output-dir",
            str(out),
            "--bootstrap",
            "5",
        ],
    )
    main()
    assert (out / "bwer_v2_summary.csv").exists()
    assert (out / "bwer_audit_report.md").exists()


def test_bwer_v2_summary_treats_none_bootstrap_method_as_missing() -> None:
    run_dir = Path("outputs") / f"test_bwer_v2_existing_ci_{uuid.uuid4().hex}" / "run"
    run_dir.mkdir(parents=True)
    _write_completed_segmentation_run(run_dir)
    write_csv(
        run_dir / "bootstrap_ci.csv",
        [
            {
                "source": "bwer_v2_posthoc",
                "status": "computed",
                "method": "posthoc_event_bootstrap",
                "bootstrap_method": "none",
                "ci_low": 0.01,
                "ci_high": 0.02,
                "bootstrap_n": 1000,
            }
        ],
    )
    out = run_dir / "bwer_v2"
    run_bwer_v2_posthoc(run_dir, out, bootstrap=5, seed=7)
    summary = read_csv_rows(out / "bwer_v2_summary.csv")[0]
    assert summary["bootstrap_method"] == "posthoc_event_bootstrap"
