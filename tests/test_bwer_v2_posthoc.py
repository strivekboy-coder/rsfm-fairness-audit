from __future__ import annotations

import json
import shutil
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
    write_csv(root / "segmentation_metrics.csv", [{"sample_id": "x", "event_id": "Pakistan", "valid_pixel_count": 512 * 512}])
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
    run_dir = Path("outputs/test_bwer_v2_posthoc/prithvi_tl_sen1floods11_official_full_512")
    if run_dir.parent.exists():
        shutil.rmtree(run_dir.parent)
    run_dir.mkdir(parents=True)
    _write_completed_segmentation_run(run_dir)
    out = run_dir / "bwer_v2"
    artifacts = run_bwer_v2_posthoc(run_dir, out, bootstrap=20, seed=7)

    expected = [
        "bwer_v2_summary",
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
    assert summary["slice_variable"] == "event_id"
    assert summary["resolution"] == "512"
    assert "valid_pixel_count" in summary["support_definition"]

    alpha = read_csv_rows(out / "alpha_sensitivity.csv")
    assert [row["alpha"] for row in alpha] == ["0.1", "0.2", "0.3", "0.4"]

    support = read_csv_rows(out / "support_sensitivity.csv")
    assert all(row["all_events_valid"] == "True" for row in support)

    reference = read_csv_rows(out / "reference_weight_sensitivity.csv")
    assert reference[0]["status"] == "not_applicable"
    assert "non-proxy" in reference[0]["reason"]

    failure = read_csv_rows(out / "event_failure_analysis.csv")
    pakistan = next(row for row in failure if row["event_id"] == "Pakistan")
    assert pakistan["tail_flag"] == "True"
    assert pakistan["IoU_rank"] == "3"


def test_run_bwer_v2_cli(monkeypatch) -> None:
    run_dir = Path("outputs/test_bwer_v2_cli/run")
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
