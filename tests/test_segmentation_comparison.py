from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from rsfm_fairness_audit.cli import main
from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.segmentation_comparison import compare_segmentation_runs


def _write_completed_run(root: Path, model: str, family: str, adaptation: str, split_protocol: str, offset: int = 0) -> None:
    (root / "bwer_v2").mkdir(parents=True)
    events = [
        ("Bolivia", 800 + offset, 120, 160, 5000),
        ("Pakistan", 620 + offset, 180, 220, 5000),
        ("Mekong", 920 + offset, 60, 50, 5000),
    ]
    event_rows = []
    for event_id, tp, fp, fn, tn in events:
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        event_rows.append(
            {
                "dataset": "sen1floods11",
                "model": model,
                "model_family": family,
                "task": "segmentation",
                "split": "test" if split_protocol == "random_chip_split" else "all",
                "event_id": event_id,
                "event": event_id,
                "adaptation_protocol": adaptation,
                "split_protocol": split_protocol,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "valid_pixel_count": tp + fp + fn + tn,
                "positive_pixel_count": tp + fn,
                "predicted_positive_pixel_count": tp + fp,
                "micro_iou": iou,
                "micro_dice": dice,
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
                "risk": 1 - iou,
            }
        )
    write_csv(root / "event_segmentation_metrics.csv", event_rows)
    write_csv(
        root / "bwer_v2" / "bwer_v2_summary.csv",
        [
            {
                "analysis_type": "raw",
                "model": model,
                "model_family": family,
                "adaptation_protocol": adaptation,
                "split_protocol": split_protocol,
                "balance_variable": "",
                "bwer": 0.12 + offset / 10000,
                "tail_slices": "Pakistan",
                "resolution": 512,
            },
            {
                "analysis_type": "standardised",
                "model": model,
                "model_family": family,
                "adaptation_protocol": adaptation,
                "split_protocol": split_protocol,
                "balance_variable": "flood_extent_bin",
                "bwer": 0.16 + offset / 10000,
                "tail_slices": "Pakistan;Bolivia",
                "resolution": 512,
            },
        ],
    )
    write_csv(
        root / "bwer_v2" / "event_failure_analysis.csv",
        [
            {"event_id": "Pakistan", "tail_flag": "True"},
            {"event_id": "Bolivia", "tail_flag": "False"},
            {"event_id": "Mekong", "tail_flag": "False"},
        ],
    )
    (root / "run_metadata.json").write_text(
        f'{{"model_family": "{family}", "adaptation_protocol": "{adaptation}", "split_protocol": "{split_protocol}", "resolution": 512}}\n',
        encoding="utf-8",
    )


def test_compare_segmentation_runs_writes_standalone_outputs() -> None:
    root = Path("outputs") / f"test_segmentation_comparison_{uuid.uuid4().hex}"
    prithvi = root / "prithvi"
    unet = root / "unet"
    _write_completed_run(prithvi, "prithvi_tl_sen1floods11", "Prithvi", "task_adapted_decoder", "standard_split")
    _write_completed_run(unet, "unet_sen1floods11_s2_512", "unet", "supervised_baseline", "random_chip_split", offset=-80)
    out = root / "comparison"
    artifacts = compare_segmentation_runs({"prithvi": prithvi, "unet": unet}, out)
    for key in ["comparison_summary", "average_vs_bwer", "event_level_comparison", "comparison_report"]:
        assert artifacts[key].exists(), key
    summary = read_csv_rows(out / "comparison_summary.csv")
    assert {row["run_name"] for row in summary} == {"prithvi", "unet"}
    assert next(row for row in summary if row["run_name"] == "unet")["split_protocol"] == "random_chip_split"
    event_rows = read_csv_rows(out / "event_level_comparison.csv")
    assert "delta_iou_vs_other" in event_rows[0]
    report = (out / "comparison_report.md").read_text(encoding="utf-8")
    assert "protocol-aware deployment-practice comparison" in report
    shutil.rmtree(root, ignore_errors=True)


def test_compare_segmentation_runs_writes_closure_outputs() -> None:
    root = Path("outputs") / f"test_segmentation_closure_{uuid.uuid4().hex}"
    runs = {
        "prithvi_tl": root / "prithvi",
        "vanilla_unet": root / "unet",
        "spectral_mndwi": root / "spectral",
        "s2_resnet34_unet": root / "resnet34",
    }
    _write_completed_run(runs["prithvi_tl"], "prithvi_tl_sen1floods11", "Prithvi", "task_adapted_decoder", "standard_split")
    _write_completed_run(runs["vanilla_unet"], "unet_sen1floods11_s2_512", "unet", "supervised_baseline", "random_chip_split", offset=-80)
    _write_completed_run(runs["spectral_mndwi"], "spectral_mndwi_fixed_ge_0p0", "spectral_rule", "diagnostic_spectral_rule", "standard_split", offset=-160)
    _write_completed_run(runs["s2_resnet34_unet"], "s2_resnet34_unet", "unet", "supervised_baseline", "random_chip_split", offset=40)
    out = root / "closure"
    artifacts = compare_segmentation_runs(runs, out, closure=True)
    for key in ["closure_comparison_summary", "closure_average_vs_bwer", "closure_event_level_comparison", "closure_tail_event_overlap", "closure_report"]:
        assert artifacts[key].exists(), key
    summary = read_csv_rows(out / "closure_comparison_summary.csv")
    assert len(summary) == 4
    overlap = read_csv_rows(out / "closure_tail_event_overlap.csv")
    assert any(row["event_id"] == "Pakistan" for row in overlap)
    report = (out / "closure_report.md").read_text(encoding="utf-8")
    assert "Future LOEO and Selective Risk Notes" in report
    assert "Spectral Baseline Check" in report
    shutil.rmtree(root, ignore_errors=True)


def test_compare_runs_cli_detects_segmentation_outputs(monkeypatch) -> None:
    root = Path("outputs") / f"test_segmentation_comparison_cli_{uuid.uuid4().hex}"
    prithvi = root / "prithvi"
    unet = root / "unet"
    _write_completed_run(prithvi, "prithvi_tl_sen1floods11", "Prithvi", "task_adapted_decoder", "standard_split")
    _write_completed_run(unet, "unet_sen1floods11_s2_512", "unet", "supervised_baseline", "random_chip_split", offset=-80)
    out = root / "comparison"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "compare-runs",
            "--dataset",
            "sen1floods11",
            "--run",
            f"prithvi={prithvi}",
            "--run",
            f"unet={unet}",
            "--output-dir",
            str(out),
        ],
    )
    main()
    assert (out / "comparison_summary.csv").exists()
    assert (out / "figures" / "average_iou_vs_raw_bwer.png").exists()
    shutil.rmtree(root, ignore_errors=True)
