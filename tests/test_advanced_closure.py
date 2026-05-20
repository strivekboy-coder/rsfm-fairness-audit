from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from rsfm_fairness_audit.advanced_closure import run_protocol_matched_comparison, run_selective_risk_audit
from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.loeo import aggregate_loeo_runs


def _write_run(root: Path, model: str, adaptation: str, split_protocol: str, confidence: bool = False, sample_offset: int = 0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    chips = []
    for event_index, event in enumerate(["Bolivia", "Pakistan", "Mekong"]):
        for chip_index in range(2):
            tp = 100 + sample_offset + event_index * 20
            fp = 10 + chip_index * 5
            fn = 15 + event_index * 7
            tn = 500
            iou = tp / (tp + fp + fn)
            row = {
                "dataset": "sen1floods11",
                "model": model,
                "model_family": "unet" if "unet" in model else "Prithvi",
                "task": "segmentation",
                "split": "test" if split_protocol != "standard_split" else "all",
                "sample_id": f"{event}_{chip_index}",
                "unit_id": f"{event}_{chip_index}",
                "event_id": event,
                "event": event,
                "country": event,
                "input_mode": "s2_6band_image_only",
                "adaptation_protocol": adaptation,
                "split_protocol": split_protocol,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "valid_pixel_count": tp + fp + fn + tn,
                "positive_pixel_count": tp + fn,
                "predicted_positive_pixel_count": tp + fp,
                "TP_plus_FN_support": tp + fn,
                "micro_iou": iou,
                "iou": iou,
                "risk": 1 - iou,
                "score": iou,
            }
            if confidence:
                row["mean_confidence"] = 0.95 - event_index * 0.1 - chip_index * 0.01
                row["confidence_source"] = "sigmoid_probability"
            chips.append(row)
    write_csv(root / "segmentation_metrics.csv", chips)
    (root / "run_metadata.json").write_text(
        f'{{"model_variant": "{model}", "adaptation_protocol": "{adaptation}", "split_protocol": "{split_protocol}", "resolution": 512}}\n',
        encoding="utf-8",
    )


def test_protocol_matched_comparison_recomputes_on_common_chips() -> None:
    root = Path("outputs") / f"test_protocol_matched_{uuid.uuid4().hex}"
    a = root / "a"
    b = root / "b"
    _write_run(a, "prithvi_tl_sen1floods11", "task_adapted_decoder", "standard_split")
    _write_run(b, "unet_sen1floods11_s2_512", "supervised_baseline", "random_chip_split", sample_offset=-10)
    out = root / "matched"
    artifacts = run_protocol_matched_comparison({"prithvi": a, "unet": b}, out)
    assert artifacts["summary"].exists()
    assert artifacts["event_level_comparison"].exists()
    summary = read_csv_rows(out / "protocol_matched_summary.csv")
    assert {row["run_name"] for row in summary} == {"prithvi", "unet"}
    report = (out / "protocol_matched_report.md").read_text(encoding="utf-8")
    assert "exact_chip_level_match: True" in report
    shutil.rmtree(root, ignore_errors=True)


def test_selective_risk_uses_chip_confidence_when_available() -> None:
    root = Path("outputs") / f"test_selective_risk_{uuid.uuid4().hex}"
    confident = root / "confident"
    no_conf = root / "no_conf"
    _write_run(confident, "unet_sen1floods11_s2_512", "supervised_baseline", "random_chip_split", confidence=True)
    _write_run(no_conf, "spectral_mndwi_fixed_ge_0p0", "diagnostic_spectral_rule", "standard_split", confidence=False)
    out = root / "selective"
    artifacts = run_selective_risk_audit({"unet": confident, "spectral": no_conf}, out, coverages=(1.0, 0.5))
    assert artifacts["availability"].exists()
    availability = read_csv_rows(out / "selective_risk_availability.csv")
    assert next(row for row in availability if row["run_name"] == "unet")["available"] == "True"
    assert next(row for row in availability if row["run_name"] == "spectral")["available"] == "False"
    summary = read_csv_rows(out / "selective_risk_summary.csv")
    assert {row["coverage_target"] for row in summary} == {"1.0", "0.5"}
    shutil.rmtree(root, ignore_errors=True)


def test_aggregate_loeo_runs_writes_bwer_compatible_outputs() -> None:
    root = Path("outputs") / f"test_loeo_aggregate_{uuid.uuid4().hex}"
    loeo_root = root / "loeo" / "unet_sen1floods11_s2_512"
    for event in ["Bolivia", "Pakistan", "Mekong"]:
        run_dir = loeo_root / event
        _write_run(run_dir, "unet_sen1floods11_s2_512", "supervised_baseline", "leave_one_event_out", confidence=True)
        rows = [row for row in read_csv_rows(run_dir / "segmentation_metrics.csv") if row["event_id"] == event]
        write_csv(run_dir / "segmentation_metrics.csv", rows)
        write_csv(run_dir / "event_segmentation_metrics.csv", rows[:1])
    out = root / "aggregate"
    artifacts = aggregate_loeo_runs(loeo_root, out)
    assert artifacts["loeo_event_level_metrics"].exists()
    assert artifacts["loeo_bwer_summary"].exists()
    rows = read_csv_rows(out / "loeo_event_level_metrics.csv")
    assert {row["event_id"] for row in rows} == {"Bolivia", "Pakistan", "Mekong"}
    report = (out / "loeo_report.md").read_text(encoding="utf-8")
    assert "leave-one-event-out" in report.lower()
    shutil.rmtree(root, ignore_errors=True)
