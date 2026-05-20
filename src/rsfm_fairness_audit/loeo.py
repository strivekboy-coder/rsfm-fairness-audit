from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_table
from rsfm_fairness_audit.audit_table import write_audit_table
from rsfm_fairness_audit.bwer_v2 import run_bwer_v2_posthoc
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.segmentation import aggregate_segmentation_metrics, build_audit_table_from_segmentation_metrics_from_rows
from rsfm_fairness_audit.slice_support import evaluate_slice_support


def _event_dir_rows(input_root: Path) -> list[tuple[str, Path, list[dict[str, str]], list[dict[str, str]]]]:
    rows = []
    for path in sorted(input_root.iterdir()):
        if not path.is_dir():
            continue
        chip_path = path / "segmentation_metrics.csv"
        event_path = path / "event_segmentation_metrics.csv"
        if chip_path.exists() and event_path.exists():
            rows.append((path.name, path, read_csv_rows(chip_path), read_csv_rows(event_path)))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def aggregate_loeo_runs(input_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    root = Path(input_root)
    output = ensure_dir(output_dir)
    runs = _event_dir_rows(root)
    if not runs:
        raise FileNotFoundError(f"No LOEO per-event run directories with segmentation outputs found under {root}")
    all_chip_rows: list[dict[str, Any]] = []
    for held_out, run_dir, chip_rows, _event_rows in runs:
        for row in chip_rows:
            item = dict(row)
            item["held_out_event"] = held_out
            item["split_protocol"] = "leave_one_event_out"
            item["split"] = "test"
            all_chip_rows.append(item)
    event_rows = aggregate_segmentation_metrics(all_chip_rows, "event_id", aggregation_level="event")
    for row in event_rows:
        row["split_protocol"] = "leave_one_event_out"
        row["split"] = "test"
    audit_rows = build_audit_table_from_segmentation_metrics_from_rows(event_rows)
    first_meta = _read_json(runs[0][1] / "run_metadata.json") or _read_json(runs[0][1] / "model_debug.json")
    first_meta.update(
        {
            "split_protocol": "leave_one_event_out",
            "loeo_input_root": str(root),
            "held_out_events": [held_out for held_out, *_ in runs],
            "loeo_completed_event_count": len(runs),
            "loeo_note": "Aggregated from per-held-out-event supervised baseline runs.",
        }
    )
    artifacts = {
        "loeo_summary": output / "loeo_summary.csv",
        "loeo_segmentation_metrics": output / "loeo_segmentation_metrics.csv",
        "loeo_event_level_metrics": output / "loeo_event_level_metrics.csv",
        "segmentation_metrics": output / "segmentation_metrics.csv",
        "event_segmentation_metrics": output / "event_segmentation_metrics.csv",
        "audit_table": output / "audit_table.csv",
        "run_metadata": output / "run_metadata.json",
        "model_debug": output / "model_debug.json",
        "loeo_report": output / "loeo_report.md",
    }
    summary_row = {
        "model": str(event_rows[0].get("model", first_meta.get("model_variant", ""))) if event_rows else first_meta.get("model_variant", ""),
        "model_family": first_meta.get("model_family", "unet"),
        "adaptation_protocol": "supervised_baseline",
        "split_protocol": "leave_one_event_out",
        "held_out_event_count": len(runs),
        "chip_count": len(all_chip_rows),
        "event_count": len(event_rows),
        "input_root": str(root),
    }
    write_csv(artifacts["loeo_summary"], [summary_row])
    write_csv(artifacts["loeo_segmentation_metrics"], all_chip_rows)
    write_csv(artifacts["segmentation_metrics"], all_chip_rows)
    write_csv(artifacts["loeo_event_level_metrics"], event_rows)
    write_csv(artifacts["event_segmentation_metrics"], event_rows)
    write_audit_table(artifacts["audit_table"], audit_rows)
    artifacts["run_metadata"].write_text(json.dumps(first_meta, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["model_debug"].write_text(json.dumps(first_meta, indent=2, sort_keys=True), encoding="utf-8")
    (output / "warnings.json").write_text(json.dumps({"warnings": ["LOEO aggregate built from completed per-event supervised-baseline runs."]}, indent=2), encoding="utf-8")
    preflight = evaluate_slice_support(
        audit_rows,
        dataset="sen1floods11",
        model=str(event_rows[0].get("model", first_meta.get("model_variant", "loeo_model"))) if event_rows else "loeo_model",
        task="segmentation",
        output_dir=output,
        candidates=["event_id", "event_id|event", "country|country"],
        score_column="micro_iou",
        risk_column="risk",
    )
    artifacts.update({f"preflight_{key}": value for key, value in preflight.items()})
    bwer = evaluate_bwer_table(
        audit_rows,
        dataset="sen1floods11",
        model=str(event_rows[0].get("model", first_meta.get("model_variant", "loeo_model"))) if event_rows else "loeo_model",
        task="segmentation",
        output_dir=output,
        slice_variable="event_id",
        balance_variable="raw",
        score_column="micro_iou",
        risk_column="risk",
        audit_level="pilot",
    )
    artifacts.update(bwer)
    bwer_v2 = run_bwer_v2_posthoc(output, output / "bwer_v2")
    artifacts.update({f"bwer_v2_{key}": value for key, value in bwer_v2.items()})
    artifacts["loeo_bwer_summary"] = output / "loeo_bwer_summary.csv"
    write_csv(artifacts["loeo_bwer_summary"], read_csv_rows(output / "bwer_v2" / "bwer_v2_summary.csv"))
    _write_loeo_report(artifacts["loeo_report"], first_meta, event_rows, [held_out for held_out, *_ in runs])
    return artifacts


def _write_loeo_report(path: Path, metadata: dict[str, Any], event_rows: Sequence[dict[str, Any]], held_out_events: Sequence[str]) -> None:
    lines = [
        "# Sen1Floods11 Leave-One-Event-Out Aggregate",
        "",
        "This output aggregates completed per-held-out-event supervised-baseline runs. It does not train models itself.",
        "",
        f"- model_variant: {metadata.get('model_variant', '')}",
        "- split_protocol: leave_one_event_out",
        f"- completed held-out events: {';'.join(held_out_events)}",
        "",
        "Each held-out event should be trained using only the remaining events and evaluated on the held-out event. This is stronger evidence than random chip split but remains model/protocol specific.",
        "",
        "| event_id | chips | micro IoU | micro Dice | risk |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in event_rows:
        lines.append(f"| {row['event_id']} | {row['sample_count']} | {row['micro_iou']:.4f} | {row['micro_dice']:.4f} | {row['risk']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
