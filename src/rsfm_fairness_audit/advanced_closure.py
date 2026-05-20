from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_table
from rsfm_fairness_audit.audit_table import write_audit_table
from rsfm_fairness_audit.bwer_v2 import run_bwer_v2_posthoc
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.segmentation import aggregate_segmentation_metrics, build_audit_table_from_segmentation_metrics_from_rows
from rsfm_fairness_audit.segmentation_comparison import compare_segmentation_runs
from rsfm_fairness_audit.slice_support import evaluate_slice_support


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _sample_id(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("chip_id") or row.get("unit_id") or "")


def _write_completed_filtered_run(run_name: str, source_dir: Path, rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    run_out = ensure_dir(output_dir / "matched_runs" / run_name)
    chip_rows = [dict(row) for row in rows]
    event_rows = aggregate_segmentation_metrics(chip_rows, "event_id", aggregation_level="event")
    audit_rows = build_audit_table_from_segmentation_metrics_from_rows(event_rows)
    write_csv(run_out / "segmentation_metrics.csv", chip_rows)
    write_csv(run_out / "segmentation_predictions.csv", chip_rows)
    write_csv(run_out / "event_segmentation_metrics.csv", event_rows)
    write_audit_table(run_out / "segmentation_audit_table.csv", audit_rows)
    write_audit_table(run_out / "audit_table.csv", audit_rows)
    metadata = _read_json(source_dir / "run_metadata.json") or _read_json(source_dir / "model_debug.json")
    metadata.update(
        {
            "protocol_matched_source_dir": str(source_dir),
            "protocol_matched_run_name": run_name,
            "protocol_matched_chip_count": len(chip_rows),
            "protocol_matched_note": "posthoc intersection over saved chip-level segmentation_metrics.csv",
        }
    )
    (run_out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (run_out / "model_debug.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    warnings = {"warnings": [f"Protocol-matched post-hoc subset generated from {source_dir}."]}
    (run_out / "warnings.json").write_text(json.dumps(warnings, indent=2), encoding="utf-8")
    try:
        preflight = evaluate_slice_support(
            audit_rows,
            dataset="sen1floods11",
            model=str(event_rows[0].get("model", run_name)) if event_rows else run_name,
            task="segmentation",
            output_dir=run_out,
            candidates=["event_id", "event_id|event", "country|country"],
            score_column="micro_iou",
            risk_column="risk",
        )
        evaluate_bwer_table(
            audit_rows,
            dataset="sen1floods11",
            model=str(event_rows[0].get("model", run_name)) if event_rows else run_name,
            task="segmentation",
            output_dir=run_out,
            slice_variable="event_id",
            balance_variable="raw",
            score_column="micro_iou",
            risk_column="risk",
            audit_level="pilot",
        )
        run_bwer_v2_posthoc(run_out, run_out / "bwer_v2")
        _ = preflight
    except ValueError as exc:
        (run_out / "bwer_not_runnable.txt").write_text(str(exc) + "\n", encoding="utf-8")
    return run_out


def run_protocol_matched_comparison(runs: Mapping[str, str | Path], output_dir: str | Path) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    availability: list[dict[str, Any]] = []
    chip_rows_by_run: dict[str, list[dict[str, str]]] = {}
    sample_sets: dict[str, set[str]] = {}
    for run_name, run_dir_value in runs.items():
        run_dir = Path(run_dir_value)
        chip_path = run_dir / "segmentation_metrics.csv"
        event_path = run_dir / "event_segmentation_metrics.csv"
        row: dict[str, Any] = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "has_chip_metrics": chip_path.exists(),
            "has_event_metrics": event_path.exists(),
            "matched_status": "pending",
            "notes": "",
        }
        if not chip_path.exists():
            row["matched_status"] = "not_applicable"
            row["notes"] = "Missing segmentation_metrics.csv; exact chip-level intersection cannot be computed."
            availability.append(row)
            continue
        rows = read_csv_rows(chip_path)
        ids = {_sample_id(item) for item in rows if _sample_id(item)}
        if not ids:
            row["matched_status"] = "not_applicable"
            row["notes"] = "No usable sample_id/chip_id/unit_id in chip-level metrics."
            availability.append(row)
            continue
        row["chip_count"] = len(rows)
        row["unique_chip_ids"] = len(ids)
        row["matched_status"] = "available"
        availability.append(row)
        chip_rows_by_run[run_name] = rows
        sample_sets[run_name] = ids
    common_ids = set.intersection(*sample_sets.values()) if sample_sets and len(sample_sets) == len(runs) else set()
    artifacts = {
        "availability": output / "protocol_matched_availability.csv",
        "summary": output / "protocol_matched_summary.csv",
        "average_vs_bwer": output / "protocol_matched_average_vs_bwer.csv",
        "event_level_comparison": output / "protocol_matched_event_level_comparison.csv",
        "tail_event_overlap": output / "protocol_matched_tail_event_overlap.csv",
        "report": output / "protocol_matched_report.md",
    }
    if not common_ids:
        write_csv(artifacts["availability"], availability)
        for key in ["summary", "average_vs_bwer", "event_level_comparison", "tail_event_overlap"]:
            write_csv(artifacts[key], [{"status": "not_applicable", "reason": "No exact common chip intersection across all requested runs."}])
        _write_protocol_matched_report(artifacts["report"], availability, [], common_ids, matched=False)
        return artifacts
    matched_dirs: dict[str, Path] = {}
    for run_name, rows in chip_rows_by_run.items():
        filtered = [row for row in rows if _sample_id(row) in common_ids]
        source_dir = Path(runs[run_name])
        matched_dirs[run_name] = _write_completed_filtered_run(run_name, source_dir, filtered, output)
    comparison_dir = ensure_dir(output / "_comparison")
    comparison_artifacts = compare_segmentation_runs(matched_dirs, comparison_dir, dataset_name="sen1floods11", closure=True)
    shutil.copyfile(comparison_artifacts["comparison_summary"], artifacts["summary"])
    shutil.copyfile(comparison_artifacts["average_vs_bwer"], artifacts["average_vs_bwer"])
    shutil.copyfile(comparison_artifacts["event_level_comparison"], artifacts["event_level_comparison"])
    shutil.copyfile(comparison_artifacts["closure_tail_event_overlap"], artifacts["tail_event_overlap"])
    write_csv(artifacts["availability"], availability)
    _write_protocol_matched_report(artifacts["report"], availability, read_csv_rows(artifacts["summary"]), common_ids, matched=True)
    return artifacts


def _write_protocol_matched_report(path: Path, availability: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, str]], common_ids: set[str], matched: bool) -> None:
    lines = [
        "# Sen1Floods11 Protocol-Matched Closure Check",
        "",
        "This is a post-hoc chip-intersection sanity check over completed outputs. It does not rerun model inference or training.",
        "",
        f"- exact_chip_level_match: {matched}",
        f"- matched_chip_count: {len(common_ids)}",
        "",
        "## Run Availability",
        "",
        "| run | chip metrics | event metrics | status | notes |",
        "|---|---:|---:|---|---|",
    ]
    for row in availability:
        lines.append(f"| {row.get('run_name')} | {row.get('has_chip_metrics')} | {row.get('has_event_metrics')} | {row.get('matched_status')} | {row.get('notes', '')} |")
    if summaries:
        ordered_avg = sorted(summaries, key=lambda row: _float(row.get("aggregate_iou")), reverse=True)
        ordered_raw = sorted(summaries, key=lambda row: _float(row.get("raw_bwer_event_id")))
        reversal = [row.get("run_name") for row in ordered_avg] != [row.get("run_name") for row in ordered_raw]
        lines.extend(
            [
                "",
                "## Matched Result",
                "",
                f"- Aggregate IoU ranking: {' > '.join(str(row.get('run_name')) for row in ordered_avg)}",
                f"- Raw-BWER ranking, lower is better: {' < '.join(str(row.get('run_name')) for row in ordered_raw)}",
                f"- Average-vs-BWER ranking reversal remains on matched subset: {reversal}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Limitation",
                "",
                "Exact chip-level matching was not possible from the provided outputs. This usually means at least one run lacks chip-level `segmentation_metrics.csv` with stable `sample_id`/`chip_id` identifiers.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _confidence_column(rows: Sequence[Mapping[str, str]]) -> str | None:
    candidates = ["mean_confidence", "confidence", "confidence_valid_mean", "water_prob_mean", "prediction_score_mean"]
    for candidate in candidates:
        values = [_float(row.get(candidate)) for row in rows if row.get(candidate, "") != ""]
        if values and any(not math.isnan(value) for value in values):
            return candidate
    return None


def run_selective_risk_audit(
    runs: Mapping[str, str | Path],
    output_dir: str | Path,
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    artifacts = {
        "availability": output / "selective_risk_availability.csv",
        "summary": output / "selective_risk_summary.csv",
        "by_event": output / "selective_risk_by_event.csv",
        "report": output / "selective_risk_report.md",
    }
    availability: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    by_event_rows: list[dict[str, Any]] = []
    for run_name, run_dir_value in runs.items():
        run_dir = Path(run_dir_value)
        chip_path = run_dir / "segmentation_metrics.csv"
        if not chip_path.exists():
            availability.append({"run_name": run_name, "available": False, "confidence_column": "", "method": "", "notes": "Missing segmentation_metrics.csv."})
            continue
        chip_rows = read_csv_rows(chip_path)
        column = _confidence_column(chip_rows)
        if column is None:
            availability.append({"run_name": run_name, "available": False, "confidence_column": "", "method": "", "notes": "No usable chip-level confidence/logit/probability summary column found."})
            continue
        availability.append(
            {
                "run_name": run_name,
                "available": True,
                "confidence_column": column,
                "method": "posthoc_chip_confidence_retention",
                "notes": "Retains whole chips by confidence summary; not pixel-level selective risk.",
            }
        )
        sorted_rows = sorted(chip_rows, key=lambda row: _float(row.get(column), -math.inf), reverse=True)
        total_valid = sum(_float(row.get("valid_pixel_count"), 0.0) for row in sorted_rows)
        for coverage in coverages:
            retained: list[dict[str, str]] = []
            retained_valid = 0.0
            target = total_valid * float(coverage)
            for row in sorted_rows:
                if retained and retained_valid >= target:
                    break
                retained.append(row)
                retained_valid += _float(row.get("valid_pixel_count"), 0.0)
            if not retained:
                continue
            event_rows = aggregate_segmentation_metrics(retained, "event_id", aggregation_level="event")
            risks = sorted([_float(row.get("risk")) for row in event_rows if not math.isnan(_float(row.get("risk")))], reverse=True)
            tail_count = max(1, int(math.ceil(0.1 * len(risks)))) if risks else 0
            mean_risk = float(sum(risks) / len(risks)) if risks else float("nan")
            tail_risk = float(sum(risks[:tail_count]) / tail_count) if tail_count else float("nan")
            bwer = tail_risk - mean_risk if not math.isnan(tail_risk) and not math.isnan(mean_risk) else float("nan")
            tail_threshold = risks[tail_count - 1] if tail_count else float("nan")
            tail_events = [str(row.get("event_id")) for row in event_rows if _float(row.get("risk")) >= tail_threshold] if tail_count else []
            counts = {
                "TP": sum(_float(row.get("TP"), 0.0) for row in retained),
                "FP": sum(_float(row.get("FP"), 0.0) for row in retained),
                "FN": sum(_float(row.get("FN"), 0.0) for row in retained),
            }
            aggregate_iou = counts["TP"] / (counts["TP"] + counts["FP"] + counts["FN"]) if (counts["TP"] + counts["FP"] + counts["FN"]) else float("nan")
            summary_rows.append(
                {
                    "run_name": run_name,
                    "coverage_target": coverage,
                    "retained_coverage": retained_valid / total_valid if total_valid else float("nan"),
                    "abstention_rate": 1.0 - (retained_valid / total_valid) if total_valid else float("nan"),
                    "retained_chips": len(retained),
                    "total_chips": len(sorted_rows),
                    "confidence_column": column,
                    "aggregate_iou": aggregate_iou,
                    "aggregate_risk": 1.0 - aggregate_iou if not math.isnan(aggregate_iou) else float("nan"),
                    "mean_event_risk": mean_risk,
                    "tail_event_risk": tail_risk,
                    "raw_bwer_event_id": bwer,
                    "tail_events": ";".join(sorted(set(tail_events))),
                    "method": "posthoc_chip_confidence_retention",
                }
            )
            for row in event_rows:
                by_event_rows.append(
                    {
                        "run_name": run_name,
                        "coverage_target": coverage,
                        "event_id": row.get("event_id"),
                        "retained_coverage": retained_valid / total_valid if total_valid else float("nan"),
                        "micro_iou": row.get("micro_iou"),
                        "micro_dice": row.get("micro_dice"),
                        "risk": row.get("risk"),
                        "valid_pixel_count": row.get("valid_pixel_count"),
                        "positive_pixel_count": row.get("positive_pixel_count"),
                        "tail_flag": str(row.get("event_id")) in set(tail_events),
                    }
                )
    write_csv(artifacts["availability"], availability)
    write_csv(artifacts["summary"], summary_rows)
    write_csv(artifacts["by_event"], by_event_rows)
    _write_selective_risk_report(artifacts["report"], availability, summary_rows)
    return artifacts


def _write_selective_risk_report(path: Path, availability: Sequence[Mapping[str, Any]], summary_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Selective Risk Availability and Post-hoc Audit",
        "",
        "This audit is post-hoc. It uses saved confidence/logit/probability summaries only and does not rerun inference.",
        "",
        "If only chip-level confidence summaries are available, results retain or abstain from whole chips. They are not pixel-level selective risk.",
        "",
        "## Availability",
        "",
        "| run | available | confidence column | method | notes |",
        "|---|---:|---|---|---|",
    ]
    for row in availability:
        lines.append(f"| {row.get('run_name')} | {row.get('available')} | {row.get('confidence_column', '')} | {row.get('method', '')} | {row.get('notes', '')} |")
    if summary_rows:
        lines.extend(
            [
                "",
                "## Outputs",
                "",
                "`selective_risk_summary.csv` and `selective_risk_by_event.csv` contain coverage-conditioned retained-chip results for runs with usable confidence summaries.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Limitation",
                "",
                "No supplied run had sufficient saved confidence/logit/probability fields for post-hoc selective risk. Future inference should save per-chip confidence at minimum and preferably per-pixel logits/probability maps for fixed-coverage selective segmentation risk.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
