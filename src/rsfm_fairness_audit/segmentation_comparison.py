from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


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


def _common(rows: list[dict[str, str]], key: str, default: str = "") -> str:
    values = [str(row.get(key, "") or "") for row in rows if str(row.get(key, "") or "")]
    if not values:
        return default
    first = values[0]
    return first if all(value == first for value in values) else "mixed"


def _metric_from_counts(rows: list[dict[str, str]]) -> dict[str, float]:
    tp = sum(_float(row.get("TP"), 0.0) for row in rows)
    fp = sum(_float(row.get("FP"), 0.0) for row in rows)
    fn = sum(_float(row.get("FN"), 0.0) for row in rows)
    iou_den = tp + fp + fn
    dice_den = (2.0 * tp) + fp + fn
    return {
        "aggregate_iou": float(tp / iou_den) if iou_den else float("nan"),
        "aggregate_dice": float((2.0 * tp) / dice_den) if dice_den else float("nan"),
    }


def _bwer_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "bwer_v2" / "bwer_v2_summary.csv"
    return read_csv_rows(path) if path.exists() else []


def _bwer_value(rows: list[dict[str, str]], analysis_type: str, balance_variable: str = "") -> dict[str, str]:
    for row in rows:
        if row.get("analysis_type") != analysis_type:
            continue
        if balance_variable and row.get("balance_variable") != balance_variable:
            continue
        if not balance_variable and row.get("balance_variable") not in {"", "raw"}:
            continue
        return row
    return {}


def _tail_flags(run_dir: Path) -> set[str]:
    path = run_dir / "bwer_v2" / "event_failure_analysis.csv"
    if not path.exists():
        return set()
    return {str(row.get("event_id", "")) for row in read_csv_rows(path) if str(row.get("tail_flag", "")).lower() == "true"}


def _event_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "event_segmentation_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing event_segmentation_metrics.csv: {path}")
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Empty event_segmentation_metrics.csv: {path}")
    return rows


def _run_summary(run_name: str, run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_rows = _event_rows(run_dir)
    bwer_rows = _bwer_rows(run_dir)
    raw = _bwer_value(bwer_rows, "raw")
    standardised = _bwer_value(bwer_rows, "standardised", "flood_extent_bin")
    metadata = _read_json(run_dir / "run_metadata.json") or _read_json(run_dir / "model_debug.json")
    counts_metrics = _metric_from_counts(event_rows)
    scored = sorted(event_rows, key=lambda row: _float(row.get("micro_iou"), _float(row.get("iou"))))
    worst = scored[0]
    best = scored[-1]
    ious = [_float(row.get("micro_iou"), _float(row.get("iou"))) for row in event_rows]
    tails = _tail_flags(run_dir)
    if not tails and raw.get("tail_slices"):
        tails = {item.strip() for item in str(raw["tail_slices"]).split(";") if item.strip()}
    model = _common(event_rows, "model", run_name)
    model_family = _common(event_rows, "model_family", str(metadata.get("model_family", "")))
    adaptation = _common(event_rows, "adaptation_protocol", str(metadata.get("adaptation_protocol", "")))
    split_protocol = _common(event_rows, "split_protocol", str(metadata.get("split_protocol", "")))
    summary = {
        "run_name": run_name,
        "model": model,
        "model_family": model_family,
        "model_variant": model,
        "adaptation_protocol": adaptation,
        "split_protocol": split_protocol,
        "eval_scope": _common(event_rows, "split", str(metadata.get("eval_split", ""))),
        "dataset": _common(event_rows, "dataset", "sen1floods11"),
        "task": _common(event_rows, "task", "segmentation"),
        "resolution": str(metadata.get("resolution", raw.get("resolution", ""))),
        "aggregate_iou": counts_metrics["aggregate_iou"],
        "aggregate_dice": counts_metrics["aggregate_dice"],
        "raw_bwer_event_id": _float(raw.get("bwer")),
        "standardised_bwer_event_id_flood_extent_bin": _float(standardised.get("bwer")),
        "worst_event": worst.get("event_id", worst.get("event", "")),
        "best_event": best.get("event_id", best.get("event", "")),
        "tail_events": ";".join(sorted(tails)),
        "event_iou_range": f"{min(ious):.6f}-{max(ious):.6f}" if ious else "",
        "run_dir": str(run_dir),
        "protocol_comparability_notes": _protocol_notes(model, adaptation, split_protocol),
    }
    long_rows: list[dict[str, Any]] = []
    for row in event_rows:
        event_id = str(row.get("event_id", row.get("event", "")))
        long_rows.append(
            {
                "event_id": event_id,
                "run_name": run_name,
                "model": model,
                "IoU": _float(row.get("micro_iou"), _float(row.get("iou"))),
                "Dice": _float(row.get("micro_dice"), _float(row.get("dice"))),
                "risk": _float(row.get("risk")),
                "precision": _float(row.get("precision")),
                "recall": _float(row.get("recall")),
                "positive_support": _float(row.get("positive_pixel_count")),
                "valid_support": _float(row.get("valid_pixel_count")),
                "tail_flag": event_id in tails,
            }
        )
    return summary, long_rows


def _protocol_notes(model: str, adaptation: str, split_protocol: str) -> str:
    notes = []
    if "prithvi" in model.lower():
        notes.append("Prithvi official task-adapted checkpoint; evaluate as adapted foundation-model route.")
    if adaptation == "supervised_baseline" or model.startswith("unet"):
        notes.append("U-Net supervised_baseline; deployment-practice baseline, not a foundation model.")
    if adaptation == "diagnostic_spectral_rule" or "spectral" in model.lower():
        notes.append("Spectral-rule diagnostic baseline; no learned weights and no SOTA claim.")
    if split_protocol == "random_chip_split":
        notes.append("random_chip_split is not event-held-out and may include event leakage.")
    return " ".join(notes)


def _add_pairwise_deltas(rows: list[dict[str, Any]], run_names: list[str]) -> list[dict[str, Any]]:
    if len(run_names) != 2:
        return rows
    by_event: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(str(row["event_id"]), {})[str(row["run_name"])] = row
    output = []
    for row in rows:
        event = str(row["event_id"])
        other = run_names[1] if row["run_name"] == run_names[0] else run_names[0]
        other_row = by_event.get(event, {}).get(other)
        item = dict(row)
        if other_row:
            item["delta_iou_vs_other"] = row["IoU"] - other_row["IoU"]
            item["delta_risk_vs_other"] = row["risk"] - other_row["risk"]
            item["other_run_name"] = other
        output.append(item)
    return output


def compare_segmentation_runs(
    runs: Mapping[str, str | Path],
    output_dir: str | Path,
    dataset_name: str = "sen1floods11",
    closure: bool = False,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    figures = ensure_dir(output / "figures")
    summaries: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    run_names = list(runs)
    for run_name, run_dir_value in runs.items():
        summary, rows = _run_summary(run_name, Path(run_dir_value))
        summary["dataset"] = summary.get("dataset") or dataset_name
        summaries.append(summary)
        event_rows.extend(rows)
    event_rows = _add_pairwise_deltas(event_rows, run_names)
    average_rows = [
        {
            "run_name": row["run_name"],
            "average_score": row["aggregate_iou"],
            "raw_bwer": row["raw_bwer_event_id"],
            "standardised_bwer": row["standardised_bwer_event_id_flood_extent_bin"],
            "model_label": row["model"],
            "protocol_label": row["adaptation_protocol"],
            "split_label": row["split_protocol"],
        }
        for row in summaries
    ]
    artifacts = {
        "comparison_summary": output / "comparison_summary.csv",
        "average_vs_bwer": output / "average_vs_bwer.csv",
        "event_level_comparison": output / "event_level_comparison.csv",
        "average_iou_vs_raw_bwer": figures / "average_iou_vs_raw_bwer.png",
        "average_iou_vs_standardised_bwer": figures / "average_iou_vs_standardised_bwer.png",
        "event_iou_comparison": figures / "event_iou_comparison.png",
        "event_risk_comparison": figures / "event_risk_comparison.png",
        "raw_vs_standardised_bwer": figures / "raw_vs_standardised_bwer.png",
        "comparison_report": output / "comparison_report.md",
    }
    if closure:
        artifacts.update(
            {
                "closure_comparison_summary": output / "closure_comparison_summary.csv",
                "closure_average_vs_bwer": output / "closure_average_vs_bwer.csv",
                "closure_event_level_comparison": output / "closure_event_level_comparison.csv",
                "closure_tail_event_overlap": output / "closure_tail_event_overlap.csv",
                "closure_report": output / "closure_report.md",
            }
        )
    write_csv(artifacts["comparison_summary"], summaries)
    write_csv(artifacts["average_vs_bwer"], average_rows)
    write_csv(artifacts["event_level_comparison"], event_rows)
    if closure:
        write_csv(artifacts["closure_comparison_summary"], summaries)
        write_csv(artifacts["closure_average_vs_bwer"], average_rows)
        write_csv(artifacts["closure_event_level_comparison"], event_rows)
        write_csv(artifacts["closure_tail_event_overlap"], _tail_overlap_rows(summaries))
    _write_figures(summaries, event_rows, artifacts)
    _write_report(artifacts["comparison_report"], summaries)
    if closure:
        _write_closure_report(artifacts["closure_report"], summaries)
    return artifacts


def _tail_overlap_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tail_sets = {
        str(row["run_name"]): {item for item in str(row.get("tail_events", "")).split(";") if item}
        for row in summaries
    }
    all_tail_events = sorted(set().union(*tail_sets.values())) if tail_sets else []
    rows = []
    for event in all_tail_events:
        present = [run for run, tails in tail_sets.items() if event in tails]
        rows.append(
            {
                "event_id": event,
                "tail_in_runs": ";".join(present),
                "tail_run_count": len(present),
                "all_runs_count": len(tail_sets),
                "persistent_tail_all_runs": len(present) == len(tail_sets),
            }
        )
    return rows


def _write_figures(summaries: list[dict[str, Any]], event_rows: list[dict[str, Any]], artifacts: Mapping[str, Path]) -> None:
    import matplotlib.pyplot as plt

    def scatter(y_key: str, path: Path, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for row in summaries:
            ax.scatter(row["aggregate_iou"], row[y_key], s=80)
            ax.annotate(str(row["run_name"]), (row["aggregate_iou"], row[y_key]), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Aggregate micro IoU")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    scatter("raw_bwer_event_id", artifacts["average_iou_vs_raw_bwer"], "Raw-BWER(event_id)")
    scatter("standardised_bwer_event_id_flood_extent_bin", artifacts["average_iou_vs_standardised_bwer"], "Standardised-BWER(event_id | flood_extent_bin)")
    _plot_event_metric(event_rows, "IoU", artifacts["event_iou_comparison"], "Event micro IoU")
    _plot_event_metric(event_rows, "risk", artifacts["event_risk_comparison"], "Event risk")
    labels = [str(row["run_name"]) for row in summaries]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, [row["raw_bwer_event_id"] for row in summaries], width, label="Raw")
    ax.bar(x + width / 2, [row["standardised_bwer_event_id_flood_extent_bin"] for row in summaries], width, label="Standardised")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("BWER")
    ax.set_title("Raw vs Standardised BWER")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(artifacts["raw_vs_standardised_bwer"], dpi=180)
    plt.close(fig)


def _plot_event_metric(rows: list[dict[str, Any]], metric: str, path: Path, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    events = sorted({str(row["event_id"]) for row in rows})
    runs = sorted({str(row["run_name"]) for row in rows})
    x = np.arange(len(events))
    width = 0.8 / max(len(runs), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(events) * 0.7), 4.5))
    for index, run in enumerate(runs):
        values = []
        for event in events:
            row = next((item for item in rows if item["event_id"] == event and item["run_name"] == run), {})
            values.append(_float(row.get(metric)))
        ax.bar(x + (index - (len(runs) - 1) / 2) * width, values, width, label=run)
    ax.set_xticks(x, events, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    ordered_avg = sorted(summaries, key=lambda row: row["aggregate_iou"], reverse=True)
    ordered_raw = sorted(summaries, key=lambda row: row["raw_bwer_event_id"])
    reversal = [row["run_name"] for row in ordered_avg] != [row["run_name"] for row in ordered_raw]
    shared_tail_notes = "; ".join(f"{row['run_name']}: {row['tail_events']}" for row in summaries)
    lines = [
        "# Sen1Floods11 Segmentation Model Comparison",
        "",
        f"- Aggregate IoU ranking: {' > '.join(row['run_name'] for row in ordered_avg)}",
        f"- Raw-BWER ranking, lower is better: {' < '.join(row['run_name'] for row in ordered_raw)}",
        f"- Average-vs-BWER ranking reversal: {reversal}",
        f"- Tail events by run: {shared_tail_notes}",
        "",
        "## Protocol Caveat",
        "",
        "Prithvi is an official task-adapted checkpoint evaluated on its completed output directory. U-Net is a supervised_baseline under the split protocol recorded in its output, commonly random_chip_split test evaluation. This is a protocol-aware deployment-practice comparison, not a pure architecture-only comparison.",
        "",
        "Do not overclaim event-held-out generalization unless each compared run records an event-held-out or leave-one-event-out split protocol.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_closure_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    ordered_avg = sorted(summaries, key=lambda row: row["aggregate_iou"], reverse=True)
    ordered_raw = sorted(summaries, key=lambda row: row["raw_bwer_event_id"])
    ordered_std = sorted(summaries, key=lambda row: row["standardised_bwer_event_id_flood_extent_bin"])
    reversal_raw = [row["run_name"] for row in ordered_avg] != [row["run_name"] for row in ordered_raw]
    tail_sets = {
        str(row["run_name"]): {item for item in str(row.get("tail_events", "")).split(";") if item}
        for row in summaries
    }
    persistent = sorted(set.intersection(*tail_sets.values())) if tail_sets and all(tail_sets.values()) else []
    spectral = [row for row in summaries if row.get("adaptation_protocol") == "diagnostic_spectral_rule" or "spectral" in str(row.get("model", "")).lower()]
    strong_unet = [row for row in summaries if "resnet34" in str(row.get("model", "")).lower()]
    lines = [
        "# Sen1Floods11 Closure Comparison",
        "",
        "This closure package compares completed native segmentation outputs only. It does not rerun model inference, training, or data preparation.",
        "",
        f"- Aggregate IoU ranking: {' > '.join(row['run_name'] for row in ordered_avg)}",
        f"- Raw-BWER ranking, lower is better: {' < '.join(row['run_name'] for row in ordered_raw)}",
        f"- Standardised-BWER ranking, lower is better: {' < '.join(row['run_name'] for row in ordered_std)}",
        f"- Average-vs-BWER ranking reversal: {reversal_raw}",
        f"- Persistent tail events across all runs with tail labels: {';'.join(persistent) if persistent else 'not established from current run set'}",
        "",
        "## Protocol-Aware Interpretation",
        "",
        "Prithvi TL is an official task-adapted foundation-model checkpoint. Vanilla U-Net and S2 ResNet34-U-Net are supervised baselines under their recorded split protocols. Spectral baselines are diagnostic fixed-rule baselines and should not be reported as learned model SOTA.",
        "",
        "The comparison is useful for deployment-practice average-vs-tail-risk analysis, but it is not a pure architecture-only comparison and does not establish event-held-out generalization unless each run records such a split.",
        "",
        "## Spectral Baseline Check",
        "",
        "Spectral runs should be interpreted as an answer to whether simple S2 water-index rules explain part of the behavior. If their tail events differ from learned models, report that as diagnostic evidence about model-specific failure modes.",
        f"- Spectral runs present: {';'.join(row['run_name'] for row in spectral) if spectral else 'none in this comparison'}",
        "",
        "## Strong U-Net-Family Check",
        "",
        "The S2 ResNet34-U-Net / AlbuNet-style run tests whether the vanilla U-Net was too weak. Compare its aggregate IoU and BWER against the vanilla U-Net, not only against Prithvi.",
        f"- S2 ResNet34-U-Net runs present: {';'.join(row['run_name'] for row in strong_unet) if strong_unet else 'none in this comparison'}",
        "",
        "## Future LOEO and Selective Risk Notes",
        "",
        "LOEO should hold out one disaster event at a time, train only on the remaining events, evaluate on the held-out event, and then write the same segmentation_metrics.csv, event_segmentation_metrics.csv, BWER v2, and comparison tables. This task intentionally does not implement that workflow.",
        "",
        "Selective Risk requires saved probability, logit, or confidence fields. Current U-Net outputs include sigmoid confidence summaries but not full saved probability maps; Prithvi TL may include confidence diagnostics depending on run settings; spectral rules have deterministic scores but no calibrated confidence. Treat selective risk as unavailable unless the completed run saves enough per-pixel or per-chip confidence outputs for fixed-coverage retention analysis.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
