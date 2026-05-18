from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter
from rsfm_fairness_audit.adapters.prithvi import PrithviAdapter
from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_table
from rsfm_fairness_audit.audit_table import write_audit_table
from rsfm_fairness_audit.io import ensure_dir, write_csv
from rsfm_fairness_audit.slice_support import evaluate_slice_support


def _binary_iou(mask: np.ndarray, prediction: np.ndarray) -> float:
    valid = mask >= 0
    if not np.any(valid):
        return float("nan")
    truth = mask[valid] == 1
    pred = prediction[valid] == 1
    union = np.logical_or(truth, pred).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(truth, pred).sum() / union)


def _pixel_accuracy(mask: np.ndarray, prediction: np.ndarray) -> float:
    valid = mask >= 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean((mask[valid] == 1) == (prediction[valid] == 1)))


def segmentation_confusion_counts(mask: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    valid = np.asarray(mask) >= 0
    if not np.any(valid):
        return {
            "valid_pixel_count": 0,
            "positive_pixel_count": 0,
            "predicted_positive_pixel_count": 0,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
        }
    truth = np.asarray(mask)[valid] == 1
    pred = np.asarray(prediction)[valid] == 1
    return {
        "valid_pixel_count": int(valid.sum()),
        "positive_pixel_count": int(truth.sum()),
        "predicted_positive_pixel_count": int(pred.sum()),
        "TP": int(np.logical_and(truth, pred).sum()),
        "FP": int(np.logical_and(~truth, pred).sum()),
        "FN": int(np.logical_and(truth, ~pred).sum()),
        "TN": int(np.logical_and(~truth, ~pred).sum()),
    }


def segmentation_metrics_from_counts(counts: dict[str, Any]) -> dict[str, float]:
    tp = float(counts.get("TP", 0) or 0)
    fp = float(counts.get("FP", 0) or 0)
    fn = float(counts.get("FN", 0) or 0)
    tn = float(counts.get("TN", 0) or 0)
    iou_den = tp + fp + fn
    dice_den = (2.0 * tp) + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    valid = tp + fp + fn + tn
    iou = 1.0 if iou_den == 0 else float(tp / iou_den)
    dice = 1.0 if dice_den == 0 else float((2.0 * tp) / dice_den)
    precision = 1.0 if precision_den == 0 else float(tp / precision_den)
    recall = 1.0 if recall_den == 0 else float(tp / recall_den)
    accuracy = float((tp + tn) / valid) if valid else float("nan")
    return {
        "iou": iou,
        "water_iou": iou,
        "micro_iou": iou,
        "dice": dice,
        "f1": dice,
        "micro_dice": dice,
        "micro_f1": dice,
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": accuracy,
        "risk": 1.0 - iou,
        "risk_source": "1_minus_iou",
    }


def _predict_from_features(features: np.ndarray) -> np.ndarray:
    if features.ndim == 3:
        score = features.mean(axis=0)
    elif features.ndim == 4:
        score = features.mean(axis=0)
    else:
        raise ValueError(f"Expected segmentation features [C,H,W] or [T,C,H,W], got {features.shape}.")
    threshold = float(np.nanmean(score))
    return (score >= threshold).astype(np.int16)


def aggregate_segmentation_metrics(
    rows: Sequence[dict[str, Any]],
    group_key: str,
    aggregation_level: str = "event",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "to_verify"))].append(row)
    output: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items()):
        counts = {
            "valid_pixel_count": int(sum(int(row.get("valid_pixel_count", 0) or 0) for row in items)),
            "positive_pixel_count": int(sum(int(row.get("positive_pixel_count", 0) or 0) for row in items)),
            "predicted_positive_pixel_count": int(sum(int(row.get("predicted_positive_pixel_count", 0) or 0) for row in items)),
            "TP": int(sum(int(row.get("TP", 0) or 0) for row in items)),
            "FP": int(sum(int(row.get("FP", 0) or 0) for row in items)),
            "FN": int(sum(int(row.get("FN", 0) or 0) for row in items)),
            "TN": int(sum(int(row.get("TN", 0) or 0) for row in items)),
        }
        metrics = segmentation_metrics_from_counts(counts)
        first = items[0]
        row: dict[str, Any] = {
            "dataset": first.get("dataset", "sen1floods11"),
            "model": first.get("model", "prithvi"),
            "task": "segmentation",
            "split": first.get("split", "all"),
            "unit_id": group,
            "sample_id": group,
            "event_id": group if group_key in {"event_id", "event"} else first.get("event_id", first.get("event", "to_verify")),
            "event": group if group_key == "event" else first.get("event", first.get("event_id", "to_verify")),
            "country": first.get("country", first.get("region", "to_verify")),
            "region": first.get("region", "to_verify"),
            "class_label": "water",
            "input_mode": first.get("input_mode", "S2"),
            "adaptation_protocol": first.get("adaptation_protocol", "frozen_encoder_lightweight_head"),
            "training_budget": first.get("training_budget", "unsupervised_threshold_head"),
            "split_protocol": first.get("split_protocol", "standard_split"),
            "aggregation_level": aggregation_level,
            "sample_count": len(items),
            "TP_plus_FN_support": counts["TP"] + counts["FN"],
            **counts,
            **metrics,
            "score": metrics["micro_iou"],
        }
        for key in ["month", "season", "sensor", "label_source"]:
            if key in first:
                row[key] = first[key]
        output.append(row)
    return output


def _group_rows(rows: Sequence[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    output = []
    for row in aggregate_segmentation_metrics(rows, group_key, aggregation_level="slice"):
        output.append(
            {
                "slice_name": group_key,
                "group": row.get(group_key, row.get("event_id", "")) if group_key in row else row["unit_id"],
                "n": row["sample_count"],
                "valid_pixel_support": row["valid_pixel_count"],
                "positive_pixel_support": row["positive_pixel_count"],
                "mean_water_iou": row["micro_iou"],
                "mean_pixel_accuracy": row["pixel_accuracy"],
            }
        )
    return output


def plot_segmentation_iou_by_group(rows: Sequence[dict[str, Any]], output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [str(row["group"]) for row in rows]
    values = [float(row["mean_water_iou"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(groups))))
    ax.bar(groups, values, color="#4C78A8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean water IoU")
    ax.set_xlabel("Group")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_segmentation_smoke(
    dataset: DatasetAdapter,
    model: PrithviAdapter,
    output_dir: str | Path,
    batch_size: int | None = None,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    figures = ensure_dir(output / "figures")
    tables = ensure_dir(output / "tables")
    metadata = dataset.load_metadata()
    model.load_model()
    batch_size = batch_size or int(getattr(model, "batch_size", 2))
    metric_rows: list[dict[str, Any]] = []
    for start in range(0, len(metadata), batch_size):
        indices = list(range(start, min(start + batch_size, len(metadata))))
        samples = [dataset.load_sample(index) for index in indices]
        batch = model.preprocess({"samples": samples, "metadata": [metadata[index] for index in indices]})
        features = model.segmentation_features(batch)
        masks = batch["masks"]
        for local_index, dataset_index in enumerate(indices):
            prediction = _predict_from_features(features[local_index])
            mask = masks[local_index].astype(np.int16)
            counts = segmentation_confusion_counts(mask, prediction)
            metrics = segmentation_metrics_from_counts(counts)
            row = dict(metadata[dataset_index])
            event_id = str(row.get("event_id") or row.get("event") or row.get("region") or "to_verify")
            row["event_id"] = event_id
            row.setdefault("event", event_id)
            row.update(
                {
                    "dataset": "sen1floods11",
                    "model": "prithvi",
                    "task": "segmentation",
                    "input_mode": "S2",
                    "adaptation_protocol": "frozen_encoder_lightweight_head",
                    "training_budget": "unsupervised_threshold_head",
                    "split_protocol": "standard_split",
                    "aggregation_level": "chip",
                    "unit_id": row.get("sample_id"),
                    "class_label": "water",
                    "TP_plus_FN_support": counts["TP"] + counts["FN"],
                    **counts,
                    **metrics,
                    "score": metrics["micro_iou"],
                }
            )
            metric_rows.append(row)

    region_rows = _group_rows(metric_rows, "region")
    event_group_key = "event_id" if "event_id" in metric_rows[0] else "event"
    event_rows = _group_rows(metric_rows, event_group_key)
    event_metric_rows = aggregate_segmentation_metrics(metric_rows, event_group_key, aggregation_level="event")
    audit_rows = build_audit_table_from_segmentation_metrics_from_rows(event_metric_rows)
    artifacts = {
        "segmentation_metrics": output / "segmentation_metrics.csv",
        "segmentation_predictions": output / "segmentation_predictions.csv",
        "event_segmentation_metrics": output / "event_segmentation_metrics.csv",
        "segmentation_audit_table": output / "segmentation_audit_table.csv",
        "audit_table": output / "audit_table.csv",
        "support_recommendations": output / "slice_support_recommendations.csv",
        "support_summary": output / "slice_support_summary.csv",
        "support_preflight_report": output / "slice_support_report.md",
        "segmentation_fairness_matrix_region": output / "segmentation_fairness_matrix_region.csv",
        "segmentation_fairness_matrix_event": output / "segmentation_fairness_matrix_event.csv",
        "tables_segmentation_metrics": tables / "segmentation_metrics.csv",
        "tables_region": tables / "segmentation_fairness_matrix_region.csv",
        "tables_event": tables / "segmentation_fairness_matrix_event.csv",
        "iou_by_group": figures / "segmentation_iou_by_group.png",
        "segmentation_report": output / "segmentation_report.md",
        "report": output / "report.md",
    }
    write_csv(artifacts["segmentation_metrics"], metric_rows)
    write_csv(artifacts["segmentation_predictions"], metric_rows)
    write_csv(artifacts["event_segmentation_metrics"], event_metric_rows)
    write_audit_table(artifacts["segmentation_audit_table"], audit_rows)
    write_audit_table(artifacts["audit_table"], audit_rows)
    write_csv(artifacts["segmentation_fairness_matrix_region"], region_rows)
    write_csv(artifacts["segmentation_fairness_matrix_event"], event_rows)
    write_csv(artifacts["tables_segmentation_metrics"], metric_rows)
    write_csv(artifacts["tables_region"], region_rows)
    write_csv(artifacts["tables_event"], event_rows)
    plot_segmentation_iou_by_group(region_rows, artifacts["iou_by_group"])
    preflight = evaluate_slice_support(
        audit_rows,
        dataset="sen1floods11",
        model="prithvi",
        task="segmentation",
        output_dir=output,
        candidates=["event_id", "event_id|event", "event_id|month", "event_id|season", "country|country"],
        score_column="micro_iou",
        risk_column="risk",
    )
    artifacts.update({f"preflight_{key}": value for key, value in preflight.items()})
    preflight_warnings = _read_warnings(output / "warnings.json")
    try:
        bwer = evaluate_bwer_table(
            audit_rows,
            dataset="sen1floods11",
            model="prithvi",
            task="segmentation",
            output_dir=output,
            slice_variable="event_id",
            balance_variable="raw",
            score_column="micro_iou",
            risk_column="risk",
            audit_level="pilot",
        )
        artifacts.update(bwer)
        _write_combined_warnings(output / "warnings.json", preflight_warnings)
    except ValueError as exc:
        (output / "bwer_not_runnable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        artifacts["bwer_not_runnable"] = output / "bwer_not_runnable.txt"
    report_path = artifacts["report"] if "bwer_not_runnable" in artifacts else artifacts["segmentation_report"]
    _write_report(report_path, region_rows, event_rows, event_metric_rows)
    return artifacts


def _read_warnings(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [str(value) for value in data.get("warnings", [])]


def _write_combined_warnings(path: Path, extra_warnings: Sequence[str]) -> None:
    combined = set(extra_warnings)
    combined.update(_read_warnings(path))
    path.write_text(json.dumps({"warnings": sorted(combined)}, indent=2), encoding="utf-8")


def build_audit_table_from_segmentation_metrics_from_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    temp_rows = []
    for row in rows:
        item = dict(row)
        item["score"] = item.get("micro_iou", item.get("iou", item.get("score", "")))
        item["risk"] = item.get("risk", "")
        temp_rows.append(item)
    from rsfm_fairness_audit.audit_table import validate_audit_table

    validate_audit_table(temp_rows)
    return temp_rows


def _write_report(
    path: Path,
    region_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    event_metric_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Prithvi-EO-2.0 Sen1Floods11 Native Segmentation Audit",
        "",
        "This run uses Prithvi-EO-2.0-300M non-TL as a frozen encoder with a lightweight unsupervised threshold head. The protocol is recorded as `frozen_encoder_lightweight_head`; it is a readiness path for native pixel-level audit, not a supervised flood fine-tune.",
        "",
        "The native segmentation audit ignores hand-label mask pixels with value -1 and computes event-level IoU/Dice/F1/precision/recall from aggregated TP/FP/FN/TN counts.",
        "",
        "BWER is a support-aware, composition-standardised, CVaR-style tail-risk statistic for deployment-relevant remote sensing slices.",
        "",
        "Chip-level Sen1Floods11 classification is a sanity audit. Native pixel-level Sen1Floods11 segmentation is the paper-grade disaster/event fairness path. Here, `event_id` is an operational disaster-event slice, not a causal country fairness attribute.",
        "",
        "If the loaded Prithvi backbone does not expose dense patch features, this path uses a transparent normalized-spectral fallback. Those fallback numbers validate pipeline wiring only and should not be interpreted as Prithvi segmentation quality.",
        "",
        "## Region Groups",
        "",
        "| group | n | mean_water_iou | mean_pixel_accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in region_rows:
        lines.append(f"| {row['group']} | {row['n']} | {row['mean_water_iou']:.4f} | {row['mean_pixel_accuracy']:.4f} |")
    lines.extend(["", "## Event Groups", "", "| group | n | mean_water_iou | mean_pixel_accuracy |", "|---|---:|---:|---:|"])
    for row in event_rows:
        lines.append(f"| {row['group']} | {row['n']} | {row['mean_water_iou']:.4f} | {row['mean_pixel_accuracy']:.4f} |")
    lines.extend(
        [
            "",
            "## Event-Level Aggregated Counts",
            "",
            "| event_id | chips | valid pixels | positive pixels | TP | FP | FN | micro IoU | micro Dice | risk |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in event_metric_rows:
        lines.append(
            f"| {row['event_id']} | {row['sample_count']} | {row['valid_pixel_count']} | {row['positive_pixel_count']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['micro_iou']:.4f} | {row['micro_dice']:.4f} | {row['risk']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
