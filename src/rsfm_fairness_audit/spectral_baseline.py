from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_table
from rsfm_fairness_audit.audit_table import write_audit_table
from rsfm_fairness_audit.bwer_v2 import run_bwer_v2_posthoc
from rsfm_fairness_audit.io import ensure_dir, write_csv
from rsfm_fairness_audit.segmentation import (
    aggregate_segmentation_metrics,
    build_audit_table_from_segmentation_metrics_from_rows,
    plot_segmentation_iou_by_group,
    segmentation_confusion_counts,
    segmentation_metrics_from_counts,
)
from rsfm_fairness_audit.slice_support import evaluate_slice_support
from rsfm_fairness_audit.unet_baseline import _load_npz_array, _prepare_image, _read_metadata


S2_BAND_NAMES = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")


@dataclass(frozen=True)
class SpectralBaselineConfig:
    data_root: Path
    output_dir: Path
    index: str = "mndwi"
    threshold: float = 0.0
    threshold_policy: str = "fixed"
    split_protocol: str = "standard_split"
    val_fraction: float = 0.15
    test_fraction: float = 0.20
    eval_split: str = "all"
    seed: int = 42
    max_samples: int | None = None
    run_bwer_v2: bool = False


def spectral_indices(image: np.ndarray, eps: float = 1e-6) -> dict[str, np.ndarray]:
    """Compute Sen1Floods11 S2 6-band spectral water scores from [C,H,W] chips."""
    arr = _prepare_image(image)
    green = arr[1]
    nir = arr[3]
    swir1 = arr[4]
    ndwi = (green - nir) / (green + nir + eps)
    mndwi = (green - swir1) / (green + swir1 + eps)
    return {
        "ndwi": np.asarray(ndwi, dtype=np.float32),
        "mndwi": np.asarray(mndwi, dtype=np.float32),
        "nir_darkness": np.asarray(nir, dtype=np.float32),
    }


def threshold_spectral_index(score: np.ndarray, index: str, threshold: float) -> np.ndarray:
    if index == "nir_darkness":
        return (score <= threshold).astype(np.int16)
    return (score >= threshold).astype(np.int16)


def _index_display(index: str, threshold: float, threshold_policy: str) -> str:
    sign = "le" if index == "nir_darkness" else "ge"
    safe_threshold = str(threshold).replace("-", "neg").replace(".", "p")
    return f"spectral_{index}_{threshold_policy}_{sign}_{safe_threshold}"


def _split_rows(rows: Sequence[dict[str, Any]], config: SpectralBaselineConfig) -> dict[str, list[dict[str, Any]]]:
    if config.split_protocol == "standard_split":
        return {"train": [], "val": list(rows), "test": list(rows), "all": list(rows)}
    if config.split_protocol != "random_chip_split":
        raise ValueError("Spectral baseline split_protocol must be standard_split or random_chip_split.")
    rng = random.Random(config.seed)
    shuffled = [dict(row) for row in rows]
    rng.shuffle(shuffled)
    test_n = max(1, int(round(len(shuffled) * config.test_fraction))) if len(shuffled) > 2 else 1
    val_n = max(1, int(round(len(shuffled) * config.val_fraction))) if len(shuffled) - test_n > 2 else 1
    test = shuffled[:test_n]
    val = shuffled[test_n : test_n + val_n]
    train = shuffled[test_n + val_n :]
    return {"train": train, "val": val, "test": test, "all": shuffled}


def _threshold_grid(index: str) -> list[float]:
    if index == "nir_darkness":
        return [round(value, 3) for value in np.linspace(0.02, 0.35, 18)]
    return [round(value, 3) for value in np.linspace(-0.35, 0.35, 29)]


def _score_threshold(data_root: Path, rows: Sequence[dict[str, Any]], index: str, threshold: float) -> dict[str, float]:
    totals = {"valid_pixel_count": 0, "positive_pixel_count": 0, "predicted_positive_pixel_count": 0, "TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for row in rows:
        image = _load_npz_array(data_root, str(row["chip_path"]), "image")
        mask = _load_npz_array(data_root, str(row["mask_path"]), "mask").astype(np.int16)
        score = spectral_indices(image)[index]
        pred = threshold_spectral_index(score, index, threshold)
        counts = segmentation_confusion_counts(mask, pred)
        for key in totals:
            totals[key] += int(counts[key])
    return segmentation_metrics_from_counts(totals)


def _select_threshold(data_root: Path, rows: Sequence[dict[str, Any]], config: SpectralBaselineConfig) -> tuple[float, list[dict[str, Any]], str]:
    if config.threshold_policy == "fixed":
        return config.threshold, [], "fixed_threshold_no_label_selection"
    if not rows:
        raise ValueError(f"threshold_policy={config.threshold_policy} requires non-empty rows for threshold selection.")
    selection_rows: list[dict[str, Any]] = []
    best_threshold = config.threshold
    best_iou = -math.inf
    for threshold in _threshold_grid(config.index):
        metrics = _score_threshold(config.data_root, rows, config.index, threshold)
        selection_rows.append(
            {
                "index": config.index,
                "threshold": threshold,
                "selection_metric": "micro_iou",
                "selection_iou": metrics["micro_iou"],
                "threshold_policy": config.threshold_policy,
            }
        )
        if metrics["micro_iou"] > best_iou:
            best_iou = metrics["micro_iou"]
            best_threshold = threshold
    if config.threshold_policy == "validation":
        return best_threshold, selection_rows, "validation_selected_threshold"
    if config.threshold_policy == "oracle_diagnostic":
        return best_threshold, selection_rows, "oracle_diagnostic_threshold_selected_on_evaluation_labels_excluded_from_primary_claims"
    raise ValueError("threshold_policy must be fixed, validation, or oracle_diagnostic.")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _chip_row(
    row: Mapping[str, Any],
    counts: Mapping[str, int],
    metrics: Mapping[str, float],
    model_name: str,
    config: SpectralBaselineConfig,
    threshold: float,
    threshold_note: str,
    index: str,
) -> dict[str, Any]:
    event_id = str(row.get("event_id") or row.get("event") or row.get("region") or "to_verify")
    item = dict(row)
    item.update(
        {
            "dataset": "sen1floods11",
            "model": model_name,
            "model_family": "spectral_rule",
            "model_variant": model_name,
            "display_name": "Sen1Floods11 spectral water-index baseline",
            "task": "segmentation",
            "input_mode": "s2_6band_image_only",
            "adaptation_protocol": "diagnostic_spectral_rule",
            "training_budget": f"index={index};threshold={threshold};threshold_policy={config.threshold_policy};no_training",
            "split_protocol": config.split_protocol,
            "checkpoint_source": "not_applicable_spectral_rule",
            "split": config.eval_split,
            "aggregation_level": "chip",
            "unit_id": item.get("sample_id"),
            "sample_id": item.get("sample_id"),
            "event_id": event_id,
            "event": str(item.get("event") or event_id),
            "country": str(item.get("country") or item.get("region") or event_id),
            "class_label": "water",
            "TP_plus_FN_support": int(counts["TP"] + counts["FN"]),
            "threshold_index": index,
            "threshold_value": threshold,
            "threshold_policy": config.threshold_policy,
            "threshold_note": threshold_note,
            "input_band_order": ",".join(S2_BAND_NAMES),
            "band_profile": item.get("band_profile", "prithvi_tl_sen1floods11"),
            "label_mapping": "0=background;1=water_flood;-1=ignore",
            **counts,
            **metrics,
            "score": metrics["micro_iou"],
        }
    )
    return item


def _write_spectral_report(path: Path, metadata: Mapping[str, Any], event_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Sen1Floods11 Spectral Baseline Audit",
        "",
        "This run is a diagnostic S2 water-index segmentation baseline, not a learned foundation model or SOTA architecture.",
        "",
        f"- model_variant: {metadata['model_variant']}",
        f"- adaptation_protocol: {metadata['adaptation_protocol']}",
        f"- split_protocol: {metadata['split_protocol']}",
        f"- threshold_policy: {metadata['threshold_policy']}",
        f"- threshold_value: {metadata['threshold_value']}",
        f"- input_bands: {metadata['input_band_order']}",
        f"- resolution: {metadata['resolution']}",
        "",
        "Fixed thresholds are primary for full-set diagnostic evaluation. Thresholds selected on evaluation labels are marked `oracle_diagnostic` and should be excluded from primary claims.",
        "",
        "Event-level BWER is interpreted as deployment slice risk, not causal country fairness.",
        "",
        "## Event-Level Aggregated Counts",
        "",
        "| event_id | chips | valid pixels | positive pixels | TP | FP | FN | micro IoU | micro Dice | risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in event_rows:
        lines.append(
            f"| {row['event_id']} | {row['sample_count']} | {row['valid_pixel_count']} | {row['positive_pixel_count']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['micro_iou']:.4f} | {row['micro_dice']:.4f} | {row['risk']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_spectral_sen1floods11(config: SpectralBaselineConfig) -> dict[str, Path]:
    output = ensure_dir(config.output_dir)
    figures = ensure_dir(output / "figures")
    tables = ensure_dir(output / "tables")
    rows = _read_metadata(config.data_root, config.max_samples)
    splits = _split_rows(rows, config)
    selection_rows = splits["val"] if config.threshold_policy == "validation" else splits.get(config.eval_split, rows)
    threshold, threshold_grid_rows, threshold_note = _select_threshold(config.data_root, selection_rows, config)
    eval_rows = rows if config.eval_split == "all" else splits[config.eval_split]
    model_name = _index_display(config.index, threshold, config.threshold_policy)
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for row in eval_rows:
        image = _load_npz_array(config.data_root, str(row["chip_path"]), "image")
        mask = _load_npz_array(config.data_root, str(row["mask_path"]), "mask").astype(np.int16)
        index_maps = spectral_indices(image)
        primary_pred = threshold_spectral_index(index_maps[config.index], config.index, threshold)
        counts = segmentation_confusion_counts(mask, primary_pred)
        metrics = segmentation_metrics_from_counts(counts)
        metric_rows.append(_chip_row(row, counts, metrics, model_name, config, threshold, threshold_note, config.index))
        for name, score in index_maps.items():
            for candidate_threshold in ([threshold] if name == config.index else [0.0] if name != "nir_darkness" else [0.15]):
                pred = threshold_spectral_index(score, name, float(candidate_threshold))
                baseline_counts = segmentation_confusion_counts(mask, pred)
                baseline_metrics = segmentation_metrics_from_counts(baseline_counts)
                diagnostic_rows.append(
                    {
                        "baseline_name": _index_display(name, float(candidate_threshold), "fixed"),
                        "sample_id": row.get("sample_id"),
                        "event_id": row.get("event_id", row.get("event", "to_verify")),
                        "threshold": candidate_threshold,
                        "threshold_policy": "fixed",
                        **baseline_counts,
                        "ground_truth_positive_pixel_ratio": baseline_metrics["ground_truth_positive_pixel_ratio"],
                        "predicted_positive_pixel_ratio": baseline_metrics["predicted_positive_pixel_ratio"],
                        "iou": baseline_metrics["iou"],
                        "dice": baseline_metrics["dice"],
                        "precision": baseline_metrics["precision"],
                        "recall": baseline_metrics["recall"],
                        "pixel_accuracy": baseline_metrics["pixel_accuracy"],
                    }
                )
    event_rows = aggregate_segmentation_metrics(metric_rows, "event_id", aggregation_level="event")
    audit_rows = build_audit_table_from_segmentation_metrics_from_rows(event_rows)
    fairness_rows = [
        {
            "slice_name": "event_id",
            "group": row["event_id"],
            "n": row["sample_count"],
            "valid_pixel_support": row["valid_pixel_count"],
            "positive_pixel_support": row["positive_pixel_count"],
            "mean_water_iou": row["micro_iou"],
            "mean_pixel_accuracy": row["pixel_accuracy"],
        }
        for row in event_rows
    ]
    data_resolution = ""
    if metric_rows:
        first_image = _prepare_image(_load_npz_array(config.data_root, str(eval_rows[0]["chip_path"]), "image"))
        data_resolution = int(first_image.shape[-1]) if first_image.shape[-1] == first_image.shape[-2] else f"{first_image.shape[-2]}x{first_image.shape[-1]}"
    metadata = {
        "model_family": "spectral_rule",
        "model_variant": model_name,
        "display_name": "Sen1Floods11 spectral water-index baseline",
        "adaptation_protocol": "diagnostic_spectral_rule",
        "split_protocol": config.split_protocol,
        "eval_split": config.eval_split,
        "threshold_index": config.index,
        "threshold_value": threshold,
        "threshold_policy": config.threshold_policy,
        "threshold_note": threshold_note,
        "resolution": data_resolution,
        "input_mode": "s2_6band_image_only",
        "input_band_order": ",".join(S2_BAND_NAMES),
        "band_profile": "prithvi_tl_sen1floods11",
        "label_mapping": "0=background;1=water_flood;-1=ignore",
        "training_budget": "no_training;fixed_or_validation_threshold_only",
    }
    artifacts = {
        "segmentation_metrics": output / "segmentation_metrics.csv",
        "segmentation_predictions": output / "segmentation_predictions.csv",
        "event_segmentation_metrics": output / "event_segmentation_metrics.csv",
        "diagnostic_baseline_per_chip": output / "diagnostic_baseline_per_chip.csv",
        "threshold_selection": output / "threshold_selection.csv",
        "segmentation_audit_table": output / "segmentation_audit_table.csv",
        "audit_table": output / "audit_table.csv",
        "segmentation_fairness_matrix_event": output / "segmentation_fairness_matrix_event.csv",
        "tables_segmentation_metrics": tables / "segmentation_metrics.csv",
        "tables_event": tables / "segmentation_fairness_matrix_event.csv",
        "iou_by_group": figures / "segmentation_iou_by_group.png",
        "run_metadata": output / "run_metadata.json",
        "model_debug": output / "model_debug.json",
        "segmentation_report": output / "segmentation_report.md",
        "report": output / "report.md",
    }
    write_csv(artifacts["segmentation_metrics"], metric_rows)
    write_csv(artifacts["segmentation_predictions"], metric_rows)
    write_csv(artifacts["event_segmentation_metrics"], event_rows)
    write_csv(artifacts["diagnostic_baseline_per_chip"], diagnostic_rows)
    write_csv(artifacts["threshold_selection"], threshold_grid_rows)
    write_audit_table(artifacts["segmentation_audit_table"], audit_rows)
    write_audit_table(artifacts["audit_table"], audit_rows)
    write_csv(artifacts["segmentation_fairness_matrix_event"], fairness_rows)
    write_csv(artifacts["tables_segmentation_metrics"], metric_rows)
    write_csv(artifacts["tables_event"], fairness_rows)
    plot_segmentation_iou_by_group(fairness_rows, artifacts["iou_by_group"])
    artifacts["run_metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["model_debug"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    preflight = evaluate_slice_support(
        audit_rows,
        dataset="sen1floods11",
        model=model_name,
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
        model=model_name,
        task="segmentation",
        output_dir=output,
        slice_variable="event_id",
        balance_variable="raw",
        score_column="micro_iou",
        risk_column="risk",
        audit_level="pilot",
    )
    artifacts.update(bwer)
    _write_spectral_report(artifacts["segmentation_report"], metadata, event_rows)
    _write_spectral_report(artifacts["report"], metadata, event_rows)
    if config.run_bwer_v2:
        artifacts.update({f"bwer_v2_{key}": value for key, value in run_bwer_v2_posthoc(output, output / "bwer_v2").items()})
    return artifacts
