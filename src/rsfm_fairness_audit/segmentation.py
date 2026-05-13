from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter
from rsfm_fairness_audit.adapters.prithvi import PrithviAdapter
from rsfm_fairness_audit.io import ensure_dir, write_csv


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


def _predict_from_features(features: np.ndarray) -> np.ndarray:
    if features.ndim == 3:
        score = features.mean(axis=0)
    elif features.ndim == 4:
        score = features.mean(axis=0)
    else:
        raise ValueError(f"Expected segmentation features [C,H,W] or [T,C,H,W], got {features.shape}.")
    threshold = float(np.nanmean(score))
    return (score >= threshold).astype(np.int16)


def _group_rows(rows: Sequence[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "to_verify"))].append(row)
    output = []
    for group, items in sorted(grouped.items()):
        ious = np.asarray([float(row["water_iou"]) for row in items], dtype=float)
        accs = np.asarray([float(row["pixel_accuracy"]) for row in items], dtype=float)
        output.append(
            {
                "slice_name": group_key,
                "group": group,
                "n": len(items),
                "mean_water_iou": float(np.nanmean(ious)),
                "mean_pixel_accuracy": float(np.nanmean(accs)),
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
            row = dict(metadata[dataset_index])
            row.update(
                {
                    "water_iou": _binary_iou(mask, prediction),
                    "pixel_accuracy": _pixel_accuracy(mask, prediction),
                }
            )
            metric_rows.append(row)

    region_rows = _group_rows(metric_rows, "region")
    event_rows = _group_rows(metric_rows, "event")
    artifacts = {
        "segmentation_metrics": output / "segmentation_metrics.csv",
        "segmentation_fairness_matrix_region": output / "segmentation_fairness_matrix_region.csv",
        "segmentation_fairness_matrix_event": output / "segmentation_fairness_matrix_event.csv",
        "tables_segmentation_metrics": tables / "segmentation_metrics.csv",
        "tables_region": tables / "segmentation_fairness_matrix_region.csv",
        "tables_event": tables / "segmentation_fairness_matrix_event.csv",
        "iou_by_group": figures / "segmentation_iou_by_group.png",
        "report": output / "report.md",
    }
    write_csv(artifacts["segmentation_metrics"], metric_rows)
    write_csv(artifacts["segmentation_fairness_matrix_region"], region_rows)
    write_csv(artifacts["segmentation_fairness_matrix_event"], event_rows)
    write_csv(artifacts["tables_segmentation_metrics"], metric_rows)
    write_csv(artifacts["tables_region"], region_rows)
    write_csv(artifacts["tables_event"], event_rows)
    plot_segmentation_iou_by_group(region_rows, artifacts["iou_by_group"])
    _write_report(artifacts["report"], region_rows, event_rows)
    return artifacts


def _write_report(path: Path, region_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Prithvi-EO-2.0 Sen1Floods11 Segmentation Smoke",
        "",
        "This run uses Prithvi-EO-2.0-300M non-TL as a frozen backbone. It is a smoke validation only, not a paper-grade flood segmentation result.",
        "",
        "The lightweight segmentation path ignores hand-label mask pixels with value -1.",
        "",
        "If the loaded Prithvi backbone does not expose dense patch features, this smoke path uses a transparent normalized-spectral fallback. Those fallback numbers validate pipeline wiring only and should not be interpreted as Prithvi segmentation quality.",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
