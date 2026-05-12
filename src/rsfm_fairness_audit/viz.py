from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from rsfm_fairness_audit.metrics import GroupMetric


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_average_vs_worst(summary_rows: Sequence[dict], path: str | Path) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in summary_rows:
        ax.scatter(row["average_performance"], row["worst_region_performance"], s=64)
        ax.annotate(row["gap_name"], (row["average_performance"], row["worst_region_performance"]), fontsize=8)
    ax.plot([0, 1], [0, 1], color="0.7", linewidth=1)
    ax.set_xlabel("Average performance")
    ax.set_ylabel("Worst-group performance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sensor_heatmap(rows: Sequence[GroupMetric], path: str | Path) -> None:
    plt = _import_pyplot()
    groups = [row.group for row in rows]
    values = np.asarray([[row.accuracy] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(4, max(2.5, 0.45 * len(groups))))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(groups)), labels=groups)
    ax.set_xticks([0], labels=["accuracy"])
    fig.colorbar(image, ax=ax, fraction=0.08)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_representation_shift(embeddings: np.ndarray, metadata: Sequence[dict], path: str | Path) -> None:
    plt = _import_pyplot()
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    regions = sorted({str(row["region"]) for row in metadata})
    colors = {region: index for index, region in enumerate(regions)}
    values = [colors[str(row["region"])] for row in metadata]

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="tab10", s=18, alpha=0.85)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, regions, title="Region", fontsize=8, loc="best")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fairness_map(metadata: Sequence[dict], predictions: np.ndarray, labels: np.ndarray, path: str | Path) -> bool:
    coords = []
    values = []
    for index, row in enumerate(metadata):
        lat = row.get("latitude")
        lon = row.get("longitude")
        if lat in (None, "", "to_verify") or lon in (None, "", "to_verify"):
            continue
        try:
            coords.append((float(lon), float(lat)))
            values.append(float(labels[index] == predictions[index]))
        except (TypeError, ValueError):
            continue
    plt = _import_pyplot()
    if not coords:
        groups = sorted({str(row.get("fallback_group") or row.get("region") or "to_verify") for row in metadata})
        values_by_group = []
        for group in groups:
            indices = [
                index
                for index, row in enumerate(metadata)
                if str(row.get("fallback_group") or row.get("region") or "to_verify") == group
            ]
            values_by_group.append(float(np.mean(labels[indices] == predictions[indices])) if indices else 0.0)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(groups, values_by_group, color="#4C78A8")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Fallback group")
        ax.set_title("Fallback grouping only - not a geographic map")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.20)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return True

    xy = np.asarray(coords, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap="RdYlGn", vmin=0, vmax=1, s=32, alpha=0.85)
    fig.colorbar(scatter, ax=ax, label="Correct prediction")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True
