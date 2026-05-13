from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.io import ensure_dir


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _float(row: Mapping[str, object], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def plot_average_vs_bwer(summary_rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    plt = _pyplot()
    ensure_dir(Path(path).parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    for index, row in enumerate(summary_rows):
        mean_risk = _float(row, "mean_risk")
        ax.scatter(1.0 - mean_risk, _float(row, "bwer"), s=60)
        label = f"{row.get('slice_variable')}|{row.get('balance_variable') or 'raw'}"
        ax.annotate(label, (1.0 - mean_risk, _float(row, "bwer")), xytext=(5, 5 + 6 * (index % 3)), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Mean score across valid slices")
    ax.set_ylabel("BWER")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bwer_by_model(summary_rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    plt = _pyplot()
    ensure_dir(Path(path).parent)
    labels = [f"{row.get('model')}:{row.get('slice_variable')}|{row.get('balance_variable') or 'raw'}" for row in summary_rows]
    values = [_float(row, "bwer", 0.0) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(labels)), 4))
    ax.bar(np.arange(len(labels)), values, color="#4C78A8")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("BWER")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_raw_vs_balanced_bwer(summary_rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    plt = _pyplot()
    ensure_dir(Path(path).parent)
    raw_by_slice = {str(row.get("slice_variable")): _float(row, "bwer") for row in summary_rows if not row.get("balance_variable")}
    rows = [row for row in summary_rows if row.get("balance_variable") and str(row.get("slice_variable")) in raw_by_slice]
    labels = [f"{row.get('slice_variable')}|{row.get('balance_variable')}" for row in rows]
    raw = [raw_by_slice[str(row.get("slice_variable"))] for row in rows]
    balanced = [_float(row, "bwer") for row in rows]
    if not rows:
        labels = [str(row.get("slice_variable")) for row in summary_rows]
        raw = [_float(row, "bwer") for row in summary_rows]
        balanced = raw
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 4))
    width = 0.35
    ax.bar(x - width / 2, raw, width, label="Raw")
    ax.bar(x + width / 2, balanced, width, label="Balanced")
    ax.set_xticks(x, labels=labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("BWER")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_worst_tail_slices(slice_rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    plt = _pyplot()
    ensure_dir(Path(path).parent)
    tail = [row for row in slice_rows if str(row.get("is_tail_slice")).lower() in {"true", "1"}]
    if not tail:
        tail = sorted(slice_rows, key=lambda row: -_float(row, "balanced_risk", 0.0))[:10]
    tail = sorted(tail, key=lambda row: _float(row, "balanced_risk", 0.0), reverse=True)[:20]
    labels = [f"{row.get('slice_variable')}={row.get('slice_value')}" for row in tail]
    values = [_float(row, "balanced_risk", 0.0) for row in tail]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(labels))))
    ax.barh(np.arange(len(labels)), values, color="#D55E00")
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Balanced risk")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_slice_risk_heatmap(slice_rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    plt = _pyplot()
    ensure_dir(Path(path).parent)
    rows = sorted(slice_rows, key=lambda row: (str(row.get("slice_variable")), str(row.get("slice_value"))))
    labels = [f"{row.get('slice_variable')}={row.get('slice_value')}" for row in rows]
    values = np.asarray([[_float(row, "balanced_risk", 0.0)] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(4.5, max(3, 0.22 * len(labels))))
    image = ax.imshow(values, cmap="magma", aspect="auto")
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    ax.set_xticks([0], labels=["risk"])
    fig.colorbar(image, ax=ax, fraction=0.08)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_bwer_figures(summary_rows: Sequence[Mapping[str, object]], slice_rows: Sequence[Mapping[str, object]], output_dir: str | Path) -> dict[str, Path]:
    figures = ensure_dir(Path(output_dir) / "figures")
    paths = {
        "average_vs_bwer": figures / "average_vs_bwer.png",
        "bwer_by_model": figures / "bwer_by_model.png",
        "raw_vs_balanced_bwer": figures / "raw_vs_balanced_bwer.png",
        "worst_tail_slices": figures / "worst_tail_slices.png",
        "slice_risk_heatmap": figures / "slice_risk_heatmap.png",
    }
    plot_average_vs_bwer(summary_rows, paths["average_vs_bwer"])
    plot_bwer_by_model(summary_rows, paths["bwer_by_model"])
    plot_raw_vs_balanced_bwer(summary_rows, paths["raw_vs_balanced_bwer"])
    plot_worst_tail_slices(slice_rows, paths["worst_tail_slices"])
    plot_slice_risk_heatmap(slice_rows, paths["slice_risk_heatmap"])
    return paths
