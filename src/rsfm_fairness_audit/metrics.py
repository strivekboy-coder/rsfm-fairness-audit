from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GroupMetric:
    slice_name: str
    group: str
    n: int
    accuracy: float
    global_accuracy: float
    drop_from_global: float
    fairness_risk_score: float


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    return float(np.mean(labels == predictions))


def group_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    metadata: Sequence[dict],
    slice_name: str,
) -> list[GroupMetric]:
    global_accuracy = accuracy(labels, predictions)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[str(row[slice_name])].append(index)

    rows: list[GroupMetric] = []
    for group, indices in sorted(groups.items()):
        idx = np.asarray(indices, dtype=np.int64)
        group_acc = accuracy(labels[idx], predictions[idx])
        drop = global_accuracy - group_acc
        rows.append(
            GroupMetric(
                slice_name=slice_name,
                group=group,
                n=len(indices),
                accuracy=group_acc,
                global_accuracy=global_accuracy,
                drop_from_global=drop,
                fairness_risk_score=max(0.0, drop),
            )
        )
    return rows


def summarize_gap(rows: Sequence[GroupMetric], gap_name: str) -> dict[str, float | str | int]:
    accuracies = np.asarray([row.accuracy for row in rows], dtype=float)
    counts = np.asarray([row.n for row in rows], dtype=float)
    if len(rows) == 0:
        return {"gap_name": gap_name, "num_groups": 0}
    worst_idx = int(np.argmin(accuracies))
    best_idx = int(np.argmax(accuracies))
    return {
        "gap_name": gap_name,
        "num_groups": len(rows),
        "average_performance": float(np.average(accuracies, weights=counts)),
        "balanced_average_performance": float(np.mean(accuracies)),
        "worst_group": rows[worst_idx].group,
        "worst_region_performance": float(accuracies[worst_idx]),
        "best_group": rows[best_idx].group,
        "best_region_performance": float(accuracies[best_idx]),
        "best_worst_gap": float(accuracies[best_idx] - accuracies[worst_idx]),
        "group_standard_deviation": float(np.std(accuracies)),
        "max_drop_from_global": float(max(row.drop_from_global for row in rows)),
        "fairness_risk_score": float(np.mean([row.fairness_risk_score for row in rows])),
    }


def raw_vs_balanced_gap(
    raw_rows: Sequence[GroupMetric],
    balanced_rows: Sequence[GroupMetric],
    slice_name: str,
) -> dict[str, float | str | int]:
    raw = summarize_gap(raw_rows, f"raw_{slice_name}_gap")
    balanced = summarize_gap(balanced_rows, f"balanced_{slice_name}_gap")
    raw_gap = float(raw["best_worst_gap"])
    balanced_gap = float(balanced["best_worst_gap"])
    return {
        "slice_name": slice_name,
        "raw_fairness_gap": raw_gap,
        "balanced_fairness_gap": balanced_gap,
        "residual_gap_after_balancing": balanced_gap,
        "gap_reduction_after_balancing": raw_gap - balanced_gap,
        "raw_worst_group": raw["worst_group"],
        "balanced_worst_group": balanced["worst_group"],
    }
