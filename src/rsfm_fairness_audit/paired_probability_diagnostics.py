from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.io import ensure_dir, write_csv


SCHEMA = "geobwer.paired_probability_diagnostics.v1"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def binary_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(targets, dtype=np.int8)
    score = np.asarray(scores, dtype=float)
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    if positive == 0 or negative == 0:
        return float("nan")
    rank_sum = float(np.sum(_average_ranks(score)[y == 1]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def binary_average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(targets, dtype=np.int8)
    positive = int(np.sum(y == 1))
    if positive == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positive)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 1e-12 else float("nan")


def _binary_confusion(target: np.ndarray, prediction: np.ndarray) -> dict[str, int | float]:
    tp = int(np.sum((target == 1) & prediction))
    fp = int(np.sum((target == 0) & prediction))
    fn = int(np.sum((target == 1) & (~prediction)))
    tn = int(np.sum((target == 0) & (~prediction)))
    denominator = (2 * tp) + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": (2 * tp / denominator) if denominator else 0.0}


def paired_label_probability_diagnostics(
    targets: np.ndarray,
    id_probabilities: np.ndarray,
    ood_probabilities: np.ndarray,
    thresholds: np.ndarray,
    label_names: Sequence[str],
    *,
    seed: int | str,
) -> list[dict[str, Any]]:
    """Describe paired score/rank and threshold effects without fitting anything.

    The diagnostic categories are operational signatures. In particular,
    ``representation_collapse_signature`` is not a causal attribution to the
    encoder because this function observes only frozen-head probabilities.
    """
    y = np.asarray(targets, dtype=np.int8)
    id_p = np.asarray(id_probabilities, dtype=float)
    ood_p = np.asarray(ood_probabilities, dtype=float)
    cutoffs = np.asarray(thresholds, dtype=float).reshape(-1)
    if y.shape != id_p.shape or y.shape != ood_p.shape:
        raise ValueError(f"Paired targets/probability shapes differ: {y.shape}, {id_p.shape}, {ood_p.shape}")
    if y.ndim != 2 or y.shape[1] != len(cutoffs) or y.shape[1] != len(label_names):
        raise ValueError("Label names and locked thresholds must match the probability columns.")
    if not np.all(np.isin(y, (0, 1))) or not np.all(np.isfinite(id_p)) or not np.all(np.isfinite(ood_p)):
        raise ValueError("Diagnostics require finite probabilities and binary targets.")
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(label_names):
        target = y[:, index]
        id_score = id_p[:, index]
        ood_score = ood_p[:, index]
        threshold = float(cutoffs[index])
        id_pred = id_score >= threshold
        ood_pred = ood_score >= threshold
        id_confusion = _binary_confusion(target, id_pred)
        ood_confusion = _binary_confusion(target, ood_pred)
        positive = target == 1
        negative = ~positive
        id_auroc = binary_auroc(target, id_score)
        ood_auroc = binary_auroc(target, ood_score)
        id_ap = binary_average_precision(target, id_score)
        ood_ap = binary_average_precision(target, ood_score)
        id_separation = float(np.mean(id_score[positive]) - np.mean(id_score[negative])) if positive.any() and negative.any() else float("nan")
        ood_separation = float(np.mean(ood_score[positive]) - np.mean(ood_score[negative])) if positive.any() and negative.any() else float("nan")
        delta_auroc = ood_auroc - id_auroc
        delta_ap = ood_ap - id_ap
        delta_separation = ood_separation - id_separation
        crossing = id_pred != ood_pred
        crossing_rate = float(np.mean(crossing))
        delta_positive_rate = float(np.mean(ood_pred) - np.mean(id_pred))
        score_std_ratio = _safe_ratio(float(np.std(ood_score)), float(np.std(id_score)))
        rank_degradation = bool(
            (math.isfinite(delta_auroc) and delta_auroc <= -0.05)
            or (math.isfinite(delta_ap) and delta_ap <= -0.05)
        )
        separation_degradation = bool(math.isfinite(delta_separation) and delta_separation <= -0.05)
        score_contraction = bool(math.isfinite(score_std_ratio) and score_std_ratio <= 0.75)
        threshold_instability = bool(crossing_rate >= 0.10 or abs(delta_positive_rate) >= 0.10)
        collapse_signature = rank_degradation and separation_degradation and score_contraction
        if collapse_signature:
            diagnosis = "representation_collapse_signature"
        elif not rank_degradation and threshold_instability:
            diagnosis = "threshold_shift_dominant"
        elif rank_degradation or separation_degradation or threshold_instability:
            diagnosis = "mixed_or_partial_degradation"
        else:
            diagnosis = "stable"
        rows.append({
            "seed": seed, "class_index": index, "class_label": str(label),
            "sample_count": len(target), "positive_support": int(np.sum(positive)),
            "negative_support": int(np.sum(negative)), "locked_threshold": threshold,
            "id_auroc": id_auroc, "ood_auroc": ood_auroc, "delta_auroc": delta_auroc,
            "id_ap": id_ap, "ood_ap": ood_ap, "delta_ap": delta_ap,
            "id_predicted_positive_rate": float(np.mean(id_pred)),
            "ood_predicted_positive_rate": float(np.mean(ood_pred)),
            "delta_predicted_positive_rate": delta_positive_rate,
            "id_f1_at_locked_threshold": id_confusion["f1"],
            "ood_f1_at_locked_threshold": ood_confusion["f1"],
            "delta_f1_at_locked_threshold": float(ood_confusion["f1"]) - float(id_confusion["f1"]),
            **{f"id_{key}": value for key, value in id_confusion.items() if key != "f1"},
            **{f"ood_{key}": value for key, value in ood_confusion.items() if key != "f1"},
            "ood_all_negative_prediction_flag": bool(not np.any(ood_pred)),
            "ood_all_positive_prediction_flag": bool(np.all(ood_pred)),
            "id_mean_probability": float(np.mean(id_score)),
            "ood_mean_probability": float(np.mean(ood_score)),
            "delta_mean_probability": float(np.mean(ood_score - id_score)),
            "id_mean_threshold_margin": float(np.mean(id_score - threshold)),
            "ood_mean_threshold_margin": float(np.mean(ood_score - threshold)),
            "delta_mean_threshold_margin": float(np.mean(ood_score - id_score)),
            "mean_absolute_paired_probability_shift": float(np.mean(np.abs(ood_score - id_score))),
            "probability_wasserstein_1": float(np.mean(np.abs(np.sort(ood_score) - np.sort(id_score)))),
            "id_score_std": float(np.std(id_score)), "ood_score_std": float(np.std(ood_score)),
            "ood_to_id_score_std_ratio": score_std_ratio,
            "id_score_separation": id_separation, "ood_score_separation": ood_separation,
            "delta_score_separation": delta_separation,
            "threshold_crossing_rate": crossing_rate,
            "negative_to_positive_crossing_rate": float(np.mean((~id_pred) & ood_pred)),
            "positive_to_negative_crossing_rate": float(np.mean(id_pred & (~ood_pred))),
            "positive_target_mean_probability_shift": float(np.mean(ood_score[positive] - id_score[positive])) if positive.any() else float("nan"),
            "negative_target_mean_probability_shift": float(np.mean(ood_score[negative] - id_score[negative])) if negative.any() else float("nan"),
            "rank_degradation_flag": rank_degradation,
            "threshold_instability_flag": threshold_instability,
            "score_separation_degradation_flag": separation_degradation,
            "score_contraction_flag": score_contraction,
            "diagnostic_signature": diagnosis,
            "diagnostic_is_causal_attribution": False,
        })
    return rows


def _mean(rows: Sequence[Mapping[str, Any]], column: str) -> float:
    values = np.asarray([float(row[column]) for row in rows], dtype=float)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def aggregate_paired_probability_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_label: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[str(row["seed"])].append(row)
        by_label[(int(row["class_index"]), str(row["class_label"]))].append(row)
    metrics = (
        "id_auroc", "ood_auroc", "delta_auroc", "id_ap", "ood_ap", "delta_ap",
        "delta_predicted_positive_rate", "delta_mean_probability",
        "mean_absolute_paired_probability_shift", "probability_wasserstein_1",
        "threshold_crossing_rate", "delta_score_separation",
        "id_f1_at_locked_threshold", "ood_f1_at_locked_threshold", "delta_f1_at_locked_threshold",
    )
    seed_rows = []
    for seed, group in sorted(by_seed.items(), key=lambda pair: int(pair[0])):
        counts = Counter(str(row["diagnostic_signature"]) for row in group)
        seed_rows.append({
            "seed": int(seed), "label_count": len(group),
            "finite_auroc_label_count": sum(math.isfinite(float(row["id_auroc"])) and math.isfinite(float(row["ood_auroc"])) for row in group),
            **{f"mean_{metric}": _mean(group, metric) for metric in metrics},
            **{f"{name}_label_count": counts.get(name, 0) for name in (
                "representation_collapse_signature", "threshold_shift_dominant",
                "mixed_or_partial_degradation", "stable",
            )},
        })
    label_rows = []
    for (index, label), group in sorted(by_label.items()):
        modes = Counter(str(row["diagnostic_signature"]) for row in group)
        label_rows.append({
            "class_index": index, "class_label": label, "seed_count": len(group),
            **{f"mean_{metric}": _mean(group, metric) for metric in metrics},
            "modal_diagnostic_signature": modes.most_common(1)[0][0],
            "representation_collapse_signature_seed_count": modes.get("representation_collapse_signature", 0),
            "threshold_shift_dominant_seed_count": modes.get("threshold_shift_dominant", 0),
        })
    return seed_rows, label_rows


def plot_paired_probability_diagnostics(
    label_summary: Sequence[Mapping[str, Any]], output_dir: str | Path
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42})
    output = ensure_dir(output_dir)
    colors = {"stable": "#009E73", "threshold_shift_dominant": "#E69F00",
              "mixed_or_partial_degradation": "#0072B2", "representation_collapse_signature": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    panels = (
        ("mean_id_auroc", "mean_ood_auroc", "AUROC"),
        ("mean_id_ap", "mean_ood_ap", "Average precision"),
    )
    for panel, (xcol, ycol, label) in enumerate(panels):
        ax = axes.flat[panel]
        for row in label_summary:
            category = str(row["modal_diagnostic_signature"])
            ax.scatter(float(row[xcol]), float(row[ycol]), color=colors[category], marker="o", s=28, edgecolor="#222", linewidth=0.3)
        ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=0.8)
        ax.set(xlabel=f"S2 ID {label}", ylabel=f"S1 OOD {label}", xlim=(0, 1), ylim=(0, 1))
        ax.text(-0.13, 1.04, chr(ord("A") + panel), transform=ax.transAxes, fontweight="bold", fontsize=10)
    axes[1, 0].barh(
        [str(row["class_label"]) for row in label_summary],
        [float(row["mean_delta_predicted_positive_rate"]) for row in label_summary],
        color=[colors[str(row["modal_diagnostic_signature"])] for row in label_summary], edgecolor="#222", linewidth=0.3,
    )
    axes[1, 0].axvline(0, color="#222", linewidth=0.8)
    axes[1, 0].set_xlabel("Predicted-positive rate shift (S1−S2)")
    axes[1, 0].text(-0.13, 1.04, "C", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=10)
    axes[1, 1].barh(
        [str(row["class_label"]) for row in label_summary],
        [float(row["mean_threshold_crossing_rate"]) for row in label_summary],
        color=[colors[str(row["modal_diagnostic_signature"])] for row in label_summary], edgecolor="#222", linewidth=0.3,
    )
    axes[1, 1].set_xlabel("Paired threshold-crossing rate")
    axes[1, 1].text(-0.13, 1.04, "D", transform=axes[1, 1].transAxes, fontweight="bold", fontsize=10)
    for ax in axes.flat:
        ax.grid(alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=name.replace("_", " ")) for name, color in colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Paired S2 ID → S1 OOD probability diagnostics", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths = []
    for suffix in (".png", ".pdf"):
        path = output / f"paired_probability_diagnostics{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def write_paired_probability_diagnostics(
    seed_inputs: Sequence[Mapping[str, Any]], output_dir: str | Path
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    all_rows: list[dict[str, Any]] = []
    for item in seed_inputs:
        all_rows.extend(paired_label_probability_diagnostics(
            np.load(Path(item["targets"]), mmap_mode="r"),
            np.load(Path(item["id_probabilities"]), mmap_mode="r"),
            np.load(Path(item["ood_probabilities"]), mmap_mode="r"),
            np.asarray(item["thresholds"], dtype=float), item["label_names"], seed=item["seed"],
        ))
    seed_rows, label_rows = aggregate_paired_probability_diagnostics(all_rows)
    label_by_seed = output / "paired_shift_probability_diagnostics_by_seed_label.csv"
    seed_summary = output / "paired_shift_probability_diagnostics_seed_summary.csv"
    label_summary = output / "paired_shift_probability_diagnostics_label_summary.csv"
    write_csv(label_by_seed, all_rows)
    write_csv(seed_summary, seed_rows)
    write_csv(label_summary, label_rows)
    figures = plot_paired_probability_diagnostics(label_rows, ensure_dir(output / "figures"))
    return {"label_by_seed": label_by_seed, "seed_summary": seed_summary, "label_summary": label_summary, "figures": figures[0].parent}


__all__ = [
    "SCHEMA", "aggregate_paired_probability_diagnostics", "binary_average_precision",
    "binary_auroc", "paired_label_probability_diagnostics",
    "plot_paired_probability_diagnostics", "write_paired_probability_diagnostics",
]
