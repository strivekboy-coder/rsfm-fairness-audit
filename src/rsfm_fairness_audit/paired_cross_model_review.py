from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rsfm_fairness_audit.io import ensure_dir, write_csv


SCHEMA = "geobwer.paired_cross_model_review.v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"None of {names!r} occur in row")


def _find(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} below {root}; found {len(matches)}")
    return matches[0]


def read_paired_result(model: str, root: str | Path) -> dict[str, Any]:
    root = Path(root)
    audit_path = _find(root, "paired_shift_result_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if str(audit.get("status", "")).lower() != "pass":
        raise ValueError(f"{model} paired audit does not pass: {audit_path}")
    deltas = _read_csv(_find(root, "paired_shift_delta_seed_panel.csv"))
    seed_diagnostics = _read_csv(_find(root, "paired_probability_seed_summary.csv"))
    label_diagnostics = _read_csv(_find(root, "paired_probability_label_summary.csv"))
    seeds = {str(row["seed"]) for row in deltas}
    if seeds != {str(row["seed"]) for row in seed_diagnostics}:
        raise ValueError(f"{model} seed mismatch between shift and probability diagnostics")
    labels = {str(row["class_label"]) for row in label_diagnostics}
    metrics = {
        "delta_mean_risk": np.mean([_number(r, "delta_mean_risk") for r in deltas]),
        "delta_tail_risk": np.mean([_number(r, "delta_tail_risk") for r in deltas]),
        "delta_geobwer": np.mean([_number(r, "delta_geobwer") for r in deltas]),
        "tail_acceleration": np.mean([_number(r, "tail_acceleration") for r in deltas]),
        "delta_auroc": np.mean([_number(r, "mean_delta_auroc") for r in seed_diagnostics]),
        "delta_ap": np.mean([_number(r, "mean_delta_ap", "delta_macro_ap") for r in seed_diagnostics]),
        "delta_f1": np.mean([_number(r, "mean_delta_f1_at_locked_threshold", "delta_macro_f1") for r in seed_diagnostics]),
        "probability_wasserstein_1": np.mean([_number(r, "mean_probability_wasserstein_1") for r in seed_diagnostics]),
        "threshold_crossing_rate": np.mean([_number(r, "mean_threshold_crossing_rate") for r in seed_diagnostics]),
        "mean_absolute_paired_probability_shift": np.mean([_number(r, "mean_mean_absolute_paired_probability_shift", "mean_absolute_paired_probability_shift") for r in seed_diagnostics]),
    }
    signatures = Counter(str(row["modal_diagnostic_signature"]) for row in label_diagnostics)
    return {
        "model": model, "root": str(root), "audit": audit, "seeds": sorted(seeds),
        "labels": labels, "metrics": {key: float(value) for key, value in metrics.items()},
        "signatures": dict(signatures), "delta_rows": deltas,
        "seed_diagnostics": seed_diagnostics, "label_diagnostics": label_diagnostics,
    }


def compare_paired_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) < 2:
        raise ValueError("Cross-model review needs at least two audited paired results")
    reference_seeds = results[0]["seeds"]
    reference_labels = results[0]["labels"]
    if any(item["seeds"] != reference_seeds for item in results[1:]):
        raise ValueError("Models do not share the same seed labels")
    if any(item["labels"] != reference_labels for item in results[1:]):
        raise ValueError("Models do not share the same label universe")
    summaries = []
    for item in results:
        metrics = item["metrics"]
        summaries.append({
            "model": item["model"], "seed_count": len(item["seeds"]),
            "label_count": len(item["labels"]), **metrics,
            "collapse_label_count": item["signatures"].get("representation_collapse_signature", 0),
            "mixed_label_count": item["signatures"].get("mixed_or_partial_degradation", 0),
            "threshold_only_label_count": item["signatures"].get("threshold_shift_dominant", 0),
            "stable_label_count": item["signatures"].get("stable", 0),
        })
    label_rows: list[dict[str, Any]] = []
    by_model = {
        item["model"]: {str(row["class_label"]): row for row in item["label_diagnostics"]}
        for item in results
    }
    for label in sorted(reference_labels):
        row: dict[str, Any] = {"class_label": label}
        ppr_signs = []
        signatures = []
        for item in results:
            model = item["model"]
            source = by_model[model][label]
            prefix = model.lower().replace(" ", "_")
            ppr = _number(source, "mean_delta_predicted_positive_rate")
            row.update({
                f"{prefix}_delta_auroc": _number(source, "mean_delta_auroc"),
                f"{prefix}_delta_ap": _number(source, "mean_delta_ap"),
                f"{prefix}_delta_ppr": ppr,
                f"{prefix}_crossing": _number(source, "mean_threshold_crossing_rate"),
                f"{prefix}_signature": source["modal_diagnostic_signature"],
            })
            ppr_signs.append(int(math.copysign(1, ppr)) if abs(ppr) > 1e-12 else 0)
            signatures.append(str(source["modal_diagnostic_signature"]))
        row["opposite_score_transport_direction"] = len(set(ppr_signs)) > 1
        row["different_modal_signature"] = len(set(signatures)) > 1
        label_rows.append(row)
    opposite = sum(bool(row["opposite_score_transport_direction"]) for row in label_rows)
    signature_difference = sum(bool(row["different_modal_signature"]) for row in label_rows)
    geobwer_span = max(row["delta_geobwer"] for row in summaries) - min(row["delta_geobwer"] for row in summaries)
    support = bool(opposite > 0 and signature_difference > 0 and geobwer_span >= 0.02)
    return {
        "schema": SCHEMA,
        "status": "pass",
        "same_seed_labels": True,
        "same_label_universe": True,
        "model_summaries": summaries,
        "label_comparison": label_rows,
        "claim_assessment": {
            "claim": "The same paired S2-to-S1 shift has different frozen-head failure geometry across models.",
            "supported": support,
            "opposite_score_transport_label_count": opposite,
            "different_modal_signature_label_count": signature_difference,
            "delta_geobwer_between_model_span": geobwer_span,
            "scope": "operational frozen-head diagnostic; not causal encoder attribution",
        },
    }


def _plot(comparison: Mapping[str, Any], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    summaries = comparison["model_summaries"]
    models = [row["model"] for row in summaries]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"][:len(models)]
    metrics = ["delta_mean_risk", "delta_tail_risk", "delta_geobwer", "tail_acceleration", "delta_auroc", "delta_ap"]
    labels = ["Δ mean risk", "Δ tail risk", "Δ GeoBWER", "Tail acceleration", "Δ AUROC", "Δ AP"]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6))
    for ax, metric, label in zip(axes.flat, metrics, labels):
        values = [float(row[metric]) for row in summaries]
        ax.bar(models, values, color=colors, edgecolor="#222222", linewidth=.35)
        ax.axhline(0, color="#333333", linewidth=.7)
        ax.set_title(label)
        ax.grid(axis="y", alpha=.18)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Same paired sensor shift, different failure geometry", fontweight="bold")
    fig.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"paired_cross_model_metric_comparison.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    label_rows = comparison["label_comparison"]
    fig, ax = plt.subplots(figsize=(8.8, 6.1))
    matrix = np.asarray([[float(row[f"{m.lower().replace(' ', '_')}_delta_ppr"]) for m in models] for row in label_rows])
    bound = max(.05, float(np.max(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="PuOr_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(models)), models)
    ax.set_yticks(range(len(label_rows)), [row["class_label"] for row in label_rows])
    ax.set_title("Label score-transport direction (Δ predicted-positive rate)", fontweight="bold")
    fig.colorbar(image, ax=ax, label="S1 OOD − S2 ID")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"paired_cross_model_label_transport.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def build_cross_model_review(model_roots: Mapping[str, str | Path], output_dir: str | Path) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    comparison = compare_paired_results([read_paired_result(model, root) for model, root in model_roots.items()])
    write_csv(output / "paired_cross_model_summary.csv", comparison["model_summaries"])
    write_csv(output / "paired_cross_model_label_comparison.csv", comparison["label_comparison"])
    figures = _plot(comparison, output)
    comparison["artifacts"] = [str(path.name) for path in figures]
    (output / "paired_cross_model_evidence.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    return comparison
