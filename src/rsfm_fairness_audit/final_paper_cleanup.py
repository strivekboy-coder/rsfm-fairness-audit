"""Paper-facing derivations from frozen GeoBWER artifacts.

This module never writes to or amends canonical experiment outputs.  It only
builds compact, auditable tables and figures below a caller-provided directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SEEDS = (42, 73, 101)
METRICS = ("primary_risk", "M", "T", "D")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized and not fields:
        raise ValueError(f"Cannot infer columns for empty table: {path}")
    columns = fields or list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def build_experiment9_tables(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {(r["task"], r["model"], int(r["seed"])): r for r in rows}
    expected = {(task, model, seed) for task in ("fmow", "reben") for model in ("dofav2", "terramind") for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"Experiment 9 cells differ from frozen 2x2x3 design: {sorted(set(indexed) ^ expected)}")

    per_seed: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task in ("fmow", "reben"):
        metric_deltas: dict[str, list[float]] = defaultdict(list)
        for seed in SEEDS:
            left, right = indexed[(task, "dofav2", seed)], indexed[(task, "terramind", seed)]
            for metric in METRICS:
                delta = float(left[metric]) - float(right[metric])
                metric_deltas[metric].append(delta)
                per_seed.append(
                    {
                        "task": task,
                        "seed": seed,
                        "metric": metric,
                        "dofav2": float(left[metric]),
                        "terramind": float(right[metric]),
                        "delta_dofav2_minus_terramind": delta,
                        "direction": "DOFAv2_higher" if delta > 0 else "TerraMind_higher" if delta < 0 else "tie",
                        "comparison_scope": "within_task_same_seed_same_support",
                        "evidence_status": "multi_seed_descriptive_paired",
                    }
                )
        for metric, values in metric_deltas.items():
            mean, sd = _mean_sd(values)
            signs = sum(v > 0 for v in values) - sum(v < 0 for v in values)
            summaries.append(
                {
                    "task": task,
                    "metric": metric,
                    "mean_delta_dofav2_minus_terramind": mean,
                    "sd": sd,
                    "min": min(values),
                    "max": max(values),
                    "positive_seed_count": sum(v > 0 for v in values),
                    "negative_seed_count": sum(v < 0 for v in values),
                    "direction_consistency": f"{abs(signs)}/3" if abs(signs) == 3 else "mixed",
                    "interpretation": (
                        "higher risk/burden for DOFAv2" if mean > 0 else "higher risk/burden for TerraMind"
                    ),
                    "claim_scope": "task-internal descriptive contrast; not a cross-task raw average",
                }
            )
    return per_seed, summaries


def build_experiment8_tables(stage_rows: list[dict[str, str]], recovery_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_map = [
        ("A_ID", "A", "test_id"),
        ("A_shifted", "A", "test_shifted"),
        ("B_threshold", "B", "test"),
        ("C_head", "C", "test"),
    ]
    output: list[dict[str, Any]] = []
    for paper_stage, stage, split in stage_map:
        matching = [r for r in stage_rows if r["stage"] == stage and r["split_role"] == split]
        if sorted(int(r["seed"]) for r in matching) != list(SEEDS):
            raise ValueError(f"Incomplete Experiment 8 stage {paper_stage}")
        for metric, source in (("AUROC", "macro_auroc"), ("M", "mean_risk"), ("T", "tail_risk_beta_0_10"), ("D", "geobwer_beta_0_10")):
            values = [float(r[source]) for r in matching]
            mean, sd = _mean_sd(values)
            output.append(
                {
                    "paper_stage": paper_stage,
                    "stage": stage,
                    "split": split,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "min": min(values),
                    "max": max(values),
                    "n_seeds": len(values),
                    "scientific_role": {
                        "A_ID": "in-distribution reference",
                        "A_shifted": "unchanged S2 head evaluated under S1 shift",
                        "B_threshold": "S1-validation threshold-only recalibration",
                        "C_head": "frozen encoder plus S1-trained task head",
                    }[paper_stage],
                }
            )

    rec_out: list[dict[str, Any]] = []
    metric_names = {"macro_auroc": "AUROC", "mean_risk": "M", "tail_risk_beta_0_10": "T", "geobwer_beta_0_10": "D"}
    for stage in ("B", "C"):
        for source, paper_metric in metric_names.items():
            matching = [r for r in recovery_rows if r["split_role"] == "test" and r["stage"] == stage and r["metric"] == source]
            if sorted(int(r["seed"]) for r in matching) != list(SEEDS):
                raise ValueError(f"Incomplete Experiment 8 recovery {stage}/{source}")
            values = [float(r["recovery"]) for r in matching]
            mean, sd = _mean_sd(values)
            rec_out.append(
                {
                    "stage": stage,
                    "metric": paper_metric,
                    "mean_recovery": mean,
                    "sd": sd,
                    "min": min(values),
                    "max": max(values),
                    "positive_seed_count": sum(v > 0 for v in values),
                    "n_seeds": 3,
                    "selection_role": "reporting_only_test_evaluation; gate_was_validation_only",
                }
            )
    return output, rec_out


def build_reben_example_tables(
    burden_rows: list[dict[str, str]], terra_labels: list[dict[str, str]], croma_labels: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    terra = {r["class_label"]: r for r in terra_labels}
    croma = {r["class_label"]: r for r in croma_labels}
    low_support_labels = {"Beaches, dunes, sands", "Coastal wetlands"}
    universe: list[dict[str, Any]] = []
    for row in burden_rows:
        label = row["class_label"]
        support = int(row["minimum_cell_support"])
        positive = int(row["positive_seed_count"])
        eligible = int(row["seed_count"]) == 3 and support >= 1000 and label not in low_support_labels and positive in (0, 3)
        tr = terra[label]
        cr = croma[label]
        universe.append(
            {
                **row,
                "minimum_cell_support": support,
                "eligible": eligible,
                "eligibility_rule": "3 seeds; stable sign (0/3 or 3/3 positive); country support>=1000; excludes pre-flagged low-positive-support labels",
                "terramind_label_delta_auroc": float(tr["mean_delta_auroc"]),
                "croma_label_delta_auroc": float(cr["mean_delta_auroc"]),
                "terramind_label_delta_ap": float(tr["mean_delta_ap"]),
                "croma_label_delta_ap": float(cr["mean_delta_ap"]),
                "terramind_label_delta_ppr": float(tr["mean_delta_predicted_positive_rate"]),
                "croma_label_delta_ppr": float(cr["mean_delta_predicted_positive_rate"]),
                "ppr_geometry_gap": abs(float(tr["mean_delta_predicted_positive_rate"]) - float(cr["mean_delta_predicted_positive_rate"])),
            }
        )

    eligible = [r for r in universe if r["eligible"]]

    def choose(label_filter, key, reverse=True):
        pool = [r for r in eligible if label_filter(r)]
        if not pool:
            raise ValueError("A predeclared reBEN example archetype has no eligible candidate")
        return sorted(pool, key=key, reverse=reverse)[0]

    chosen: list[tuple[str, dict[str, Any], str]] = []
    chosen.append((
        "shared_high_burden",
        choose(lambda r: r["positive_seed_count"] == "3", lambda r: (float(r["mean_delta_risk"]), r["minimum_cell_support"])),
        "largest stable TerraMind country×label risk increase among eligible cells",
    ))
    chosen.append((
        "model_specific_overprediction_geometry",
        choose(lambda r: r["positive_seed_count"] == "3", lambda r: (r["ppr_geometry_gap"], r["minimum_cell_support"])),
        "largest matched TerraMind-vs-CROMA label-level predicted-positive-rate geometry gap",
    ))
    chosen.append((
        "opposite_score_transport",
        choose(lambda r: r["terramind_label_delta_ppr"] < 0 < r["croma_label_delta_ppr"], lambda r: (r["ppr_geometry_gap"], r["minimum_cell_support"])),
        "largest label-level PPR gap with TerraMind decreasing and CROMA increasing predicted-positive rate",
    ))
    chosen.append((
        "localized_risk_reduction",
        choose(lambda r: r["positive_seed_count"] == "0", lambda r: (float(r["mean_delta_risk"]), -r["minimum_cell_support"]), reverse=False),
        "most negative stable TerraMind country×label risk change among eligible cells",
    ))

    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for archetype, row, rule in chosen:
        cell = (row["country"], row["class_label"])
        if cell in used:
            continue
        used.add(cell)
        selected.append({"archetype": archetype, "selection_rule": rule, **row})
    return universe, selected


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def clean(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(clean(row.get(c, "")) for c in columns) + " |" for row in rows]
    return "\n".join([header, rule, *body]) + "\n"


def ensure_finite(rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    for row in rows:
        for field in fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {field}: {row}")
