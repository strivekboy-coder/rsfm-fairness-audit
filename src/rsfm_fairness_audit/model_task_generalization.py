from __future__ import annotations

"""Experiment 9 task-wise 2x2 model x task comparison.

Raw GeoBWER values are never averaged across tasks.  Cross-task interaction is
reported only after orienting metrics as risks and standardising within task.
"""

import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


SCHEMA = "geobwer.experiment9.model_x_task_generalization.v1"


class ModelTaskGeneralizationError(RuntimeError):
    pass


def _float(row: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    raise ModelTaskGeneralizationError(f"None of {keys} is finite in GeoBWER row.")


def _seed(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ModelTaskGeneralizationError(f"Cannot infer seed from {path}")


def _candidate_seed_dirs(root: Path) -> list[Path]:
    candidates = list(root.glob("seed_*")) + list((root / "probe_seeds").glob("seed_*"))
    return sorted({path.resolve() for path in candidates if path.is_dir()}, key=_seed)


def _one(seed_dir: Path, pattern: str, preferred_parent: tuple[str, ...]) -> Path:
    candidates = [path for path in seed_dir.rglob(pattern) if path.parent.name in preferred_parent]
    if len(candidates) != 1:
        raise ModelTaskGeneralizationError(f"Expected one {pattern} below {seed_dir}; found {candidates}")
    return candidates[0]


def summarize_cell(root: str | Path, *, model: str, task: str) -> list[dict[str, Any]]:
    root = Path(root)
    output = []
    for seed_dir in _candidate_seed_dirs(root):
        summary_path = _one(seed_dir, "geobwer_summary.csv", ("geobwer", "geobwer_raw"))
        country = next((row for row in read_csv_rows(summary_path) if str(row.get("axis")) == "country"), None)
        if country is None:
            raise ModelTaskGeneralizationError(f"No country axis in {summary_path}")
        audit_path = _one(seed_dir, "formal_audit_table.csv", ("formal_outputs",))
        audit = read_csv_rows(audit_path)
        risk = np.asarray([float(row["risk"]) for row in audit], dtype=float)
        if not len(risk) or not np.all(np.isfinite(risk)):
            raise ModelTaskGeneralizationError(f"Invalid formal risks in {audit_path}")
        sample_ids = sorted({str(row["sample_id"]) for row in audit})
        import hashlib
        sample_hash = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
        probability_path = audit_path.parent / "probabilities.npz"
        if not probability_path.is_file():
            raise ModelTaskGeneralizationError(f"Missing formal probability artifact: {probability_path}")
        with np.load(probability_path, allow_pickle=False) as formal:
            probability_sample_ids = [str(value) for value in formal["sample_id"]]
            targets = np.asarray(formal["targets"])
            class_names = [str(value) for value in formal["class_names"]]
        if sorted(probability_sample_ids) != sample_ids or targets.shape[0] != len(probability_sample_ids):
            raise ModelTaskGeneralizationError(f"Formal probability/target rows do not match {audit_path}")
        target_digest = hashlib.sha256()
        for sample_id, target in sorted(zip(probability_sample_ids, targets), key=lambda item: item[0]):
            target_digest.update(sample_id.encode())
            target_digest.update(np.asarray(target).tobytes(order="C"))
        target_digest.update("\n".join(class_names).encode())
        output.append({
            "model": model, "task": task, "seed": _seed(seed_dir),
            "primary_risk": float(np.mean(risk)),
            "primary_score": float(1.0 - np.mean(risk)),
            "M": _float(country, "mean_risk"), "T": _float(country, "tail_risk"),
            "D": _float(country, "bwer", "geobwer"),
            "evidence_status": country.get("evidence_status", ""),
            "risk_spec_signature": country.get("risk_spec_signature", ""),
            "sample_count": len(sample_ids), "sample_id_set_hash": sample_hash,
            "target_and_class_mapping_hash": target_digest.hexdigest(),
            "audit_table": str(audit_path), "geobwer_summary": str(summary_path),
        })
    if not output:
        raise ModelTaskGeneralizationError(f"No seed outputs found under {root}")
    return output


def analyze_model_task_matrix(
    cells: Mapping[tuple[str, str], str | Path], output_dir: str | Path,
    *, models: tuple[str, str] = ("dofav2", "terramind"), tasks: tuple[str, str] = ("fmow", "reben"),
    expected_seeds: tuple[int, ...] = (42, 73, 101),
) -> dict[str, Path]:
    expected = {(model, task) for model in models for task in tasks}
    if set(cells) != expected:
        raise ModelTaskGeneralizationError(f"Expected exactly four cells {sorted(expected)}; observed {sorted(cells)}")
    rows = [row for key, root in cells.items() for row in summarize_cell(root, model=key[0], task=key[1])]
    contracts = []
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        sample_hashes = {row["sample_id_set_hash"] for row in task_rows}
        target_hashes = {row["target_and_class_mapping_hash"] for row in task_rows}
        risk_specs = {row["risk_spec_signature"] for row in task_rows}
        seeds_by_model = {model: sorted(int(row["seed"]) for row in task_rows if row["model"] == model) for model in models}
        contract = {"task": task, "same_test_sample_support": len(sample_hashes) == 1,
                    "same_targets_and_class_mapping": len(target_hashes) == 1,
                    "same_risk_spec": len(risk_specs) == 1 and "" not in risk_specs,
                    "same_seeds": len({tuple(value) for value in seeds_by_model.values()}) == 1,
                    "expected_seeds_present": all(value == sorted(expected_seeds) for value in seeds_by_model.values()),
                    "seeds_by_model": seeds_by_model, "sample_hashes": sorted(sample_hashes),
                    "target_hashes": sorted(target_hashes), "risk_specs": sorted(risk_specs)}
        contract["valid"] = bool(contract["same_test_sample_support"] and contract["same_targets_and_class_mapping"]
                                 and contract["same_risk_spec"] and contract["same_seeds"] and contract["expected_seeds_present"])
        contracts.append(contract)
    if not all(item["valid"] for item in contracts):
        raise ModelTaskGeneralizationError(f"Within-task comparison contract failed: {contracts}")
    ranks = []
    for task in tasks:
        for metric in ("primary_risk", "M", "T", "D"):
            means = {model: float(np.mean([float(row[metric]) for row in rows if row["task"] == task and row["model"] == model])) for model in models}
            ordered = sorted(means, key=means.get)
            for rank, model in enumerate(ordered, start=1):
                ranks.append({"task": task, "metric": metric, "model": model, "mean": means[model], "rank": rank,
                              "lower_is_better": True})
    effects = []
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        common_seeds = sorted({int(row["seed"]) for row in task_rows})
        for metric in ("primary_risk", "M", "T", "D"):
            values = np.asarray([float(row[metric]) for row in task_rows], dtype=float)
            scale = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            deltas = []
            for seed in common_seeds:
                by_model = {row["model"]: float(row[metric]) for row in task_rows if int(row["seed"]) == seed}
                deltas.append(by_model[models[0]] - by_model[models[1]])
            raw_delta = float(np.mean(deltas))
            effects.append({"task": task, "metric": metric, "contrast": f"{models[0]}_minus_{models[1]}",
                            "raw_within_task_delta": raw_delta, "within_task_scale": scale,
                            "standardized_effect": raw_delta / scale if scale > 1e-12 else float("nan"),
                            "positive_means_first_model_worse": True})
    interactions = []
    for metric in ("primary_risk", "M", "T", "D"):
        by_task = {row["task"]: float(row["standardized_effect"]) for row in effects if row["metric"] == metric}
        interactions.append({"metric": metric, "interaction": f"effect_{tasks[0]}_minus_effect_{tasks[1]}",
                             "standardized_difference_in_differences": by_task[tasks[0]] - by_task[tasks[1]],
                             "descriptive_not_formal_causal_interaction": True})
    consistency = []
    for task in tasks:
        for model in models:
            means = {metric: float(np.mean([float(row[metric]) for row in rows if row["task"] == task and row["model"] == model])) for metric in ("M", "T", "D")}
            opponent = {metric: float(np.mean([float(row[metric]) for row in rows if row["task"] == task and row["model"] != model])) for metric in ("M", "T", "D")}
            signs = {metric: int(np.sign(means[metric] - opponent[metric])) for metric in means}
            consistency.append({"task": task, "model": model, **{f"delta_{key}": means[key] - opponent[key] for key in means},
                                "mean_tail_consistency": signs["M"] == signs["T"],
                                "mean_gap_consistency": signs["M"] == signs["D"],
                                "levelling_down_signature": signs["M"] > 0 and signs["D"] < 0})
    output = ensure_dir(output_dir)
    paths = {"cells": output / "experiment9_cell_seed_metrics.csv", "ranks": output / "experiment9_within_task_ranks.csv",
             "effects": output / "experiment9_standardized_effects.csv", "interaction": output / "experiment9_model_task_interaction.csv",
             "consistency": output / "experiment9_mean_tail_consistency.csv", "manifest": output / "experiment9_manifest.json"}
    write_csv(paths["cells"], rows); write_csv(paths["ranks"], ranks); write_csv(paths["effects"], effects)
    write_csv(paths["interaction"], interactions); write_csv(paths["consistency"], consistency)
    paths["manifest"].write_text(json.dumps({"schema": SCHEMA, "status": "complete", "contracts": contracts,
                                              "raw_geobwer_averaged_across_tasks": False,
                                              "comparison_estimand": "model_pipeline_under_common_task_sample_label_riskspec_contract",
                                              "backbone_only_causal_claim": False,
                                              "standardization": "pooled_seed_scale_within_each_task_only",
                                              "interaction_status": "descriptive"}, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


__all__ = ["ModelTaskGeneralizationError", "analyze_model_task_matrix", "summarize_cell"]
