from __future__ import annotations

"""Experiment 9 task-wise 2x2 model x task comparison.

Raw GeoBWER values are never averaged across tasks.  Cross-task interaction is
reported only after orienting metrics as risks and standardising within task.
"""

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.risk_spec import RiskSpec


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


def _is_derived_geobwer_path(seed_dir: Path, path: Path) -> bool:
    relative_parts = path.relative_to(seed_dir).parts[:-1]
    return any(
        part == "uncertainty_extensions"
        or part.startswith("conformal_risk_control")
        or part.startswith("selective_")
        for part in relative_parts
    )


def _canonical_geobwer_summary(seed_dir: Path) -> Path:
    for relative in (
        Path("geobwer") / "geobwer_summary.csv",
        Path("geobwer_raw") / "geobwer_summary.csv",
    ):
        candidate = seed_dir / relative
        if candidate.is_file():
            return candidate

    candidates = sorted([
        path
        for path in seed_dir.rglob("geobwer_summary.csv")
        if path.parent.name in {"geobwer", "geobwer_raw"}
        and not _is_derived_geobwer_path(seed_dir, path)
    ])
    if len(candidates) != 1:
        raise ModelTaskGeneralizationError(
            f"Canonical geobwer/geobwer_summary.csv and geobwer_raw/geobwer_summary.csv are missing "
            f"below {seed_dir}; expected one non-derived legacy fallback, found {candidates}"
        )
    return candidates[0]


def _canonical_formal_artifact(seed_dir: Path, filename: str) -> Path:
    candidate = seed_dir / "formal_outputs" / filename
    if not candidate.is_file():
        raise ModelTaskGeneralizationError(f"Missing canonical formal artifact: {candidate}")
    return candidate


def _json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelTaskGeneralizationError(f"Cannot read canonical lineage JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ModelTaskGeneralizationError(f"Canonical lineage JSON must contain an object: {path}")
    return dict(payload)


def _semantic_field(
    name: str,
    sources: tuple[Mapping[str, Any], ...],
    *,
    seed_dir: Path,
    allow_empty: bool = False,
) -> Any:
    for source in sources:
        if name in source and source[name] is not None and (allow_empty or source[name] != ""):
            return source[name]
    raise ModelTaskGeneralizationError(
        f"Cannot verify estimand-relevant RiskSpec field {name!r} from canonical artifacts below {seed_dir}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _formal_risk_semantics(
    *,
    task: str,
    audit: list[dict[str, Any]],
    probability_sample_ids: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray | None,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registered = {
        "fmow": ("multiclass_classification", "multiclass", "risk", "zero_one_misclassification_loss"),
        "reben": ("multilabel_classification", "multilabel", "hamming_loss", "labelwise_hamming_loss"),
    }
    if task not in registered:
        raise ModelTaskGeneralizationError(f"No registered Experiment 9 risk semantics for task={task!r}")
    expected_formal_task, task_adapter, loss_name, definition = registered[task]
    observed_tasks = {str(row.get("task", "")).strip() for row in audit}
    if observed_tasks != {expected_formal_task}:
        raise ModelTaskGeneralizationError(
            f"Formal task semantics mismatch for Experiment 9 task={task}: observed={sorted(observed_tasks)}"
        )
    if (
        probabilities.ndim != 2
        or len(class_names) != probabilities.shape[1]
        or len(set(class_names)) != len(class_names)
        or not all(class_names)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ModelTaskGeneralizationError(f"Invalid canonical probability/class shape in {audit_path.parent}")

    if task_adapter == "multiclass":
        if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
            raise ModelTaskGeneralizationError("fMoW targets must be one integer class index per sample.")
        if np.any(targets < 0) or np.any(targets >= len(class_names)):
            raise ModelTaskGeneralizationError("fMoW targets contain an index outside the canonical class mapping.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4):
            raise ModelTaskGeneralizationError("fMoW canonical probability rows must sum to one.")
        expected_risk = (np.argmax(probabilities, axis=1) != targets).astype(float)
        target_semantics = "single_label_integer_class_index"
        formal_support = "binary_{0,1}"
        decision_rule = "argmax_over_canonical_class_mapping"
    else:
        if targets.shape != probabilities.shape or np.any((targets != 0) & (targets != 1)):
            raise ModelTaskGeneralizationError("reBEN targets must be binary multilabel vectors aligned to probabilities.")
        if thresholds is None:
            raise ModelTaskGeneralizationError(
                "Cannot verify legacy reBEN Hamming risk without canonical validation-locked thresholds."
            )
        threshold_values = np.asarray(thresholds, dtype=float)
        if threshold_values.ndim == 0:
            threshold_values = np.full(probabilities.shape[1], float(threshold_values))
        if threshold_values.shape != (probabilities.shape[1],) or np.any(threshold_values <= 0.0) or np.any(threshold_values >= 1.0):
            raise ModelTaskGeneralizationError("Canonical reBEN thresholds do not align with the label columns.")
        expected_risk = np.mean((probabilities >= threshold_values[None, :]) != targets, axis=1)
        target_semantics = "binary_multilabel_vector"
        formal_support = f"fractions_k_over_{probabilities.shape[1]}_for_k_0_to_{probabilities.shape[1]}"
        decision_rule = "validation_locked_per_label_thresholds"

    if len(probability_sample_ids) != len(expected_risk) or len(set(probability_sample_ids)) != len(expected_risk):
        raise ModelTaskGeneralizationError("Canonical probability sample IDs are empty or duplicated.")
    risk_by_sample: dict[str, float] = {}
    for row in audit:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in risk_by_sample:
            raise ModelTaskGeneralizationError(f"Formal audit sample IDs are empty or duplicated in {audit_path}")
        try:
            risk_by_sample[sample_id] = float(row["risk"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelTaskGeneralizationError(f"Formal risk is missing or invalid in {audit_path}") from exc
    if set(risk_by_sample) != set(probability_sample_ids):
        raise ModelTaskGeneralizationError(f"Formal risks and probabilities have different sample support in {audit_path}")
    observed_risk = np.asarray([risk_by_sample[sample_id] for sample_id in probability_sample_ids], dtype=float)
    if not np.all(np.isfinite(observed_risk)) or not np.allclose(observed_risk, expected_risk, atol=1e-7, rtol=0.0):
        raise ModelTaskGeneralizationError(
            f"Formal risk values do not implement registered {definition} semantics in {audit_path}"
        )

    class_mapping_hash = hashlib.sha256("\n".join(class_names).encode()).hexdigest()
    risk_definition = {
        "name": loss_name,
        "semantics": definition,
        "direction": "higher_is_worse",
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "unit": "independent_unit",
        "aggregation": "mean_within_slice",
        "formal_support": formal_support,
        "decision_rule": decision_rule,
        "verified_by_recomputation": True,
    }
    classification_targets = {
        "formal_task": expected_formal_task,
        "task_adapter": task_adapter,
        "target_semantics": target_semantics,
        "class_count": len(class_names),
        "class_mapping_hash": class_mapping_hash,
    }
    return risk_definition, classification_targets


def _semantic_risk_contract(
    *,
    task: str,
    seed_dir: Path,
    summary_path: Path,
    summary_row: Mapping[str, Any],
    audit_path: Path,
    audit: list[dict[str, Any]],
    probability_path: Path,
    probability_sample_ids: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray | None,
) -> dict[str, Any]:
    protocol_path = summary_path.parent / "geobwer_protocol.json"
    manifest_path = audit_path.parent / "formal_output_manifest.json"
    protocol = _json_mapping(protocol_path)
    manifest = _json_mapping(manifest_path)
    manifest_protocol = manifest.get("protocol", {})
    if manifest_protocol and not isinstance(manifest_protocol, Mapping):
        raise ModelTaskGeneralizationError(f"formal_output_manifest protocol must be an object: {manifest_path}")
    sources = (protocol, summary_row, dict(manifest_protocol))

    loss_name = str(_semantic_field("loss_name", sources, seed_dir=seed_dir))
    task_adapter = str(_semantic_field("task_adapter", sources, seed_dir=seed_dir))
    risk_definition, target_semantics = _formal_risk_semantics(
        task=task,
        audit=audit,
        probability_sample_ids=probability_sample_ids,
        probabilities=probabilities,
        targets=targets,
        class_names=class_names,
        thresholds=thresholds,
        audit_path=audit_path,
    )
    if loss_name != risk_definition["name"] or task_adapter != target_semantics["task_adapter"]:
        raise ModelTaskGeneralizationError(
            f"Canonical protocol loss/task adapter disagrees with verified formal risk semantics below {seed_dir}"
        )

    risk_spec_payload = protocol.get("risk_spec")
    if not isinstance(risk_spec_payload, Mapping):
        candidate = manifest_protocol.get("risk_spec") if isinstance(manifest_protocol, Mapping) else None
        risk_spec_payload = candidate if isinstance(candidate, Mapping) else None
    try:
        risk_spec = (
            RiskSpec.from_mapping(risk_spec_payload)
            if isinstance(risk_spec_payload, Mapping)
            else RiskSpec(name=loss_name, task_adapter=task_adapter)
        )
    except (TypeError, ValueError) as exc:
        raise ModelTaskGeneralizationError(f"Invalid canonical RiskSpec metadata below {seed_dir}") from exc
    if risk_spec.name != loss_name or risk_spec.task_adapter != task_adapter:
        raise ModelTaskGeneralizationError(f"RiskSpec disagrees with canonical loss/task semantics below {seed_dir}")

    explicit_signature = str(summary_row.get("risk_spec_signature", "")).strip()
    if explicit_signature and explicit_signature != risk_spec.signature:
        raise ModelTaskGeneralizationError(
            f"Explicit RiskSpec signature does not match canonical semantic fields below {seed_dir}"
        )
    verification_source = "explicit_signature" if explicit_signature else "legacy_reconstructed"

    axis = str(summary_row.get("axis", "")).strip()
    if not axis or any(axis not in row or str(row.get(axis, "")).strip() == "" for row in audit):
        raise ModelTaskGeneralizationError(f"Cannot verify slice axis {axis!r} from {audit_path}")
    all_slices = {str(row[axis]) for row in audit}
    excluded = {value for value in str(summary_row.get("excluded_groups", "")).split(";") if value}
    if not excluded.issubset(all_slices):
        raise ModelTaskGeneralizationError(f"GeoBWER excluded groups are absent from canonical audit axis={axis}")

    estimand_fields = {
        "beta": float(_semantic_field("beta", sources, seed_dir=seed_dir)),
        "deployment_weighting": str(_semantic_field("deployment_weighting", sources, seed_dir=seed_dir)),
        "audit_measure": str(_semantic_field("audit_measure", sources, seed_dir=seed_dir)),
        "partition_rule": str(_semantic_field("partition_rule", sources, seed_dir=seed_dir)),
        "missingness_rule": str(_semantic_field("missingness_rule", sources, seed_dir=seed_dir)),
        "standardization_target": str(_semantic_field("standardization_target", sources, seed_dir=seed_dir)),
        "standardization_weights": _semantic_field("standardization_weights", sources, seed_dir=seed_dir),
        "support_rule": str(_semantic_field("support_rule", sources, seed_dir=seed_dir)),
        "inference_target": str(_semantic_field("inference_target", sources, seed_dir=seed_dir)),
        "estimand_scope": str(_semantic_field("estimand_scope", sources, seed_dir=seed_dir)),
        "group_variable": str(_semantic_field("group_variable", sources, seed_dir=seed_dir)),
        "balance_variable": str(_semantic_field("balance_variable", sources, seed_dir=seed_dir, allow_empty=True)),
        "independent_unit_column": str(_semantic_field("independent_unit_column", sources, seed_dir=seed_dir)),
        "metric_version": str(_semantic_field("metric_version", sources, seed_dir=seed_dir)),
    }
    if not 0.0 < estimand_fields["beta"] <= 1.0:
        raise ModelTaskGeneralizationError(f"Invalid beta in canonical RiskSpec contract below {seed_dir}")

    semantic_contract = {
        "task": task,
        "risk_spec": risk_spec.to_dict(),
        "verified_formal_risk_definition": risk_definition,
        "classification_target_semantics": target_semantics,
        "primary_geobwer_estimand": {
            **estimand_fields,
            "slice_axis": axis,
            "included_slice_universe": sorted(all_slices - excluded),
            "excluded_slices": sorted(excluded),
        },
    }
    contract_json = _canonical_json(semantic_contract)
    return {
        "risk_spec_signature": explicit_signature,
        "risk_spec_verification_source": verification_source,
        "risk_spec_semantic_contract": contract_json,
        "risk_spec_semantic_contract_hash": hashlib.sha256(contract_json.encode()).hexdigest(),
        "risk_spec_verification_artifacts": _canonical_json({
            "geobwer_protocol": str(protocol_path) if protocol_path.is_file() else "missing",
            "geobwer_summary": str(summary_path),
            "formal_audit_table": str(audit_path),
            "probabilities": str(probability_path),
            "formal_output_manifest": str(manifest_path) if manifest_path.is_file() else "missing",
        }),
    }


def _risk_spec_comparison_contract(
    task: str, task_rows: list[dict[str, Any]], models: tuple[str, str]
) -> dict[str, Any]:
    semantic_contracts = {str(row["risk_spec_semantic_contract"]) for row in task_rows}
    verification = [
        {
            "model": row["model"],
            "seed": int(row["seed"]),
            "risk_spec_verification_source": row["risk_spec_verification_source"],
            "explicit_signature": row["risk_spec_signature"],
            "semantic_contract_hash": row["risk_spec_semantic_contract_hash"],
            "semantic_contract": json.loads(str(row["risk_spec_semantic_contract"])),
            "verification_artifacts": json.loads(str(row["risk_spec_verification_artifacts"])),
        }
        for row in sorted(task_rows, key=lambda value: (str(value["model"]), int(value["seed"])))
    ]
    return {
        "risk_spec_contract_equality": len(semantic_contracts) == 1,
        "same_risk_spec": len(semantic_contracts) == 1,
        "explicit_risk_spec_signatures": sorted({
            str(row["risk_spec_signature"]) for row in task_rows if str(row["risk_spec_signature"])
        }),
        "risk_spec_verification_sources": sorted({str(row["risk_spec_verification_source"]) for row in task_rows}),
        "risk_spec_verification": verification,
        "models": list(models),
        "task": task,
    }


def summarize_cell(root: str | Path, *, model: str, task: str) -> list[dict[str, Any]]:
    root = Path(root)
    output = []
    for seed_dir in _candidate_seed_dirs(root):
        summary_path = _canonical_geobwer_summary(seed_dir)
        country = next((row for row in read_csv_rows(summary_path) if str(row.get("axis")) == "country"), None)
        if country is None:
            raise ModelTaskGeneralizationError(f"No country axis in {summary_path}")
        audit_path = _canonical_formal_artifact(seed_dir, "formal_audit_table.csv")
        audit = read_csv_rows(audit_path)
        risk = np.asarray([float(row["risk"]) for row in audit], dtype=float)
        if not len(risk) or not np.all(np.isfinite(risk)):
            raise ModelTaskGeneralizationError(f"Invalid formal risks in {audit_path}")
        sample_ids = sorted({str(row["sample_id"]) for row in audit})
        sample_hash = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
        probability_path = _canonical_formal_artifact(seed_dir, "probabilities.npz")
        with np.load(probability_path, allow_pickle=False) as formal:
            probability_sample_ids = [str(value) for value in formal["sample_id"]]
            probabilities = np.asarray(formal["probabilities"])
            targets = np.asarray(formal["targets"])
            class_names = [str(value) for value in formal["class_names"]]
            thresholds = (
                np.asarray(formal["thresholds"])
                if "thresholds" in formal.files
                else np.asarray(formal["threshold"])
                if "threshold" in formal.files
                else None
            )
        if sorted(probability_sample_ids) != sample_ids or targets.shape[0] != len(probability_sample_ids):
            raise ModelTaskGeneralizationError(f"Formal probability/target rows do not match {audit_path}")
        target_digest = hashlib.sha256()
        for sample_id, target in sorted(zip(probability_sample_ids, targets), key=lambda item: item[0]):
            target_digest.update(sample_id.encode())
            target_digest.update(np.asarray(target).tobytes(order="C"))
        target_digest.update("\n".join(class_names).encode())
        risk_lineage = _semantic_risk_contract(
            task=task,
            seed_dir=seed_dir,
            summary_path=summary_path,
            summary_row=country,
            audit_path=audit_path,
            audit=audit,
            probability_path=probability_path,
            probability_sample_ids=probability_sample_ids,
            probabilities=probabilities,
            targets=targets,
            class_names=class_names,
            thresholds=thresholds,
        )
        output.append({
            "model": model, "task": task, "seed": _seed(seed_dir),
            "primary_risk": float(np.mean(risk)),
            "primary_score": float(1.0 - np.mean(risk)),
            "M": _float(country, "mean_risk"), "T": _float(country, "tail_risk"),
            "D": _float(country, "bwer", "geobwer"),
            "evidence_status": country.get("evidence_status", ""),
            **risk_lineage,
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
        risk_spec_contract = _risk_spec_comparison_contract(task, task_rows, models)
        seeds_by_model = {model: sorted(int(row["seed"]) for row in task_rows if row["model"] == model) for model in models}
        contract = {"task": task, "same_test_sample_support": len(sample_hashes) == 1,
                    "same_targets_and_class_mapping": len(target_hashes) == 1,
                    **risk_spec_contract,
                    "same_seeds": len({tuple(value) for value in seeds_by_model.values()}) == 1,
                    "expected_seeds_present": all(value == sorted(expected_seeds) for value in seeds_by_model.values()),
                    "seeds_by_model": seeds_by_model, "sample_hashes": sorted(sample_hashes),
                    "target_hashes": sorted(target_hashes)}
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
