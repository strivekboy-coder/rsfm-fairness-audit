from __future__ import annotations

"""Validation-locked reBEN S2->S1 adaptation ladder (Experiment 8).

Stage A is a read-only reference to the frozen paired-shift campaign.  Stages B
and C write only into a new experiment directory.  The test split is never used
for threshold selection or for the C->D decision.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.optimization_phase1 import distribution_metrics
from rsfm_fairness_audit.paired_probability_diagnostics import binary_auroc
from rsfm_fairness_audit.reben_phase1_runners import (
    _cache,
    validate_paired_cache_contract,
)
from rsfm_fairness_audit.reben_sensor_audit import (
    compute_multilabel_metrics,
    default_reben_class_names,
    select_thresholds_from_validation,
)
from rsfm_fairness_audit.reben_terramind_campaign import train_streaming_multilabel_probe


SCHEMA = "geobwer.reben.s2_to_s1_adaptation_ablation.v1"


class RebenAdaptationError(RuntimeError):
    pass


def _metadata_class_names(cache: Mapping[str, Any]) -> list[str]:
    value = cache["metadata"][0].get("class_names") if cache["metadata"] else None
    return [str(item) for item in value] if isinstance(value, list) and len(value) == 19 else default_reben_class_names()


def _thresholds(path: Path) -> np.ndarray:
    rows = read_csv_rows(path)
    values = sorted(rows, key=lambda row: int(row["class_index"]))
    result = np.asarray([float(row["threshold"]) for row in values], dtype=np.float32)
    if result.shape != (19,):
        raise RebenAdaptationError(f"Expected 19 frozen thresholds in {path}.")
    return result


def _load_checkpoint_probabilities(
    checkpoint: Path,
    embeddings_path: Path,
    output_path: Path,
    *,
    batch_size: int,
    device: str,
) -> Path:
    """Inference only: apply an existing linear head without changing it."""
    if output_path.is_file():
        values = np.load(output_path, mmap_mode="r")
        expected = len(np.load(embeddings_path, mmap_mode="r"))
        if values.shape == (expected, 19):
            print(f"[exp8:inference] reuse {output_path}")
            return output_path
        raise RebenAdaptationError(f"Stale probability output under new experiment directory: {output_path}")
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - Colab path
        raise RebenAdaptationError("PyTorch is required for linear-head inference.") from exc
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dimension = int(payload["embedding_dim"])
    model = nn.Linear(dimension, int(payload.get("label_count", 19)))
    model.load_state_dict(payload["state_dict"])
    resolved = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RebenAdaptationError("CUDA requested for Experiment 8 head inference but unavailable.")
    model = model.to(resolved).eval()
    mean = np.asarray(payload["embedding_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(payload["embedding_std"], dtype=np.float32).reshape(-1)
    values = np.load(embeddings_path, mmap_mode="r")
    if values.ndim != 2 or values.shape[1] != dimension:
        raise RebenAdaptationError("Embedding dimension does not match the frozen head.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(values), 19))
    gpu = torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "none"
    print(f"[exp8:inference] device={resolved} gpu={gpu} model_device={next(model.parameters()).device} rows={len(values)}")
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            x = (np.asarray(values[start:end], dtype=np.float32) - mean) / np.maximum(std, 1e-6)
            target[start:end] = torch.sigmoid(model(torch.as_tensor(x, device=resolved))).cpu().numpy()
            if end == len(values) or end % 32768 == 0:
                print(f"[exp8:inference] {end}/{len(values)}")
    target.flush()
    del target
    return output_path


def _align_array(source_cache: Mapping[str, Any], reference_cache: Mapping[str, Any], source: np.ndarray) -> np.ndarray:
    lookup = {str(row["sample_id"]): index for index, row in enumerate(source_cache["metadata"])}
    try:
        order = np.asarray([lookup[str(row["sample_id"])] for row in reference_cache["metadata"]], dtype=np.int64)
    except KeyError as exc:
        raise RebenAdaptationError(f"Paired cache is missing sample {exc.args[0]}") from exc
    if len(order) != len(source_cache["metadata"]):
        raise RebenAdaptationError("Paired caches do not have one-to-one sample support.")
    return np.asarray(source[order])


def _decision_partition(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split validation by independent unit so threshold fitting and go/no-go are disjoint."""
    units = sorted({str(row["independent_unit_id"]) for row in rows})
    calibration_units = {
        unit for unit in units
        if int(hashlib.sha256(f"{seed}:{unit}".encode()).hexdigest()[:8], 16) % 2 == 0
    }
    if not calibration_units or calibration_units == set(units):
        calibration_units = set(units[::2])
    calibration = np.asarray([i for i, row in enumerate(rows) if str(row["independent_unit_id"]) in calibration_units], dtype=np.int64)
    decision = np.asarray([i for i, row in enumerate(rows) if str(row["independent_unit_id"]) not in calibration_units], dtype=np.int64)
    if not len(calibration) or not len(decision):
        raise RebenAdaptationError("Validation partition produced an empty calibration or decision role.")
    return calibration, decision


def _binary_slice(axis: str, key: str, truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int8).reshape(-1)
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    return {
        "slice_axis": axis,
        "slice_value": key,
        "risk": float(np.mean(pred != (truth == 1))),
        "f1": (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "support": int(truth.size),
        "positive_support": int(np.sum(truth)),
    }


def _slice_metrics(targets: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray,
                   metadata: Sequence[Mapping[str, Any]], class_names: Sequence[str]) -> list[dict[str, Any]]:
    pred = probabilities >= thresholds[None, :]
    countries = np.asarray([str(row.get("country", "")) for row in metadata], dtype=str)
    result = [_binary_slice("class_label", class_names[index], targets[:, index], pred[:, index]) for index in range(targets.shape[1])]
    for country in sorted(set(countries) - {""}):
        selected = countries == country
        result.append(_binary_slice("country", country, targets[selected], pred[selected]))
        for index, label in enumerate(class_names):
            result.append(_binary_slice("country_x_label", f"{country} × {label}", targets[selected, index], pred[selected, index]))
    return result


def _evaluate(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *, stage: str,
    split_role: str,
    sensor: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate, _ = compute_multilabel_metrics(targets, probabilities, thresholds, class_names)
    aggregate["macro_auroc"] = float(np.nanmean([
        binary_auroc(targets[:, index], probabilities[:, index]) for index in range(targets.shape[1])
    ]))
    slices = _slice_metrics(targets, probabilities, thresholds, metadata, class_names)
    countries = [row for row in slices if row["slice_axis"] == "country"]
    dist = distribution_metrics({row["slice_value"]: float(row["risk"]) for row in countries})
    for key in ("profile", "weights"):
        dist.pop(key, None)
    return {"stage": stage, "split_role": split_role, "sensor": sensor, **aggregate, **dist}, slices


def directional_recovery(id_value: float, shifted_value: float, adapted_value: float, *, higher_is_better: bool) -> dict[str, Any]:
    available = (id_value - shifted_value) if higher_is_better else (shifted_value - id_value)
    restored = (adapted_value - shifted_value) if higher_is_better else (shifted_value - adapted_value)
    if not all(math.isfinite(value) for value in (id_value, shifted_value, adapted_value)) or available <= 1e-12:
        return {"recovery": float("nan"), "recovery_clipped": float("nan"), "overshoot": False, "identifiable": False}
    recovery = restored / available
    return {
        "recovery": recovery,
        "recovery_clipped": float(np.clip(recovery, 0.0, 1.0)),
        "overshoot": bool(recovery > 1.0),
        "identifiable": True,
    }


def _compare_slices(stage_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    base = {(str(row["slice_axis"]), str(row["slice_value"])): row for row in baseline_rows}
    result = []
    for row in stage_rows:
        key = (str(row["slice_axis"]), str(row["slice_value"]))
        if key not in base:
            continue
        delta = float(row["risk"]) - float(base[key]["risk"])
        result.append({**row, "baseline_shifted_risk": base[key]["risk"], "delta_vs_shifted": delta,
                       "no_harm": bool(delta <= tolerance), "harm": bool(delta > tolerance)})
    return result


def _validate_frozen_seed(source: Path, checkpoint: Path, thresholds_path: Path) -> None:
    contract_path = source / "paired_shift_contract.json"
    if not contract_path.is_file():
        raise RebenAdaptationError(f"Frozen Stage A contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    failures = []
    if contract.get("status") != "complete":
        failures.append("status_not_complete")
    if contract.get("same_head") is not True or contract.get("test_used_for_selection") is not False:
        failures.append("same_head_or_test_selection_contract_failed")
    # Legacy frozen paired-shift contracts predate the optional model_family
    # provenance field. Preserve compatibility without weakening validation of
    # an explicitly declared family.
    if "model_family" in contract and str(contract["model_family"]).lower() != "terramind":
        failures.append("unexpected_model_family")
    if contract.get("checkpoint_sha256") != file_sha256(checkpoint):
        failures.append("checkpoint_hash_mismatch")
    if contract.get("thresholds_sha256") != file_sha256(thresholds_path):
        failures.append("threshold_hash_mismatch")
    if failures:
        raise RebenAdaptationError(f"Frozen Stage A contract failed for {source}: {failures}")


def run_reben_adaptation_ablation(
    s2_cache_root: str | Path,
    s1_cache_root: str | Path,
    frozen_baseline_root: str | Path,
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (42, 73, 101),
    epochs: int = 100,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: str = "auto",
    stop_recovery: float = 0.80,
    tolerance: float = 0.002,
    geobwer_protocol: str | Path = Path("configs/geobwer/reben.yaml"),
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    protocol = BWERProtocol.from_mapping(load_yaml(geobwer_protocol))
    if not math.isclose(protocol.beta, 0.10) or protocol.deployment_weighting != "equal":
        raise RebenAdaptationError("Experiment 8 is frozen to beta=.10 and equal country deployment weights.")
    preflight = validate_paired_cache_contract(s2_cache_root, s1_cache_root)
    s2 = {split: _cache(Path(s2_cache_root), split) for split in ("train", "val", "test")}
    s1 = {split: _cache(Path(s1_cache_root), split) for split in ("train", "val", "test")}
    class_names = _metadata_class_names(s2["train"])
    all_summary: list[dict[str, Any]] = []
    all_slices: list[dict[str, Any]] = []
    all_recovery: list[dict[str, Any]] = []
    all_no_harm: list[dict[str, Any]] = []
    all_consistency: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    frozen = Path(frozen_baseline_root)
    for seed_value in seeds:
        seed = int(seed_value)
        source = frozen / f"seed_{seed}"
        checkpoint = source / "s2_trained_probe/linear_probe.pt"
        required = [checkpoint, source / "s2_trained_probe/s2_validation_probabilities.npy",
                    source / "s2_trained_probe/s2_id_test_probabilities.npy",
                    source / "s2_trained_probe/s1_ood_test_probabilities.npy",
                    source / "s2_validation_locked_thresholds.csv"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RebenAdaptationError("Frozen A artifacts are incomplete; refusing to retrain them: " + ", ".join(missing))
        _validate_frozen_seed(source, checkpoint, required[-1])
        seed_out = output / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        a_threshold = _thresholds(required[-1])
        a_s2_val = np.load(required[1], mmap_mode="r")
        a_s2_test = np.load(required[2], mmap_mode="r")
        a_s1_test = np.load(required[3], mmap_mode="r")
        a_s1_val_path = _load_checkpoint_probabilities(checkpoint, s1["val"]["dir"] / "embeddings.npy", seed_out / "B/s1_validation_probabilities.npy", batch_size=batch_size, device=device)
        a_s1_val = np.load(a_s1_val_path, mmap_mode="r")
        a_s2_val_aligned = _align_array(s2["val"], s1["val"], a_s2_val)
        calibration_idx, decision_idx = _decision_partition(s1["val"]["metadata"], seed)
        b_decision_thresholds = select_thresholds_from_validation(s1["val"]["labels"][calibration_idx], a_s1_val[calibration_idx])
        b_test_thresholds = select_thresholds_from_validation(s1["val"]["labels"], a_s1_val)
        write_csv(seed_out / "B/s1_validation_locked_thresholds.csv", [
            {"class_index": i, "class_label": class_names[i], "threshold": float(value), "test_used_for_selection": False}
            for i, value in enumerate(b_test_thresholds)
        ])
        c_probs, c_checkpoint = train_streaming_multilabel_probe(
            s1["train"]["dir"] / "embeddings.npy", s1["train"]["dir"] / "labels.npy",
            {"s1_validation": s1["val"]["dir"] / "embeddings.npy", "s1_test": s1["test"]["dir"] / "embeddings.npy"},
            seed_out / "C/s1_trained_probe", epochs=epochs, learning_rate=learning_rate,
            weight_decay=weight_decay, batch_size=batch_size, device=device, seed=seed,
            cache_signature=f"exp8:C:{preflight['s1_contract']['sample_id_hashes']}",
        )
        c_s1_val = np.load(c_probs["s1_validation"], mmap_mode="r")
        c_s1_test_unaligned = np.load(c_probs["s1_test"], mmap_mode="r")
        c_s1_test = _align_array(s1["test"], s2["test"], c_s1_test_unaligned)
        c_decision_thresholds = select_thresholds_from_validation(s1["val"]["labels"][calibration_idx], c_s1_val[calibration_idx])
        c_test_thresholds = select_thresholds_from_validation(s1["val"]["labels"], c_s1_val)
        write_csv(seed_out / "C/s1_validation_locked_thresholds.csv", [
            {"class_index": i, "class_label": class_names[i], "threshold": float(value), "test_used_for_selection": False}
            for i, value in enumerate(c_test_thresholds)
        ])
        evaluations = [
            ("A", "validation_decision_id", "S2", a_s2_val_aligned[decision_idx], s1["val"]["labels"][decision_idx], a_threshold, [s1["val"]["metadata"][i] for i in decision_idx]),
            ("A", "validation_decision_shifted", "S1", a_s1_val[decision_idx], s1["val"]["labels"][decision_idx], a_threshold, [s1["val"]["metadata"][i] for i in decision_idx]),
            ("B", "validation_decision", "S1", a_s1_val[decision_idx], s1["val"]["labels"][decision_idx], b_decision_thresholds, [s1["val"]["metadata"][i] for i in decision_idx]),
            ("C", "validation_decision", "S1", c_s1_val[decision_idx], s1["val"]["labels"][decision_idx], c_decision_thresholds, [s1["val"]["metadata"][i] for i in decision_idx]),
            ("A", "test_id", "S2", a_s2_test, s2["test"]["labels"], a_threshold, s2["test"]["metadata"]),
            ("A", "test_shifted", "S1", a_s1_test, s2["test"]["labels"], a_threshold, s2["test"]["metadata"]),
            ("B", "test", "S1", a_s1_test, s2["test"]["labels"], b_test_thresholds, s2["test"]["metadata"]),
            ("C", "test", "S1", c_s1_test, s2["test"]["labels"], c_test_thresholds, s2["test"]["metadata"]),
        ]
        evaluated: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for stage, role, sensor, probs, targets, cutoffs, metadata in evaluations:
            evaluated[(stage, role)] = _evaluate(np.asarray(probs), np.asarray(targets), np.asarray(cutoffs), metadata, class_names, stage=stage, split_role=role, sensor=sensor)
            summary, slices = evaluated[(stage, role)]
            all_summary.append({"seed": seed, **summary})
            all_slices.extend({"seed": seed, "stage": stage, "split_role": role, **row} for row in slices)
        for role, id_role, shifted_role in (("validation_decision", "validation_decision_id", "validation_decision_shifted"), ("test", "test_id", "test_shifted")):
            id_summary = evaluated[("A", id_role)][0]
            shifted_summary = evaluated[("A", shifted_role)][0]
            shifted_slices = evaluated[("A", shifted_role)][1]
            for stage in ("B", "C"):
                adapted_summary = evaluated[(stage, role)][0]
                for metric, higher in (("macro_auroc", True), ("macro_ap", True), ("macro_f1", True), ("mean_risk", False), ("tail_risk_beta_0_10", False), ("geobwer_beta_0_10", False)):
                    recovery = directional_recovery(float(id_summary[metric]), float(shifted_summary[metric]), float(adapted_summary[metric]), higher_is_better=higher)
                    all_recovery.append({"seed": seed, "split_role": role, "stage": stage, "metric": metric,
                                         "id_value": id_summary[metric], "shifted_value": shifted_summary[metric], "adapted_value": adapted_summary[metric], **recovery})
                current = {(row["stage"], row["metric"]): row for row in all_recovery if row["seed"] == seed and row["split_role"] == role}
                ordinary = float(current[(stage, "macro_auroc")]["recovery_clipped"])
                mean = float(current[(stage, "mean_risk")]["recovery_clipped"])
                tail = float(current[(stage, "tail_risk_beta_0_10")]["recovery_clipped"])
                adapted = evaluated[(stage, role)][0]
                levelling_down = bool(float(adapted["mean_risk"]) > float(shifted_summary["mean_risk"]) + tolerance
                                       and float(adapted["geobwer_beta_0_10"]) < float(shifted_summary["geobwer_beta_0_10"]) - tolerance)
                all_consistency.append({
                    "seed": seed, "split_role": role, "stage": stage,
                    "ordinary_auroc_recovery": ordinary, "mean_risk_recovery": mean, "tail_risk_recovery": tail,
                    "ordinary_minus_tail_recovery": ordinary - tail,
                    "ordinary_recovered_without_tail": bool(ordinary >= stop_recovery and tail < stop_recovery),
                    "mean_tail_recovery_consistent": bool(abs(mean - tail) <= 0.15),
                    "levelling_down_flag": levelling_down,
                })
                comparisons = _compare_slices(evaluated[(stage, role)][1], shifted_slices, tolerance)
                all_no_harm.extend({"seed": seed, "split_role": role, "stage": stage, **row} for row in comparisons)
        decision_recovery = {(row["stage"], row["metric"]): row for row in all_recovery if row["seed"] == seed and row["split_role"] == "validation_decision"}
        c_summary = evaluated[("C", "validation_decision")][0]
        a_shift = evaluated[("A", "validation_decision_shifted")][0]
        levelling_down = bool(float(c_summary["mean_risk"]) > float(a_shift["mean_risk"]) + tolerance and float(c_summary["geobwer_beta_0_10"]) < float(a_shift["geobwer_beta_0_10"]) - tolerance)
        criteria = {
            "macro_auroc_recovery_at_least_target": float(decision_recovery[("C", "macro_auroc")]["recovery_clipped"]) >= stop_recovery,
            "mean_risk_recovery_at_least_target": float(decision_recovery[("C", "mean_risk")]["recovery_clipped"]) >= stop_recovery,
            "tail_risk_recovery_at_least_target": float(decision_recovery[("C", "tail_risk_beta_0_10")]["recovery_clipped"]) >= stop_recovery,
            "ordinary_no_harm": float(c_summary["macro_auroc"]) >= float(a_shift["macro_auroc"]) - tolerance,
            "mean_no_harm": float(c_summary["mean_risk"]) <= float(a_shift["mean_risk"]) + tolerance,
            "tail_no_harm": float(c_summary["tail_risk_beta_0_10"]) <= float(a_shift["tail_risk_beta_0_10"]) + tolerance,
            "no_levelling_down": not levelling_down,
        }
        decisions.append({"seed": seed, "decision": "stop_at_C" if all(criteria.values()) else "consider_D", "criteria": criteria,
                          "validation_roles_disjoint": True, "test_used_for_decision": False,
                          "c_checkpoint": str(c_checkpoint), "c_checkpoint_sha256": file_sha256(c_checkpoint)})
    summary_path = output / "adaptation_stage_metrics.csv"
    recovery_path = output / "adaptation_recovery.csv"
    slices_path = output / "adaptation_slice_patterns.csv"
    no_harm_path = output / "adaptation_no_harm_slices.csv"
    consistency_path = output / "adaptation_mean_tail_consistency.csv"
    write_csv(summary_path, all_summary)
    write_csv(recovery_path, all_recovery)
    write_csv(slices_path, all_slices)
    write_csv(no_harm_path, all_no_harm)
    write_csv(consistency_path, all_consistency)
    aggregate_stop = all(item["decision"] == "stop_at_C" for item in decisions)
    gate_path = output / "stage_d_gate.json"
    gate_path.write_text(json.dumps({
        "schema": "geobwer.reben.adaptation_stage_d_gate.v1", "decision": "do_not_run_D" if aggregate_stop else "D_eligible_for_consideration",
        "rule": f"all seeds must recover >= {stop_recovery:.2f} on validation-only macro AUROC, country mean risk, and beta=.10 tail risk, with no harm and no levelling-down",
        "test_used_for_decision": False, "D_implemented_or_run": False,
        "interpretation": "D remains gated because representation adaptation is justified only if frozen-encoder head retraining is insufficient.",
        "per_seed": decisions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = output / "experiment8_manifest.json"
    manifest.write_text(json.dumps({
        "schema": SCHEMA, "status": "A_reused_B_C_complete_D_gated", "seeds": list(map(int, seeds)),
        "frozen_A_root": str(frozen.resolve()), "frozen_A_modified": False,
        "frozen_A_manifest_sha256": file_sha256(frozen / "paired_shift_panel_manifest.json") if (frozen / "paired_shift_panel_manifest.json").is_file() else "",
        "risk_spec_signature": protocol.risk_spec.signature, "protocol_signature": protocol.signature,
        "paired_cache_preflight": preflight, "test_used_for_selection": False,
        "validation_partition": "independent_unit_hash_50_50_threshold_calibration_vs_stop_go_decision",
        "recovery_definition": "directional_fraction_of_A_shift_gap_restored; raw plus clipped_[0,1]",
        "ordinary_performance_recovery_is_not_assumed_to_imply_tail_recovery": True,
        "stage_d_gate": str(gate_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metrics": summary_path, "recovery": recovery_path, "slices": slices_path, "no_harm": no_harm_path,
            "mean_tail_consistency": consistency_path, "stage_d_gate": gate_path, "manifest": manifest}


__all__ = ["RebenAdaptationError", "directional_recovery", "run_reben_adaptation_ablation"]
