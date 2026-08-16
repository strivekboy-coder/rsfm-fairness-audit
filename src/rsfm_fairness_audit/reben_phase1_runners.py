from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.io import ensure_dir, write_csv
from rsfm_fairness_audit.optimization_phase1 import distribution_metrics
from rsfm_fairness_audit.reben_sensor_audit import (
    RebenRunLabels,
    compute_multilabel_metrics,
    default_reben_class_names,
    expand_predictions_to_label_audit_rows,
    run_reben_multilabel_bwer,
    select_thresholds_from_validation,
)
from rsfm_fairness_audit.reben_terramind_campaign import train_streaming_multilabel_probe


DEFAULT_BUDGETS = (0.05, 0.10, 0.25, 0.50, 1.00)


def _metadata(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        for key in ("sample_id", "country", "source_tile_id", "independent_unit_id"):
            if not str(row.get(key, "")).strip():
                raise ValueError(f"Missing {key} in {path} for sample={row.get('sample_id')}")
    return rows


def _cache(root: Path, split: str) -> dict[str, Any]:
    directory = root / split
    embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy", mmap_mode="r")
    metadata = _metadata(directory / "metadata.jsonl")
    if embeddings.ndim != 2 or labels.shape != (len(embeddings), 19) or len(metadata) != len(embeddings):
        raise ValueError(f"Invalid or misaligned embedding cache: {directory}")
    return {"dir": directory, "embeddings": embeddings, "labels": labels, "metadata": metadata}


def _sample_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_cache_contract(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    caches = {split: _cache(root, split) for split in ("train", "val", "test")}
    ids = {split: {str(row["sample_id"]) for row in cache["metadata"]} for split, cache in caches.items()}
    overlap = {
        f"{left}__{right}": len(ids[left] & ids[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlap.values()):
        raise ValueError(f"Sample leakage across reBEN splits: {overlap}")
    dimensions = {int(cache["embeddings"].shape[1]) for cache in caches.values()}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions differ across splits: {sorted(dimensions)}")
    return {
        "schema": "geobwer.reben.embedding_cache_contract.v1",
        "root": str(root),
        "split_counts": {split: len(cache["metadata"]) for split, cache in caches.items()},
        "sample_id_hashes": {split: _sample_hash(cache["metadata"]) for split, cache in caches.items()},
        "sample_overlap_counts": overlap,
        "embedding_dimension": dimensions.pop(),
        "source_tile_overlap_is_reported_not_failed": True,
    }


def _nested_unit_order(metadata: Sequence[Mapping[str, Any]], labels: np.ndarray, seed: int) -> list[str]:
    by_unit: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        by_unit.setdefault(str(row["independent_unit_id"]), []).append(index)
    unit_ids = sorted(by_unit)
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(unit_ids)[rng.permutation(len(unit_ids))])
    unit_labels = {
        unit_id: np.max(np.asarray(labels[indices], dtype=np.int8), axis=0).astype(bool)
        for unit_id, indices in by_unit.items()
    }
    uncovered = np.any(np.asarray(labels, dtype=bool), axis=0)
    cover: list[str] = []
    remaining = set(unit_ids)
    # Greedy set cover adds at most a small number of units and prevents a 5%
    # budget from accidentally omitting a rare label. The rest stays seeded-random.
    while np.any(uncovered):
        candidates = [unit for unit in shuffled if unit in remaining]
        best = max(candidates, key=lambda unit: int(np.sum(unit_labels[unit] & uncovered)))
        gain = unit_labels[best] & uncovered
        if not np.any(gain):
            break
        cover.append(best)
        remaining.remove(best)
        uncovered[gain] = False
    return cover + [unit for unit in shuffled if unit in remaining]


def validate_paired_cache_contract(s2_cache_root: str | Path, s1_cache_root: str | Path) -> dict[str, Any]:
    s2_root, s1_root = Path(s2_cache_root), Path(s1_cache_root)
    s2_contract = validate_cache_contract(s2_root)
    s1_contract = validate_cache_contract(s1_root)
    s2_test = _cache(s2_root, "test")
    s1_test = _cache(s1_root, "test")
    if int(s2_test["embeddings"].shape[1]) != int(s1_test["embeddings"].shape[1]):
        raise ValueError("S1 and S2 embeddings must share the same feature dimension for one unchanged head.")
    order, _ = _align_test_caches(s2_test, s1_test)
    return {
        "schema": "geobwer.reben.paired_sensor_shift.preflight.v1",
        "status": "ready",
        "same_embedding_dimension": True,
        "paired_test_sample_count": len(order),
        "paired_sample_ids_targets_and_metadata": True,
        "s2_train_only_for_probe": True,
        "s2_validation_only_for_thresholds": True,
        "same_head_required": True,
        "s2_contract": s2_contract,
        "s1_contract": s1_contract,
    }


def _materialize_budget(
    cache: Mapping[str, Any], selected_units: set[str], output: Path
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    indices = np.asarray(
        [index for index, row in enumerate(cache["metadata"]) if str(row["independent_unit_id"]) in selected_units],
        dtype=np.int64,
    )
    if not len(indices):
        raise ValueError("A label budget selected zero training rows.")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "train_indices.npy", indices)
    rows = [dict(cache["metadata"][index]) for index in indices]
    (output / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return indices, rows


def _group_distribution(audit_rows: Sequence[Mapping[str, Any]], axis: str = "country") -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in audit_rows:
        key = str(row.get(axis, "")).strip()
        if key:
            grouped.setdefault(key, []).append(float(row["risk_binary_error"]))
    return distribution_metrics({key: float(np.mean(values)) for key, values in grouped.items()})


def run_label_budget_campaign(
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    seeds: Sequence[int] = (42, 73, 101),
    epochs: int = 100,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: str = "auto",
) -> dict[str, Path]:
    cache_root = Path(cache_root)
    output = ensure_dir(output_dir)
    contract = validate_cache_contract(cache_root)
    (output / "embedding_cache_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    train = _cache(cache_root, "train")
    val = _cache(cache_root, "val")
    test = _cache(cache_root, "test")
    class_names = default_reben_class_names()
    # Prefer canonical class names embedded in metadata when present.
    metadata_names = train["metadata"][0].get("class_names") if train["metadata"] else None
    if isinstance(metadata_names, list) and len(metadata_names) == 19:
        class_names = [str(value) for value in metadata_names]
    summaries: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    unique_units = {str(row["independent_unit_id"]) for row in train["metadata"]}
    for seed in seeds:
        order = _nested_unit_order(train["metadata"], train["labels"], seed)
        previous: set[str] = set()
        for budget in sorted(set(float(value) for value in budgets)):
            if not 0.0 < budget <= 1.0:
                raise ValueError("Budgets must be in (0,1].")
            count = max(1, int(math.ceil(len(order) * budget)))
            selected = set(order[:count])
            if not previous.issubset(selected):
                raise RuntimeError("Nested label-budget invariant failed.")
            previous = selected
            tag = f"seed_{seed}/budget_{int(round(100 * budget)):03d}"
            train_indices, selected_rows = _materialize_budget(train, selected, output / "subsets" / tag)
            probabilities, checkpoint = train_streaming_multilabel_probe(
                train["dir"] / "embeddings.npy", train["dir"] / "labels.npy",
                {"validation": val["dir"] / "embeddings.npy", "test": test["dir"] / "embeddings.npy"},
                output / "probes" / tag,
                epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                batch_size=batch_size, device=device, seed=seed,
                cache_signature=f"nested_units:{_sample_hash(selected_rows)}",
                train_indices=train_indices,
            )
            val_prob = np.load(probabilities["validation"], mmap_mode="r")
            test_prob = np.load(probabilities["test"], mmap_mode="r")
            thresholds = select_thresholds_from_validation(val["labels"], val_prob)
            aggregate, _ = compute_multilabel_metrics(test["labels"], test_prob, thresholds, class_names)
            labels = RebenRunLabels(
                model_family="terramind", model_variant="TerraMind-1.0-base",
                sensor_mode="S2", input_mode="s2_image_only",
                adaptation_protocol="frozen_encoder_nested_label_budget_linear_probe",
                split_protocol="official_split_nested_independent_units", eval_scope="test",
                band_profile="reben_s2_l2a",
            )
            audit_rows = expand_predictions_to_label_audit_rows(
                test["metadata"], test["labels"], test_prob, thresholds, labels, class_names
            )
            dist = _group_distribution(audit_rows)
            dist.pop("profile")
            dist.pop("weights")
            bwer_dir = output / "bwer" / tag
            run_reben_multilabel_bwer(
                audit_rows, bwer_dir, model_name=f"terramind_s2_seed_{seed}_budget_{budget:.2f}",
                split="test", risk_column="risk_binary_error", min_support=20,
            )
            summaries.append({
                "seed": seed, "budget_fraction": budget,
                "selected_independent_units": len(selected), "total_independent_units": len(unique_units),
                "selected_samples": len(selected_rows), "test_samples": len(test["metadata"]),
                "checkpoint": str(checkpoint), **aggregate, **dist,
            })
            selections.extend(
                {"seed": seed, "budget_fraction": budget, "independent_unit_id": unit}
                for unit in sorted(selected)
            )
    summary_path = output / "label_budget_curves.csv"
    selection_path = output / "nested_budget_unit_selections.csv"
    write_csv(summary_path, summaries)
    write_csv(selection_path, selections)
    manifest = output / "label_budget_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "geobwer.reben.label_budget.v1", "status": "complete",
        "budgets": list(budgets), "seeds": list(seeds),
        "validation_and_test_fixed": True, "nested_by_independent_unit": True,
        "test_used_for_selection": False,
    }, indent=2), encoding="utf-8")
    return {"summary": summary_path, "selections": selection_path, "manifest": manifest}


def _align_test_caches(s2: Mapping[str, Any], s1: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    s1_index = {str(row["sample_id"]): index for index, row in enumerate(s1["metadata"])}
    s2_ids = [str(row["sample_id"]) for row in s2["metadata"]]
    if set(s2_ids) != set(s1_index):
        raise ValueError("S1 and S2 OOD test caches do not have identical sample IDs.")
    order = np.asarray([s1_index[sample_id] for sample_id in s2_ids], dtype=np.int64)
    if not np.array_equal(np.asarray(s2["labels"]), np.asarray(s1["labels"][order])):
        raise ValueError("S1 and S2 paired test targets differ.")
    for index, other in enumerate(order):
        for key in ("country", "source_tile_id", "independent_unit_id"):
            if str(s2["metadata"][index][key]) != str(s1["metadata"][int(other)][key]):
                raise ValueError(f"Paired metadata mismatch for {key}, sample={s2_ids[index]}")
    return order, np.asarray(s2["labels"])


def run_paired_sensor_shift(
    s2_cache_root: str | Path,
    s1_cache_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: str = "auto",
    aligned_s1_embeddings_path: str | Path | None = None,
) -> dict[str, Path]:
    s2_root, s1_root = Path(s2_cache_root), Path(s1_cache_root)
    output = ensure_dir(output_dir)
    paired_preflight = validate_paired_cache_contract(s2_root, s1_root)
    s2_contract = paired_preflight["s2_contract"]
    s1_contract = paired_preflight["s1_contract"]
    s2 = {split: _cache(s2_root, split) for split in ("train", "val", "test")}
    s1_test = _cache(s1_root, "test")
    if int(s2["train"]["embeddings"].shape[1]) != int(s1_test["embeddings"].shape[1]):
        raise ValueError("S1 and S2 embeddings must share the same feature dimension for one unchanged head.")
    s1_order, targets = _align_test_caches(s2["test"], s1_test)
    aligned_s1_path = Path(aligned_s1_embeddings_path) if aligned_s1_embeddings_path else output / "aligned_s1_test_embeddings.npy"
    if not aligned_s1_path.exists():
        aligned_s1_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(aligned_s1_path, np.asarray(s1_test["embeddings"][s1_order], dtype=np.float32))
    aligned_shape = np.load(aligned_s1_path, mmap_mode="r").shape
    if aligned_shape != (len(s1_order), int(s2["test"]["embeddings"].shape[1])):
        raise ValueError(f"Aligned S1 cache shape mismatch: {aligned_shape}")
    probabilities, checkpoint = train_streaming_multilabel_probe(
        s2["train"]["dir"] / "embeddings.npy", s2["train"]["dir"] / "labels.npy",
        {
            "s2_validation": s2["val"]["dir"] / "embeddings.npy",
            "s2_id_test": s2["test"]["dir"] / "embeddings.npy",
            "s1_ood_test": aligned_s1_path,
        },
        output / "s2_trained_probe", epochs=epochs, learning_rate=learning_rate,
        weight_decay=weight_decay, batch_size=batch_size, device=device, seed=seed,
        cache_signature=f"s2:{s2_contract['sample_id_hashes']}|s1:{s1_contract['sample_id_hashes']['test']}",
    )
    validation_prob = np.load(probabilities["s2_validation"], mmap_mode="r")
    thresholds = select_thresholds_from_validation(s2["val"]["labels"], validation_prob)
    class_names = default_reben_class_names()
    rows_by_domain: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for domain, key, sensor in (("ID", "s2_id_test", "S2"), ("OOD", "s1_ood_test", "S1")):
        probability = np.load(probabilities[key], mmap_mode="r")
        aggregate, _ = compute_multilabel_metrics(targets, probability, thresholds, class_names)
        labels = RebenRunLabels(
            model_family="terramind", model_variant="TerraMind-1.0-base",
            sensor_mode=sensor, input_mode="same_s2_trained_head",
            adaptation_protocol="s2_train_s2_validation_locked_same_head_paired_sensor_shift",
            split_protocol="official_paired_test_common_support", eval_scope="test",
            band_profile=f"reben_{sensor.lower()}",
        )
        audit = expand_predictions_to_label_audit_rows(
            s2["test"]["metadata"], targets, probability, thresholds, labels, class_names
        )
        rows_by_domain[domain] = audit
        dist = _group_distribution(audit)
        dist.pop("profile")
        dist.pop("weights")
        summary_rows.append({"domain": domain, "sensor": sensor, **aggregate, **dist})
        write_csv(output / f"{domain.lower()}_label_audit.csv", audit)
        run_reben_multilabel_bwer(
            audit, output / f"{domain.lower()}_bwer", model_name=f"terramind_same_head_{domain.lower()}",
            split="test", risk_column="risk_binary_error", min_support=20,
        )
    by_domain = {row["domain"]: row for row in summary_rows}
    delta = {
        "comparison": "S1_OOD_minus_S2_ID",
        "delta_mean_risk": by_domain["OOD"]["mean_risk"] - by_domain["ID"]["mean_risk"],
        "delta_tail_risk": by_domain["OOD"]["tail_risk_beta_0_10"] - by_domain["ID"]["tail_risk_beta_0_10"],
        "delta_geobwer": by_domain["OOD"]["geobwer_beta_0_10"] - by_domain["ID"]["geobwer_beta_0_10"],
    }
    delta["tail_acceleration_minus_mean"] = delta["delta_tail_risk"] - delta["delta_mean_risk"]
    delta["levelling_down_flag"] = bool(delta["delta_mean_risk"] > 0 and delta["delta_geobwer"] < 0)
    summary_path = output / "paired_shift_summary.csv"
    delta_path = output / "paired_shift_delta.csv"
    write_csv(summary_path, summary_rows)
    write_csv(delta_path, [delta])
    contract = output / "paired_shift_contract.json"
    contract.write_text(json.dumps({
        "schema": "geobwer.reben.paired_sensor_shift.v1", "status": "complete",
        "train_domain": "S2", "threshold_calibration_domain": "S2_validation",
        "id_domain": "S2_test", "ood_domain": "S1_test", "same_head": True,
        "paired_common_support": True, "test_used_for_selection": False,
        "effective_robustness_claimed": False,
        "checkpoint": str(checkpoint), "preflight": paired_preflight,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary_path, "delta": delta_path, "contract": contract}


def run_paired_sensor_shift_panel(
    s2_cache_root: str | Path,
    s1_cache_root: str | Path,
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (42, 73, 101),
    epochs: int = 100,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: str = "auto",
) -> dict[str, Path]:
    import csv

    output = ensure_dir(output_dir)
    preflight = validate_paired_cache_contract(s2_cache_root, s1_cache_root)
    (output / "paired_shift_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    s2_test = _cache(Path(s2_cache_root), "test")
    s1_test = _cache(Path(s1_cache_root), "test")
    order, _ = _align_test_caches(s2_test, s1_test)
    shared = output / "shared/aligned_s1_test_embeddings.npy"
    shared.parent.mkdir(parents=True, exist_ok=True)
    if not shared.exists():
        np.save(shared, np.asarray(s1_test["embeddings"][order], dtype=np.float32))
    summaries: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for seed in seeds:
        artifacts = run_paired_sensor_shift(
            s2_cache_root, s1_cache_root, output / f"seed_{int(seed)}", seed=int(seed),
            epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            batch_size=batch_size, device=device, aligned_s1_embeddings_path=shared,
        )
        with artifacts["summary"].open("r", encoding="utf-8-sig", newline="") as handle:
            summaries.extend({"seed": int(seed), **row} for row in csv.DictReader(handle))
        with artifacts["delta"].open("r", encoding="utf-8-sig", newline="") as handle:
            deltas.extend({"seed": int(seed), **row} for row in csv.DictReader(handle))
    summary_path = output / "paired_shift_seed_panel.csv"
    delta_path = output / "paired_shift_delta_seed_panel.csv"
    write_csv(summary_path, summaries)
    write_csv(delta_path, deltas)
    manifest = output / "paired_shift_panel_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "geobwer.reben.paired_sensor_shift_panel.v1", "status": "complete",
        "seeds": [int(seed) for seed in seeds], "same_s2_trained_head_within_seed": True,
        "shared_aligned_s1_cache": str(shared), "test_used_for_selection": False,
    }, indent=2), encoding="utf-8")
    return {"summary": summary_path, "delta": delta_path, "manifest": manifest}


__all__ = [
    "DEFAULT_BUDGETS",
    "run_label_budget_campaign",
    "run_paired_sensor_shift",
    "run_paired_sensor_shift_panel",
    "validate_cache_contract",
    "validate_paired_cache_contract",
]
