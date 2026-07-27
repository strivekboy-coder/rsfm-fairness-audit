from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.bwer_schema import artifact_signature, class_mapping_hash, validate_formal_audit_rows
from rsfm_fairness_audit.io import ensure_dir, write_csv


def file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text[:160] or "sample"


def _json_native(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    return value


def _validate_probabilities(probabilities: np.ndarray, *, multilabel: bool) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [samples, classes] with at least two classes.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must be finite and lie in [0, 1].")
    if not multilabel and not np.allclose(values.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4):
        raise ValueError("Multiclass probability rows must sum to one.")
    return values


def _base_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    model: str,
    task: str,
    split: str,
    split_role: str,
    protocol: BWERProtocol,
    model_signature: str,
    dataset_signature: str,
    independent_unit_column: str,
) -> list[dict[str, Any]]:
    if split_role not in {"test", "evaluation"}:
        raise ValueError("Formal audit output may contain only test/evaluation rows; calibration must be exported separately.")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(sample_rows):
        row = dict(source)
        sample_id = str(row.get("sample_id", row.get("image_id", row.get("id", index))))
        if not sample_id or sample_id in seen:
            raise ValueError(f"sample_id must be non-empty and unique; duplicate={sample_id!r}.")
        seen.add(sample_id)
        independent = row.get(independent_unit_column, row.get("independent_unit_id", sample_id))
        if independent is None or str(independent).strip() == "":
            raise ValueError(f"Missing independent unit for sample_id={sample_id}.")
        row.update(
            {
                "dataset": dataset,
                "model": model,
                "task": task,
                "split": split,
                "sample_id": sample_id,
                "independent_unit_id": str(independent),
                "split_role": split_role,
                "model_signature": model_signature,
                "dataset_signature": dataset_signature,
                "protocol_hash": protocol.signature,
                "metric_version": protocol.metric_version,
            }
        )
        output.append(row)
    return output


@dataclass(frozen=True)
class FormalOutputBundle:
    output_dir: Path
    audit_table: Path
    probability_artifact: Path
    class_mapping: Path
    manifest: Path
    row_count: int
    class_mapping_hash: str
    model_signature: str
    dataset_signature: str


def _multiclass_target_indices(
    targets: Sequence[int] | Sequence[str],
    class_to_index: Mapping[str, int],
) -> tuple[np.ndarray, str]:
    values = list(targets)
    if all(
        isinstance(value, numbers.Integral)
        and not isinstance(value, (bool, np.bool_))
        for value in values
    ):
        return np.asarray(values, dtype=np.int64), "integer_indices"
    if all(isinstance(value, str) for value in values):
        missing = sorted({value for value in values if value not in class_to_index})
        if missing:
            raise ValueError(
                "String targets must be class labels present in class_names; "
                f"missing labels: {missing[:10]}"
            )
        return (
            np.asarray([class_to_index[value] for value in values], dtype=np.int64),
            "string_class_labels",
        )
    raise ValueError(
        "Multiclass targets must be uniformly integer class indices or uniformly "
        "string class labels; mixed or ambiguous target types are not accepted."
    )


def _write_manifest(
    output: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    probability_artifact: Path,
    mapping_path: Path,
    protocol: BWERProtocol,
    model_lineage: Mapping[str, Any],
    dataset_lineage: Mapping[str, Any],
    output_schema: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "output_schema": output_schema,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "protocol": protocol.to_dict(),
        "protocol_hash": protocol.signature,
        "model_lineage": _json_native(model_lineage),
        "dataset_lineage": _json_native(dataset_lineage),
        "artifacts": {
            "formal_audit_table": "formal_audit_table.csv",
            "probability_artifact": str(probability_artifact.relative_to(output)),
            "probability_sha256": file_sha256(probability_artifact),
            "class_mapping": str(mapping_path.relative_to(output)),
            "class_mapping_sha256": file_sha256(mapping_path),
        },
        "extra": _json_native(extra or {}),
    }
    path = output / "formal_output_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_multiclass_bundle(
    output_dir: str | Path,
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    targets: Sequence[int] | Sequence[str],
    class_names: Sequence[Any],
    dataset: str,
    model: str,
    split: str,
    protocol: BWERProtocol,
    model_lineage: Mapping[str, Any],
    dataset_lineage: Mapping[str, Any],
    independent_unit_column: str = "independent_unit_id",
    split_role: str = "evaluation",
) -> FormalOutputBundle:
    output = ensure_dir(output_dir)
    probs = _validate_probabilities(np.asarray(probabilities), multilabel=False)
    classes = tuple(str(value) for value in class_names)
    if len(classes) != probs.shape[1] or len(set(classes)) != len(classes):
        raise ValueError("class_names must be unique and align with the probability columns.")
    if len(sample_rows) != len(probs) or len(targets) != len(probs):
        raise ValueError("sample_rows, probabilities, and targets must be aligned.")
    class_to_index = {name: index for index, name in enumerate(classes)}
    target_indices, target_encoding = _multiclass_target_indices(
        targets, class_to_index
    )
    if np.any(target_indices < 0) or np.any(target_indices >= len(classes)):
        raise ValueError("targets contain a class absent from class_names.")
    model_sig = artifact_signature(model_lineage)
    dataset_sig = artifact_signature(dataset_lineage)
    mapping_hash = class_mapping_hash(classes)
    rows = _base_rows(
        sample_rows,
        dataset=dataset,
        model=model,
        task="multiclass_classification",
        split=split,
        split_role=split_role,
        protocol=protocol,
        model_signature=model_sig,
        dataset_signature=dataset_sig,
        independent_unit_column=independent_unit_column,
    )
    predicted = np.argmax(probs, axis=1)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1) / math.log(len(classes))
    probability_path = output / "probabilities.npz"
    np.savez_compressed(
        probability_path,
        sample_id=np.asarray([row["sample_id"] for row in rows], dtype=str),
        probabilities=probs,
        targets=target_indices,
        class_names=np.asarray(classes, dtype=str),
    )
    mapping_path = output / "class_mapping.json"
    mapping_path.write_text(
        json.dumps({"classes": list(classes), "class_to_index": class_to_index, "class_mapping_hash": mapping_hash}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    relative_probability = str(probability_path.relative_to(output))
    for index, row in enumerate(rows):
        true_index = int(target_indices[index])
        pred_index = int(predicted[index])
        row.update(
            {
                "label": classes[true_index],
                "prediction": classes[pred_index],
                "risk": float(pred_index != true_index),
                "log_loss": float(-math.log(max(float(probs[index, true_index]), 1e-12))),
                "confidence": float(probs[index, pred_index]),
                "normalized_entropy": float(entropy[index]),
                "probabilities_path": relative_probability,
                "probability_row": index,
                "class_mapping_hash": mapping_hash,
            }
        )
    validation = validate_formal_audit_rows(
        rows,
        task_adapter="multiclass",
        required_cluster_column=protocol.spatial_block_column
        if protocol.inference_method == "spatial_maxt"
        else protocol.cluster_column
        if protocol.inference_method != "none"
        else None,
        require_spatial_block=protocol.inference_method == "spatial_maxt",
        require_probabilities=True,
        expected_protocol_hash=protocol.signature,
        expected_metric_version=protocol.metric_version,
    )
    if not validation.ok:
        raise ValueError("Formal multiclass output failed validation: " + " | ".join(validation.errors))
    audit_path = output / "formal_audit_table.csv"
    write_csv(audit_path, rows)
    manifest = _write_manifest(
        output,
        rows=rows,
        probability_artifact=probability_path,
        mapping_path=mapping_path,
        protocol=protocol,
        model_lineage=model_lineage,
        dataset_lineage=dataset_lineage,
        output_schema="geobwer.multiclass.v1",
        extra={"target_encoding": target_encoding},
    )
    return FormalOutputBundle(output, audit_path, probability_path, mapping_path, manifest, len(rows), mapping_hash, model_sig, dataset_sig)


def write_multilabel_bundle(
    output_dir: str | Path,
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    targets: Sequence[Sequence[int]] | np.ndarray,
    class_names: Sequence[Any],
    dataset: str,
    model: str,
    split: str,
    protocol: BWERProtocol,
    model_lineage: Mapping[str, Any],
    dataset_lineage: Mapping[str, Any],
    threshold: float | Sequence[float] = 0.5,
    independent_unit_column: str = "independent_unit_id",
    split_role: str = "evaluation",
) -> FormalOutputBundle:
    output = ensure_dir(output_dir)
    probs = _validate_probabilities(np.asarray(probabilities), multilabel=True)
    labels = np.asarray(targets, dtype=np.int8)
    classes = tuple(str(value) for value in class_names)
    if labels.shape != probs.shape or len(sample_rows) != len(probs) or len(classes) != probs.shape[1]:
        raise ValueError("sample_rows, probabilities, targets, and class_names must align.")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("Multilabel targets must be binary.")
    threshold_array = np.asarray(threshold, dtype=np.float32)
    if threshold_array.ndim == 0:
        threshold_array = np.full(probs.shape[1], float(threshold_array), dtype=np.float32)
    if threshold_array.shape != (probs.shape[1],) or np.any(threshold_array <= 0.0) or np.any(threshold_array >= 1.0):
        raise ValueError("threshold must be a scalar or one value in (0,1) per class.")
    model_sig = artifact_signature(model_lineage)
    dataset_sig = artifact_signature(dataset_lineage)
    mapping_hash = class_mapping_hash(classes)
    rows = _base_rows(
        sample_rows,
        dataset=dataset,
        model=model,
        task="multilabel_classification",
        split=split,
        split_role=split_role,
        protocol=protocol,
        model_signature=model_sig,
        dataset_signature=dataset_sig,
        independent_unit_column=independent_unit_column,
    )
    predictions = (probs >= threshold_array[None, :]).astype(np.int8)
    probability_path = output / "probabilities.npz"
    np.savez_compressed(
        probability_path,
        sample_id=np.asarray([row["sample_id"] for row in rows], dtype=str),
        probabilities=probs,
        targets=labels,
        class_names=np.asarray(classes, dtype=str),
        thresholds=threshold_array,
        # Compatibility alias for BWER 1.x consumers. New code should read
        # ``thresholds`` because validation may calibrate one threshold per class.
        threshold=threshold_array,
    )
    mapping_path = output / "class_mapping.json"
    mapping_path.write_text(
        json.dumps({"classes": list(classes), "class_mapping_hash": mapping_hash}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    relative_probability = str(probability_path.relative_to(output))
    for index, row in enumerate(rows):
        positives = labels[index] == 1
        false_negatives = int(np.sum(positives & (predictions[index] == 0)))
        positive_count = int(np.sum(positives))
        row.update(
            {
                "risk": float(np.mean(predictions[index] != labels[index])),
                "false_negative_rate": float(false_negatives / positive_count) if positive_count else 0.0,
                "label_cardinality": positive_count,
                "prediction_cardinality": int(np.sum(predictions[index])),
                "confidence": float(np.mean(np.maximum(probs[index], 1.0 - probs[index]))),
                "probabilities_path": relative_probability,
                "probability_row": index,
                "class_mapping_hash": mapping_hash,
            }
        )
    validation = validate_formal_audit_rows(
        rows,
        task_adapter="multilabel",
        required_cluster_column=protocol.spatial_block_column
        if protocol.inference_method == "spatial_maxt"
        else protocol.cluster_column
        if protocol.inference_method != "none"
        else None,
        require_spatial_block=protocol.inference_method == "spatial_maxt",
        require_probabilities=True,
        expected_protocol_hash=protocol.signature,
        expected_metric_version=protocol.metric_version,
    )
    if not validation.ok:
        raise ValueError("Formal multilabel output failed validation: " + " | ".join(validation.errors))
    audit_path = output / "formal_audit_table.csv"
    write_csv(audit_path, rows)
    manifest = _write_manifest(
        output,
        rows=rows,
        probability_artifact=probability_path,
        mapping_path=mapping_path,
        protocol=protocol,
        model_lineage=model_lineage,
        dataset_lineage=dataset_lineage,
        output_schema="geobwer.multilabel.v1",
        extra={"decision_thresholds": threshold_array.tolist(), "threshold_policy": "frozen_validation_per_class"},
    )
    return FormalOutputBundle(output, audit_path, probability_path, mapping_path, manifest, len(rows), mapping_hash, model_sig, dataset_sig)


def write_segmentation_bundle(
    output_dir: str | Path,
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    positive_probability_maps: Sequence[np.ndarray],
    target_masks: Sequence[np.ndarray],
    dataset: str,
    model: str,
    split: str,
    protocol: BWERProtocol,
    model_lineage: Mapping[str, Any],
    dataset_lineage: Mapping[str, Any],
    valid_masks: Sequence[np.ndarray] | None = None,
    threshold: float = 0.5,
    independent_unit_column: str = "independent_unit_id",
    split_role: str = "evaluation",
) -> FormalOutputBundle:
    output = ensure_dir(output_dir)
    if not (len(sample_rows) == len(positive_probability_maps) == len(target_masks)) or not sample_rows:
        raise ValueError("sample_rows, probability maps, and target masks must be non-empty and aligned.")
    if valid_masks is not None and len(valid_masks) != len(sample_rows):
        raise ValueError("valid_masks must align with sample_rows.")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be in (0, 1).")
    model_sig = artifact_signature(model_lineage)
    dataset_sig = artifact_signature(dataset_lineage)
    classes = ("background", "positive")
    mapping_hash = class_mapping_hash(classes)
    rows = _base_rows(
        sample_rows,
        dataset=dataset,
        model=model,
        task="binary_segmentation",
        split=split,
        split_role=split_role,
        protocol=protocol,
        model_signature=model_sig,
        dataset_signature=dataset_sig,
        independent_unit_column=independent_unit_column,
    )
    map_dir = ensure_dir(output / "probability_maps")
    index_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        probability = np.asarray(positive_probability_maps[index], dtype=np.float32).squeeze()
        target = np.asarray(target_masks[index]).squeeze()
        if probability.shape != target.shape or probability.ndim != 2:
            raise ValueError(f"Probability/target shape mismatch for sample_id={row['sample_id']}: {probability.shape} vs {target.shape}.")
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0) or np.any(probability > 1.0):
            raise ValueError(f"Invalid probability map for sample_id={row['sample_id']}.")
        valid = np.ones_like(target, dtype=bool) if valid_masks is None else np.asarray(valid_masks[index], dtype=bool).squeeze()
        if valid.shape != target.shape or not np.any(valid):
            raise ValueError(f"Valid mask is empty or misaligned for sample_id={row['sample_id']}.")
        truth = target.astype(bool)
        prediction = probability >= float(threshold)
        tp = int(np.sum(valid & prediction & truth))
        fp = int(np.sum(valid & prediction & ~truth))
        fn = int(np.sum(valid & ~prediction & truth))
        tn = int(np.sum(valid & ~prediction & ~truth))
        union = tp + fp + fn
        iou = float(tp / union) if union else 1.0
        map_path = map_dir / f"{index:07d}_{_safe_name(row['sample_id'])}.npz"
        np.savez_compressed(map_path, positive_probability=probability, target=target.astype(np.int8), valid=valid.astype(np.uint8))
        relative = str(map_path.relative_to(output))
        row.update(
            {
                "risk": 1.0 - iou,
                "iou": iou,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "positive_support": int(np.sum(valid & truth)),
                "valid_pixel_support": int(np.sum(valid)),
                "confidence": float(np.mean(np.maximum(probability[valid], 1.0 - probability[valid]))),
                "probability_map_path": relative,
                "class_mapping_hash": mapping_hash,
            }
        )
        index_records.append({"sample_id": row["sample_id"], "probability_map_path": relative, "sha256": file_sha256(map_path)})
    index_path = output / "probability_maps_index.json"
    index_path.write_text(json.dumps(index_records, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping_path = output / "class_mapping.json"
    mapping_path.write_text(
        json.dumps({"classes": list(classes), "positive_class": "positive", "class_mapping_hash": mapping_hash}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    validation = validate_formal_audit_rows(
        rows,
        task_adapter="segmentation",
        required_cluster_column=protocol.spatial_block_column
        if protocol.inference_method == "spatial_maxt"
        else protocol.cluster_column
        if protocol.inference_method != "none"
        else None,
        require_spatial_block=protocol.inference_method == "spatial_maxt",
        require_probabilities=True,
        expected_protocol_hash=protocol.signature,
        expected_metric_version=protocol.metric_version,
    )
    if not validation.ok:
        raise ValueError("Formal segmentation output failed validation: " + " | ".join(validation.errors))
    audit_path = output / "formal_audit_table.csv"
    write_csv(audit_path, rows)
    manifest = _write_manifest(
        output,
        rows=rows,
        probability_artifact=index_path,
        mapping_path=mapping_path,
        protocol=protocol,
        model_lineage=model_lineage,
        dataset_lineage=dataset_lineage,
        output_schema="geobwer.binary_segmentation.v1",
        extra={"decision_threshold": threshold, "map_count": len(rows)},
    )
    return FormalOutputBundle(output, audit_path, index_path, mapping_path, manifest, len(rows), mapping_hash, model_sig, dataset_sig)


__all__ = [
    "FormalOutputBundle",
    "file_sha256",
    "write_multiclass_bundle",
    "write_multilabel_bundle",
    "write_segmentation_bundle",
]
