from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_schema import class_mapping_hash


def _as_probability_matrix(values: Sequence[Sequence[float]], *, multilabel: bool = False) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must be finite and in [0, 1].")
    if not multilabel and not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Every multiclass probability row must sum to one.")
    return probabilities


def multiclass_audit_rows(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    class_names: Sequence[Any] | None = None,
    sample_ids: Sequence[Any] | None = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    clipped_log_loss: float = 1e-12,
) -> list[dict[str, Any]]:
    probs = _as_probability_matrix(probabilities)
    y = np.asarray(labels, dtype=int)
    if len(y) != len(probs) or np.any(y < 0) or np.any(y >= probs.shape[1]):
        raise ValueError("labels must align with probabilities and index valid classes.")
    names = tuple(class_names) if class_names is not None else tuple(range(probs.shape[1]))
    if len(names) != probs.shape[1]:
        raise ValueError("class_names length must equal the probability dimension.")
    ids = tuple(sample_ids) if sample_ids is not None else tuple(range(len(y)))
    if len(ids) != len(y):
        raise ValueError("sample_ids must align with labels.")
    meta = metadata or ({},) * len(y)
    if len(meta) != len(y):
        raise ValueError("metadata must align with labels.")
    mapping_hash = class_mapping_hash(names)
    prediction = np.argmax(probs, axis=1)
    rows: list[dict[str, Any]] = []
    for index in range(len(y)):
        entropy = float(-np.sum(probs[index] * np.log(np.clip(probs[index], clipped_log_loss, 1.0))))
        row = dict(meta[index])
        row.update(
            {
                "sample_id": str(ids[index]),
                "label": int(y[index]),
                "prediction": int(prediction[index]),
                "score": float(prediction[index] == y[index]),
                "risk": float(prediction[index] != y[index]),
                "log_loss": float(-math.log(max(float(probs[index, y[index]]), clipped_log_loss))),
                "confidence": float(np.max(probs[index])),
                "entropy": entropy,
                "probability_vector": tuple(float(value) for value in probs[index]),
                "class_mapping_hash": mapping_hash,
            }
        )
        rows.append(row)
    return rows


def multilabel_audit_rows(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Sequence[int | bool]],
    *,
    threshold: float | Sequence[float] = 0.5,
    class_names: Sequence[Any] | None = None,
    sample_ids: Sequence[Any] | None = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    probs = _as_probability_matrix(probabilities, multilabel=True)
    y = np.asarray(targets, dtype=int)
    if y.shape != probs.shape or np.any((y != 0) & (y != 1)):
        raise ValueError("targets must be a binary matrix aligned with probabilities.")
    threshold_array = np.asarray(threshold, dtype=float)
    if threshold_array.ndim == 0:
        threshold_array = np.full(probs.shape[1], float(threshold_array), dtype=float)
    if threshold_array.shape != (probs.shape[1],) or np.any(threshold_array < 0.0) or np.any(threshold_array > 1.0):
        raise ValueError("threshold must be a scalar or one value in [0,1] per class.")
    names = tuple(class_names) if class_names is not None else tuple(range(probs.shape[1]))
    if len(names) != probs.shape[1]:
        raise ValueError("class_names length must equal the probability dimension.")
    ids = tuple(sample_ids) if sample_ids is not None else tuple(range(len(y)))
    meta = metadata or ({},) * len(y)
    if len(ids) != len(y) or len(meta) != len(y):
        raise ValueError("sample_ids and metadata must align with targets.")
    pred = probs >= threshold_array[None, :]
    mapping_hash = class_mapping_hash(names)
    rows: list[dict[str, Any]] = []
    for index in range(len(y)):
        positives = int(np.sum(y[index]))
        false_negatives = int(np.sum((y[index] == 1) & (~pred[index])))
        row = dict(meta[index])
        row.update(
            {
                "sample_id": str(ids[index]),
                "risk": float(np.mean(pred[index] != y[index])),
                "hamming_loss": float(np.mean(pred[index] != y[index])),
                "false_negative_loss": float(false_negatives / positives) if positives else 0.0,
                "confidence": float(np.mean(np.maximum(probs[index], 1.0 - probs[index]))),
                "probability_vector": tuple(float(value) for value in probs[index]),
                "target_vector": tuple(int(value) for value in y[index]),
                "prediction_vector": tuple(bool(value) for value in pred[index]),
                "class_mapping_hash": mapping_hash,
            }
        )
        rows.append(row)
    return rows


def segmentation_count_risk(row: Mapping[str, Any], *, loss: str = "one_minus_iou") -> float:
    def value(*keys: str) -> float:
        for key in keys:
            if key in row and row[key] not in {None, ""}:
                return float(row[key])
        raise ValueError(f"Missing segmentation count; expected one of {keys}.")

    tp = value("TP", "tp")
    fp = value("FP", "fp")
    fn = value("FN", "fn")
    if loss == "one_minus_iou":
        denominator = tp + fp + fn
        score = 1.0 if denominator == 0.0 else tp / denominator
        return float(1.0 - score)
    if loss == "one_minus_dice":
        denominator = 2.0 * tp + fp + fn
        score = 1.0 if denominator == 0.0 else 2.0 * tp / denominator
        return float(1.0 - score)
    if loss == "false_negative_rate":
        denominator = tp + fn
        return 0.0 if denominator == 0.0 else float(fn / denominator)
    raise ValueError("loss must be one_minus_iou, one_minus_dice, or false_negative_rate.")


def segmentation_audit_rows(
    count_rows: Sequence[Mapping[str, Any]],
    *,
    primary_loss: str = "one_minus_iou",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, source in enumerate(count_rows):
        row = dict(source)
        row.setdefault("sample_id", str(index))
        row["risk"] = segmentation_count_risk(row, loss=primary_loss)
        row["one_minus_iou"] = segmentation_count_risk(row, loss="one_minus_iou")
        row["one_minus_dice"] = segmentation_count_risk(row, loss="one_minus_dice")
        row["false_negative_rate"] = segmentation_count_risk(row, loss="false_negative_rate")
        output.append(row)
    return output


def conformal_audit_rows(
    prediction_sets: Sequence[Sequence[Any]],
    targets: Sequence[Any],
    *,
    sample_ids: Sequence[Any] | None = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    number_of_classes: int | None = None,
) -> list[dict[str, Any]]:
    if len(prediction_sets) != len(targets):
        raise ValueError("prediction_sets and targets must align.")
    ids = tuple(sample_ids) if sample_ids is not None else tuple(range(len(targets)))
    meta = metadata or ({},) * len(targets)
    if len(ids) != len(targets) or len(meta) != len(targets):
        raise ValueError("sample_ids and metadata must align with targets.")
    output: list[dict[str, Any]] = []
    for index, (values, target) in enumerate(zip(prediction_sets, targets)):
        prediction_set = tuple(dict.fromkeys(values))
        covered = target in prediction_set
        row = dict(meta[index])
        row.update(
            {
                "sample_id": str(ids[index]),
                "prediction_set": prediction_set,
                "covered": bool(covered),
                "risk": float(not covered),
                "miscoverage_loss": float(not covered),
                "set_size": len(prediction_set),
                "set_size_fraction": float(len(prediction_set) / number_of_classes) if number_of_classes else "",
            }
        )
        output.append(row)
    return output


@dataclass(frozen=True)
class SelectiveAudit:
    rows: tuple[dict[str, Any], ...]
    requested_coverage: float
    retained_coverage: float
    threshold: float


def selective_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    coverage: float,
    confidence_column: str = "confidence",
) -> SelectiveAudit:
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("coverage must be in (0, 1].")
    if not rows:
        raise ValueError("rows must not be empty.")
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for index, source in enumerate(rows):
        if confidence_column not in source:
            raise ValueError(f"Missing confidence column: {confidence_column}")
        confidence = float(source[confidence_column])
        if not math.isfinite(confidence):
            raise ValueError("confidence values must be finite.")
        ranked.append((confidence, str(source.get("sample_id", index)), dict(source)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    retained_n = max(1, int(math.ceil(float(coverage) * len(ranked))))
    selected = tuple(item[2] for item in ranked[:retained_n])
    return SelectiveAudit(
        rows=selected,
        requested_coverage=float(coverage),
        retained_coverage=float(retained_n / len(ranked)),
        threshold=float(ranked[retained_n - 1][0]),
    )
