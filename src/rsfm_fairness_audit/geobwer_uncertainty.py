from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


class UncertaintyProtocolError(ValueError):
    """Raised when calibration/test separation or task geometry is invalid."""


def _multiclass_probabilities(values: Sequence[Sequence[float]]) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise UncertaintyProtocolError("Multiclass probabilities must have shape [N,K], K>=2.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0) or not np.all(np.isfinite(probabilities)):
        raise UncertaintyProtocolError("Probabilities must be finite and in [0,1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4):
        raise UncertaintyProtocolError("Multiclass probability rows must sum to one.")
    return probabilities


def conformal_quantile(scores: Sequence[float], *, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise UncertaintyProtocolError("Calibration scores must be a non-empty finite vector.")
    if not 0.0 < alpha < 1.0:
        raise UncertaintyProtocolError("alpha must be in (0,1).")
    rank = min(int(math.ceil((len(values) + 1) * (1.0 - alpha))), len(values))
    return float(np.sort(values)[rank - 1])


def multiclass_nonconformity_scores(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    method: str = "lac",
    raps_lambda: float = 0.01,
    raps_k_reg: int = 5,
) -> np.ndarray:
    probs = _multiclass_probabilities(probabilities)
    y = np.asarray(targets, dtype=int)
    if y.shape != (len(probs),) or np.any(y < 0) or np.any(y >= probs.shape[1]):
        raise UncertaintyProtocolError("targets must be valid class indices aligned with probabilities.")
    if method == "lac":
        return 1.0 - probs[np.arange(len(probs)), y]
    if method not in {"aps", "raps"}:
        raise UncertaintyProtocolError("method must be lac, aps, or raps.")
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative_before = np.cumsum(sorted_probs, axis=1) - sorted_probs
    inverse = np.empty_like(order)
    inverse[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])[None, :]
    ranks = inverse[np.arange(len(probs)), y]
    # Deterministic APS uses probability mass strictly ahead of the candidate.
    # The prediction set below applies exactly this same score, preserving the
    # split-conformal guarantee without an unrecorded random boundary variable.
    scores = cumulative_before[np.arange(len(probs)), ranks]
    if method == "raps":
        if raps_lambda < 0.0 or raps_k_reg < 0:
            raise UncertaintyProtocolError("RAPS lambda/k_reg must be nonnegative.")
        scores = scores + raps_lambda * np.maximum(ranks + 1 - int(raps_k_reg), 0)
    return scores


def multiclass_prediction_sets(
    probabilities: Sequence[Sequence[float]],
    threshold: float | Sequence[float],
    *,
    method: str = "lac",
    raps_lambda: float = 0.01,
    raps_k_reg: int = 5,
) -> np.ndarray:
    probs = _multiclass_probabilities(probabilities)
    q = np.asarray(threshold, dtype=float)
    if q.ndim == 0:
        q = np.full(len(probs), float(q))
    if q.shape != (len(probs),) or np.any(np.isnan(q)) or np.any(np.isneginf(q)):
        raise UncertaintyProtocolError(
            "threshold must be scalar or one value per test sample; +inf is allowed "
            "for a conservative unsupported-location prediction set."
        )
    if method == "lac":
        return (1.0 - probs) <= q[:, None]
    if method not in {"aps", "raps"}:
        raise UncertaintyProtocolError("method must be lac, aps, or raps.")
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    scores = np.cumsum(sorted_probs, axis=1) - sorted_probs
    if method == "raps":
        ranks = np.arange(1, probs.shape[1] + 1)
        scores = scores + raps_lambda * np.maximum(ranks - int(raps_k_reg), 0)[None, :]
    selected_sorted = scores <= q[:, None]
    selected = np.zeros_like(selected_sorted, dtype=bool)
    selected[np.arange(len(probs))[:, None], order] = selected_sorted
    return selected


@dataclass(frozen=True)
class MulticlassConformalModel:
    alpha: float
    method: str
    global_threshold: float
    group_thresholds: tuple[tuple[str, float], ...] = ()
    group_support: tuple[tuple[str, int], ...] = ()
    minimum_group_calibration_support: int = 0
    raps_lambda: float = 0.01
    raps_k_reg: int = 5

    def threshold_for(self, groups: Sequence[Any] | None, count: int) -> np.ndarray:
        if groups is None or not self.group_thresholds:
            return np.full(count, self.global_threshold, dtype=float)
        if len(groups) != count:
            raise UncertaintyProtocolError("test_groups must align with test probabilities.")
        values = dict(self.group_thresholds)
        return np.asarray([values.get(str(group), self.global_threshold) for group in groups], dtype=float)


def fit_multiclass_conformal(
    calibration_probabilities: Sequence[Sequence[float]],
    calibration_targets: Sequence[int],
    *,
    alpha: float = 0.10,
    method: str = "lac",
    calibration_groups: Sequence[Any] | None = None,
    minimum_group_calibration_support: int = 100,
    raps_lambda: float = 0.01,
    raps_k_reg: int = 5,
) -> MulticlassConformalModel:
    scores = multiclass_nonconformity_scores(
        calibration_probabilities,
        calibration_targets,
        method=method,
        raps_lambda=raps_lambda,
        raps_k_reg=raps_k_reg,
    )
    global_threshold = conformal_quantile(scores, alpha=alpha)
    thresholds: list[tuple[str, float]] = []
    supports: list[tuple[str, int]] = []
    if calibration_groups is not None:
        if len(calibration_groups) != len(scores):
            raise UncertaintyProtocolError("calibration_groups must align with calibration rows.")
        group_array = np.asarray([str(value) for value in calibration_groups], dtype=object)
        for group in sorted(set(group_array.tolist())):
            mask = group_array == group
            support = int(np.sum(mask))
            supports.append((group, support))
            if support >= minimum_group_calibration_support:
                thresholds.append((group, conformal_quantile(scores[mask], alpha=alpha)))
    return MulticlassConformalModel(
        alpha=float(alpha),
        method=method,
        global_threshold=global_threshold,
        group_thresholds=tuple(thresholds),
        group_support=tuple(supports),
        minimum_group_calibration_support=int(minimum_group_calibration_support),
        raps_lambda=float(raps_lambda),
        raps_k_reg=int(raps_k_reg),
    )


def multiclass_conformal_audit_rows(
    model: MulticlassConformalModel,
    test_probabilities: Sequence[Sequence[float]],
    test_targets: Sequence[int],
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    test_groups: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    probabilities = _multiclass_probabilities(test_probabilities)
    targets = np.asarray(test_targets, dtype=int)
    if len(sample_rows) != len(probabilities) or targets.shape != (len(probabilities),):
        raise UncertaintyProtocolError("sample_rows/targets must align with test probabilities.")
    thresholds = model.threshold_for(test_groups, len(probabilities))
    sets = multiclass_prediction_sets(
        probabilities,
        thresholds,
        method=model.method,
        raps_lambda=model.raps_lambda,
        raps_k_reg=model.raps_k_reg,
    )
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sample_rows):
        covered = bool(sets[index, targets[index]])
        row = dict(source)
        row.update(
            {
                "risk": float(not covered),
                "miscoverage_loss": float(not covered),
                "covered": covered,
                "set_size": int(np.sum(sets[index])),
                "set_size_fraction": float(np.mean(sets[index])),
                "conformal_alpha": model.alpha,
                "conformal_method": model.method,
                "conformal_threshold": float(thresholds[index]),
                "coverage_target_violation": float((not covered) - model.alpha),
            }
        )
        rows.append(row)
    return rows


@dataclass(frozen=True)
class SelectiveThreshold:
    target_coverage: float
    confidence_threshold: float
    calibration_coverage: float


def fit_selective_threshold(calibration_confidence: Sequence[float], *, target_coverage: float) -> SelectiveThreshold:
    confidence = np.asarray(calibration_confidence, dtype=float)
    if confidence.ndim != 1 or len(confidence) == 0 or not np.all(np.isfinite(confidence)):
        raise UncertaintyProtocolError("calibration_confidence must be a non-empty finite vector.")
    if not 0.0 < target_coverage <= 1.0:
        raise UncertaintyProtocolError("target_coverage must be in (0,1].")
    retained = max(1, int(math.ceil(target_coverage * len(confidence))))
    threshold = float(np.sort(confidence)[::-1][retained - 1])
    return SelectiveThreshold(
        target_coverage=float(target_coverage),
        confidence_threshold=threshold,
        calibration_coverage=float(np.mean(confidence >= threshold)),
    )


def apply_selective_threshold(
    model: SelectiveThreshold,
    test_risk: Sequence[float],
    test_confidence: Sequence[float],
    *,
    sample_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    risk = np.asarray(test_risk, dtype=float)
    confidence = np.asarray(test_confidence, dtype=float)
    if len(risk) != len(confidence) or len(sample_rows) != len(risk):
        raise UncertaintyProtocolError("risk, confidence, and sample_rows must align.")
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sample_rows):
        accepted = bool(confidence[index] >= model.confidence_threshold)
        row = dict(source)
        row.update(
            {
                "accepted": accepted,
                "risk": float(risk[index]) if accepted else float("nan"),
                "base_risk": float(risk[index]),
                "confidence": float(confidence[index]),
                "selective_threshold": model.confidence_threshold,
                "target_coverage": model.target_coverage,
            }
        )
        rows.append(row)
    return rows


@dataclass(frozen=True)
class ConformalRiskControlModel:
    alpha: float
    probability_threshold: float
    calibration_empirical_risk: float
    calibration_corrected_risk: float
    calibration_samples: int
    risk_name: str


def _per_unit_false_negative_risk(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    valid_masks: np.ndarray | None = None,
) -> np.ndarray:
    predicted = probabilities >= threshold
    valid = np.ones_like(targets, dtype=bool) if valid_masks is None else np.asarray(valid_masks, dtype=bool)
    if valid.shape != targets.shape:
        raise UncertaintyProtocolError("valid_masks must align with probabilities/targets.")
    positives = (targets == 1) & valid
    denominator = positives.reshape(len(positives), -1).sum(axis=1)
    false_negative = (positives & ~predicted).reshape(len(positives), -1).sum(axis=1)
    return np.divide(false_negative, denominator, out=np.zeros_like(false_negative, dtype=float), where=denominator > 0)


def fit_false_negative_crc(
    calibration_probabilities: Sequence[Any] | np.ndarray,
    calibration_targets: Sequence[Any] | np.ndarray,
    *,
    alpha: float = 0.10,
    maximum_candidates: int = 4096,
    risk_name: str = "false_negative_rate",
    valid_masks: Sequence[Any] | np.ndarray | None = None,
) -> ConformalRiskControlModel:
    probabilities = np.asarray(calibration_probabilities, dtype=float)
    targets = np.asarray(calibration_targets, dtype=int)
    if probabilities.shape != targets.shape or probabilities.ndim < 2 or len(probabilities) == 0:
        raise UncertaintyProtocolError("CRC probabilities and binary targets must share shape [N,...].")
    valid = np.ones_like(targets, dtype=bool) if valid_masks is None else np.asarray(valid_masks, dtype=bool)
    if valid.shape != targets.shape:
        raise UncertaintyProtocolError("CRC valid_masks must align with probabilities/targets.")
    if np.any(((targets != 0) & (targets != 1)) & valid) or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise UncertaintyProtocolError("CRC targets must be binary on valid elements and probabilities in [0,1].")
    if not 0.0 < alpha < 1.0:
        raise UncertaintyProtocolError("alpha must be in (0,1).")
    valid_probabilities = probabilities[valid]
    if valid_probabilities.size == 0:
        raise UncertaintyProtocolError("CRC calibration contains no valid elements.")
    values = np.unique(valid_probabilities)
    if len(values) > maximum_candidates:
        values = np.unique(np.quantile(values, np.linspace(0.0, 1.0, maximum_candidates)))
    candidates = np.unique(np.concatenate(([0.0], values, [1.0])))
    chosen: tuple[float, float, float] | None = None
    n = len(probabilities)
    for threshold in candidates:
        empirical = float(np.mean(_per_unit_false_negative_risk(probabilities, targets, float(threshold), valid)))
        corrected = (n / (n + 1.0)) * empirical + 1.0 / (n + 1.0)
        if corrected <= alpha:
            chosen = (float(threshold), empirical, corrected)
    if chosen is None:
        raise UncertaintyProtocolError(
            "No CRC threshold satisfies the requested risk level after finite-sample correction. "
            "The calibration sample is too small for this alpha."
        )
    return ConformalRiskControlModel(
        alpha=float(alpha),
        probability_threshold=chosen[0],
        calibration_empirical_risk=chosen[1],
        calibration_corrected_risk=chosen[2],
        calibration_samples=n,
        risk_name=risk_name,
    )


def crc_audit_rows(
    model: ConformalRiskControlModel,
    test_probabilities: Sequence[Any] | np.ndarray,
    test_targets: Sequence[Any] | np.ndarray,
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    valid_masks: Sequence[Any] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    probabilities = np.asarray(test_probabilities, dtype=float)
    targets = np.asarray(test_targets, dtype=int)
    if probabilities.shape != targets.shape or len(sample_rows) != len(probabilities):
        raise UncertaintyProtocolError("CRC test arrays and sample_rows must align.")
    valid = np.ones_like(targets, dtype=bool) if valid_masks is None else np.asarray(valid_masks, dtype=bool)
    if valid.shape != targets.shape:
        raise UncertaintyProtocolError("CRC valid_masks must align with test arrays.")
    risk = _per_unit_false_negative_risk(probabilities, targets, model.probability_threshold, valid)
    predicted = probabilities >= model.probability_threshold
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sample_rows):
        valid_count = int(np.sum(valid[index]))
        if valid_count == 0:
            raise UncertaintyProtocolError(
                f"CRC test unit {index} contains no valid elements."
            )
        row = dict(source)
        row.update(
            {
                "risk": float(risk[index]),
                model.risk_name: float(risk[index]),
                "risk_target": model.alpha,
                "risk_target_excess": float(max(risk[index] - model.alpha, 0.0)),
                "prediction_set_fraction": float(np.sum(predicted[index] & valid[index]) / valid_count),
                "crc_probability_threshold": model.probability_threshold,
                "crc_calibration_samples": model.calibration_samples,
            }
        )
        rows.append(row)
    return rows


__all__ = [
    "ConformalRiskControlModel",
    "MulticlassConformalModel",
    "SelectiveThreshold",
    "UncertaintyProtocolError",
    "apply_selective_threshold",
    "conformal_quantile",
    "crc_audit_rows",
    "fit_false_negative_crc",
    "fit_multiclass_conformal",
    "fit_selective_threshold",
    "multiclass_conformal_audit_rows",
    "multiclass_nonconformity_scores",
    "multiclass_prediction_sets",
]
