from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_core import compute_geobwer


@dataclass(frozen=True)
class ClusterUncertaintyResult:
    method: str
    alpha: float
    threshold: float
    calibration_cluster_count: int
    test_cluster_count: int
    marginal_coverage: float
    cluster_mean_coverage: float
    worst_cluster_coverage: float
    mean_set_size: float
    set_size_fraction: float
    mean_cluster_miscoverage: float
    tail_cluster_miscoverage: float
    cluster_miscoverage_geobwer: float
    calibration_risk_ucb: float | None
    evidence_status: str
    guarantee_scope: str


def _validate(
    probabilities: np.ndarray, targets: np.ndarray, clusters: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets, dtype=int)
    c = np.asarray([str(value) for value in clusters], dtype=object)
    if p.ndim != 2 or p.shape[0] != len(y) or len(y) != len(c) or p.shape[1] < 2:
        raise ValueError("Multiclass probabilities, targets and clusters are not aligned.")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("Probabilities must be finite and in [0,1].")
    if not np.allclose(p.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4):
        raise ValueError("Probability rows must sum to one.")
    if np.any(y < 0) or np.any(y >= p.shape[1]) or any(not value for value in c):
        raise ValueError("Targets and cluster IDs are invalid.")
    return p, y, c


def _higher_quantile(values: np.ndarray, level: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(len(ordered), max(1, int(math.ceil(level * len(ordered)))))
    return float(ordered[rank - 1])


def _evaluate_lac(
    probabilities: np.ndarray,
    targets: np.ndarray,
    clusters: np.ndarray,
    *, threshold: float, alpha: float, method: str,
    calibration_clusters: int, calibration_risk_ucb: float | None,
    evidence_status: str, guarantee_scope: str,
) -> ClusterUncertaintyResult:
    sets = (1.0 - probabilities) <= float(threshold)
    covered = sets[np.arange(len(targets)), targets]
    cluster_risks = {
        name: float(1.0 - np.mean(covered[clusters == name]))
        for name in sorted(set(clusters.tolist()))
    }
    card = compute_geobwer(cluster_risks, beta=0.10)
    return ClusterUncertaintyResult(
        method=method, alpha=float(alpha), threshold=float(threshold),
        calibration_cluster_count=int(calibration_clusters), test_cluster_count=len(cluster_risks),
        marginal_coverage=float(np.mean(covered)),
        cluster_mean_coverage=float(1.0 - np.mean(list(cluster_risks.values()))),
        worst_cluster_coverage=float(1.0 - max(cluster_risks.values())),
        mean_set_size=float(np.mean(sets.sum(axis=1))),
        set_size_fraction=float(np.mean(sets.sum(axis=1)) / probabilities.shape[1]),
        mean_cluster_miscoverage=card.mean_risk, tail_cluster_miscoverage=card.tail_risk,
        cluster_miscoverage_geobwer=card.bwer,
        calibration_risk_ucb=calibration_risk_ucb,
        evidence_status=evidence_status, guarantee_scope=guarantee_scope,
    )


def cluster_max_lac(
    calibration_probabilities: np.ndarray,
    calibration_targets: np.ndarray,
    calibration_clusters: Sequence[Any],
    test_probabilities: np.ndarray,
    test_targets: np.ndarray,
    test_clusters: Sequence[Any],
    *, alpha: float = 0.10, cluster_design_valid: bool = True,
    min_calibration_clusters: int = 20,
) -> ClusterUncertaintyResult:
    """Split conformal using one maximum nonconformity score per cluster.

    Under exchangeable independent clusters this targets simultaneous coverage
    of every row in a new cluster.  It is intentionally conservative and never
    falls back to the full class set for unsupported local neighborhoods.
    """

    cp, cy, cc = _validate(calibration_probabilities, calibration_targets, calibration_clusters)
    tp, ty, tc = _validate(test_probabilities, test_targets, test_clusters)
    if set(cc.tolist()) & set(tc.tolist()):
        raise ValueError("Calibration and test clusters must be disjoint.")
    row_scores = 1.0 - cp[np.arange(len(cy)), cy]
    names = sorted(set(cc.tolist()))
    maxima = np.asarray([np.max(row_scores[cc == name]) for name in names], dtype=float)
    level = min(1.0, math.ceil((len(maxima) + 1) * (1.0 - alpha)) / len(maxima))
    threshold = _higher_quantile(maxima, level)
    enough_clusters = len(names) >= int(min_calibration_clusters)
    status = "formal_confirmed" if cluster_design_valid and enough_clusters else (
        "not_identified" if cluster_design_valid else "descriptive_only"
    )
    scope = (
        "finite-sample simultaneous-within-new-cluster coverage under exchangeable independent clusters"
        if cluster_design_valid else
        "empirical cluster-aware comparator; cluster independence/exchangeability not certified"
    )
    return _evaluate_lac(tp, ty, tc, threshold=threshold, alpha=alpha,
        method="cluster_max_lac", calibration_clusters=len(names), calibration_risk_ucb=None,
        evidence_status=status, guarantee_scope=scope)


def cluster_hoeffding_crc_lac(
    calibration_probabilities: np.ndarray,
    calibration_targets: np.ndarray,
    calibration_clusters: Sequence[Any],
    test_probabilities: np.ndarray,
    test_targets: np.ndarray,
    test_clusters: Sequence[Any],
    *, alpha: float = 0.10, delta: float = 0.05, cluster_design_valid: bool = True,
    min_calibration_clusters: int = 20,
) -> ClusterUncertaintyResult:
    """Choose the smallest LAC set whose mean cluster risk has a valid UCB."""

    cp, cy, cc = _validate(calibration_probabilities, calibration_targets, calibration_clusters)
    tp, ty, tc = _validate(test_probabilities, test_targets, test_clusters)
    if set(cc.tolist()) & set(tc.tolist()):
        raise ValueError("Calibration and test clusters must be disjoint.")
    scores = 1.0 - cp[np.arange(len(cy)), cy]
    names = sorted(set(cc.tolist())); radius = math.sqrt(math.log(1.0 / delta) / (2.0 * len(names)))
    # Mean cluster risk is a weighted empirical survival function.  Giving
    # each row weight 1/(G*n_g) is exactly equivalent to averaging the G
    # within-cluster risks, but avoids an O(n^2) threshold scan.
    counts = {name: int(np.sum(cc == name)) for name in names}
    weights = np.asarray([1.0 / (len(names) * counts[str(name)]) for name in cc], dtype=float)
    order = np.argsort(scores, kind="mergesort")
    ordered_scores, ordered_weights = scores[order], weights[order]
    selected = None; selected_ucb = None
    cumulative = 0.0
    index = 0
    while index < len(ordered_scores):
        threshold = float(ordered_scores[index])
        stop = index
        while stop < len(ordered_scores) and ordered_scores[stop] == threshold:
            cumulative += float(ordered_weights[stop]); stop += 1
        ucb = min(1.0, float(max(0.0, 1.0 - cumulative) + radius))
        if ucb <= alpha:
            selected, selected_ucb = float(threshold), ucb
            break
        index = stop
    if selected is None:
        selected, selected_ucb = 1.0, min(1.0, radius)
        identified = False
    else:
        identified = True
    enough_clusters = len(names) >= int(min_calibration_clusters)
    status = "formal_confirmed" if cluster_design_valid and identified and enough_clusters else (
        "not_identified" if cluster_design_valid else "descriptive_only"
    )
    scope = (
        "high-probability mean new-cluster risk control under independent exchangeable clusters"
        if cluster_design_valid else
        "empirical cluster-aware comparator; cluster independence/exchangeability not certified"
    )
    return _evaluate_lac(tp, ty, tc, threshold=selected, alpha=alpha,
        method="cluster_hoeffding_crc_lac", calibration_clusters=len(names),
        calibration_risk_ucb=selected_ucb, evidence_status=status, guarantee_scope=scope)
