from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_inference import certify_geobwer_from_band, simultaneous_group_risk_band
from rsfm_fairness_audit.bwer_protocol import Validity


CLUSTER_ELIGIBILITY_VERSION = "geobwer_cluster_eligibility_v1"


class ClusterEvidenceLevel(str, Enum):
    NOT_OBSERVED = "not_observed"
    DESCRIPTIVE_ONLY = "descriptive_only"
    INFERENTIAL_ELIGIBLE = "inferential_eligible"


@dataclass(frozen=True)
class ClusterEligibilityRule:
    min_units_per_group: int
    min_clusters_per_group: int
    min_total_clusters: int
    calibration_signature: str = ""
    version: str = CLUSTER_ELIGIBILITY_VERSION

    def __post_init__(self) -> None:
        if min(self.min_units_per_group, self.min_clusters_per_group, self.min_total_clusters) < 1:
            raise ValueError("Cluster eligibility thresholds must be positive.")
        if self.version != CLUSTER_ELIGIBILITY_VERSION:
            raise ValueError(f"Unsupported cluster eligibility version={self.version!r}.")

    @property
    def signature(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GroupClusterEligibility:
    group: str
    unit_count: int
    cluster_count: int
    level: ClusterEvidenceLevel
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ClusterEligibilityAssessment:
    groups: tuple[GroupClusterEligibility, ...]
    total_clusters: int
    rule_signature: str
    inferential_groups: tuple[str, ...]
    descriptive_groups: tuple[str, ...]
    not_observed_groups: tuple[str, ...]


@dataclass(frozen=True)
class ClusterEligibilityCandidate:
    min_clusters_per_group: int
    valid_simulations: int
    simultaneous_coverage: float
    coverage_ci_low: float
    coverage_ci_high: float
    false_positive_rate: float
    false_positive_ci_low: float
    false_positive_ci_high: float
    passes: bool


@dataclass(frozen=True)
class ClusterEligibilityCalibration:
    candidates: tuple[ClusterEligibilityCandidate, ...]
    selected_min_clusters_per_group: int | None
    confidence_level: float
    coverage_tolerance: float
    false_positive_tolerance: float
    n_simulations: int
    n_bootstrap: int
    seed: int
    scenario: str
    signature: str


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def calibrate_cluster_eligibility_rule(
    *,
    candidate_min_clusters: Sequence[int] = (2, 4, 6, 8, 10, 15, 20, 30),
    group_count: int = 10,
    rows_per_cluster: int = 4,
    intracluster_correlation: float = 0.50,
    confidence_level: float = 0.95,
    coverage_tolerance: float = 0.05,
    false_positive_tolerance: float = 0.02,
    n_simulations: int = 500,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> ClusterEligibilityCalibration:
    """Calibrate the lowest per-group cluster support for the max-T engine.

    The frozen stress scenario combines unequal group support, repeated rows
    within clusters, a common null risk, and exact ties.  It calibrates the
    inference engine, not any task's test data.
    """

    candidates = tuple(sorted(set(int(value) for value in candidate_min_clusters)))
    if not candidates or min(candidates) < 2:
        raise ValueError("candidate_min_clusters must contain integers >=2.")
    if group_count < 3 or rows_per_cluster < 1:
        raise ValueError("group_count must be >=3 and rows_per_cluster positive.")
    if not 0.0 <= intracluster_correlation < 1.0:
        raise ValueError("intracluster_correlation must be in [0,1).")
    if n_simulations < 100 or n_bootstrap < 100:
        raise ValueError("Formal calibration requires at least 100 simulations and bootstrap draws.")
    alpha = 1.0 - float(confidence_level)
    coverage_gate = confidence_level - coverage_tolerance
    fpr_gate = alpha + false_positive_tolerance
    records: list[ClusterEligibilityCandidate] = []
    for candidate in candidates:
        coverage_count = 0
        false_positive_count = 0
        valid_count = 0
        for simulation in range(n_simulations):
            rng = np.random.default_rng(seed + 100_003 * candidate + simulation)
            losses: list[float] = []
            groups: list[str] = []
            clusters: list[str] = []
            for group_index in range(group_count):
                # Deterministic heterogeneity tests the advertised minimum in
                # the presence of larger groups without changing the null mean.
                cluster_count = candidate + (group_index % 3) * max(1, candidate // 2)
                for cluster_index in range(cluster_count):
                    cluster_effect = rng.normal()
                    for _ in range(rows_per_cluster):
                        residual = rng.normal()
                        latent = (
                            math.sqrt(intracluster_correlation) * cluster_effect
                            + math.sqrt(1.0 - intracluster_correlation) * residual
                        )
                        # Symmetric clipping preserves the common null mean 0.5.
                        losses.append(float(np.clip(0.5 + 0.18 * latent, 0.0, 1.0)))
                        groups.append(f"g{group_index:02d}")
                        clusters.append(f"g{group_index:02d}_c{cluster_index:04d}")
            band = simultaneous_group_risk_band(
                losses,
                groups,
                clusters,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                seed=seed + simulation,
                min_clusters_per_group=candidate,
            )
            if band.validity != Validity.VALID:
                continue
            valid_count += 1
            lower, upper = dict(band.lower), dict(band.upper)
            coverage_count += int(all(lower[group] <= 0.5 <= upper[group] for group in lower))
            false_positive_count += int(
                certify_geobwer_from_band(band, beta=0.10).lower_confidence_bound > 0.0
            )
        denominator = max(valid_count, 1)
        coverage_ci = _wilson(coverage_count, denominator)
        fpr_ci = _wilson(false_positive_count, denominator)
        records.append(
            ClusterEligibilityCandidate(
                min_clusters_per_group=candidate,
                valid_simulations=valid_count,
                simultaneous_coverage=coverage_count / denominator,
                coverage_ci_low=coverage_ci[0],
                coverage_ci_high=coverage_ci[1],
                false_positive_rate=false_positive_count / denominator,
                false_positive_ci_low=fpr_ci[0],
                false_positive_ci_high=fpr_ci[1],
                passes=(
                    valid_count == n_simulations
                    and coverage_ci[0] >= coverage_gate
                    and fpr_ci[1] <= fpr_gate
                ),
            )
        )
    passing = [record.min_clusters_per_group for record in records if record.passes]
    payload = {
        "candidates": [asdict(record) for record in records],
        "selected_min_clusters_per_group": min(passing) if passing else None,
        "confidence_level": confidence_level,
        "coverage_tolerance": coverage_tolerance,
        "false_positive_tolerance": false_positive_tolerance,
        "n_simulations": n_simulations,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "scenario": "unequal_support_repeated_cluster_rows_common_null_ties_v1",
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ClusterEligibilityCalibration(
        candidates=tuple(records),
        selected_min_clusters_per_group=min(passing) if passing else None,
        confidence_level=float(confidence_level),
        coverage_tolerance=float(coverage_tolerance),
        false_positive_tolerance=float(false_positive_tolerance),
        n_simulations=int(n_simulations),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        scenario=payload["scenario"],
        signature=signature,
    )


def assess_cluster_eligibility(
    unit_support: Mapping[Any, int],
    cluster_support: Mapping[Any, int],
    *,
    total_clusters: int,
    rule: ClusterEligibilityRule,
    required_group_universe: Sequence[Any] | None = None,
) -> ClusterEligibilityAssessment:
    units = {str(key): int(value) for key, value in unit_support.items()}
    clusters = {str(key): int(value) for key, value in cluster_support.items()}
    universe = set(units) | set(clusters)
    if required_group_universe is not None:
        universe |= {str(value) for value in required_group_universe}
    records: list[GroupClusterEligibility] = []
    for group in sorted(universe):
        unit_count = units.get(group, 0)
        cluster_count = clusters.get(group, 0)
        reasons: list[str] = []
        if unit_count == 0:
            level = ClusterEvidenceLevel.NOT_OBSERVED
            reasons.append("zero_observed_units")
        elif unit_count < rule.min_units_per_group:
            level = ClusterEvidenceLevel.DESCRIPTIVE_ONLY
            reasons.append("insufficient_unit_support")
        elif cluster_count < rule.min_clusters_per_group:
            level = ClusterEvidenceLevel.DESCRIPTIVE_ONLY
            reasons.append("insufficient_within_group_clusters")
        elif int(total_clusters) < rule.min_total_clusters and not rule.calibration_signature:
            level = ClusterEvidenceLevel.DESCRIPTIVE_ONLY
            reasons.append("small_cluster_design_not_calibrated")
        else:
            level = ClusterEvidenceLevel.INFERENTIAL_ELIGIBLE
        records.append(
            GroupClusterEligibility(
                group=group,
                unit_count=unit_count,
                cluster_count=cluster_count,
                level=level,
                reasons=tuple(reasons),
            )
        )
    return ClusterEligibilityAssessment(
        groups=tuple(records),
        total_clusters=int(total_clusters),
        rule_signature=rule.signature,
        inferential_groups=tuple(record.group for record in records if record.level == ClusterEvidenceLevel.INFERENTIAL_ELIGIBLE),
        descriptive_groups=tuple(record.group for record in records if record.level == ClusterEvidenceLevel.DESCRIPTIVE_ONLY),
        not_observed_groups=tuple(record.group for record in records if record.level == ClusterEvidenceLevel.NOT_OBSERVED),
    )
