from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_core import BWERPointEstimate, compute_geobwer, fractional_tail_allocation, normalize_deployment_weights
from rsfm_fairness_audit.bwer_protocol import Validity


@dataclass(frozen=True)
class SimultaneousRiskBand:
    validity: Validity
    estimates: tuple[tuple[str, float], ...]
    standard_errors: tuple[tuple[str, float], ...]
    lower: tuple[tuple[str, float], ...]
    upper: tuple[tuple[str, float], ...]
    critical_value: float
    confidence_level: float
    cluster_count: int
    clusters_per_group: tuple[tuple[str, int], ...]
    bootstrap_replicates: int
    risk_lower_bound: float | None = None
    risk_upper_bound: float | None = None
    message: str = ""

    def estimate_dict(self) -> dict[str, float]:
        return dict(self.estimates)

    def half_width_dict(self) -> dict[str, float]:
        lo = dict(self.lower)
        hi = dict(self.upper)
        return {group: max(self.estimate_dict()[group] - lo[group], hi[group] - self.estimate_dict()[group]) for group in self.estimate_dict()}


@dataclass(frozen=True)
class CertifiedBWER:
    validity: Validity
    point: BWERPointEstimate
    ci_low: float
    ci_high: float
    lower_confidence_bound: float
    radius: float
    weighted_sum_radius: float
    total_variation_radius: float
    radius_method: str
    parameter_lower_bound: float
    parameter_upper_bound: float
    confidence_level: float
    band: SimultaneousRiskBand
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        output = self.point.to_dict()
        output.update(
            {
                "validity": self.validity.value,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "lower_confidence_bound": self.lower_confidence_bound,
                "certification_radius": self.radius,
                "weighted_sum_radius": self.weighted_sum_radius,
                "total_variation_radius": self.total_variation_radius,
                "certification_radius_method": self.radius_method,
                "parameter_lower_bound": self.parameter_lower_bound,
                "parameter_upper_bound": self.parameter_upper_bound,
                "confidence_level": self.confidence_level,
                "cluster_count": self.band.cluster_count,
                "critical_value": self.band.critical_value,
                "message": self.message,
            }
        )
        return output


@dataclass(frozen=True)
class PairedBWERComparison:
    validity: Validity
    model_a: str
    model_b: str
    delta_bwer: float
    ci_low: float
    ci_high: float
    direct_multiplier_ci_low: float
    direct_multiplier_ci_high: float
    common_groups: tuple[str, ...]
    common_units: int
    cluster_count: int
    confidence_level: float
    message: str = ""


@dataclass(frozen=True)
class HonestConfirmationFold:
    discovery_partition: str
    confirmation_partition: str
    discovery_clusters: int
    confirmation_clusters: int
    selected_mass: tuple[tuple[str, float], ...]
    discovery_apparent_bwer: float
    confirmed_tail_risk: float
    confirmed_mean_risk: float
    confirmed_contrast: float


@dataclass(frozen=True)
class HonestConfirmedBWER:
    """Cross-fitted tail confirmation with selection and evaluation separated.

    The confirmed contrast is deliberately allowed to be negative. Truncating it
    at zero would hide failed confirmation and would reintroduce an optimistic
    post-selection interpretation.
    """

    validity: Validity
    beta: float
    cross_fitted_contrast: float
    both_directions_positive: bool
    groups: tuple[str, ...]
    folds: tuple[HonestConfirmationFold, ...]
    seed: int
    message: str = ""


def _prepare_cluster_influence(
    losses: Sequence[float],
    groups: Sequence[Any],
    clusters: Sequence[Any],
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(losses, dtype=float)
    g = np.asarray([str(value) for value in groups], dtype=object)
    c = np.asarray([str(value) for value in clusters], dtype=object)
    if not (len(y) == len(g) == len(c)) or len(y) == 0:
        raise ValueError("losses, groups, and clusters must be non-empty and aligned.")
    if not np.all(np.isfinite(y)):
        raise ValueError("losses must be finite.")
    group_names = tuple(sorted(set(g.tolist())))
    cluster_names = tuple(sorted(set(c.tolist())))
    if len(group_names) < 2:
        raise ValueError("At least two groups are required.")
    if len(cluster_names) < 2:
        raise ValueError("At least two clusters are required.")
    estimates = np.zeros(len(group_names), dtype=float)
    influence = np.zeros((len(cluster_names), len(group_names)), dtype=float)
    cluster_counts = np.zeros(len(group_names), dtype=int)
    cluster_index = {name: index for index, name in enumerate(cluster_names)}
    for group_index, group in enumerate(group_names):
        mask = g == group
        n_group = int(np.sum(mask))
        if n_group == 0:
            raise RuntimeError("Internal group support error.")
        estimate = float(np.mean(y[mask]))
        estimates[group_index] = estimate
        observed_clusters = set(c[mask].tolist())
        cluster_counts[group_index] = len(observed_clusters)
        for value, cluster in zip(y[mask], c[mask]):
            influence[cluster_index[str(cluster)], group_index] += (float(value) - estimate) / n_group
    correction = math.sqrt(len(cluster_names) / (len(cluster_names) - 1))
    influence *= correction
    return group_names, cluster_names, estimates, influence, cluster_counts


def _multiplier_draws(
    influence: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    multiplier: str,
) -> tuple[np.ndarray, np.ndarray]:
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100 for a formal simultaneous band.")
    rng = np.random.default_rng(seed)
    if multiplier == "rademacher":
        weights = rng.choice(np.asarray([-1.0, 1.0]), size=(n_bootstrap, influence.shape[0]))
    elif multiplier == "normal":
        weights = rng.normal(size=(n_bootstrap, influence.shape[0]))
    else:
        raise ValueError("multiplier must be rademacher or normal.")
    # GeoFM group-by-cluster influence matrices are usually very sparse: a
    # country/event occupies only a small fraction of global blocks. Dense
    # BLAS multiplies every zero and makes validation-layout Monte Carlo
    # needlessly expensive. Preserve the exact calculation while exploiting
    # sparse columns when fewer than 35% of entries are non-zero.
    nonzero_fraction = float(np.count_nonzero(influence) / max(influence.size, 1))
    if nonzero_fraction < 0.35:
        perturbations = np.zeros((n_bootstrap, influence.shape[1]), dtype=float)
        for column in range(influence.shape[1]):
            active = np.flatnonzero(influence[:, column])
            if len(active):
                perturbations[:, column] = weights[:, active] @ influence[active, column]
    else:
        perturbations = weights @ influence
    standard_errors = np.sqrt(np.sum(influence * influence, axis=0))
    return perturbations, standard_errors


def simultaneous_group_risk_band(
    losses: Sequence[float],
    groups: Sequence[Any],
    clusters: Sequence[Any],
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
    multiplier: str = "rademacher",
    risk_bounds: tuple[float, float] | None = (0.0, 1.0),
    min_clusters_per_group: int = 2,
) -> SimultaneousRiskBand:
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")
    names, cluster_names, estimates, influence, cluster_counts = _prepare_cluster_influence(losses, groups, clusters)
    if int(np.min(cluster_counts)) < int(min_clusters_per_group):
        return SimultaneousRiskBand(
            validity=Validity.INSUFFICIENT_INDEPENDENT_UNITS,
            estimates=tuple(zip(names, estimates.tolist())),
            standard_errors=(),
            lower=(),
            upper=(),
            critical_value=float("nan"),
            confidence_level=float(confidence_level),
            cluster_count=len(cluster_names),
            clusters_per_group=tuple(zip(names, cluster_counts.tolist())),
            bootstrap_replicates=0,
            message=f"At least one group has fewer than {min_clusters_per_group} independent clusters.",
        )
    perturbations, standard_errors = _multiplier_draws(influence, n_bootstrap=n_bootstrap, seed=seed, multiplier=multiplier)
    active = standard_errors > np.finfo(float).eps
    if np.any(active):
        studentized = np.abs(perturbations[:, active] / standard_errors[active])
        max_statistics = np.max(studentized, axis=1)
        critical = float(np.quantile(max_statistics, confidence_level, method="higher"))
    else:
        critical = 0.0
    half_width = critical * standard_errors
    lower = estimates - half_width
    upper = estimates + half_width
    if risk_bounds is not None:
        risk_low, risk_high = map(float, risk_bounds)
        lower = np.maximum(lower, risk_low)
        upper = np.minimum(upper, risk_high)
    return SimultaneousRiskBand(
        validity=Validity.VALID,
        estimates=tuple(zip(names, estimates.tolist())),
        standard_errors=tuple(zip(names, standard_errors.tolist())),
        lower=tuple(zip(names, lower.tolist())),
        upper=tuple(zip(names, upper.tolist())),
        critical_value=critical,
        confidence_level=float(confidence_level),
        cluster_count=len(cluster_names),
        clusters_per_group=tuple(zip(names, cluster_counts.tolist())),
        bootstrap_replicates=n_bootstrap,
        risk_lower_bound=None if risk_bounds is None else float(risk_bounds[0]),
        risk_upper_bound=None if risk_bounds is None else float(risk_bounds[1]),
    )


def simultaneous_standardized_risk_band(
    losses: Sequence[float],
    groups: Sequence[Any],
    balance_levels: Sequence[Any],
    clusters: Sequence[Any],
    *,
    target_weights: Mapping[Any, float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
    multiplier: str = "rademacher",
    risk_bounds: tuple[float, float] | None = (0.0, 1.0),
    min_clusters_per_group: int = 2,
) -> SimultaneousRiskBand:
    """Simultaneous band for a strict common-composition standardized risk."""

    y = np.asarray(losses, dtype=float)
    g = np.asarray([str(value) for value in groups], dtype=object)
    z = np.asarray([str(value) for value in balance_levels], dtype=object)
    c = np.asarray([str(value) for value in clusters], dtype=object)
    if not (len(y) == len(g) == len(z) == len(c)) or len(y) == 0:
        raise ValueError("losses, groups, balance_levels, and clusters must be non-empty and aligned.")
    if not np.all(np.isfinite(y)):
        raise ValueError("losses must be finite.")
    group_names = tuple(sorted(set(g.tolist())))
    cluster_names = tuple(sorted(set(c.tolist())))
    raw_weights = {str(key): float(value) for key, value in target_weights.items()}
    levels = tuple(sorted(raw_weights))
    total = sum(raw_weights.values())
    if total <= 0.0 or any(not math.isfinite(value) or value < 0.0 for value in raw_weights.values()):
        raise ValueError("target_weights must be finite, non-negative, and have positive mass.")
    weights = {level: raw_weights[level] / total for level in levels}
    if len(group_names) < 2 or len(cluster_names) < 2:
        raise ValueError("At least two groups and two clusters are required.")
    cluster_index = {name: index for index, name in enumerate(cluster_names)}
    influence = np.zeros((len(cluster_names), len(group_names)), dtype=float)
    estimates = np.zeros(len(group_names), dtype=float)
    cluster_counts = np.zeros(len(group_names), dtype=int)
    for group_index, group in enumerate(group_names):
        group_clusters: set[str] = set()
        for level in levels:
            mask = (g == group) & (z == level)
            n_cell = int(np.sum(mask))
            if n_cell == 0:
                return SimultaneousRiskBand(
                    validity=Validity.NOT_IDENTIFIED,
                    estimates=(),
                    standard_errors=(),
                    lower=(),
                    upper=(),
                    critical_value=float("nan"),
                    confidence_level=float(confidence_level),
                    cluster_count=len(cluster_names),
                    clusters_per_group=(),
                    bootstrap_replicates=0,
                    message=f"Missing strict standardization cell: group={group}, balance={level}.",
                )
            cell_mean = float(np.mean(y[mask]))
            estimates[group_index] += weights[level] * cell_mean
            group_clusters.update(str(value) for value in c[mask])
            for value, cluster in zip(y[mask], c[mask]):
                influence[cluster_index[str(cluster)], group_index] += weights[level] * (float(value) - cell_mean) / n_cell
        cluster_counts[group_index] = len(group_clusters)
    correction = math.sqrt(len(cluster_names) / (len(cluster_names) - 1))
    influence *= correction
    if int(np.min(cluster_counts)) < int(min_clusters_per_group):
        return SimultaneousRiskBand(
            validity=Validity.INSUFFICIENT_INDEPENDENT_UNITS,
            estimates=tuple(zip(group_names, estimates.tolist())),
            standard_errors=(),
            lower=(),
            upper=(),
            critical_value=float("nan"),
            confidence_level=float(confidence_level),
            cluster_count=len(cluster_names),
            clusters_per_group=tuple(zip(group_names, cluster_counts.tolist())),
            bootstrap_replicates=0,
            message=f"At least one group has fewer than {min_clusters_per_group} independent clusters.",
        )
    perturbations, standard_errors = _multiplier_draws(influence, n_bootstrap=n_bootstrap, seed=seed, multiplier=multiplier)
    active = standard_errors > np.finfo(float).eps
    critical = 0.0
    if np.any(active):
        max_statistics = np.max(np.abs(perturbations[:, active] / standard_errors[active]), axis=1)
        critical = float(np.quantile(max_statistics, confidence_level, method="higher"))
    half_width = critical * standard_errors
    lower = estimates - half_width
    upper = estimates + half_width
    if risk_bounds is not None:
        risk_low, risk_high = map(float, risk_bounds)
        lower = np.maximum(lower, risk_low)
        upper = np.minimum(upper, risk_high)
    return SimultaneousRiskBand(
        validity=Validity.VALID,
        estimates=tuple(zip(group_names, estimates.tolist())),
        standard_errors=tuple(zip(group_names, standard_errors.tolist())),
        lower=tuple(zip(group_names, lower.tolist())),
        upper=tuple(zip(group_names, upper.tolist())),
        critical_value=critical,
        confidence_level=float(confidence_level),
        cluster_count=len(cluster_names),
        clusters_per_group=tuple(zip(group_names, cluster_counts.tolist())),
        bootstrap_replicates=n_bootstrap,
        risk_lower_bound=None if risk_bounds is None else float(risk_bounds[0]),
        risk_upper_bound=None if risk_bounds is None else float(risk_bounds[1]),
    )


def certify_geobwer_from_band(
    band: SimultaneousRiskBand,
    *,
    beta: float,
    deployment_weights: Mapping[Any, float] | None = None,
) -> CertifiedBWER:
    if band.validity != Validity.VALID:
        raise ValueError(f"Cannot certify GeoBWER from invalid band: {band.validity.value}")
    estimates = band.estimate_dict()
    weights = normalize_deployment_weights(tuple(estimates), deployment_weights)
    point = compute_geobwer(estimates, beta, weights)
    errors = band.half_width_dict()
    tail_error, _ = fractional_tail_allocation(errors, beta, weights)
    mean_error = float(sum(weights[group] * errors[group] for group in estimates))
    weighted_sum_radius = float(tail_error + mean_error)
    # Risk-envelope geometry gives ||q-mu||_1 <= 2(1-beta) for every
    # q_g <= mu_g / beta. Combining this with the simultaneous coordinate
    # band yields a second valid Lipschitz radius. Taking the smaller of two
    # valid radii preserves coverage and can be much sharper near beta=1.
    total_variation_radius = float(2.0 * (1.0 - float(beta)) * max(errors.values(), default=0.0))
    radius = min(weighted_sum_radius, total_variation_radius)
    radius_method = (
        "total_variation_envelope"
        if total_variation_radius < weighted_sum_radius
        else "weighted_tail_plus_mean"
    )
    parameter_lower_bound = 0.0
    if band.risk_lower_bound is not None and band.risk_upper_bound is not None:
        risk_range = max(0.0, band.risk_upper_bound - band.risk_lower_bound)
        parameter_upper_bound = float((1.0 - float(beta)) * risk_range)
    else:
        parameter_upper_bound = float("inf")
    lower = max(parameter_lower_bound, point.bwer - radius)
    upper = min(parameter_upper_bound, point.bwer + radius)
    return CertifiedBWER(
        validity=Validity.VALID,
        point=point,
        ci_low=lower,
        ci_high=upper,
        lower_confidence_bound=lower,
        radius=radius,
        weighted_sum_radius=weighted_sum_radius,
        total_variation_radius=total_variation_radius,
        radius_method=radius_method,
        parameter_lower_bound=parameter_lower_bound,
        parameter_upper_bound=parameter_upper_bound,
        confidence_level=band.confidence_level,
        band=band,
    )


def certified_geobwer(
    losses: Sequence[float],
    groups: Sequence[Any],
    clusters: Sequence[Any],
    *,
    beta: float = 0.10,
    deployment_weights: Mapping[Any, float] | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
    multiplier: str = "rademacher",
    min_clusters_per_group: int = 2,
) -> CertifiedBWER:
    band = simultaneous_group_risk_band(
        losses,
        groups,
        clusters,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
        multiplier=multiplier,
        min_clusters_per_group=min_clusters_per_group,
    )
    if band.validity != Validity.VALID:
        estimates = band.estimate_dict()
        point = compute_geobwer(estimates, beta, deployment_weights)
        return CertifiedBWER(
            validity=band.validity,
            point=point,
            ci_low=float("nan"),
            ci_high=float("nan"),
            lower_confidence_bound=float("nan"),
            radius=float("nan"),
            weighted_sum_radius=float("nan"),
            total_variation_radius=float("nan"),
            radius_method="not_certified",
            parameter_lower_bound=0.0,
            parameter_upper_bound=float("nan"),
            confidence_level=confidence_level,
            band=band,
            message=band.message,
        )
    return certify_geobwer_from_band(band, beta=beta, deployment_weights=deployment_weights)


def paired_bwer_comparison(
    losses_a: Sequence[float],
    losses_b: Sequence[float],
    groups: Sequence[Any],
    clusters: Sequence[Any],
    *,
    model_a: str = "model_a",
    model_b: str = "model_b",
    beta: float = 0.10,
    deployment_weights: Mapping[Any, float] | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
    multiplier: str = "rademacher",
) -> PairedBWERComparison:
    if not (len(losses_a) == len(losses_b) == len(groups) == len(clusters)) or len(losses_a) == 0:
        raise ValueError("Paired inputs must be non-empty and aligned.")
    names_a, clusters_a, estimates_a, influence_a, counts_a = _prepare_cluster_influence(losses_a, groups, clusters)
    names_b, clusters_b, estimates_b, influence_b, counts_b = _prepare_cluster_influence(losses_b, groups, clusters)
    if names_a != names_b or clusters_a != clusters_b:
        raise RuntimeError("Paired influence matrices did not align.")
    influence = np.concatenate([influence_a, influence_b], axis=1)
    perturbations, standard_errors = _multiplier_draws(influence, n_bootstrap=n_bootstrap, seed=seed, multiplier=multiplier)
    active = standard_errors > np.finfo(float).eps
    critical = 0.0
    if np.any(active):
        max_statistics = np.max(np.abs(perturbations[:, active] / standard_errors[active]), axis=1)
        critical = float(np.quantile(max_statistics, confidence_level, method="higher"))
    width_a = critical * standard_errors[: len(names_a)]
    width_b = critical * standard_errors[len(names_a) :]
    risks_a = dict(zip(names_a, estimates_a.tolist()))
    risks_b = dict(zip(names_b, estimates_b.tolist()))
    weights = normalize_deployment_weights(names_a, deployment_weights)
    point_a = compute_geobwer(risks_a, beta, weights)
    point_b = compute_geobwer(risks_b, beta, weights)
    delta = point_a.bwer - point_b.bwer
    tail_a, _ = fractional_tail_allocation(dict(zip(names_a, width_a.tolist())), beta, weights)
    tail_b, _ = fractional_tail_allocation(dict(zip(names_b, width_b.tolist())), beta, weights)
    radius_a = tail_a + sum(weights[group] * width_a[index] for index, group in enumerate(names_a))
    radius_b = tail_b + sum(weights[group] * width_b[index] for index, group in enumerate(names_b))
    formal_radius = float(radius_a + radius_b)
    direct_deltas = np.empty(n_bootstrap, dtype=float)
    for replicate in range(n_bootstrap):
        perturbed_a = {group: float(np.clip(estimates_a[index] + perturbations[replicate, index], 0.0, 1.0)) for index, group in enumerate(names_a)}
        perturbed_b = {
            group: float(np.clip(estimates_b[index] + perturbations[replicate, len(names_a) + index], 0.0, 1.0))
            for index, group in enumerate(names_b)
        }
        direct_deltas[replicate] = compute_geobwer(perturbed_a, beta, weights).bwer - compute_geobwer(perturbed_b, beta, weights).bwer
    alpha = 1.0 - confidence_level
    direct_low, direct_high = np.quantile(direct_deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    validity = Validity.VALID
    message = "Formal CI propagates a joint simultaneous group-risk band; direct multiplier CI is a non-smooth sensitivity interval."
    if min(int(np.min(counts_a)), int(np.min(counts_b))) < 2:
        validity = Validity.INSUFFICIENT_INDEPENDENT_UNITS
        message = "At least one group has fewer than two paired clusters."
    return PairedBWERComparison(
        validity=validity,
        model_a=model_a,
        model_b=model_b,
        delta_bwer=float(delta),
        ci_low=float(delta - formal_radius),
        ci_high=float(delta + formal_radius),
        direct_multiplier_ci_low=float(direct_low),
        direct_multiplier_ci_high=float(direct_high),
        common_groups=names_a,
        common_units=len(losses_a),
        cluster_count=len(clusters_a),
        confidence_level=float(confidence_level),
        message=message,
    )


def honest_confirmed_bwer(
    losses: Sequence[float],
    groups: Sequence[Any],
    clusters: Sequence[Any],
    *,
    beta: float = 0.10,
    deployment_weights: Mapping[Any, float] | None = None,
    seed: int = 42,
    min_clusters_per_group_per_partition: int = 2,
) -> HonestConfirmedBWER:
    """Confirm a discovered bad tail on data not used to select that tail.

    Independent clusters are randomly assigned to two frozen partitions.  Each
    direction selects an exact fractional tail using one partition and evaluates
    the same tail allocation on the other.  The symmetric mean is a cross-fitted
    diagnostic, while the two directional contrasts remain visible so a single
    favorable split cannot be presented as confirmation.

    This estimand requires every audited group to be estimable in both
    partitions.  It is therefore not identified when a group is itself a single
    independent unit (for example, an event-level slice with one event).
    """

    y = np.asarray(losses, dtype=float)
    g = np.asarray([str(value) for value in groups], dtype=object)
    c = np.asarray([str(value) for value in clusters], dtype=object)
    if not (len(y) == len(g) == len(c)) or len(y) == 0:
        raise ValueError("losses, groups, and clusters must be non-empty and aligned.")
    if not np.all(np.isfinite(y)):
        raise ValueError("losses must be finite.")
    if min_clusters_per_group_per_partition < 1:
        raise ValueError("min_clusters_per_group_per_partition must be positive.")
    group_names = tuple(sorted(set(g.tolist())))
    cluster_names = np.asarray(sorted(set(c.tolist())), dtype=object)
    if len(group_names) < 2:
        raise ValueError("At least two groups are required.")
    if len(cluster_names) < 2:
        return HonestConfirmedBWER(
            validity=Validity.INSUFFICIENT_INDEPENDENT_UNITS,
            beta=float(beta),
            cross_fitted_contrast=float("nan"),
            both_directions_positive=False,
            groups=group_names,
            folds=(),
            seed=seed,
            message="At least two independent clusters are required for honest confirmation.",
        )
    weights = normalize_deployment_weights(group_names, deployment_weights)
    rng = np.random.default_rng(seed)
    shuffled = cluster_names.copy()
    rng.shuffle(shuffled)
    partition_a = set(str(value) for value in shuffled[::2])
    partition_b = set(str(value) for value in shuffled[1::2])
    partitions = (("A", partition_a), ("B", partition_b))
    if not partition_a or not partition_b:
        return HonestConfirmedBWER(
            validity=Validity.INSUFFICIENT_INDEPENDENT_UNITS,
            beta=float(beta),
            cross_fitted_contrast=float("nan"),
            both_directions_positive=False,
            groups=group_names,
            folds=(),
            seed=seed,
            message="Both confirmation partitions must contain independent clusters.",
        )

    def group_risks_for(cluster_set: set[str]) -> tuple[dict[str, float], dict[str, int]] | None:
        mask_partition = np.asarray([str(value) in cluster_set for value in c], dtype=bool)
        risks: dict[str, float] = {}
        cluster_support: dict[str, int] = {}
        for group in group_names:
            mask = mask_partition & (g == group)
            support = len(set(str(value) for value in c[mask]))
            cluster_support[group] = support
            if support < min_clusters_per_group_per_partition or not np.any(mask):
                return None
            risks[group] = float(np.mean(y[mask]))
        return risks, cluster_support

    risks_by_partition: dict[str, dict[str, float]] = {}
    supports_by_partition: dict[str, dict[str, int]] = {}
    for name, cluster_set in partitions:
        result = group_risks_for(cluster_set)
        if result is None:
            return HonestConfirmedBWER(
                validity=Validity.NOT_IDENTIFIED,
                beta=float(beta),
                cross_fitted_contrast=float("nan"),
                both_directions_positive=False,
                groups=group_names,
                folds=(),
                seed=seed,
                message=(
                    "Every group must have at least "
                    f"{min_clusters_per_group_per_partition} independent clusters in both frozen partitions. "
                    "Use the fixed-slice simultaneous-band analysis when group and independent unit coincide."
                ),
            )
        risks_by_partition[name], supports_by_partition[name] = result

    folds: list[HonestConfirmationFold] = []
    for discovery_name, confirmation_name in (("A", "B"), ("B", "A")):
        discovery_risks = risks_by_partition[discovery_name]
        confirmation_risks = risks_by_partition[confirmation_name]
        discovery_point = compute_geobwer(discovery_risks, beta, weights)
        selected = dict(discovery_point.allocation.selected_mass)
        confirmed_tail = float(sum(selected[group] * confirmation_risks[group] for group in group_names) / float(beta))
        confirmed_mean = float(sum(weights[group] * confirmation_risks[group] for group in group_names))
        confirmed_contrast = confirmed_tail - confirmed_mean
        folds.append(
            HonestConfirmationFold(
                discovery_partition=discovery_name,
                confirmation_partition=confirmation_name,
                discovery_clusters=len(partition_a if discovery_name == "A" else partition_b),
                confirmation_clusters=len(partition_b if confirmation_name == "B" else partition_a),
                selected_mass=tuple(sorted(selected.items())),
                discovery_apparent_bwer=discovery_point.bwer,
                confirmed_tail_risk=confirmed_tail,
                confirmed_mean_risk=confirmed_mean,
                confirmed_contrast=confirmed_contrast,
            )
        )
    contrast = float(np.mean([fold.confirmed_contrast for fold in folds]))
    return HonestConfirmedBWER(
        validity=Validity.VALID,
        beta=float(beta),
        cross_fitted_contrast=contrast,
        both_directions_positive=all(fold.confirmed_contrast > 0.0 for fold in folds),
        groups=group_names,
        folds=tuple(folds),
        seed=seed,
        message=(
            "The cross-fitted contrast is a post-selection diagnostic. Primary formal claims should still use "
            "the pre-specified fixed-slice simultaneous confidence band."
        ),
    )


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = (np.radians(lon2 - lon1) + np.pi) % (2.0 * np.pi) - np.pi
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def equal_area_block_ids(latitudes: Sequence[float], longitudes: Sequence[float], *, cell_km: float) -> list[str]:
    """Assign deterministic Lambert cylindrical equal-area cells.

    The cells are equal-area in the cylindrical coordinate plane and wrap at
    the antimeridian. They are a dependency-free formal fallback; H3/S2 IDs may
    be supplied directly as clusters when those libraries are available.
    """

    if float(cell_km) <= 0.0:
        raise ValueError("cell_km must be positive.")
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    if lat.shape != lon.shape or lat.ndim != 1 or not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("latitudes and longitudes must be aligned finite vectors.")
    if np.any(lat < -90.0) or np.any(lat > 90.0):
        raise ValueError("latitudes must be in [-90, 90].")
    wrapped_lon = (lon + 180.0) % 360.0 - 180.0
    x = EARTH_RADIUS_KM * np.radians(wrapped_lon)
    y = EARTH_RADIUS_KM * np.sin(np.radians(lat))
    x_origin = -math.pi * EARTH_RADIUS_KM
    y_origin = -EARTH_RADIUS_KM
    ix = np.floor((x - x_origin) / float(cell_km)).astype(np.int64)
    iy = np.floor((y - y_origin) / float(cell_km)).astype(np.int64)
    scale = f"{float(cell_km):.3f}".rstrip("0").rstrip(".")
    return [f"cea_{scale}km_{x_value}_{y_value}" for x_value, y_value in zip(ix, iy)]


@dataclass(frozen=True)
class SpatialRangeEstimate:
    range_km: float
    near_correlation: float
    pair_count: int
    distance_bins_km: tuple[float, ...]
    correlations: tuple[float, ...]


def estimate_spatial_correlation_range(
    losses: Sequence[float],
    groups: Sequence[Any],
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    *,
    max_pairs: int = 100_000,
    bins: int = 12,
    seed: int = 42,
) -> SpatialRangeEstimate:
    y = np.asarray(losses, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    g = np.asarray([str(value) for value in groups], dtype=object)
    if not (len(y) == len(lat) == len(lon) == len(g)) or len(y) < 3:
        raise ValueError("At least three aligned observations are required.")
    residual = y.copy()
    for group in set(g.tolist()):
        mask = g == group
        residual[mask] -= np.mean(residual[mask])
    variance = float(np.mean(residual * residual))
    if variance <= np.finfo(float).eps:
        return SpatialRangeEstimate(0.0, 0.0, 0, (), ())
    rng = np.random.default_rng(seed)
    possible = len(y) * (len(y) - 1) // 2
    pair_n = int(min(max_pairs, possible))
    if possible <= max_pairs and len(y) <= 2000:
        left, right = np.triu_indices(len(y), k=1)
    else:
        left = rng.integers(0, len(y), size=pair_n * 2)
        right = rng.integers(0, len(y), size=pair_n * 2)
        keep = left != right
        left, right = left[keep][:pair_n], right[keep][:pair_n]
    distances = haversine_km(lat[left], lon[left], lat[right], lon[right])
    products = residual[left] * residual[right] / variance
    positive_distances = distances[distances > 0.0]
    if len(positive_distances) == 0:
        return SpatialRangeEstimate(0.0, 0.0, len(distances), (), ())
    edges = np.unique(np.quantile(positive_distances, np.linspace(0.0, 1.0, bins + 1)))
    centers: list[float] = []
    correlations: list[float] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        mask = (distances >= start) & (distances <= stop if stop == edges[-1] else distances < stop)
        if np.sum(mask) < 10:
            continue
        centers.append(float(np.median(distances[mask])))
        correlations.append(float(np.clip(np.mean(products[mask]), -1.0, 1.0)))
    if not centers:
        return SpatialRangeEstimate(float(np.median(positive_distances)), 0.0, len(distances), (), ())
    near = correlations[0]
    threshold = max(0.05, 0.1 * max(near, 0.0))
    range_km = centers[-1]
    for distance, correlation in zip(centers, correlations):
        if correlation <= threshold:
            range_km = distance
            break
    return SpatialRangeEstimate(float(range_km), float(near), len(distances), tuple(centers), tuple(correlations))


@dataclass(frozen=True)
class BlockCandidateCalibration:
    cell_km: float
    cluster_count: int
    range_adequate: bool
    valid_simulations: int
    null_coverage: float
    null_coverage_ci_low: float
    null_coverage_ci_high: float
    false_positive_rate: float
    false_positive_ci_low: float
    false_positive_ci_high: float
    moderate_tail_power: float
    moderate_tail_power_ci_low: float
    moderate_tail_power_ci_high: float
    power_target_met: bool
    passes: bool


@dataclass(frozen=True)
class SpatialBlockCalibration:
    validity: Validity
    selected_cell_km: float | None
    range_estimate: SpatialRangeEstimate
    candidates: tuple[BlockCandidateCalibration, ...]
    simulation_repetitions: int
    message: str = ""


def calibrate_spatial_block_scale(
    losses: Sequence[float],
    groups: Sequence[Any],
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    *,
    candidate_cell_km: Sequence[float] | None = None,
    confidence_level: float = 0.95,
    n_simulations: int = 200,
    n_bootstrap: int = 500,
    seed: int = 42,
    beta: float = 0.10,
    minimum_moderate_tail_power: float = 0.80,
    require_power_gate: bool = False,
) -> SpatialBlockCalibration:
    if n_simulations < 100:
        raise ValueError("n_simulations must be at least 100 for a formal coverage/power gate.")
    if not 0.0 < float(minimum_moderate_tail_power) < 1.0:
        raise ValueError("minimum_moderate_tail_power must be in (0,1).")
    range_estimate = estimate_spatial_correlation_range(losses, groups, latitudes, longitudes, seed=seed)
    base = max(range_estimate.range_km, 1.0)
    candidates = tuple(sorted(set(float(value) for value in (candidate_cell_km or (base, 1.5 * base, 2.0 * base)) if float(value) > 0.0)))
    if not candidates:
        raise ValueError("At least one positive candidate cell size is required.")
    group_values = np.asarray([str(value) for value in groups], dtype=object)
    group_names = tuple(sorted(set(group_values.tolist())))
    if len(group_names) < 2:
        raise ValueError("At least two groups are required for block calibration.")
    block_ids = {cell: np.asarray(equal_area_block_ids(latitudes, longitudes, cell_km=cell), dtype=object) for cell in candidates}
    truth_blocks = np.asarray(equal_area_block_ids(latitudes, longitudes, cell_km=base), dtype=object)
    truth_names = tuple(sorted(set(truth_blocks.tolist())))
    rho = float(np.clip(range_estimate.near_correlation, 0.05, 0.85))
    bad_count = max(1, int(math.ceil(beta * len(group_names))))
    bad_groups = set(group_names[:bad_count])
    alpha = 1.0 - confidence_level
    def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
        if total <= 0:
            return float("nan"), float("nan")
        proportion = successes / total
        denominator = 1.0 + z * z / total
        center = (proportion + z * z / (2.0 * total)) / denominator
        radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
        return max(0.0, center - radius), min(1.0, center + radius)

    records: list[BlockCandidateCalibration] = []
    for candidate in candidates:
        # Common random numbers make power/coverage differences attributable
        # to block scale rather than a different Monte Carlo sample.
        rng = np.random.default_rng(seed)
        null_cover = 0
        false_positive = 0
        power = 0
        valid_simulations = 0
        candidate_clusters = block_ids[candidate]
        for simulation in range(n_simulations):
            block_effects = dict(zip(truth_names, rng.normal(size=len(truth_names))))
            spatial = np.asarray([block_effects[str(block)] for block in truth_blocks], dtype=float)
            noise = rng.normal(size=len(group_values))
            null_loss = np.clip(0.5 + 0.18 * (math.sqrt(rho) * spatial + math.sqrt(1.0 - rho) * noise), 0.0, 1.0)
            signal_loss = np.clip(null_loss + np.asarray([0.10 if group in bad_groups else 0.0 for group in group_values]), 0.0, 1.0)
            band = simultaneous_group_risk_band(
                null_loss,
                group_values,
                candidate_clusters,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                seed=seed + simulation,
                min_clusters_per_group=2,
            )
            if band.validity != Validity.VALID:
                continue
            valid_simulations += 1
            low = dict(band.lower)
            high = dict(band.upper)
            null_cover += int(all(low[group] <= 0.5 <= high[group] for group in group_names))
            false_positive += int(certify_geobwer_from_band(band, beta=beta).lower_confidence_bound > 0.0)
            signal_band = simultaneous_group_risk_band(
                signal_loss,
                group_values,
                candidate_clusters,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                seed=seed + 10_000 + simulation,
                min_clusters_per_group=2,
            )
            if signal_band.validity == Validity.VALID:
                power += int(certify_geobwer_from_band(signal_band, beta=beta).lower_confidence_bound > 0.0)
        denominator = max(valid_simulations, 1)
        coverage = null_cover / denominator
        fpr = false_positive / denominator
        candidate_power = power / denominator
        coverage_ci = wilson(null_cover, denominator)
        fpr_ci = wilson(false_positive, denominator)
        power_ci = wilson(power, denominator)
        range_adequate = bool(candidate + 1e-12 >= max(range_estimate.range_km, 1.0))
        power_target_met = bool(power_ci[0] >= minimum_moderate_tail_power)
        # Inference validity and sensitivity are deliberately separated.
        # Requiring a fixed power level would confound an honest but imprecise
        # dataset with an invalid block design. Power remains visible and is a
        # ranking criterion; callers may pre-register it as a hard gate.
        passes = (
            valid_simulations == n_simulations
            and range_adequate
            and coverage_ci[1] >= confidence_level
            and fpr_ci[0] <= alpha
            and (power_target_met or not require_power_gate)
        )
        records.append(
            BlockCandidateCalibration(
                cell_km=candidate,
                cluster_count=len(set(candidate_clusters.tolist())),
                range_adequate=range_adequate,
                valid_simulations=valid_simulations,
                null_coverage=coverage,
                null_coverage_ci_low=coverage_ci[0],
                null_coverage_ci_high=coverage_ci[1],
                false_positive_rate=fpr,
                false_positive_ci_low=fpr_ci[0],
                false_positive_ci_high=fpr_ci[1],
                moderate_tail_power=candidate_power,
                moderate_tail_power_ci_low=power_ci[0],
                moderate_tail_power_ci_high=power_ci[1],
                power_target_met=power_target_met,
                passes=passes,
            )
        )
    eligible = [record for record in records if record.passes]
    if not eligible:
        return SpatialBlockCalibration(
            validity=Validity.SPATIAL_BLOCK_NOT_CALIBRATED,
            selected_cell_km=None,
            range_estimate=range_estimate,
            candidates=tuple(records),
            simulation_repetitions=n_simulations,
            message=(
                "No candidate block scale covered the validation-estimated spatial range and passed the "
                "pre-registered coverage/false-positive gates."
            ),
        )
    selected = sorted(eligible, key=lambda record: (-record.moderate_tail_power, record.cell_km))[0]
    return SpatialBlockCalibration(
        validity=Validity.VALID,
        selected_cell_km=selected.cell_km,
        range_estimate=range_estimate,
        candidates=tuple(records),
        simulation_repetitions=n_simulations,
    )
