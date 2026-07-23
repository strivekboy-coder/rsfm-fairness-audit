from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_core import compute_geobwer, legacy_whole_slice_bwer
from rsfm_fairness_audit.bwer_inference import (
    certify_geobwer_from_band,
    honest_confirmed_bwer,
    paired_bwer_comparison,
    simultaneous_group_risk_band,
)
from rsfm_fairness_audit.bwer_protocol import Validity
from rsfm_fairness_audit.io import ensure_dir, write_csv


@dataclass(frozen=True)
class ValidationGate:
    name: str
    estimate: float
    required: str
    passes: bool
    family: str
    interpretation: str


@dataclass(frozen=True)
class GeoBWERValidationResult:
    validity: Validity
    property_checks: tuple[ValidationGate, ...]
    simulation_checks: tuple[ValidationGate, ...]
    simulation_repetitions: int
    bootstrap_replicates: int
    seed: int

    @property
    def passes(self) -> bool:
        return self.validity == Validity.VALID and all(
            check.passes for check in (*self.property_checks, *self.simulation_checks)
        )


def _property_checks(seed: int) -> tuple[ValidationGate, ...]:
    rng = np.random.default_rng(seed)
    risks = rng.uniform(0.05, 0.95, size=20)
    mapping = {f"g{index}": float(value) for index, value in enumerate(risks)}
    groups = len(mapping)
    embedding_error = max(
        abs(
            compute_geobwer(mapping, beta=tail_count / groups).bwer
            - legacy_whole_slice_bwer(mapping, beta=tail_count / groups)
        )
        for tail_count in range(1, groups + 1)
    )
    beta_grid = np.linspace(0.001, 1.0, 1000)
    profile = np.asarray([compute_geobwer(mapping, beta=float(beta)).bwer for beta in beta_grid])
    beta_limit_error = abs(
        compute_geobwer(mapping, beta=1e-10).bwer
        - (float(np.max(risks)) - float(np.mean(risks)))
    )
    endpoint_error = abs(compute_geobwer(mapping, beta=1.0).bwer)
    monotonicity_violation = max(0.0, float(np.max(np.diff(profile))))
    translated = {group: risk + 0.025 for group, risk in mapping.items()}
    translation_error = abs(compute_geobwer(translated, beta=0.10).bwer - compute_geobwer(mapping, beta=0.10).bwer)
    scaled = {group: 0.5 * risk for group, risk in mapping.items()}
    homogeneity_error = abs(compute_geobwer(scaled, beta=0.10).bwer - 0.5 * compute_geobwer(mapping, beta=0.10).bwer)
    original = {"a": 0.10, "b": 0.30, "c": 0.80}
    cloned = {"a": 0.10, "b": 0.30, "c1": 0.80, "c2": 0.80}
    clone_error = abs(
        compute_geobwer(original, beta=0.20, deployment_weights={"a": 0.2, "b": 0.3, "c": 0.5}).bwer
        - compute_geobwer(
            cloned,
            beta=0.20,
            deployment_weights={"a": 0.2, "b": 0.3, "c1": 0.2, "c2": 0.3},
        ).bwer
    )
    tied_a = {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.2, "e": 0.1}
    tied_b = {"c": 0.9, "a": 0.9, "b": 0.9, "e": 0.1, "d": 0.2}
    tie_error = abs(compute_geobwer(tied_a, beta=0.40).bwer - compute_geobwer(tied_b, beta=0.40).bwer)
    fine_weights_array = rng.dirichlet(np.ones(12))
    fine_risks_array = rng.uniform(0.05, 0.95, size=12)
    fine_risks = {f"f{index}": float(value) for index, value in enumerate(fine_risks_array)}
    fine_weights = {f"f{index}": float(value) for index, value in enumerate(fine_weights_array)}
    coarse_risks: dict[str, float] = {}
    coarse_weights: dict[str, float] = {}
    for parent_index in range(4):
        child_indices = range(3 * parent_index, 3 * parent_index + 3)
        parent_weight = float(sum(fine_weights_array[index] for index in child_indices))
        coarse_weights[f"p{parent_index}"] = parent_weight
        coarse_risks[f"p{parent_index}"] = float(
            sum(
                fine_weights_array[index] * fine_risks_array[index]
                for index in child_indices
            )
            / parent_weight
        )
    refinement_violation = max(
        0.0,
        max(
            compute_geobwer(coarse_risks, beta=beta, deployment_weights=coarse_weights).bwer
            - compute_geobwer(fine_risks, beta=beta, deployment_weights=fine_weights).bwer
            for beta in (0.05, 0.10, 0.20, 0.50, 1.0)
        ),
    )
    specifications = (
        ("legacy_embedding", embedding_error, "error <= 1e-12", embedding_error <= 1e-12, "BWER1 is embedded at equal-weight integer tail masses."),
        ("worst_group_limit", beta_limit_error, "error <= 1e-8", beta_limit_error <= 1e-8, "The beta-to-zero limit equals the worst-group excess risk."),
        ("beta_one_endpoint", endpoint_error, "error <= 1e-12", endpoint_error <= 1e-12, "The full deployment mass has zero excess over itself."),
        ("beta_profile_monotonicity", monotonicity_violation, "violation <= 1e-12", monotonicity_violation <= 1e-12, "Increasing tail mass cannot increase upper-tail excess risk."),
        ("translation_invariance", translation_error, "error <= 1e-12", translation_error <= 1e-12, "A common risk offset does not create disparity."),
        ("positive_homogeneity", homogeneity_error, "error <= 1e-12", homogeneity_error <= 1e-12, "Rescaling the loss rescales GeoBWER."),
        ("measure_preserving_clone_invariance", clone_error, "error <= 1e-12", clone_error <= 1e-12, "Splitting a group without changing deployment mass does not change the metric."),
        ("measure_preserving_refinement_monotonicity", refinement_violation, "violation <= 1e-12", refinement_violation <= 1e-12, "Refining a partition while preserving parent audit mass cannot reduce upper-tail excess risk."),
        ("tie_permutation_invariance", tie_error, "error <= 1e-12", tie_error <= 1e-12, "Tail-boundary label ordering does not change the metric value."),
    )
    return tuple(
        ValidationGate(name, float(value), required, bool(passes), "population_property", interpretation)
        for name, value, required, passes, interpretation in specifications
    )


def _simulated_panel(
    rng: np.random.Generator,
    true_risks: Sequence[float],
    *,
    clusters: int,
    units_per_group: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_count = len(true_risks)
    if len(units_per_group) != group_count:
        raise ValueError("units_per_group must align with true_risks.")
    losses: list[float] = []
    groups: list[str] = []
    cluster_ids: list[str] = []
    for cluster_index in range(clusters):
        shared = rng.choice(np.asarray([-0.025, 0.025]))
        for group_index, (mean, unit_count) in enumerate(zip(true_risks, units_per_group)):
            individual = rng.choice(np.asarray([-0.075, 0.075]), size=int(unit_count))
            losses.extend((float(mean) + shared + individual).tolist())
            groups.extend([f"g{group_index}"] * int(unit_count))
            cluster_ids.extend([f"c{cluster_index}"] * int(unit_count))
    values = np.asarray(losses, dtype=float)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError("Validation panel left the bounded loss domain.")
    return values, np.asarray(groups, dtype=object), np.asarray(cluster_ids, dtype=object)


def _simulation_checks(
    *,
    repetitions: int,
    n_bootstrap: int,
    seed: int,
    confidence_level: float,
) -> tuple[ValidationGate, ...]:
    if repetitions < 100:
        raise ValueError("Formal validation requires at least 100 simulation repetitions.")
    if n_bootstrap < 100:
        raise ValueError("Formal validation requires at least 100 multiplier replicates.")
    rng = np.random.default_rng(seed)
    group_count = 10
    beta = 0.10
    null_truth = np.full(group_count, 0.25, dtype=float)
    alternative_truth = null_truth.copy()
    alternative_truth[0] = 0.37
    imbalanced_units = np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=int)
    balanced_units = np.full(group_count, 3, dtype=int)
    counts = {
        "null_group_band_coverage": 0,
        "null_bwer_interval_coverage": 0,
        "null_false_positive": 0,
        "alternative_bwer_interval_coverage": 0,
        "alternative_positive_certificate": 0,
        "imbalanced_group_band_coverage": 0,
        "paired_delta_interval_coverage": 0,
        "honest_confirmation_positive": 0,
    }
    null_plugin: list[float] = []
    interval_width_ratios: list[float] = []
    sharpened_never_wider: list[bool] = []
    null_bwer = compute_geobwer({f"g{i}": value for i, value in enumerate(null_truth)}, beta=beta).bwer
    alternative_bwer = compute_geobwer(
        {f"g{i}": value for i, value in enumerate(alternative_truth)}, beta=beta
    ).bwer
    model_b_truth = alternative_truth.copy()
    model_b_truth[0] = 0.32
    paired_truth = alternative_bwer - compute_geobwer(
        {f"g{i}": value for i, value in enumerate(model_b_truth)}, beta=beta
    ).bwer
    for repetition in range(repetitions):
        null_loss, group, cluster = _simulated_panel(
            rng,
            null_truth,
            clusters=80,
            units_per_group=balanced_units,
        )
        null_band = simultaneous_group_risk_band(
            null_loss,
            group,
            cluster,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed + repetition,
        )
        if null_band.validity != Validity.VALID:
            raise RuntimeError("The balanced null validation panel unexpectedly failed inference preflight.")
        null_lower, null_upper = dict(null_band.lower), dict(null_band.upper)
        counts["null_group_band_coverage"] += int(
            all(null_lower[f"g{i}"] <= null_truth[i] <= null_upper[f"g{i}"] for i in range(group_count))
        )
        null_certificate = certify_geobwer_from_band(null_band, beta=beta)
        interval_width_ratios.append(
            null_certificate.radius / null_certificate.weighted_sum_radius
            if null_certificate.weighted_sum_radius > 0.0
            else 1.0
        )
        sharpened_never_wider.append(
            null_certificate.radius <= null_certificate.weighted_sum_radius + 1e-15
        )
        counts["null_bwer_interval_coverage"] += int(
            null_certificate.ci_low <= null_bwer <= null_certificate.ci_high
        )
        counts["null_false_positive"] += int(null_certificate.lower_confidence_bound > 0.0)
        null_plugin.append(null_certificate.point.bwer)

        alternative_loss, alt_group, alt_cluster = _simulated_panel(
            rng,
            alternative_truth,
            clusters=80,
            units_per_group=balanced_units,
        )
        alternative_band = simultaneous_group_risk_band(
            alternative_loss,
            alt_group,
            alt_cluster,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed + 10_000 + repetition,
        )
        alternative_certificate = certify_geobwer_from_band(alternative_band, beta=beta)
        interval_width_ratios.append(
            alternative_certificate.radius / alternative_certificate.weighted_sum_radius
            if alternative_certificate.weighted_sum_radius > 0.0
            else 1.0
        )
        sharpened_never_wider.append(
            alternative_certificate.radius <= alternative_certificate.weighted_sum_radius + 1e-15
        )
        counts["alternative_bwer_interval_coverage"] += int(
            alternative_certificate.ci_low <= alternative_bwer <= alternative_certificate.ci_high
        )
        counts["alternative_positive_certificate"] += int(
            alternative_certificate.lower_confidence_bound > 0.0
        )
        honest = honest_confirmed_bwer(
            alternative_loss,
            alt_group,
            alt_cluster,
            beta=beta,
            seed=seed + 20_000 + repetition,
        )
        counts["honest_confirmation_positive"] += int(
            honest.validity == Validity.VALID and honest.both_directions_positive
        )

        imbalanced_loss, imbalanced_group, imbalanced_cluster = _simulated_panel(
            rng,
            alternative_truth,
            clusters=80,
            units_per_group=imbalanced_units,
        )
        imbalanced_band = simultaneous_group_risk_band(
            imbalanced_loss,
            imbalanced_group,
            imbalanced_cluster,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed + 30_000 + repetition,
        )
        imbalanced_lower, imbalanced_upper = dict(imbalanced_band.lower), dict(imbalanced_band.upper)
        counts["imbalanced_group_band_coverage"] += int(
            all(
                imbalanced_lower[f"g{i}"] <= alternative_truth[i] <= imbalanced_upper[f"g{i}"]
                for i in range(group_count)
            )
        )

        paired_base, paired_group, paired_cluster = _simulated_panel(
            rng,
            null_truth,
            clusters=80,
            units_per_group=balanced_units,
        )
        paired_a = paired_base + np.asarray(
            [alternative_truth[int(str(value)[1:])] - null_truth[int(str(value)[1:])] for value in paired_group]
        )
        paired_b = paired_base + np.asarray(
            [model_b_truth[int(str(value)[1:])] - null_truth[int(str(value)[1:])] for value in paired_group]
        )
        comparison = paired_bwer_comparison(
            paired_a,
            paired_b,
            paired_group,
            paired_cluster,
            beta=beta,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed + 40_000 + repetition,
        )
        counts["paired_delta_interval_coverage"] += int(
            comparison.validity == Validity.VALID
            and comparison.ci_low <= paired_truth <= comparison.ci_high
        )

    rates = {name: value / repetitions for name, value in counts.items()}
    alpha = 1.0 - confidence_level
    mean_null_plugin = float(np.mean(null_plugin))
    mean_width_ratio = float(np.mean(interval_width_ratios))
    never_wider_rate = float(np.mean(sharpened_never_wider))
    specifications = (
        ("null_group_band_coverage", rates["null_group_band_coverage"], f">= {confidence_level - 0.05:.2f}", rates["null_group_band_coverage"] >= confidence_level - 0.05, "Familywise coverage of all fixed group risks under the null."),
        ("null_bwer_interval_coverage", rates["null_bwer_interval_coverage"], f">= {confidence_level - 0.05:.2f}", rates["null_bwer_interval_coverage"] >= confidence_level - 0.05, "Coverage after propagating the simultaneous group-risk band through GeoBWER."),
        ("null_certificate_false_positive_rate", rates["null_false_positive"], f"<= {alpha + 0.05:.2f}", rates["null_false_positive"] <= alpha + 0.05, "False positive rate for a positive certified disparity under equal true risks."),
        ("alternative_bwer_interval_coverage", rates["alternative_bwer_interval_coverage"], f">= {confidence_level - 0.05:.2f}", rates["alternative_bwer_interval_coverage"] >= confidence_level - 0.05, "GeoBWER interval coverage under a pre-specified tail alternative."),
        ("alternative_positive_certificate_power", rates["alternative_positive_certificate"], ">= 0.80", rates["alternative_positive_certificate"] >= 0.80, "Power to certify a bounded 0.12 tail-risk elevation in the reference panel."),
        ("support_imbalance_group_band_coverage", rates["imbalanced_group_band_coverage"], f">= {confidence_level - 0.05:.2f}", rates["imbalanced_group_band_coverage"] >= confidence_level - 0.05, "Familywise risk-band coverage when group sample counts differ five-fold."),
        ("paired_common_support_delta_coverage", rates["paired_delta_interval_coverage"], f">= {confidence_level - 0.05:.2f}", rates["paired_delta_interval_coverage"] >= confidence_level - 0.05, "Coverage of the conservative paired common-support GeoBWER difference interval."),
        ("honest_tail_confirmation_power", rates["honest_confirmation_positive"], ">= 0.80", rates["honest_confirmation_positive"] >= 0.80, "Both discovery-to-confirmation directions preserve a positive tail contrast."),
        ("null_plugin_selection_bias_visible", mean_null_plugin, "> 0", mean_null_plugin > 0.0, "The plug-in score exposes the expected winner's-curse bias; certification, not concealment, controls it."),
        ("sharpened_lipschitz_never_wider", never_wider_rate, "= 1.00", never_wider_rate == 1.0, "The minimum of the weighted and total-variation envelopes is never wider than the previous valid radius."),
        ("sharpened_lipschitz_mean_width_ratio", mean_width_ratio, "< 1.00", mean_width_ratio < 1.0, "Unclipped interval width relative to the previous weighted tail-plus-mean propagation."),
    )
    return tuple(
        ValidationGate(name, float(value), required, bool(passes), "finite_sample_simulation", interpretation)
        for name, value, required, passes, interpretation in specifications
    )


def validate_geobwer_design(
    *,
    repetitions: int = 200,
    n_bootstrap: int = 500,
    seed: int = 20260722,
    confidence_level: float = 0.95,
) -> GeoBWERValidationResult:
    properties = _property_checks(seed)
    simulations = _simulation_checks(
        repetitions=repetitions,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
        confidence_level=confidence_level,
    )
    validity = Validity.VALID if all(check.passes for check in (*properties, *simulations)) else Validity.INVALID_PROTOCOL
    return GeoBWERValidationResult(
        validity=validity,
        property_checks=properties,
        simulation_checks=simulations,
        simulation_repetitions=repetitions,
        bootstrap_replicates=n_bootstrap,
        seed=seed,
    )


def write_validation_report(result: GeoBWERValidationResult, output_dir: str | Path) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    checks = (*result.property_checks, *result.simulation_checks)
    csv_path = output / "geobwer_validation_gates.csv"
    write_csv(csv_path, checks)
    json_path = output / "geobwer_validation_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "schema": "geobwer.validation.v1",
                "validity": result.validity.value,
                "passes": result.passes,
                "simulation_repetitions": result.simulation_repetitions,
                "bootstrap_replicates": result.bootstrap_replicates,
                "seed": result.seed,
                "checks": [asdict(check) for check in checks],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = output / "geobwer_validation_report.md"
    lines = [
        "# GeoBWER production validation gate",
        "",
        f"- Overall validity: `{result.validity.value}`",
        f"- All gates passed: `{str(result.passes).lower()}`",
        f"- Simulation repetitions: `{result.simulation_repetitions}`",
        f"- Multiplier replicates per interval: `{result.bootstrap_replicates}`",
        "",
        "| Family | Gate | Estimate | Requirement | Pass |",
        "|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| {check.family} | {check.name} | {check.estimate:.6g} | {check.required} | {str(check.passes).lower()} |"
        for check in checks
    )
    lines.extend(
        [
            "",
            "This gate validates the production implementation, not a duplicate research-only formula. "
            "It is a prerequisite for formal model runs and is not itself empirical evidence about a GeoFM.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": json_path, "gates": csv_path, "report": report_path}


def run_validation_gate(
    output_dir: str | Path,
    *,
    repetitions: int = 200,
    n_bootstrap: int = 500,
    seed: int = 20260722,
) -> dict[str, Path]:
    result = validate_geobwer_design(
        repetitions=repetitions,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    artifacts = write_validation_report(result, output_dir)
    if not result.passes:
        failed = ", ".join(
            check.name for check in (*result.property_checks, *result.simulation_checks) if not check.passes
        )
        raise RuntimeError(f"GeoBWER production validation failed: {failed}")
    return artifacts


__all__ = [
    "GeoBWERValidationResult",
    "ValidationGate",
    "run_validation_gate",
    "validate_geobwer_design",
    "write_validation_report",
]
