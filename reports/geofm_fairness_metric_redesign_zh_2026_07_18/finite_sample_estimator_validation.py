"""Read-only simulation stress tests for the final BWER/P15 design.

The script uses synthetic Bernoulli slice risks and the saved fMoW support
distribution. It does not train a model, alter canonical outputs, or claim that
the simulations are empirical paper results.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "diagnostics"
BETA = 0.10
Z_95_ONE_SIDED = 1.6448536269514722


def normalize_weights(weights: np.ndarray | None, groups: int) -> np.ndarray:
    if weights is None:
        return np.full(groups, 1.0 / groups)
    values = np.asarray(weights, dtype=float)
    if len(values) != groups or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Invalid deployment weights")
    return values / values.sum()


def tail_mass_and_risk(
    risks: np.ndarray,
    beta: float = BETA,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Exact upper-tail mass and AVaR for a discrete deployment measure."""
    values = np.asarray(risks, dtype=float)
    mu = normalize_weights(weights, len(values))
    if not 0 < beta <= 1:
        raise ValueError("beta must lie in (0, 1]")
    order = np.argsort(-values, kind="mergesort")
    selected = np.zeros(len(values), dtype=float)
    remaining = beta
    for index in order:
        take = min(float(mu[index]), remaining)
        selected[index] = take
        remaining -= take
        if remaining <= 1e-15:
            break
    return selected, float(np.dot(selected, values) / beta)


def bwer(
    risks: np.ndarray,
    beta: float = BETA,
    weights: np.ndarray | None = None,
) -> float:
    values = np.asarray(risks, dtype=float)
    mu = normalize_weights(weights, len(values))
    return tail_mass_and_risk(values, beta, mu)[1] - float(np.dot(mu, values))


def bwer1(risks: np.ndarray, beta: float = BETA) -> float:
    values = np.asarray(risks, dtype=float)
    k = max(1, int(math.ceil(beta * len(values))))
    return float(np.mean(np.sort(values)[-k:]) - np.mean(values))


def selected_contrast(
    selection_risks: np.ndarray,
    evaluation_risks: np.ndarray,
    beta: float = BETA,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Evaluate a discovery-selected AVaR direction on independent risks."""
    mu = normalize_weights(weights, len(selection_risks))
    selected_mass, _ = tail_mass_and_risk(selection_risks, beta, mu)
    coefficients = selected_mass / beta - mu
    return float(np.dot(coefficients, evaluation_risks)), coefficients


def empirical_bayes_risks(counts: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Simple beta-binomial moment shrinkage used only as a comparator."""
    counts = np.asarray(counts, dtype=float)
    support = np.asarray(support, dtype=float)
    raw = counts / support
    pooled = float(counts.sum() / support.sum())
    observed_var = float(np.var(raw, ddof=1)) if len(raw) > 1 else 0.0
    sampling_var = float(np.mean(pooled * (1.0 - pooled) / support))
    between_var = max(observed_var - sampling_var, 0.0)
    if between_var <= 1e-12 or pooled in {0.0, 1.0}:
        concentration = 1e6
    else:
        concentration = max(pooled * (1.0 - pooled) / between_var - 1.0, 0.0)
        concentration = min(concentration, 1e6)
    return (counts + concentration * pooled) / (support + concentration)


def bootstrap_bias_corrected(
    counts: np.ndarray,
    support: np.ndarray,
    rng: np.random.Generator,
    beta: float = BETA,
    draws: int = 80,
) -> float:
    raw = counts / support
    apparent = bwer(raw, beta)
    boot = np.empty(draws, dtype=float)
    for index in range(draws):
        simulated = rng.binomial(support, raw) / support
        boot[index] = bwer(simulated, beta)
    return float(2.0 * apparent - np.mean(boot))


def honest_lcb(
    discovery_counts: np.ndarray,
    discovery_support: np.ndarray,
    evaluation_counts: np.ndarray,
    evaluation_support: np.ndarray,
    beta: float = BETA,
) -> tuple[float, float, float, float]:
    discovery_risk = discovery_counts / discovery_support
    evaluation_risk = evaluation_counts / evaluation_support
    contrast, coefficients = selected_contrast(discovery_risk, evaluation_risk, beta)
    variance = float(
        np.sum(
            coefficients**2
            * evaluation_risk
            * (1.0 - evaluation_risk)
            / evaluation_support
        )
    )
    wald_lcb = float(contrast - Z_95_ONE_SIDED * math.sqrt(max(variance, 0.0)))
    # Conditional on the discovery fold, the evaluation contrast is a weighted
    # sum of independent bounded Bernoulli observations. Hoeffding therefore
    # yields a finite-sample one-sided bound without a normal approximation.
    hoeffding_radius = math.sqrt(
        0.5 * math.log(1.0 / 0.05) * float(np.sum(coefficients**2 / evaluation_support))
    )
    log_delta = math.log(1.0 / 0.05)
    variance_upper = float(np.sum(coefficients**2 / (4.0 * evaluation_support)))
    max_individual_weight = float(np.max(np.abs(coefficients) / evaluation_support))
    bernstein_radius = math.sqrt(2.0 * variance_upper * log_delta) + max_individual_weight * log_delta / 3.0
    return contrast, wald_lcb, float(contrast - hoeffding_radius), float(contrast - bernstein_radius)


def hoeffding_band_lcb(
    observed_risks: np.ndarray,
    support: np.ndarray,
    beta: float = BETA,
    delta: float = 0.05,
) -> float:
    """Conservative finite-sample certificate from simultaneous Hoeffding bands.

    If every group error is bounded by e_g, the BWER error is bounded by
    AVaR_beta(e) + E[e]. This protects data-dependent tail selection.
    """
    groups = len(observed_risks)
    radii = np.sqrt(np.log(2.0 * groups / delta) / (2.0 * support))
    radius = tail_mass_and_risk(radii, beta)[1] + float(np.mean(radii))
    return float(bwer(observed_risks, beta) - radius)


def load_fmow_support() -> np.ndarray:
    base = (
        ROOT
        / "outputs/019e9c6b-cca4-7fa2-aea5-cb2a55798073"
        / "presentations/rsfm-bwer-progress-update/assets/canonical/02_fmow"
    )
    path = base / "fmow_sentinel_resnet50_30k_location_disjoint_audit_table.csv"
    frame = pd.read_csv(path, low_memory=False)
    grouped = frame.dropna(subset=["country", "risk"]).groupby("country")["risk"].size()
    return grouped[grouped >= 20].to_numpy(dtype=int)


def scenario_definitions() -> list[dict[str, object]]:
    fmow_support = load_fmow_support()
    groups = len(fmow_support)
    tail_n = int(math.ceil(BETA * groups))

    def tail_profile(base: float, tail: float) -> np.ndarray:
        values = np.full(groups, base, dtype=float)
        values[:tail_n] = tail
        return values

    return [
        {
            "scenario": "fmow_support_null",
            "support": fmow_support,
            "true_risk": np.full(groups, 0.80),
        },
        {
            "scenario": "fmow_support_moderate_tail",
            "support": fmow_support,
            "true_risk": tail_profile(0.80, 0.86),
        },
        {
            "scenario": "fmow_support_strong_tail",
            "support": fmow_support,
            "true_risk": tail_profile(0.80, 0.92),
        },
        {
            "scenario": "small_G_null",
            "support": np.array([28, 35, 42, 50, 63, 75, 90, 110, 130, 160, 190]),
            "true_risk": np.full(11, 0.25),
        },
        {
            "scenario": "small_G_two_bad",
            "support": np.array([28, 35, 42, 50, 63, 75, 90, 110, 130, 160, 190]),
            "true_risk": np.array([0.55, 0.48, *([0.25] * 9)]),
        },
    ]


def estimator_simulation(seed: int = 20260721, repetitions: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for definition in scenario_definitions():
        scenario = str(definition["scenario"])
        support = np.asarray(definition["support"], dtype=int)
        true_risk = np.asarray(definition["true_risk"], dtype=float)
        truth = bwer(true_risk)
        estimates: dict[str, list[float]] = {
            "apparent_plugin": [],
            "crossfit_confirmed": [],
            "empirical_bayes": [],
            "bootstrap_bias_corrected": [],
        }
        honest_wald_lcbs: list[float] = []
        honest_hoeffding_lcbs: list[float] = []
        honest_bernstein_lcbs: list[float] = []
        band_lcbs: list[float] = []
        for _ in range(repetitions):
            n_a = np.maximum(2, support // 2)
            n_b = np.maximum(2, support - n_a)
            k_a = rng.binomial(n_a, true_risk)
            k_b = rng.binomial(n_b, true_risk)
            counts = k_a + k_b
            risks = counts / support
            estimates["apparent_plugin"].append(bwer(risks))
            a_to_b, _ = selected_contrast(k_a / n_a, k_b / n_b)
            b_to_a, _ = selected_contrast(k_b / n_b, k_a / n_a)
            estimates["crossfit_confirmed"].append(0.5 * (a_to_b + b_to_a))
            estimates["empirical_bayes"].append(bwer(empirical_bayes_risks(counts, support)))
            estimates["bootstrap_bias_corrected"].append(
                bootstrap_bias_corrected(counts, support, rng)
            )
            _, wald_lcb, hoeffding_lcb, bernstein_lcb = honest_lcb(k_a, n_a, k_b, n_b)
            honest_wald_lcbs.append(wald_lcb)
            honest_hoeffding_lcbs.append(hoeffding_lcb)
            honest_bernstein_lcbs.append(bernstein_lcb)
            band_lcbs.append(hoeffding_band_lcb(risks, support))
        for estimator, values in estimates.items():
            array = np.asarray(values)
            rows.append(
                {
                    "scenario": scenario,
                    "groups": len(support),
                    "min_support": int(support.min()),
                    "median_support": float(np.median(support)),
                    "max_support": int(support.max()),
                    "true_bwer": truth,
                    "method": estimator,
                    "mean": float(array.mean()),
                    "bias": float(array.mean() - truth),
                    "rmse": float(np.sqrt(np.mean((array - truth) ** 2))),
                    "sd": float(array.std(ddof=1)),
                    "negative_fraction": float(np.mean(array < 0)),
                    "repetitions": repetitions,
                }
            )
        for method, lcbs in {
            "honest_split_wald_lcb": honest_wald_lcbs,
            "honest_split_hoeffding_lcb": honest_hoeffding_lcbs,
            "honest_split_bernstein_lcb": honest_bernstein_lcbs,
            "hoeffding_band_lcb": band_lcbs,
        }.items():
            array = np.asarray(lcbs)
            rows.append(
                {
                    "scenario": scenario,
                    "groups": len(support),
                    "min_support": int(support.min()),
                    "median_support": float(np.median(support)),
                    "max_support": int(support.max()),
                    "true_bwer": truth,
                    "method": method,
                    "mean": float(array.mean()),
                    "bias": "",
                    "rmse": "",
                    "sd": float(array.std(ddof=1)),
                    "negative_fraction": float(np.mean(array < 0)),
                    "lcb_coverage": float(np.mean(array <= truth)),
                    "positive_certificate_rate": float(np.mean(array > 0)),
                    "repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def split_allocation_simulation(seed: int = 20260722, repetitions: int = 500) -> pd.DataFrame:
    """Power/selection trade-off for an honest one-way confirmation split."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for definition in scenario_definitions():
        scenario = str(definition["scenario"])
        if scenario not in {"fmow_support_moderate_tail", "fmow_support_strong_tail", "small_G_two_bad"}:
            continue
        support = np.asarray(definition["support"], dtype=int)
        true_risk = np.asarray(definition["true_risk"], dtype=float)
        true_bwer = bwer(true_risk)
        true_mass, _ = tail_mass_and_risk(true_risk)
        for discovery_fraction in (0.25, 0.33, 0.50, 0.67):
            stats = {"contrast": [], "wald_lcb": [], "hoeffding_lcb": [], "tail_overlap": []}
            for _ in range(repetitions):
                n_a = np.maximum(2, np.floor(support * discovery_fraction).astype(int))
                n_b = np.maximum(2, support - n_a)
                k_a = rng.binomial(n_a, true_risk)
                k_b = rng.binomial(n_b, true_risk)
                contrast, wald_lcb, hoeffding_lcb, _ = honest_lcb(k_a, n_a, k_b, n_b)
                selected_mass, _ = tail_mass_and_risk(k_a / n_a)
                stats["contrast"].append(contrast)
                stats["wald_lcb"].append(wald_lcb)
                stats["hoeffding_lcb"].append(hoeffding_lcb)
                stats["tail_overlap"].append(float(np.minimum(selected_mass, true_mass).sum() / BETA))
            rows.append(
                {
                    "scenario": scenario,
                    "true_bwer": true_bwer,
                    "discovery_fraction": discovery_fraction,
                    "evaluation_fraction": 1.0 - discovery_fraction,
                    "mean_confirmed_contrast": float(np.mean(stats["contrast"])),
                    "mean_tail_mass_overlap": float(np.mean(stats["tail_overlap"])),
                    "wald_positive_certificate_rate": float(np.mean(np.asarray(stats["wald_lcb"]) > 0)),
                    "wald_lcb_coverage": float(np.mean(np.asarray(stats["wald_lcb"]) <= true_bwer)),
                    "hoeffding_positive_certificate_rate": float(np.mean(np.asarray(stats["hoeffding_lcb"]) > 0)),
                    "hoeffding_lcb_coverage": float(np.mean(np.asarray(stats["hoeffding_lcb"]) <= true_bwer)),
                    "repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def property_checks(seed: int = 721) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.0, size=10)
    embedding_error = max(
        abs(bwer1(values, k / len(values)) - bwer(values, k / len(values)))
        for k in range(1, len(values) + 1)
    )
    beta_grid = np.linspace(0.001, 1.0, 1000)
    profile = np.array([bwer(values, float(beta)) for beta in beta_grid])

    original_risk = np.array([0.1, 0.3, 0.8])
    original_mu = np.array([0.2, 0.3, 0.5])
    cloned_risk = np.array([0.1, 0.3, 0.8, 0.8])
    cloned_mu = np.array([0.2, 0.3, 0.2, 0.3])

    tied = np.array([0.9, 0.9, 0.9, 0.2, 0.1])
    tied_permuted = tied[[2, 0, 1, 4, 3]]
    return pd.DataFrame(
        [
            {"property": "BWER1_embedding_equal_weight_integer_beta", "error": embedding_error, "passes": embedding_error < 1e-12},
            {"property": "beta_to_zero_limit", "error": abs(bwer(values, 1e-8) - (values.max() - values.mean())), "passes": abs(bwer(values, 1e-8) - (values.max() - values.mean())) < 1e-10},
            {"property": "beta_one_endpoint", "error": abs(bwer(values, 1.0)), "passes": abs(bwer(values, 1.0)) < 1e-12},
            {"property": "beta_profile_nonincreasing", "error": float(np.max(np.diff(profile))), "passes": bool(np.all(np.diff(profile) <= 1e-12))},
            {"property": "measure_preserving_clone_invariance", "error": abs(bwer(original_risk, 0.2, original_mu) - bwer(cloned_risk, 0.2, cloned_mu)), "passes": abs(bwer(original_risk, 0.2, original_mu) - bwer(cloned_risk, 0.2, cloned_mu)) < 1e-12},
            {"property": "tie_value_permutation_invariance", "error": abs(bwer(tied, 0.4) - bwer(tied_permuted, 0.4)), "passes": abs(bwer(tied, 0.4) - bwer(tied_permuted, 0.4)) < 1e-12},
        ]
    )


def standardisation_counterexample() -> pd.DataFrame:
    # Group A has both classes; group B is missing class 2. Per-group
    # renormalisation compares different target compositions and creates a gap.
    q = np.array([0.5, 0.5])
    risk_a = np.array([0.1, 0.9])
    risk_b = np.array([0.1, np.nan])
    renorm_a = float(np.dot(q, risk_a))
    available = ~np.isnan(risk_b)
    renorm_b = float(np.dot(q[available] / q[available].sum(), risk_b[available]))
    common_a = float(risk_a[0])
    common_b = float(risk_b[0])
    return pd.DataFrame(
        [
            {
                "method": "per_group_renormalize",
                "group_a_standardised_risk": renorm_a,
                "group_b_standardised_risk": renorm_b,
                "absolute_gap": abs(renorm_a - renorm_b),
                "scientific_target": "different composition in each group",
            },
            {
                "method": "strict_common_support",
                "group_a_standardised_risk": common_a,
                "group_b_standardised_risk": common_b,
                "absolute_gap": abs(common_a - common_b),
                "scientific_target": "same identifiable composition",
            },
        ]
    )


def cluster_clone_counterexample() -> pd.DataFrame:
    """Show why repeated cluster draws need replicate-local identities.

    Suppose a superpopulation bootstrap draws three event clusters and obtains
    [high, high, low]. If the two high-event clones retain the same slice ID,
    a later group-by collapses them and silently changes the empirical event
    measure from weights (1/3, 1/3, 1/3) to (1/2, 1/2).
    """
    correct_replica = np.array([1.0, 1.0, 0.0])
    collapsed_replica = np.array([1.0, 0.0])
    return pd.DataFrame(
        [
            {
                "implementation": "replicate_local_unique_slice_ids",
                "replicate_event_risks": "1.0;1.0;0.0",
                "mean_risk": float(correct_replica.mean()),
                "bwer_beta_0_5": bwer(correct_replica, beta=0.5),
                "target": "three-draw empirical event distribution",
            },
            {
                "implementation": "original_ids_collapsed_by_groupby",
                "replicate_event_risks": "1.0;0.0",
                "mean_risk": float(collapsed_replica.mean()),
                "bwer_beta_0_5": bwer(collapsed_replica, beta=0.5),
                "target": "two-unique-event distribution (wrong target)",
            },
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    estimator_simulation().to_csv(OUT / "estimator_simulation.csv", index=False)
    split_allocation_simulation().to_csv(OUT / "split_allocation_simulation.csv", index=False)
    print(f"Wrote finite-sample estimator stress tests to {OUT}")


if __name__ == "__main__":
    main()
