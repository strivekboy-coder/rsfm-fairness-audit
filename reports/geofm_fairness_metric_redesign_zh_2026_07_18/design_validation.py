"""Lightweight design validation for the GeoBWER / Certified BWER Profile.

This script validates the population functional, demonstrates known failure
modes, and stress-tests simultaneous-band propagation under spatial
dependence. It never trains a model or modifies canonical experiment outputs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent / "diagnostics"
BETA = 0.10


def normalize_weights(weights: np.ndarray | None, groups: int) -> np.ndarray:
    if weights is None:
        return np.full(groups, 1.0 / groups)
    values = np.asarray(weights, dtype=float)
    if len(values) != groups or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("invalid deployment weights")
    return values / values.sum()


def fractional_tail(
    risks: np.ndarray,
    beta: float = BETA,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Exact upper-tail AVaR and selected deployment mass."""
    values = np.asarray(risks, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("risks must be a non-empty finite vector")
    if not 0 < beta <= 1:
        raise ValueError("beta must lie in (0, 1]")
    mu = normalize_weights(weights, len(values))
    order = np.argsort(-values, kind="mergesort")
    selected = np.zeros(len(values), dtype=float)
    remaining = beta
    for index in order:
        take = min(float(mu[index]), remaining)
        selected[index] = take
        remaining -= take
        if remaining <= 1e-15:
            break
    return float(np.dot(selected, values) / beta), selected


def bwer(
    risks: np.ndarray,
    beta: float = BETA,
    weights: np.ndarray | None = None,
) -> float:
    values = np.asarray(risks, dtype=float)
    mu = normalize_weights(weights, len(values))
    tail, _ = fractional_tail(values, beta, mu)
    return tail - float(np.dot(mu, values))


def legacy_bwer(risks: np.ndarray, beta: float = BETA) -> float:
    values = np.asarray(risks, dtype=float)
    k = max(1, int(math.ceil(beta * len(values))))
    return float(np.mean(np.sort(values)[-k:]) - np.mean(values))


def property_checks(seed: int = 20260721) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.0, size=10)
    beta_grid = np.linspace(0.001, 1.0, 1000)
    profile = np.array([bwer(values, beta) for beta in beta_grid])
    original_risk = np.array([0.1, 0.3, 0.8])
    original_mu = np.array([0.2, 0.3, 0.5])
    cloned_risk = np.array([0.1, 0.3, 0.8, 0.8])
    cloned_mu = np.array([0.2, 0.3, 0.2, 0.3])
    tied = np.array([0.9, 0.9, 0.9, 0.2, 0.1])
    permuted = tied[[2, 0, 1, 4, 3]]
    checks = {
        "legacy_embedding": max(
            abs(legacy_bwer(values, k / len(values)) - bwer(values, k / len(values)))
            for k in range(1, len(values) + 1)
        ),
        "beta_zero_limit": abs(bwer(values, 1e-8) - (values.max() - values.mean())),
        "beta_one_endpoint": abs(bwer(values, 1.0)),
        "beta_profile_nonincreasing": max(0.0, float(np.max(np.diff(profile)))),
        "measure_preserving_clone": abs(
            bwer(original_risk, 0.2, original_mu)
            - bwer(cloned_risk, 0.2, cloned_mu)
        ),
        "tie_permutation": abs(bwer(tied, 0.4) - bwer(permuted, 0.4)),
        "constant_risk_zero": abs(bwer(np.full(17, 0.42), 0.1)),
    }
    return pd.DataFrame(
        {"property": list(checks), "error": list(checks.values()), "passes": [x < 1e-10 for x in checks.values()]}
    )


def implementation_counterexamples() -> pd.DataFrame:
    q = np.array([0.5, 0.5])
    risk_a = np.array([0.1, 0.9])
    risk_b = np.array([0.1, np.nan])
    available = np.isfinite(risk_b)
    renorm_a = float(np.dot(q, risk_a))
    renorm_b = float(np.dot(q[available] / q[available].sum(), risk_b[available]))
    correct_replica = np.array([1.0, 1.0, 0.0])
    collapsed_replica = np.array([1.0, 0.0])
    return pd.DataFrame(
        [
            {
                "case": "per_group_renormalize",
                "observed_value": abs(renorm_a - renorm_b),
                "correct_value": 0.0,
                "failure": "different target composition creates a spurious standardized gap",
            },
            {
                "case": "cluster_equals_slice_id_collapse",
                "observed_value": bwer(collapsed_replica, 0.5),
                "correct_value": bwer(correct_replica, 0.5),
                "failure": "repeated event draws collapse and change the bootstrap estimand",
            },
            {
                "case": "legacy_small_G_tail_mass",
                "observed_value": math.ceil(0.1 * 11) / 11,
                "correct_value": 0.1,
                "failure": "ceil(beta*G) audits 18.18% rather than 10% of deployment mass",
            },
        ]
    )


def _ar1(rng: np.random.Generator, length: int, rho: float) -> np.ndarray:
    values = np.empty(length, dtype=float)
    values[0] = rng.normal()
    scale = math.sqrt(1.0 - rho**2)
    for index in range(1, length):
        values[index] = rho * values[index - 1] + scale * rng.normal()
    return values


def _contributions(samples: np.ndarray, block_length: int) -> np.ndarray:
    """Cluster contributions to group means for multiplier inference.

    samples has shape (spatial_position, within_position, group). Contiguous
    spatial positions are kept together when block_length > 1.
    """
    positions, within, groups = samples.shape
    centered = samples - samples.mean(axis=(0, 1), keepdims=True)
    units: list[np.ndarray] = []
    for start in range(0, positions, block_length):
        stop = min(start + block_length, positions)
        units.append(centered[start:stop].sum(axis=(0, 1)) / (positions * within))
    return np.stack(units).reshape(-1, groups)


def _iid_contributions(samples: np.ndarray) -> np.ndarray:
    positions, within, groups = samples.shape
    centered = samples - samples.mean(axis=(0, 1), keepdims=True)
    return centered.reshape(positions * within, groups) / (positions * within)


def _band_interval(
    estimate: np.ndarray,
    contributions: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float, float]:
    """Multiplier simultaneous risk bands propagated through BWER."""
    se = np.sqrt(np.sum(contributions**2, axis=0))
    se = np.maximum(se, 1e-12)
    multipliers = rng.choice((-1.0, 1.0), size=(draws, len(contributions)))
    deviations = multipliers @ contributions
    maxima = np.max(np.abs(deviations) / se, axis=1)
    critical = float(np.quantile(maxima, 0.95, method="higher"))
    error = critical * se
    radius = fractional_tail(error, BETA)[0] + float(np.mean(error))
    point = bwer(estimate, BETA)
    return point - radius, point + radius, radius


def spatial_band_simulation(
    seed: int = 20260722,
    repetitions: int = 180,
    multiplier_draws: int = 199,
) -> pd.DataFrame:
    """Compare independence, one-position, and spatial-superblock inference."""
    rng = np.random.default_rng(seed)
    positions, within, groups = 48, 6, 30
    loading = np.sin(np.linspace(0, 2 * np.pi, groups, endpoint=False))
    scenarios = {
        "spatial_null": np.full(groups, 0.30),
        "spatial_moderate_tail": np.r_[np.full(3, 0.34), np.full(groups - 3, 0.30)],
        "spatial_strong_tail": np.r_[np.full(3, 0.44), np.full(groups - 3, 0.30)],
    }
    rows: list[dict[str, object]] = []
    for scenario, truth_risk in scenarios.items():
        truth = bwer(truth_risk)
        records = {name: {"cover": [], "positive": [], "width": []} for name in ("iid", "position_cluster", "spatial_superblock_6")}
        plugin: list[float] = []
        for _ in range(repetitions):
            spatial = 0.075 * np.tanh(_ar1(rng, positions, rho=0.72))
            noise = 0.035 * rng.choice((-1.0, 1.0), size=(positions, within, groups))
            samples = truth_risk[None, None, :] + spatial[:, None, None] * loading[None, None, :] + noise
            estimate = samples.mean(axis=(0, 1))
            plugin.append(bwer(estimate))
            contribution_sets = {
                "iid": _iid_contributions(samples),
                "position_cluster": _contributions(samples, 1),
                "spatial_superblock_6": _contributions(samples, 6),
            }
            for method, contributions in contribution_sets.items():
                lower, upper, radius = _band_interval(estimate, contributions, rng, multiplier_draws)
                records[method]["cover"].append(lower <= truth <= upper)
                records[method]["positive"].append(lower > 0.0)
                records[method]["width"].append(2.0 * radius)
        rows.append(
            {
                "scenario": scenario,
                "method": "apparent_plugin",
                "true_bwer": truth,
                "mean_point": float(np.mean(plugin)),
                "bias": float(np.mean(plugin) - truth),
                "coverage": "",
                "positive_certificate_rate": "",
                "mean_interval_width": "",
                "repetitions": repetitions,
            }
        )
        for method, stats in records.items():
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "true_bwer": truth,
                    "mean_point": float(np.mean(plugin)),
                    "bias": float(np.mean(plugin) - truth),
                    "coverage": float(np.mean(stats["cover"])),
                    "positive_certificate_rate": float(np.mean(stats["positive"])),
                    "mean_interval_width": float(np.mean(stats["width"])),
                    "repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    property_checks().to_csv(OUT / "population_properties.csv", index=False)
    implementation_counterexamples().to_csv(OUT / "implementation_counterexamples.csv", index=False)
    spatial_band_simulation().to_csv(OUT / "spatial_band_simulation.csv", index=False)
    print(f"wrote design diagnostics to {OUT}")


if __name__ == "__main__":
    main()
