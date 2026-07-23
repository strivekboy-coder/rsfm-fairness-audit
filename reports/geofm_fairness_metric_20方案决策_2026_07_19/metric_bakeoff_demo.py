"""Read-only metric bake-off for the GeoFM fairness redesign review.

This script does not train or run inference.  It uses synthetic risk vectors and
already-saved canonical slice CSVs.  Outputs are diagnostics, not paper results.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import linprog, minimize_scalar
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "diagnostics"
BETA = 0.10


def normalise_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    if len(w) != n or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and match risks")
    return w / w.sum()


def weighted_mean(risks: np.ndarray, weights: np.ndarray | None = None) -> float:
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    return float(np.dot(w, r))


def weighted_fractional_avar(
    risks: np.ndarray, beta: float = BETA, weights: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Upper-tail AVaR with exact fractional mass at the boundary."""
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    order = np.argsort(-r, kind="mergesort")
    remaining = beta
    selected_mass = np.zeros(len(r), dtype=float)
    numerator = 0.0
    for idx in order:
        take = min(float(w[idx]), remaining)
        if take > 0:
            selected_mass[idx] = take
            numerator += take * float(r[idx])
            remaining -= take
        if remaining <= 1e-15:
            break
    return numerator / beta, selected_mass


def bwer1(risks: np.ndarray, beta: float = BETA) -> float:
    r = np.asarray(risks, dtype=float)
    k = max(1, int(math.ceil(beta * len(r))))
    return float(np.mean(np.sort(r)[-k:]) - np.mean(r))


def bwer2(
    risks: np.ndarray, beta: float = BETA, weights: np.ndarray | None = None
) -> float:
    return weighted_fractional_avar(risks, beta, weights)[0] - weighted_mean(risks, weights)


def bwer_s(risks: np.ndarray, beta: float = BETA) -> float:
    """Bounded absolute scaling for losses in [0, 1]; linear in BWER2."""
    return bwer2(risks, beta) / (1.0 - beta)


def max_excess(risks: np.ndarray, weights: np.ndarray | None = None) -> float:
    return float(np.max(risks) - weighted_mean(risks, weights))


def gini_absolute(risks: np.ndarray, weights: np.ndarray | None = None) -> float:
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    return float(0.5 * np.sum((w[:, None] * w[None, :]) * np.abs(r[:, None] - r[None, :])))


def theil(risks: np.ndarray, weights: np.ndarray | None = None) -> float:
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    m = float(np.dot(w, r))
    if m <= 0:
        return 0.0
    x = r / m
    terms = np.zeros_like(x)
    positive = x > 0
    terms[positive] = x[positive] * np.log(x[positive])
    return float(np.dot(w, terms))


def atkinson_half(risks: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Atkinson index with epsilon=0.5, which is defined at zero risk."""
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    m = float(np.dot(w, r))
    if m <= 0:
        return 0.0
    equally_distributed_equivalent = float(np.dot(w, np.sqrt(np.maximum(r, 0.0))) ** 2)
    return float(1.0 - equally_distributed_equivalent / m)


def error_burden_tv(risks: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Total variation between deployment mass and normalized error burden."""
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    m = float(np.dot(w, r))
    if m <= 0:
        return 0.0
    error_mass = w * r / m
    return float(0.5 * np.abs(error_mass - w).sum())


def bwer_n(risks: np.ndarray, beta: float = BETA, weights: np.ndarray | None = None) -> float:
    """Normalized top-burden/Lorenz companion from the AI2 report."""
    m = weighted_mean(risks, weights)
    if m <= 0:
        return 0.0
    return float((beta / (1.0 - beta)) * bwer2(risks, beta, weights) / m)


def upper_semideviation(risks: np.ndarray, weights: np.ndarray | None = None) -> float:
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    m = float(np.dot(w, r))
    return float(np.sqrt(np.dot(w, np.maximum(r - m, 0.0) ** 2)))


def expectile_gap(
    risks: np.ndarray, tau: float = 0.90, weights: np.ndarray | None = None
) -> float:
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    lo, hi = float(np.min(r)), float(np.max(r))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        balance = tau * np.dot(w, np.maximum(r - mid, 0.0)) - (1.0 - tau) * np.dot(
            w, np.maximum(mid - r, 0.0)
        )
        if balance > 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi) - np.dot(w, r))


def evar_gap(
    risks: np.ndarray, beta: float = BETA, weights: np.ndarray | None = None
) -> float:
    """Entropic Value-at-Risk gap, optimized over the Chernoff parameter."""
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    if float(np.ptp(r)) <= 1e-15:
        return 0.0
    logw = np.log(w)

    def objective(log_z: float) -> float:
        z = math.exp(log_z)
        return float((logsumexp(logw + z * r) - math.log(beta)) / z)

    result = minimize_scalar(objective, bounds=(-8.0, 12.0), method="bounded")
    return float(result.fun - np.dot(w, r))


def spectral_gap(
    risks: np.ndarray,
    beta_grid: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40),
    weights: np.ndarray | None = None,
) -> float:
    """A fixed equal mixture of AVaR deviations (a spectral comparator)."""
    return float(np.mean([bwer2(risks, beta, weights) for beta in beta_grid]))


def target_exceedance_inequity(
    risks: np.ndarray, target: float = 0.25, weights: np.ndarray | None = None
) -> float:
    """Jensen gap for violation of an absolute reliability target."""
    r = np.asarray(risks, dtype=float)
    w = normalise_weights(weights, len(r))
    m = float(np.dot(w, r))
    return float(np.dot(w, np.maximum(r - target, 0.0)) - max(m - target, 0.0))


def metric_bundle(risks: np.ndarray, beta: float = BETA) -> dict[str, float]:
    r = np.asarray(risks, dtype=float)
    return {
        "mean_risk": float(np.mean(r)),
        "BWER1": bwer1(r, beta),
        "BWER2": bwer2(r, beta),
        "BWER-S": bwer_s(r, beta),
        "max_excess": max_excess(r),
        "spectral_gap": spectral_gap(r),
        "expectile_gap": expectile_gap(r),
        "EVaR_gap": evar_gap(r, beta),
        "Gini_abs": gini_absolute(r),
        "Theil": theil(r),
        "Atkinson_e0.5": atkinson_half(r),
        "EB_TV": error_burden_tv(r),
        "BWER_N": bwer_n(r, beta),
        "upper_semideviation": upper_semideviation(r),
        "target_exceedance_0.25": target_exceedance_inequity(r, 0.25),
    }


def crossfit_bwer_from_binomial(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    support_a: np.ndarray,
    support_b: np.ndarray,
    beta: float = BETA,
) -> float:
    ra = counts_a / support_a
    rb = counts_b / support_b
    w = np.full(len(ra), 1.0 / len(ra))
    _, mass_a = weighted_fractional_avar(ra, beta, w)
    _, mass_b = weighted_fractional_avar(rb, beta, w)
    a_to_b = float(np.dot(mass_a, rb) / beta - np.mean(rb))
    b_to_a = float(np.dot(mass_b, ra) / beta - np.mean(ra))
    return 0.5 * (a_to_b + b_to_a)


def synthetic_scenarios() -> pd.DataFrame:
    scenarios = {
        "equal_low": np.full(20, 0.02),
        "equal_high": np.full(20, 0.70),
        "single_hotspot": np.array([0.70, *([0.20] * 19)]),
        "two_hotspots": np.array([0.70, 0.60, *([0.20] * 18)]),
        "diffuse_gradient": np.linspace(0.05, 0.55, 20),
        "scaled_two_hotspots": np.array([0.007, 0.006, *([0.002] * 18)]),
    }
    rows = []
    for name, risks in scenarios.items():
        rows.append({"scenario": name, **metric_bundle(risks)})
    return pd.DataFrame(rows)


def null_bias_simulation(seed: int = 719, repetitions: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for groups in (10, 50, 200):
        for support_mode in ("equal", "heterogeneous"):
            collected: dict[str, list[float]] = {
                "BWER1": [],
                "BWER2": [],
                "crossfit_BWER2": [],
                "max_excess": [],
                "Gini_abs": [],
                "EB_TV": [],
                "spectral_gap": [],
            }
            for _ in range(repetitions):
                if support_mode == "equal":
                    n = np.full(groups, 40, dtype=int)
                else:
                    n = rng.integers(10, 201, size=groups)
                n_a = np.maximum(2, n // 2)
                n_b = np.maximum(2, n - n_a)
                ka = rng.binomial(n_a, 0.20)
                kb = rng.binomial(n_b, 0.20)
                r = (ka + kb) / (n_a + n_b)
                values = metric_bundle(r)
                for key in ("BWER1", "BWER2", "max_excess", "Gini_abs", "EB_TV", "spectral_gap"):
                    collected[key].append(values[key])
                collected["crossfit_BWER2"].append(crossfit_bwer_from_binomial(ka, kb, n_a, n_b))
            for metric, vals in collected.items():
                arr = np.asarray(vals)
                rows.append(
                    {
                        "true_group_disparity": 0.0,
                        "groups": groups,
                        "support_mode": support_mode,
                        "metric": metric,
                        "mean_estimate": float(arr.mean()),
                        "median_estimate": float(np.median(arr)),
                        "p95_estimate": float(np.quantile(arr, 0.95)),
                        "sd_estimate": float(arr.std(ddof=1)),
                        "repetitions": repetitions,
                    }
                )
    return pd.DataFrame(rows)


def path_laplacian(n: int) -> np.ndarray:
    adjacency = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    return np.diag(adjacency.sum(axis=1)) - adjacency


def connected_scan_gap(risks: np.ndarray, window: int) -> float:
    r = np.asarray(risks, dtype=float)
    return float(max(np.mean(r[i : i + window]) for i in range(len(r) - window + 1)) - np.mean(r))


def gtv_swer_path(risks: np.ndarray, beta: float, kappa: float) -> float:
    """LP for the path-graph TV-constrained CVaR risk envelope."""
    r = np.asarray(risks, dtype=float)
    n = len(r)
    edges = [(i, i + 1) for i in range(n - 1)]
    m = len(edges)
    # Variables are w_0..w_(n-1), z_0..z_(m-1).
    c = np.concatenate([-r / n, np.zeros(m)])
    a_ub, b_ub = [], []
    for e, (i, j) in enumerate(edges):
        row = np.zeros(n + m)
        row[i], row[j], row[n + e] = 1.0, -1.0, -1.0
        a_ub.append(row)
        b_ub.append(0.0)
        row = np.zeros(n + m)
        row[i], row[j], row[n + e] = -1.0, 1.0, -1.0
        a_ub.append(row)
        b_ub.append(0.0)
    row = np.zeros(n + m)
    row[n:] = 1.0
    a_ub.append(row)
    b_ub.append(kappa)
    a_eq = np.zeros((1, n + m))
    a_eq[0, :n] = 1.0 / n
    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0 / beta)] * n + [(0.0, None)] * m,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    robust_risk = -float(result.fun)
    return robust_risk - float(np.mean(r))


def spatial_counterexamples() -> pd.DataFrame:
    n = 20
    clustered = np.full(n, 0.1)
    clustered[8:12] = 0.9
    scattered = np.full(n, 0.1)
    scattered[[1, 6, 11, 16]] = 0.9
    boundary = np.full(n, 0.1)
    boundary[:4] = 0.9
    lap = path_laplacian(n)
    rows = []
    for name, risks in {"clustered": clustered, "scattered": scattered, "boundary_cluster": boundary}.items():
        smoothed = expm(-0.5 * lap) @ risks
        rows.append(
            {
                "scenario": name,
                "mean": float(np.mean(risks)),
                "BWER2_beta0.2": bwer2(risks, 0.20),
                "connected_window4_gap": connected_scan_gap(risks, 4),
                "Heat_BWER_t0.5": bwer2(smoothed, 0.20),
                "GTV_SWER_kappa4": gtv_swer_path(risks, 0.20, 4.0),
                "GTV_SWER_kappa8": gtv_swer_path(risks, 0.20, 8.0),
            }
        )
    # Same samples and hotspot, but shift the aggregation boundary.
    for shift in range(4):
        rolled = np.roll(clustered, -shift)
        aggregated = rolled.reshape(5, 4).mean(axis=1)
        rows.append(
            {
                "scenario": f"MAUP_grid_shift_{shift}",
                "mean": float(np.mean(aggregated)),
                "BWER2_beta0.2": bwer2(aggregated, 0.20),
                "connected_window4_gap": np.nan,
                "Heat_BWER_t0.5": np.nan,
                "GTV_SWER_kappa4": np.nan,
                "GTV_SWER_kappa8": np.nan,
                "aggregated_risks": ";".join(f"{x:.3f}" for x in aggregated),
            }
        )
    return pd.DataFrame(rows)


def canonical_slice_vectors() -> list[tuple[str, str, str, np.ndarray, str]]:
    base = ROOT / "outputs/019e9c6b-cca4-7fa2-aea5-cb2a55798073/presentations/rsfm-bwer-progress-update/assets/canonical"
    vectors: list[tuple[str, str, str, np.ndarray, str]] = []

    sen_dir = base / "01_sen1floods"
    sen_files = {
        "Prithvi-TL": "sen1floods11_protocol_matched_matched_runs_prithvi_tl_bwer_by_slice.csv",
        "ResNet34-U-Net": "sen1floods11_protocol_matched_matched_runs_s2_resnet34_unet_bwer_by_slice.csv",
        "MNDWI": "sen1floods11_protocol_matched_matched_runs_spectral_mndwi_bwer_by_slice.csv",
        "Vanilla-U-Net": "sen1floods11_protocol_matched_matched_runs_vanilla_unet_bwer_by_slice.csv",
    }
    for model, filename in sen_files.items():
        frame = pd.read_csv(sen_dir / filename)
        frame = frame[(frame["slice_variable"] == "event_id") & (frame["balance_variable"].fillna("") == "") & frame["is_valid_slice"]]
        vectors.append(("Sen1Floods11", model, "event", frame["balanced_risk"].to_numpy(float), str(sen_dir / filename)))

    fmow_dir = base / "02_fmow"
    dofa_path = fmow_dir / "fmow_sentinel_dofa_vitb_linear_probe_30k_location_disjoint_scaled10000_bwer_bwer_by_slice.csv"
    dofa = pd.read_csv(dofa_path, low_memory=False)
    dofa = dofa[(dofa["slice_variable"] == "country") & (dofa["balance_variable"].fillna("") == "") & dofa["is_valid_slice"]]
    vectors.append(("fMoW-Sentinel", "DOFA", "country", dofa["balanced_risk"].to_numpy(float), str(dofa_path)))
    resnet_path = fmow_dir / "fmow_sentinel_resnet50_30k_location_disjoint_audit_table.csv"
    resnet = pd.read_csv(resnet_path, low_memory=False)
    grouped = resnet.dropna(subset=["country", "risk"]).groupby("country")["risk"].agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= 20]
    vectors.append(("fMoW-Sentinel", "ResNet50", "country", grouped["mean"].to_numpy(float), str(resnet_path)))

    reben_dir = base / "03_reben"
    for model, stem in {
        "CROMA-S1": "bwer_croma_s1_bwer_by_slice.csv",
        "CROMA-S2": "bwer_croma_s2_bwer_by_slice.csv",
        "CROMA-S1+S2": "bwer_croma_s1_plus_s2_bwer_by_slice.csv",
    }.items():
        path = reben_dir / stem
        frame = pd.read_csv(path)
        frame = frame[(frame["slice_variable"] == "country") & (frame["balance_variable"].fillna("") == "") & frame["is_valid_slice"]]
        vectors.append(("reBEN", model, "country", frame["balanced_risk"].to_numpy(float), str(path)))
    return vectors


def canonical_posthoc() -> pd.DataFrame:
    rows = []
    for dataset, model, axis, risks, source in canonical_slice_vectors():
        rows.append(
            {
                "dataset": dataset,
                "model_or_condition": model,
                "axis": axis,
                "n_valid_slices": len(risks),
                "source": source,
                **metric_bundle(risks),
            }
        )
    return pd.DataFrame(rows)


def fmow_actual_support_null(seed: int = 1907, repetitions: int = 2000) -> pd.DataFrame:
    """Homogeneous-risk null using the actual valid-country support distributions.

    This is a diagnostic parametric null for 0/1 error.  It does not replace a
    location-aware paired bootstrap, but quantifies tail-selection bias at the
    observed country counts.
    """
    rng = np.random.default_rng(seed)
    base = ROOT / "outputs/019e9c6b-cca4-7fa2-aea5-cb2a55798073/presentations/rsfm-bwer-progress-update/assets/canonical/02_fmow"
    inputs: list[tuple[str, np.ndarray, np.ndarray]] = []

    dofa = pd.read_csv(
        base / "fmow_sentinel_dofa_vitb_linear_probe_30k_location_disjoint_scaled10000_bwer_bwer_by_slice.csv",
        low_memory=False,
    )
    dofa = dofa[(dofa["slice_variable"] == "country") & (dofa["balance_variable"].fillna("") == "") & dofa["is_valid_slice"]]
    inputs.append(("DOFA", dofa["balanced_risk"].to_numpy(float), dofa["sample_count"].to_numpy(int)))

    resnet = pd.read_csv(base / "fmow_sentinel_resnet50_30k_location_disjoint_audit_table.csv", low_memory=False)
    grouped = resnet.dropna(subset=["country", "risk"]).groupby("country")["risk"].agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= 20]
    inputs.append(("ResNet50", grouped["mean"].to_numpy(float), grouped["size"].to_numpy(int)))

    rows = []
    for model, observed_risks, support in inputs:
        pooled_p = float(np.dot(support, observed_risks) / support.sum())
        simulated_bwer = np.empty(repetitions)
        simulated_max = np.empty(repetitions)
        for j in range(repetitions):
            risks = rng.binomial(support, pooled_p) / support
            simulated_bwer[j] = bwer2(risks)
            simulated_max[j] = max_excess(risks)
        observed = bwer2(observed_risks)
        rows.append(
            {
                "model": model,
                "n_valid_countries": len(support),
                "min_support": int(support.min()),
                "median_support": float(np.median(support)),
                "max_support": int(support.max()),
                "homogeneous_null_risk": pooled_p,
                "observed_BWER2": observed,
                "null_mean_BWER2": float(simulated_bwer.mean()),
                "null_p95_BWER2": float(np.quantile(simulated_bwer, 0.95)),
                "observed_minus_null_mean": float(observed - simulated_bwer.mean()),
                "parametric_upper_tail_fraction": float((np.sum(simulated_bwer >= observed) + 1) / (repetitions + 1)),
                "null_mean_max_excess": float(simulated_max.mean()),
                "repetitions": repetitions,
                "limitation": "Independent Bernoulli country null; ignores location dependence, metadata uncertainty, and model pairing.",
            }
        )
    return pd.DataFrame(rows)


WEIGHTS = {
    "construct_validity": 15,
    "mathematical_definition": 12,
    "finite_sample_inference": 15,
    "partition_support_robustness": 12,
    "cross_task_transfer": 12,
    "interpretability_adoption": 10,
    "reproducibility_anti_gaming": 10,
    "novelty_geo_specificity": 8,
    "low_use_cost": 3,
    "current_empirical_validation": 3,
}


SCORES = {
    "BWER1_baseline": [7, 4, 3, 3, 9, 9, 5, 4, 9, 8],
    "BWER2_fractional_core": [9, 9, 6, 6, 9, 9, 8, 5, 9, 7],
    "P01_BWER-S_bounded_absolute": [9, 8, 6, 6, 9, 8, 8, 5, 9, 3],
    "P02_Rawlsian_max_excess": [8, 8, 2, 2, 9, 9, 7, 2, 10, 8],
    "P03_spectral_tail_gap": [8.5, 9, 7, 6, 9, 5, 5, 4, 7, 4],
    "P04_expectile_gap": [7, 8, 8, 6, 9, 5, 6, 4, 8, 2],
    "P05_EVaR_gap": [7.5, 9, 6, 5, 7, 4, 5, 4, 6, 2],
    "P06_absolute_Gini_deviation": [7, 8, 7, 8, 9, 9, 8, 3, 10, 7],
    "P07_generalized_entropy_Theil": [6, 8, 5, 8, 8, 6, 7, 3, 9, 4],
    "P08_error_burden_TV": [7.5, 8, 7, 9, 9, 9, 9, 5, 10, 5],
    "P09_Atkinson": [5.5, 8, 6, 8, 8, 6, 6, 3, 9, 3],
    "P10_BWER-N_Lorenz_companion": [7, 6, 4, 6, 9, 8, 7, 4, 10, 3],
    "P11_f_DRO_robust_excess": [6.5, 9, 6, 7, 7, 6, 5, 4, 5, 3],
    "P12_Wasserstein_distributional_parity": [6.5, 8, 5, 5, 5, 5, 5, 5, 3, 1],
    "P13_target_exceedance_inequity": [7, 8, 7, 7, 8, 8, 7, 6, 9, 2],
    "P14_hierarchical_Bayesian_BWER": [8, 8, 9, 7, 7, 5, 4, 5, 3, 1],
    "P15_certified_crossfit_BWER_profile": [9.5, 9, 9, 7, 9, 8, 9, 7, 6, 3],
    "P16_rich_subgroup_LCB_auditor": [8, 8, 7, 8, 7, 8, 6, 7, 4, 3],
    "P17_SPAD_BWER": [8, 8, 7, 9, 5, 5, 5, 8, 3, 2],
    "P18_connected_spatial_C_SWER": [8.5, 8, 8, 8, 5, 9, 6, 9, 3, 2],
    "P19_graph_heat_BWER": [8, 8, 6, 5, 5, 5, 4, 9, 3, 2],
    "P20_graph_TV_risk_envelope_GTV_SWER": [8.5, 9, 6, 7, 6, 6, 4, 10, 2, 1],
}


def scorecard() -> pd.DataFrame:
    columns = list(WEIGHTS)
    rows = []
    for candidate, values in SCORES.items():
        row = {"candidate": candidate, **dict(zip(columns, values))}
        row["weighted_total_100"] = sum(WEIGHTS[c] * row[c] for c in columns) / 10.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("weighted_total_100", ascending=False)


def weight_sensitivity(seed: int = 20260719, draws: int = 10000) -> pd.DataFrame:
    """Perturb frozen weights with a Dirichlet centered on the review weights."""
    rng = np.random.default_rng(seed)
    frame = scorecard().set_index("candidate")
    columns = list(WEIGHTS)
    base = np.array([WEIGHTS[c] for c in columns], dtype=float)
    sampled_weights = rng.dirichlet(base * 0.75, size=draws)
    scores = frame[columns].to_numpy(float)
    totals = scores @ sampled_weights.T
    winners = np.argmax(totals, axis=0)
    ranks = np.argsort(np.argsort(-totals, axis=0), axis=0) + 1
    rows = []
    for i, candidate in enumerate(frame.index):
        rows.append(
            {
                "candidate": candidate,
                "win_fraction": float(np.mean(winners == i)),
                "top3_fraction": float(np.mean(ranks[i] <= 3)),
                "mean_rank": float(np.mean(ranks[i])),
                "median_rank": float(np.median(ranks[i])),
                "draws": draws,
            }
        )
    return pd.DataFrame(rows).sort_values(["win_fraction", "top3_fraction"], ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic_scenarios().to_csv(OUT / "synthetic_scenario_metrics.csv", index=False)
    null_bias_simulation().to_csv(OUT / "null_bias_summary.csv", index=False)
    spatial_counterexamples().to_csv(OUT / "spatial_counterexamples.csv", index=False)
    canonical_posthoc().to_csv(OUT / "canonical_posthoc_metrics.csv", index=False)
    fmow_actual_support_null().to_csv(OUT / "fmow_actual_support_null.csv", index=False)
    scorecard().to_csv(OUT / "candidate_scorecard.csv", index=False)
    weight_sensitivity().to_csv(OUT / "weight_sensitivity.csv", index=False)
    print(f"Wrote diagnostics to {OUT}")


if __name__ == "__main__":
    main()
