"""Analyze frozen AlphaEarth--WorldCover disagreement against boundary distance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def residualize(values: np.ndarray, frame: pd.DataFrame, controls: list[str]) -> np.ndarray:
    design = pd.get_dummies(frame[controls].astype(str), drop_first=True, dtype=float)
    matrix = np.column_stack([np.ones(len(frame)), design.to_numpy(dtype=float)])
    return values - matrix @ np.linalg.lstsq(matrix, values, rcond=None)[0]


def partial_spearman(frame: pd.DataFrame) -> float:
    y = residualize(rank(frame.risk.to_numpy()), frame, ["country_iso3", "class_label"])
    x = residualize(rank(np.log1p(frame.boundary_distance_m.to_numpy())), frame, ["country_iso3", "class_label"])
    return float(np.corrcoef(x, y)[0, 1])


def spatial_bootstrap(frame: pd.DataFrame, x_resid: np.ndarray, y_resid: np.ndarray, n_boot: int, seed: int):
    groups = {key: value.index.to_numpy() for key, value in frame.groupby("spatial_block_id")}
    keys = np.asarray(sorted(groups), dtype=object)
    if len(keys) < 4:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        selected = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[key] for key in selected])
        values.append(float(np.corrcoef(x_resid[indices], y_resid[indices])[0, 1]))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975)), len(values)


def plot_result(bins: pd.DataFrame, summary: dict, output: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(bins))
    se = bins["std"] / np.sqrt(bins["count"].clip(lower=1))
    ax.errorbar(x, bins["mean"], yerr=1.96 * se, fmt="o-", color="#0072B2", markerfacecolor="white", capsize=2, lw=1.2)
    ax.set_xticks(x, bins["distance_bin_m"].astype(str), rotation=35, ha="right")
    ax.set_xlabel("Distance to nearest WorldCover class edge (m)")
    ax.set_ylabel("Mean disagreement risk")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E7EB", lw=.5)
    ax.text(-.14, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax = axes[1]
    estimate = float(summary["partial_spearman_controlling_country_and_class"])
    low, high = float(summary["partial_spatial_cluster_ci_low"]), float(summary["partial_spatial_cluster_ci_high"])
    ax.axvline(0, color="#6F7780", lw=.8)
    ax.errorbar([estimate], [0], xerr=[[estimate - low], [high - estimate]], fmt="o", color="#D55E00", capsize=3)
    ax.set_yticks([0], ["Partial Spearman ρ"])
    ax.set_xlabel("Association with log(1 + boundary distance)")
    ax.set_ylim(-.7, .7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(-.14, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"alphaearth_boundary_distance_diagnostic.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from rsfm_fairness_audit.paper_supplementary import write_csv, write_json

    predictions = pd.read_csv(args.predictions)
    boundary = pd.read_csv(args.boundary)
    required_pred = {"sample_id", "risk", "country_iso3", "spatial_block_id", "class_label"}
    required_boundary = {"sample_id", "boundary_distance_m"}
    if not required_pred.issubset(predictions) or not required_boundary.issubset(boundary):
        raise ValueError(f"Missing fields: predictions={required_pred-set(predictions)}, boundary={required_boundary-set(boundary)}")
    boundary = boundary.drop_duplicates("sample_id")
    frame = predictions.merge(boundary[["sample_id", "boundary_distance_m"]], on="sample_id", how="inner", validate="one_to_one")
    frame["risk"] = pd.to_numeric(frame.risk, errors="coerce")
    frame["boundary_distance_m"] = pd.to_numeric(frame.boundary_distance_m, errors="coerce")
    frame = frame[np.isfinite(frame.risk) & np.isfinite(frame.boundary_distance_m) & (frame.boundary_distance_m >= 0)].copy()
    if len(frame) != len(predictions):
        raise ValueError(f"Boundary merge must retain the complete frozen evaluation support; retained {len(frame)}/{len(predictions)} rows.")
    rho = float(pd.Series(frame.risk).corr(pd.Series(np.log1p(frame.boundary_distance_m)), method="spearman"))
    ranked_risk = rank(frame.risk.to_numpy())
    ranked_distance = rank(np.log1p(frame.boundary_distance_m.to_numpy()))
    y_resid = residualize(ranked_risk, frame, ["country_iso3", "class_label"])
    x_resid = residualize(ranked_distance, frame, ["country_iso3", "class_label"])
    partial = float(np.corrcoef(x_resid, y_resid)[0, 1])
    low, high, valid = spatial_bootstrap(frame.reset_index(drop=True), x_resid, y_resid, args.n_boot, 20260911)
    summary = [{
        "sample_count": len(frame), "spatial_block_count": frame.spatial_block_id.nunique(),
        "raw_spearman_risk_vs_log1p_distance": rho,
        "partial_spearman_controlling_country_and_class": partial,
        "partial_spatial_cluster_ci_low": low, "partial_spatial_cluster_ci_high": high,
        "valid_bootstrap_replicates": valid,
        "bootstrap_semantics": "spatial-cluster bootstrap of fixed full-sample rank residuals",
        "interpretation": "WorldCover-proxy boundary diagnostic; not independent ground truth or causal attribution",
    }]
    edges = [-.001, 10, 30, 100, 300, 1000, np.inf]
    labels = ["0-10", "10-30", "30-100", "100-300", "300-1000", ">1000"]
    frame["distance_bin_m"] = pd.cut(frame.boundary_distance_m, bins=edges, labels=labels)
    bins = frame.groupby("distance_bin_m", observed=False).risk.agg(["mean", "count", "std"]).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "alphaearth_boundary_distance_summary.csv", summary)
    bins.to_csv(args.output_dir / "alphaearth_risk_by_boundary_distance.csv", index=False)
    plot_result(bins, summary[0], args.output_dir)
    write_json(args.output_dir / "manifest.json", {
        "schema": "geobwer.paper_supplement.alphaearth_boundary.v1", "status": "complete",
        "distance_source": "ESA WorldCover 2021 class-edge distance sampled in Earth Engine",
        "reference_product_is_proxy": True, "frozen_outputs_modified": False, "model_rerun": False,
    })
    print(json.dumps(summary[0], indent=2))


if __name__ == "__main__":
    main()
