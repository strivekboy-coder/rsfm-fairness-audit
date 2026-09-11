"""Merge DEM export and relate predeclared event descriptors to frozen risk."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DESCRIPTORS = [
    "reference_flood_fraction_mean",
    "s1_water_land_robust_contrast_mean",
    "s2_water_land_robust_contrast_mean",
    "dem_elevation_std_mean",
    "dem_relief_mean",
]


def spearman(x, y):
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def plot_associations(event: pd.DataFrame, associations: list[dict], output: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
    labels = {
        "reference_flood_fraction_mean": "Mapped flood fraction",
        "s1_water_land_robust_contrast_mean": "S1 water-land contrast (IQR units)",
        "s2_water_land_robust_contrast_mean": "S2 water-land contrast (IQR units)",
        "dem_elevation_std_mean": "Elevation SD (m)",
        "dem_relief_mean": "Elevation relief (m)",
    }
    lookup = {row["descriptor"]: row for row in associations}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
    axes = axes.ravel()
    for index, descriptor in enumerate(DESCRIPTORS):
        ax = axes[index]
        frame = event[["event_id", descriptor, "consensus_event_risk"]].dropna()
        ax.scatter(frame[descriptor], frame.consensus_event_risk, s=24, facecolors="white", edgecolors="#0072B2", linewidths=1)
        for row in frame.itertuples():
            ax.annotate(str(row.event_id), (getattr(row, descriptor), row.consensus_event_risk), xytext=(2, 2), textcoords="offset points", fontsize=5.8, color="#4B535B")
        if len(frame) >= 2:
            coefficients = np.polyfit(frame[descriptor], frame.consensus_event_risk, 1)
            xx = np.linspace(frame[descriptor].min(), frame[descriptor].max(), 50)
            ax.plot(xx, coefficients[0] * xx + coefficients[1], color="#D55E00", lw=1, ls="--")
        row = lookup.get(descriptor, {})
        rho = row.get("spearman_rho")
        title = f"Spearman ρ = {float(rho):.2f}" if rho is not None else "Insufficient verified values"
        ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold")
        ax.set_xlabel(labels[descriptor])
        ax.set_ylabel("Consensus event risk" if index % 3 == 0 else "")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-.17, 1.05, chr(97 + index), transform=ax.transAxes, fontweight="bold", fontsize=10)
    axes[-1].axis("off")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"sen1_event_descriptor_associations.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--chip-descriptors", type=Path, required=True)
    parser.add_argument("--dem-export", type=Path, required=True)
    parser.add_argument("--slice-atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from rsfm_fairness_audit.paper_supplementary import write_csv, write_json

    chip = pd.read_csv(args.chip_descriptors)
    dem = pd.read_csv(args.dem_export)
    if "sample_id" not in dem:
        raise ValueError("DEM export lacks sample_id.")
    rename = {}
    for old, new in (("elevation_stdDev", "dem_elevation_std"), ("elevation_min", "dem_elevation_min"), ("elevation_max", "dem_elevation_max"), ("mean", "dem_elevation_mean"), ("stdDev", "dem_elevation_std"), ("min", "dem_elevation_min"), ("max", "dem_elevation_max")):
        if old in dem and new not in dem:
            rename[old] = new
    dem = dem.rename(columns=rename)
    merged = chip.merge(dem, on="sample_id", how="left", validate="one_to_one")
    if {"dem_elevation_min", "dem_elevation_max"}.issubset(merged):
        merged["dem_relief"] = pd.to_numeric(merged.dem_elevation_max, errors="coerce") - pd.to_numeric(merged.dem_elevation_min, errors="coerce")
    else:
        merged["dem_relief"] = np.nan
    numeric = ["reference_flood_fraction", "s1_water_land_robust_contrast", "s2_water_land_robust_contrast", "dem_elevation_std", "dem_relief"]
    for column in numeric:
        if column not in merged:
            merged[column] = np.nan
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    event = merged.groupby("event_id", as_index=False)[numeric].mean().rename(columns={column: f"{column}_mean" for column in numeric})

    slices = pd.read_csv(args.slice_atlas)
    mask = (
        slices.dataset.eq("Sen1Floods11") & slices.slice_axis.eq("event")
        & slices.model_family.isin(["supervised_resnet34_unet", "terramind_v1_base"])
        & slices.eligible_for_primary_metric.astype(str).str.lower().isin(["true", "1"])
    )
    risk = slices.loc[mask].copy()
    risk["risk"] = pd.to_numeric(risk.risk, errors="coerce")
    consensus = risk.groupby("slice_value", as_index=False).risk.mean().rename(columns={"slice_value": "event_id", "risk": "consensus_event_risk"})
    event = event.merge(consensus, on="event_id", how="inner", validate="one_to_one")
    if len(event) != 11:
        raise ValueError(f"Expected 11 event rows after merge; found {len(event)}")
    associations = []
    for descriptor in DESCRIPTORS:
        valid = event[[descriptor, "consensus_event_risk"]].dropna()
        if len(valid) < 8 or valid[descriptor].nunique() < 3:
            associations.append({"descriptor": descriptor, "n_events": len(valid), "status": "insufficient_verified_values"})
            continue
        estimate = spearman(valid[descriptor], valid.consensus_event_risk)
        leave_one_out = [spearman(valid.drop(index=i)[descriptor], valid.drop(index=i).consensus_event_risk) for i in valid.index]
        associations.append({
            "descriptor": descriptor, "n_events": len(valid), "spearman_rho": estimate,
            "leave_one_event_out_min": float(np.nanmin(leave_one_out)),
            "leave_one_event_out_max": float(np.nanmax(leave_one_out)),
            "sign_stable_leave_one_event_out": bool(all(np.sign(value) == np.sign(estimate) for value in leave_one_out if np.isfinite(value))),
            "status": "descriptive_n11",
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event.to_csv(args.output_dir / "sen1_event_descriptors_with_consensus_risk.csv", index=False)
    write_csv(args.output_dir / "sen1_event_descriptor_associations.csv", associations)
    plot_associations(event, associations, args.output_dir)
    write_json(args.output_dir / "manifest.json", {
        "schema": "geobwer.paper_supplement.sen1_event_association.v1", "status": "complete_descriptive",
        "event_count": len(event), "risk_definition": "mean event risk across six replicated U-Net/TerraMind model-modality cells and three seeds",
        "selection_policy": "descriptor families fixed before outcome analysis", "causal_claim": False,
        "frozen_outputs_modified": False, "model_rerun": False,
    })
    print(json.dumps({"events": len(event), "associations": associations}, indent=2))


if __name__ == "__main__":
    main()
