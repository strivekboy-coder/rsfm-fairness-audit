"""Compute audit-safe cluster-bootstrap intervals for selected headline M/T/D."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def first_present(columns, names):
    for name in names:
        if name in columns:
            return name
    return None


def derive_reben_tile(value: str) -> str:
    match = re.search(r"(?:^|_)(T\d{2}[A-Z]{3})(?:_|$)", str(value).upper())
    return match.group(1) if match else ""


def verify_canonical_point(result: dict, spec: dict) -> dict:
    summary_path = str(spec.get("canonical_summary_path", "")).strip()
    if not summary_path:
        return {"point_reconstruction_verified": False, "point_verification_source": "unavailable"}
    path = Path(summary_path)
    if not path.is_file():
        raise ValueError(f"Canonical point summary is missing: {path}")
    frame = pd.read_csv(path)
    axis_col = first_present(frame.columns, ["axis", "slice_variable"])
    if not axis_col:
        raise ValueError(f"Canonical summary lacks axis field: {path}")
    requested = str(spec.get("canonical_axis", spec.get("axis", "")))
    aliases = {requested, "country_iso3" if requested == "country" else requested}
    rows = frame[frame[axis_col].astype(str).isin(aliases)].copy()
    if "audit_measure" in rows and str(spec.get("balance_col", "")):
        balanced = rows[rows.audit_measure.astype(str).str.lower().eq("balanced")]
        if len(balanced):
            rows = balanced
    if len(rows) != 1:
        raise ValueError(f"Expected one canonical {requested} summary row in {path}; found {len(rows)}")
    row = rows.iloc[0]
    expected = {
        "M": float(row["mean_risk"]),
        "T": float(row["tail_risk"]),
        "D": float(row["bwer"]),
    }
    differences = {metric: abs(float(result[metric]) - value) for metric, value in expected.items()}
    if any(value > 1e-9 for value in differences.values()):
        observed = {metric: float(result[metric]) for metric in expected}
        raise ValueError(f"Reconstructed point estimand does not match canonical M/T/D: expected={expected}, observed={observed}, abs_diff={differences}")
    return {
        "point_reconstruction_verified": True,
        "point_verification_source": str(path),
        "point_verification_max_abs_difference": max(differences.values()),
    }


def plot_intervals(results: list[dict], output: Path) -> None:
    usable = pd.DataFrame([row for row in results if row.get("status") == "computed"])
    if usable.empty:
        return
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
    usable["label"] = usable.apply(lambda row: f"{row['task']} · {row['model']} · seed {row['seed']}", axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, max(2.6, .27 * len(usable) + 1.25)), sharey=True, constrained_layout=True)
    for index, (ax, metric) in enumerate(zip(axes, ("M", "T", "D"))):
        y = np.arange(len(usable))
        value = usable[metric].astype(float).to_numpy()
        low = usable[f"{metric}_ci_low"].astype(float).to_numpy()
        high = usable[f"{metric}_ci_high"].astype(float).to_numpy()
        ax.errorbar(value, y, xerr=np.vstack([value - low, high - value]), fmt="o", color="#0072B2" if metric != "D" else "#D55E00", capsize=2, lw=.9, markersize=3)
        ax.set_yticks(y, usable.label if index == 0 else [])
        ax.set_xlabel(metric if metric == "M" else rf"${metric}_{{0.10}}$")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#E2E7EB", lw=.5)
        ax.text(-.16, 1.03, chr(97 + index), transform=ax.transAxes, fontweight="bold", fontsize=10)
    axes[0].invert_yaxis()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"headline_mtd_cluster_intervals.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def aggregate(path: Path, *, axis: str, balance_col: str = "", chunksize: int = 250000):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    risk_col = first_present(header, ["risk", "risk_binary_error", "risk_0_1", "error"])
    can_derive_label_error = {"label_true", "label_prediction"}.issubset(header)
    cluster_col = first_present(header, ["source_tile_id", "site_id", "original_sequence_key", "spatial_block_id", "location_id", "event_id"])
    sample_col = first_present(header, ["sample_id", "patch_id", "unit_id"])
    if axis == "country":
        slice_cols = [first_present(header, ["country", "country_iso3"])]
    elif axis == "label":
        slice_cols = [first_present(header, ["class_label", "worldcover_class_name", "landcover"])]
    elif axis == "country_label":
        slice_cols = [first_present(header, ["country", "country_iso3"]), first_present(header, ["class_label", "worldcover_class_name", "landcover"])]
    elif axis == "event":
        slice_cols = [first_present(header, ["event_id", "event"])]
    else:
        raise ValueError(f"Unknown axis: {axis}")
    if (not risk_col and not can_derive_label_error) or any(value is None for value in slice_cols):
        raise ValueError(f"Cannot resolve risk/slice columns in {path}: risk={risk_col}, slices={slice_cols}")
    risk_inputs = [risk_col] if risk_col else ["label_true", "label_prediction"]
    if balance_col and balance_col not in header:
        raise ValueError(f"Missing requested balance column {balance_col!r} in {path}")
    usecols = sorted({*risk_inputs, *(value for value in slice_cols if value), *(value for value in (cluster_col, sample_col, balance_col) if value)})
    totals = defaultdict(lambda: [0.0, 0])
    resolved_risk = risk_col or "derived_label_mismatch_hamming_primitive"
    for number, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), 1):
        effective_cluster_col = cluster_col
        if not effective_cluster_col:
            if not sample_col:
                raise ValueError(f"No independent cluster or sample lineage in {path}")
            if "reben" in str(path).lower() or "label_audit" in path.name:
                chunk["__cluster"] = chunk[sample_col].astype(str).map(derive_reben_tile)
                effective_cluster_col = "__cluster"
            else:
                raise ValueError(f"No declared cluster column in {path}; sample IDs are not silently treated as independent.")
        elif effective_cluster_col == "location_id" and "fmow" in str(path).lower():
            raise ValueError("Legacy fMoW location_id is not accepted as an independent site without original-sequence lineage.")
        if risk_col:
            risk = pd.to_numeric(chunk[risk_col], errors="coerce")
        else:
            truth = pd.to_numeric(chunk["label_true"], errors="coerce")
            prediction = pd.to_numeric(chunk["label_prediction"], errors="coerce")
            risk = (truth != prediction).astype(float).where(truth.notna() & prediction.notna())
        valid = risk.notna() & chunk[effective_cluster_col].astype(str).str.len().gt(0)
        for col in slice_cols:
            valid &= chunk[col].astype(str).str.len().gt(0)
        if balance_col:
            valid &= chunk[balance_col].astype(str).str.len().gt(0)
        part_columns = [effective_cluster_col, *slice_cols, *([balance_col] if balance_col else [])]
        part = chunk.loc[valid, part_columns].copy()
        part["__risk"] = risk.loc[valid]
        part["__slice"] = part[slice_cols].astype(str).agg(" × ".join, axis=1)
        group_columns = [effective_cluster_col, "__slice", *([balance_col] if balance_col else [])]
        grouped = part.groupby(group_columns).__risk.agg(["sum", "count"])
        for key_values, row in grouped.iterrows():
            if balance_col:
                cluster, slice_value, balance_value = key_values
            else:
                cluster, slice_value = key_values
                balance_value = ""
            key = (str(cluster), str(slice_value), str(balance_value))
            totals[key][0] += float(row["sum"])
            totals[key][1] += int(row["count"])
        print(f"[cluster aggregate] {path.name} chunk {number}", flush=True)
    resolved_cluster = cluster_col or "derived_source_tile_from_sample_id"
    return totals, {"risk_column": resolved_risk, "cluster_column": resolved_cluster, "slice_columns": slice_cols, "balance_column": balance_col}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from rsfm_fairness_audit.paper_supplementary import ClusterSliceSufficient, cluster_bootstrap_mtd, cluster_bootstrap_standardized_mtd, write_csv, write_json

    specs = json.loads(args.spec_json.read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        raise ValueError("spec-json must contain a list of input contracts.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, inventory = [], []
    for spec in specs:
        path = Path(spec["path"])
        base = {key: spec.get(key, "") for key in ("task", "model", "condition", "seed", "axis", "evidence_scope")}
        if not path.is_file():
            inventory.append({**base, "path": str(path), "status": "missing_canonical_input"})
            continue
        try:
            balance_col = str(spec.get("balance_col", ""))
            totals, lineage = aggregate(path, axis=str(spec["axis"]), balance_col=balance_col)
            records = [ClusterSliceSufficient(cluster=k[0], slice_value=k[1], balance_value=k[2], risk_sum=v[0], count=v[1]) for k, v in totals.items()]
            bootstrap = cluster_bootstrap_standardized_mtd if balance_col else cluster_bootstrap_mtd
            result = bootstrap(records, beta=float(spec.get("beta", .1)), n_boot=args.n_boot, seed=int(spec.get("bootstrap_seed", 20260828)))
            if result.get("status") in {"computed", "unstable_bootstrap_support"}:
                result.update(verify_canonical_point(result, spec))
            results.append({**base, "path": str(path), **lineage, **{k: v for k, v in result.items() if k != "clusters_per_slice"}})
            inventory.append({**base, "path": str(path), "status": result["status"], "minimum_clusters_per_slice": result.get("minimum_clusters_per_slice", 0)})
        except Exception as exc:
            inventory.append({**base, "path": str(path), "status": "unverifiable", "reason": str(exc)})
    write_csv(args.output_dir / "headline_mtd_cluster_intervals.csv", results or inventory)
    write_csv(args.output_dir / "headline_mtd_interval_inventory.csv", inventory)
    plot_intervals(results, args.output_dir)
    write_json(args.output_dir / "manifest.json", {
        "schema": "geobwer.paper_supplement.cluster_mtd_intervals.v1", "status": "complete_with_explicit_unavailable_cases",
        "n_boot": args.n_boot, "input_spec": str(args.spec_json), "frozen_outputs_modified": False,
        "three_seed_sd_is_cluster_ci": False, "model_rerun": False,
    })
    print(json.dumps({"computed_rows": len(results), "inventory_rows": len(inventory), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
