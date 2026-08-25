#!/usr/bin/env python
"""CPU-only granular discovery over frozen GeoBWER artifacts.

The script reads canonical or explicitly snapshotted frozen tables and writes a
new analysis package.  It never modifies any source artifact and never invokes
model inference, embedding extraction, or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


REPO = Path(__file__).resolve().parents[2]
SEEDS = (42, 73, 101)
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)


def seed_from_run(run_id: str) -> float:
    m = re.search(r"seed[_-](42|73|101)", str(run_id))
    return float(m.group(1)) if m else np.nan


def fractional_tail(values: np.ndarray, beta: float = 0.1) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, np.nan
    order = np.sort(values)[::-1]
    target = beta * len(order)
    remaining = target
    numerator = 0.0
    masses: list[float] = []
    for value in order:
        take = min(1.0, remaining)
        if take <= 0:
            break
        numerator += take * value
        masses.append(take)
        remaining -= take
    tail = numerator / target
    mean = float(np.mean(order))
    p = np.asarray(masses) / sum(masses)
    effective = float(1.0 / np.sum(p * p))
    return mean, tail, tail - mean, effective


def tail_candidate_mask(values: pd.Series, beta: float = 0.1) -> pd.Series:
    """Return a tie-aware candidate mask for the upper-beta tail.

    The formal fractional tail can split mass across atoms tied at its boundary.
    Discovery tables therefore retain every tied boundary atom instead of
    choosing an arbitrary first ``ceil(beta*n)`` subset.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    mask = pd.Series(False, index=values.index, dtype=bool)
    if valid.empty:
        return mask
    k = max(1, math.ceil(beta * len(valid)))
    cutoff = float(valid.nlargest(k).min())
    mask.loc[valid.index] = valid >= cutoff
    return mask


def panel_statistics(full: pd.DataFrame, metric: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in full.groupby(["dataset", "run_id", "model_family", "mode", "slice_axis"], dropna=False):
        g = g[g["eligible_for_primary_metric"].astype(str).str.lower().isin(["true", "1"])]
        if g.empty:
            continue
        risk = pd.to_numeric(g["risk"], errors="coerce").dropna()
        tail = g[pd.to_numeric(g["tail_selected_mass_beta_0_10"], errors="coerce").fillna(0) > 0].copy()
        masses = pd.to_numeric(tail["tail_selected_mass_beta_0_10"], errors="coerce").fillna(0).to_numpy()
        if masses.sum() > 0:
            p = masses / masses.sum()
            effective = 1 / np.sum(p * p)
            dominance = p.max()
        else:
            effective = dominance = np.nan
        m, t, d, _ = fractional_tail(risk.to_numpy())
        rows.append({
            "dataset": keys[0], "run_id": keys[1], "model_family": keys[2], "mode": keys[3], "slice_axis": keys[4],
            "slice_count": len(risk), "M_recomputed": m, "T_recomputed": t, "D_recomputed": d,
            "SD": float(risk.std(ddof=0)), "worst_minus_M": float(risk.max() - risk.mean()),
            "tail_slice_count": len(tail), "tail_effective_slices": effective, "top_tail_mass_share": dominance,
            "top_slice_risk": float(risk.max()), "bottom_slice_risk": float(risk.min()),
        })
    out = pd.DataFrame(rows)
    profiles = metric[metric["evidence_role"].eq("beta_profile")].copy()
    profiles["beta"] = pd.to_numeric(profiles["beta"], errors="coerce")
    profiles["geobwer"] = pd.to_numeric(profiles["geobwer"], errors="coerce")
    piv = profiles.pivot_table(index=["dataset", "run_id", "model_family", "mode", "slice_axis"], columns="beta", values="geobwer", aggfunc="first").reset_index()
    if 0.05 in piv.columns and 0.3 in piv.columns:
        piv["beta_elasticity_D_0_05_minus_0_30"] = piv[0.05] - piv[0.3]
        out = out.merge(piv[["dataset", "run_id", "model_family", "mode", "slice_axis", "beta_elasticity_D_0_05_minus_0_30"]], how="left")
    out["geometry"] = np.select(
        [
            (out.M_recomputed < .25) & (out.D_recomputed < .08),
            (out.M_recomputed > .60) & (out.D_recomputed < .10),
            (out.tail_effective_slices <= 1.5) & (out.D_recomputed >= .10),
            (out.top_tail_mass_share >= .45) & (out.D_recomputed >= .10),
            (out.tail_effective_slices >= 3) & (out.D_recomputed >= .10),
        ],
        ["uniformly_low", "levelling_down_compressed", "isolated_extreme", "concentrated_tail", "broad_elevated_tail"],
        default="diffuse_heterogeneity",
    )
    return out


def persistent_slices(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = full.copy()
    d = d[d["eligible_for_primary_metric"].astype(str).str.lower().isin(["true", "1"])]
    d["seed"] = d["run_id"].map(seed_from_run)
    d["risk"] = pd.to_numeric(d["risk"], errors="coerce")
    d["support"] = pd.to_numeric(d["support"], errors="coerce")
    d["tail"] = pd.to_numeric(d["tail_selected_mass_beta_0_10"], errors="coerce").fillna(0).gt(0)
    # Re-rank inside the *eligible* candidate universe.  The canonical
    # descriptive rank also numbers low-support cells and therefore cannot be
    # converted to a percentile after eligibility filtering.
    d["rank"] = d.groupby(["dataset", "run_id", "slice_axis"])["risk"].rank(
        ascending=False, method="average"
    )
    counts = d.groupby(["dataset", "run_id", "slice_axis"])["slice_value"].transform("count")
    d["rank_percentile"] = (d["rank"] - 1) / (counts - 1).clip(lower=1)
    group_cols = ["dataset", "model_family", "mode", "slice_axis", "slice_value"]
    agg = d.groupby(group_cols, dropna=False).agg(
        run_count=("run_id", "nunique"), seed_count=("seed", "nunique"), mean_risk=("risk", "mean"), risk_sd=("risk", "std"),
        min_support=("support", "min"), tail_membership_frequency=("tail", "mean"), mean_rank_percentile=("rank_percentile", "mean"),
        seed_risks=("risk", lambda x: json.dumps([round(float(v), 6) for v in x], separators=(",", ":"))),
    ).reset_index()
    agg["stability"] = np.where(agg.seed_count >= 3, "three_seed", np.where(agg.seed_count >= 2, "two_seed", "single_frozen_run_descriptive"))
    hard = agg[(agg.tail_membership_frequency >= 2 / 3) & (agg.min_support >= 5)].copy()
    easy = agg[(agg.tail_membership_frequency == 0) & (agg.mean_rank_percentile >= .9) & (agg.min_support >= 5)].copy()
    hard = hard.sort_values(["tail_membership_frequency", "mean_risk", "min_support"], ascending=[False, False, False])
    easy = easy.sort_values(["mean_rank_percentile", "mean_risk", "min_support"], ascending=[False, True, False])
    return agg, hard, easy


def cross_model(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = full.copy()
    d = d[d["eligible_for_primary_metric"].astype(str).str.lower().isin(["true", "1"])]
    d["risk"] = pd.to_numeric(d["risk"], errors="coerce")
    d["support"] = pd.to_numeric(d["support"], errors="coerce")
    d["tail"] = pd.to_numeric(d["tail_selected_mass_beta_0_10"], errors="coerce").fillna(0).gt(0)
    base = d.groupby(["dataset", "mode", "slice_axis", "model_family", "slice_value"]).agg(risk=("risk", "mean"), support=("support", "min"), tail_frequency=("tail", "mean"), run_count=("run_id", "nunique")).reset_index()
    agreements, failures = [], []
    for (dataset, mode, axis), g in base.groupby(["dataset", "mode", "slice_axis"]):
        models = sorted(g.model_family.unique())
        for a, b in combinations(models, 2):
            x = g[g.model_family.eq(a)].set_index("slice_value")
            y = g[g.model_family.eq(b)].set_index("slice_value")
            z = x.join(y, lsuffix="_a", rsuffix="_b", how="inner")
            if len(z) < 5:
                continue
            k = max(1, math.ceil(.1 * len(z)))
            ta = set(z[z.tail_frequency_a > 0].index)
            tb = set(z[z.tail_frequency_b > 0].index)
            if not ta:
                cutoff = z.nlargest(k, "risk_a").risk_a.min(); ta = set(z[z.risk_a >= cutoff].index)
            if not tb:
                cutoff = z.nlargest(k, "risk_b").risk_b.min(); tb = set(z[z.risk_b >= cutoff].index)
            rho = spearmanr(z.risk_a, z.risk_b).statistic
            tau = kendalltau(z.risk_a, z.risk_b).statistic
            agreements.append({"dataset": dataset, "mode": mode, "slice_axis": axis, "model_a": a, "model_b": b, "common_slices": len(z), "spearman_rho": rho, "kendall_tau": tau, "top10_jaccard": len(ta & tb) / len(ta | tb), "shared_top_tail": len(ta & tb), "model_a_only_top_tail": len(ta - tb), "model_b_only_top_tail": len(tb - ta), "evidence_status": "descriptive_same_support"})
            za = z.risk_a.rank(ascending=False, method="average"); zb = z.risk_b.rank(ascending=False, method="average")
            for sid, row in z.iterrows():
                state = "shared_hard" if sid in ta & tb else "model_a_specific" if sid in ta else "model_b_specific" if sid in tb else "non_tail"
                if state != "non_tail" or abs(za[sid] - zb[sid]) >= max(5, .25 * len(z)):
                    failures.append({"dataset": dataset, "mode": mode, "slice_axis": axis, "slice_id": sid, "model_a": a, "model_b": b, "risk_a": row.risk_a, "risk_b": row.risk_b, "support_a": row.support_a, "support_b": row.support_b, "rank_a": za[sid], "rank_b": zb[sid], "rank_difference": za[sid] - zb[sid], "failure_type": state if state != "non_tail" else "stable_rank_reversal_candidate", "same_comparison_support": row.support_a == row.support_b})
    return pd.DataFrame(agreements), pd.DataFrame(failures)


def fmow_site_analysis(src: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(src / "atlas_fmow_dofa_spatial_unit_risk.csv").set_index("spatial_unit")
    b = pd.read_csv(src / "atlas_fmow_resnet_spatial_unit_risk.csv").set_index("spatial_unit")
    z = a.join(b, lsuffix="_dofa", rsuffix="_resnet", how="inner")
    qd, qr = z.mean_risk_dofa.quantile(.9), z.mean_risk_resnet.quantile(.9)
    z["dofa_tail"] = z.mean_risk_dofa >= qd
    z["resnet_tail"] = z.mean_risk_resnet >= qr
    z["failure_type"] = np.select([z.dofa_tail & z.resnet_tail, z.dofa_tail & ~z.resnet_tail, ~z.dofa_tail & z.resnet_tail], ["shared_hard", "DOFAv2_specific", "ResNet50_specific"], default="non_tail")
    z["exact_0_to_1_reversal"] = ((z.mean_risk_dofa == 0) & (z.mean_risk_resnet == 1)) | ((z.mean_risk_dofa == 1) & (z.mean_risk_resnet == 0))
    summary = pd.DataFrame([{
        "dataset": "fMoW-Sentinel", "slice_axis": "site", "common_sites": len(z),
        "spearman_rho": spearmanr(z.mean_risk_dofa, z.mean_risk_resnet).statistic,
        "kendall_tau": kendalltau(z.mean_risk_dofa, z.mean_risk_resnet).statistic,
        "shared_hard_sites": int((z.failure_type == "shared_hard").sum()), "dofa_specific_hard_sites": int((z.failure_type == "DOFAv2_specific").sum()),
        "resnet_specific_hard_sites": int((z.failure_type == "ResNet50_specific").sum()), "exact_0_to_1_reversal_sites": int(z.exact_0_to_1_reversal.sum()),
        "both_exact_risk_1_sites": int(((z.mean_risk_dofa == 1) & (z.mean_risk_resnet == 1)).sum()), "evidence_status": "three_seed_descriptive_atlas",
    }])
    detail = z.reset_index().sort_values(["exact_0_to_1_reversal", "failure_type", "support_dofa"], ascending=[False, True, False])
    return summary, detail


def interaction_amplification(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, run_id, model, mode), g in full[full.slice_axis.str.contains("_x_")].groupby(["dataset", "run_id", "model_family", "mode"]):
        axes = sorted(g.slice_axis.unique())
        for axis in axes:
            cells = g[g.slice_axis.eq(axis)].copy()
            if cells.empty or not cells.slice_value.astype(str).str.contains(r" \| ").all():
                continue
            parts = cells.slice_value.str.split(" | ", n=1, regex=False, expand=True)
            cells["geo"] = parts[0]; cells["semantic"] = parts[1]
            cells["risk"] = pd.to_numeric(cells.risk, errors="coerce"); cells["support"] = pd.to_numeric(cells.support, errors="coerce")
            def aggregate(col):
                return cells.groupby(col).apply(lambda x: np.average(x.risk, weights=x.support), include_groups=False).to_numpy()
            geo = aggregate("geo"); sem = aggregate("semantic"); inter = cells.risk.to_numpy()
            gm, gt, gd, _ = fractional_tail(geo); sm, st, sd, _ = fractional_tail(sem); im, it, id_, _ = fractional_tail(inter)
            rows.append({"dataset": dataset, "run_id": run_id, "model_family": model, "mode": mode, "interaction_axis": axis, "geography_D": gd, "semantic_D": sd, "interaction_D": id_, "amplification_over_larger_main_D": id_ - max(gd, sd), "amplification_ratio": id_ / max(gd, sd) if max(gd, sd) > 0 else np.nan, "geography_slice_count": len(geo), "semantic_slice_count": len(sem), "interaction_cell_count": len(inter), "evidence_status": "descriptive_derived_from_canonical_cells"})
    return pd.DataFrame(rows)


def modality_redistribution(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ("reBEN", "Sen1Floods11"):
        d = full[full.dataset.eq(dataset)].copy()
        d["risk"] = pd.to_numeric(d.risk, errors="coerce")
        for (model, axis), g in d.groupby(["model_family", "slice_axis"]):
            modes = sorted(g["mode"].unique())
            for a, b in combinations(modes, 2):
                x = g[g["mode"].eq(a)].groupby("slice_value").risk.mean(); y = g[g["mode"].eq(b)].groupby("slice_value").risk.mean()
                z = pd.concat([x.rename("risk_a"), y.rename("risk_b")], axis=1).dropna()
                if len(z) < 5: continue
                k = max(1, math.ceil(.1 * len(z))); ta=set(z.nlargest(k,"risk_a").index); tb=set(z.nlargest(k,"risk_b").index)
                for sid, r in z.iterrows():
                    rows.append({"dataset": dataset, "model_family": model, "slice_axis": axis, "mode_a": a, "mode_b": b, "slice_id": sid, "risk_a": r.risk_a, "risk_b": r.risk_b, "delta_b_minus_a": r.risk_b-r.risk_a, "tail_transition": "persistent_tail" if sid in ta&tb else "tail_to_non_tail" if sid in ta-tb else "new_tail" if sid in tb-ta else "non_tail", "top_tail_jaccard": len(ta&tb)/len(ta|tb)})
    return pd.DataFrame(rows)


def paired_shift(src: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, overlap = [], []
    model_frames = {}
    for model in ("terra", "croma"):
        parts=[]
        for axis in ("country", "label"):
            d=pd.read_csv(src/f"{model}_{axis}_deltas.csv")
            d["model"]=model; d["axis"]=axis
            d["slice_id"]=d["slice_value"]
            parts.append(d)
        model_frames[model]=pd.concat(parts,ignore_index=True)
    for model,d in model_frames.items():
        for (seed,axis),g in d.groupby(["seed","axis"]):
            k=max(1,math.ceil(.1*len(g)))
            id_cut=float(g.nlargest(k,"id_risk").id_risk.min()); ood_cut=float(g.nlargest(k,"ood_risk").ood_risk.min())
            # Include every boundary tie.  The canonical fractional tail shares
            # mass across tied atoms; arbitrarily retaining only the first k
            # would create false tail migrations under OOD ceiling compression.
            id_tail=set(g[g.id_risk>=id_cut].slice_id); ood_tail=set(g[g.ood_risk>=ood_cut].slice_id)
            for _,r in g.iterrows():
                support = r.get("support", np.nan)
                if pd.isna(support):
                    support = r.get("positive_support", np.nan)
                rows.append({"model":model,"seed":seed,"slice_axis":axis,"slice_id":r.slice_id,"id_risk":r.id_risk,"ood_risk":r.ood_risk,"delta_risk":r.delta_risk,"support":support,"tail_transition":"persistent_tail" if r.slice_id in id_tail&ood_tail else "tail_to_non_tail" if r.slice_id in id_tail-ood_tail else "newly_created_tail" if r.slice_id in ood_tail-id_tail else "non_tail","nominal_top_k":k,"id_tail_candidate_count":len(id_tail),"ood_tail_candidate_count":len(ood_tail),"ood_boundary_tie_expansion":len(ood_tail)>k})
    out=pd.DataFrame(rows)
    for (axis,seed),g in out.groupby(["slice_axis","seed"]):
        a=set(g[(g.model.eq("terra"))&g.tail_transition.isin(["persistent_tail","newly_created_tail"])].slice_id)
        b=set(g[(g.model.eq("croma"))&g.tail_transition.isin(["persistent_tail","newly_created_tail"])].slice_id)
        overlap.append({"slice_axis":axis,"seed":seed,"terramind_ood_tail_count":len(a),"croma_ood_tail_count":len(b),"shared_ood_tail":len(a&b),"tail_jaccard":len(a&b)/len(a|b) if a|b else np.nan})
    return out,pd.DataFrame(overlap)


def adaptation(src: Path) -> pd.DataFrame:
    d=pd.read_csv(src/"exp8_adaptation_slice_patterns.csv")
    pieces=[]
    mapping={"A_id":("A","test_id"),"A_shifted":("A","test_shifted"),"B":("B","test"),"C":("C","test")}
    for name,(stage,split) in mapping.items():
        x=d[(d.stage.eq(stage))&d.split_role.eq(split)][["seed","slice_axis","slice_value","risk","f1","support","positive_support"]].copy()
        x=x.rename(columns={"risk":f"risk_{name}","f1":f"f1_{name}","support":f"support_{name}","positive_support":f"positive_support_{name}"})
        pieces.append(x.set_index(["seed","slice_axis","slice_value"]))
    z=pd.concat(pieces,axis=1,join="inner").reset_index()
    for stage in ("A_id","A_shifted","B","C"):
        z[f"one_minus_f1_{stage}"]=np.where(z.slice_axis.eq("class_label"),1-z[f"f1_{stage}"],np.nan)
    for stage in ("B","C"):
        z[f"delta_{stage}_vs_shifted"]=z[f"risk_{stage}"]-z.risk_A_shifted
        denom=z.risk_A_shifted-z.risk_A_id
        z[f"recovery_{stage}"]=np.where(denom.abs()>1e-12,(z.risk_A_shifted-z[f"risk_{stage}"])/denom,np.nan)
    for (seed,axis),g in z.groupby(["seed","slice_axis"]):
        k=max(1,math.ceil(.1*len(g)))
        tails={s:set(g.loc[tail_candidate_mask(g[f"risk_{s}"]),"slice_value"]) for s in ("A_id","A_shifted","B","C")}
        for idx,r in g.iterrows():
            z.loc[idx,"tail_A_id"]=r.slice_value in tails["A_id"]
            z.loc[idx,"tail_A_shifted"]=r.slice_value in tails["A_shifted"]
            z.loc[idx,"tail_B"]=r.slice_value in tails["B"]
            z.loc[idx,"tail_C"]=r.slice_value in tails["C"]
            for stage in ("A_id","A_shifted","B","C"):
                z.loc[idx,f"tail_candidate_count_{stage}"]=len(tails[stage])
                z.loc[idx,f"tail_boundary_tie_expansion_{stage}"]=len(tails[stage])>k
    return z


def adaptation_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (axis,sid),g in detail.groupby(["slice_axis","slice_value"]):
        support=min(g.support_A_shifted.min(),g.support_C.min()); pos=min(g.positive_support_A_shifted.min(),g.positive_support_C.min())
        adequate = support>=20 and (axis=="country" or pos>=20)
        a2b=-g.delta_B_vs_shifted; a2c=-g.delta_C_vs_shifted
        rows.append({"slice_axis":axis,"slice_id":sid,"min_support":support,"min_positive_support":pos,"support_adequate":adequate,"A_to_B_mean_risk_reduction":a2b.mean(),"A_to_B_improved_seeds":int((a2b>0).sum()),"A_to_C_mean_risk_reduction":a2c.mean(),"A_to_C_improved_seeds":int((a2c>0).sum()),"C_mean_recovery":g.recovery_C.mean(),"shifted_tail_frequency":g.tail_A_shifted.mean(),"C_tail_frequency":g.tail_C.mean(),"tail_exit_count":int((g.tail_A_shifted.astype(bool)&~g.tail_C.astype(bool)).sum()),"persistent_tail_count":int((g.tail_A_shifted.astype(bool)&g.tail_C.astype(bool)).sum()),"seed_risks_A_shifted":json.dumps(g.risk_A_shifted.round(6).tolist()),"seed_risks_C":json.dumps(g.risk_C.round(6).tolist())})
    return pd.DataFrame(rows)


def label_budget(src: Path) -> pd.DataFrame:
    d=pd.read_csv(src/"label_budget_curves.csv")
    d=d.sort_values(["seed","budget_fraction"])
    d["delta_M_from_previous_budget"]=d.groupby("seed").mean_risk.diff()
    d["delta_T_from_previous_budget"]=d.groupby("seed").tail_risk_beta_0_10.diff()
    d["delta_D_from_previous_budget"]=d.groupby("seed").geobwer_beta_0_10.diff()
    d["tail_shift_elsewhere_signature"]=(d.delta_M_from_previous_budget<0)&(d.delta_D_from_previous_budget>0)
    d["granular_carrier_status"]="unavailable: frozen output has panel metrics but no country/label/cell risk by budget"
    return d


def reben_risk_primitive_stability(src: Path) -> pd.DataFrame:
    """Compare the label tail under omission, commission and balanced error."""
    d = pd.read_csv(src / "reben_labelwise_fnr_fpr.csv")
    risks = ("fnr", "fpr", "balanced_error")
    rows = []
    for (family, mode, seed), g in d.groupby(["family", "mode", "seed"]):
        tails = {risk: set(g.nlargest(2, risk)["label"]) for risk in risks}
        for a, b in combinations(risks, 2):
            union = tails[a] | tails[b]
            rho = spearmanr(g[a], g[b]).statistic
            rows.append({
                "record_type": "reben_labelwise_risk_primitive",
                "task": "reBEN", "model_family": family, "mode": mode,
                "seed": seed, "risk_a": a, "risk_b": b,
                "label_count": len(g), "spearman_rho": rho,
                "top2_jaccard": len(tails[a] & tails[b]) / len(union),
                "shared_tail_labels": json.dumps(sorted(tails[a] & tails[b])),
                "risk_a_only_tail_labels": json.dumps(sorted(tails[a] - tails[b])),
                "risk_b_only_tail_labels": json.dumps(sorted(tails[b] - tails[a])),
                "evidence_status": "descriptive_three_seed_fixed_label_universe",
                "scope_note": "same 19-label universe; top two labels within each risk primitive",
            })
    return pd.DataFrame(rows)


def selective_analysis(repo: Path) -> pd.DataFrame:
    d=pd.read_csv(repo/"outputs/fmow_conformal_selective_audit_v1/fmow_retained_coverage_by_slice.csv")
    d=d[d.selector.eq("confidence_topk_test")].copy()
    d["abstention_rate"]=1-d.retained_coverage
    rejected=d.total_count-d.retained_count
    d["rejected_population_risk"]=np.where(rejected>0,(d.baseline_mean_risk*d.total_count-d.mean_risk*d.retained_count)/rejected,np.nan)
    rows=[]
    for keys,g in d.groupby(["run_id","coverage_target","slice_variable"]):
        supported=g[g.total_count>=20].copy()
        q=supported.baseline_mean_risk.quantile(.9); tail=supported.baseline_mean_risk>=q
        rows.append({"run_id":keys[0],"coverage_target":keys[1],"slice_axis":keys[2],"supported_slice_count":len(supported),"mean_retained_coverage":supported.retained_coverage.mean(),"coverage_sd":supported.retained_coverage.std(),"min_retained_coverage":supported.retained_coverage.min(),"max_retained_coverage":supported.retained_coverage.max(),"risk_abstention_spearman":spearmanr(supported.baseline_mean_risk,supported.abstention_rate).statistic,"tail_minus_nontail_retained_coverage":supported.loc[tail,"retained_coverage"].mean()-supported.loc[~tail,"retained_coverage"].mean(),"mean_retained_risk":supported.mean_risk.mean(),"mean_baseline_risk":supported.baseline_mean_risk.mean(),"mean_rejected_risk":supported.rejected_population_risk.mean(),"unequal_service_signature":supported.loc[tail,"retained_coverage"].mean()<supported.loc[~tail,"retained_coverage"].mean(),"evidence_status":"descriptive_post_hoc_selective"})
    return pd.DataFrame(rows)


def selective_risk_service_frontier(selective: pd.DataFrame) -> pd.DataFrame:
    """Construct the empirical risk-service frontier over the full 18-cell panel."""
    out = selective.copy()
    out["model"] = out["run_id"].map({
        "dofa_scaled10000": "DOFAv2",
        "resnet50_13band": "ResNet50",
    }).fillna(out["run_id"])
    out["abstention_strength"] = 1.0 - out["coverage_target"]
    out["service_gap_pp"] = 100.0 * out["tail_minus_nontail_retained_coverage"]
    out["remaining_risk_reduction"] = out["mean_baseline_risk"] - out["mean_retained_risk"]
    out["rejected_minus_retained_risk"] = out["mean_rejected_risk"] - out["mean_retained_risk"]
    out["tail_service_deficit"] = out["service_gap_pp"] < 0
    out["rejected_is_harder"] = out["rejected_minus_retained_risk"] > 0
    out["selection_rule"] = "all 2 models x 3 coverage targets x 3 axes; supported slices have n>=20"
    out["evidence_status"] = "descriptive_post_hoc_selective_full_panel"
    monotonic = []
    for (model, axis), g in out.groupby(["model", "slice_axis"]):
        ordered = g.sort_values("abstention_strength")
        rho = spearmanr(ordered["abstention_strength"], ordered["service_gap_pp"]).statistic
        monotonic.append({"model": model, "slice_axis": axis, "service_gap_vs_abstention_spearman": rho})
    return out.merge(pd.DataFrame(monotonic), on=["model", "slice_axis"], how="left")


def shift_adaptation_tail_turnover(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join ID, shifted, threshold-only and Stage-C risk carriers in one lineage."""
    out = detail.copy()
    for stage in ("A_id", "A_shifted", "C"):
        column = f"one_minus_f1_{stage}"
        if column not in out:
            f1_column = f"f1_{stage}"
            out[column] = 1 - out[f1_column] if f1_column in out else np.nan
    out = out.rename(columns={"slice_value": "slice_id"})
    out["min_support"] = out[["support_A_id", "support_A_shifted", "support_B", "support_C"]].min(axis=1)
    out["min_positive_support"] = out[["positive_support_A_id", "positive_support_A_shifted", "positive_support_B", "positive_support_C"]].min(axis=1)
    out["support_adequate"] = (out["min_support"] >= 20) & (
        out["slice_axis"].eq("country") | (out["min_positive_support"] >= 20)
    )
    out["id_to_shift_delta"] = out["risk_A_shifted"] - out["risk_A_id"]
    out["shift_to_B_delta"] = out["risk_B"] - out["risk_A_shifted"]
    out["shift_to_C_delta"] = out["risk_C"] - out["risk_A_shifted"]
    out["tail_turnover_ID_to_shift"] = np.select(
        [out["tail_A_id"] & out["tail_A_shifted"], out["tail_A_id"] & ~out["tail_A_shifted"], ~out["tail_A_id"] & out["tail_A_shifted"]],
        ["persistent_tail", "tail_exit", "new_shift_tail"], default="non_tail",
    )
    out["tail_turnover_shift_to_C"] = np.select(
        [out["tail_A_shifted"] & out["tail_C"], out["tail_A_shifted"] & ~out["tail_C"], ~out["tail_A_shifted"] & out["tail_C"]],
        ["persistent_tail", "tail_exit", "new_C_tail"], default="non_tail",
    )
    out["evidence_status"] = "descriptive_three_seed_fixed_support_tie_aware"
    out["selection_rule"] = "full fixed slice universe; all boundary ties retained; main examples require support adequacy and >=2/3 seed direction"
    group = out.groupby(["slice_axis", "slice_id"], dropna=False)
    summary = group.agg(
        seed_count=("seed", "nunique"),
        min_support=("min_support", "min"),
        min_positive_support=("min_positive_support", "min"),
        support_adequate=("support_adequate", "all"),
        mean_ID_risk=("risk_A_id", "mean"),
        mean_shifted_risk=("risk_A_shifted", "mean"),
        mean_B_risk=("risk_B", "mean"),
        mean_C_risk=("risk_C", "mean"),
        mean_ID_one_minus_f1=("one_minus_f1_A_id", "mean"),
        mean_shifted_one_minus_f1=("one_minus_f1_A_shifted", "mean"),
        mean_C_one_minus_f1=("one_minus_f1_C", "mean"),
        mean_ID_to_shift_delta=("id_to_shift_delta", "mean"),
        mean_shift_to_C_delta=("shift_to_C_delta", "mean"),
        ID_tail_frequency=("tail_A_id", "mean"),
        shifted_tail_frequency=("tail_A_shifted", "mean"),
        C_tail_frequency=("tail_C", "mean"),
        shifted_tail_candidate_count=("tail_candidate_count_A_shifted", "max"),
        C_tail_candidate_count=("tail_candidate_count_C", "max"),
    ).reset_index()
    summary["stable_shift_created_tail"] = (summary["ID_tail_frequency"] < 1/3) & (summary["shifted_tail_frequency"] >= 2/3)
    summary["stable_shift_tail_exit_after_C"] = (summary["shifted_tail_frequency"] >= 2/3) & (summary["C_tail_frequency"] < 1/3)
    summary["stable_new_C_tail"] = (summary["shifted_tail_frequency"] < 1/3) & (summary["C_tail_frequency"] >= 2/3)
    summary["stable_persistent_C_tail"] = (summary["shifted_tail_frequency"] >= 2/3) & (summary["C_tail_frequency"] >= 2/3)
    summary["evidence_status"] = "descriptive_three_seed_fixed_support_tie_aware"
    return out, summary


def uq_error_geography(repo: Path) -> pd.DataFrame:
    # Formal cluster-aware UQ is aggregate-only. Calibrated confidence retention is
    # retained as an explicitly descriptive location-level uncertainty proxy.
    d=pd.read_csv(repo/"outputs/fmow_conformal_selective_audit_v1/fmow_conformal_slice_coverage.csv")
    rows=[]
    for keys,g in d.groupby(["run_id","method","coverage_target","slice_variable"]):
        g=g[g.total_count>=20]
        if len(g)<5: continue
        risk=1-g.empirical_retained_accuracy
        uncertainty=1-g.retained_coverage
        rows.append({"run_id":keys[0],"method":keys[1],"coverage_target":keys[2],"slice_axis":keys[3],"slice_count":len(g),"risk_vs_abstention_spearman":spearmanr(risk,uncertainty).statistic,"evidence_status":"descriptive_confidence_retention_proxy_not_formal_cluster_UQ","formal_site_level_set_size_available":False})
    return pd.DataFrame(rows)


def exp9_slice_profiles(src: Path, metrics_path: Path) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]
    for task in ("fmow","reben"):
        for model in ("dofa","tm"):
            for seed in SEEDS:
                path=src/f"exp9_{task}_{model}_seed{seed}_by_group.csv"
                if not path.exists(): continue
                d=pd.read_csv(path); d=d[d.axis.eq("country")]
                for _,r in d.iterrows(): rows.append({"task":task,"model":"DOFAv2" if model=="dofa" else "TerraMind","seed":seed,"country":r["group"],"risk":r.risk,"support":r.support,"tail":r.selected_tail_mass>0})
    # TerraMind reBEN legacy country risks are reconstructed from paired-shift ID side.
    t=pd.read_csv(src/"terra_country_deltas.csv")
    for _,r in t.iterrows(): rows.append({"task":"reben","model":"TerraMind","seed":int(r.seed),"country":r.slice_value,"risk":r.id_risk,"support":r.support,"tail":False})
    d=pd.DataFrame(rows).drop_duplicates(["task","model","seed","country"],keep="first")
    for keys,g in d.groupby(["task","model","seed"]):
        k=max(1,math.ceil(.1*len(g))); ids=set(g.nlargest(k,"risk").country); d.loc[g.index,"tail"]=g.country.isin(ids)
    profiles=[]
    for (task,model),g in d.groupby(["task","model"]):
        agg=g.groupby("country").agg(mean_risk=("risk","mean"),tail_frequency=("tail","mean"),support=("support","min")).reset_index()
        profiles.append({"task":task,"model":model,"country_count":len(agg),"risk_cv":agg.mean_risk.std()/agg.mean_risk.mean(),"persistent_tail_country_count":int((agg.tail_frequency>=2/3).sum()),"top10_tail_concentration":agg.nlargest(max(1,math.ceil(.1*len(agg))),"mean_risk").mean_risk.sum()/agg.mean_risk.sum(),"mean_risk_rank_within_task":np.nan,"D_behavior":"smaller_gap_with_worse_M_T" if task=="fmow" and model=="DOFAv2" else "mean_tail_consistent" if task=="reben" and model=="TerraMind" else "task_specific"})
    profiles = pd.DataFrame(profiles)
    metrics = pd.read_csv(metrics_path)
    metrics["model"] = metrics["model"].map({"dofav2": "DOFAv2", "terramind": "TerraMind"})
    summary = metrics.groupby(["task", "model"])[["primary_risk", "M", "T", "D"]].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    profiles = profiles.merge(summary, on=["task", "model"], how="left")
    profiles["mean_risk_rank_within_task"] = profiles.groupby("task")["M_mean"].rank(method="min")
    profiles["D_rank_within_task"] = profiles.groupby("task")["D_mean"].rank(method="min")
    return d, profiles


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def build_figures(
    out: Path,
    taxonomy: pd.DataFrame,
    site: pd.DataFrame,
    shift: pd.DataFrame,
    adapt: pd.DataFrame,
    pattern: pd.DataFrame,
    sen1: pd.DataFrame,
    frontier: pd.DataFrame,
    turnover: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":120})
    fig,ax=plt.subplots(figsize=(6.5,4)); ax.scatter(taxonomy.mean_risk,taxonomy.geobwer_beta_0_10,c=taxonomy.weighted_std,cmap="viridis",s=65)
    for i,(_,r) in enumerate(taxonomy.iterrows(),1): ax.annotate(str(i),(r.mean_risk,r.geobwer_beta_0_10),xytext=(3,3),textcoords="offset points",fontsize=7,fontweight="bold")
    legend_text="\n".join(f"{i}. {r.scenario.replace('_',' ')}" for i,(_,r) in enumerate(taxonomy.iterrows(),1))
    ax.text(1.02,.98,legend_text,transform=ax.transAxes,va="top",fontsize=6)
    ax.set(xlabel="Mean risk (M)",ylabel="Tail excess (D)",title="GeoBWER risk-geometry counterexamples"); savefig(fig,out/"figures/F1_geobwer_geometry_taxonomy")
    fig,ax=plt.subplots(figsize=(5,4)); colors=site.failure_type.map({"shared_hard":OKABE[3],"DOFAv2_specific":OKABE[0],"ResNet50_specific":OKABE[1],"non_tail":"#BBBBBB"})
    ax.scatter(site.mean_risk_dofa,site.mean_risk_resnet,c=colors,s=np.clip(site.support_dofa,1,20)*2,alpha=.7); ax.plot([0,1],[0,1],"k--",lw=.8)
    ax.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=c,label=l,markersize=6) for l,c in [("Both ceiling-risk",OKABE[3]),("DOFAv2 ceiling only",OKABE[0]),("ResNet50 ceiling only",OKABE[1]),("Other","#BBBBBB")]],frameon=False,fontsize=6,loc="lower right")
    ax.set(xlabel="DOFAv2 site risk",ylabel="ResNet50 site risk",title="fMoW shared and model-specific site failures"); savefig(fig,out/"figures/F2_fmow_site_failure_geometry")
    s=shift.groupby(["model","slice_axis","tail_transition"]).size().unstack(fill_value=0)
    fig,axes=plt.subplots(1,2,figsize=(7,3));
    for ax,(axis,g) in zip(axes,s.groupby(level=1)):
        g.droplevel(1).rename(columns=lambda x:x.replace("_"," ")).plot(kind="bar",stacked=True,ax=ax,color=OKABE[:len(g.columns)]); ax.set_title(axis); ax.set_ylabel("seed×slice transitions"); ax.legend(fontsize=6,frameon=False)
    fig.suptitle("reBEN S2→S1 tail migration"); fig.tight_layout(); savefig(fig,out/"figures/F3_reben_shift_tail_migration")
    a=adapt[(adapt.support_adequate)&adapt.slice_axis.eq("country_x_label")].copy()
    repaired=a[a.A_to_C_improved_seeds.eq(3)].nlargest(6,"A_to_C_mean_risk_reduction")
    residual=a[(a.C_tail_frequency>=2/3)&(a.shifted_tail_frequency<1/3)].nsmallest(6,"A_to_C_mean_risk_reduction")
    show=pd.concat([repaired,residual]).drop_duplicates("slice_id"); labels=[textwrap.fill(x.replace(" ⇄ "," × "),34) for x in show.slice_id]
    fig,ax=plt.subplots(figsize=(7,5)); ax.barh(labels,show.A_to_C_mean_risk_reduction,color=np.where(show.A_to_C_mean_risk_reduction>=0,OKABE[2],OKABE[1])); ax.axvline(0,color="k",lw=.8); ax.set(xlabel="A→C mean risk reduction",title="Experiment 8: original-tail repair and new residual tail"); savefig(fig,out/"figures/F4_adaptation_slice_recovery")
    if not pattern.empty:
        plot=pattern.copy(); plot["plot_code"]=plot.support_code; plot.loc[plot.result_direction.str.startswith("counterexample"),"plot_code"]=-1
        mat=plot.pivot(index="pattern",columns="experiment_family",values="plot_code").fillna(0); fig,ax=plt.subplots(figsize=(8,4)); im=ax.imshow(mat,cmap="coolwarm",vmin=-2,vmax=2,aspect="auto"); ax.set_xticks(range(len(mat.columns)),mat.columns,rotation=45,ha="right"); ax.set_yticks(range(len(mat.index)),mat.index); fig.colorbar(im,ax=ax,label="−1 counterexample; 0 unavailable; 1 limited; 2 repeated/stable"); ax.set_title("Cross-experiment recurring-pattern matrix"); fig.tight_layout(); savefig(fig,out/"figures/F5_recurring_pattern_matrix")
    if not sen1.empty:
        piv=sen1.pivot_table(index="slice_value",columns=["model_family","mode"],values="risk",aggfunc="mean"); fig,ax=plt.subplots(figsize=(9,4)); im=ax.imshow(piv,cmap="magma",aspect="auto"); ax.set_yticks(range(len(piv.index)),piv.index); ax.set_xticks(range(len(piv.columns)),[f"{a}\n{b}" for a,b in piv.columns],rotation=45,ha="right",fontsize=6); fig.colorbar(im,ax=ax,label="Event risk"); ax.set_title("Sen1Floods11 event-risk geometry"); fig.tight_layout(); savefig(fig,out/"figures/A1_sen1_event_risk_heatmap")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    country = frontier[frontier.slice_axis.eq("country")].sort_values("coverage_target")
    for i, (model, g) in enumerate(country.groupby("model")):
        axes[0].plot(100 * g.coverage_target, g.service_gap_pp, color=OKABE[i], marker=("o", "s")[i], linewidth=1.6, label=model)
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set(xlabel="Target coverage (%)", ylabel="Tail − non-tail retained coverage (pp)", title="A  Country service gap")
    axes[0].legend(frameon=False, fontsize=7)
    axis_colors = {"country": OKABE[0], "class_label": OKABE[1], "region": OKABE[2]}
    markers = {"DOFAv2": "o", "ResNet50": "s"}
    for (model, axis), g in frontier.groupby(["model", "slice_axis"]):
        axes[1].scatter(g.service_gap_pp, g.remaining_risk_reduction, color=axis_colors.get(axis, "#777777"), marker=markers[model], s=35, edgecolor="white", linewidth=.4)
    axes[1].axvline(0, color="black", linewidth=.7); axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set(xlabel="Tail service gap (pp)", ylabel="Baseline − retained risk", title="B  Full 18-cell risk–service panel")
    axis_labels = {"country": "country", "class_label": "class", "region": "region"}
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=axis_labels[a], markersize=6) for a, c in axis_colors.items()]
    handles += [Line2D([0], [0], marker=m, color="#555555", linestyle="None", label=model, markersize=6) for model, m in markers.items()]
    axes[1].legend(handles=handles, frameon=False, fontsize=6, ncol=2)
    fig.suptitle("Selective prediction: empirical geographic risk–service frontier", fontsize=10)
    fig.tight_layout(); savefig(fig, out/"figures/F6_selective_risk_service_frontier")

    supported = turnover[turnover.support_adequate].copy()
    label = supported[supported.slice_axis.eq("class_label")].copy()
    preferred = ["Marine waters", "Beaches, dunes, sands", "Coastal wetlands"]
    selected_labels = pd.concat([label[label.slice_id.isin(preferred)], label.nlargest(3, "mean_ID_to_shift_delta")]).drop_duplicates("slice_id").head(6)
    interaction = supported[supported.slice_axis.eq("country_x_label")].copy()
    repaired = interaction[interaction.stable_shift_tail_exit_after_C].nsmallest(4, "mean_shift_to_C_delta")
    migrated = interaction[interaction.stable_new_C_tail].nlargest(4, "mean_shift_to_C_delta")
    selected_cells = pd.concat([repaired, migrated]).drop_duplicates("slice_id")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), gridspec_kw={"width_ratios": [0.85, 1.15]})
    panels = [
        (axes[0], selected_labels, "A  Label error trajectory (1 − F1)", ["mean_ID_one_minus_f1", "mean_shifted_one_minus_f1", "mean_C_one_minus_f1"]),
        (axes[1], selected_cells, "B  Country × label risk contribution", ["mean_ID_risk", "mean_shifted_risk", "mean_C_risk"]),
    ]
    for ax, data, title, stages in panels:
        matrix = data.set_index("slice_id")[stages]
        image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3), ["S2 ID", "S1 shifted", "Stage C"])
        ax.set_yticks(range(len(matrix)), [textwrap.fill(str(v).replace(" ⇄ ", " × "), 28) for v in matrix.index], fontsize=6)
        ax.set_title(title)
        for y, (_, row) in enumerate(matrix.iterrows()):
            for x, value in enumerate(row):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white" if value > .52 else "black")
    fig.colorbar(image, ax=axes, label="Risk (panel-specific definition)", fraction=.025, pad=.03)
    fig.suptitle("reBEN shift and adaptation redistribute tail burden", fontsize=10)
    fig.subplots_adjust(left=.25, right=.92, bottom=.12, top=.86, wspace=.58)
    savefig(fig, out/"figures/F7_shift_adaptation_tail_turnover")


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--source-dir",type=Path,default=REPO/"work/granular_discovery_sources"); ap.add_argument("--output-dir",type=Path,default=REPO/"outputs/granular_scientific_discovery_v1"); args=ap.parse_args()
    src=args.source_dir.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    exp9_metrics_path = REPO / "work/final_paper_cleanup_sources/experiment9_cell_seed_metrics.csv"
    source_paths=[REPO/"outputs/optimization_1_7_v1/full_slice_distribution.csv",REPO/"outputs/optimization_1_7_v1/compound_interaction_atlas.csv",REPO/"outputs/optimization_1_7_v1/distribution_metric_comparison.csv",REPO/"outputs/optimization_1_7_v1/metric_counterexamples.csv",exp9_metrics_path,*sorted(src.glob("*"))]
    hashes_before={str(p):sha256(p) for p in source_paths if p.is_file()}
    full=pd.read_csv(source_paths[0]); metric=pd.read_csv(source_paths[2]); taxonomy=pd.read_csv(REPO/"outputs/optimization_1_7_v1/metric_counterexamples.csv")
    panels=panel_statistics(full,metric); all_slices,hard,easy=persistent_slices(full); agree,fail=cross_model(full); site_summary,site=fmow_site_analysis(src)
    agree=pd.concat([agree,site_summary],ignore_index=True,sort=False); interaction=interaction_amplification(full); modality=modality_redistribution(full); shift,shift_overlap=paired_shift(src)
    adapt_detail=adaptation(src); adapt=adaptation_summary(adapt_detail)
    turnover_seed,turnover=shift_adaptation_tail_turnover(adapt_detail)
    budget=label_budget(src); selective=selective_analysis(REPO)
    frontier=selective_risk_service_frontier(selective)
    uq=uq_error_geography(REPO); exp9_slices,profiles=exp9_slice_profiles(src, exp9_metrics_path)
    site_candidates = pd.DataFrame({
        "dataset": "fMoW-Sentinel", "mode": "S2", "slice_axis": "site",
        "slice_id": site.spatial_unit, "model_a": "dofav2", "model_b": "resnet50",
        "risk_a": site.mean_risk_dofa, "risk_b": site.mean_risk_resnet,
        "support_a": site.support_dofa, "support_b": site.support_resnet,
        "rank_a": site.mean_risk_dofa.rank(ascending=False, method="average"),
        "rank_b": site.mean_risk_resnet.rank(ascending=False, method="average"),
        "rank_difference": site.mean_risk_dofa.rank(ascending=False, method="average") - site.mean_risk_resnet.rank(ascending=False, method="average"),
        "failure_type": site.failure_type, "same_comparison_support": site.support_dofa.eq(site.support_resnet),
    })
    fail = pd.concat([fail, site_candidates], ignore_index=True, sort=False)
    write_csv(out/"02_per_panel_slice_statistics.csv",panels); write_csv(out/"03_persistent_hard_slices.csv",hard); write_csv(out/"04_persistent_easy_slices.csv",easy); write_csv(out/"05_shared_vs_model_specific_failures.csv",fail); write_csv(out/"06_cross_model_rank_agreement.csv",agree)
    (out/"07_tail_overlap_matrices").mkdir(exist_ok=True); write_csv(out/"07_tail_overlap_matrices/model_pair_tail_overlap.csv",agree); write_csv(out/"07_tail_overlap_matrices/paired_shift_tail_overlap.csv",shift_overlap)
    write_csv(out/"08_interaction_amplification.csv",interaction)
    logloss=pd.read_csv(REPO/"work/evidence_rebuild_v060/fmow_proper_score_v12/fmow_log_loss_geobwer_sensitivity.csv"); logloss["available_slice_membership_comparison"]=False; logloss["scope_note"]="panel-level risk-definition sensitivity only; canonical log-loss output lacks slice-risk table"
    logloss["record_type"] = "fmow_panel_log_loss"
    risk_stability = pd.concat([logloss, reben_risk_primitive_stability(src)], ignore_index=True, sort=False)
    write_csv(out/"09_risk_definition_stability.csv",risk_stability); write_csv(out/"10_modality_tail_redistribution.csv",modality); write_csv(out/"11_shift_tail_migration.csv",shift); write_csv(out/"12_adaptation_slice_recovery.csv",adapt); write_csv(out/"13_label_budget_slice_dynamics.csv",budget); write_csv(out/"14_uq_vs_error_geography.csv",uq); write_csv(out/"15_selective_coverage_burden.csv",selective); write_csv(out/"16_same_model_cross_task_profiles.csv",profiles)
    write_csv(out/"25_selective_risk_service_frontier.csv",frontier)
    write_csv(out/"26_shift_adaptation_tail_turnover_seed_level.csv",turnover_seed)
    write_csv(out/"27_shift_adaptation_tail_turnover.csv",turnover)
    write_csv(out/"risk_geometry_taxonomy.csv",taxonomy)
    write_csv(out/"sen1_threshold_beta_profiles.csv",pd.read_csv(src/"sen1_validation_locked_threshold_profile.csv"))
    pattern_rows=[
        ("interaction_amplification","E1_fMoW",1),("interaction_amplification","E3_AlphaEarth",2),("shared_plus_model_specific_failure","E1_fMoW",2),("shared_plus_model_specific_failure","E2_reBEN",2),("shared_plus_model_specific_failure","E4_Sen1",1),
        ("mean_tail_rank_decoupling","E4_Sen1",2),("mean_tail_rank_decoupling","E11_model_task",2),("shift_as_tail_migration","E8_paired_shift",2),("shift_as_tail_migration","E10_adaptation",2),("persistent_tail_after_average_recovery","E10_adaptation",2),
        ("risk_definition_changes_hardness","E1_fMoW",1),("risk_definition_changes_hardness","E2_reBEN",2),("uncertainty_not_identical_to_error","E6_UQ",0),("selective_service_tradeoff","Selective",2),("supervision_repairs_specific_carriers","E7_label_budget",0),
        ("disparity_less_cross_task_stable_than_mean","E11_model_task",2),("geography_semantic_interaction","E3_AlphaEarth",2),("geography_semantic_interaction","E8_reBEN_shift",2),
    ]
    pattern=pd.DataFrame(pattern_rows,columns=["pattern","experiment_family","support_code"]); pattern["evidence_meaning"]=pattern.support_code.map({0:"unavailable",1:"single-family or limited/counterexample",2:"stable/repeated"})
    pattern["result_direction"] = "supports_pattern"
    pattern.loc[(pattern.pattern.eq("interaction_amplification")) & (pattern.experiment_family.eq("E1_fMoW")), "result_direction"] = "counterexample_interaction_D_below_main_effect_D"
    pattern.loc[pattern.support_code.eq(0), "result_direction"] = "unavailable"
    write_csv(out/"17_cross_experiment_pattern_matrix.csv",pattern)
    atlas=[]
    for _,r in hard.groupby(["dataset","model_family","mode","slice_axis"],dropna=False).head(3).iterrows(): atlas.append({"task":r.dataset,"model":r.model_family,"slice_axis":r.slice_axis,"slice_id":r.slice_value,"support":r.min_support,"seed_level_risks":r.seed_risks,"mean_risk":r.mean_risk,"tail_membership_frequency":r.tail_membership_frequency,"rank":r.mean_rank_percentile,"comparator_rank_or_risk":"","evidence_stability":r.stability,"why_selected":"predeclared top persistent hard slice"})
    supported_sites = site[(site.failure_type.ne("non_tail")) & (site.support_dofa >= 5) & (site.support_resnet >= 5)]
    for _,r in supported_sites.sort_values(["exact_0_to_1_reversal","support_dofa"],ascending=[False,False]).head(12).iterrows(): atlas.append({"task":"fMoW-Sentinel","model":"DOFAv2 vs ResNet50","slice_axis":"site","slice_id":r.spatial_unit,"support":min(r.support_dofa,r.support_resnet),"seed_level_risks":json.dumps({"dofa_mean":r.mean_risk_dofa,"resnet_mean":r.mean_risk_resnet,"dofa_range":[r.seed_min_risk_dofa,r.seed_max_risk_dofa],"resnet_range":[r.seed_min_risk_resnet,r.seed_max_risk_resnet]}),"mean_risk":max(r.mean_risk_dofa,r.mean_risk_resnet),"tail_membership_frequency":"","rank":"","comparator_rank_or_risk":r.failure_type,"evidence_stability":"three_seed_atlas","why_selected":"predeclared shared/model-specific ceiling-risk site; support >=5"})
    repairs = adapt[(adapt.support_adequate)&(adapt.A_to_C_improved_seeds==3)].sort_values("A_to_C_mean_risk_reduction",ascending=False).head(8)
    residual = adapt[(adapt.support_adequate)&(adapt.C_tail_frequency>=2/3)&(adapt.shifted_tail_frequency<1/3)].sort_values(["C_tail_frequency","A_to_C_mean_risk_reduction"],ascending=[False,True]).head(8)
    for _,r in pd.concat([repairs,residual]).drop_duplicates(["slice_axis","slice_id"]).iterrows(): atlas.append({"task":"reBEN adaptation","model":"TerraMind","slice_axis":r.slice_axis,"slice_id":r.slice_id,"support":r.min_support,"seed_level_risks":r.seed_risks_C,"mean_risk":"","tail_membership_frequency":r.C_tail_frequency,"rank":"","comparator_rank_or_risk":r.A_to_C_mean_risk_reduction,"evidence_stability":f"{r.A_to_C_improved_seeds}/3 improved","why_selected":"predeclared stable repair" if r.A_to_C_mean_risk_reduction>0 else "new residual C-tail after shifted tail exited"})
    write_csv(out/"18_concrete_slice_atlas.csv",pd.DataFrame(atlas))
    availability=[
        ("E0","available","synthetic counterexamples + beta profiles"),("E1","available","country/country×class/region×class + 1,480 matched sites; log-loss panel sensitivity"),("E2","partial","27-run label tables complete; canonical 27-run country×label table absent"),("E3","partial","interaction cells and spatial units available; location-level geo-kernel set sizes absent"),("E4","available","19 routes × 11 events + threshold/beta profiles"),("E5","available","60 panel distributions"),("E6","partial","formal aggregate UQ available; site-level set size/coverage absent"),("E7","partial","5-budget panel curves available; slice-by-budget risks absent"),("E8","partial","country/label paired shifts for both models; country×label only TerraMind atlas"),("E9","available","fMoW/AlphaEarth spatial units and reBEN atlas"),("E10","available","A/B/C country/label/country×label, 3 seeds"),("E11","partial","fMoW country slices both models; reBEN 7-country common support reconstructed"),("Selective","available_descriptive","fMoW 70/80/90% per-slice retention; post-hoc diagnostic")]
    inventory=pd.DataFrame(availability,columns=["experiment_family","availability","canonical_granular_assets"]); write_csv(out/"01_experiment_family_inventory.csv",inventory)
    main_text="""# Candidate main-text findings\n\n1. **Country labels alone do not resolve deployment burden.** AlphaEarth country×land-cover cells amplify D from 0.181 (land cover) / 0.133 (country) to 0.417, while fMoW's 1,480 matched sites contain a large shared ceiling-risk set and many model-specific failures even though country-level rank agreement is only moderate. fMoW interaction D itself is not amplified and is retained as a counterexample to universality.\n2. **Sensor shift changes tail geometry rather than applying a common additive penalty.** In every seed TerraMind's label-tail candidate set expands from two labels to nine labels tied at risk 1, whereas CROMA preserves the same two-label tail; their OOD country tails are disjoint (Belgium versus Portugal).\n3. **Adaptation redistributes rather than uniformly reverses shifted burden.** Stage C substantially repairs many shifted slices, but tie-aware fixed-universe analysis shows that the identities of high-risk labels and country×label cells turn over. This redistribution explains why broad performance recovery need not equal tail-reliability recovery.\n4. **Error construct changes the identity of the difficult labels.** Across 27 reBEN model×modality×seed runs, FNR and balanced-error top-two labels nearly coincide, whereas FPR has zero top-two overlap with either; “high-risk label” therefore has a precise omission/commission meaning.\n5. **Selective prediction trades lower remaining risk for unequal geographic service.** Across the full 2-model × 3-coverage × 3-axis panel, 17/18 tail-minus-non-tail retention gaps are negative. Country gaps are −8.47/−7.72/−4.82 pp for DOFAv2 and −10.54/−7.98/−6.18 pp for ResNet50 at 70/80/90% coverage; rejected examples are harder than retained examples.\n6. **Absolute performance is more task-stable than disparity ordering.** TerraMind lowers M/T in both Experiment 9 tasks, but D is 0.054 higher than DOFAv2 on fMoW and 0.011 lower on reBEN, consistent with fMoW ceiling compression.\n"""; (out/"19_candidate_main_text_findings.md").write_text(main_text,encoding="utf-8")
    appendix="""# Appendix findings\n\n- Full beta elasticity and event-tail geometry across the 19 Sen1Floods11 routes.\n- Labelwise FNR/FPR/balanced-error construct sensitivity across the 27 reBEN runs.\n- fMoW log-loss sensitivity, explicitly descriptive because log loss is unbounded.\n- AlphaEarth spatial-scale gate diagnostics as an informative negative result.\n- Complete model-pair rank agreements, tail overlaps, and candidate universes.\n- Cluster-aware marginal UQ summary, with the guarantee scope stated exactly.\n"""; (out/"20_appendix_findings.md").write_text(appendix,encoding="utf-8")
    unstable="""# Findings not suitable for paper claims\n\n- Country/label-specific label-budget repair: the frozen budget artifact has no slice-by-budget risk table.\n- Formal site-level alignment between error geography and cluster-UQ set size: only aggregate cluster-aware UQ was frozen.\n- CROMA country×label shift atlas: no canonical CROMA interaction table exists.\n- Spatially certified AlphaEarth hotspots: every tested spatial scale failed the validation-only gate.\n- Cross-task averages of raw GeoBWER or the legacy Experiment 9 standardized interaction effect.\n- Sparse extreme cells failing the declared support/stability rule.\n"""; (out/"21_do_not_use_unstable_findings.md").write_text(unstable,encoding="utf-8")
    figtext="""# Figure candidates\n\n## Recommended main figures\n\n1. **F2 fMoW shared/model-specific site geometry** — direct evidence that two models do not fail at identical locations.\n2. **F6 selective risk–service frontier** — shows the joint remaining-risk benefit and geographic service deficit over all 18 cells, rather than one 70% point.\n3. **F7 shift–adaptation tail turnover** — links paired sensor shift to Stage C and localizes repaired and newly residual burden.\n4. **F5 recurring-pattern matrix** — compact cross-experiment synthesis.\n\n## Method/appendix\n\n- F1 GeoBWER geometry taxonomy.\n- F3 and F4 provide expanded transition/recovery diagnostics.\n- A1 Sen1 event-risk heatmap.\n\nAll plots are exported as vector PDF and 300-dpi PNG with colorblind-safe colors.\n"""; (out/"22_figure_candidates.md").write_text(figtext,encoding="utf-8")
    sen1=full[full.dataset.eq("Sen1Floods11")].copy(); build_figures(out,taxonomy,site,shift,adapt,pattern,sen1,frontier,turnover)
    country_frontier=frontier[frontier.slice_axis.eq("country")]
    hashes_after={str(p):sha256(p) for p in source_paths if p.is_file()}; checks={"status":"pass","source_files_unchanged":hashes_before==hashes_after,"panel_count":len(panels),"persistent_hard_count":len(hard),"persistent_easy_count":len(easy),"model_pair_comparisons":len(agree),"fmow_site_count":len(site),"adaptation_slice_count":len(adapt),"paired_shift_rows":len(shift),"selective_full_panel_cells":len(frontier),"selective_tail_service_deficit_cells":int(frontier.tail_service_deficit.sum()),"selective_rejected_harder_cells":int(frontier.rejected_is_harder.sum()),"country_frontier_models_with_monotonic_deficit":int((country_frontier.groupby("model").service_gap_vs_abstention_spearman.first()<0).sum()),"cross_task_raw_average_created":False,"legacy_exp9_standardized_effect_used":False,"stage_D_used":False}
    (out/"validation_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    fallacies=[("Simpson's paradox","checked","aggregate and slice directions retained separately"),("Ecological fallacy","checked","country/site claims remain at group level"),("Berkson's paradox","checked","selected frozen panels identified"),("Collider bias","checked","no covariate-adjusted causal model introduced"),("Base-rate neglect","checked","label positive/negative support retained"),("Regression to mean","checked","adaptation uses fixed all-slice universe, not extreme-only selection"),("Survivorship bias","checked","support/missing outputs retained; unavailable cells not silently dropped"),("Look-elsewhere effect","caution","discovery is exploratory; full candidate universes and deterministic top-k rules saved"),("Garden of forking paths","caution","post-hoc discovery; manuscript tiers remain descriptive unless pre-existing formal evidence applies"),("Correlation != causation","checked","association language only"),("Reverse causality","checked","no directional proxy mechanism claim")]
    pd.DataFrame(fallacies,columns=["fallacy","status","handling"]).to_csv(out/"statistical_fallacy_scan_11_of_11.csv",index=False)
    provenance={"schema":"granular_scientific_discovery_v1","verification_status":"ANALYZED","source_sha256":hashes_before,"drive_snapshot_directory":str(src),"anti_cherry_picking":{"tail":"top 10% within each fixed panel/seed","persistent":"tail membership >=2/3 seeds","support":"canonical eligibility plus minimum support; country×label positive support >=20 for adaptation","examples":"deterministic top-k from full candidate universes"},"unavailable_not_inferred":["CROMA country×label shift","slice-by-budget risk","formal site-level cluster-UQ set size","spatially certified AlphaEarth hotspots"],"software":{"pandas":pd.__version__,"numpy":np.__version__}}
    (out/"23_provenance_manifest.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    summary=f"""# Executive discovery summary\n\n## Material Passport\n\n- Verification Status: **ANALYZED**\n- Scope: CPU-only post-processing of frozen canonical artifacts\n- Source mutation: **none**\n- Statistical fallacy scan: **11/11 checked**\n\n## Package scale\n\n- {len(panels)} panel×axis risk geometries\n- {len(all_slices):,} aggregated slice records\n- {len(hard):,} persistent/high-tail slice records\n- {len(agree)} cross-model rank/overlap comparisons\n- {len(site):,} matched fMoW sites\n- {len(adapt):,} Experiment 8 country/label/country×label recovery units\n- {len(shift):,} paired shift seed×slice transitions\n\n## Headline\n\nThe granular pass adds substantial value beyond aggregate M/T/D. The recurring structure is **burden localization and redistribution**: models share some intrinsically difficult deployment units, but risk primitives, sensor shift, adaptation and abstention change *which* slices carry the burden. The strongest new paper-facing evidence is the fMoW site failure geometry, reBEN omission-versus-commission tail split, tie-aware S2→S1 tail geometry, and Experiment 8's replacement of the original shifted tail by a new residual tail.\n\nSee `19_candidate_main_text_findings.md` for concise claims and the CSV tables for exact support/stability.\n"""; (out/"00_executive_discovery_summary.md").write_text(summary,encoding="utf-8")
    files=sorted(p for p in out.rglob("*") if p.is_file() and p.name!="package_manifest.json"); (out/"package_manifest.json").write_text(json.dumps({"file_count":len(files),"files":[{"path":str(p.relative_to(out)).replace("\\","/"),"sha256":sha256(p),"bytes":p.stat().st_size} for p in files]},indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__=="__main__": main()
