#!/usr/bin/env python
"""Build the final paper-facing package from frozen/derived source snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rsfm_fairness_audit.final_paper_cleanup import (  # noqa: E402
    build_experiment8_tables,
    build_experiment9_tables,
    build_reben_example_tables,
    markdown_table,
    read_csv,
    sha256_file,
    write_csv,
)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_exp9(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=False)
    colors = {"primary_risk": "#0072B2", "M": "#009E73", "T": "#D55E00", "D": "#CC79A7"}
    for ax, task in zip(axes, ("fmow", "reben")):
        rows = [r for r in summary if r["task"] == task]
        x = np.arange(4)
        ax.bar(x, [r["mean_delta_dofav2_minus_terramind"] for r in rows], color=[colors[r["metric"]] for r in rows])
        ax.errorbar(x, [r["mean_delta_dofav2_minus_terramind"] for r in rows], yerr=[r["sd"] for r in rows], fmt="none", ecolor="#222222", capsize=3)
        ax.axhline(0, color="#333333", linewidth=.8)
        ax.set_xticks(x, [r["metric"] for r in rows])
        ax.set_title("fMoW" if task == "fmow" else "reBEN")
        ax.set_ylabel("DOFAv2 − TerraMind (risk; higher is worse)")
        ax.text(.02, .96, "3/3 seeds, same direction", transform=ax.transAxes, va="top", fontsize=8)
    fig.suptitle("Experiment 9: task-wise paired model contrasts", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "experiment9_taskwise_paired_contrasts")


def _plot_exp8(stage, out):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    order = ["A_ID", "A_shifted", "B_threshold", "C_head"]
    labels = ["A: S2 ID", "A: S1 shift", "B: thresholds", "C: S1 head"]
    colors = ["#56B4E9", "#D55E00", "#E69F00", "#009E73"]
    for ax, metric in zip(axes, ("AUROC", "M", "T", "D")):
        rows = {r["paper_stage"]: r for r in stage if r["metric"] == metric}
        means = [rows[k]["mean"] for k in order]
        sds = [rows[k]["sd"] for k in order]
        x = np.arange(4)
        ax.bar(x, means, color=colors)
        ax.errorbar(x, means, yerr=sds, fmt="none", ecolor="#222222", capsize=3)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(metric)
        if metric != "AUROC":
            ax.set_ylabel("Risk (lower is better)")
        else:
            ax.set_ylabel("AUROC (higher is better)")
    fig.suptitle("Experiment 8: adaptation ladder on frozen paired S2→S1 shift", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "experiment8_adaptation_ladder")


def _plot_reben(selected, out):
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    labels = [f"{r['country']} ×\n{r['class_label']}" for r in selected]
    values = [float(r["mean_delta_risk"]) for r in selected]
    errors = [float(r["delta_risk_sd"]) for r in selected]
    colors = ["#D55E00" if v >= 0 else "#0072B2" for v in values]
    x = np.arange(len(values))
    ax.bar(x, values, color=colors)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="#222222", capsize=3)
    ax.axhline(0, color="#222222", linewidth=.8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("TerraMind S1−S2 Δrisk (higher is worse)")
    ax.set_title("reBEN: pre-specified country×label shift archetypes", fontweight="bold")
    ax.text(.01, -.28, "Examples selected deterministically from the saved 190-cell candidate universe; CROMA comparison is label-level.", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    _save(fig, out / "reben_country_label_shift_examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=REPO / "work/final_paper_cleanup_sources")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/final_paper_package_v1")
    args = parser.parse_args()
    src, out = args.source_dir.resolve(), args.output_dir.resolve()
    csv_out, fig_out, table_out, prov_out = (out / p for p in ("paper_facing_csv", "figures", "tables", "provenance"))
    for p in (csv_out, fig_out, table_out, prov_out):
        p.mkdir(parents=True, exist_ok=True)

    source_files = sorted(src.glob("*.csv"))
    source_hashes_before = {str(p): sha256_file(p) for p in source_files}

    exp9_seed, exp9_summary = build_experiment9_tables(read_csv(src / "experiment9_cell_seed_metrics.csv"))
    exp8_stage, exp8_recovery = build_experiment8_tables(
        read_csv(src / "experiment8_adaptation_stage_metrics.csv"),
        read_csv(src / "experiment8_adaptation_recovery.csv"),
    )
    universe, selected = build_reben_example_tables(
        read_csv(src / "reben_terramind_country_label_burden.csv"),
        read_csv(src / "reben_terramind_label_diagnostics.csv"),
        read_csv(src / "reben_croma_label_diagnostics.csv"),
    )

    write_csv(csv_out / "experiment9_taskwise_paired_deltas_v2.csv", exp9_seed)
    write_csv(csv_out / "experiment9_taskwise_contrast_summary_v2.csv", exp9_summary)
    exp9_joint = []
    for task in ("fmow", "reben"):
        by_metric = {r["metric"]: r for r in exp9_summary if r["task"] == task}
        levelling = (
            by_metric["M"]["mean_delta_dofav2_minus_terramind"] > 0
            and by_metric["T"]["mean_delta_dofav2_minus_terramind"] > 0
            and by_metric["D"]["mean_delta_dofav2_minus_terramind"] < 0
        )
        exp9_joint.append({
            "task": task,
            "dofav2_minus_terramind_delta_M": by_metric["M"]["mean_delta_dofav2_minus_terramind"],
            "dofav2_minus_terramind_delta_T": by_metric["T"]["mean_delta_dofav2_minus_terramind"],
            "dofav2_minus_terramind_delta_D": by_metric["D"]["mean_delta_dofav2_minus_terramind"],
            "levelling_down_signature_if_D_read_alone": levelling,
            "joint_interpretation": ("DOFAv2 has worse M and T despite smaller D; smaller D is not an advantage" if levelling else "DOFAv2 has higher M, T and D; no smaller-D levelling-down pattern"),
            "evidence_status": "three-seed descriptive paired pattern",
        })
    write_csv(csv_out / "experiment9_joint_mtd_interpretation_v2.csv", exp9_joint)
    write_csv(csv_out / "experiment8_adaptation_stage_summary.csv", exp8_stage)
    write_csv(csv_out / "experiment8_recovery_summary.csv", exp8_recovery)
    write_csv(csv_out / "reben_shift_example_candidate_universe.csv", universe)
    write_csv(csv_out / "reben_shift_examples_selected.csv", selected)

    audit_card = [
        {"dataset": "fMoW-Sentinel", "task": "62-class location recognition", "deployment_unit": "country/site", "model_panel": "DOFAv2, ResNet50; TerraMind in Exp9", "risk": "0–1 classification error", "metric_scope": "DOFAv2 country, 3-seed mean", "M": .79183432, "T": .99874464, "D": .20691031, "main_finding": "high tail burden and site-specific model reversals; cluster-aware marginal coverage is achievable", "evidence_status": "formal_partial main comparison; formal_confirmed marginal cluster UQ"},
        {"dataset": "reBEN", "task": "19-label land-cover classification", "deployment_unit": "country; label; country×label", "model_panel": "CROMA, TerraMind, ResNet; S1/S2/fusion", "risk": "Hamming loss (plus labelwise sensitivity)", "metric_scope": "TerraMind S2 country, 3-seed mean", "M": .08831335, "T": .11932455, "D": .03101120, "main_finding": "sensor shift creates recurrent but model-specific geographic/label failure geometry", "evidence_status": "formal_partial fixed-universe; paired multi-seed descriptive"},
        {"dataset": "AlphaEarth–WorldCover", "task": "embedding-probe reference-map agreement", "deployment_unit": "country×land-cover", "model_panel": "AlphaEarth embeddings + linear probe", "risk": "disagreement with WorldCover proxy", "metric_scope": "country×land-cover descriptive", "M": .23632868, "T": .59465900, "D": .35833032, "main_finding": "large cross-slice concentration; spatial gate failure prevents formal spatial extrapolation", "evidence_status": "descriptive association; spatial gate not passed"},
        {"dataset": "Sen1Floods11", "task": "flood segmentation", "deployment_unit": "event", "model_panel": "U-Net, TerraMind, Prithvi; S1/S2/fusion routes", "risk": "1 − chip IoU", "metric_scope": "TerraMind fusion seed 73 event example", "M": .44706331, "T": .60241020, "D": .15534690, "main_finding": "mean performance and event-tail burden do not move identically across modality/model routes", "evidence_status": "multi-route descriptive; spatial gate not passed"},
    ]
    write_csv(csv_out / "four_task_primary_audit_card.csv", audit_card)

    claims = [
        {"claim_id": "C1", "claim": "Average performance can conceal concentrated deployment burden; M, T and D expose distinct components.", "canonical_artifact": "four-task canonical GeoBWER summaries", "evidence_status": "cross-task convergent descriptive + task-specific formal evidence", "strong_wording": "Across four EO tasks, tail risk repeatedly exceeded mean risk, and the gap varied by deployment slice.", "material_overstatement_to_avoid": "Do not compare or average raw D across tasks."},
        {"claim_id": "C2", "claim": "S2→S1 shift produces geographically and label-structured burden in reBEN.", "canonical_artifact": "frozen paired-shift outputs; country×label atlas", "evidence_status": "paired multi-seed descriptive", "strong_wording": "The paired shift increased risk in all countries while concentrating additional burden in recurrent labels and country×label cells.", "material_overstatement_to_avoid": "Do not call separately adapted predictors a causal sensor effect."},
        {"claim_id": "C3", "claim": "Threshold recalibration alone is insufficient; an S1-trained head restores most performance and tail reliability.", "canonical_artifact": "Experiment 8 frozen A/B/C tables and validation gate", "evidence_status": "validation-gated ablation; 3-seed test description", "strong_wording": "Threshold-only recalibration was ineffective, whereas frozen-encoder head retraining substantially recovered AUROC, M and T; D recovery was less complete.", "material_overstatement_to_avoid": "Stage D was not run because the pre-specified gate stopped at C."},
        {"claim_id": "C4", "claim": "Model ordering is task dependent and M/T/D must be interpreted jointly.", "canonical_artifact": "Experiment 9 task-wise paired contrasts", "evidence_status": "multi-seed descriptive interaction pattern", "strong_wording": "TerraMind had lower primary risk, M and T on both tasks, but D reversed direction between fMoW and reBEN across all seeds; fMoW therefore exhibits a levelling-down signature if D is read alone.", "material_overstatement_to_avoid": "Do not attribute the pattern to backbone alone or use the deprecated standardized interaction magnitude."},
        {"claim_id": "C5", "claim": "Formal uncertainty statements depend on the deployment unit.", "canonical_artifact": "fMoW cluster-aware conformal/CRC outputs", "evidence_status": "formal_confirmed marginal cluster coverage", "strong_wording": "Cluster-aware calibration achieved the target marginal coverage, at the cost of large prediction sets.", "material_overstatement_to_avoid": "Marginal cluster coverage is not worst-site coverage."},
        {"claim_id": "C6", "claim": "The AlphaEarth geographic kernel comparison is an informative negative result.", "canonical_artifact": "AlphaEarth validation-only spatial gates", "evidence_status": "spatial gate not passed", "strong_wording": "No tested spatial scale passed the pre-specified gate, so the observed heterogeneity remains descriptive rather than spatially certified.", "material_overstatement_to_avoid": "WorldCover is a reference proxy, not perfect ground truth."},
    ]
    write_csv(csv_out / "final_claim_registry.csv", claims)

    placement = [
        {"item": "GeoBWER definition and M/T/D", "placement": "main", "reason": "core estimand"},
        {"item": "Four-task Audit Card", "placement": "main", "reason": "paper-level synthesis"},
        {"item": "Exp8 A/B/C ladder", "placement": "main", "reason": "mechanism/ablation finding"},
        {"item": "Exp9 task-wise paired contrasts", "placement": "main", "reason": "task dependence"},
        {"item": "reBEN country×label examples", "placement": "main or one main + remainder appendix", "reason": "localizes shift burden"},
        {"item": "Selective BWER", "placement": "appendix", "reason": "uncertainty-extension robustness, not core estimand"},
        {"item": "Full beta profiles and labelwise sensitivity", "placement": "appendix", "reason": "robustness/construct sensitivity"},
        {"item": "AlphaEarth geo-kernel gate", "placement": "appendix plus concise main negative", "reason": "informative boundary"},
        {"item": "Exp9 legacy standardized effects", "placement": "do not use", "reason": "mechanically scaled; distinct from Standardised GeoBWER"},
    ]
    write_csv(csv_out / "main_text_appendix_placement.csv", placement)

    _plot_exp9(exp9_summary, fig_out)
    _plot_exp8(exp8_stage, fig_out)
    _plot_reben(selected, fig_out)

    (table_out / "experiment9_taskwise_contrasts.md").write_text(markdown_table(exp9_summary, ["task", "metric", "mean_delta_dofav2_minus_terramind", "sd", "min", "max", "direction_consistency"]), encoding="utf-8")
    (table_out / "experiment8_adaptation_table.md").write_text(markdown_table(exp8_stage, ["paper_stage", "metric", "mean", "sd", "scientific_role"]), encoding="utf-8")
    (table_out / "four_task_audit_card.md").write_text(markdown_table(audit_card, ["dataset", "task", "risk", "metric_scope", "M", "T", "D", "main_finding", "evidence_status"]), encoding="utf-8")

    bullets = """# Manuscript-ready result bullets\n\n- Across the four EO tasks, deployment-tail risk exceeded average risk, while the excess burden (D=T−M) varied by geography, label, event, modality and model.\n- In reBEN, the unchanged S2 head failed under paired S1 input. Validation-only threshold recalibration did not restore reliability, whereas an S1-trained head on the frozen encoder recovered most AUROC, M and T; D recovery was more variable.\n- Experiment 9 showed task-dependent model behavior: TerraMind had lower primary risk, M and T than DOFAv2 on both fMoW and reBEN, while the D contrast reversed between tasks in all three seeds. On fMoW, DOFAv2's smaller D co-occurred with worse M and T—a levelling-down signature that demonstrates why the three quantities must be read jointly.\n- reBEN shift burden was not spatially or semantically uniform. Both TerraMind and CROMA degraded overall, but matched label diagnostics showed different score-transport and country×label failure geometry.\n- fMoW provides a positive formal uncertainty result: cluster-aware calibration achieved marginal coverage, although prediction sets were large and worst-site coverage was not guaranteed.\n- AlphaEarth exhibited pronounced cross-slice disagreement with the WorldCover reference proxy, but no tested spatial scale passed the validation-only certification gate; this is an informative negative for spatial extrapolation.\n"""
    (out / "manuscript_ready_result_bullets.md").write_text(bullets, encoding="utf-8")

    policy = """# Statistical and presentation policy\n\n1. Report Experiment 9 only as task-wise, same-seed DOFAv2−TerraMind contrasts in primary risk, M, T and D. Never average raw GeoBWER across tasks.\n2. The legacy `standardized_effects` / standardized interaction from Experiment 9 is deprecated for paper claims because its scale is mechanically tied to the two-model pooled dispersion. It is not the project method called **Standardised GeoBWER**, which standardises group composition under a prespecified class mixture.\n3. Interpret M, T and D jointly. A lower D is not automatically better if M and T rose (levelling-down check).\n4. Experiment 8 selection stopped at C using validation-only criteria. Test values describe the frozen decision; they did not tune the gate.\n5. Conformal/CRC concerns prediction-set uncertainty. Cluster bootstrap/simultaneous intervals concern uncertainty in risk and GeoBWER estimates. Geo-kernel comparisons remain descriptive where the spatial gate did not pass.\n"""
    (out / "statistical_presentation_policy.md").write_text(policy, encoding="utf-8")

    source_hashes_after = {str(p): sha256_file(p) for p in source_files}
    checks = {
        "status": "pass",
        "source_files_unchanged": source_hashes_before == source_hashes_after,
        "experiment9_cell_count": len(exp9_seed),
        "experiment9_has_cross_task_raw_average": False,
        "experiment9_all_direction_consistent": all(r["direction_consistency"] == "3/3" for r in exp9_summary),
        "experiment9_fmow_levelling_down_signature": exp9_joint[0]["levelling_down_signature_if_D_read_alone"],
        "experiment9_reben_levelling_down_signature": exp9_joint[1]["levelling_down_signature_if_D_read_alone"],
        "experiment8_stage_rows": len(exp8_stage),
        "experiment8_stage_D_present": False,
        "reben_candidate_universe_rows": len(universe),
        "reben_selected_rows": len(selected),
        "audit_card_tasks": len(audit_card),
        "legacy_standardized_effects_packaged": False,
    }
    if not all((checks["source_files_unchanged"], checks["experiment9_all_direction_consistent"], len(universe) == 190, len(audit_card) == 4)):
        raise RuntimeError(checks)
    (out / "consistency_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    provenance = {
        "package": "final_paper_package_v1",
        "derivation": "CPU-only paper-facing transformation; no model inference or frozen-output mutation",
        "source_sha256": source_hashes_before,
        "drive_lineage": {
            "experiment9_analysis": "/content/drive/MyDrive/rsfm_fairness_audit/outputs/experiment9_model_task_analysis_v1",
            "experiment8": "/content/drive/MyDrive/rsfm_fairness_audit/outputs/experiment8_reben_adaptation_v1",
            "reben_paired_shift": "canonical paired-shift summary/atlas artifacts",
        },
        "presentation_decisions": [
            "task-wise raw contrasts only for Experiment 9",
            "Stage D is a validation-gated stop, not a completed stage",
            "CROMA country×label values are not inferred where no canonical table exists",
        ],
    }
    (prov_out / "provenance_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = """# Final paper package\n\nThis directory contains CPU-only, paper-facing derivatives of frozen GeoBWER evidence. It does not replace, amend or back-fill any canonical output.\n\n## Main files\n\n- `paper_facing_csv/`: auditable numeric tables, claim registry and placement map.\n- `figures/`: publication-ready PNG (300 dpi) and vector PDF figures.\n- `tables/`: compact Markdown tables for manuscript drafting.\n- `manuscript_ready_result_bullets.md`: concise result language.\n- `statistical_presentation_policy.md`: rules preventing cross-task/raw-standardisation misinterpretation.\n- `provenance/provenance_manifest.json`: source hashes and scientific lineage.\n- `consistency_checks.json`: CPU validation outcomes.\n\nThe complete reBEN 190-cell candidate universe is retained next to the four deterministically selected examples, so the presentation examples are reproducible rather than post-hoc cherry-picked.\n"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    package_files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "package_manifest.json")
    package_manifest = {
        "package": "final_paper_package_v1",
        "file_count": len(package_files),
        "files": [
            {"path": str(p.relative_to(out)).replace("\\", "/"), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in package_files
        ],
    }
    (out / "package_manifest.json").write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
