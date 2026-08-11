from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


DEFAULT_OUTPUT = Path("outputs/unified_paper_package_v3")
PROTOCOL_ROOT = Path("outputs/fmow_protocol_contrast_v1")
SELECTIVE_ROOT = Path("outputs/fmow_conformal_selective_audit_v1")
SOCIAL_ROOT = Path("outputs/fmow_social_spatial_interpretation_v1")
RANDOM_SOCIAL_ROOT = Path("outputs/fmow_random_split_social_spatial_v1")
EVIDENCE_POLICY = PROJECT_ROOT / "configs" / "analysis" / "evidence_status_v060.json"


def _evidence_policy(path: Path = EVIDENCE_POLICY) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Invalid evidence policy: {path}")
    return [dict(record) for record in records]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def _main_result_rows() -> list[dict[str, Any]]:
    return [
        {"dataset": "fMoW", "run_id": "resnet50_13band", "model": "ResNet50", "axis": "geography", "protocol": "location_disjoint", "protocol_status": "valid_benchmark_formal_partial", "estimate_validity": "valid", "comparison_design": "same_registered_panel", "inference_strength": "formal_partial", "presentation_role": "main_benchmark", "aggregate_metric": "accuracy", "aggregate_value": 0.20002233638597275, "raw_bwer": 0.17361207464336303, "standardised_bwer": 0.14234836453364907},
        {"dataset": "fMoW", "run_id": "dofa_scaled10000", "model": "DOFA scaled", "axis": "geography", "protocol": "location_disjoint", "protocol_status": "valid_benchmark_formal_partial", "estimate_validity": "valid", "comparison_design": "same_registered_panel", "inference_strength": "formal_partial", "presentation_role": "main_benchmark", "aggregate_metric": "accuracy", "aggregate_value": 0.17768595041322313, "raw_bwer": 0.16141538857738702, "standardised_bwer": 0.1269780950367737},
        {"dataset": "reBEN", "run_id": "croma_s1", "model": "CROMA S1", "axis": "sensor/geography", "protocol": "multilabel_remote_sensing", "protocol_status": "valid_benchmark_descriptive", "estimate_validity": "valid", "comparison_design": "same_panel_sensor_condition", "inference_strength": "descriptive", "presentation_role": "main_benchmark", "aggregate_metric": "macro_ap", "aggregate_value": 0.4957580772190985, "raw_bwer": 0.1588281739799881, "standardised_bwer": 0.1588281739799881},
        {"dataset": "reBEN", "run_id": "croma_s2", "model": "CROMA S2", "axis": "sensor/geography", "protocol": "multilabel_remote_sensing", "protocol_status": "valid_benchmark_descriptive", "estimate_validity": "valid", "comparison_design": "same_panel_sensor_condition", "inference_strength": "descriptive", "presentation_role": "main_benchmark", "aggregate_metric": "macro_ap", "aggregate_value": 0.5817858667102239, "raw_bwer": 0.10553858910927061, "standardised_bwer": 0.10553858910927061},
        {"dataset": "reBEN", "run_id": "croma_s1_s2", "model": "CROMA S1+S2", "axis": "sensor/geography", "protocol": "multilabel_remote_sensing", "protocol_status": "valid_benchmark_descriptive", "estimate_validity": "valid", "comparison_design": "same_panel_sensor_condition", "inference_strength": "descriptive", "presentation_role": "main_benchmark", "aggregate_metric": "macro_ap", "aggregate_value": 0.6080083479682447, "raw_bwer": 0.07242186493060693, "standardised_bwer": 0.07242186493060693},
        {"dataset": "Sen1Floods11", "run_id": "prithvi_tl", "model": "Prithvi TL", "axis": "event", "protocol": "event_tail_segmentation", "protocol_status": "valid_protocol_aware_benchmark", "estimate_validity": "valid", "comparison_design": "protocol_aware", "inference_strength": "descriptive", "presentation_role": "main_benchmark", "aggregate_metric": "iou", "aggregate_value": 0.8051987522468903, "raw_bwer": 0.1175003367140734, "standardised_bwer": 0.15655975905862524},
        {"dataset": "Sen1Floods11", "run_id": "unet_vanilla", "model": "Vanilla U-Net", "axis": "event", "protocol": "event_tail_segmentation", "protocol_status": "valid_protocol_aware_benchmark", "estimate_validity": "valid", "comparison_design": "protocol_aware", "inference_strength": "descriptive", "presentation_role": "main_benchmark", "aggregate_metric": "iou", "aggregate_value": 0.751110769217214, "raw_bwer": 0.3188765524378785, "standardised_bwer": 0.2295156256914025},
        {"dataset": "Sen1Floods11", "run_id": "mndwi", "model": "MNDWI", "axis": "event", "protocol": "event_tail_segmentation", "protocol_status": "valid_protocol_aware_benchmark", "estimate_validity": "valid", "comparison_design": "protocol_aware", "inference_strength": "descriptive", "presentation_role": "main_benchmark", "aggregate_metric": "iou", "aggregate_value": 0.595166113508361, "raw_bwer": 0.15529557268478827, "standardised_bwer": 0.10040281744951218},
        {"dataset": "Sen1Floods11", "run_id": "s2_resnet34_unet", "model": "S2 ResNet34 U-Net", "axis": "event", "protocol": "event_tail_segmentation", "protocol_status": "valid_protocol_aware_benchmark", "estimate_validity": "valid", "comparison_design": "protocol_aware", "inference_strength": "descriptive", "presentation_role": "main_benchmark", "aggregate_metric": "iou", "aggregate_value": 0.7896841162497235, "raw_bwer": 0.17733362156729465, "standardised_bwer": 0.20953834627006696},
    ]


def _claim_support_table() -> list[dict[str, Any]]:
    return [
        {"claim_id": "C1", "claim": "Aggregate performance is insufficient for deployment reliability.", "claim_strength": "strong", "evidence_files": "unified main results; fMoW selective audit; reBEN; Sen1", "formal_status": "cross_experiment_supported", "primary_caveat": "BWER is slice-audit evidence, not a universal fairness guarantee."},
        {"claim_id": "C2", "claim": "Random-split fMoW accuracy is overly optimistic relative to location-disjoint evaluation.", "claim_strength": "strong_protocol_contrast", "evidence_files": "fmow_protocol_contrast_v1", "formal_status": "protocol_contrast_not_deployment", "primary_caveat": "Random split remains sanity evidence only."},
        {"claim_id": "C3", "claim": "Within the registered fMoW location-disjoint benchmark panel, ResNet50 has the higher aggregate score while DOFA has the lower reported geography-BWER point estimate.", "claim_strength": "moderate_protocol_aware", "evidence_files": "drive_real_audit_v1; fmow_conformal_selective_audit_v1", "formal_status": "valid_point_estimate_formal_partial", "primary_caveat": "Retain the ranking as benchmark evidence; same-seed common-support inference does not certify universal no-harm or universal model superiority."},
        {"claim_id": "C4", "claim": "The fMoW aggregate-vs-BWER divergence persists under confidence-conditioned audit.", "claim_strength": "strong_posthoc", "evidence_files": "rank_divergence_under_selective_audit.csv", "formal_status": "posthoc_selective_diagnostic", "primary_caveat": "Calibrated threshold is weaker than full conformal prediction because full probability vectors are unavailable."},
        {"claim_id": "C5", "claim": "fMoW geography risk is spatially structured but not reducible to GDP, population density, or urbanization.", "claim_strength": "moderate_exploratory", "evidence_files": "fmow_social_spatial_interpretation_v1", "formal_status": "exploratory_association", "primary_caveat": "Country-level associations are non-causal and support-filtered."},
        {"claim_id": "C6", "claim": "The retained reBEN benchmark point estimates favor CROMA S1+S2 on aggregate and country-GeoBWER summaries while residual label and country tail risks remain.", "claim_strength": "moderate_descriptive", "evidence_files": "reBEN 27-run and labelwise outputs", "formal_status": "valid_point_estimates_partial_identification", "primary_caveat": "Country-cluster inference remains support-limited; this does not invalidate the reported benchmark ranking."},
        {"claim_id": "C7", "claim": "The retained 19-model Sen1Floods11 benchmark shows material chip- and event-level tail-risk variation across models and modalities.", "claim_strength": "moderate_descriptive", "evidence_files": "Sen1 19-model event-tail outputs", "formal_status": "valid_protocol_aware_descriptive", "primary_caveat": "Report test90 and combined105 separately; the single Bolivia event has no identified between-event gap."},
    ]


def _experiment_status_matrix() -> list[dict[str, Any]]:
    return [
        {"experiment": "fMoW location-disjoint Step3", "dataset": "fMoW", "status": "complete", "evidence_role": "formal geography deployment audit", "allowed_claims": "Raw/standardised BWER; selective BWER; calibrated confidence-threshold diagnostics", "blocked_claims": "Full conformal prediction without full probability vectors or true-class probabilities"},
        {"experiment": "fMoW random split sanity", "dataset": "fMoW", "status": "complete", "evidence_role": "protocol contrast", "allowed_claims": "Random split aggregate optimism relative to location-disjoint", "blocked_claims": "Deployment generalization"},
        {"experiment": "fMoW social-spatial v1.1", "dataset": "fMoW", "status": "complete", "evidence_role": "exploratory interpretation", "allowed_claims": "Support-filtered non-causal country indicator associations", "blocked_claims": "Causal socio-economic explanation"},
        {"experiment": "reBEN CROMA sensor mode", "dataset": "reBEN", "status": "complete", "evidence_role": "sensor/probability-aware deployment slice audit", "allowed_claims": "Aggregate and BWER comparison across S1/S2/fusion", "blocked_claims": "Global fairness guarantee"},
        {"experiment": "Sen1Floods11 event-tail segmentation", "dataset": "Sen1Floods11", "status": "complete", "evidence_role": "event-tail segmentation audit", "allowed_claims": "Event tail-risk evidence", "blocked_claims": "Conformal/selective segmentation without probability maps"},
        {"experiment": "AlphaEarth/GEE land-cover", "dataset": "AlphaEarth", "status": "point_estimates_complete_spatial_gate_failed", "evidence_role": "reference-map agreement benchmark plus honest calibration failure", "allowed_claims": "Raw and class-standardised point/support results; global-versus-cluster empirical comparator", "blocked_claims": "Formal spatial finite-sample guarantee; interpretation as ground-truth fairness"},
    ]


def _metric_scope_matrix() -> list[dict[str, Any]]:
    return [
        {"metric": "Raw-BWER", "scope": "Slice worst-case excess risk over normalized audit tables", "formal_for": "Valid point estimation across registered tasks; inference strength is recorded separately", "caveat": "Low support limits generalization but does not automatically delete a valid point estimate."},
        {"metric": "Standardised-BWER", "scope": "Worst-case excess risk after balancing over available balance variable", "formal_for": "Valid point estimation where the standardization cells are identified", "caveat": "Missing balance policy and support thresholds remain part of the inference claim."},
        {"metric": "Selective-BWER", "scope": "BWER over confidence-retained examples", "formal_for": "fMoW post-hoc location-disjoint diagnostic", "caveat": "Formal for retained audit rows, not a new model evaluation."},
        {"metric": "Calibrated confidence-threshold BWER", "scope": "Calibration/test split threshold diagnostic using confidence", "formal_for": "fMoW diagnostic only", "caveat": "Weaker than APS/RAPS conformal prediction."},
        {"metric": "Protocol contrast", "scope": "Random split vs location-disjoint aggregate and BWER comparison", "formal_for": "fMoW protocol discussion", "caveat": "Random split is not deployment evidence."},
        {"metric": "Social-spatial association", "scope": "Support-filtered country risk vs World Bank indicators", "formal_for": "Exploratory interpretation", "caveat": "Associational and country-level only; no causal claim."},
    ]


def _protocol_summary(protocol_root: Path) -> list[dict[str, Any]]:
    rows = _read(protocol_root / "fmow_random_vs_location_disjoint_summary.csv")
    return [
        {
            "model_family": row.get("model_family"),
            "random_accuracy": row.get("random_accuracy"),
            "location_disjoint_accuracy": row.get("location_disjoint_accuracy"),
            "accuracy_drop_random_to_location": row.get("accuracy_drop_random_to_location"),
            "random_country_raw_bwer": row.get("random_country_raw_bwer"),
            "location_country_raw_bwer": row.get("location_country_raw_bwer"),
            "evidence_scope": "random split is sanity/protocol contrast; location-disjoint is formal deployment protocol",
        }
        for row in rows
    ]


def _selective_social_summary(selective_root: Path, social_root: Path) -> list[dict[str, Any]]:
    divergence = _read(selective_root / "rank_divergence_under_selective_audit.csv")
    assoc = _read(social_root / "fmow_risk_indicator_association_v1_1.csv")
    n_div = sum(str(row.get("rank_diverges")).lower() == "true" for row in divergence)
    n_total = len(divergence)
    baseline_assoc = [row for row in assoc if row.get("scenario") == "baseline" and row.get("association_status") == "available"]
    max_abs = max((_float(row.get("pearson_r"), 0.0) for row in baseline_assoc), key=abs, default=0.0)
    return [
        {"summary_item": "rank_divergence_under_selective_audit", "value": f"{n_div}/{n_total}", "interpretation": "The retained benchmark point estimates show aggregate-vs-BWER rank divergence across baseline, top-k, and calibrated-threshold settings; this is not a universal-superiority claim."},
        {"summary_item": "social_indicator_join", "value": "available", "interpretation": "World Bank GDP per capita, population density, urbanization, and income metadata were joined by ISO3 where available."},
        {"summary_item": "largest_baseline_indicator_correlation_abs", "value": max_abs, "interpretation": "Country-level socio-economic indicators show weak exploratory associations after support filtering."},
    ]


def _bwer_robustness_summary(selective_root: Path) -> list[dict[str, Any]]:
    divergence = _read(selective_root / "rank_divergence_under_selective_audit.csv")
    return [
        {"evidence_line": "fMoW support-filtered selective audit", "status": "complete", "summary": "Support-filtered country summaries reduce small-country overinterpretation.", "source_file": "fmow_support_filtered_slice_summary.csv"},
        {"evidence_line": "fMoW rank divergence", "status": "complete", "summary": f"{sum(str(row.get('rank_diverges')).lower() == 'true' for row in divergence)}/{len(divergence)} settings show aggregate-best != BWER-best.", "source_file": "rank_divergence_under_selective_audit.csv"},
        {"evidence_line": "Missing policy and support sensitivity", "status": "available_from_prior_audit_where_generated", "summary": "Use existing robustness outputs as sensitivity evidence; do not infer missing sensitivity tables.", "source_file": "outputs/bwer_robustness_v1 and drive_real_audit_v1"},
    ]


def _random_social_summary(random_social_root: Path) -> list[dict[str, Any]]:
    assoc_path = random_social_root / "fmow_random_split_risk_indicator_association.csv"
    rows = _read(assoc_path)
    available = [row for row in rows if row.get("association_status") == "available"]
    max_abs = max((_float(row.get("pearson_r"), 0.0) for row in available), key=abs, default=0.0)
    strongest = next((row for row in available if abs(_float(row.get("pearson_r"), 0.0)) == abs(max_abs)), {})
    return [
        {
            "experiment_id": "fmow_random_split_social_spatial_v1",
            "protocol": "random_split_sanity",
            "association_rows": len(rows),
            "largest_abs_pearson_r": max_abs,
            "strongest_run_id": strongest.get("run_id", ""),
            "strongest_indicator": strongest.get("indicator", ""),
            "claim_scope": "sanity/protocol contrast only; exploratory association; not causal and not deployment evidence",
            "source_file": str(assoc_path) if assoc_path.exists() else "unavailable",
        }
    ]


def _write_markdown(output: Path, random_social_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Path]:
    files = {
        "paper_ready_main_findings_v3": output / "paper_ready_main_findings_v3.md",
        "next_experiment_readiness_alphaearth_gee": output / "next_experiment_readiness_alphaearth_gee.md",
        "manuscript_outline_v3": output / "manuscript_outline_v3.md",
        "thesis_chapter_outline_v3": output / "thesis_chapter_outline_v3.md",
    }
    files["paper_ready_main_findings_v3"].write_text(
        "# Paper-ready main findings v3\n\n"
        "1. Aggregate performance is insufficient for deployment reliability across geography, sensor mode, and event-tail audits.\n"
        "2. fMoW random split is a protocol contrast: it produces much higher aggregate accuracy than location-disjoint evaluation, but it is not deployment evidence.\n"
        "3. In the retained fMoW location-disjoint benchmark, ResNet50 has the higher aggregate point score and DOFA scaled the lower geography-BWER point estimate; common-support inference remains partial and does not establish universal superiority.\n"
        "4. fMoW geography risk is spatially structured, but support-filtered World Bank GDP, population density, and urbanization associations are weak and non-causal.\n"
        "5. Random-split fMoW social-spatial associations are reported only as sanity/protocol contrast and do not replace location-disjoint deployment interpretation.\n"
        "6. Retained reBEN benchmark estimates favor CROMA fusion on aggregate and broad GeoBWER summaries while country inference remains support-limited; labelwise sensitivity is reported separately.\n"
        "7. The retained Sen1Floods11 19-model benchmark exposes chip- and event-tail variation; test90 and combined105 remain separate and single-event regimes are labelled as structurally unidentified.\n"
        "8. AlphaEarth raw and class-standardised agreement estimates are retained, while the failed validation-only spatial gate blocks only the formal spatial guarantee.\n",
        encoding="utf-8",
    )
    files["next_experiment_readiness_alphaearth_gee"].write_text(
        "# AlphaEarth/GEE land-cover readiness\n\n"
        "The empirical AlphaEarth point-estimate campaign has completed. Retain raw and class-standardised reference-map agreement results; the validation-only spatial scale gate failed, so no formal spatial certification is available.\n\n"
        "Task design: fixed land-cover classification or pixel-level segmentation with explicit geography, biome, urban/rural, and income-region slices. Use a location-disjoint split when possible.\n\n"
        "GEE export needs: tile IDs, ISO3/country, region, biome/ecoregion if available, acquisition dates, labels, and model-ready image chips or probability tables. Export manifests before chips.\n\n"
        "Local BWER mapping: normalize predictions into audit tables with label, prediction, confidence/probabilities where available, slice fields, support counts, preflight, Raw-BWER, Standardised-BWER, and robustness summaries.\n\n"
        "Expected value: adds a land-cover deployment axis and can test whether AlphaEarth-style representations reduce geography tail risk.\n\n"
        "Risks: GEE quota, label noise, temporal mismatch, country support imbalance, and accidental overclaiming if only random/geographically mixed splits are available.\n",
        encoding="utf-8",
    )
    files["manuscript_outline_v3"].write_text(
        "# Manuscript outline v3\n\n"
        "1. Introduction: deployment reliability requires slice-aware evidence beyond aggregate scores.\n"
        "2. Methods: normalized audit tables, support-aware preflight, Raw-BWER, Standardised-BWER, selective diagnostics.\n"
        "3. Experiments: fMoW geography, reBEN sensor/geography, Sen1Floods11 event tails.\n"
        "4. Results: aggregate-vs-BWER divergence, random-vs-location protocol contrast, confidence-conditioned fMoW audit.\n"
        "5. Interpretation: fMoW social-spatial structure and limits of country-level indicators.\n"
        "6. Limitations: post-hoc calibration, missing probability maps, support thresholds, non-causal associations.\n"
        "7. Next experiments: AlphaEarth/GEE land-cover audit.\n",
        encoding="utf-8",
    )
    files["thesis_chapter_outline_v3"].write_text(
        "# Thesis chapter outline v3\n\n"
        "1. Remote-sensing foundation models and deployment reliability.\n"
        "2. BWER as a normalized slice-fairness audit contract.\n"
        "3. Geography audit on fMoW: protocol contrast, selective-BWER, and social-spatial interpretation.\n"
        "4. Sensor-mode audit on reBEN.\n"
        "5. Event-tail audit on Sen1Floods11.\n"
        "6. Robustness, caveats, and reproducibility package.\n"
        "7. AlphaEarth/GEE design for the next deployment axis.\n",
        encoding="utf-8",
    )
    return files


def _write_figures(output: Path, main_rows: list[dict[str, Any]], protocol_rows: list[dict[str, Any]], social_rows: list[dict[str, str]]) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    figures = ensure_dir(output / "figures")
    paths: dict[str, Path] = {}

    def save(fig: Any, name: str) -> None:
        png = figures / f"{name}.png"
        pdf = figures / f"{name}.pdf"
        fig.tight_layout()
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
        paths[f"{name}_png"] = png
        paths[f"{name}_pdf"] = pdf

    colors = {"fMoW": "#2F5DA8", "reBEN": "#3A7D44", "Sen1Floods11": "#A04D3A"}

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    axes = ["Geography\nfMoW", "Sensor/geography\nreBEN", "Event tails\nSen1", "Land cover\nAlphaEarth"]
    strength = [3, 2.5, 2, 0.6]
    ax.bar(axes, strength, color=["#2F5DA8", "#3A7D44", "#A04D3A", "#777777"])
    ax.set_ylim(0, 3.4)
    ax.set_ylabel("Evidence maturity")
    ax.set_title("Deployment audit axes")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "unified_deployment_axes_overview_v3")

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for row in main_rows:
        ax.scatter(row["aggregate_value"], row["raw_bwer"], s=58, color=colors.get(row["dataset"], "#666666"))
        ax.text(row["aggregate_value"], row["raw_bwer"], row["model"], fontsize=7)
    ax.set_xlabel("Aggregate metric value")
    ax.set_ylabel("Raw-BWER")
    ax.set_title("Aggregate score does not determine tail risk")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "aggregate_vs_bwer_main_result_v3")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = [row["model_family"] for row in protocol_rows]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, [_float(row["random_accuracy"]) for row in protocol_rows], width, label="random split", color="#8C6D31")
    ax.bar(x + width / 2, [_float(row["location_disjoint_accuracy"]) for row in protocol_rows], width, label="location-disjoint", color="#2F5DA8")
    ax.set_xticks(x)
    ax.set_xticklabels(["ResNet50", "DOFA"])
    ax.set_ylabel("Accuracy")
    ax.set_title("fMoW protocol contrast")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fmow_protocol_contrast_v3")

    fmow = [row for row in main_rows if row["dataset"] == "fMoW"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar([row["model"] for row in fmow], [row["raw_bwer"] for row in fmow], color=["#2F5DA8", "#6B8E23"])
    ax.set_ylabel("Country Raw-BWER")
    ax.set_title("fMoW aggregate-best vs BWER-best")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fmow_aggregate_vs_selective_bwer_v3")

    assoc = [row for row in social_rows if row.get("scenario") == "baseline" and row.get("association_status") == "available"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    labels = [f"{row.get('run_id','').replace('_scaled10000','')}\n{row.get('indicator','')}" for row in assoc]
    vals = [_float(row.get("pearson_r"), 0.0) for row in assoc]
    ax.bar(range(len(vals)), vals, color="#5E6C84")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title("Support-filtered fMoW risk vs country indicators")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fmow_social_spatial_summary_v3")

    reben = [row for row in main_rows if row["dataset"] == "reBEN"]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(reben))
    ax1.bar(x - 0.18, [row["aggregate_value"] for row in reben], width=0.36, color="#3A7D44", label="macro AP")
    ax1.bar(x + 0.18, [row["raw_bwer"] for row in reben], width=0.36, color="#B08D57", label="country BWER")
    ax1.set_xticks(x)
    ax1.set_xticklabels([row["model"].replace("CROMA ", "") for row in reben])
    ax1.set_title("reBEN sensor-mode summary")
    ax1.legend(frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    save(fig, "reben_sensor_mode_summary_v3")

    sen1 = [row for row in main_rows if row["dataset"] == "Sen1Floods11"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(sen1))
    ax.bar(x - 0.18, [row["aggregate_value"] for row in sen1], width=0.36, color="#A04D3A", label="IoU")
    ax.bar(x + 0.18, [row["raw_bwer"] for row in sen1], width=0.36, color="#5A6F8F", label="Raw-BWER")
    ax.set_xticks(x)
    ax.set_xticklabels([row["model"] for row in sen1], rotation=25, ha="right")
    ax.set_title("Sen1Floods11 event-tail summary")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "sen1_event_tail_risk_summary_v3")

    claims = _claim_support_table()
    strength_map = {"strong": 3.0, "strong_protocol_contrast": 2.6, "strong_posthoc": 2.4, "moderate_protocol_aware": 2.2, "moderate_descriptive": 2.0, "moderate": 1.8, "moderate_exploratory": 1.5}
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    vals = [strength_map.get(row["claim_strength"], 1.0) for row in claims]
    ax.imshow([vals], aspect="auto", cmap="YlGnBu", vmin=0, vmax=3)
    ax.set_yticks([])
    ax.set_xticks(range(len(claims)))
    ax.set_xticklabels([row["claim_id"] for row in claims])
    ax.set_title("Claim strength matrix")
    for i, val in enumerate(vals):
        ax.text(i, 0, f"{val:.1f}", ha="center", va="center", fontsize=8)
    save(fig, "claim_strength_matrix_v3")

    return paths


def build_unified_paper_package(
    output_dir: Path = DEFAULT_OUTPUT,
    protocol_root: Path = PROTOCOL_ROOT,
    selective_root: Path = SELECTIVE_ROOT,
    social_root: Path = SOCIAL_ROOT,
    random_social_root: Path = RANDOM_SOCIAL_ROOT,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    main_rows = _main_result_rows()
    protocol_rows = _protocol_summary(protocol_root)
    social_rows = _read(social_root / "fmow_risk_indicator_association_v1_1.csv")
    random_social_rows = _random_social_summary(random_social_root)

    artifacts = {
        "claim_support_table_v3": output / "claim_support_table_v3.csv",
        "experiment_status_matrix_v3": output / "experiment_status_matrix_v3.csv",
        "metric_scope_and_caveat_matrix_v3": output / "metric_scope_and_caveat_matrix_v3.csv",
        "unified_results_narrative_table_v3": output / "unified_results_narrative_table_v3.csv",
        "bwer_robustness_summary_v3": output / "bwer_robustness_summary_v3.csv",
        "fmow_selective_social_spatial_summary_v3": output / "fmow_selective_social_spatial_summary_v3.csv",
        "fmow_protocol_contrast_summary_v3": output / "fmow_protocol_contrast_summary_v3.csv",
        "fmow_random_split_social_spatial_summary_v3": output / "fmow_random_split_social_spatial_summary_v3.csv",
        "current_evidence_policy_v3": output / "current_evidence_policy_v3.csv",
    }
    write_csv(artifacts["claim_support_table_v3"], _claim_support_table())
    write_csv(artifacts["experiment_status_matrix_v3"], _experiment_status_matrix())
    write_csv(artifacts["metric_scope_and_caveat_matrix_v3"], _metric_scope_matrix())
    write_csv(artifacts["unified_results_narrative_table_v3"], main_rows)
    write_csv(artifacts["bwer_robustness_summary_v3"], _bwer_robustness_summary(selective_root))
    write_csv(artifacts["fmow_selective_social_spatial_summary_v3"], _selective_social_summary(selective_root, social_root))
    write_csv(artifacts["fmow_protocol_contrast_summary_v3"], protocol_rows)
    write_csv(artifacts["fmow_random_split_social_spatial_summary_v3"], random_social_rows)
    write_csv(artifacts["current_evidence_policy_v3"], _evidence_policy())
    artifacts.update(_write_markdown(output, random_social_rows))
    artifacts.update({f"figure_{name}": path for name, path in _write_figures(output, main_rows, protocol_rows, social_rows).items()})
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for name, path in build_unified_paper_package(args.out).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
