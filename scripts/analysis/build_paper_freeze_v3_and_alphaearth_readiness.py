from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, write_csv


FREEZE_ROOT = Path("outputs/final_paper_assets/rsfm_bwer_paper_freeze_v3")
ALPHA_ROOT = Path("outputs/alphaearth_gee_readiness_v1")

SOURCE_DIRS = [
    Path("outputs/unified_paper_package_v3"),
    Path("outputs/fmow_protocol_contrast_v1"),
    Path("outputs/fmow_conformal_selective_audit_v1"),
    Path("outputs/fmow_social_spatial_interpretation_v1"),
    Path("outputs/fmow_random_split_social_spatial_v1"),
    Path("outputs/drive_real_audit_v1"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


def _copy_sources(freeze_root: Path) -> list[dict[str, Any]]:
    copied_rows: list[dict[str, Any]] = []
    copied_root = ensure_dir(freeze_root / "source_outputs")
    for source in SOURCE_DIRS:
        dest = copied_root / source.name
        status = "copied" if source.exists() else "missing"
        if source.exists():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
        files = _iter_files(dest) if dest.exists() else []
        copied_rows.append(
            {
                "source_output": str(source),
                "freeze_copy": str(dest),
                "status": status,
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
            }
        )
    return copied_rows


def _asset_index(freeze_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in _iter_files(freeze_root / "source_outputs"):
        rel = path.relative_to(freeze_root).as_posix()
        rows.append(
            {
                "relative_path": rel,
                "file_name": path.name,
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "asset_group": rel.split("/")[1] if "/" in rel else "root",
            }
        )
    return sorted(rows, key=lambda row: row["relative_path"])


def _formal_table() -> list[dict[str, str]]:
    return [
        {"asset_or_analysis": "fMoW location-disjoint Step3", "status": "valid_point_estimates_formal_partial", "allowed_claim": "Retain the benchmark ranking: ResNet50 has the higher aggregate point score and DOFA scaled the lower geography-BWER point estimate in the registered panel", "caveat": "Common-support inference does not certify universal superiority or no-harm disparity reduction."},
        {"asset_or_analysis": "fMoW selective-BWER", "status": "post_hoc_formal_diagnostic", "allowed_claim": "Aggregate-vs-BWER rank divergence persists under confidence-conditioned retained sets", "caveat": "No new model inference; calibrated threshold is weaker than full conformal prediction."},
        {"asset_or_analysis": "fMoW calibrated confidence-threshold", "status": "diagnostic", "allowed_claim": "Calibration/test split threshold diagnostic from available confidence/max_probability", "caveat": "No APS/RAPS conformal claim without full probability vectors or true-class probabilities."},
        {"asset_or_analysis": "fMoW social-spatial v1.1", "status": "exploratory_diagnostic", "allowed_claim": "Country risk is spatially structured and weakly associated with simple World Bank indicators", "caveat": "Non-causal; support_count >= 20 for country interpretation."},
        {"asset_or_analysis": "fMoW random split protocol contrast", "status": "sanity", "allowed_claim": "Random split aggregate accuracy is overly optimistic relative to location-disjoint protocol", "caveat": "Not deployment evidence."},
        {"asset_or_analysis": "fMoW random split social-spatial", "status": "sanity", "allowed_claim": "Exploratory random-split country indicator associations", "caveat": "Not deployment evidence and not a substitute for location-disjoint social-spatial interpretation."},
        {"asset_or_analysis": "Drive-real audit v1", "status": "formal_contract_evidence", "allowed_claim": "Real Drive outputs support the audit contract findings", "caveat": "Raw zips are referenced/read-only, not repackaged as raw experiment data."},
        {"asset_or_analysis": "reBEN 27-run and labelwise sensitivity", "status": "valid_point_estimates_partial_identification", "allowed_claim": "Retain model, modality, country, and labelwise benchmark patterns", "caveat": "Country-cluster inference is support-limited; labelwise sensitivity is not geographic inference."},
        {"asset_or_analysis": "Sen1Floods11 19-model event panel", "status": "valid_protocol_aware_descriptive", "allowed_claim": "Retain test90, combined105, model/modality rankings, and event-tail findings", "caveat": "Single-event regimes have no identified between-event gap; no formal spatial guarantee."},
        {"asset_or_analysis": "AlphaEarth/GEE", "status": "point_estimates_complete_spatial_gate_failed", "allowed_claim": "Retain raw and class-standardised reference-map agreement point/support results", "caveat": "No formal spatial certification; not ground-truth fairness."},
    ]


def _write_freeze_markdown(freeze_root: Path, source_rows: list[dict[str, Any]], asset_rows: list[dict[str, Any]]) -> dict[str, Path]:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    manifest = freeze_root / "MANIFEST.md"
    manifest.write_text(
        "# RSFM BWER paper freeze v3\n\n"
        f"Generated: {generated}\n\n"
        "This freeze copies the current paper-facing evidence assets into a stable package. It does not train models, run inference, modify raw experiment outputs, or delete intermediate files.\n\n"
        "## Included source outputs\n\n"
        + "\n".join(f"- `{row['source_output']}` -> `{row['freeze_copy']}` ({row['status']}, {row['file_count']} files)" for row in source_rows)
        + "\n\n## Index\n\n"
        f"- `asset_index.csv`: {len(asset_rows)} files with size and SHA256.\n"
        "- `formal_vs_diagnostic_vs_sanity_table.csv`: claim-scope table.\n"
        "- `source_output_references.csv`: source-to-freeze mapping.\n"
        "- `reproducibility_notes.md`: commands and caveats.\n"
        "- `rsfm_bwer_paper_freeze_v3.zip`: archive of this freeze directory.\n",
        encoding="utf-8",
    )
    repro = freeze_root / "reproducibility_notes.md"
    repro.write_text(
        "# Reproducibility notes\n\n"
        "Run from repository root:\n\n"
        "```powershell\n"
        "python scripts\\analysis\\build_paper_freeze_v3_and_alphaearth_readiness.py\n"
        "python -m pytest tests\\test_paper_freeze_and_alphaearth_readiness.py\n"
        "```\n\n"
        "The freeze is a copy/reference package over completed outputs. It intentionally excludes raw training data, raw image arrays, and any model reruns. Valid point estimates and benchmark rankings are retained even when inference is partial or descriptive. Random-split assets are marked sanity/protocol contrast only; formal certification status is recorded separately from estimate validity.\n",
        encoding="utf-8",
    )
    return {"MANIFEST": manifest, "reproducibility_notes": repro}


def _zip_freeze(freeze_root: Path) -> Path:
    zip_path = freeze_root / "rsfm_bwer_paper_freeze_v3.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _iter_files(freeze_root):
            if path == zip_path:
                continue
            archive.write(path, path.relative_to(freeze_root))
    return zip_path


def build_paper_freeze(freeze_root: Path = FREEZE_ROOT) -> dict[str, Path]:
    freeze = ensure_dir(freeze_root)
    source_rows = _copy_sources(freeze)
    asset_rows = _asset_index(freeze)
    artifacts = {
        "asset_index": freeze / "asset_index.csv",
        "formal_vs_diagnostic_vs_sanity_table": freeze / "formal_vs_diagnostic_vs_sanity_table.csv",
        "source_output_references": freeze / "source_output_references.csv",
    }
    write_csv(artifacts["asset_index"], asset_rows)
    write_csv(artifacts["formal_vs_diagnostic_vs_sanity_table"], _formal_table())
    write_csv(artifacts["source_output_references"], source_rows)
    artifacts.update(_write_freeze_markdown(freeze, source_rows, asset_rows))
    artifacts["zip"] = _zip_freeze(freeze)
    return artifacts


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def build_alphaearth_readiness(output_dir: Path = ALPHA_ROOT) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    artifacts = {
        "alphaearth_research_design": output / "alphaearth_research_design.md",
        "gee_export_manifest_template": output / "gee_export_manifest_template.csv",
        "alphaearth_audit_table_schema": output / "alphaearth_audit_table_schema.csv",
        "alphaearth_bwer_pipeline_mapping": output / "alphaearth_bwer_pipeline_mapping.md",
        "alphaearth_feasibility_checklist": output / "alphaearth_feasibility_checklist.md",
        "alphaearth_risk_register": output / "alphaearth_risk_register.csv",
        "alphaearth_colab_or_gee_steps": output / "alphaearth_colab_or_gee_steps.md",
        "alphaearth_expected_claims_and_blockers": output / "alphaearth_expected_claims_and_blockers.md",
    }
    _write_text(
        artifacts["alphaearth_research_design"],
        """
# AlphaEarth/GEE pilot research design

This is a runnable but not-yet-executed pilot plan. No empirical AlphaEarth claims are made here.

## Candidate data

- AlphaEarth embeddings if available in Google Earth Engine or as a clean exportable feature source.
- Dynamic World labels/probabilities for land-cover agreement or supervised classification.
- ESA WorldCover labels for independent land-cover agreement checks.
- LUCAS point labels if feasible after license, spatial join, and support checks.

## Task

Pilot a land-cover / land-use classification or agreement audit. Prefer exporting metadata, labels, model scores, confidence/probability summaries, and slice fields before large imagery.

## Deployment axes

Country, biome/ecoregion, urban-rural, income group, and support strata. Location-disjoint or spatial-block split is preferred; random split is sanity only.

## Metrics

Aggregate accuracy/F1, Raw-BWER, Standardised-BWER, and Selective-BWER if confidence/probability fields exist.
""",
    )
    write_csv(
        artifacts["gee_export_manifest_template"],
        [
            {
                "sample_id": "",
                "tile_id": "",
                "split": "train|calibration|test",
                "spatial_block_id": "",
                "latitude": "",
                "longitude": "",
                "country_iso3": "",
                "admin_region": "",
                "biome_or_ecoregion": "",
                "urban_rural": "",
                "income_group": "",
                "support_stratum": "",
                "label_source": "DynamicWorld|ESAWorldCover|LUCAS|other",
                "label": "",
                "prediction": "",
                "confidence": "",
                "probability_vector_ref": "",
                "gee_asset_id": "",
                "export_status": "planned",
                "notes": "",
            }
        ],
    )
    write_csv(
        artifacts["alphaearth_audit_table_schema"],
        [
            {"field": "sample_id", "required": "yes", "description": "Stable sample/tile identifier."},
            {"field": "dataset", "required": "yes", "description": "Dataset or label source."},
            {"field": "task", "required": "yes", "description": "land_cover_classification or agreement_audit."},
            {"field": "split", "required": "yes", "description": "train/calibration/test; formal claims use spatial-block or location-disjoint test."},
            {"field": "label", "required": "yes", "description": "Reference class label."},
            {"field": "prediction", "required": "yes", "description": "Predicted/agreement class."},
            {"field": "correct", "required": "yes", "description": "1 if prediction equals label, otherwise 0."},
            {"field": "risk", "required": "yes", "description": "1 - correct for classification."},
            {"field": "confidence", "required": "optional", "description": "Max probability or confidence if available; needed for Selective-BWER."},
            {"field": "country_iso3", "required": "yes", "description": "Country deployment slice."},
            {"field": "biome_or_ecoregion", "required": "recommended", "description": "Biome/ecoregion deployment slice."},
            {"field": "urban_rural", "required": "recommended", "description": "Urban/rural deployment slice."},
            {"field": "income_group", "required": "recommended", "description": "World Bank income group joined by ISO3."},
            {"field": "support_stratum", "required": "recommended", "description": "Predefined support bin."},
            {"field": "spatial_block_id", "required": "recommended", "description": "Block identifier for leakage-aware split checks."},
        ],
    )
    _write_text(
        artifacts["alphaearth_bwer_pipeline_mapping"],
        """
# AlphaEarth BWER pipeline mapping

1. GEE/Colab export: create manifest/audit rows with labels, predictions, confidence if available, spatial metadata, and deployment slices.
2. Local normalization: convert exported tables into the RSFM audit-table schema.
3. Preflight: verify required fields, class mapping, support counts, and leakage constraints.
4. Formal BWER: run Raw-BWER by country, biome/ecoregion, urban-rural, income group, and support strata.
5. Standardised-BWER: balance over label/class where support allows, using the existing missing-balance policy.
6. Selective-BWER: only if confidence/probability exists; otherwise record unavailable.
7. Package: report aggregate metrics, BWER summaries, support diagnostics, caveats, and figures.
""",
    )
    _write_text(
        artifacts["alphaearth_feasibility_checklist"],
        """
# AlphaEarth/GEE feasibility checklist

- [ ] Confirm AlphaEarth embedding availability and export permissions in GEE.
- [ ] Select pilot geography and target label source.
- [ ] Confirm Dynamic World and ESA WorldCover label compatibility.
- [ ] Decide spatial-block or location-disjoint split design.
- [ ] Export metadata/probability/audit table before imagery.
- [ ] Validate ISO3, biome/ecoregion, urban-rural, and income-group joins.
- [ ] Check support counts before interpreting worst slices.
- [ ] Run local audit preflight before BWER.
- [ ] Mark random split as sanity only if used.
""",
    )
    write_csv(
        artifacts["alphaearth_risk_register"],
        [
            {"risk": "AlphaEarth embeddings unavailable or restricted in GEE", "impact": "blocks AlphaEarth-specific empirical claim", "mitigation": "fall back to readiness-only or use exportable embedding source after verification", "owner": "GEE/Colab"},
            {"risk": "Dynamic World and ESA WorldCover label mismatch", "impact": "ambiguous task target", "mitigation": "frame as agreement audit or harmonize classes explicitly", "owner": "analysis"},
            {"risk": "Spatial leakage", "impact": "invalid deployment claim", "mitigation": "use spatial blocks/location-disjoint split and leakage checks", "owner": "analysis"},
            {"risk": "Low country/biome support", "impact": "unstable worst slices", "mitigation": "support preflight and support-filtered summaries", "owner": "local"},
            {"risk": "No confidence/probabilities", "impact": "Selective-BWER unavailable", "mitigation": "record unavailable; do not fabricate confidence", "owner": "GEE/Colab"},
            {"risk": "Large imagery exports exceed quota", "impact": "slow or failed pilot", "mitigation": "export metadata/probability/audit tables first", "owner": "GEE"},
        ],
    )
    _write_text(
        artifacts["alphaearth_colab_or_gee_steps"],
        """
# AlphaEarth Colab/GEE steps

## Requires GEE/Colab

1. Verify AlphaEarth embedding collection or export source.
2. Choose pilot AOI and label source: Dynamic World, ESA WorldCover, optionally LUCAS.
3. Generate spatial-block IDs and candidate splits.
4. Export manifest/audit tables with metadata, labels, predictions/agreement labels, and confidence/probability fields if available.
5. Export small validation samples first; avoid large imagery until schema passes.

## Runs locally

1. Validate exported CSV schema against `alphaearth_audit_table_schema.csv`.
2. Run audit contract/preflight.
3. Run aggregate metrics and BWER summaries.
4. Generate support diagnostics, robustness checks, and paper assets.
""",
    )
    _write_text(
        artifacts["alphaearth_expected_claims_and_blockers"],
        """
# AlphaEarth expected claims and blockers

## Expected claims after execution

- Whether AlphaEarth-style representations reduce land-cover deployment tail risk across geography/ecology slices.
- Whether aggregate land-cover accuracy/F1 hides country, biome/ecoregion, urban-rural, income-group, or support-stratum risks.
- Whether confidence-conditioned Selective-BWER is possible, if confidence/probability fields are exported.

## Current blockers

- No GEE/Colab export has been run.
- AlphaEarth availability and export fields must be verified.
- No empirical AlphaEarth audit table exists yet.
- No AlphaEarth BWER, selective, or social-spatial claim is currently supported.
""",
    )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze paper assets v3 and prepare AlphaEarth/GEE readiness package.")
    parser.add_argument("--freeze-out", type=Path, default=FREEZE_ROOT)
    parser.add_argument("--alpha-out", type=Path, default=ALPHA_ROOT)
    args = parser.parse_args()
    artifacts = {}
    artifacts.update({f"freeze_{k}": v for k, v in build_paper_freeze(args.freeze_out).items()})
    artifacts.update({f"alpha_{k}": v for k, v in build_alphaearth_readiness(args.alpha_out).items()})
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
