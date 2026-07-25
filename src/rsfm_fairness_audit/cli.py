from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_from_file, run_audit_from_outputs
from rsfm_fairness_audit.advanced_closure import run_protocol_matched_comparison, run_selective_risk_audit
from rsfm_fairness_audit.bwer_v2 import run_bwer_v2_posthoc
from rsfm_fairness_audit.fmow_sentinel_enrichment import (
    FmowMetadataEnrichmentConfig,
    run_fmow_sentinel_metadata_enrichment,
)
from rsfm_fairness_audit.fmow_sentinel_classification import (
    FmowBwerConfig,
    FmowClassificationConfig,
    compare_fmow_runs,
    run_fmow_geography_bwer,
    run_fmow_sentinel_classification,
)
from rsfm_fairness_audit.fmow_sentinel_preflight import FmowPreflightConfig, run_fmow_sentinel_preflight
from rsfm_fairness_audit.fmow_step3_contract import (
    FmowStep3PackageConfig,
    FmowStep3ValidationConfig,
    package_fmow_step3_handoff,
    validate_fmow_step3_results,
)
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.geobwer import audit_rows as geobwer_audit_rows
from rsfm_fairness_audit.geobwer import compare as geobwer_compare
from rsfm_fairness_audit.geobwer_extensions import (
    run_multiclass_spatial_upgrade,
    run_multiclass_uncertainty_suite,
    run_multilabel_uncertainty_suite,
)
from rsfm_fairness_audit.geobwer_inventory import inventory_artifacts, write_inventory_report
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.loeo import aggregate_loeo_runs
from rsfm_fairness_audit.pipeline import (
    build_real_adapters,
    compare_model_runs,
    compare_sensor_mode_runs,
    run_dummy_pipeline,
    run_real_pipeline,
)
from rsfm_fairness_audit.preflight import checks_to_json, run_real_preflight
from rsfm_fairness_audit.reben_sensor_audit import (
    compute_selective_risk,
    read_label_expanded_predictions,
    run_reben_multilabel_bwer,
)
from rsfm_fairness_audit.segmentation import run_segmentation_smoke
from rsfm_fairness_audit.spectral_baseline import SpectralBaselineConfig, run_spectral_sen1floods11
from rsfm_fairness_audit.spatial_conformal import SpatialConformalConfig
from rsfm_fairness_audit.slice_support import evaluate_slice_support_from_files
from rsfm_fairness_audit.unet_baseline import UnetConfig, run_unet_sen1floods11


def _parse_wavelengths(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _load_geobwer_protocol(path: Path) -> BWERProtocol:
    if path.suffix.lower() == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
    else:
        values = load_yaml(path)
    if not isinstance(values, dict):
        raise ValueError("GeoBWER protocol must be a YAML/JSON mapping.")
    return BWERProtocol.from_mapping(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsfm-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dummy = subparsers.add_parser("run-dummy", help="Run the CPU-only synthetic fairness audit.")
    dummy.add_argument("--output-dir", type=Path, default=Path("outputs/dummy_smoke"))
    dummy.add_argument("--num-samples", type=int, default=240)
    dummy.add_argument("--seed", type=int, default=7)

    real = subparsers.add_parser("run-real", help="Run a subset-first real dataset/model smoke audit.")
    real.add_argument("--dataset", choices=["bigearthnet", "ben_ge", "sen1floods11"], required=True)
    real.add_argument("--model", choices=["dofa", "croma", "prithvi", "prithvi_tl_sen1floods11"], required=True)
    real.add_argument("--data-root", "--dataset-root", dest="data_root", type=Path, required=True)
    real.add_argument("--metadata-path", type=Path)
    real.add_argument("--subset-size", "--max-samples", dest="subset_size", type=int)
    real.add_argument("--subset-manifest-path", type=Path)
    real.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    real.add_argument("--sensor-mode", choices=["S1", "S2", "S1+S2"], default="S2")
    real.add_argument("--output-dir", type=Path, default=Path("outputs/runs/dofa_bigearthnet_subset"))
    real.add_argument("--chunk-size", type=int, default=256)
    real.add_argument("--streaming-embeddings", default="false", help="true/false; write embedding chunks during extraction.")
    real.add_argument("--model-config", "--config", dest="model_config", type=Path, help="YAML config for the real model adapter.")
    real.add_argument(
        "--dofa-wavelengths",
        type=str,
        help="Comma-separated official wavelength list matching the subset band order.",
    )
    real.add_argument(
        "--allow-torch-hub-download",
        action="store_true",
        help="Explicitly opt into the official torch.hub DOFA loading path, which may download weights.",
    )

    check = subparsers.add_parser("check-real", help="Preflight-check a real dataset/model smoke run.")
    check.add_argument("--dataset", choices=["bigearthnet", "ben_ge", "sen1floods11"], required=True)
    check.add_argument("--model", choices=["dofa", "croma", "prithvi", "prithvi_tl_sen1floods11"], required=True)
    check.add_argument("--model-config", type=Path, required=True)
    check.add_argument("--data-root", type=Path, required=True)
    check.add_argument("--metadata-path", type=Path)
    check.add_argument("--subset-manifest-path", type=Path)
    check.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    check.add_argument("--sensor-mode", choices=["S1", "S2", "S1+S2"], default="S2")
    check.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    compare = subparsers.add_parser("compare-runs", help="Compare completed runs; segmentation outputs use protocol-aware BWER v2 comparison.")
    compare.add_argument("--dataset", default="bigearthnet")
    compare.add_argument(
        "--run",
        action="append",
        required=True,
        help="Model/run pair in the form model_name=path_to_output_dir. Repeat for DOFA and CROMA.",
    )
    compare.add_argument("--output-dir", type=Path, default=Path("outputs/model_comparison"))
    compare.add_argument("--closure", action="store_true", help="Also emit Sen1Floods11 closure_* comparison outputs.")

    sensor_compare = subparsers.add_parser("compare-sensor-modes", help="Compare CROMA SAR/optical/both completed runs.")
    sensor_compare.add_argument("--dataset", default="ben_ge")
    sensor_compare.add_argument(
        "--run",
        action="append",
        required=True,
        help="Sensor-mode/run pair in the form sar=path, optical=path, or both=path.",
    )
    sensor_compare.add_argument("--output-dir", type=Path, default=Path("outputs/comparisons/croma_sensor_modes"))

    protocol_matched = subparsers.add_parser("protocol-match-runs", help="Post-hoc chip-intersection comparison for completed Sen1Floods11 segmentation runs.")
    protocol_matched.add_argument("--run", action="append", required=True, help="Run pair in the form run_name=path_to_output_dir.")
    protocol_matched.add_argument("--output-dir", type=Path, default=Path("outputs/comparisons/sen1floods11_protocol_matched"))

    selective = subparsers.add_parser("run-selective-risk", help="Post-hoc confidence-conditioned selective risk audit for completed segmentation runs.")
    selective.add_argument("--run", action="append", required=True, help="Run pair in the form run_name=path_to_output_dir.")
    selective.add_argument("--output-dir", type=Path, default=Path("outputs/comparisons/sen1floods11_selective_risk"))
    selective.add_argument("--coverage", action="append", type=float, help="Coverage level to evaluate. Repeat as needed; default is 1.0,0.9,0.8,0.7,0.6,0.5.")

    loeo = subparsers.add_parser("aggregate-loeo", help="Aggregate completed leave-one-event-out supervised baseline run directories.")
    loeo.add_argument("--input-root", type=Path, required=True, help="Directory containing one completed run subdirectory per held-out event.")
    loeo.add_argument("--output-dir", type=Path, required=True, help="Directory for LOEO aggregate outputs.")

    fmow = subparsers.add_parser("preflight-fmow-sentinel", help="Build fMoW-Sentinel metadata, subset, raster, and audit-table preflight outputs.")
    fmow.add_argument("--metadata-csv", action="append", type=Path, required=True, help="Input fMoW-Sentinel metadata CSV. Repeat for train/val/test files.")
    fmow.add_argument("--output-dir", type=Path, required=True)
    fmow.add_argument("--data-root", type=Path, help="Optional root used to resolve relative image_path values during raster inspection.")
    fmow.add_argument("--split", dest="split_protocol", default="official_split", choices=["official_split", "location_split", "location_disjoint", "region_split", "time_split", "custom_stratified_subset"])
    fmow.add_argument("--filter-split", action="append", default=[], help="Optional official split value to include, e.g. train. Repeat as needed.")
    fmow.add_argument("--subset-max-per-split", type=int, default=5000)
    fmow.add_argument("--min-support", type=int, default=20)
    fmow.add_argument("--inspect-rasters", action="store_true")
    fmow.add_argument("--raster-sample-size", type=int, default=256)
    fmow.add_argument("--metadata-only", action="store_true")
    fmow.add_argument("--seed", type=int, default=42)
    fmow.add_argument("--country-region-map", type=Path, help="Optional verified CSV mapping country to continent/UN region/region.")

    fmow_enrich = subparsers.add_parser("enrich-fmow-sentinel-metadata", help="Join SatMAE fMoW-Sentinel CSVs with optional fMoW/GPS/geography metadata.")
    fmow_enrich.add_argument("--satmae-csv", action="append", type=Path, required=True, help="SatMAE fMoW-Sentinel train/val/test CSV. Repeat as needed.")
    fmow_enrich.add_argument("--external-metadata-csv", "--external-metadata", dest="external_metadata_csv", action="append", type=Path, default=[], help="Optional original fMoW/GPS/geography CSV to join. Repeat as needed.")
    fmow_enrich.add_argument("--country-region-map", type=Path, help="Optional verified CSV mapping country to continent/UN region/region.")
    fmow_enrich.add_argument("--output-dir", type=Path, required=True)
    fmow_enrich.add_argument("--join-key", default="auto", help="auto or + separated key fields, e.g. category+location_id+image_id or location_id.")
    fmow_enrich.add_argument("--no-infer-split-from-filename", action="store_true", help="Do not fill missing split from train/val/test CSV filenames.")

    fmow_cls = subparsers.add_parser("run-fmow-sentinel-classification", help="Run Step 3 fMoW-Sentinel image-only classification prototype.")
    fmow_cls.add_argument("--metadata-csv", type=Path, required=True, help="Final enriched fMoW-Sentinel metadata or subset manifest CSV.")
    fmow_cls.add_argument("--data-root", type=Path, help="Root used to resolve relative image_path values.")
    fmow_cls.add_argument("--output-dir", type=Path, required=True)
    fmow_cls.add_argument("--model", choices=["supervised_stats", "dofa", "dofav2", "resnet50"], default="supervised_stats")
    fmow_cls.add_argument("--model-config", type=Path, help="DOFA config. Defaults to the frozen v1/v2 task config selected by --model.")
    fmow_cls.add_argument("--probe", choices=["linear", "nearest_centroid"], default="linear", help="Probe used for --model dofa. Formal path is linear.")
    fmow_cls.add_argument("--probe-epochs", type=int, default=200)
    fmow_cls.add_argument("--probe-learning-rate", type=float, default=1e-2)
    fmow_cls.add_argument("--embedding-cache-dir", type=Path, help="Optional cache directory for frozen encoder embeddings.")
    fmow_cls.add_argument("--dofa-input-scale", type=float, help="Override DOFA input_scale from config, e.g. 10000 for raw fMoW-Sentinel reflectance-like values.")
    fmow_cls.add_argument("--dofa-embedding-pooling", choices=["flatten", "mean_tokens"], help="DOFA feature pooling for ablation runs. Formal completed run used flatten.")
    fmow_cls.add_argument("--train-split", default="train")
    fmow_cls.add_argument("--eval-split", default="val")
    fmow_cls.add_argument("--max-samples", type=int, help="Backward-compatible per-split prototype cap. Omit or pass 0 for all rows.")
    fmow_cls.add_argument("--max-samples-per-split", type=int, help="Explicit per-split cap applied after train/eval split filtering.")
    fmow_cls.add_argument("--image-size", type=int, default=96)
    fmow_cls.add_argument("--batch-size", type=int, default=32)
    fmow_cls.add_argument("--epochs", type=int, default=20)
    fmow_cls.add_argument("--learning-rate", type=float, default=1e-3)
    fmow_cls.add_argument("--weight-decay", type=float, default=1e-4)
    fmow_cls.add_argument("--checkpoint-metric", choices=["macro_f1", "accuracy"], default="macro_f1", help="Validation metric used for ResNet-50 checkpoint selection.")
    fmow_cls.add_argument("--num-workers", type=int, default=2)
    fmow_cls.add_argument("--device", default="auto")
    fmow_cls.add_argument("--norm-stats", type=Path, help="Optional train-only norm_stats.json to reuse for ResNet-50.")
    fmow_cls.add_argument("--amp", default="true", help="true/false; enable CUDA AMP for ResNet-50.")
    fmow_cls.add_argument("--seed", type=int, default=42)
    fmow_cls.add_argument("--split-protocol", choices=["official_split", "location_split", "location_disjoint", "random_split_sanity", "region_split", "time_split", "custom_stratified_subset"], default="official_split")
    fmow_cls.add_argument("--eval-scope", default="val")
    fmow_cls.add_argument("--band-profile", default="sentinel2_13band_fmow")
    fmow_cls.add_argument("--allow-torch-hub-download", action="store_true", help="Explicitly allow DOFA torch.hub download for Colab runs.")
    fmow_cls.add_argument("--run-bwer", action="store_true", help="Run geography BWER immediately after writing predictions.")
    fmow_cls.add_argument("--bwer-bootstrap", type=int, default=0)
    fmow_cls.add_argument("--write-formal-outputs", action="store_true", help="Write the complete probability matrix and versioned GeoBWER audit contract.")
    fmow_cls.add_argument("--geobwer-protocol", type=Path, default=Path("configs/geobwer/fmow_sentinel.yaml"))

    fmow_bwer = subparsers.add_parser("run-fmow-geography-bwer", help="Run post-hoc geography BWER on completed fMoW-Sentinel predictions.")
    fmow_bwer.add_argument("--input-dir", type=Path, required=True, help="Completed fMoW-Sentinel run directory containing audit_table.csv or predictions.csv.")
    fmow_bwer.add_argument("--output-dir", type=Path, required=True)
    fmow_bwer.add_argument("--audit-table", type=Path)
    fmow_bwer.add_argument("--bootstrap", type=int, default=0)
    fmow_bwer.add_argument("--seed", type=int, default=42)

    fmow_compare = subparsers.add_parser("compare-fmow-runs", help="Compare completed fMoW-Sentinel classification+BWER runs.")
    fmow_compare.add_argument("--run", action="append", required=True, help="Run pair in the form run_name=path_to_output_dir.")
    fmow_compare.add_argument("--output-dir", type=Path, required=True)

    fmow_validate = subparsers.add_parser("validate-fmow-step3-results", help="Validate fMoW-Sentinel Step 3 result contract and write handoff readiness files.")
    fmow_validate.add_argument("--run-dir", type=Path, required=True)
    fmow_validate.add_argument("--output-dir", type=Path, help="Directory for validation outputs; defaults to --run-dir.")
    fmow_validate.add_argument("--run-name")
    fmow_validate.add_argument("--archive-source-url", default="https://stacks.stanford.edu/file/druid:vg497cb6002/fmow-sentinel.tar.gz")
    fmow_validate.add_argument("--full-archive-downloaded-locally", choices=["true", "false", "unknown"], default="unknown")
    fmow_validate.add_argument("--full-extraction-avoided", default="true")
    fmow_validate.add_argument("--streaming-partial-extraction-excluded", default="true")

    fmow_package = subparsers.add_parser("package-fmow-step3-handoff", help="Package a completed fMoW-Sentinel Step 3 result directory into a small handoff zip.")
    fmow_package.add_argument("--run-dir", type=Path, required=True)
    fmow_package.add_argument("--output-zip", type=Path)
    fmow_package.add_argument("--include-rasters", action="store_true", help="Include raster/image files. Off by default.")

    reben_bwer = subparsers.add_parser("run-reben-multilabel-bwer", help="Run post-hoc multi-label BWER for BigEarthNet v2.0 / reBEN label-expanded predictions.")
    reben_bwer.add_argument("--predictions", type=Path, required=True, help="Label-expanded predictions/audit table with one row per sample x class.")
    reben_bwer.add_argument("--output-dir", type=Path, required=True)
    reben_bwer.add_argument("--model-name", required=True)
    reben_bwer.add_argument("--split", default="validation")
    reben_bwer.add_argument("--risk-column", choices=["risk_bce", "risk_binary_error"], default="risk_bce")
    reben_bwer.add_argument("--alpha", type=float, default=0.1)
    reben_bwer.add_argument("--min-support", type=int, default=20)
    reben_bwer.add_argument("--selective-risk", action="store_true")

    seg = subparsers.add_parser("run-segmentation-real", help="Run native Sen1Floods11 segmentation metrics, preflight, and BWER audit.")
    seg.add_argument("--dataset", choices=["sen1floods11"], required=True)
    seg.add_argument("--model", choices=["prithvi", "prithvi_tl_sen1floods11"], required=True)
    seg.add_argument("--data-root", "--dataset-root", dest="data_root", type=Path, required=True)
    seg.add_argument("--metadata-path", type=Path)
    seg.add_argument("--subset-size", "--max-samples", dest="subset_size", type=int)
    seg.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    seg.add_argument("--output-dir", type=Path, default=Path("outputs/prithvi_sen1floods11_seg64"))
    seg.add_argument("--model-config", "--config", dest="model_config", type=Path, required=True)
    seg.add_argument("--debug-samples", type=int, default=0, help="Save raw output/probability diagnostics and quick-look PNGs for the first N chips.")

    unet = subparsers.add_parser("run-unet-sen1floods11", help="Train and evaluate the supervised U-Net Sen1Floods11 segmentation baseline.")
    unet.add_argument("--data-root", "--dataset-root", dest="data_root", type=Path, required=True)
    unet.add_argument("--output-dir", type=Path, default=Path("outputs/unet_sen1floods11_full_512"))
    unet.add_argument("--epochs", type=int, default=50)
    unet.add_argument("--batch-size", type=int, default=4)
    unet.add_argument("--learning-rate", type=float, default=1e-3)
    unet.add_argument("--weight-decay", type=float, default=1e-4)
    unet.add_argument("--base-channels", type=int, default=16)
    unet.add_argument("--architecture", choices=["vanilla_unet", "s2_resnet34_unet"], default="vanilla_unet")
    unet.add_argument("--pretrained-encoder", action="store_true", help="Use torchvision ResNet34 ImageNet weights and adapt conv1 from 3 to 6 bands.")
    unet.add_argument("--early-stopping-patience", type=int, default=10)
    unet.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    unet.add_argument("--split-protocol", choices=["random_chip_split", "event_held_out", "leave_one_event_out"], default="random_chip_split")
    unet.add_argument("--val-fraction", type=float, default=0.15)
    unet.add_argument("--test-fraction", type=float, default=0.20)
    unet.add_argument("--held-out-event", dest="held_out_events", action="append", default=[])
    unet.add_argument("--seed", type=int, default=42)
    unet.add_argument("--device", default="auto")
    unet.add_argument("--amp", default="true", help="true/false; enable CUDA AMP when a CUDA device is used.")
    unet.add_argument("--max-samples", type=int, help="Limit chips for a smoke run. Use 0 or omit for all chips.")
    unet.add_argument("--eval-split", choices=["train", "val", "test", "all"], default="test")
    unet.add_argument("--run-bwer-v2", action="store_true", help="Run post-hoc BWER-Audit v2 after U-Net evaluation.")
    unet.add_argument("--debug-samples", type=int, default=0, help="Reserved for future quick-look debug samples.")

    spectral = subparsers.add_parser("run-spectral-sen1floods11", help="Evaluate diagnostic NDWI/MNDWI/NIR Sen1Floods11 spectral segmentation baselines.")
    spectral.add_argument("--data-root", "--dataset-root", dest="data_root", type=Path, required=True)
    spectral.add_argument("--output-dir", type=Path, default=Path("outputs/spectral_mndwi_sen1floods11_full_512"))
    spectral.add_argument("--index", choices=["ndwi", "mndwi", "nir_darkness"], default="mndwi")
    spectral.add_argument("--threshold", type=float, default=0.0)
    spectral.add_argument("--threshold-policy", choices=["fixed", "validation", "oracle_diagnostic"], default="fixed")
    spectral.add_argument("--split-protocol", choices=["standard_split", "random_chip_split"], default="standard_split")
    spectral.add_argument("--val-fraction", type=float, default=0.15)
    spectral.add_argument("--test-fraction", type=float, default=0.20)
    spectral.add_argument("--eval-split", choices=["val", "test", "all"], default="all")
    spectral.add_argument("--seed", type=int, default=42)
    spectral.add_argument("--max-samples", type=int, help="Limit chips for a smoke run. Use 0 or omit for all chips.")
    spectral.add_argument("--run-bwer-v2", action="store_true", help="Run post-hoc BWER-Audit v2 after spectral evaluation.")

    bwer = subparsers.add_parser("evaluate-bwer", help="Evaluate BWER slice fairness from a normalized audit table.")
    bwer.add_argument("--audit-table", type=Path, required=True)
    bwer.add_argument("--dataset", required=True)
    bwer.add_argument("--model", required=True)
    bwer.add_argument("--task", required=True)
    bwer.add_argument("--slice-config", type=Path, default=Path("configs/slice_taxonomy.yaml"))
    bwer.add_argument("--output-dir", type=Path, required=True)
    bwer.add_argument("--slice-variable")
    bwer.add_argument("--balance-variable")
    bwer.add_argument("--tail-fraction", type=float)
    bwer.add_argument("--bootstrap", type=int, default=0)
    bwer.add_argument("--cluster-key")
    bwer.add_argument("--seed", type=int, default=42)
    bwer.add_argument("--weighting", choices=["uniform", "empirical"], default="uniform")
    bwer.add_argument("--missing-balance-policy", choices=["renormalize", "invalidate", "overlap"], default="renormalize")
    bwer.add_argument("--score-column")
    bwer.add_argument("--risk-column")
    bwer.add_argument("--selective-coverage", type=float, help="Reserved hook for future fixed-coverage selective_risk runs, e.g. 0.8.")
    bwer.add_argument("--audit-level", choices=["smoke", "pilot", "paper"], default="pilot")

    bwer_v2 = subparsers.add_parser("run-bwer-v2", help="Generate post-hoc BWER-Audit v2 outputs from a completed audit output directory.")
    bwer_v2.add_argument("--input-dir", type=Path, required=True, help="Completed run output directory containing event_segmentation_metrics.csv.")
    bwer_v2.add_argument("--output-dir", type=Path, required=True, help="Directory to write BWER-Audit v2 outputs, usually <input-dir>/bwer_v2.")
    bwer_v2.add_argument("--bootstrap", type=int, default=1000, help="Post-hoc event bootstrap replicates for Raw-BWER CI.")
    bwer_v2.add_argument("--seed", type=int, default=42)

    audit = subparsers.add_parser("run-audit", help="Build an audit table from existing outputs and evaluate BWER.")
    audit.add_argument("--predictions", type=Path)
    audit.add_argument("--metadata", type=Path)
    audit.add_argument("--segmentation-metrics", type=Path)
    audit.add_argument("--dataset", required=True)
    audit.add_argument("--model", required=True)
    audit.add_argument("--task", required=True)
    audit.add_argument("--slice-config", type=Path, default=Path("configs/slice_taxonomy.yaml"))
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--slice-variable")
    audit.add_argument("--balance-variable")
    audit.add_argument("--tail-fraction", type=float)
    audit.add_argument("--bootstrap", type=int, default=0)
    audit.add_argument("--cluster-key")
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--weighting", choices=["uniform", "empirical"], default="uniform")
    audit.add_argument("--missing-balance-policy", choices=["renormalize", "invalidate", "overlap"], default="renormalize")
    audit.add_argument("--score-column")
    audit.add_argument("--risk-column")
    audit.add_argument("--selective-coverage", type=float, help="Reserved hook for future fixed-coverage selective_risk runs, e.g. 0.8.")
    audit.add_argument("--audit-level", choices=["smoke", "pilot", "paper"], default="pilot")

    support = subparsers.add_parser("preflight-bwer", help="Check slice support before paper-grade BWER runs.")
    support.add_argument("--audit-table", type=Path)
    support.add_argument("--predictions", type=Path)
    support.add_argument("--metadata", type=Path)
    support.add_argument("--segmentation-metrics", type=Path)
    support.add_argument("--dataset", required=True)
    support.add_argument("--model", required=True)
    support.add_argument("--task", required=True)
    support.add_argument("--slice-config", type=Path, default=Path("configs/slice_taxonomy.yaml"))
    support.add_argument("--output-dir", type=Path, required=True)
    support.add_argument(
        "--candidate",
        action="append",
        help="Candidate in the form slice or slice|balance, for example climatezone|class_label. Repeat as needed.",
    )
    support.add_argument("--min-samples-per-slice", type=int)
    support.add_argument("--min-units-required", type=int)
    support.add_argument("--min-slices-required", type=int)
    support.add_argument("--score-column")
    support.add_argument("--risk-column")

    geobwer = subparsers.add_parser("geobwer-audit", help="Run the versioned fractional GeoBWER audit with fail-fast formal inference.")
    geobwer.add_argument("--audit-table", type=Path, required=True)
    geobwer.add_argument("--protocol", type=Path, default=Path("configs/geobwer/default.yaml"))
    geobwer.add_argument("--group-column", action="append", help="Slice column; repeat for multiple pre-registered axes. Defaults to protocol group_variable.")
    geobwer.add_argument("--loss-column", help="Loss/risk column; defaults to protocol loss_name.")
    geobwer.add_argument("--unit-column", help="Independent unit column; defaults to protocol independent_unit_column.")
    geobwer.add_argument("--cluster-column", help="Cluster/spatial block column; defaults to the protocol.")
    geobwer.add_argument("--balance-column", help="Optional standardisation variable.")
    geobwer.add_argument("--output-dir", type=Path, required=True)
    geobwer.add_argument("--bootstrap", type=int, default=2000)
    geobwer.add_argument("--seed", type=int, default=42)
    geobwer.add_argument("--require-probabilities", action="store_true")
    geobwer.add_argument("--diagnostic", action="store_true", help="Allow descriptive execution when the formal schema/inference unit is unavailable.")

    geobwer_cmp = subparsers.add_parser("geobwer-compare", help="Paired common-unit GeoBWER model comparison.")
    geobwer_cmp.add_argument("--model-a-table", type=Path, required=True)
    geobwer_cmp.add_argument("--model-b-table", type=Path, required=True)
    geobwer_cmp.add_argument("--model-a", default="model_a")
    geobwer_cmp.add_argument("--model-b", default="model_b")
    geobwer_cmp.add_argument("--protocol", type=Path, default=Path("configs/geobwer/default.yaml"))
    geobwer_cmp.add_argument("--group-column", help="Defaults to protocol group_variable.")
    geobwer_cmp.add_argument("--loss-column", help="Defaults to protocol loss_name.")
    geobwer_cmp.add_argument("--unit-column", help="Defaults to protocol independent_unit_column.")
    geobwer_cmp.add_argument("--cluster-column", help="Defaults to the protocol cluster/spatial block column.")
    geobwer_cmp.add_argument("--output-dir", type=Path, required=True)
    geobwer_cmp.add_argument("--bootstrap", type=int, default=2000)
    geobwer_cmp.add_argument("--seed", type=int, default=42)

    geobwer_inventory = subparsers.add_parser("geobwer-inventory", help="Inspect existing CSV/ZIP artifacts without rerunning models.")
    geobwer_inventory.add_argument("--input", action="append", type=Path, required=True, help="File or directory to scan; repeat as needed.")
    geobwer_inventory.add_argument("--protocol", type=Path, default=Path("configs/geobwer/default.yaml"))
    geobwer_inventory.add_argument("--output-dir", type=Path, required=True)

    geobwer_mc_uncertainty = subparsers.add_parser(
        "geobwer-multiclass-uncertainty",
        help="Run validation-calibrated LAC/APS/RAPS and selective GeoBWER on a formal multiclass bundle.",
    )
    geobwer_mc_uncertainty.add_argument("--calibration-probabilities", type=Path, required=True)
    geobwer_mc_uncertainty.add_argument("--calibration-manifest", type=Path)
    geobwer_mc_uncertainty.add_argument("--test-formal-dir", type=Path, required=True)
    geobwer_mc_uncertainty.add_argument("--protocol", type=Path, required=True)
    geobwer_mc_uncertainty.add_argument("--group-column", action="append", required=True)
    geobwer_mc_uncertainty.add_argument("--conformal-method", action="append", choices=["lac", "aps", "raps"])
    geobwer_mc_uncertainty.add_argument("--selective-coverage", action="append", type=float)
    geobwer_mc_uncertainty.add_argument("--alpha", type=float, default=0.10)
    geobwer_mc_uncertainty.add_argument("--bootstrap", type=int, default=2000)
    geobwer_mc_uncertainty.add_argument("--seed", type=int, default=42)
    geobwer_mc_uncertainty.add_argument(
        "--spatial-localization",
        action="store_true",
        help="Run the preregistered geographic-kernel comparator when its spatial preflight passes.",
    )
    geobwer_mc_uncertainty.add_argument("--output-dir", type=Path, required=True)

    geobwer_ml_uncertainty = subparsers.add_parser(
        "geobwer-multilabel-uncertainty",
        help="Run validation-calibrated false-negative CRC and selective GeoBWER on a formal multilabel bundle.",
    )
    geobwer_ml_uncertainty.add_argument("--calibration-probabilities", type=Path, required=True)
    geobwer_ml_uncertainty.add_argument("--calibration-manifest", type=Path)
    geobwer_ml_uncertainty.add_argument("--test-formal-dir", type=Path, required=True)
    geobwer_ml_uncertainty.add_argument("--protocol", type=Path, required=True)
    geobwer_ml_uncertainty.add_argument("--group-column", action="append", required=True)
    geobwer_ml_uncertainty.add_argument("--selective-coverage", action="append", type=float)
    geobwer_ml_uncertainty.add_argument("--crc-alpha", type=float, default=0.10)
    geobwer_ml_uncertainty.add_argument("--bootstrap", type=int, default=2000)
    geobwer_ml_uncertainty.add_argument("--seed", type=int, default=42)
    geobwer_ml_uncertainty.add_argument(
        "--spatial-localization-preflight",
        action="store_true",
        help="Record all-task spatial-localization eligibility without replacing multilabel CRC.",
    )
    geobwer_ml_uncertainty.add_argument("--output-dir", type=Path, required=True)

    geobwer_spatial_upgrade = subparsers.add_parser(
        "geobwer-spatial-conformal-upgrade",
        help="Add coordinates to existing multiclass calibration probabilities and run the spatial comparator without rerunning the model.",
    )
    geobwer_spatial_upgrade.add_argument("--calibration-probabilities", type=Path, required=True)
    geobwer_spatial_upgrade.add_argument("--calibration-metadata", type=Path, required=True)
    geobwer_spatial_upgrade.add_argument("--calibration-manifest", type=Path)
    geobwer_spatial_upgrade.add_argument("--test-formal-dir", type=Path, required=True)
    geobwer_spatial_upgrade.add_argument("--protocol", type=Path, required=True)
    geobwer_spatial_upgrade.add_argument("--group-column", action="append", required=True)
    geobwer_spatial_upgrade.add_argument("--conformal-method", action="append", choices=["lac", "aps", "raps"])
    geobwer_spatial_upgrade.add_argument("--alpha", type=float, default=0.10)
    geobwer_spatial_upgrade.add_argument("--bootstrap", type=int, default=2000)
    geobwer_spatial_upgrade.add_argument("--seed", type=int, default=42)
    geobwer_spatial_upgrade.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run-dummy":
        artifacts = run_dummy_pipeline(args.output_dir, num_samples=args.num_samples, seed=args.seed)
        print(f"Dummy fairness audit complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-real":
        dataset, model = build_real_adapters(
            dataset_name=args.dataset,
            model_name=args.model,
            data_root=args.data_root,
            metadata_path=args.metadata_path,
            subset_manifest_path=args.subset_manifest_path,
            subset_size=args.subset_size,
            split=args.split,
            sensor_mode=args.sensor_mode,
            dofa_wavelengths=_parse_wavelengths(args.dofa_wavelengths),
            allow_torch_hub_download=args.allow_torch_hub_download,
            model_config=args.model_config,
        )
        artifacts = run_real_pipeline(
            dataset,
            model,
            args.output_dir,
            args.dataset,
            args.model,
            chunk_size=args.chunk_size,
            streaming_embeddings=_parse_bool(args.streaming_embeddings),
        )
        print(f"Real smoke audit complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "check-real":
        checks = run_real_preflight(
            model=args.model,
            dataset=args.dataset,
            model_config=args.model_config,
            data_root=args.data_root,
            metadata_path=args.metadata_path,
            subset_manifest_path=args.subset_manifest_path,
            split=args.split,
            sensor_mode=args.sensor_mode,
        )
        if args.json:
            print(checks_to_json(checks))
        else:
            for check in checks:
                print(f"[{check.status.upper()}] {check.name}: {check.message}")
        if any(check.status == "fail" for check in checks):
            raise SystemExit(1)
    elif args.command == "compare-runs":
        runs = {}
        for item in args.run:
            if "=" not in item:
                raise SystemExit("--run must use the form model_name=path_to_output_dir")
            model_name, run_dir = item.split("=", 1)
            runs[model_name.strip()] = Path(run_dir.strip())
        artifacts = compare_model_runs(runs, args.output_dir, dataset_name=args.dataset, closure=args.closure)
        print(f"Model comparison complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "compare-sensor-modes":
        runs = {}
        for item in args.run:
            if "=" not in item:
                raise SystemExit("--run must use the form sensor_mode=path_to_output_dir")
            sensor_mode, run_dir = item.split("=", 1)
            runs[sensor_mode.strip()] = Path(run_dir.strip())
        artifacts = compare_sensor_mode_runs(runs, args.output_dir, dataset_name=args.dataset)
        print(f"Sensor-mode comparison complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "protocol-match-runs":
        runs = {}
        for item in args.run:
            if "=" not in item:
                raise SystemExit("--run must use the form run_name=path_to_output_dir")
            run_name, run_dir = item.split("=", 1)
            runs[run_name.strip()] = Path(run_dir.strip())
        artifacts = run_protocol_matched_comparison(runs, args.output_dir)
        print(f"Protocol-matched comparison complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-selective-risk":
        runs = {}
        for item in args.run:
            if "=" not in item:
                raise SystemExit("--run must use the form run_name=path_to_output_dir")
            run_name, run_dir = item.split("=", 1)
            runs[run_name.strip()] = Path(run_dir.strip())
        coverages = tuple(args.coverage) if args.coverage else (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
        artifacts = run_selective_risk_audit(runs, args.output_dir, coverages=coverages)
        print(f"Selective Risk audit complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "aggregate-loeo":
        artifacts = aggregate_loeo_runs(args.input_root, args.output_dir)
        print(f"LOEO aggregate complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "preflight-fmow-sentinel":
        artifacts = run_fmow_sentinel_preflight(
            FmowPreflightConfig(
                metadata_csvs=tuple(args.metadata_csv),
                output_dir=args.output_dir,
                data_root=args.data_root,
                split_protocol=args.split_protocol,
                filter_splits=tuple(args.filter_split or ()),
                subset_max_per_split=args.subset_max_per_split,
                seed=args.seed,
                metadata_only=args.metadata_only,
                inspect_rasters=args.inspect_rasters,
                raster_sample_size=args.raster_sample_size,
                min_support=args.min_support,
                country_region_map=args.country_region_map,
            )
        )
        print(f"fMoW-Sentinel preflight complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "enrich-fmow-sentinel-metadata":
        artifacts = run_fmow_sentinel_metadata_enrichment(
            FmowMetadataEnrichmentConfig(
                satmae_csvs=tuple(args.satmae_csv),
                external_metadata_csvs=tuple(args.external_metadata_csv or ()),
                output_dir=args.output_dir,
                join_key=args.join_key,
                infer_split_from_filename=not args.no_infer_split_from_filename,
                country_region_map=args.country_region_map,
            )
        )
        print(f"fMoW-Sentinel metadata enrichment complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-fmow-sentinel-classification":
        max_samples = None if args.max_samples in (None, 0) else args.max_samples
        artifacts = run_fmow_sentinel_classification(
            FmowClassificationConfig(
                metadata_csv=args.metadata_csv,
                data_root=args.data_root,
                output_dir=args.output_dir,
                model=args.model,
                model_config=args.model_config
                or (Path("configs/models/dofav2_fmow_sentinel.yaml") if args.model == "dofav2" else Path("configs/models/dofa_fmow_sentinel.yaml")),
                probe=args.probe,
                probe_epochs=args.probe_epochs,
                probe_learning_rate=args.probe_learning_rate,
                embedding_cache_dir=args.embedding_cache_dir,
                dofa_input_scale=args.dofa_input_scale,
                dofa_embedding_pooling=args.dofa_embedding_pooling,
                train_split=args.train_split,
                eval_split=args.eval_split,
                max_samples=max_samples,
                max_samples_per_split=None if args.max_samples_per_split in (None, 0) else args.max_samples_per_split,
                image_size=args.image_size,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                checkpoint_metric=args.checkpoint_metric,
                num_workers=args.num_workers,
                device=args.device,
                norm_stats=args.norm_stats,
                seed=args.seed,
                split_protocol=args.split_protocol,
                eval_scope=args.eval_scope,
                band_profile=args.band_profile,
                allow_torch_hub_download=args.allow_torch_hub_download,
                amp=_parse_bool(args.amp),
                run_bwer=args.run_bwer,
                bwer_bootstrap=args.bwer_bootstrap,
                write_formal_outputs=args.write_formal_outputs,
                geobwer_protocol=args.geobwer_protocol,
            )
        )
        print(f"fMoW-Sentinel classification prototype complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-fmow-geography-bwer":
        artifacts = run_fmow_geography_bwer(
            FmowBwerConfig(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                audit_table=args.audit_table,
                bootstrap=args.bootstrap,
                seed=args.seed,
            )
        )
        print(f"fMoW-Sentinel geography BWER complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "compare-fmow-runs":
        runs = {}
        for item in args.run:
            if "=" not in item:
                raise SystemExit("--run must use the form run_name=path_to_output_dir")
            run_name, run_dir = item.split("=", 1)
            runs[run_name.strip()] = Path(run_dir.strip())
        artifacts = compare_fmow_runs(runs, args.output_dir)
        print(f"fMoW-Sentinel run comparison complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "validate-fmow-step3-results":
        local_archive = None
        if args.full_archive_downloaded_locally != "unknown":
            local_archive = args.full_archive_downloaded_locally == "true"
        artifacts = validate_fmow_step3_results(
            FmowStep3ValidationConfig(
                run_dir=args.run_dir,
                output_dir=args.output_dir,
                run_name=args.run_name,
                archive_source_url=args.archive_source_url,
                full_archive_downloaded_locally=local_archive,
                full_extraction_avoided=_parse_bool(args.full_extraction_avoided),
                streaming_partial_extraction_excluded=_parse_bool(args.streaming_partial_extraction_excluded),
            )
        )
        print(f"fMoW-Sentinel Step 3 validation complete: {args.output_dir or args.run_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "package-fmow-step3-handoff":
        artifacts = package_fmow_step3_handoff(
            FmowStep3PackageConfig(
                run_dir=args.run_dir,
                output_zip=args.output_zip,
                include_rasters=args.include_rasters,
            )
        )
        print("fMoW-Sentinel Step 3 handoff package complete.")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-reben-multilabel-bwer":
        rows = read_label_expanded_predictions(args.predictions)
        artifacts = run_reben_multilabel_bwer(
            rows,
            args.output_dir,
            model_name=args.model_name,
            split=args.split,
            risk_column=args.risk_column,
            alpha=args.alpha,
            min_support=args.min_support,
        )
        if args.selective_risk:
            from rsfm_fairness_audit.io import write_csv

            selective_path = args.output_dir / "selective_risk_summary.csv"
            write_csv(selective_path, compute_selective_risk(rows, risk_column=args.risk_column))
            artifacts["selective_risk_summary"] = selective_path
        print("reBEN multi-label BWER complete.")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-segmentation-real":
        dataset, model = build_real_adapters(
            dataset_name=args.dataset,
            model_name=args.model,
            data_root=args.data_root,
            metadata_path=args.metadata_path,
            subset_manifest_path=None,
            subset_size=args.subset_size,
            split=args.split,
            sensor_mode="S2",
            model_config=args.model_config,
        )
        artifacts = run_segmentation_smoke(dataset, model, args.output_dir, debug_samples=args.debug_samples)
        print(f"Real segmentation smoke complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-unet-sen1floods11":
        max_samples = None if args.max_samples in (None, 0) else args.max_samples
        artifacts = run_unet_sen1floods11(
            UnetConfig(
                data_root=args.data_root,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                base_channels=args.base_channels,
                architecture=args.architecture,
                pretrained_encoder=args.pretrained_encoder,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                split_protocol=args.split_protocol,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                held_out_events=tuple(args.held_out_events or ()),
                seed=args.seed,
                device=args.device,
                amp=_parse_bool(args.amp),
                max_samples=max_samples,
                eval_split=args.eval_split,
                run_bwer_v2=args.run_bwer_v2,
                debug_samples=args.debug_samples,
            )
        )
        print(f"U-Net Sen1Floods11 baseline complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-spectral-sen1floods11":
        max_samples = None if args.max_samples in (None, 0) else args.max_samples
        artifacts = run_spectral_sen1floods11(
            SpectralBaselineConfig(
                data_root=args.data_root,
                output_dir=args.output_dir,
                index=args.index,
                threshold=args.threshold,
                threshold_policy=args.threshold_policy,
                split_protocol=args.split_protocol,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                eval_split=args.eval_split,
                seed=args.seed,
                max_samples=max_samples,
                run_bwer_v2=args.run_bwer_v2,
            )
        )
        print(f"Spectral Sen1Floods11 baseline complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "evaluate-bwer":
        artifacts = evaluate_bwer_from_file(
            audit_table=args.audit_table,
            dataset=args.dataset,
            model=args.model,
            task=args.task,
            output_dir=args.output_dir,
            slice_config=args.slice_config,
            slice_variable=args.slice_variable,
            balance_variable=args.balance_variable,
            tail_fraction=args.tail_fraction,
            bootstrap=args.bootstrap,
            cluster_key=args.cluster_key,
            seed=args.seed,
            weighting=args.weighting,
            missing_balance_policy=args.missing_balance_policy,
            score_column=args.score_column,
            risk_column=args.risk_column,
            selective_coverage=args.selective_coverage,
            audit_level=args.audit_level,
        )
        print(f"BWER audit complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-bwer-v2":
        artifacts = run_bwer_v2_posthoc(args.input_dir, args.output_dir, bootstrap=args.bootstrap, seed=args.seed)
        print(f"BWER-Audit v2 post-hoc analysis complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "run-audit":
        artifacts = run_audit_from_outputs(
            predictions=args.predictions,
            metadata=args.metadata,
            segmentation_metrics=args.segmentation_metrics,
            dataset=args.dataset,
            model=args.model,
            task=args.task,
            output_dir=args.output_dir,
            slice_config=args.slice_config,
            slice_variable=args.slice_variable,
            balance_variable=args.balance_variable,
            tail_fraction=args.tail_fraction,
            bootstrap=args.bootstrap,
            cluster_key=args.cluster_key,
            seed=args.seed,
            weighting=args.weighting,
            missing_balance_policy=args.missing_balance_policy,
            score_column=args.score_column,
            risk_column=args.risk_column,
            selective_coverage=args.selective_coverage,
            audit_level=args.audit_level,
        )
        print(f"BWER audit complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "preflight-bwer":
        artifacts = evaluate_slice_support_from_files(
            audit_table=args.audit_table,
            predictions=args.predictions,
            metadata=args.metadata,
            segmentation_metrics=args.segmentation_metrics,
            dataset=args.dataset,
            model=args.model,
            task=args.task,
            output_dir=args.output_dir,
            slice_config=args.slice_config,
            candidates=args.candidate,
            min_samples_per_slice=args.min_samples_per_slice,
            min_units_required=args.min_units_required,
            min_slices_required=args.min_slices_required,
            score_column=args.score_column,
            risk_column=args.risk_column,
        )
        print(f"BWER support preflight complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "geobwer-audit":
        protocol = _load_geobwer_protocol(args.protocol)
        rows = read_csv_rows(args.audit_table)
        groups = args.group_column or [protocol.group_variable]
        result = geobwer_audit_rows(
            rows,
            group_columns=groups,
            protocol=protocol,
            loss_column=args.loss_column or protocol.loss_name,
            unit_column=args.unit_column or protocol.independent_unit_column,
            cluster_column=args.cluster_column,
            balance_column=args.balance_column or protocol.balance_variable or None,
            formal=not args.diagnostic,
            require_probabilities=args.require_probabilities,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        artifacts = result.to_report(args.output_dir)
        print(f"GeoBWER audit complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "geobwer-compare":
        protocol = _load_geobwer_protocol(args.protocol)
        rows_a = read_csv_rows(args.model_a_table)
        rows_b = read_csv_rows(args.model_b_table)
        unit_column = args.unit_column or protocol.independent_unit_column
        group_column = args.group_column or protocol.group_variable
        loss_column = args.loss_column or protocol.loss_name
        cluster_column = args.cluster_column or (
            protocol.spatial_block_column if protocol.inference_method == "spatial_maxt" else protocol.cluster_column
        )
        index_a = {str(row.get(unit_column)): row for row in rows_a}
        index_b = {str(row.get(unit_column)): row for row in rows_b}
        if len(index_a) != len(rows_a) or len(index_b) != len(rows_b):
            raise ValueError("geobwer-compare requires one row per independent unit in each table.")
        common = tuple(sorted(set(index_a) & set(index_b)))
        if len(common) < 2:
            raise ValueError("Fewer than two common independent units.")
        for unit in common:
            if str(index_a[unit].get(group_column)) != str(index_b[unit].get(group_column)):
                raise ValueError(f"Group mismatch for paired unit={unit}.")
            if str(index_a[unit].get(cluster_column)) != str(index_b[unit].get(cluster_column)):
                raise ValueError(f"Cluster mismatch for paired unit={unit}.")
        result = geobwer_compare(
            loss_a=[float(index_a[unit][loss_column]) for unit in common],
            loss_b=[float(index_b[unit][loss_column]) for unit in common],
            groups=[index_a[unit][group_column] for unit in common],
            unit_id=common,
            cluster_id=[index_a[unit][cluster_column] for unit in common],
            protocol=protocol,
            model_a=args.model_a,
            model_b=args.model_b,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            spatial_localization_config=(
                SpatialConformalConfig()
                if args.spatial_localization_preflight
                else None
            ),
        )
        output = ensure_dir(args.output_dir)
        row = asdict(result)
        row["validity"] = result.validity.value
        row["common_groups"] = ";".join(result.common_groups)
        row["protocol_hash"] = protocol.signature
        write_csv(output / "geobwer_model_comparison.csv", [row])
        (output / "geobwer_model_comparison.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"GeoBWER paired comparison complete: {args.output_dir}")
    elif args.command == "geobwer-inventory":
        protocol = _load_geobwer_protocol(args.protocol)
        records = inventory_artifacts(args.input, protocol)
        artifacts = write_inventory_report(records, protocol, args.output_dir)
        print(f"GeoBWER artifact inventory complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "geobwer-multiclass-uncertainty":
        protocol = _load_geobwer_protocol(args.protocol)
        artifacts = run_multiclass_uncertainty_suite(
            args.calibration_probabilities,
            args.test_formal_dir,
            args.output_dir,
            protocol=protocol,
            group_columns=tuple(args.group_column),
            calibration_manifest=args.calibration_manifest,
            conformal_methods=tuple(args.conformal_method or ("lac", "aps", "raps")),
            selective_coverages=tuple(args.selective_coverage or (0.5, 0.7, 0.8, 0.9)),
            alpha=args.alpha,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        print(f"GeoBWER multiclass uncertainty audit complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "geobwer-multilabel-uncertainty":
        protocol = _load_geobwer_protocol(args.protocol)
        artifacts = run_multilabel_uncertainty_suite(
            args.calibration_probabilities,
            args.test_formal_dir,
            args.output_dir,
            protocol=protocol,
            group_columns=tuple(args.group_column),
            calibration_manifest=args.calibration_manifest,
            selective_coverages=tuple(args.selective_coverage or (0.5, 0.7, 0.8, 0.9)),
            crc_alpha=args.crc_alpha,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        print(f"GeoBWER multilabel uncertainty audit complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    elif args.command == "geobwer-spatial-conformal-upgrade":
        protocol = _load_geobwer_protocol(args.protocol)
        artifacts = run_multiclass_spatial_upgrade(
            args.calibration_probabilities,
            args.calibration_metadata,
            args.test_formal_dir,
            args.output_dir,
            protocol=protocol,
            group_columns=tuple(args.group_column),
            source_calibration_manifest=args.calibration_manifest,
            conformal_methods=tuple(args.conformal_method or ("lac", "aps", "raps")),
            alpha=args.alpha,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        print(f"GeoBWER spatial conformal upgrade complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
