from __future__ import annotations

import argparse
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
from rsfm_fairness_audit.loeo import aggregate_loeo_runs
from rsfm_fairness_audit.pipeline import (
    build_real_adapters,
    compare_model_runs,
    compare_sensor_mode_runs,
    run_dummy_pipeline,
    run_real_pipeline,
)
from rsfm_fairness_audit.preflight import checks_to_json, run_real_preflight
from rsfm_fairness_audit.segmentation import run_segmentation_smoke
from rsfm_fairness_audit.spectral_baseline import SpectralBaselineConfig, run_spectral_sen1floods11
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
    fmow_cls.add_argument("--model", choices=["supervised_stats", "dofa", "resnet50"], default="supervised_stats")
    fmow_cls.add_argument("--model-config", type=Path, default=Path("configs/models/dofa_fmow_sentinel.yaml"))
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
                model_config=args.model_config,
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


if __name__ == "__main__":
    main()
