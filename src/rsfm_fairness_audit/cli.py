from __future__ import annotations

import argparse
from pathlib import Path

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_from_file, run_audit_from_outputs
from rsfm_fairness_audit.bwer_v2 import run_bwer_v2_posthoc
from rsfm_fairness_audit.pipeline import (
    build_real_adapters,
    compare_model_runs,
    compare_sensor_mode_runs,
    run_dummy_pipeline,
    run_real_pipeline,
)
from rsfm_fairness_audit.preflight import checks_to_json, run_real_preflight
from rsfm_fairness_audit.segmentation import run_segmentation_smoke
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

    sensor_compare = subparsers.add_parser("compare-sensor-modes", help="Compare CROMA SAR/optical/both completed runs.")
    sensor_compare.add_argument("--dataset", default="ben_ge")
    sensor_compare.add_argument(
        "--run",
        action="append",
        required=True,
        help="Sensor-mode/run pair in the form sar=path, optical=path, or both=path.",
    )
    sensor_compare.add_argument("--output-dir", type=Path, default=Path("outputs/comparisons/croma_sensor_modes"))

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
    unet.add_argument("--early-stopping-patience", type=int, default=10)
    unet.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    unet.add_argument("--split-protocol", choices=["random_chip_split", "event_held_out"], default="random_chip_split")
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
        artifacts = compare_model_runs(runs, args.output_dir, dataset_name=args.dataset)
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
