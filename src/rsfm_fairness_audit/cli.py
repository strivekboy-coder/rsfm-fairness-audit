from __future__ import annotations

import argparse
from pathlib import Path

from rsfm_fairness_audit.pipeline import build_real_adapters, run_dummy_pipeline, run_real_pipeline
from rsfm_fairness_audit.preflight import checks_to_json, run_real_preflight


def _parse_wavelengths(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsfm-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dummy = subparsers.add_parser("run-dummy", help="Run the CPU-only synthetic fairness audit.")
    dummy.add_argument("--output-dir", type=Path, default=Path("outputs/dummy_smoke"))
    dummy.add_argument("--num-samples", type=int, default=240)
    dummy.add_argument("--seed", type=int, default=7)

    real = subparsers.add_parser("run-real", help="Run a subset-first real dataset/model smoke audit.")
    real.add_argument("--dataset", choices=["bigearthnet"], required=True)
    real.add_argument("--model", choices=["dofa"], required=True)
    real.add_argument("--data-root", type=Path, required=True)
    real.add_argument("--metadata-path", type=Path)
    real.add_argument("--subset-size", type=int)
    real.add_argument("--subset-manifest-path", type=Path)
    real.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    real.add_argument("--sensor-mode", choices=["S1", "S2", "S1+S2"], default="S2")
    real.add_argument("--output-dir", type=Path, default=Path("outputs/runs/dofa_bigearthnet_subset"))
    real.add_argument("--model-config", type=Path, help="YAML config for the real model adapter.")
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
    check.add_argument("--dataset", choices=["bigearthnet"], required=True)
    check.add_argument("--model", choices=["dofa"], required=True)
    check.add_argument("--model-config", type=Path, required=True)
    check.add_argument("--data-root", type=Path, required=True)
    check.add_argument("--metadata-path", type=Path)
    check.add_argument("--subset-manifest-path", type=Path)
    check.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    check.add_argument("--sensor-mode", choices=["S1", "S2", "S1+S2"], default="S2")
    check.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
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
        artifacts = run_real_pipeline(dataset, model, args.output_dir, args.dataset, args.model)
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


if __name__ == "__main__":
    main()
