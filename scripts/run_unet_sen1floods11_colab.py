from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=cwd or repo_root, check=True, env=env)


def _mount_drive(no_mount_drive: bool) -> None:
    if no_mount_drive:
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        print("[info] google.colab is not available; assuming files are local.", flush=True)
        return
    drive.mount("/content/drive")


def _extract_prepared_zip(prepared_zip: Path, data_parent: Path, expected_dir: str) -> Path:
    if not prepared_zip.exists():
        raise FileNotFoundError(f"Prepared data zip not found: {prepared_zip}")
    data_parent.mkdir(parents=True, exist_ok=True)
    target = data_parent / expected_dir
    if (target / "metadata.csv").exists():
        print(f"[info] Prepared data already extracted: {target}", flush=True)
        return target
    if target.exists():
        shutil.rmtree(target)
    print(f"[stage] Extracting {prepared_zip} to {data_parent}", flush=True)
    with zipfile.ZipFile(prepared_zip) as zf:
        zf.extractall(data_parent)
    if (target / "metadata.csv").exists():
        return target
    matches = [path for path in data_parent.iterdir() if path.is_dir() and (path / "metadata.csv").exists()]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"Could not locate extracted Sen1Floods11 prepared directory under {data_parent}")


def _zip_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    print(f"[stage] Writing output zip: {dst}", flush=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src.parent))


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _validate_output_dir(output_dir: Path) -> None:
    required = [
        "segmentation_metrics.csv",
        "event_segmentation_metrics.csv",
        "audit_table.csv",
        "bwer_summary.csv",
        "bwer_by_slice.csv",
        "support_diagnostics.csv",
        "warnings.json",
        "report.md",
        "model_debug.json",
        "run_metadata.json",
        "bwer_v2/bwer_v2_summary.csv",
        "bwer_v2/standardised_bwer.csv",
        "bwer_v2/event_failure_analysis.csv",
        "bwer_v2/bwer_audit_report.md",
    ]
    missing = [rel for rel in required if not (output_dir / rel).exists()]
    if missing:
        raise RuntimeError(f"U-Net output is incomplete; missing files: {missing}")
    for rel in ["segmentation_metrics.csv", "event_segmentation_metrics.csv", "bwer_summary.csv", "bwer_v2/bwer_v2_summary.csv"]:
        count = _csv_row_count(output_dir / rel)
        if count <= 0:
            raise RuntimeError(f"U-Net output validation failed: {rel} has no data rows.")
    summary_text = (output_dir / "bwer_v2" / "bwer_v2_summary.csv").read_text(encoding="utf-8")
    if "supervised_baseline" not in summary_text or "unet" not in summary_text:
        raise RuntimeError("U-Net BWER v2 summary does not preserve supervised_baseline/unet metadata.")
    print(f"[stage] Output validation passed: {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab helper for the supervised U-Net Sen1Floods11 baseline.")
    parser.add_argument(
        "--prepared-zip",
        type=Path,
        default=Path("/content/drive/MyDrive/rsfm_fairness_audit/prepared_zips/sen1floods11_prithvi_tl_official_full_512.zip"),
        help="Existing prepared Sen1Floods11 512 zip. This script reads it but never modifies it.",
    )
    parser.add_argument("--data-parent", type=Path, default=Path("/content/data"))
    parser.add_argument("--prepared-dir-name", default="sen1floods11_tl_official_full_512")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/outputs/unet_sen1floods11_full_512"))
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("/content/drive/MyDrive/rsfm_fairness_audit/outputs/unet_sen1floods11_full_512.zip"),
        help="Final enriched output zip containing the original U-Net outputs plus bwer_v2/.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--architecture", choices=["vanilla_unet", "s2_resnet34_unet"], default="vanilla_unet")
    parser.add_argument("--pretrained-encoder", action="store_true")
    parser.add_argument("--split-protocol", choices=["random_chip_split", "event_held_out", "leave_one_event_out"], default="random_chip_split")
    parser.add_argument("--held-out-event", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0, help="Use >0 for a smoke run; 0 means full extracted dataset.")
    parser.add_argument("--eval-split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--force", action="store_true", help="Remove an existing output directory before running.")
    parser.add_argument("--skip-output-validation", action="store_true", help="Skip required-file checks before writing the final zip.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _mount_drive(args.no_mount_drive)
    data_root = _extract_prepared_zip(args.prepared_zip, args.data_parent, args.prepared_dir_name)
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists():
        raise RuntimeError(f"Output directory already exists: {args.output_dir}. Pass --force to remove it before running.")
    cmd = [
        sys.executable,
        "-m",
        "rsfm_fairness_audit.cli",
        "run-unet-sen1floods11",
        "--data-root",
        str(data_root),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--architecture",
        args.architecture,
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--split-protocol",
        args.split_protocol,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--eval-split",
        args.eval_split,
        "--run-bwer-v2",
    ]
    if args.pretrained_encoder:
        cmd.append("--pretrained-encoder")
    for event in args.held_out_event:
        cmd.extend(["--held-out-event", event])
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    _run(cmd)
    if not args.skip_output_validation:
        _validate_output_dir(args.output_dir)
    _zip_dir(args.output_dir, args.output_zip)
    print(f"[done] U-Net output directory: {args.output_dir}", flush=True)
    print(f"[done] U-Net output zip: {args.output_zip}", flush=True)


if __name__ == "__main__":
    main()
