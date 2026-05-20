from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts.run_sen1floods11_comparison_colab import _extract_run, _mount_drive, _zip_dir


def _run(cmd: list[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _validate_closure(output_dir: Path) -> None:
    required = [
        "closure_comparison_summary.csv",
        "closure_average_vs_bwer.csv",
        "closure_event_level_comparison.csv",
        "closure_tail_event_overlap.csv",
        "closure_report.md",
        "comparison_summary.csv",
        "figures/average_iou_vs_raw_bwer.png",
        "figures/average_iou_vs_standardised_bwer.png",
    ]
    missing = [rel for rel in required if not (output_dir / rel).exists()]
    if missing:
        raise RuntimeError(f"Closure output is incomplete; missing files: {missing}")
    if _row_count(output_dir / "closure_comparison_summary.csv") < 4:
        raise RuntimeError("Closure comparison should contain four runs: Prithvi, vanilla U-Net, spectral, and S2 ResNet34-U-Net.")
    print(f"[stage] Closure validation passed: {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab helper for the Sen1Floods11 Closure Core Package comparison.")
    drive_root = Path("/content/drive/MyDrive/rsfm_fairness_audit")
    parser.add_argument("--prithvi-zip", type=Path, default=drive_root / "outputs" / "prithvi_tl_sen1floods11_official_full_512.zip")
    parser.add_argument("--unet-zip", type=Path, default=drive_root / "outputs" / "unet_sen1floods11_full_512.zip")
    parser.add_argument("--spectral-zip", type=Path, default=drive_root / "outputs" / "spectral_mndwi_sen1floods11_full_512.zip")
    parser.add_argument("--resnet34-unet-zip", type=Path, default=drive_root / "outputs" / "s2_resnet34_unet_sen1floods11_full_512.zip")
    parser.add_argument("--outputs-parent", type=Path, default=Path("/content/outputs"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("/content/outputs/comparisons/sen1floods11_closure"))
    parser.add_argument("--comparison-zip", type=Path, default=drive_root / "outputs" / "comparisons" / "sen1floods11_closure.zip")
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _mount_drive(args.no_mount_drive)
    prithvi = _extract_run(args.prithvi_zip, args.outputs_parent, "prithvi_tl_sen1floods11_official_full_512", args.force)
    unet = _extract_run(args.unet_zip, args.outputs_parent, "unet_sen1floods11_full_512", args.force)
    spectral = _extract_run(args.spectral_zip, args.outputs_parent, "spectral_mndwi_sen1floods11_full_512", args.force)
    resnet34 = _extract_run(args.resnet34_unet_zip, args.outputs_parent, "s2_resnet34_unet_sen1floods11_full_512", args.force)
    if args.comparison_dir.exists() and args.force:
        shutil.rmtree(args.comparison_dir)
    elif args.comparison_dir.exists():
        raise RuntimeError(f"Closure directory already exists: {args.comparison_dir}. Pass --force to replace it.")
    _run(
        [
            sys.executable,
            "-m",
            "rsfm_fairness_audit.cli",
            "compare-runs",
            "--dataset",
            "sen1floods11",
            "--run",
            f"prithvi_tl={prithvi}",
            "--run",
            f"vanilla_unet={unet}",
            "--run",
            f"spectral_mndwi={spectral}",
            "--run",
            f"s2_resnet34_unet={resnet34}",
            "--output-dir",
            str(args.comparison_dir),
            "--closure",
        ]
    )
    _validate_closure(args.comparison_dir)
    _zip_dir(args.comparison_dir, args.comparison_zip)


if __name__ == "__main__":
    main()
