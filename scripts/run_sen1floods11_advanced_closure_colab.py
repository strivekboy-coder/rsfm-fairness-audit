from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab helper for Sen1Floods11 advanced closure post-hoc checks.")
    drive_root = Path("/content/drive/MyDrive/rsfm_fairness_audit")
    parser.add_argument("--prithvi-zip", type=Path, default=drive_root / "outputs" / "prithvi_tl_sen1floods11_official_full_512.zip")
    parser.add_argument("--unet-zip", type=Path, default=drive_root / "outputs" / "unet_sen1floods11_full_512.zip")
    parser.add_argument("--spectral-zip", type=Path, default=drive_root / "outputs" / "spectral_mndwi_sen1floods11_full_512.zip")
    parser.add_argument("--resnet34-unet-zip", type=Path, default=drive_root / "outputs" / "s2_resnet34_unet_sen1floods11_full_512.zip")
    parser.add_argument("--outputs-parent", type=Path, default=Path("/content/outputs"))
    parser.add_argument("--protocol-matched-dir", type=Path, default=Path("/content/outputs/comparisons/sen1floods11_protocol_matched"))
    parser.add_argument("--protocol-matched-zip", type=Path, default=drive_root / "outputs" / "comparisons" / "sen1floods11_protocol_matched.zip")
    parser.add_argument("--selective-risk-dir", type=Path, default=Path("/content/outputs/comparisons/sen1floods11_selective_risk"))
    parser.add_argument("--selective-risk-zip", type=Path, default=drive_root / "outputs" / "comparisons" / "sen1floods11_selective_risk.zip")
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _mount_drive(args.no_mount_drive)
    runs = {
        "prithvi_tl": _extract_run(args.prithvi_zip, args.outputs_parent, "prithvi_tl_sen1floods11_official_full_512", args.force),
        "vanilla_unet": _extract_run(args.unet_zip, args.outputs_parent, "unet_sen1floods11_full_512", args.force),
        "spectral_mndwi": _extract_run(args.spectral_zip, args.outputs_parent, "spectral_mndwi_sen1floods11_full_512", args.force),
        "s2_resnet34_unet": _extract_run(args.resnet34_unet_zip, args.outputs_parent, "s2_resnet34_unet_sen1floods11_full_512", args.force),
    }
    for directory in [args.protocol_matched_dir, args.selective_risk_dir]:
        if directory.exists() and args.force:
            shutil.rmtree(directory)
        elif directory.exists():
            raise RuntimeError(f"Output directory already exists: {directory}. Pass --force to replace it.")
    run_args = []
    for name, path in runs.items():
        run_args.extend(["--run", f"{name}={path}"])
    _run([sys.executable, "-m", "rsfm_fairness_audit.cli", "protocol-match-runs", *run_args, "--output-dir", str(args.protocol_matched_dir)])
    _run([sys.executable, "-m", "rsfm_fairness_audit.cli", "run-selective-risk", *run_args, "--output-dir", str(args.selective_risk_dir)])
    _zip_dir(args.protocol_matched_dir, args.protocol_matched_zip)
    _zip_dir(args.selective_risk_dir, args.selective_risk_zip)


if __name__ == "__main__":
    main()
