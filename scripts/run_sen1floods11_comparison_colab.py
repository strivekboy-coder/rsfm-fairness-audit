from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def _mount_drive(no_mount_drive: bool) -> None:
    if no_mount_drive:
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        print("[info] google.colab is not available; assuming files are local.", flush=True)
        return
    drive.mount("/content/drive")


def _run(cmd: list[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def _extract_run(zip_path: Path, output_parent: Path, expected_dir_name: str, force: bool) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing run zip: {zip_path}")
    output_parent.mkdir(parents=True, exist_ok=True)
    target = output_parent / expected_dir_name
    if target.exists() and force:
        shutil.rmtree(target)
    if (target / "event_segmentation_metrics.csv").exists():
        print(f"[info] Reusing extracted run: {target}", flush=True)
        return target
    print(f"[stage] Extracting {zip_path} to {output_parent}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_parent)
    if not (target / "event_segmentation_metrics.csv").exists():
        raise RuntimeError(f"Expected extracted run directory with event metrics: {target}")
    return target


def _zip_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src.parent))
    print(f"[done] Comparison zip: {dst}", flush=True)


def _validate_comparison(output_dir: Path) -> None:
    required = [
        "comparison_summary.csv",
        "average_vs_bwer.csv",
        "event_level_comparison.csv",
        "comparison_report.md",
        "figures/average_iou_vs_raw_bwer.png",
        "figures/average_iou_vs_standardised_bwer.png",
    ]
    missing = [rel for rel in required if not (output_dir / rel).exists()]
    if missing:
        raise RuntimeError(f"Comparison output is incomplete; missing files: {missing}")
    print(f"[stage] Comparison validation passed: {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab helper for standalone Sen1Floods11 Prithvi-vs-U-Net comparison.")
    drive_root = Path("/content/drive/MyDrive/rsfm_fairness_audit")
    parser.add_argument("--prithvi-zip", type=Path, default=drive_root / "outputs" / "prithvi_tl_sen1floods11_official_full_512.zip")
    parser.add_argument("--unet-zip", type=Path, default=drive_root / "outputs" / "unet_sen1floods11_full_512.zip")
    parser.add_argument("--outputs-parent", type=Path, default=Path("/content/outputs"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("/content/outputs/comparisons/sen1floods11_prithvi_vs_unet_512"))
    parser.add_argument("--comparison-zip", type=Path, default=drive_root / "outputs" / "comparisons" / "sen1floods11_prithvi_vs_unet_512.zip")
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _mount_drive(args.no_mount_drive)
    prithvi_dir = _extract_run(args.prithvi_zip, args.outputs_parent, "prithvi_tl_sen1floods11_official_full_512", args.force)
    unet_dir = _extract_run(args.unet_zip, args.outputs_parent, "unet_sen1floods11_full_512", args.force)
    if args.comparison_dir.exists() and args.force:
        shutil.rmtree(args.comparison_dir)
    elif args.comparison_dir.exists():
        raise RuntimeError(f"Comparison directory already exists: {args.comparison_dir}. Pass --force to replace it.")
    _run(
        [
            sys.executable,
            "-m",
            "rsfm_fairness_audit.cli",
            "compare-runs",
            "--dataset",
            "sen1floods11",
            "--run",
            f"prithvi={prithvi_dir}",
            "--run",
            f"unet={unet_dir}",
            "--output-dir",
            str(args.comparison_dir),
        ]
    )
    _validate_comparison(args.comparison_dir)
    _zip_dir(args.comparison_dir, args.comparison_zip)


if __name__ == "__main__":
    main()
