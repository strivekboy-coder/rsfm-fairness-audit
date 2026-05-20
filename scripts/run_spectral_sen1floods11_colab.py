from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts.run_unet_sen1floods11_colab import _extract_prepared_zip, _mount_drive, _zip_dir


def _run(cmd: list[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True, env=env)


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
    ]
    missing = [rel for rel in required if not (output_dir / rel).exists()]
    if missing:
        raise RuntimeError(f"Spectral output is incomplete; missing files: {missing}")
    for rel in ["segmentation_metrics.csv", "event_segmentation_metrics.csv", "bwer_v2/bwer_v2_summary.csv"]:
        if _csv_row_count(output_dir / rel) <= 0:
            raise RuntimeError(f"Spectral output validation failed: {rel} has no data rows.")
    text = (output_dir / "bwer_v2" / "bwer_v2_summary.csv").read_text(encoding="utf-8")
    if "diagnostic_spectral_rule" not in text:
        raise RuntimeError("Spectral BWER v2 summary does not preserve diagnostic_spectral_rule metadata.")
    print(f"[stage] Spectral output validation passed: {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab helper for Sen1Floods11 spectral baseline evaluation.")
    drive_root = Path("/content/drive/MyDrive/rsfm_fairness_audit")
    parser.add_argument("--prepared-zip", type=Path, default=drive_root / "prepared_zips" / "sen1floods11_prithvi_tl_official_full_512.zip")
    parser.add_argument("--data-parent", type=Path, default=Path("/content/data"))
    parser.add_argument("--prepared-dir-name", default="sen1floods11_tl_official_full_512")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/outputs/spectral_mndwi_sen1floods11_full_512"))
    parser.add_argument("--output-zip", type=Path, default=drive_root / "outputs" / "spectral_mndwi_sen1floods11_full_512.zip")
    parser.add_argument("--index", choices=["ndwi", "mndwi", "nir_darkness"], default="mndwi")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--threshold-policy", choices=["fixed", "validation", "oracle_diagnostic"], default="fixed")
    parser.add_argument("--eval-split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-output-validation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _mount_drive(args.no_mount_drive)
    data_root = _extract_prepared_zip(args.prepared_zip, args.data_parent, args.prepared_dir_name)
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists():
        raise RuntimeError(f"Output directory already exists: {args.output_dir}. Pass --force to replace it.")
    cmd = [
        sys.executable,
        "-m",
        "rsfm_fairness_audit.cli",
        "run-spectral-sen1floods11",
        "--data-root",
        str(data_root),
        "--output-dir",
        str(args.output_dir),
        "--index",
        args.index,
        "--threshold",
        str(args.threshold),
        "--threshold-policy",
        args.threshold_policy,
        "--eval-split",
        args.eval_split,
        "--run-bwer-v2",
    ]
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    _run(cmd)
    if not args.skip_output_validation:
        _validate_output_dir(args.output_dir)
    _zip_dir(args.output_dir, args.output_zip)
    print(f"[done] Spectral output directory: {args.output_dir}", flush=True)
    print(f"[done] Spectral output zip: {args.output_zip}", flush=True)


if __name__ == "__main__":
    main()
