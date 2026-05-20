from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.run_unet_sen1floods11_colab import _extract_prepared_zip, _mount_drive, _zip_dir


def _run(cmd: list[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def _events_from_metadata(data_root: Path) -> list[str]:
    with (data_root / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    events = sorted({str(row.get("event_id") or row.get("event") or row.get("region")) for row in rows if row})
    return [event for event in events if event and event != "None"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab helper for resumable Sen1Floods11 leave-one-event-out U-Net runs.")
    drive_root = Path("/content/drive/MyDrive/rsfm_fairness_audit")
    parser.add_argument("--prepared-zip", type=Path, default=drive_root / "prepared_zips" / "sen1floods11_prithvi_tl_official_full_512.zip")
    parser.add_argument("--data-parent", type=Path, default=Path("/content/data"))
    parser.add_argument("--prepared-dir-name", default="sen1floods11_tl_official_full_512")
    parser.add_argument("--architecture", choices=["vanilla_unet", "s2_resnet34_unet"], default="vanilla_unet")
    parser.add_argument("--output-root", type=Path, default=Path("/content/outputs/loeo/sen1floods11"))
    parser.add_argument("--aggregate-dir", type=Path, default=Path("/content/outputs/comparisons/sen1floods11_loeo"))
    parser.add_argument("--aggregate-zip", type=Path, default=drive_root / "outputs" / "comparisons" / "sen1floods11_loeo.zip")
    parser.add_argument("--held-out-event", action="append", default=[], help="Repeat to run a subset of events; default is all events.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0, help="Smoke only; full LOEO should leave this as 0.")
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _mount_drive(args.no_mount_drive)
    data_root = _extract_prepared_zip(args.prepared_zip, args.data_parent, args.prepared_dir_name)
    model_variant = "s2_resnet34_unet" if args.architecture == "s2_resnet34_unet" else "unet_sen1floods11_s2_512"
    model_root = args.output_root / model_variant
    events = args.held_out_event or _events_from_metadata(data_root)
    if not args.aggregate_only:
        for event in events:
            event_dir = model_root / event
            if (event_dir / "event_segmentation_metrics.csv").exists() and not args.force:
                print(f"[info] Reusing completed LOEO event: {event}", flush=True)
                continue
            if event_dir.exists() and args.force:
                shutil.rmtree(event_dir)
            cmd = [
                sys.executable,
                "-m",
                "rsfm_fairness_audit.cli",
                "run-unet-sen1floods11",
                "--data-root",
                str(data_root),
                "--output-dir",
                str(event_dir),
                "--architecture",
                args.architecture,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--split-protocol",
                "leave_one_event_out",
                "--held-out-event",
                event,
                "--eval-split",
                "test",
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--run-bwer-v2",
            ]
            if args.max_samples:
                cmd.extend(["--max-samples", str(args.max_samples)])
            _run(cmd)
    if args.aggregate_dir.exists() and args.force:
        shutil.rmtree(args.aggregate_dir)
    _run(
        [
            sys.executable,
            "-m",
            "rsfm_fairness_audit.cli",
            "aggregate-loeo",
            "--input-root",
            str(model_root),
            "--output-dir",
            str(args.aggregate_dir / model_variant),
        ]
    )
    _zip_dir(args.aggregate_dir, args.aggregate_zip)


if __name__ == "__main__":
    main()
