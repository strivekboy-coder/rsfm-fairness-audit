from __future__ import annotations

"""One-shot Colab entrypoint for cached CROMA paired S2->S1 sensitivity."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/rsfm_fairness_audit")
DRIVE_PANEL_ROOT = DRIVE_PROJECT_ROOT / "outputs/geobwer_final_v3/reben_full_panel/croma"
LOCAL_CACHE_ROOT = Path("/content/reben_croma_paired_shift_cache")
LOCAL_OUTPUT = Path("/content/reben_croma_paired_shift_v1")
DRIVE_OUTPUT = DRIVE_PROJECT_ROOT / "outputs/reben_croma_paired_shift_v1"
REQUIRED_FILES = ("embeddings.npy", "labels.npy", "metadata.jsonl", "embedding_cache_manifest.json")


def _mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:  # pragma: no cover - Colab-only guard
        raise RuntimeError("This staging entrypoint must run inside Google Colab.") from exc
    drive.mount("/content/drive", force_remount=False)


def _stage_cache(sensor: str) -> Path:
    source = DRIVE_PANEL_ROOT / sensor / "shared_embedding_cache"
    target = LOCAL_CACHE_ROOT / sensor
    if not source.is_dir():
        raise FileNotFoundError(f"Missing formal CROMA cache root: {source}")
    for split in ("train", "val", "test"):
        for filename in REQUIRED_FILES:
            src = source / split / filename
            dst = target / split / filename
            if not src.is_file():
                raise FileNotFoundError(f"Missing required cached artifact: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                print(f"[croma-paired:stage] reuse {dst} ({dst.stat().st_size:,} bytes)")
                continue
            print(f"[croma-paired:stage] copy {src} -> {dst} ({src.stat().st_size:,} bytes)")
            shutil.copy2(src, dst)
    return target


def _publish() -> None:
    DRIVE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[croma-paired:publish] syncing {LOCAL_OUTPUT} -> {DRIVE_OUTPUT}")
    shutil.copytree(LOCAL_OUTPUT, DRIVE_OUTPUT, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    _mount_drive()
    s2 = _stage_cache("s2")
    s1 = _stage_cache("s1")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_reben_croma_paired_shift_colab.py"),
        "--s2-cache-root", str(s2),
        "--s1-cache-root", str(s1),
        "--output-dir", str(LOCAL_OUTPUT),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    print("[croma-paired:run] " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    _publish()
    print(f"[croma-paired:complete] {DRIVE_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
