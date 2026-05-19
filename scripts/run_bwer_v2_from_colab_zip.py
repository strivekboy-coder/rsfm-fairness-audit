from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path


DRIVE_ROOT = Path("/content/drive/MyDrive/rsfm_fairness_audit")
PROJECT_DIR = Path("/content/rsfm-fairness-audit")
INPUT_ZIP = DRIVE_ROOT / "outputs" / "prithvi_tl_sen1floods11_official_full_512.zip"
CONTENT_OUTPUTS = Path("/content/outputs")
RUN_DIR = CONTENT_OUTPUTS / "prithvi_tl_sen1floods11_official_full_512"
BWER_V2_DIR = RUN_DIR / "bwer_v2"
ENRICHED_ZIP = DRIVE_ROOT / "outputs" / "prithvi_tl_sen1floods11_official_full_512_with_bwer_v2.zip"


def _mount_drive_if_colab() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return
    drive.mount("/content/drive")


def _unzip_input() -> None:
    if not INPUT_ZIP.exists():
        raise FileNotFoundError(f"Missing input zip: {INPUT_ZIP}")
    CONTENT_OUTPUTS.mkdir(parents=True, exist_ok=True)
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    with zipfile.ZipFile(INPUT_ZIP) as zf:
        zf.extractall(CONTENT_OUTPUTS)
    if not RUN_DIR.exists():
        candidates = [path for path in CONTENT_OUTPUTS.iterdir() if path.is_dir() and path.name.startswith("prithvi_tl_sen1floods11_official_full_512")]
        if len(candidates) == 1:
            candidates[0].rename(RUN_DIR)
    if not RUN_DIR.exists():
        raise RuntimeError(f"Unzipped archive did not create expected run directory: {RUN_DIR}")


def _run_bwer_v2() -> None:
    cmd = [
        "python",
        "-m",
        "rsfm_fairness_audit.cli",
        "run-bwer-v2",
        "--input-dir",
        str(RUN_DIR),
        "--output-dir",
        str(BWER_V2_DIR),
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_DIR if PROJECT_DIR.exists() else None, check=True)


def _zip_enriched_run() -> None:
    ENRICHED_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if ENRICHED_ZIP.exists():
        ENRICHED_ZIP.unlink()
    with zipfile.ZipFile(ENRICHED_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in RUN_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(CONTENT_OUTPUTS))
    print(f"Enriched output zip written to: {ENRICHED_ZIP}")


def main() -> None:
    _mount_drive_if_colab()
    _unzip_input()
    _run_bwer_v2()
    _zip_enriched_run()


if __name__ == "__main__":
    main()
