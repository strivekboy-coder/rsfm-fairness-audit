from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


DRIVE_ROOT = Path("/content/drive/MyDrive/rsfm_fairness_audit")
PROJECT_DIR = Path("/content/rsfm-fairness-audit")
DEFAULT_RUN_NAME = "prithvi_tl_sen1floods11_official_full_512"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unzip a completed audit run, add BWER v2 outputs, and write one canonical final result zip.")
    parser.add_argument(
        "--input-zip",
        type=Path,
        default=DRIVE_ROOT / "outputs" / f"{DEFAULT_RUN_NAME}.zip",
        help="Existing completed audit output zip. This can also be a manually uploaded /content/*.zip file.",
    )
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR, help="Cloned repository directory.")
    parser.add_argument("--content-outputs", type=Path, default=Path("/content/outputs"))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Expected unzipped run directory name.")
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("/content") / f"{DEFAULT_RUN_NAME}.zip",
        help="Canonical final result zip containing the original output files plus bwer_v2/. Defaults to /content for reliable manual download/upload.",
    )
    parser.add_argument("--no-mount-drive", action="store_true", help="Skip Drive mounting; useful for manual uploads/downloads.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser


def _mount_drive_if_colab() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return
    drive.mount("/content/drive")


def _unzip_input(input_zip: Path, content_outputs: Path, run_dir: Path, run_name: str) -> None:
    if not input_zip.exists():
        raise FileNotFoundError(f"Missing input zip: {input_zip}")
    content_outputs.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    with zipfile.ZipFile(input_zip) as zf:
        zf.extractall(content_outputs)
    if not run_dir.exists():
        candidates = [path for path in content_outputs.iterdir() if path.is_dir() and path.name.startswith(run_name)]
        if len(candidates) == 1:
            candidates[0].rename(run_dir)
    if not run_dir.exists():
        raise RuntimeError(f"Unzipped archive did not create expected run directory: {run_dir}")


def _run_bwer_v2(project_dir: Path, run_dir: Path, bwer_v2_dir: Path, bootstrap: int) -> None:
    cmd = [
        "python",
        "-m",
        "rsfm_fairness_audit.cli",
        "run-bwer-v2",
        "--input-dir",
        str(run_dir),
        "--output-dir",
        str(bwer_v2_dir),
        "--bootstrap",
        str(bootstrap),
    ]
    env = os.environ.copy()
    src = project_dir / "src"
    if src.exists():
        env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=project_dir if project_dir.exists() else None, env=env, check=True)


def _zip_enriched_run(run_dir: Path, content_outputs: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in run_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(content_outputs))
    print(f"Enriched output zip written to: {output_zip}")


def main() -> None:
    args = _parser().parse_args()
    if not args.no_mount_drive:
        _mount_drive_if_colab()
    run_dir = args.content_outputs / args.run_name
    bwer_v2_dir = run_dir / "bwer_v2"
    _unzip_input(args.input_zip, args.content_outputs, run_dir, args.run_name)
    _run_bwer_v2(args.project_dir, run_dir, bwer_v2_dir, args.bootstrap)
    _zip_enriched_run(run_dir, args.content_outputs, args.output_zip)


if __name__ == "__main__":
    main()
