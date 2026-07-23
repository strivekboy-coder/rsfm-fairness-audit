from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.drive_artifact_preflight import run_drive_preflight  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only inventory of existing Google Drive artifacts before GeoBWER formal jobs.")
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        help="Drive file/folder to inspect. Repeat as needed; defaults to the project output root and AlphaEarth source root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v2/00_drive_preflight"),
    )
    args = parser.parse_args()
    roots = args.root or [
        Path("/content/drive/MyDrive/rsfm_fairness_audit/outputs"),
        Path("/content/drive/MyDrive/rsfm_fairness_audit_alphaearth_full_v2_150k"),
    ]
    artifacts = run_drive_preflight(roots, args.output_dir)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
