from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.fmow_formal_split import write_fmow_formal_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the frozen train/calibration/test fMoW manifest by category-scoped location."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-source-split", default="train")
    parser.add_argument("--holdout-source-split", default="val")
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    artifacts = write_fmow_formal_split(
        args.source_manifest,
        args.output_dir,
        train_source_split=args.train_source_split,
        holdout_source_split=args.holdout_source_split,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
