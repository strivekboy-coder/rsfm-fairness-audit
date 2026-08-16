from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.optimization_phase1 import run_optimization_1_7  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPU post-processing for approved optimization items 1-7.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optimization_1_7_v1"))
    parser.add_argument("--snapshot-v050", type=Path, default=Path("work/drive_snapshot_v050"))
    parser.add_argument("--reben-npz", type=Path, default=Path("work/drive_snapshot_v060/reben_probability_npz"))
    args = parser.parse_args()
    artifacts = run_optimization_1_7(
        PROJECT_ROOT, args.output_dir,
        snapshot_v050=args.snapshot_v050, reben_npz=args.reben_npz,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
