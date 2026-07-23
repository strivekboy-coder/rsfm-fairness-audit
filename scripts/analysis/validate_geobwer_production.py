from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.geobwer_validation import run_validation_gate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the production GeoBWER implementation before formal model runs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    artifacts = run_validation_gate(
        args.output_dir,
        repetitions=args.repetitions,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(f"[geobwer:validation] all gates passed: {args.output_dir}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
