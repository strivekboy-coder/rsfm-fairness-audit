from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rsfm_fairness_audit.alphaearth_existing_upgrade import (  # noqa: E402
    AlphaEarthExistingUpgradeConfig,
    run_alphaearth_existing_upgrade,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only GeoBWER 1.1 upgrade of frozen AlphaEarth formal "
            "calibration/test probabilities."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/geobwer/alphaearth.yaml",
    )
    parser.add_argument("--audit-bootstrap", type=int, default=2000)
    parser.add_argument("--conformal-alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_alphaearth_existing_upgrade(
        AlphaEarthExistingUpgradeConfig(
            source_root=args.source_root,
            output_dir=args.output_dir,
            persistent_output_dir=args.persistent_output_dir,
            protocol_path=args.protocol,
            audit_bootstrap=args.audit_bootstrap,
            conformal_alpha=args.conformal_alpha,
            seed=args.seed,
        )
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    print("ALPHAEARTH_EXISTING_GEOBWER_UPGRADE=PASS")


if __name__ == "__main__":
    main()
