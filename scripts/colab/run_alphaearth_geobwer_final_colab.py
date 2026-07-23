from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.alphaearth_geobwer_campaign import (  # noqa: E402
    AlphaEarthCampaignConfig,
    run_alphaearth_geobwer_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upgrade existing AlphaEarth full probabilities to the final GeoBWER contract without GEE/model reruns."
    )
    parser.add_argument("--all-split-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--persistent-output-dir",
        type=Path,
        help="Drive mirror for completed post-processing; keep --output-dir on local /content.",
    )
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/geobwer/alphaearth.yaml")
    parser.add_argument("--calibration-simulations", type=int, default=200)
    parser.add_argument("--calibration-bootstrap", type=int, default=500)
    parser.add_argument("--audit-bootstrap", type=int, default=2000)
    parser.add_argument("--minimum-moderate-tail-power", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_alphaearth_geobwer_campaign(
        AlphaEarthCampaignConfig(
            all_split_predictions=args.all_split_predictions,
            output_dir=args.output_dir,
            persistent_output_dir=args.persistent_output_dir,
            protocol_path=args.protocol,
            calibration_simulations=args.calibration_simulations,
            calibration_bootstrap=args.calibration_bootstrap,
            audit_bootstrap=args.audit_bootstrap,
            minimum_moderate_tail_power=args.minimum_moderate_tail_power,
            seed=args.seed,
        )
    )
    print(f"[alphaearth:geobwer] complete: {args.output_dir}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
