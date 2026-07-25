from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.fmow_superclass_postprocess import (  # noqa: E402
    run_fmow_superclass_postprocess,
)


def _seeds(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated integer seeds."
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen fMoW superclass analysis after all matching "
            "DOFAv2 and ResNet-50 seeds are complete."
        )
    )
    parser.add_argument("--axis-role-freeze", type=Path, required=True)
    parser.add_argument("--feasibility-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--geography-contract", type=Path, required=True)
    parser.add_argument("--dofa-source-root", type=Path, required=True)
    parser.add_argument(
        "--dofa-provenance-overlay", type=Path, required=True
    )
    parser.add_argument("--resnet-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/geobwer/fmow_sentinel.yaml",
    )
    parser.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_fmow_superclass_postprocess(
        axis_role_freeze=args.axis_role_freeze,
        feasibility_dir=args.feasibility_dir,
        taxonomy_path=args.taxonomy,
        geography_contract=args.geography_contract,
        dofa_source_root=args.dofa_source_root,
        dofa_provenance_overlay=args.dofa_provenance_overlay,
        resnet_source_root=args.resnet_source_root,
        output_dir=args.output_dir,
        persistent_output_dir=args.persistent_output_dir,
        geobwer_protocol=args.protocol,
        seeds=args.seeds,
    )
    print("[fmow:superclass-frozen] complete")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
