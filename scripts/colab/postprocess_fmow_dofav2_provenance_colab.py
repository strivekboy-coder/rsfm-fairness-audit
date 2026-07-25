from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.fmow_dofav2_postprocess import (  # noqa: E402
    postprocess_fmow_dofav2_provenance,
)


def _seeds(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected comma-separated integer seeds.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive geography-provenance overlay for an existing DOFAv2 fMoW run."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--geography-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = postprocess_fmow_dofav2_provenance(
        args.source_root,
        args.geography_contract,
        args.output_dir,
        seeds=args.seeds,
    )
    print("[fmow:dofav2-provenance] complete")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
