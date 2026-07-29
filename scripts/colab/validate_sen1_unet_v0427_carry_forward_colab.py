from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rsfm_fairness_audit.sen1_amp_carry_forward import (  # noqa: E402
    build_carry_forward_manifest,
)


def _seeds(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only validation of completed v0.4.27 Sen1 U-Net seeds for "
            "explicit v0.4.28 carry-forward."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-run-log", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mode", default="S1")
    parser.add_argument("--seeds", type=_seeds, default=(42, 73))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = build_carry_forward_manifest(
        project_root=PROJECT_ROOT,
        source_root=args.source_root,
        source_run_log=args.source_run_log,
        output_path=args.output_manifest,
        mode=args.mode,
        seeds=args.seeds,
    )
    print("SEN1_V0427_CARRY_FORWARD=PASS")
    print(f"CARRY_FORWARD_MANIFEST={output}")


if __name__ == "__main__":
    main()
