from __future__ import annotations

import argparse
from pathlib import Path

from rsfm_fairness_audit.pipeline import run_dummy_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsfm-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dummy = subparsers.add_parser("run-dummy", help="Run the CPU-only synthetic fairness audit.")
    dummy.add_argument("--output-dir", type=Path, default=Path("outputs/dummy_smoke"))
    dummy.add_argument("--num-samples", type=int, default=240)
    dummy.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run-dummy":
        artifacts = run_dummy_pipeline(args.output_dir, num_samples=args.num_samples, seed=args.seed)
        print(f"Dummy fairness audit complete: {args.output_dir}")
        for name, path in artifacts.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
