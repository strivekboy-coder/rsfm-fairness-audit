from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.fmow_superclass_feasibility import (  # noqa: E402
    DEFAULT_CONFIRMATORY_MIN_SITES,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_MIN_SITES,
    DEFAULT_TAXONOMY,
    scan_fmow_superclass_feasibility,
)
from rsfm_fairness_audit.io import read_csv_rows  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the outcome-independent fMoW superclass support "
            "feasibility scan."
        )
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--geography-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--min-sites", type=int, default=DEFAULT_MIN_SITES)
    parser.add_argument(
        "--confirmatory-min-sites",
        type=int,
        default=DEFAULT_CONFIRMATORY_MIN_SITES,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("[fmow:superclass-feasibility] model outputs used: false")
    print(f"[fmow:superclass-feasibility] metadata: {args.metadata_csv}")
    print(
        "[fmow:superclass-feasibility] geography contract: "
        f"{args.geography_contract}"
    )
    print(f"[fmow:superclass-feasibility] taxonomy: {args.taxonomy}")
    artifacts = scan_fmow_superclass_feasibility(
        args.metadata_csv,
        args.geography_contract,
        args.output_dir,
        taxonomy_path=args.taxonomy,
        split=args.split,
        min_samples=args.min_samples,
        min_sites=args.min_sites,
        confirmatory_min_sites=args.confirmatory_min_sites,
    )
    print("[fmow:superclass-feasibility] complete")
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    print("\n=== FEASIBILITY SUMMARY ===")
    for row in read_csv_rows(artifacts["summary"]):
        print(row)


if __name__ == "__main__":
    main()
