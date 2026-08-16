from __future__ import annotations

import argparse
from pathlib import Path

from rsfm_fairness_audit.geographic_risk_atlas import build_geographic_risk_atlas


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CPU-only three-task geographic risk atlas from frozen audit tables.")
    parser.add_argument("--alphaearth-csv", type=Path)
    parser.add_argument("--fmow-csv", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--reben-paired-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fmow = {}
    for value in args.fmow_csv:
        if "=" not in value:
            parser.error("--fmow-csv must be NAME=CSV")
        name, path = value.split("=", 1); fmow[name] = Path(path)
    result = build_geographic_risk_atlas(
        args.output_dir, alphaearth_csv=args.alphaearth_csv,
        fmow_csvs=fmow, reben_paired_dir=args.reben_paired_dir,
    )
    print(f"status={result['status']}")
    print(f"asset_count={len(result['readiness'])}")


if __name__ == "__main__":
    main()
