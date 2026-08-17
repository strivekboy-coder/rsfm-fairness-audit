from __future__ import annotations

import argparse
from pathlib import Path

from rsfm_fairness_audit.geographic_risk_association import build_geographic_risk_association


def _mapping(values: list[str], parser: argparse.ArgumentParser) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            parser.error("mapping arguments must be NAME=CSV")
        name, path = value.split("=", 1)
        output[name] = Path(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered CPU-only geographic risk/covariate associations.")
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphaearth-sample-csv", type=Path)
    parser.add_argument("--alphaearth-external-csv", type=Path)
    parser.add_argument("--fmow-sample-csv", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--fmow-external-csv", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--fmow-site-count", type=int, default=1480)
    args = parser.parse_args()
    result = build_geographic_risk_association(
        args.atlas_dir, args.output_dir,
        alphaearth_sample_csv=args.alphaearth_sample_csv,
        fmow_sample_csvs=_mapping(args.fmow_sample_csv, parser),
        alphaearth_external_csv=args.alphaearth_external_csv,
        fmow_external_csvs=_mapping(args.fmow_external_csv, parser),
        n_boot=args.n_boot,
        fmow_expected_site_count=args.fmow_site_count,
    )
    print(f"status={result['status']}")
    print(f"completed_associations={len(result['results'])}")


if __name__ == "__main__":
    main()
