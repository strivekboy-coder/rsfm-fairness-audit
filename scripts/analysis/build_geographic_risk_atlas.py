from __future__ import annotations

import argparse
from pathlib import Path

from rsfm_fairness_audit.geographic_risk_atlas import build_geographic_risk_atlas


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CPU-only three-task geographic risk atlas from frozen audit tables.")
    alpha = parser.add_mutually_exclusive_group()
    alpha.add_argument("--alphaearth-csv", type=Path, help="Explicit sample-level CSV with coordinates, spatial_block_id, and risk.")
    alpha.add_argument("--alphaearth-root", type=Path, help="Canonical AlphaEarth result root; discovers formal_outputs/formal_audit_table.csv and validates its contract.")
    parser.add_argument("--fmow-csv", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--fmow-seed-count", action="append", default=[], metavar="NAME=N")
    parser.add_argument("--fmow-site-count", type=int, default=1480)
    parser.add_argument("--reben-paired-dir", type=Path)
    parser.add_argument("--reben-model-paired-dir", action="append", default=[], metavar="MODEL=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fmow: dict[str, list[Path]] = {}
    for value in args.fmow_csv:
        if "=" not in value:
            parser.error("--fmow-csv must be NAME=CSV")
        name, path = value.split("=", 1); fmow.setdefault(name, []).append(Path(path))
    expected = {}
    for value in args.fmow_seed_count:
        if "=" not in value:
            parser.error("--fmow-seed-count must be NAME=N")
        name, count = value.split("=", 1)
        try:
            expected[name] = int(count)
        except ValueError:
            parser.error("--fmow-seed-count N must be an integer")
    reben_models = {}
    for value in args.reben_model_paired_dir:
        if "=" not in value:
            parser.error("--reben-model-paired-dir must be MODEL=DIR")
        name, path = value.split("=", 1)
        reben_models[name] = Path(path)
    result = build_geographic_risk_atlas(
        args.output_dir, alphaearth_csv=args.alphaearth_csv, alphaearth_root=args.alphaearth_root,
        fmow_csvs=fmow, fmow_expected_seed_counts=expected,
        reben_paired_dir=args.reben_paired_dir,
        reben_model_paired_dirs=reben_models,
        fmow_expected_site_count=args.fmow_site_count,
    )
    print(f"status={result['status']}")
    print(f"asset_count={len(result['readiness'])}")


if __name__ == "__main__":
    main()
