from __future__ import annotations

import argparse
from pathlib import Path

from rsfm_fairness_audit.geographic_risk_covariates import prepare_geographic_risk_covariates


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed official geographic covariates from Google Earth Engine.")
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--alphaearth-sample-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ee-project", required=True)
    parser.add_argument("--batch-size", type=int, default=400)
    args = parser.parse_args()

    import ee  # type: ignore[import-not-found]

    try:
        ee.Initialize(project=args.ee_project)
    except Exception:
        print("[covariates] Earth Engine authentication required", flush=True)
        ee.Authenticate()
        ee.Initialize(project=args.ee_project)
    print(f"[covariates] Earth Engine project: {args.ee_project}", flush=True)
    result = prepare_geographic_risk_covariates(
        atlas_dir=args.atlas_dir, alphaearth_sample_csv=args.alphaearth_sample_csv,
        output_dir=args.output_dir, cache_dir=args.cache_dir,
        batch_size=args.batch_size, ee_module=ee,
    )
    print(f"status={result['status']}")
    print(f"manifest={args.output_dir / 'geographic_covariate_manifest.json'}")


if __name__ == "__main__":
    main()
