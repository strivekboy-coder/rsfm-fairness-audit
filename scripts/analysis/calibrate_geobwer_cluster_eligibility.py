from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rsfm_fairness_audit.cluster_eligibility import calibrate_cluster_eligibility_rule


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only GeoBWER cluster eligibility calibration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", default="10,15,20,30,40,50,75")
    parser.add_argument("--group-count", type=int, default=8)
    parser.add_argument("--rows-per-cluster", type=int, default=3)
    parser.add_argument("--intracluster-correlation", type=float, default=0.5)
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage-tolerance", type=float, default=0.05)
    parser.add_argument("--false-positive-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration artifact: {output}")
    result = calibrate_cluster_eligibility_rule(
        candidate_min_clusters=tuple(int(value) for value in args.candidates.split(",")),
        group_count=args.group_count,
        rows_per_cluster=args.rows_per_cluster,
        intracluster_correlation=args.intracluster_correlation,
        n_simulations=args.simulations,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        coverage_tolerance=args.coverage_tolerance,
        false_positive_tolerance=args.false_positive_tolerance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(f"GEOBWER_CLUSTER_ELIGIBILITY_CALIBRATION={output}")
    print(f"selected_min_clusters_per_group={result.selected_min_clusters_per_group}")
    print(f"calibration_signature={result.signature}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
