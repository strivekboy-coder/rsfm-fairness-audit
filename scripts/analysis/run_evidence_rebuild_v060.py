from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsfm_fairness_audit.evidence_rebuild_v060 import (
    run_alphaearth_validation_spatial_recalibration,
    run_fmow_same_seed_paired_v12,
    run_reben_fixed_universe_v12,
    run_sen1_event_geobwer,
    run_cluster_uncertainty_v060,
    seal_evidence_output,
    run_fmow_proper_score_sensitivity,
    build_evidence_status_matrix,
    run_reben_labelwise_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only GeoBWER v0.6 evidence rebuild")
    sub = parser.add_subparsers(dest="stage", required=True)
    fmow = sub.add_parser("fmow-paired")
    for seed in (42, 73, 101):
        fmow.add_argument(f"--dofa-{seed}", type=Path, required=True)
        fmow.add_argument(f"--resnet-{seed}", type=Path, required=True)
    fmow.add_argument("--output-dir", type=Path, required=True)
    fmow.add_argument("--bootstrap", type=int, default=2000)
    fmow.add_argument("--min-clusters", type=int, default=75)

    reben = sub.add_parser("reben-partial")
    reben.add_argument("--unified-metrics", type=Path, required=True)
    reben.add_argument("--support-universe", type=Path, required=True)
    reben.add_argument("--output-dir", type=Path, required=True)
    reben.add_argument("--min-clusters", type=int, default=75)

    sen1 = sub.add_parser("sen1-event")
    sen1.add_argument("--event-metrics", type=Path, required=True)
    sen1.add_argument("--output-dir", type=Path, required=True)

    alpha = sub.add_parser("alpha-spatial")
    alpha.add_argument("--calibration-bundle-b64", type=Path, required=True)
    alpha.add_argument("--test-bundle-b64", type=Path, required=True)
    alpha.add_argument("--eval-metadata", type=Path, required=True)
    alpha.add_argument("--protocol", type=Path, default=Path("configs/geobwer/alphaearth.yaml"))
    alpha.add_argument("--output-dir", type=Path, required=True)
    alpha.add_argument("--simulations", type=int, default=200)
    alpha.add_argument("--calibration-bootstrap", type=int, default=500)
    alpha.add_argument("--audit-bootstrap", type=int, default=2000)

    cluster = sub.add_parser("cluster-uncertainty")
    cluster.add_argument("--fmow-calibration", type=Path, required=True)
    cluster.add_argument("--fmow-test", type=Path, required=True)
    cluster.add_argument("--fmow-test-table", type=Path, required=True)
    cluster.add_argument("--alpha-calibration-b64", type=Path, required=True)
    cluster.add_argument("--alpha-test-b64", type=Path, required=True)
    cluster.add_argument("--alpha-metadata", type=Path, required=True)
    cluster.add_argument("--output-dir", type=Path, required=True)

    seal = sub.add_parser("seal")
    seal.add_argument("--output-dir", type=Path, required=True)

    proper = sub.add_parser("fmow-proper-score")
    for seed in (42, 73, 101):
        proper.add_argument(f"--dofa-{seed}", type=Path, required=True)
        proper.add_argument(f"--resnet-{seed}", type=Path, required=True)
    proper.add_argument("--output-dir", type=Path, required=True)

    matrix = sub.add_parser("matrix")
    matrix.add_argument("--records", type=Path, required=True)
    matrix.add_argument("--output-dir", type=Path, required=True)

    reben_labelwise = sub.add_parser("reben-labelwise")
    reben_labelwise.add_argument("--probability-dir", type=Path, required=True)
    reben_labelwise.add_argument("--unified-metrics", type=Path, required=True)
    reben_labelwise.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.stage == "fmow-paired":
        paths = run_fmow_same_seed_paired_v12(
            dofa_tables={seed: getattr(args, f"dofa_{seed}") for seed in (42, 73, 101)},
            resnet_tables={seed: getattr(args, f"resnet_{seed}") for seed in (42, 73, 101)},
            output_dir=args.output_dir,
            n_bootstrap=args.bootstrap,
            min_clusters=args.min_clusters,
        )
    elif args.stage == "reben-partial":
        paths = run_reben_fixed_universe_v12(
            unified_metrics=args.unified_metrics,
            support_universe=args.support_universe,
            output_dir=args.output_dir,
            min_clusters=args.min_clusters,
        )
    elif args.stage == "sen1-event":
        paths = run_sen1_event_geobwer(
            event_metrics=args.event_metrics,
            output_dir=args.output_dir,
        )
    elif args.stage == "alpha-spatial":
        paths = run_alphaearth_validation_spatial_recalibration(
            calibration_bundle_b64=args.calibration_bundle_b64,
            test_bundle_b64=args.test_bundle_b64,
            eval_metadata=args.eval_metadata,
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            n_simulations=args.simulations,
            calibration_bootstrap=args.calibration_bootstrap,
            audit_bootstrap=args.audit_bootstrap,
        )
    elif args.stage == "cluster-uncertainty":
        paths = run_cluster_uncertainty_v060(
            fmow_calibration_npz=args.fmow_calibration,
            fmow_test_npz=args.fmow_test,
            fmow_test_table=args.fmow_test_table,
            alpha_calibration_bundle_b64=args.alpha_calibration_b64,
            alpha_test_bundle_b64=args.alpha_test_b64,
            alpha_eval_metadata=args.alpha_metadata,
            output_dir=args.output_dir,
        )
    elif args.stage == "fmow-proper-score":
        paths = run_fmow_proper_score_sensitivity(
            dofa_tables={seed: getattr(args, f"dofa_{seed}") for seed in (42, 73, 101)},
            resnet_tables={seed: getattr(args, f"resnet_{seed}") for seed in (42, 73, 101)},
            output_dir=args.output_dir,
        )
    elif args.stage == "matrix":
        records = json.loads(args.records.read_text(encoding="utf-8"))
        paths = build_evidence_status_matrix(task_records=records, output_dir=args.output_dir)
    elif args.stage == "reben-labelwise":
        paths = run_reben_labelwise_sensitivity(
            probability_dir=args.probability_dir,
            unified_metrics=args.unified_metrics,
            output_dir=args.output_dir,
        )
    else:
        paths = {"completion": seal_evidence_output(args.output_dir)}
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
