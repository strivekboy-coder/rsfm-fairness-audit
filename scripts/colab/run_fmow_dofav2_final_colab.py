from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.fmow_dofav2_campaign import (  # noqa: E402
    FmowDOFAv2CampaignConfig,
    run_fmow_dofav2_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final DOFAv2 train/calibrate/test fMoW-Sentinel GeoBWER campaign.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dofa-repo", type=Path, required=True, help="Official DOFA clone checked out at the pinned commit.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Verified dofav2_vit_base_e150.pth checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--persistent-output-dir",
        type=Path,
        help="Drive mirror for embeddings/probe/formal outputs; keep --output-dir on local /content.",
    )
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/geobwer/fmow_sentinel.yaml")
    parser.add_argument("--calibration-split", default="calibration")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--audit-bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--diagnostic-max-per-split",
        type=int,
        help="Run only a bounded real-data embedding smoke test; never produces formal evidence.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_fmow_dofav2_campaign(
        FmowDOFAv2CampaignConfig(
            metadata_csv=args.metadata_csv,
            data_root=args.data_root,
            model_config=args.model_config,
            output_dir=args.output_dir,
            persistent_output_dir=args.persistent_output_dir,
            geobwer_protocol=args.protocol,
            model_repo_path=args.dofa_repo,
            model_checkpoint_path=args.checkpoint,
            calibration_split=args.calibration_split,
            test_split=args.test_split,
            batch_size=args.batch_size,
            probe_epochs=args.probe_epochs,
            probe_learning_rate=args.probe_learning_rate,
            audit_bootstrap=args.audit_bootstrap,
            device=args.device,
            seed=args.seed,
            max_samples_per_split=args.diagnostic_max_per_split,
            diagnostic_only=args.diagnostic_max_per_split is not None,
        )
    )
    print(f"[fmow:dofav2] complete: {args.output_dir}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
