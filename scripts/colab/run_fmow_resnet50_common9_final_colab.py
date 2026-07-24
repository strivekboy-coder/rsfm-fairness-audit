from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.fmow_resnet50_campaign import (  # noqa: E402
    FmowResNet50CampaignConfig,
    run_fmow_resnet50_campaign,
)


def _seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected comma-separated integer seeds.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal multi-seed common-9-band ResNet-50 fMoW-Sentinel GeoBWER baseline."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/geobwer/fmow_sentinel.yaml",
    )
    parser.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audit-bootstrap", type=int, default=2000)
    parser.add_argument("--pretrained-encoder", action="store_true")
    parser.add_argument("--diagnostic-max-per-split", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_fmow_resnet50_campaign(
        FmowResNet50CampaignConfig(
            metadata_csv=args.metadata_csv,
            data_root=args.data_root,
            output_dir=args.output_dir,
            persistent_output_dir=args.persistent_output_dir,
            geobwer_protocol=args.protocol,
            seeds=args.seeds,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            pretrained_encoder=args.pretrained_encoder,
            device=args.device,
            audit_bootstrap=args.audit_bootstrap,
            diagnostic_max_samples_per_split=args.diagnostic_max_per_split,
        )
    )
    print(f"[fmow:resnet50] complete: {artifacts['campaign_manifest']}")


if __name__ == "__main__":
    main()
