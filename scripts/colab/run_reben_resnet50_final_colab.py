from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.reben_resnet50_campaign import (  # noqa: E402
    SENSOR_MODES,
    RebenResNet50Config,
    run_reben_resnet50_campaign,
)


def _seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected comma-separated integer seeds.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal supervised ResNet-50 S1/S2/S1+S2 reBEN GeoBWER panel."
    )
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--metadata-snow-cloud-parquet", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/geobwer/reben.yaml",
    )
    parser.add_argument("--mode", action="append", choices=SENSOR_MODES)
    parser.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audit-bootstrap", type=int, default=2000)
    parser.add_argument("--pretrained-encoder", action="store_true")
    parser.add_argument("--diagnostic-max-samples", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_reben_resnet50_campaign(
        RebenResNet50Config(
            lmdb_root=args.lmdb_root,
            metadata_parquet=args.metadata_parquet,
            metadata_snow_cloud_parquet=args.metadata_snow_cloud_parquet,
            output_dir=args.output_dir,
            persistent_output_dir=args.persistent_output_dir,
            geobwer_protocol=args.protocol,
            sensor_modes=tuple(args.mode or SENSOR_MODES),
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
            diagnostic_max_samples=args.diagnostic_max_samples,
        )
    )
    print(f"[reben:resnet50] complete: {artifacts['campaign_manifest']}")


if __name__ == "__main__":
    main()
