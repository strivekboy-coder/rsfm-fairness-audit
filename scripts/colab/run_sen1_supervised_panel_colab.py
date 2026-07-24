from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.sen1_supervised_campaign import (  # noqa: E402
    SENSOR_MODES,
    Sen1SupervisedConfig,
    run_sen1_supervised_campaign,
)


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected one or more comma-separated integer seeds.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Protocol-matched ResNet34-U-Net S1/S2/S1+S2 Sen1Floods11 panel. "
            "Writes validation and test probability maps; GeoBWER finalization waits for "
            "the common all-model spatial calibration."
        )
    )
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--s2-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument("--mode", action="append", choices=SENSOR_MODES)
    parser.add_argument("--seeds", type=_csv_ints, default=(42, 73, 101))
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--pretrained-encoder",
        action="store_true",
        help="Optional ImageNet-encoder sensitivity; the formal sensor-symmetric baseline is from scratch.",
    )
    parser.add_argument(
        "--diagnostic-max-samples",
        type=int,
        help="Bound every split for a real GPU smoke; output is explicitly non-formal.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_sen1_supervised_campaign(
        Sen1SupervisedConfig(
            s1_root=args.s1_root,
            s2_root=args.s2_root,
            label_root=args.label_root,
            train_split=args.train_split,
            validation_split=args.val_split,
            test_split=args.test_split,
            output_dir=args.output_dir,
            persistent_output_dir=args.persistent_output_dir,
            sensor_modes=tuple(args.mode or SENSOR_MODES),
            seeds=args.seeds,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            pretrained_encoder=args.pretrained_encoder,
            device=args.device,
            diagnostic_max_samples=args.diagnostic_max_samples,
        )
    )
    print(f"[sen1:supervised] complete: {artifacts['campaign_manifest']}")


if __name__ == "__main__":
    main()
