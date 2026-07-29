from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.prithvi_sen1_campaign import (  # noqa: E402
    PrithviSen1CampaignConfig,
    run_prithvi_sen1_probability_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate the official Prithvi Sen1Floods11 task checkpoint to the GeoBWER 1.1 "
            "probability-map contract on the official validation/test members."
        )
    )
    parser.add_argument("--prepared-data-root", type=Path, required=True)
    parser.add_argument("--prepared-metadata-csv", type=Path)
    parser.add_argument("--bolivia-prepared-data-root", type=Path, required=True)
    parser.add_argument("--bolivia-prepared-metadata-csv", type=Path)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument(
        "--bolivia-split",
        "--heldout-event-split",
        dest="bolivia_split",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--diagnostic-max-samples", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_prithvi_sen1_probability_campaign(
        PrithviSen1CampaignConfig(
            prepared_data_root=args.prepared_data_root,
            prepared_metadata_csv=args.prepared_metadata_csv,
            bolivia_prepared_data_root=args.bolivia_prepared_data_root,
            bolivia_prepared_metadata_csv=args.bolivia_prepared_metadata_csv,
            model_config=args.model_config,
            train_split=args.train_split,
            validation_split=args.val_split,
            test_split=args.test_split,
            bolivia_split=args.bolivia_split,
            output_dir=args.output_dir,
            persistent_output_dir=args.persistent_output_dir,
            batch_size=args.batch_size,
            device=args.device,
            diagnostic_max_samples=args.diagnostic_max_samples,
        )
    )
    print(f"[prithvi:sen1] probability migration complete: {artifacts['manifest']}")
    if args.diagnostic_max_samples is not None:
        manifest = json.loads(Path(artifacts["manifest"]).read_text(encoding="utf-8"))
        expected = int(args.diagnostic_max_samples)
        if (
            int(manifest.get("validation_count", -1)) != expected
            or int(manifest.get("test_count", -1)) != expected
            or int(manifest.get("bolivia_holdout_count", -1)) != expected
            or manifest.get("formal_evidence") is not False
        ):
            raise RuntimeError("Prithvi-only diagnostic contract did not preserve the requested bounded split counts.")
        for split in ("validation", "test", "bolivia_holdout"):
            validation = manifest["split_runtime_validation"][split]
            if validation.get("full_probability_layout") != "[B,2,H,W]":
                raise RuntimeError(f"Prithvi-only {split} output did not preserve full class probabilities.")
            if float(validation.get("maximum_probability_sum_error", 1.0)) > 1e-5:
                raise RuntimeError(f"Prithvi-only {split} probabilities failed the unit-sum contract.")
        print("PRITHVI_ONLY_GPU_PROBE=PASS")
    print("[prithvi:sen1] GeoBWER finalization remains blocked until the all-model "
          "validation-only spatial calibration is frozen.")


if __name__ == "__main__":
    main()
