from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.reben_terramind_campaign import (  # noqa: E402
    RebenTerraMindConfig,
    run_reben_terramind_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Final reBEN TerraMind frozen-encoder train/validation/test campaign for one sensor mode."
    )
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--metadata-snow-cloud-parquet", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--persistent-output-dir",
        type=Path,
        help="Drive mirror for chunks and completed stages; keep --output-dir on local /content.",
    )
    parser.add_argument("--sensor-mode", choices=["S1", "S2", "S1+S2"], required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Pinned official TerraMind_v1_base.pt; its SHA-256 is verified before inference.",
    )
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/geobwer/reben.yaml")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-chunk-size", type=int, default=4096)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--s1-unit-policy",
        choices=["already_db", "linear_power_to_db", "linear_amplitude_to_db"],
        default="already_db",
        help="Must match the inspected raw LMDB payload; the formal default is Sentinel-1 dB.",
    )
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--max-samples", type=int, help="Required with --diagnostic-only; forbidden in formal runs.")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_reben_terramind_campaign(
        RebenTerraMindConfig(
            lmdb_root=args.lmdb_root,
            metadata_parquet=args.metadata_parquet,
            metadata_snow_cloud_parquet=args.metadata_snow_cloud_parquet,
            output_dir=args.output_dir,
            persistent_output_dir=args.persistent_output_dir,
            sensor_mode=args.sensor_mode,
            terramind_checkpoint_path=args.checkpoint,
            geobwer_protocol=args.protocol,
            device=args.device,
            batch_size=args.batch_size,
            embedding_chunk_size=args.embedding_chunk_size,
            probe_epochs=args.probe_epochs,
            probe_learning_rate=args.probe_learning_rate,
            probe_weight_decay=args.probe_weight_decay,
            probe_batch_size=args.probe_batch_size,
            seed=args.seed,
            max_samples=args.max_samples,
            n_bootstrap=args.n_bootstrap,
            s1_unit_policy=args.s1_unit_policy,
            diagnostic_only=args.diagnostic_only,
        )
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
