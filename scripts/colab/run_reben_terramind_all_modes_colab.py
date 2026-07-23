from __future__ import annotations

"""Run the frozen reBEN TerraMind sensor campaign and paired model panel once.

This is the formal orchestration entry point.  The single-mode runner remains
available for recovery, but a paper run should use this script so S1, S2 and
S1+S2 share one protocol and are compared on exactly the same test units.
"""

import argparse
import gc
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.adapters.terramind import (  # noqa: E402
    TERRAMIND_OFFICIAL_REVISION,
    validate_terramind_checkpoint,
)
from rsfm_fairness_audit.bwer_protocol import BWERProtocol  # noqa: E402
from rsfm_fairness_audit.config import load_yaml  # noqa: E402
from rsfm_fairness_audit.geobwer_panel import run_geobwer_model_panel  # noqa: E402
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output  # noqa: E402
from rsfm_fairness_audit.reben_terramind_campaign import (  # noqa: E402
    RebenTerraMindConfig,
    run_reben_terramind_campaign,
)


MODES = ("S1", "S2", "S1+S2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal reBEN TerraMind S1/S2/S1+S2 campaign plus paired common-unit GeoBWER panel."
    )
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--metadata-snow-cloud-parquet", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--persistent-output-dir",
        type=Path,
        help="Drive mirror; --output-dir must remain on local /content storage.",
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
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    args = build_parser().parse_args()
    _, checkpoint_sha256 = validate_terramind_checkpoint(args.checkpoint)
    hydrate_output(args.output_dir, args.persistent_output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = BWERProtocol.from_mapping(load_yaml(args.protocol))
    model_tables: dict[str, Path] = {}
    mode_manifests: dict[str, str] = {}
    for mode in MODES:
        slug = mode.lower().replace("+", "_plus_")
        local_dir = args.output_dir / slug
        persistent_dir = args.persistent_output_dir / slug if args.persistent_output_dir else None
        artifacts = run_reben_terramind_campaign(
            RebenTerraMindConfig(
                lmdb_root=args.lmdb_root,
                metadata_parquet=args.metadata_parquet,
                metadata_snow_cloud_parquet=args.metadata_snow_cloud_parquet,
                output_dir=local_dir,
                persistent_output_dir=persistent_dir,
                sensor_mode=mode,
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
                n_bootstrap=args.n_bootstrap,
                s1_unit_policy=args.s1_unit_policy,
            )
        )
        model_name = f"terramind_v1_base_{slug}"
        model_tables[model_name] = artifacts["formal_audit_table"]
        mode_manifests[mode] = str(artifacts["run_manifest"])
        persist_output(local_dir, persistent_dir, label=f"reben-{mode}-complete")
        _release_accelerator_memory()

    panel = run_geobwer_model_panel(
        model_tables,
        args.output_dir / "model_panel",
        protocol=protocol,
        group_column="country",
        cluster_column="source_tile_id",
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    manifest = args.output_dir / "campaign_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.reben.terramind_all_modes.v1",
                "formal_evidence": True,
                "sensor_modes": list(MODES),
                "terramind_revision": TERRAMIND_OFFICIAL_REVISION,
                "terramind_checkpoint_sha256": checkpoint_sha256,
                "mode_manifests": mode_manifests,
                "model_panel": {name: str(path) for name, path in vars(panel).items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    persist_output(args.output_dir, args.persistent_output_dir, label="reben-all-modes-complete")
    print(f"[reben:terramind] formal campaign complete: {manifest}")


if __name__ == "__main__":
    main()
