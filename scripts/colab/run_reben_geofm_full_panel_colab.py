from __future__ import annotations

"""Final six-model reBEN architecture-by-modality GeoBWER campaign."""

import argparse
import gc
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.adapters.terramind import validate_terramind_checkpoint  # noqa: E402
from rsfm_fairness_audit.bwer_protocol import BWERProtocol  # noqa: E402
from rsfm_fairness_audit.config import load_yaml  # noqa: E402
from rsfm_fairness_audit.geobwer_panel import run_geobwer_model_panel  # noqa: E402
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output  # noqa: E402
from rsfm_fairness_audit.reben_croma_geobwer_campaign import (  # noqa: E402
    RebenCROMAConfig,
    calibrate_reben_croma_train_normalization,
    run_reben_croma_campaign,
    validate_croma_assets,
)
from rsfm_fairness_audit.reben_terramind_campaign import (  # noqa: E402
    RebenTerraMindConfig,
    run_reben_terramind_campaign,
)
from rsfm_fairness_audit.reben_resnet50_campaign import (  # noqa: E402
    RebenResNet50Config,
    run_reben_resnet50_campaign,
)


MODES = ("S1", "S2", "S1+S2")


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Expected unique comma-separated integer seeds.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal CROMA/TerraMind x S1/S2/S1+S2 reBEN campaign and paired GeoBWER panel."
    )
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--metadata-snow-cloud-parquet", type=Path)
    parser.add_argument("--croma-repo", type=Path, required=True)
    parser.add_argument("--croma-checkpoint", type=Path, required=True)
    parser.add_argument("--terramind-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/geobwer/reben.yaml")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-chunk-size", type=int, default=4096)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--seeds", type=_csv_ints, default=(42, 73, 101))
    parser.add_argument(
        "--seed",
        type=int,
        help="Legacy single-seed diagnostic override; formal campaigns require at least three seeds.",
    )
    parser.add_argument("--supervised-max-epochs", type=int, default=30)
    parser.add_argument("--supervised-patience", type=int, default=5)
    parser.add_argument("--supervised-batch-size", type=int, default=128)
    parser.add_argument("--supervised-num-workers", type=int, default=4)
    parser.add_argument(
        "--supervised-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--supervised-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--supervised-prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--supervised-host-to-device-non-blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--supervised-learning-rate", type=float, default=3e-4)
    parser.add_argument("--supervised-weight-decay", type=float, default=1e-4)
    parser.add_argument("--supervised-pretrained-encoder", action="store_true")
    parser.add_argument(
        "--s1-unit-policy",
        choices=["already_db", "linear_power_to_db", "linear_amplitude_to_db"],
        default="already_db",
        help="TerraMind S1 unit contract; confirm from the real LMDB preflight.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--diagnostic-max-samples",
        type=int,
        help="Run the six real adapter paths on bounded split samples and stop before formal auditing.",
    )
    return parser


def _release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _common_kwargs(
    args: argparse.Namespace,
    output: Path,
    persistent: Path | None,
    *,
    seed: int,
    embedding_cache_root: Path,
    persistent_embedding_cache_root: Path | None,
) -> dict[str, object]:
    return {
        "lmdb_root": args.lmdb_root,
        "metadata_parquet": args.metadata_parquet,
        "metadata_snow_cloud_parquet": args.metadata_snow_cloud_parquet,
        "output_dir": output,
        "persistent_output_dir": persistent,
        "geobwer_protocol": args.protocol,
        "device": args.device,
        "batch_size": args.batch_size,
        "embedding_chunk_size": args.embedding_chunk_size,
        "probe_epochs": args.probe_epochs,
        "probe_learning_rate": args.probe_learning_rate,
        "probe_weight_decay": args.probe_weight_decay,
        "probe_batch_size": args.probe_batch_size,
        "seed": seed,
        "embedding_cache_root": embedding_cache_root,
        "persistent_embedding_cache_root": persistent_embedding_cache_root,
        "n_bootstrap": args.n_bootstrap,
        "max_samples": args.diagnostic_max_samples,
        "diagnostic_only": args.diagnostic_max_samples is not None,
    }


def main() -> None:
    args = build_parser().parse_args()
    seeds = (int(args.seed),) if args.seed is not None else tuple(map(int, args.seeds))
    if args.diagnostic_max_samples is None and len(seeds) < 3:
        raise RuntimeError("The formal reBEN panel requires at least three independent probe/training seeds.")
    croma_assets = validate_croma_assets(args.croma_repo, args.croma_checkpoint)
    _, terramind_sha256 = validate_terramind_checkpoint(args.terramind_checkpoint)
    hydrate_output(args.output_dir, args.persistent_output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = BWERProtocol.from_mapping(load_yaml(args.protocol))
    normalization_path = args.output_dir / "croma_train_normalization.json"
    calibration_config = RebenCROMAConfig(
        **_common_kwargs(
            args,
            args.output_dir / "croma" / "s1_plus_s2",
            args.persistent_output_dir / "croma" / "s1_plus_s2" if args.persistent_output_dir else None,
            seed=seeds[0],
            embedding_cache_root=args.output_dir / "croma" / "s1_plus_s2" / "shared_embedding_cache",
            persistent_embedding_cache_root=(
                args.persistent_output_dir / "croma" / "s1_plus_s2" / "shared_embedding_cache"
                if args.persistent_output_dir
                else None
            ),
        ),
        sensor_mode="S1+S2",
        croma_checkpoint_path=args.croma_checkpoint,
        croma_repo_path=args.croma_repo,
        normalization_stats_path=normalization_path,
    )
    calibrate_reben_croma_train_normalization(calibration_config, normalization_path)
    persist_output(args.output_dir, args.persistent_output_dir, label="croma-train-normalization")

    model_tables: dict[str, Path] = {}
    run_manifests: dict[str, str] = {}
    for architecture in ("croma", "terramind"):
        for mode in MODES:
            slug = mode.lower().replace("+", "_plus_")
            shared_cache = args.output_dir / architecture / slug / "shared_embedding_cache"
            persistent_shared_cache = (
                args.persistent_output_dir / architecture / slug / "shared_embedding_cache"
                if args.persistent_output_dir
                else None
            )
            for seed in seeds:
                local = args.output_dir / architecture / slug / f"seed_{seed}"
                persistent = (
                    args.persistent_output_dir / architecture / slug / f"seed_{seed}"
                    if args.persistent_output_dir
                    else None
                )
                common = _common_kwargs(
                    args,
                    local,
                    persistent,
                    seed=seed,
                    embedding_cache_root=shared_cache,
                    persistent_embedding_cache_root=persistent_shared_cache,
                )
                if architecture == "croma":
                    artifacts = run_reben_croma_campaign(
                        RebenCROMAConfig(
                            **common,
                            sensor_mode=mode,
                            croma_checkpoint_path=args.croma_checkpoint,
                            croma_repo_path=args.croma_repo,
                            normalization_stats_path=normalization_path,
                        )
                    )
                    model_name = f"croma_base_{slug}_seed_{seed}"
                else:
                    artifacts = run_reben_terramind_campaign(
                        RebenTerraMindConfig(
                            **common,
                            sensor_mode=mode,
                            terramind_checkpoint_path=args.terramind_checkpoint,
                            s1_unit_policy=args.s1_unit_policy,
                        )
                    )
                    model_name = f"terramind_v1_base_{slug}_seed_{seed}"
                if args.diagnostic_max_samples is None:
                    model_tables[model_name] = artifacts["formal_audit_table"]
                    run_manifests[model_name] = str(artifacts["run_manifest"])
                else:
                    run_manifests[model_name] = str(artifacts["diagnostic_manifest"])
                _release_memory()

    supervised = run_reben_resnet50_campaign(
        RebenResNet50Config(
            lmdb_root=args.lmdb_root,
            metadata_parquet=args.metadata_parquet,
            metadata_snow_cloud_parquet=args.metadata_snow_cloud_parquet,
            output_dir=args.output_dir / "supervised_resnet50",
            persistent_output_dir=(
                args.persistent_output_dir / "supervised_resnet50"
                if args.persistent_output_dir
                else None
            ),
            geobwer_protocol=args.protocol,
            sensor_modes=MODES,
            seeds=seeds,
            max_epochs=args.supervised_max_epochs,
            patience=args.supervised_patience,
            batch_size=args.supervised_batch_size,
            num_workers=args.supervised_num_workers,
            pin_memory=args.supervised_pin_memory,
            persistent_workers=args.supervised_persistent_workers,
            prefetch_factor=args.supervised_prefetch_factor,
            host_to_device_non_blocking=(
                args.supervised_host_to_device_non_blocking
            ),
            learning_rate=args.supervised_learning_rate,
            weight_decay=args.supervised_weight_decay,
            pretrained_encoder=args.supervised_pretrained_encoder,
            device=args.device,
            audit_bootstrap=args.n_bootstrap,
            diagnostic_max_samples=args.diagnostic_max_samples,
        )
    )
    for mode in MODES:
        slug = mode.lower().replace("+", "_plus_")
        for seed in seeds:
            run_key = f"resnet50_{slug}_seed_{seed}"
            model_name = f"resnet50_supervised_{slug}_seed_{seed}"
            artifacts = supervised["runs"][run_key]
            if args.diagnostic_max_samples is None:
                model_tables[model_name] = Path(artifacts["formal_audit_table"])
                run_manifests[model_name] = str(artifacts["run_manifest"])
            else:
                run_manifests[model_name] = str(artifacts["diagnostic_manifest"])

    if args.diagnostic_max_samples is not None:
        manifest = args.output_dir / "diagnostic_panel_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "geobwer.reben.geofm_full_panel_diagnostic.v1",
                    "formal_evidence": False,
                    "reason": "explicit_bounded_real_gpu_smoke",
                    "max_samples_per_split": args.diagnostic_max_samples,
                    "seeds": list(seeds),
                    "croma_assets": croma_assets,
                    "terramind_checkpoint_sha256": terramind_sha256,
                    "normalization_stats": str(normalization_path),
                    "run_manifests": run_manifests,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        persist_output(args.output_dir, args.persistent_output_dir, label="reben-geofm-panel-diagnostic-complete")
        print(f"[reben:full-panel] diagnostic complete: {manifest}")
        return

    primary_contrasts = tuple(
        pair
        for seed in seeds
        for mode in ("s1", "s2", "s1_plus_s2")
        for pair in (
            (
                f"croma_base_{mode}_seed_{seed}",
                f"terramind_v1_base_{mode}_seed_{seed}",
            ),
            (
                f"croma_base_{mode}_seed_{seed}",
                f"resnet50_supervised_{mode}_seed_{seed}",
            ),
            (
                f"terramind_v1_base_{mode}_seed_{seed}",
                f"resnet50_supervised_{mode}_seed_{seed}",
            ),
        )
    )
    panel = run_geobwer_model_panel(
        model_tables,
        args.output_dir / "model_panel",
        protocol=protocol,
        group_column="country",
        cluster_column="source_tile_id",
        comparison_pairs=primary_contrasts,
        n_bootstrap=args.n_bootstrap,
        seed=seeds[0],
    )
    manifest = args.output_dir / "campaign_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.reben.geofm_full_panel.v2",
                "formal_evidence": True,
                "design": "3_architectures_x_3_modalities_x_3_seeds",
                "seeds": list(seeds),
                "croma_assets": croma_assets,
                "terramind_checkpoint_sha256": terramind_sha256,
                "normalization_stats": str(normalization_path),
                "run_manifests": run_manifests,
                "primary_contrasts": [list(pair) for pair in primary_contrasts],
                "model_panel": {name: str(path) for name, path in vars(panel).items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    persist_output(args.output_dir, args.persistent_output_dir, label="reben-geofm-full-panel-complete")
    print(f"[reben:full-panel] complete: {manifest}")


if __name__ == "__main__":
    main()
