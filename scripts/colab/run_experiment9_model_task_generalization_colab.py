from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.fmow_terramind_campaign import FmowTerraMindConfig, run_fmow_terramind_campaign  # noqa: E402
from rsfm_fairness_audit.model_task_generalization import analyze_model_task_matrix  # noqa: E402
from rsfm_fairness_audit.reben_dofav2_campaign import RebenDOFAv2Config, run_reben_dofav2_campaign  # noqa: E402


def _seeds(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Expected unique comma-separated seeds.")
    return parsed


def _release() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 9 missing-cell runners and 2x2 analysis.")
    commands = parser.add_subparsers(dest="command", required=True)
    reben = commands.add_parser("dofa-reben", help="Run the missing DOFAv2 x reBEN S2 cell.")
    reben.add_argument("--lmdb-root", type=Path, required=True)
    reben.add_argument("--metadata-parquet", type=Path, required=True)
    reben.add_argument("--metadata-snow-cloud-parquet", type=Path)
    reben.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs/models/dofav2_fmow_sentinel.yaml")
    reben.add_argument("--dofa-repo", type=Path, required=True)
    reben.add_argument("--checkpoint", type=Path, required=True)
    reben.add_argument("--output-dir", type=Path, required=True)
    reben.add_argument("--persistent-output-dir", type=Path)
    reben.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    reben.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    reben.add_argument("--batch-size", type=int, default=32)
    reben.add_argument("--embedding-chunk-size", type=int, default=4096)
    reben.add_argument("--probe-epochs", type=int, default=100)
    reben.add_argument("--probe-batch-size", type=int, default=512)
    reben.add_argument("--n-bootstrap", type=int, default=2000)
    reben.add_argument("--diagnostic-max-samples", type=int)
    fmow = commands.add_parser("terramind-fmow", help="Run the missing TerraMind x fMoW-Sentinel cell.")
    fmow.add_argument("--metadata-csv", type=Path, required=True)
    fmow.add_argument("--data-root", type=Path, required=True)
    fmow.add_argument("--checkpoint", type=Path, required=True)
    fmow.add_argument("--output-dir", type=Path, required=True)
    fmow.add_argument("--persistent-output-dir", type=Path)
    fmow.add_argument("--embedding-cache-dir", type=Path)
    fmow.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    fmow.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    fmow.add_argument("--batch-size", type=int, default=32)
    fmow.add_argument("--probe-epochs", type=int, default=200)
    fmow.add_argument("--probe-batch-size", type=int, default=512)
    fmow.add_argument("--audit-bootstrap", type=int, default=2000)
    fmow.add_argument("--diagnostic-max-per-split", type=int)
    analysis = commands.add_parser("analyze", help="Analyze four completed cells without GPU work.")
    analysis.add_argument("--dofa-fmow-root", type=Path, required=True)
    analysis.add_argument("--terramind-fmow-root", type=Path, required=True)
    analysis.add_argument("--dofa-reben-root", type=Path, required=True)
    analysis.add_argument("--terramind-reben-root", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "dofa-reben":
        shared = args.output_dir / "shared_embedding_cache"
        persistent_shared = args.persistent_output_dir / "shared_embedding_cache" if args.persistent_output_dir else None
        panel_runs = {}
        for seed in args.seeds:
            artifacts = run_reben_dofav2_campaign(RebenDOFAv2Config(
                lmdb_root=args.lmdb_root, metadata_parquet=args.metadata_parquet,
                metadata_snow_cloud_parquet=args.metadata_snow_cloud_parquet,
                output_dir=args.output_dir / f"seed_{seed}",
                persistent_output_dir=args.persistent_output_dir / f"seed_{seed}" if args.persistent_output_dir else None,
                embedding_cache_root=shared, persistent_embedding_cache_root=persistent_shared,
                model_config=args.model_config, dofa_repo_path=args.dofa_repo, dofa_checkpoint_path=args.checkpoint,
                device=args.device, batch_size=args.batch_size, embedding_chunk_size=args.embedding_chunk_size,
                probe_epochs=args.probe_epochs, probe_batch_size=args.probe_batch_size, seed=seed,
                n_bootstrap=args.n_bootstrap, max_samples=args.diagnostic_max_samples,
                diagnostic_only=args.diagnostic_max_samples is not None,
            ))
            print(f"[experiment9:dofa-reben] seed={seed} artifacts={artifacts}")
            panel_runs[str(seed)] = {name: str(path) for name, path in artifacts.items()}
            _release()
        panel = args.output_dir / "experiment9_dofav2_reben_panel_manifest.json"
        payload = {"schema": "geobwer.experiment9.dofav2_reben_panel.v1",
                   "status": "diagnostic" if args.diagnostic_max_samples is not None else "complete",
                   "formal_evidence": args.diagnostic_max_samples is None,
                   "seeds": list(args.seeds), "shared_embedding_cache": str(shared), "runs": panel_runs}
        panel.parent.mkdir(parents=True, exist_ok=True)
        panel.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.persistent_output_dir:
            persistent_panel = args.persistent_output_dir / panel.name
            persistent_panel.parent.mkdir(parents=True, exist_ok=True)
            persistent_panel.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"panel_manifest: {panel}")
        return
    if args.command == "terramind-fmow":
        artifacts = run_fmow_terramind_campaign(FmowTerraMindConfig(
            metadata_csv=args.metadata_csv, data_root=args.data_root, terramind_checkpoint_path=args.checkpoint,
            output_dir=args.output_dir, persistent_output_dir=args.persistent_output_dir,
            embedding_cache_dir=args.embedding_cache_dir, device=args.device, batch_size=args.batch_size,
            probe_epochs=args.probe_epochs, probe_batch_size=args.probe_batch_size,
            audit_bootstrap=args.audit_bootstrap, seeds=args.seeds,
            max_samples_per_split=args.diagnostic_max_per_split,
            diagnostic_only=args.diagnostic_max_per_split is not None,
        ))
        print(f"[experiment9:terramind-fmow] artifacts={artifacts}")
        return
    artifacts = analyze_model_task_matrix({
        ("dofav2", "fmow"): args.dofa_fmow_root,
        ("terramind", "fmow"): args.terramind_fmow_root,
        ("dofav2", "reben"): args.dofa_reben_root,
        ("terramind", "reben"): args.terramind_reben_root,
    }, args.output_dir)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
