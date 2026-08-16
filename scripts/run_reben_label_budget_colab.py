from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.reben_phase1_runners import run_label_budget_campaign, validate_cache_contract  # noqa: E402
from rsfm_fairness_audit.reben_phase1_postprocess import postprocess_label_budget  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nested TerraMind reBEN label-budget curves from frozen S2 embeddings.")
    parser.add_argument("--cache-root", type=Path, help="Root containing train/val/test embedding caches.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true", help="Audit and plot an already completed result directory without retraining.")
    args = parser.parse_args()
    if args.preflight_only and args.postprocess_only:
        parser.error("--preflight-only and --postprocess-only are mutually exclusive")
    if args.postprocess_only:
        artifacts = postprocess_label_budget(args.output_dir, expected_seeds=args.seeds or (42, 73, 101))
        for name, path in artifacts.items():
            print(f"{name}: {path}")
        return 0
    if args.cache_root is None:
        parser.error("--cache-root is required unless --postprocess-only is used")
    if args.preflight_only:
        import json
        contract = validate_cache_contract(args.cache_root)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "label_budget_preflight.json"
        path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        print(path)
        return 0
    artifacts = run_label_budget_campaign(
        args.cache_root, args.output_dir, seeds=args.seeds or (42, 73, 101),
        epochs=args.epochs, batch_size=args.batch_size, device=args.device,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
