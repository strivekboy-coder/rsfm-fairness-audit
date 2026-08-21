from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.reben_adaptation_ablation import run_reben_adaptation_ablation  # noqa: E402
from rsfm_fairness_audit.reben_phase1_runners import validate_paired_cache_contract  # noqa: E402


def _seeds(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Expected unique comma-separated seeds.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 8: validation-locked reBEN S2->S1 adaptation ladder A-C.")
    parser.add_argument("--s2-cache-root", type=Path, required=True)
    parser.add_argument("--s1-cache-root", type=Path, required=True)
    parser.add_argument("--frozen-baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/geobwer/reben.yaml")
    parser.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--stop-recovery", type=float, default=0.80)
    parser.add_argument("--no-harm-tolerance", type=float, default=0.002)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        report = validate_paired_cache_contract(args.s2_cache_root, args.s1_cache_root)
        print(report)
        return
    artifacts = run_reben_adaptation_ablation(
        args.s2_cache_root, args.s1_cache_root, args.frozen_baseline_root, args.output_dir,
        seeds=args.seeds, epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, batch_size=args.batch_size, device=args.device,
        stop_recovery=args.stop_recovery, tolerance=args.no_harm_tolerance,
        geobwer_protocol=args.protocol,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
