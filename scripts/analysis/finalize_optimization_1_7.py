from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.reben_phase1_postprocess import (  # noqa: E402
    build_final_optimization_evidence,
    postprocess_label_budget,
    postprocess_paired_sensor_shift,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit items 6-7 and freeze the final optimization 1-7 evidence manifest.")
    parser.add_argument("--base-result-dir", type=Path, default=Path("outputs/optimization_1_7_v1"))
    parser.add_argument("--label-budget-dir", type=Path, required=True)
    parser.add_argument("--paired-shift-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true", help="Write a non-final readiness manifest when item 6 or 7 is absent.")
    parser.add_argument("--skip-result-postprocess", action="store_true")
    args = parser.parse_args()

    if not args.skip_result_postprocess:
        if (args.label_budget_dir / "label_budget_curves.csv").is_file():
            postprocess_label_budget(args.label_budget_dir)
        if (args.paired_shift_dir / "paired_shift_seed_panel.csv").is_file():
            postprocess_paired_sensor_shift(args.paired_shift_dir)
    artifacts = build_final_optimization_evidence(
        PROJECT_ROOT,
        args.base_result_dir,
        args.label_budget_dir,
        args.paired_shift_dir,
        args.output_dir,
        allow_pending=args.allow_pending,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
