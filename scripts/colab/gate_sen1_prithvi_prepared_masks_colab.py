from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rsfm_fairness_audit.sen1_prithvi_mask_gate import (  # noqa: E402
    gate_prithvi_prepared_masks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only hard gate for all 431+15 prepared Prithvi masks."
    )
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--core-metadata", type=Path)
    parser.add_argument("--bolivia-root", type=Path, required=True)
    parser.add_argument("--bolivia-metadata", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = gate_prithvi_prepared_masks(
        core_root=args.core_root,
        core_metadata=args.core_metadata,
        bolivia_root=args.bolivia_root,
        bolivia_metadata=args.bolivia_metadata,
    )
    if args.output_json:
        if args.output_json.exists():
            raise RuntimeError(f"Refusing to overwrite gate evidence: {args.output_json}")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("SEN1_PRITHVI_431_PLUS_15_MASK_GATE=PASS")


if __name__ == "__main__":
    main()
