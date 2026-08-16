from __future__ import annotations

import argparse
from pathlib import Path

from rsfm_fairness_audit.paired_cross_model_review import build_cross_model_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare audited reBEN paired S2-to-S1 results across models without retraining.")
    parser.add_argument("--model-result", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    roots = {}
    for value in args.model_result:
        if "=" not in value:
            parser.error("--model-result must be NAME=DIR")
        name, root = value.split("=", 1)
        roots[name] = Path(root)
    result = build_cross_model_review(roots, args.output_dir)
    print(f"status={result['status']}")
    print(f"failure_geometry_supported={result['claim_assessment']['supported']}")


if __name__ == "__main__":
    main()
