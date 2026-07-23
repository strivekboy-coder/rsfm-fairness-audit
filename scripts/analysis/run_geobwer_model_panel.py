from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.bwer_protocol import BWERProtocol  # noqa: E402
from rsfm_fairness_audit.config import load_yaml  # noqa: E402
from rsfm_fairness_audit.geobwer_panel import run_geobwer_model_panel  # noqa: E402


def _model_table(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/formal_audit_table.csv.")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both model name and table path are required.")
    return name.strip(), Path(raw_path)


def _comparison(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use MODEL_A=MODEL_B for a pre-specified contrast.")
    left, right = (item.strip() for item in value.split("=", 1))
    if not left or not right or left == right:
        raise argparse.ArgumentTypeError("A contrast requires two distinct non-empty model names.")
    return left, right


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired common-unit GeoBWER model panel.")
    parser.add_argument("--model-table", action="append", type=_model_table, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column")
    parser.add_argument("--loss-column", default="risk")
    parser.add_argument("--unit-column")
    parser.add_argument("--cluster-column")
    parser.add_argument(
        "--comparison",
        action="append",
        type=_comparison,
        help="Pre-specified MODEL_A=MODEL_B contrast; repeat. Default compares every pair.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tables = dict(args.model_table)
    if len(tables) != len(args.model_table):
        raise ValueError("Model names in --model-table must be unique.")
    protocol = BWERProtocol.from_mapping(load_yaml(args.protocol))
    artifacts = run_geobwer_model_panel(
        tables,
        args.output_dir,
        protocol=protocol,
        group_column=args.group_column,
        loss_column=args.loss_column,
        unit_column=args.unit_column,
        cluster_column=args.cluster_column,
        comparison_pairs=args.comparison,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    for name, path in vars(artifacts).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
