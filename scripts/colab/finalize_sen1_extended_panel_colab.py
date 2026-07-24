from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.sen1_extended_panel import (  # noqa: E402
    Sen1ExtendedPanelConfig,
    run_sen1_extended_panel,
)


def _seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(result) < 3 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Formal panel requires at least three unique seeds.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize the common-scale TerraMind/baseline/Prithvi Sen1 GeoBWER panel."
    )
    parser.add_argument("--terramind-root", type=Path, required=True)
    parser.add_argument("--supervised-root", type=Path, required=True)
    parser.add_argument("--prithvi-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/geobwer/sen1floods11.yaml",
    )
    parser.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    parser.add_argument("--audit-bootstrap", type=int, default=2000)
    parser.add_argument("--crc-alpha", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_sen1_extended_panel(
        Sen1ExtendedPanelConfig(
            terramind_root=args.terramind_root,
            supervised_root=args.supervised_root,
            prithvi_root=args.prithvi_root,
            output_dir=args.output_dir,
            persistent_output_dir=args.persistent_output_dir,
            protocol_path=args.protocol,
            metadata_csv=args.metadata_csv,
            audit_bootstrap=args.audit_bootstrap,
            crc_alpha=args.crc_alpha,
            seeds=args.seeds,
        )
    )
    print(f"[sen1:extended-panel] complete: {artifacts['manifest']}")


if __name__ == "__main__":
    main()
