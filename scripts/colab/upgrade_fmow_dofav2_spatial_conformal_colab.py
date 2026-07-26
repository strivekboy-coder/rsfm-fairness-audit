from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.geobwer_extensions import (  # noqa: E402
    run_multiclass_spatial_upgrade,
)
from rsfm_fairness_audit.persistent_cache import persist_output  # noqa: E402
from rsfm_fairness_audit.spatial_conformal import (  # noqa: E402
    SpatialConformalConfig,
)


def _seeds(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected comma-separated integer seeds.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only GeoConformal upgrade for completed fMoW DOFAv2 seeds. "
            "Source probabilities and formal outputs remain immutable."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/geobwer/fmow_sentinel.yaml",
    )
    parser.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for seed in args.seeds:
        source = args.source_root / "probe_seeds" / f"seed_{seed}"
        output = args.output_dir / f"seed_{seed}"
        artifacts = run_multiclass_spatial_upgrade(
            source / "calibration_probabilities.npz",
            args.calibration_metadata,
            source / "formal_outputs",
            output,
            protocol=args.protocol,
            group_columns=(
                "country",
                "region",
                "class_label",
                "country_class",
                "region_class",
            ),
            source_calibration_manifest=source / "calibration_manifest.json",
            alpha=0.10,
            n_bootstrap=args.bootstrap,
            seed=seed,
            spatial_conformal_config=SpatialConformalConfig(),
        )
        print(f"[fmow:dofav2-spatial-upgrade] seed={seed} complete")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
    persist_output(
        args.output_dir,
        args.persistent_output_dir,
        label="fmow-dofav2-spatial-conformal-upgrade-complete",
    )


if __name__ == "__main__":
    main()
