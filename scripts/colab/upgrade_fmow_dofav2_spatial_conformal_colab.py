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
from rsfm_fairness_audit.fmow_spatial_upgrade import (  # noqa: E402
    FmowSpatialUpgradeError,
    completion_signature,
    derive_legacy_dofa_calibration,
    validate_completion_contract,
    write_completion_contract,
)
from rsfm_fairness_audit.formal_outputs import file_sha256  # noqa: E402
from rsfm_fairness_audit.persistent_cache import (  # noqa: E402
    hydrate_output,
    persist_output,
)
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
    frozen_source = args.source_root.resolve()
    for label, candidate in (
        ("output-dir", args.output_dir),
        ("persistent-output-dir", args.persistent_output_dir),
    ):
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if (
            resolved == frozen_source
            or frozen_source in resolved.parents
            or resolved in frozen_source.parents
        ):
            raise FmowSpatialUpgradeError(
                f"{label} must not overlap the frozen DOFA source directory."
            )
    hydrate_output(args.output_dir, args.persistent_output_dir)
    for seed in args.seeds:
        source = args.source_root / "probe_seeds" / f"seed_{seed}"
        output = args.output_dir / f"seed_{seed}"
        legacy = source / "calibration_predictions.npz"
        modern = source / "calibration_probabilities.npz"
        if legacy.is_file():
            calibration_source = legacy
            lineage_source = args.source_root / "run_manifest.json"
        elif modern.is_file():
            calibration_source = modern
            lineage_source = source / "calibration_manifest.json"
        else:
            raise FmowSpatialUpgradeError(
                f"seed={seed} has neither legacy nor modern calibration probabilities."
            )
        signature_payload = {
            "schema": "geobwer.fmow.dofav2_spatial_upgrade_signature.v1",
            "seed": int(seed),
            "calibration_source_sha256": file_sha256(calibration_source),
            "calibration_metadata_sha256": file_sha256(args.calibration_metadata),
            "test_formal_manifest_sha256": file_sha256(
                source / "formal_outputs" / "formal_output_manifest.json"
            ),
            "lineage_source_sha256": file_sha256(lineage_source),
            "protocol_sha256": file_sha256(args.protocol),
            "bootstrap": int(args.bootstrap),
            "spatial_config": SpatialConformalConfig().__dict__,
        }
        expected_signature = completion_signature(signature_payload)
        if validate_completion_contract(output, expected_signature):
            print(f"[fmow:dofav2-spatial-upgrade] seed={seed} completed; skipping")
            continue
        output.mkdir(parents=True, exist_ok=True)
        if legacy.is_file():
            calibration_source, calibration_manifest = derive_legacy_dofa_calibration(
                args.source_root,
                args.calibration_metadata,
                output / "legacy_calibration",
                seed=seed,
                test_formal_dir=source / "formal_outputs",
                expected_count=4485,
            )
        else:
            calibration_manifest = source / "calibration_manifest.json"
        artifacts = run_multiclass_spatial_upgrade(
            calibration_source,
            args.calibration_metadata,
            source / "formal_outputs",
            output / "spatial_conformal",
            protocol=args.protocol,
            group_columns=(
                "country",
                "region",
                "class_label",
                "country_class",
                "region_class",
            ),
            source_calibration_manifest=calibration_manifest,
            alpha=0.10,
            n_bootstrap=args.bootstrap,
            seed=seed,
            spatial_conformal_config=SpatialConformalConfig(),
        )
        print(f"[fmow:dofav2-spatial-upgrade] seed={seed} complete")
        for name, path in artifacts.items():
            print(f"{name}: {path}")
        contract = write_completion_contract(
            output,
            seed=seed,
            signature_payload=signature_payload,
        )
        print(f"completion_contract: {contract}")
        persist_output(
            args.output_dir,
            args.persistent_output_dir,
            label=f"fmow-dofav2-spatial-conformal-seed-{seed}-complete",
        )


if __name__ == "__main__":
    main()
