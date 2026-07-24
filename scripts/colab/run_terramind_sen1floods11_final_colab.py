from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.formal_outputs import file_sha256  # noqa: E402
from rsfm_fairness_audit.adapters.terramind import (  # noqa: E402
    TERRAMIND_OFFICIAL_REVISION,
    TERRAMIND_OFFICIAL_SHA256,
    validate_terratorch_runtime,
    validate_terramind_checkpoint,
)
from rsfm_fairness_audit.bwer_protocol import BWERProtocol  # noqa: E402
from rsfm_fairness_audit.geobwer_extensions import run_segmentation_uncertainty_suite  # noqa: E402
from rsfm_fairness_audit.geobwer_panel import run_geobwer_model_panel  # noqa: E402
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output  # noqa: E402
from rsfm_fairness_audit.sen1floods11_formal import (  # noqa: E402
    calibrate_common_sen1_spatial_blocks,
    finalize_sen1floods11_segmentation,
    load_sen1_probability_units,
)
from rsfm_fairness_audit.terramind_sen1_config import (  # noqa: E402
    prepare_terramind_sen1_splits,
    write_terramind_sen1floods11_config,
)


MODES = ("S1", "S2", "S1+S2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Final, resumable TerraMind S1/S2/S1+S2 Sen1Floods11 campaign with formal probability exports."
    )
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--s2-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Pinned official TerraMind_v1_base.pt; SHA-256 is verified before any formal fit.",
    )
    parser.add_argument(
        "--persistent-output-dir",
        type=Path,
        help="Drive mirror for completed stages. Keep --output-dir on /content; never train directly on Drive.",
    )
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/geobwer/sen1floods11.yaml")
    parser.add_argument("--mode", action="append", choices=MODES, help="Repeat to restrict modes; default is all three.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-simulations", type=int, default=200)
    parser.add_argument("--calibration-bootstrap", type=int, default=500)
    parser.add_argument("--audit-bootstrap", type=int, default=2000)
    parser.add_argument("--crc-alpha", type=float, default=0.10)
    parser.add_argument("--minimum-moderate-tail-power", type=float, default=0.80)
    parser.add_argument("--checkpoint-mirror-every-n-epochs", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate configs without training/inference.")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run one real train/validation batch per mode with Lightning fast_dev_run, then stop.",
    )
    return parser


def _terratorch_command() -> list[str]:
    validate_terratorch_runtime()
    executable = shutil.which("terratorch")
    if executable:
        return [executable]
    raise RuntimeError("The official `terratorch` CLI is unavailable. Install terratorch>=1.2.5,<1.3 in Colab.")


def _run(command: list[str]) -> None:
    print("[terramind:campaign]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _split_count(path: Path) -> int:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Split file is empty: {path}")
    return len(lines)


def _probability_artifact(export_root: Path, value: Any) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        if raw.exists():
            return raw
        return export_root / "samples" / raw.name
    return export_root / raw


def _export_complete(path: Path, expected: int) -> bool:
    index_parts = sorted((path / "index_parts").glob("*.jsonl"))
    if not index_parts or not list(path.glob("writer_manifest_rank_*.json")):
        return False
    rows: dict[str, dict[str, Any]] = {}
    for part in index_parts:
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                sample_id = str(item["sample_id"])
                if sample_id in rows and rows[sample_id] != item:
                    raise RuntimeError(f"Conflicting duplicate probability index for sample_id={sample_id}.")
                rows[sample_id] = item
    return len(rows) == expected and all(
        _probability_artifact(path, row["probability_path"]).exists() for row in rows.values()
    )


def _checkpoint(run_dir: Path) -> Path | None:
    best = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    if len(best) > 1:
        raise RuntimeError(f"More than one best checkpoint exists under {run_dir}; provenance is ambiguous: {best}")
    return best[0] if best else None


def _fit_if_needed(
    config: Path,
    run_dir: Path,
    *,
    backbone_checkpoint_sha256: str,
    dry_run: bool,
) -> Path | None:
    protocol_path = run_dir / "fit_protocol.json"
    current_protocol = {
        "schema": "geobwer.terramind.fit_protocol.v1",
        "config_sha256": file_sha256(config),
        "backbone_checkpoint_sha256": backbone_checkpoint_sha256,
        "backbone_revision": TERRAMIND_OFFICIAL_REVISION,
    }
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != current_protocol:
            raise RuntimeError(
                f"Fit protocol changed under an existing output directory: {run_dir}. "
                "Use a new output directory; do not resume a checkpoint produced by another protocol."
            )
    else:
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(current_protocol, indent=2), encoding="utf-8")
    best = _checkpoint(run_dir)
    marker = run_dir / "fit_complete.json"
    if best is not None and marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if (
            payload.get("checkpoint_sha256") == file_sha256(best)
            and payload.get("fit_protocol_sha256") == file_sha256(protocol_path)
        ):
            print(f"[terramind:campaign] reusing complete checkpoint {best}")
            return best
    if dry_run:
        return best
    command = _terratorch_command() + ["fit", "-c", str(config)]
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        command += ["--ckpt_path", str(last)]
    _run(command)
    best = _checkpoint(run_dir)
    if best is None:
        raise RuntimeError(f"TerraTorch fit finished without one best checkpoint under {run_dir}.")
    marker.write_text(
        json.dumps(
            {
                "checkpoint": str(best),
                "checkpoint_sha256": file_sha256(best),
                "fit_protocol": str(protocol_path),
                "fit_protocol_sha256": file_sha256(protocol_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best


def _predict_if_needed(
    config: Path,
    checkpoint: Path | None,
    output: Path,
    *,
    expected: int,
    dry_run: bool,
) -> None:
    if _export_complete(output, expected):
        print(f"[terramind:campaign] reusing complete probability export {output}")
        return
    if dry_run:
        return
    if checkpoint is None:
        raise RuntimeError("Prediction requires a completed checkpoint.")
    _run(_terratorch_command() + ["predict", "-c", str(config), "--ckpt_path", str(checkpoint)])
    if not _export_complete(output, expected):
        raise RuntimeError(
            f"Probability export failed completeness check: expected={expected}, path={output}. "
            "Do not proceed to BWER with partial predictions."
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run and args.smoke_only:
        raise ValueError("--dry-run and --smoke-only are mutually exclusive.")
    if args.dry_run and not args.checkpoint.is_file():
        backbone_checkpoint_sha256 = TERRAMIND_OFFICIAL_SHA256
        print(
            "[terramind:campaign] dry-run only: checkpoint bytes are absent; generated configs retain the "
            "official expected checkpoint identity.",
            flush=True,
        )
    else:
        _, backbone_checkpoint_sha256 = validate_terramind_checkpoint(args.checkpoint)
    hydrate_output(args.output_dir, args.persistent_output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_report = prepare_terramind_sen1_splits(
        {
            "s1_root": args.s1_root,
            "s2_root": args.s2_root,
            "label_root": args.label_root,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
        },
        args.output_dir / "terratorch_splits",
    )
    terratorch_splits = {
        name: Path(path) for name, path in source_report["terratorch_split_paths"].items()
    }
    (args.output_dir / "source_preflight.json").write_text(
        json.dumps(
            {
                **source_report,
                "terramind_checkpoint": str(args.checkpoint),
                "terramind_checkpoint_sha256": backbone_checkpoint_sha256,
                "terramind_revision": TERRAMIND_OFFICIAL_REVISION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    persist_output(args.output_dir, args.persistent_output_dir, label="source-preflight")
    modes = tuple(args.mode or MODES)
    if set(modes) != set(MODES) and not (args.dry_run or args.smoke_only):
        raise RuntimeError(
            "The final primary campaign requires S1, S2, and S1+S2 together so block calibration and comparisons "
            "use a common model set. Use --dry-run to inspect a subset of configs."
        )
    checkpoints: dict[str, Path | None] = {}
    validation_exports: dict[str, Path] = {}
    test_exports: dict[str, Path] = {}
    for mode in modes:
        slug = mode.lower().replace("+", "_plus_")
        run_dir = args.output_dir / slug
        config_dir = run_dir / "configs"
        validation_output = run_dir / "probabilities" / "validation"
        test_output = run_dir / "probabilities" / "test"
        common = {
            "sensor_mode": mode,
            "s1_root": args.s1_root,
            "s2_root": args.s2_root,
            "label_root": args.label_root,
            "train_split": terratorch_splits["train"],
            "val_split": terratorch_splits["validation"],
            "test_split": terratorch_splits["test"],
            "run_dir": run_dir,
            "backbone_checkpoint_path": args.checkpoint,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_epochs": args.max_epochs,
            "fast_dev_run": args.smoke_only,
            "persistent_checkpoint_dir": (
                args.persistent_output_dir / slug / "checkpoints"
                if args.persistent_output_dir is not None
                else None
            ),
            "checkpoint_mirror_every_n_epochs": args.checkpoint_mirror_every_n_epochs,
        }
        fit_config = write_terramind_sen1floods11_config(config_dir / "fit.yaml", **common)
        if args.smoke_only:
            _run(_terratorch_command() + ["fit", "-c", str(fit_config)])
            smoke_manifest = run_dir / "diagnostic_manifest.json"
            smoke_manifest.write_text(
                json.dumps(
                    {
                        "schema": "geobwer.sen1floods11.terramind_diagnostic.v1",
                        "formal_evidence": False,
                        "reason": "real_gpu_fast_dev_run",
                        "sensor_mode": mode,
                        "fit_config": str(fit_config),
                        "fit_config_sha256": file_sha256(fit_config),
                        "pretraining_checkpoint_sha256": backbone_checkpoint_sha256,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            persist_output(
                run_dir,
                args.persistent_output_dir / slug if args.persistent_output_dir else None,
                label=f"{mode}-diagnostic",
            )
            continue
        validation_config = write_terramind_sen1floods11_config(
            config_dir / "predict_validation.yaml",
            **common,
            prediction_split="validation",
            probability_output_dir=validation_output,
        )
        test_config = write_terramind_sen1floods11_config(
            config_dir / "predict_test.yaml",
            **common,
            prediction_split="test",
            probability_output_dir=test_output,
        )
        checkpoint = _fit_if_needed(
            fit_config,
            run_dir,
            backbone_checkpoint_sha256=backbone_checkpoint_sha256,
            dry_run=args.dry_run,
        )
        persist_output(run_dir, args.persistent_output_dir / slug if args.persistent_output_dir else None, label=f"{mode}-fit")
        _predict_if_needed(
            validation_config,
            checkpoint,
            validation_output,
            expected=_split_count(terratorch_splits["validation"]),
            dry_run=args.dry_run,
        )
        persist_output(
            validation_output,
            args.persistent_output_dir / slug / "probabilities" / "validation"
            if args.persistent_output_dir
            else None,
            label=f"{mode}-validation-predictions",
        )
        checkpoints[mode] = checkpoint
        validation_exports[mode] = validation_output
        test_exports[mode] = test_output
        # Ensure the test config is materialized and schema-checkable before any long run.
        assert test_config.exists()
    if args.smoke_only:
        diagnostic_manifest = args.output_dir / "diagnostic_panel_manifest.json"
        diagnostic_manifest.write_text(
            json.dumps(
                {
                    "schema": "geobwer.sen1floods11.terramind_panel_diagnostic.v1",
                    "formal_evidence": False,
                    "reason": "real_gpu_fast_dev_run",
                    "modes": list(modes),
                    "pretraining_checkpoint_sha256": backbone_checkpoint_sha256,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        persist_output(args.output_dir, args.persistent_output_dir, label="sen1-terramind-diagnostic-complete")
        print(f"[terramind:campaign] diagnostic complete: {diagnostic_manifest}")
        return
    if args.dry_run:
        persist_output(args.output_dir, args.persistent_output_dir, label="dry-run-configs")
        print(f"[terramind:campaign] dry-run configs ready under {args.output_dir}")
        return

    calibration_path = args.output_dir / "common_spatial_block_calibration.json"
    calibrate_common_sen1_spatial_blocks(
        validation_exports,
        calibration_path,
        metadata_csv=args.metadata_csv,
        n_simulations=args.calibration_simulations,
        n_bootstrap=args.calibration_bootstrap,
        seed=args.seed,
        minimum_moderate_tail_power=args.minimum_moderate_tail_power,
    )
    persist_output(args.output_dir, args.persistent_output_dir, label="spatial-calibration")
    terratorch_version = importlib.metadata.version("terratorch")
    panel_tables: dict[str, Path] = {}
    panel_protocol: BWERProtocol | None = None
    for mode in modes:
        slug = mode.lower().replace("+", "_plus_")
        run_dir = args.output_dir / slug
        test_config = run_dir / "configs" / "predict_test.yaml"
        _predict_if_needed(
            test_config,
            checkpoints[mode],
            test_exports[mode],
            expected=_split_count(terratorch_splits["test"]),
            dry_run=False,
        )
        persist_output(
            test_exports[mode],
            args.persistent_output_dir / slug / "probabilities" / "test"
            if args.persistent_output_dir
            else None,
            label=f"{mode}-test-predictions",
        )
        bundle = finalize_sen1floods11_segmentation(
            test_exports[mode],
            run_dir / "formal_outputs",
            model_name=f"terramind_v1_base_{slug}",
            checkpoint_path=checkpoints[mode],
            pretraining_checkpoint_path=args.checkpoint,
            pretraining_checkpoint_sha256=backbone_checkpoint_sha256,
            protocol_path=args.protocol,
            block_calibration_path=calibration_path,
            metadata_csv=args.metadata_csv,
            split="test",
            sensor_mode=mode,
            terratorch_version=terratorch_version,
        )
        calibration_rows, calibration_probabilities, calibration_targets, calibration_valid = load_sen1_probability_units(
            validation_exports[mode], metadata_csv=args.metadata_csv
        )
        formal_manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
        resolved_protocol = BWERProtocol.from_mapping(formal_manifest["protocol"])
        panel_tables[f"terramind_v1_base_{slug}"] = bundle.audit_table
        panel_protocol = resolved_protocol
        run_segmentation_uncertainty_suite(
            calibration_probabilities,
            calibration_targets,
            bundle.output_dir,
            run_dir / "uncertainty_extensions",
            protocol=resolved_protocol,
            group_columns=("event_id",),
            calibration_valid_masks=calibration_valid,
            calibration_sample_ids=[str(row["sample_id"]) for row in calibration_rows],
            crc_alpha=args.crc_alpha,
            n_bootstrap=args.audit_bootstrap,
            seed=args.seed,
        )
        persist_output(run_dir, args.persistent_output_dir / slug if args.persistent_output_dir else None, label=f"{mode}-formal")
    if set(modes) == set(MODES):
        if panel_protocol is None:
            raise RuntimeError("Internal error: the Sen1 model-panel protocol was not resolved.")
        run_geobwer_model_panel(
            panel_tables,
            args.output_dir / "model_panel",
            protocol=panel_protocol,
            group_column="event_id",
            cluster_column="spatial_block_id",
            n_bootstrap=args.audit_bootstrap,
            seed=args.seed,
        )
    persist_output(args.output_dir, args.persistent_output_dir, label="campaign-complete")
    print(f"[terramind:campaign] complete: {args.output_dir}")


if __name__ == "__main__":
    main()
