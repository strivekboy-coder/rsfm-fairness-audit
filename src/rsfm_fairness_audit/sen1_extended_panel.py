from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.geobwer_extensions import run_segmentation_uncertainty_suite
from rsfm_fairness_audit.geobwer_panel import run_geobwer_model_panel
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output
from rsfm_fairness_audit.sen1floods11_formal import (
    finalize_sen1_probability_export,
    load_sen1_probability_units,
)
from rsfm_fairness_audit.spatial_conformal import SpatialConformalConfig


class Sen1ExtendedPanelError(RuntimeError):
    """Raised when the frozen Sen1 cross-model panel is incomplete."""


@dataclass(frozen=True)
class Sen1ExtendedPanelConfig:
    terramind_root: Path
    supervised_root: Path
    prithvi_root: Path
    output_dir: Path
    protocol_path: Path
    metadata_csv: Path
    persistent_output_dir: Path | None = None
    audit_bootstrap: int = 2000
    crc_alpha: float = 0.10
    seeds: tuple[int, ...] = (42, 73, 101)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Sen1ExtendedPanelError(f"Required manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Sen1ExtendedPanelError(f"Manifest is not a JSON object: {path}")
    return value


def _mode_slug(mode: str) -> str:
    return str(mode).lower().replace("+", "_plus_")


def _finalize_one(
    *,
    model_name: str,
    validation_export: Path,
    test_export: Path,
    output_dir: Path,
    protocol_path: Path,
    block_calibration_path: Path,
    metadata_csv: Path,
    model_lineage: Mapping[str, Any],
    dataset_lineage: Mapping[str, Any],
    crc_alpha: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[Path, BWERProtocol]:
    for path in (validation_export, test_export):
        if not path.is_dir():
            raise Sen1ExtendedPanelError(f"Probability export is missing: {path}")
    bundle = finalize_sen1_probability_export(
        test_export,
        output_dir / "formal_outputs",
        model_name=model_name,
        protocol_path=protocol_path,
        block_calibration_path=block_calibration_path,
        model_lineage=model_lineage,
        dataset_lineage=dataset_lineage,
        metadata_csv=metadata_csv,
        split="test",
    )
    rows, probabilities, targets, valid = load_sen1_probability_units(
        validation_export, metadata_csv=metadata_csv
    )
    manifest = _json(bundle.manifest)
    protocol = BWERProtocol.from_mapping(manifest["protocol"])
    run_segmentation_uncertainty_suite(
        probabilities,
        targets,
        bundle.output_dir,
        output_dir / "uncertainty_extensions",
        protocol=protocol,
        group_columns=("event_id",),
        calibration_valid_masks=valid,
        calibration_sample_ids=[str(row["sample_id"]) for row in rows],
        calibration_sample_rows=rows,
        crc_alpha=crc_alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
        spatial_localization_config=SpatialConformalConfig(),
    )
    return bundle.audit_table, protocol


def _terramind_tables(
    root: Path,
    seeds: tuple[int, ...],
) -> tuple[dict[str, Path], BWERProtocol | None]:
    tables: dict[str, Path] = {}
    protocol: BWERProtocol | None = None
    for mode in ("S1", "S2", "S1+S2"):
        slug = _mode_slug(mode)
        for seed in seeds:
            run_name = f"terramind_v1_base_{slug}_seed_{seed}"
            formal = root / slug / f"seed_{seed}" / "formal_outputs"
            manifest = _json(formal / "formal_output_manifest.json")
            observed_name = str(manifest.get("model_lineage", {}).get("model", ""))
            if observed_name != run_name:
                raise Sen1ExtendedPanelError(
                    f"TerraMind model identity drift: expected={run_name}, observed={observed_name!r}."
                )
            table = formal / "formal_audit_table.csv"
            if not table.is_file():
                raise Sen1ExtendedPanelError(f"Missing TerraMind formal table: {table}")
            tables[run_name] = table
            current = BWERProtocol.from_mapping(manifest["protocol"])
            if protocol is not None and current.signature != protocol.signature:
                raise Sen1ExtendedPanelError("TerraMind formal outputs do not share one protocol hash.")
            protocol = current
    return tables, protocol


def run_sen1_extended_panel(config: Sen1ExtendedPanelConfig) -> dict[str, Path]:
    """Finalize supervised/Prithvi outputs and build the frozen common-unit panel."""

    hydrate_output(config.output_dir, config.persistent_output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = config.terramind_root / "common_spatial_block_calibration.json"
    calibration = _json(calibration_path)
    calibrated_models = set(map(str, calibration.get("models", {})))

    tables, panel_protocol = _terramind_tables(config.terramind_root, config.seeds)
    required_calibration_models = set(tables)
    completed: dict[str, dict[str, Any]] = {}

    for mode in ("S1", "S2", "S1+S2"):
        slug = _mode_slug(mode)
        for seed in config.seeds:
            model_name = f"resnet34_unet_{slug}_seed_{seed}"
            required_calibration_models.add(model_name)
            run_root = config.supervised_root / slug / f"seed_{seed}"
            source_manifest = _json(run_root / "run_manifest.json")
            if (
                str(source_manifest.get("sensor_mode")) != mode
                or int(source_manifest.get("seed", -1)) != seed
            ):
                raise Sen1ExtendedPanelError(f"Supervised run identity drift at {run_root}.")
            checkpoint = run_root / "best_resnet34_unet.pt"
            table, current_protocol = _finalize_one(
                model_name=model_name,
                validation_export=run_root / "probabilities" / "validation",
                test_export=run_root / "probabilities" / "test",
                output_dir=config.output_dir / model_name,
                protocol_path=config.protocol_path,
                block_calibration_path=calibration_path,
                metadata_csv=config.metadata_csv,
                model_lineage={
                    "model": model_name,
                    "architecture": "resnet34_unet",
                    "pretrained_encoder": bool(
                        source_manifest.get("adaptation_protocol", "").startswith(
                            "supervised_from_scratch_decoder_imagenet"
                        )
                    ),
                    "adaptation_protocol": source_manifest.get("adaptation_protocol"),
                    "sensor_mode": mode,
                    "input_channels": source_manifest.get("input_channels"),
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "selection_data": "official_validation",
                    "full_probability_export": True,
                },
                dataset_lineage={
                    "dataset": "Sen1Floods11-v1.1-HandLabeled",
                    "split_protocol": "official_252_89_90_membership",
                    "split": "test",
                },
                crc_alpha=config.crc_alpha,
                n_bootstrap=config.audit_bootstrap,
                seed=seed,
            )
            if panel_protocol is not None and current_protocol.signature != panel_protocol.signature:
                raise Sen1ExtendedPanelError("Supervised output protocol differs from TerraMind.")
            panel_protocol = current_protocol
            tables[model_name] = table
            completed[model_name] = {"role": "protocol_matched_supervised_baseline"}

    prithvi_manifest = _json(config.prithvi_root / "campaign_manifest.json")
    prithvi_name = str(prithvi_manifest.get("model") or "prithvi_tl_sen1floods11")
    required_calibration_models.add(prithvi_name)
    prithvi_table, current_protocol = _finalize_one(
        model_name=prithvi_name,
        validation_export=config.prithvi_root / "probabilities" / "validation",
        test_export=config.prithvi_root / "probabilities" / "test",
        output_dir=config.output_dir / prithvi_name,
        protocol_path=config.protocol_path,
        block_calibration_path=calibration_path,
        metadata_csv=config.metadata_csv,
        model_lineage={
            "model": prithvi_name,
            "architecture": "prithvi_tl",
            "adaptation_protocol": prithvi_manifest.get("adaptation_protocol"),
            "sensor_mode": "S2",
            "role": "task_specific_external_validity_reference",
            "checkpoint_path": prithvi_manifest.get("checkpoint_path"),
            "checkpoint_sha256": prithvi_manifest.get("checkpoint_sha256"),
            "model_load_diagnostics": prithvi_manifest.get("model_load_diagnostics"),
            "full_probability_export": True,
        },
        dataset_lineage={
            "dataset": "Sen1Floods11-v1.1-HandLabeled",
            "split_protocol": "official_252_89_90_membership",
            "split": "test",
        },
        crc_alpha=config.crc_alpha,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seeds[0],
    )
    if panel_protocol is not None and current_protocol.signature != panel_protocol.signature:
        raise Sen1ExtendedPanelError("Prithvi output protocol differs from TerraMind.")
    panel_protocol = current_protocol
    tables[prithvi_name] = prithvi_table
    completed[prithvi_name] = {"role": "task_specific_external_validity_reference"}

    missing_calibration = sorted(required_calibration_models - calibrated_models)
    if missing_calibration:
        raise Sen1ExtendedPanelError(
            "The common spatial scale was not calibrated against every formal model: "
            + ", ".join(missing_calibration)
        )
    if panel_protocol is None:
        raise Sen1ExtendedPanelError("No formal protocol could be resolved.")

    primary_pairs: list[tuple[str, str]] = []
    secondary_pairs: list[tuple[str, str]] = []
    external_pairs: list[tuple[str, str]] = []
    for seed in config.seeds:
        for mode in ("s1", "s2", "s1_plus_s2"):
            primary_pairs.append(
                (
                    f"terramind_v1_base_{mode}_seed_{seed}",
                    f"resnet34_unet_{mode}_seed_{seed}",
                )
            )
        for family in ("terramind_v1_base", "resnet34_unet"):
            secondary_pairs.extend(
                [
                    (
                        f"{family}_s1_seed_{seed}",
                        f"{family}_s1_plus_s2_seed_{seed}",
                    ),
                    (
                        f"{family}_s2_seed_{seed}",
                        f"{family}_s1_plus_s2_seed_{seed}",
                    ),
                ]
            )
        external_pairs.append((prithvi_name, f"terramind_v1_base_s2_seed_{seed}"))
    all_pairs = tuple(primary_pairs + secondary_pairs + external_pairs)
    panel = run_geobwer_model_panel(
        tables,
        config.output_dir / "model_panel",
        protocol=panel_protocol,
        group_column="event_id",
        cluster_column="spatial_block_id",
        comparison_pairs=all_pairs,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seeds[0],
    )
    comparison_design = config.output_dir / "comparison_design.json"
    comparison_design.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.comparison_design.v1",
                "primary_same_protocol_pairs": primary_pairs,
                "secondary_sensor_modality_pairs": secondary_pairs,
                "external_validity_pairs": external_pairs,
                "prithvi_role": "task_specific_external_validity_reference_not_training_budget_matched",
                "multiplicity_family": "all_pre_registered_pairs_jointly",
                "common_spatial_calibration_sha256": file_sha256(calibration_path),
                "calibrated_models": sorted(calibrated_models),
                "completed_models": completed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = config.output_dir / "campaign_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.extended_panel.v1",
                "models": sorted(tables),
                "seeds": list(config.seeds),
                "protocol_hash": panel_protocol.signature,
                "comparison_design": str(comparison_design),
                "model_panel_summary": str(panel.model_summary),
                "paired_comparisons": str(panel.paired_comparisons),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    persist_output(config.output_dir, config.persistent_output_dir, label="sen1-extended-panel-complete")
    return {
        "manifest": manifest,
        "comparison_design": comparison_design,
        "model_panel_summary": panel.model_summary,
        "paired_comparisons": panel.paired_comparisons,
    }


__all__ = [
    "Sen1ExtendedPanelConfig",
    "Sen1ExtendedPanelError",
    "run_sen1_extended_panel",
]
