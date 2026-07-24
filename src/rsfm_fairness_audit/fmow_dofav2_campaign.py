from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.dofa import DOFAAdapter
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.fmow_sentinel_classification import (
    FmowClassificationConfig,
    _cached_dofa_embeddings,
    _limit_rows,
    _load_metadata,
    _row_hash,
    _split_rows,
)
from rsfm_fairness_audit.fmow_formal_split import fmow_site_id
from rsfm_fairness_audit.formal_outputs import FormalOutputBundle, file_sha256, write_multiclass_bundle
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.geobwer_extensions import run_multiclass_uncertainty_suite
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output
from rsfm_fairness_audit.probe_selection import (
    MulticlassProbeSearchConfig,
    fit_selected_multiclass_probe,
)


class FmowDOFAv2CampaignError(RuntimeError):
    """Raised when the final train/calibrate/test fMoW chain is invalid."""


_FROZEN_DOFA_CONFIG: dict[str, Any] = {
    "model_variant": "dofav2_vit_base",
    "model_release": "dofav2_vit_base_e150",
    "repo_revision": DOFAAdapter.official_dofav2_repo_revision,
    "architecture_source_repo": DOFAAdapter.official_dofav2_architecture_repo,
    "architecture_source_revision": DOFAAdapter.official_dofav2_architecture_revision,
    "required_timm_version": DOFAAdapter.official_dofav2_timm_version,
    "checkpoint_sha256": DOFAAdapter.official_dofav2_checkpoint_sha256,
    "require_exact_checkpoint_match": True,
    "band_profile": "sentinel2_9_legacy",
    "image_size": 224,
    "input_scale": 39.21568627450981,
    "embedding_layer": "forward_features",
    "embedding_pooling": "mean_tokens",
    "allow_torch_hub_download": False,
}


@dataclass(frozen=True)
class FmowDOFAv2CampaignConfig:
    metadata_csv: Path
    data_root: Path
    model_config: Path
    output_dir: Path
    persistent_output_dir: Path | None = None
    geobwer_protocol: Path = Path("configs/geobwer/fmow_sentinel.yaml")
    model_repo_path: Path | None = None
    model_checkpoint_path: Path | None = None
    train_split: str = "train"
    calibration_split: str = "calibration"
    test_split: str = "test"
    split_protocol: str = "location_disjoint"
    image_size: int = 224
    band_profile: str = "sentinel2_9_legacy"
    batch_size: int = 16
    probe_epochs: int = 200
    probe_learning_rate: float = 1e-3
    probe_learning_rates: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    probe_patience: int = 20
    probe_inner_validation_fraction: float = 0.15
    probe_batch_size: int = 512
    weight_decay: float = 1e-4
    device: str = "auto"
    max_samples_per_split: int | None = None
    diagnostic_only: bool = False
    audit_bootstrap: int = 2000
    conformal_alpha: float = 0.10
    seed: int = 42
    seeds: tuple[int, ...] = (42, 73, 101)

    def __post_init__(self) -> None:
        if self.max_samples_per_split is not None and not self.diagnostic_only:
            raise ValueError(
                "The final DOFAv2 campaign does not permit split subsampling; set diagnostic_only=True in a separate output path."
            )
        if self.diagnostic_only and self.max_samples_per_split is None:
            raise ValueError("diagnostic_only requires max_samples_per_split.")
        if self.batch_size <= 0 or self.probe_epochs <= 0:
            raise ValueError("batch_size and probe_epochs must be positive.")
        if not self.seeds or len(set(map(int, self.seeds))) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique.")
        if not self.diagnostic_only and len(self.seeds) < 3:
            raise ValueError("Formal DOFAv2 probe inference requires at least three training seeds.")


def _validate_frozen_model_config(values: Mapping[str, Any]) -> None:
    """Refuse silent deviations from the preregistered DOFAv2 pipeline."""

    mismatches: list[str] = []
    for key, expected in _FROZEN_DOFA_CONFIG.items():
        observed = values.get(key)
        if isinstance(expected, float):
            try:
                matches = bool(np.isclose(float(observed), expected, rtol=0.0, atol=1e-12))
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected
        if not matches:
            mismatches.append(f"{key}: expected={expected!r}, observed={observed!r}")
    try:
        coverage = float(values.get("minimum_checkpoint_key_coverage"))
    except (TypeError, ValueError):
        coverage = float("nan")
    if not np.isclose(coverage, 0.90, rtol=0.0, atol=1e-12):
        mismatches.append(
            "minimum_checkpoint_key_coverage: expected=0.9, "
            f"observed={values.get('minimum_checkpoint_key_coverage')!r}"
        )
    if mismatches:
        raise FmowDOFAv2CampaignError(
            "Formal DOFAv2 protocol differs from the frozen configuration: " + "; ".join(mismatches)
        )


def _site_set(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    try:
        return {fmow_site_id(row) for row in rows}
    except RuntimeError as exc:
        raise FmowDOFAv2CampaignError(str(exc)) from exc


def _validate_split_contract(
    train_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> None:
    if not train_rows or not calibration_rows or not test_rows:
        raise FmowDOFAv2CampaignError("Train, calibration, and test splits must all be non-empty.")
    train_sites = _site_set(train_rows)
    calibration_sites = _site_set(calibration_rows)
    test_sites = _site_set(test_rows)
    if train_sites & calibration_sites or train_sites & test_sites or calibration_sites & test_sites:
        raise FmowDOFAv2CampaignError("fMoW category-scoped site leakage exists across train/calibration/test.")
    sample_sets = [
        {str(row.get("sample_id", "")).strip() for row in split_rows}
        for split_rows in (train_rows, calibration_rows, test_rows)
    ]
    if any("" in values for values in sample_sets) or sample_sets[0] & sample_sets[1] or sample_sets[0] & sample_sets[2] or sample_sets[1] & sample_sets[2]:
        raise FmowDOFAv2CampaignError("fMoW sample IDs must be non-empty and disjoint across splits.")


def _formal_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        sample_id = str(row.get("sample_id", "")).strip()
        country = str(row.get("country", "")).strip()
        category = str(row.get("category", "")).strip()
        region = str(row.get("region") or row.get("un_region") or row.get("continent") or "").strip()
        if not sample_id or not country or not category:
            raise FmowDOFAv2CampaignError(
                "Every formal fMoW row requires sample_id, verified country, and category."
            )
        row.update(
            {
                "sample_id": sample_id,
                "independent_unit_id": sample_id,
                "location_id": str(row.get("location_id", "")).strip(),
                "site_id": fmow_site_id(row),
                "country": country,
                "class_label": category,
                "region": region,
                "country_class": f"{country}|{category}" if country and category else "",
                "region_class": f"{region}|{category}" if region and category else "",
            }
        )
        output.append(row)
    return output


def run_fmow_dofav2_campaign(config: FmowDOFAv2CampaignConfig) -> dict[str, Path]:
    hydrate_output(config.output_dir, config.persistent_output_dir)
    output = ensure_dir(config.output_dir)
    rows = _load_metadata(config.metadata_csv)
    limit = config.max_samples_per_split
    train_rows = _limit_rows(_split_rows(rows, config.train_split), limit, config.seed)
    calibration_rows = _limit_rows(_split_rows(rows, config.calibration_split), limit, config.seed + 1)
    test_rows = _limit_rows(_split_rows(rows, config.test_split), limit, config.seed + 2)
    _validate_split_contract(train_rows, calibration_rows, test_rows)

    model_config_values = load_yaml(config.model_config)
    model_config_values.update(
        {
            "device": config.device,
            "batch_size": config.batch_size,
            "band_profile": config.band_profile,
            "image_size": config.image_size,
        }
    )
    if config.model_repo_path is not None:
        model_config_values["repo_path"] = str(config.model_repo_path)
    if config.model_checkpoint_path is not None:
        model_config_values["checkpoint_path"] = str(config.model_checkpoint_path)
    _validate_frozen_model_config(model_config_values)
    adapter = DOFAAdapter.from_config(model_config_values)
    if adapter.repo_path is None or adapter.checkpoint_path is None:
        raise FmowDOFAv2CampaignError(
            "The final DOFAv2 campaign requires explicit official repo and checkpoint paths."
        )
    resolved_model_config = output / "resolved_dofav2_model_config.json"
    resolved_model_config.write_text(
        json.dumps(model_config_values, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if not adapter.model_release.startswith("dofav2"):
        raise FmowDOFAv2CampaignError("The final campaign requires a verified DOFAv2 model release.")
    adapter.load_model()
    runner_config = FmowClassificationConfig(
        metadata_csv=config.metadata_csv,
        data_root=config.data_root,
        output_dir=output,
        model="dofav2",
        model_config=config.model_config,
        probe="linear",
        probe_epochs=config.probe_epochs,
        probe_learning_rate=config.probe_learning_rate,
        train_split=config.train_split,
        eval_split=config.test_split,
        image_size=config.image_size,
        band_profile=config.band_profile,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        device=config.device,
        split_protocol=config.split_protocol,
        seed=config.seed,
    )
    train_x, train_y, train_ok, train_warnings, train_cache, train_cache_meta = _cached_dofa_embeddings(
        train_rows, config.train_split, runner_config, adapter, output
    )
    calibration_x, calibration_y, calibration_ok, calibration_warnings, calibration_cache, calibration_cache_meta = _cached_dofa_embeddings(
        calibration_rows, config.calibration_split, runner_config, adapter, output
    )
    test_x, test_y, test_ok, test_warnings, test_cache, test_cache_meta = _cached_dofa_embeddings(
        test_rows, config.test_split, runner_config, adapter, output
    )
    persist_output(output, config.persistent_output_dir, label="dofav2-embeddings-complete")
    if train_warnings or calibration_warnings or test_warnings:
        warning_path = output / "readability_warnings.json"
        warning_path.write_text(
            json.dumps(
                {
                    "train": train_warnings,
                    "calibration": calibration_warnings,
                    "test": test_warnings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if len(train_ok) != len(train_rows) or len(calibration_ok) != len(calibration_rows) or len(test_ok) != len(test_rows):
        raise FmowDOFAv2CampaignError(
            "Formal fMoW campaign refuses unreadable-row dropping; repair paths/prepared data before inference."
        )
    if config.diagnostic_only:
        embedding_dimensions = {
            "train": list(train_x.shape),
            "calibration": list(calibration_x.shape),
            "test": list(test_x.shape),
        }
        if any(len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0 for shape in embedding_dimensions.values()):
            raise FmowDOFAv2CampaignError(f"Invalid diagnostic embedding shapes: {embedding_dimensions}")
        if len({shape[1] for shape in embedding_dimensions.values()}) != 1:
            raise FmowDOFAv2CampaignError(f"Embedding dimensions differ across splits: {embedding_dimensions}")
        if not all(np.all(np.isfinite(values)) for values in (train_x, calibration_x, test_x)):
            raise FmowDOFAv2CampaignError("Diagnostic embeddings contain NaN or infinity.")
        diagnostic = output / "diagnostic_manifest.json"
        diagnostic.write_text(
            json.dumps(
                {
                    "schema": "geobwer.fmow.dofav2_diagnostic.v1",
                    "formal_evidence": False,
                    "reason": "explicit_bounded_real_gpu_smoke",
                    "max_samples_per_split": config.max_samples_per_split,
                    "embedding_shapes": embedding_dimensions,
                    "checkpoint_sha256": adapter.actual_checkpoint_sha256,
                    "repo_revision": adapter.actual_repo_revision,
                    "checkpoint_load_report": adapter.checkpoint_load_report,
                    "architecture_source_repo": adapter.architecture_source_repo,
                    "architecture_source_revision": adapter.architecture_source_revision,
                    "timm_version": adapter.actual_timm_version,
                    "patch_size": adapter.official_dofav2_patch_size,
                    "embedding_semantics": adapter.official_dofav2_embedding_semantics,
                    "band_profile": config.band_profile,
                    "input_scale": adapter.input_scale,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        persist_output(output, config.persistent_output_dir, label="dofav2-diagnostic-complete")
        return {"diagnostic_manifest": diagnostic}
    search_config = MulticlassProbeSearchConfig(
        learning_rates=tuple(float(value) for value in config.probe_learning_rates),
        max_epochs=config.probe_epochs,
        patience=config.probe_patience,
        inner_validation_fraction=config.probe_inner_validation_fraction,
        batch_size=config.probe_batch_size,
        weight_decay=config.weight_decay,
    )
    train_groups = [fmow_site_id(row) for row in train_ok]
    per_seed: dict[int, dict[str, Any]] = {}
    calibration_components: list[np.ndarray] = []
    test_components: list[np.ndarray] = []
    seed_summary_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_output = output / "probe_seeds" / f"seed_{int(seed)}"
        selected = fit_selected_multiclass_probe(
            train_x,
            train_y,
            train_groups,
            {"calibration": calibration_x, "test": test_x},
            seed_output,
            config=search_config,
            seed=int(seed),
            device=config.device,
        )
        per_seed[int(seed)] = selected
        calibration_component = np.asarray(
            selected["predictions"]["calibration"]["probabilities"], dtype=np.float32
        )
        test_component = np.asarray(
            selected["predictions"]["test"]["probabilities"], dtype=np.float32
        )
        calibration_components.append(calibration_component)
        test_components.append(test_component)
        class_to_index_seed = selected["class_to_index"]
        test_index = np.asarray(
            [class_to_index_seed[str(label)] for label in test_y], dtype=np.int64
        )
        predicted = np.argmax(test_component, axis=1)
        seed_summary_rows.append(
            {
                "seed": int(seed),
                "selected_learning_rate": selected["selection"]["selected_learning_rate"],
                "selected_epoch": selected["selection"]["selected_epoch"],
                "inner_validation_cross_entropy": min(
                    float(row["inner_validation_cross_entropy"])
                    for row in selected["selection"]["candidates"]
                ),
                "test_accuracy": float(np.mean(predicted == test_index)),
                "test_log_loss": float(
                    -np.mean(
                        np.log(
                            np.clip(
                                test_component[np.arange(len(test_index)), test_index],
                                1e-12,
                                1.0,
                            )
                        )
                    )
                ),
                "checkpoint": str(selected["checkpoint"]),
                "checkpoint_sha256": file_sha256(selected["checkpoint"]),
            }
        )
    class_name_sets = {tuple(value["classes"]) for value in per_seed.values()}
    if len(class_name_sets) != 1:
        raise FmowDOFAv2CampaignError("Probe seeds produced inconsistent class mappings.")
    class_names = list(next(iter(class_name_sets)))
    calibration_probabilities = np.mean(
        np.stack(calibration_components, axis=0), axis=0
    ).astype(np.float32)
    test_probabilities = np.mean(np.stack(test_components, axis=0), axis=0).astype(
        np.float32
    )
    probe_checkpoint = output / "probe_panel_manifest.json"
    probe_checkpoint.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.dofav2_probe_panel.v2",
                "estimand": "seed_ensemble_predictive_distribution",
                "seed_robustness_required": True,
                "seeds": list(config.seeds),
                "search_config": asdict(search_config),
                "components": {
                    str(seed): {
                        "checkpoint": str(value["checkpoint"]),
                        "checkpoint_sha256": file_sha256(value["checkpoint"]),
                        "selection_manifest": str(value["manifest"]),
                        "selection_manifest_sha256": file_sha256(value["manifest"]),
                    }
                    for seed, value in per_seed.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(output / "probe_seed_robustness.csv", seed_summary_rows)
    probe_metadata = {
        "probe": "linear_multiseed_selected",
        "checkpoint_path": str(probe_checkpoint),
        "seeds": list(config.seeds),
        "class_to_index": per_seed[int(config.seeds[0])]["class_to_index"],
    }
    probe_debug = {
        "schema": "geobwer.fmow.dofav2_probe_debug.v2",
        "search_data": "outer_train_only_category_scoped_site_disjoint_inner_holdout",
        "seed_summary": seed_summary_rows,
    }
    persist_output(output, config.persistent_output_dir, label="dofav2-probe-complete")
    if len(class_names) != 62:
        raise FmowDOFAv2CampaignError(f"Expected the frozen 62-class fMoW mapping, got {len(class_names)} classes.")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    if set(calibration_y) - set(class_to_index) or set(test_y) - set(class_to_index):
        raise FmowDOFAv2CampaignError("Calibration/test labels are absent from the train-frozen class mapping.")
    calibration_targets = np.asarray([class_to_index[label] for label in calibration_y], dtype=np.int64)
    test_targets = np.asarray([class_to_index[label] for label in test_y], dtype=np.int64)

    protocol = BWERProtocol.from_mapping(load_yaml(config.geobwer_protocol))
    probe_checkpoint = Path(str(probe_metadata["checkpoint_path"]))
    model_lineage = {
        "model": adapter.model_release,
        "model_variant": adapter.model_variant,
        "official_repo_path": str(adapter.repo_path),
        "official_repo_revision": adapter.actual_repo_revision,
        "architecture_source_repo": adapter.architecture_source_repo,
        "architecture_source_revision": adapter.architecture_source_revision,
        "timm_version": adapter.actual_timm_version,
        "patch_size": adapter.official_dofav2_patch_size,
        "backbone_checkpoint": str(adapter.checkpoint_path),
        "backbone_checkpoint_sha256": adapter.actual_checkpoint_sha256,
        "checkpoint_load_report": adapter.checkpoint_load_report,
        "probe_checkpoint": str(probe_checkpoint),
        "probe_checkpoint_sha256": file_sha256(probe_checkpoint),
        "adaptation_protocol": "frozen_encoder_train_only_selected_multiseed_linear_probe",
        "probe_estimand": "seed_ensemble_predictive_distribution_with_seed_robustness",
        "probe_seeds": list(config.seeds),
        "probe_model_selection": "outer_train_only_category_scoped_site_disjoint_inner_holdout",
        "band_profile": config.band_profile,
        "image_size": config.image_size,
        "input_scale": adapter.input_scale,
        "embedding_pooling": adapter.embedding_pooling,
        "embedding_semantics": adapter.official_dofav2_embedding_semantics,
        "resolved_model_config": str(resolved_model_config),
        "resolved_model_config_sha256": file_sha256(resolved_model_config),
    }
    dataset_lineage = {
        "dataset": "fMoW-Sentinel",
        "metadata_sha256": file_sha256(config.metadata_csv),
        "split_protocol": config.split_protocol,
        "train_split": config.train_split,
        "calibration_split": config.calibration_split,
        "test_split": config.test_split,
        "train_row_hash": _row_hash(train_ok),
        "calibration_row_hash": _row_hash(calibration_ok),
        "test_row_hash": _row_hash(test_ok),
    }
    seed_geobwer_artifacts: dict[str, dict[str, str]] = {}
    for seed, selected in per_seed.items():
        seed_dir = output / "probe_seeds" / f"seed_{seed}"
        seed_probabilities = np.asarray(
            selected["predictions"]["test"]["probabilities"], dtype=np.float32
        )
        seed_bundle = write_multiclass_bundle(
            seed_dir / "formal_outputs",
            sample_rows=_formal_rows(test_ok),
            probabilities=seed_probabilities,
            targets=test_targets,
            class_names=class_names,
            dataset="fmow_sentinel",
            model=f"{adapter.model_release}_seed_{seed}",
            split=config.test_split,
            protocol=protocol,
            model_lineage={
                **model_lineage,
                "model": f"{adapter.model_release}_seed_{seed}",
                "probe_estimand": "single_training_seed",
                "probe_seed": seed,
                "probe_checkpoint": str(selected["checkpoint"]),
                "probe_checkpoint_sha256": file_sha256(selected["checkpoint"]),
                "probe_selection_manifest": str(selected["manifest"]),
                "probe_selection_manifest_sha256": file_sha256(selected["manifest"]),
            },
            dataset_lineage=dataset_lineage,
            independent_unit_column="sample_id",
            split_role="evaluation",
        )
        seed_rows = read_csv_rows(seed_bundle.audit_table)
        seed_audit = audit_rows(
            seed_rows,
            group_columns=("country", "region", "class_label"),
            protocol=protocol,
            loss_column="risk",
            unit_column="independent_unit_id",
            cluster_column="site_id",
            formal=True,
            require_probabilities=True,
            n_bootstrap=config.audit_bootstrap,
            seed=seed,
        ).to_report(seed_dir / "geobwer_raw")
        seed_geobwer_artifacts[str(seed)] = {
            key: str(value) for key, value in seed_audit.items()
        }
        country_summary = next(
            (
                row
                for row in read_csv_rows(seed_audit["summary"])
                if str(row.get("axis")) == "country"
            ),
            {},
        )
        for row in seed_summary_rows:
            if int(row["seed"]) == int(seed):
                row["country_geobwer"] = country_summary.get("geobwer", "")
                row["country_geobwer_lcb"] = country_summary.get("geobwer_lcb", "")
                row["country_geobwer_ucb"] = country_summary.get("geobwer_ucb", "")
                row["country_geobwer_validity"] = country_summary.get("validity", "")
    write_csv(output / "probe_seed_robustness.csv", seed_summary_rows)
    bundle: FormalOutputBundle = write_multiclass_bundle(
        output / "formal_outputs",
        sample_rows=_formal_rows(test_ok),
        probabilities=test_probabilities,
        targets=test_targets,
        class_names=class_names,
        dataset="fmow_sentinel",
        model=f"{adapter.model_release}_seed_ensemble",
        split=config.test_split,
        protocol=protocol,
        model_lineage={
            **model_lineage,
            "model": f"{adapter.model_release}_seed_ensemble",
        },
        dataset_lineage=dataset_lineage,
        independent_unit_column="sample_id",
        split_role="evaluation",
    )
    calibration_path = output / "calibration_probabilities.npz"
    np.savez_compressed(
        calibration_path,
        probabilities=calibration_probabilities,
        targets=calibration_targets,
        class_names=np.asarray(class_names, dtype=str),
        sample_id=np.asarray([row["sample_id"] for row in calibration_ok], dtype=str),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    calibration_manifest = output / "calibration_manifest.json"
    calibration_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.multiclass_calibration.v2",
                "split_role": "calibration",
                "test_rows_used": False,
                "probabilities_sha256": file_sha256(calibration_path),
                "sample_count": len(calibration_ok),
                "class_mapping": class_names,
                "class_mapping_source": "train_only_multiseed_selected_linear_probe",
                "model_selection_data": "outer_train_only",
                "seed_ensemble": list(config.seeds),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    formal_rows = read_csv_rows(bundle.audit_table)
    axes = tuple(
        column
        for column in ("country", "region", "class_label", "country_class", "region_class")
        if all(str(row.get(column, "")).strip() for row in formal_rows)
    )
    audit = audit_rows(
        formal_rows,
        group_columns=axes,
        protocol=protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="site_id",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seed,
    )
    audit_artifacts = audit.to_report(output / "geobwer_raw")
    standardized = audit_rows(
        formal_rows,
        group_columns=("country",),
        protocol=protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="site_id",
        balance_column="class_label",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seed,
    )
    standardized_artifacts = standardized.to_report(output / "geobwer_standardized")
    partial_protocol = replace(
        protocol,
        missingness_rule="partial_bounds",
        metadata=tuple(
            sorted(
                {
                    **dict(protocol.metadata),
                    "standardization_sensitivity": "partial_identification_bounds",
                }.items()
            )
        ),
    )
    partial_standardized = audit_rows(
        formal_rows,
        group_columns=("country",),
        protocol=partial_protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="site_id",
        balance_column="class_label",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seed,
    )
    partial_standardized_artifacts = partial_standardized.to_report(
        output / "geobwer_standardized_partial_bounds"
    )
    uncertainty_artifacts = run_multiclass_uncertainty_suite(
        calibration_path,
        bundle.output_dir,
        output / "uncertainty_extensions",
        protocol=protocol,
        group_columns=axes,
        calibration_manifest=calibration_manifest,
        alpha=config.conformal_alpha,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seed,
    )
    run_manifest = output / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow_dofav2_campaign.v2",
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "model_lineage": model_lineage,
                "dataset_lineage": dataset_lineage,
                "embedding_caches": {
                    "train": {"path": str(train_cache), **train_cache_meta},
                    "calibration": {"path": str(calibration_cache), **calibration_cache_meta},
                    "test": {"path": str(test_cache), **test_cache_meta},
                },
                "probe_debug": probe_debug,
                "formal_output_manifest": str(bundle.manifest),
                "calibration_manifest": str(calibration_manifest),
                "geobwer_artifacts": {key: str(value) for key, value in audit_artifacts.items()},
                "standardized_artifacts": {key: str(value) for key, value in standardized_artifacts.items()},
                "partial_standardized_artifacts": {
                    key: str(value) for key, value in partial_standardized_artifacts.items()
                },
                "seed_geobwer_artifacts": seed_geobwer_artifacts,
                "probe_seed_robustness": str(output / "probe_seed_robustness.csv"),
                "uncertainty_artifacts": {key: str(value) for key, value in uncertainty_artifacts.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    persist_output(output, config.persistent_output_dir, label="dofav2-formal-campaign-complete")
    return {
        "formal_audit_table": bundle.audit_table,
        "formal_output_manifest": bundle.manifest,
        "calibration_probabilities": calibration_path,
        "calibration_manifest": calibration_manifest,
        "probe_checkpoint": probe_checkpoint,
        "geobwer_summary": audit_artifacts["summary"],
        "standardized_summary": standardized_artifacts["summary"],
        "partial_standardized_summary": partial_standardized_artifacts["summary"],
        "probe_seed_robustness": output / "probe_seed_robustness.csv",
        "uncertainty_summary": uncertainty_artifacts["summary"],
        "run_manifest": run_manifest,
    }


__all__ = [
    "FmowDOFAv2CampaignConfig",
    "FmowDOFAv2CampaignError",
    "run_fmow_dofav2_campaign",
]
