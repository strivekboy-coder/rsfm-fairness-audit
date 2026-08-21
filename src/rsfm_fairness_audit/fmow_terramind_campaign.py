from __future__ import annotations

"""TerraMind x fMoW-Sentinel missing cell for Experiment 9.

The task/sample/RiskSpec contract matches the frozen fMoW campaign.  TerraMind
uses its native 12-band S2L2A input (B10 removed from the 13-band source), so
the result is explicitly a model-pipeline comparison rather than an isolated
causal effect of backbone identity.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.terramind import TerraMindAdapter
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.fmow_dofav2_campaign import (
    _coordinate_or_nan,
    _formal_rows,
    _validate_split_contract,
    _write_fmow_metadata_preflight,
)
from rsfm_fairness_audit.fmow_formal_split import fmow_site_id
from rsfm_fairness_audit.fmow_sentinel_classification import (
    _limit_rows,
    _load_metadata,
    _row_hash,
    _split_rows,
    load_fmow_sentinel_image,
)
from rsfm_fairness_audit.formal_outputs import file_sha256, write_multiclass_bundle
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output
from rsfm_fairness_audit.probe_selection import MulticlassProbeSearchConfig, fit_selected_multiclass_probe


class FmowTerraMindError(RuntimeError):
    pass


@dataclass(frozen=True)
class FmowTerraMindConfig:
    metadata_csv: Path
    data_root: Path
    terramind_checkpoint_path: Path
    output_dir: Path
    persistent_output_dir: Path | None = None
    embedding_cache_dir: Path | None = None
    geobwer_protocol: Path = Path("configs/geobwer/fmow_sentinel.yaml")
    train_split: str = "train"
    calibration_split: str = "calibration"
    test_split: str = "test"
    split_protocol: str = "location_disjoint"
    image_size: int = 224
    band_profile: str = "sentinel2_12_fmow_l2a"
    batch_size: int = 32
    embedding_chunk_size: int = 4096
    probe_epochs: int = 200
    probe_learning_rates: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    probe_patience: int = 20
    probe_inner_validation_fraction: float = 0.15
    probe_batch_size: int = 512
    weight_decay: float = 1e-4
    device: str = "auto"
    max_samples_per_split: int | None = None
    diagnostic_only: bool = False
    audit_bootstrap: int = 2000
    seeds: tuple[int, ...] = (42, 73, 101)

    def __post_init__(self) -> None:
        if self.band_profile != "sentinel2_12_fmow_l2a" or self.image_size != 224:
            raise ValueError("Formal TerraMind x fMoW is frozen to 12-band S2L2A at 224 px.")
        if self.max_samples_per_split is not None and not self.diagnostic_only:
            raise ValueError("Subsampling is diagnostic-only.")
        if self.diagnostic_only and self.max_samples_per_split is None:
            raise ValueError("diagnostic_only requires max_samples_per_split.")
        if not self.diagnostic_only and len(self.seeds) < 3:
            raise ValueError("Formal seed robustness requires at least three seeds.")


def _signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _extract_embeddings(
    rows: Sequence[Mapping[str, Any]], split: str, config: FmowTerraMindConfig,
    adapter: TerraMindAdapter, cache_root: Path,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]], Path]:
    # Verify and load the pinned model before accepting a cache signature. This
    # binds reuse to the observed checkpoint hash rather than only its path.
    adapter.load_model()
    split_dir = cache_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = split_dir / "embeddings.npy"
    labels_path = split_dir / "labels.json"
    metadata_path = split_dir / "metadata.jsonl"
    manifest_path = split_dir / "embedding_cache_manifest.json"
    lineage = {
        "schema": "geobwer.fmow.terramind_embedding_cache.v1",
        "adapter": {key: value for key, value in adapter.provenance().items() if key != "preprocessing_report"},
        "split": split,
        "row_hash": _row_hash(rows, config.data_root),
        "row_count": len(rows),
        "band_profile": config.band_profile,
        "image_size": config.image_size,
        "metadata_sha256": file_sha256(config.metadata_csv),
    }
    signature = _signature(lineage)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_signature") != signature:
            raise FmowTerraMindError(f"Cache protocol changed under {split_dir}; use a new output directory.")
        if embedding_path.is_file() and labels_path.is_file() and metadata_path.is_file():
            embeddings = np.load(embedding_path, mmap_mode="r")
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]
            if embeddings.shape[0] == len(rows) == len(labels) == len(metadata):
                print(f"[fmow:terramind] reuse embeddings split={split} shape={embeddings.shape}")
                return embeddings, [str(value) for value in labels], metadata, manifest_path
    chunks: list[np.ndarray] = []
    labels: list[str] = []
    metadata: list[dict[str, Any]] = []
    for start in range(0, len(rows), config.batch_size):
        selected = rows[start : start + config.batch_size]
        samples = []
        for row in selected:
            chip = load_fmow_sentinel_image(row, config.data_root, config.image_size, config.band_profile)
            samples.append({"image": {"S2": chip}})
        prepared = adapter.preprocess({"samples": samples, "metadata": selected})
        chunks.append(adapter.extract_embeddings(prepared))
        labels.extend(str(row["category"]) for row in selected)
        metadata.extend(dict(row) for row in selected)
        end = min(start + config.batch_size, len(rows))
        if end == len(rows) or end % config.embedding_chunk_size < config.batch_size:
            print(f"[fmow:terramind] split={split} embeddings={end}/{len(rows)}")
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    if embeddings.shape[0] != len(rows) or not np.all(np.isfinite(embeddings)):
        raise FmowTerraMindError("TerraMind embedding extraction is incomplete or non-finite.")
    np.save(embedding_path, embeddings)
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    metadata_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in metadata), encoding="utf-8")
    manifest_path.write_text(json.dumps({**lineage, "cache_signature": signature, "embedding_shape": list(embeddings.shape),
                                         "observed_preprocessing_report": adapter.provenance().get("preprocessing_report", {})}, ensure_ascii=False, indent=2), encoding="utf-8")
    return embeddings, labels, metadata, manifest_path


def _test_metrics(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    prediction = np.argmax(probabilities, axis=1)
    return float(np.mean(prediction == targets)), float(-np.mean(np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))))


def run_fmow_terramind_campaign(config: FmowTerraMindConfig) -> dict[str, Path]:
    hydrate_output(config.output_dir, config.persistent_output_dir)
    output = ensure_dir(config.output_dir)
    rows = _load_metadata(config.metadata_csv)
    preflight = _write_fmow_metadata_preflight(rows, output / "fmow_metadata_preflight.json",
                                                expected_splits=(config.train_split, config.calibration_split, config.test_split))
    if not config.diagnostic_only and not preflight["ok"]:
        raise FmowTerraMindError("fMoW metadata preflight failed: " + ", ".join(preflight["errors"]))
    limit = config.max_samples_per_split
    train_rows = _limit_rows(_split_rows(rows, config.train_split), limit, 42)
    calibration_rows = _limit_rows(_split_rows(rows, config.calibration_split), limit, 43)
    test_rows = _limit_rows(_split_rows(rows, config.test_split), limit, 44)
    _validate_split_contract(train_rows, calibration_rows, test_rows)
    adapter = TerraMindAdapter(sensor_mode="S2", input_profile="reben_l2a", model_name="terramind_v1_base",
                               model_release="terramind_v1_base_iccv2025", device=config.device,
                               image_size=config.image_size, merge_method="mean", embedding_pooling="mean_tokens",
                               pretrained=True, checkpoint_path=config.terramind_checkpoint_path, strict_range_check=True)
    cache_root = config.embedding_cache_dir or output / "embedding_cache"
    train_x, train_y, train_ok, train_manifest = _extract_embeddings(train_rows, config.train_split, config, adapter, cache_root)
    calibration_x, calibration_y, calibration_ok, calibration_manifest_cache = _extract_embeddings(calibration_rows, config.calibration_split, config, adapter, cache_root)
    test_x, test_y, test_ok, test_manifest = _extract_embeddings(test_rows, config.test_split, config, adapter, cache_root)
    persist_output(output, config.persistent_output_dir, label="terramind-embeddings-complete")
    if config.diagnostic_only:
        path = output / "diagnostic_manifest.json"
        path.write_text(json.dumps({"schema": "geobwer.fmow.terramind_diagnostic.v1", "formal_evidence": False,
                                    "embedding_shapes": {"train": list(train_x.shape), "calibration": list(calibration_x.shape), "test": list(test_x.shape)},
                                    "band_profile": config.band_profile, "pipeline_level_comparison": True}, indent=2), encoding="utf-8")
        persist_output(output, config.persistent_output_dir, label="terramind-diagnostic-complete")
        return {"diagnostic_manifest": path}
    search = MulticlassProbeSearchConfig(learning_rates=config.probe_learning_rates, max_epochs=config.probe_epochs,
                                         patience=config.probe_patience, inner_validation_fraction=config.probe_inner_validation_fraction,
                                         batch_size=config.probe_batch_size, weight_decay=config.weight_decay)
    groups = [fmow_site_id(row) for row in train_ok]
    selected_by_seed: dict[int, dict[str, Any]] = {}
    calibration_components, test_components, seed_rows = [], [], []
    for seed in config.seeds:
        selected = fit_selected_multiclass_probe(train_x, train_y, groups, {"calibration": calibration_x, "test": test_x},
                                                 output / "probe_seeds" / f"seed_{seed}", config=search, seed=seed, device=config.device)
        selected_by_seed[seed] = selected
        calibration_components.append(np.asarray(selected["predictions"]["calibration"]["probabilities"], dtype=np.float32))
        test_prob = np.asarray(selected["predictions"]["test"]["probabilities"], dtype=np.float32)
        test_components.append(test_prob)
        index = np.asarray([selected["class_to_index"][label] for label in test_y])
        accuracy, log_loss = _test_metrics(test_prob, index)
        seed_rows.append({"seed": seed, "test_accuracy": accuracy, "test_log_loss": log_loss,
                          "selected_learning_rate": selected["selection"]["selected_learning_rate"],
                          "selected_epoch": selected["selection"]["selected_epoch"], "checkpoint": str(selected["checkpoint"]),
                          "checkpoint_sha256": file_sha256(selected["checkpoint"])})
    class_sets = {tuple(value["classes"]) for value in selected_by_seed.values()}
    if len(class_sets) != 1 or len(next(iter(class_sets))) != 62:
        raise FmowTerraMindError("Probe seeds did not preserve the frozen 62-class mapping.")
    class_names = list(next(iter(class_sets)))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    calibration_targets = np.asarray([class_to_index[label] for label in calibration_y], dtype=np.int64)
    test_targets = np.asarray([class_to_index[label] for label in test_y], dtype=np.int64)
    protocol = BWERProtocol.from_mapping(load_yaml(config.geobwer_protocol))
    probe_panel_manifest = output / "probe_panel_manifest.json"
    probe_panel_manifest.write_text(json.dumps({
        "schema": "geobwer.fmow.terramind_probe_panel.v1",
        "selection_data": "outer_train_only_site_disjoint_inner_holdout",
        "test_used_for_selection": False,
        "seeds": list(config.seeds),
        "search_config": asdict(search),
        "components": {
            str(seed): {
                "checkpoint": str(selected["checkpoint"]),
                "checkpoint_sha256": file_sha256(selected["checkpoint"]),
                "selection_manifest": str(selected["manifest"]),
                "selection_manifest_sha256": file_sha256(selected["manifest"]),
            }
            for seed, selected in selected_by_seed.items()
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    model_lineage = {**adapter.provenance(), "adaptation_protocol": "frozen_encoder_train_only_selected_multiseed_linear_probe",
                     "probe_model_selection": "outer_train_only_site_disjoint_inner_holdout", "probe_seeds": list(config.seeds),
                     "probe_panel_manifest": str(probe_panel_manifest),
                     "probe_panel_manifest_sha256": file_sha256(probe_panel_manifest),
                     "band_profile": config.band_profile, "comparison_scope": "model_pipeline_not_backbone_causal",
                     "input_band_parity_with_dofav2": False}
    dataset_lineage = {"dataset": "fMoW-Sentinel", "metadata_sha256": file_sha256(config.metadata_csv),
                       "split_protocol": config.split_protocol, "train_row_hash": _row_hash(train_ok),
                       "calibration_row_hash": _row_hash(calibration_ok), "test_row_hash": _row_hash(test_ok)}
    seed_audits = {}
    for seed, selected in selected_by_seed.items():
        seed_dir = output / "probe_seeds" / f"seed_{seed}"
        bundle = write_multiclass_bundle(seed_dir / "formal_outputs", sample_rows=_formal_rows(test_ok),
                                         probabilities=selected["predictions"]["test"]["probabilities"], targets=test_targets,
                                         class_names=class_names, dataset="fmow_sentinel", model=f"terramind_v1_base_seed_{seed}",
                                         split=config.test_split, protocol=protocol,
                                         model_lineage={**model_lineage, "probe_seed": seed, "probe_checkpoint": str(selected["checkpoint"])},
                                         dataset_lineage=dataset_lineage, independent_unit_column="sample_id", split_role="evaluation")
        report = audit_rows(read_csv_rows(bundle.audit_table), group_columns=("country", "region", "class_label"), protocol=protocol,
                            loss_column="risk", unit_column="independent_unit_id", cluster_column="site_id", formal=True,
                            require_probabilities=True, n_bootstrap=config.audit_bootstrap, seed=seed).to_report(seed_dir / "geobwer_raw")
        seed_audits[str(seed)] = {key: str(value) for key, value in report.items()}
    write_csv(output / "probe_seed_robustness.csv", seed_rows)
    ensemble_test = np.mean(np.stack(test_components), axis=0).astype(np.float32)
    ensemble_calibration = np.mean(np.stack(calibration_components), axis=0).astype(np.float32)
    ensemble_bundle = write_multiclass_bundle(output / "formal_outputs", sample_rows=_formal_rows(test_ok), probabilities=ensemble_test,
                                               targets=test_targets, class_names=class_names, dataset="fmow_sentinel",
                                               model="terramind_v1_base_seed_ensemble", split=config.test_split, protocol=protocol,
                                               model_lineage=model_lineage, dataset_lineage=dataset_lineage,
                                               independent_unit_column="sample_id", split_role="evaluation")
    ensemble_audit = audit_rows(read_csv_rows(ensemble_bundle.audit_table), group_columns=("country", "region", "class_label"), protocol=protocol,
                                loss_column="risk", unit_column="independent_unit_id", cluster_column="site_id", formal=True,
                                require_probabilities=True, n_bootstrap=config.audit_bootstrap, seed=config.seeds[0]).to_report(output / "geobwer_raw")
    calibration_path = output / "calibration_probabilities.npz"
    np.savez_compressed(calibration_path, probabilities=ensemble_calibration, targets=calibration_targets,
                        class_names=np.asarray(class_names), sample_id=np.asarray([row["sample_id"] for row in calibration_ok]),
                        latitude=np.asarray([_coordinate_or_nan(row, "latitude", "lat") for row in calibration_ok]),
                        longitude=np.asarray([_coordinate_or_nan(row, "longitude", "lon", "lng") for row in calibration_ok]),
                        split_role=np.asarray("calibration"), test_rows_used=np.asarray(False))
    calibration_manifest = output / "calibration_manifest.json"
    calibration_manifest.write_text(json.dumps({
        "schema": "geobwer.fmow.terramind_calibration.v1",
        "split": config.calibration_split,
        "role": "uncertainty_calibration_not_probe_selection",
        "test_rows_used": False,
        "probabilities": str(calibration_path),
        "probabilities_sha256": file_sha256(calibration_path),
        "sample_count": len(calibration_targets),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps({"schema": "geobwer.fmow.terramind_campaign.v1", "config": asdict(config),
                                    "status": "complete", "formal_output_manifest": str(ensemble_bundle.manifest),
                                    "probe_panel_manifest": str(probe_panel_manifest),
                                    "calibration_manifest": str(calibration_manifest),
                                    "seed_geobwer_artifacts": seed_audits, "ensemble_geobwer_artifacts": {key: str(value) for key, value in ensemble_audit.items()},
                                    "embedding_manifests": [str(train_manifest), str(calibration_manifest_cache), str(test_manifest)],
                                    "test_used_for_model_selection": False, "raw_cross_task_geobwer_averaging_allowed": False,
                                    "pipeline_level_comparison": True}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    persist_output(output, config.persistent_output_dir, label="terramind-fmow-complete")
    return {"formal_audit_table": ensemble_bundle.audit_table, "geobwer_summary": ensemble_audit["summary"],
            "probe_seed_robustness": output / "probe_seed_robustness.csv", "probe_panel_manifest": probe_panel_manifest,
            "calibration_probabilities": calibration_path, "calibration_manifest": calibration_manifest,
            "run_manifest": manifest}


__all__ = ["FmowTerraMindConfig", "FmowTerraMindError", "run_fmow_terramind_campaign"]
