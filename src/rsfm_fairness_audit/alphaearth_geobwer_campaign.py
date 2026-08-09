from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_inference import calibrate_spatial_block_scale, equal_area_block_ids
from rsfm_fairness_audit.bwer_protocol import BWERProtocol, Validity
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.formal_outputs import FormalOutputBundle, file_sha256, write_multiclass_bundle
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.geobwer_extensions import run_multiclass_uncertainty_suite
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output
from rsfm_fairness_audit.spatial_conformal import SpatialConformalConfig


class AlphaEarthCampaignError(RuntimeError):
    """Raised when the existing AlphaEarth predictions cannot support a formal upgrade."""


@dataclass(frozen=True)
class AlphaEarthCampaignConfig:
    all_split_predictions: Path
    output_dir: Path
    persistent_output_dir: Path | None = None
    protocol_path: Path = Path("configs/geobwer/alphaearth.yaml")
    model_name: str = "alphaearth_hist_gradient_boosting"
    candidate_cell_km: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0)
    calibration_max_per_country: int = 50
    calibration_simulations: int = 200
    calibration_bootstrap: int = 500
    audit_bootstrap: int = 2000
    minimum_moderate_tail_power: float = 0.80
    coverage_tolerance: float = 0.02
    false_positive_tolerance: float = 0.01
    conformal_alpha: float = 0.10
    seed: int = 42


def _text(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def _float(row: Mapping[str, Any], *names: str) -> float:
    value = _text(row, *names)
    try:
        return float(value)
    except ValueError as exc:
        raise AlphaEarthCampaignError(f"Missing or invalid numeric field {names}: {value!r}") from exc


def _read_predictions(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...], tuple[str, ...]]:
    if not path.exists():
        raise AlphaEarthCampaignError(f"AlphaEarth all-split prediction table does not exist: {path}")
    print(f"[alphaearth:geobwer] reading {path}", flush=True)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        probability_columns = tuple(name for name in fieldnames if name.startswith("prob_"))
        rows = [dict(row) for row in reader]
    if len(probability_columns) < 2:
        raise AlphaEarthCampaignError("Full per-class probability columns prob_* are required.")
    if not rows:
        raise AlphaEarthCampaignError("AlphaEarth prediction table is empty.")
    class_names = tuple(name[len("prob_") :] for name in probability_columns)
    if len(set(class_names)) != len(class_names):
        raise AlphaEarthCampaignError("Duplicate AlphaEarth probability class columns.")
    return rows, probability_columns, class_names


def _split_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calibration = [dict(row) for row in rows if _text(row, "split").lower() in {"calibration", "validation", "val"}]
    test = [dict(row) for row in rows if _text(row, "split").lower() == "test"]
    if not calibration or not test:
        raise AlphaEarthCampaignError("Both calibration and test rows are required in all_split_predictions.")
    calibration_ids = {_text(row, "sample_id") for row in calibration}
    test_ids = {_text(row, "sample_id") for row in test}
    if "" in calibration_ids or "" in test_ids or calibration_ids & test_ids:
        raise AlphaEarthCampaignError("Calibration/test sample IDs must be non-empty and disjoint.")
    legacy_calibration_blocks = {_text(row, "spatial_block_id") for row in calibration}
    legacy_test_blocks = {_text(row, "spatial_block_id") for row in test}
    if "" in legacy_calibration_blocks or "" in legacy_test_blocks:
        raise AlphaEarthCampaignError("The frozen spatial split requires legacy spatial_block_id on every row.")
    if legacy_calibration_blocks & legacy_test_blocks:
        raise AlphaEarthCampaignError("Legacy spatial-block leakage exists between calibration and test.")
    return calibration, test


def _probabilities_targets(
    rows: Sequence[Mapping[str, Any]],
    probability_columns: Sequence[str],
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(
        [[float(row[column]) for column in probability_columns] for row in rows], dtype=np.float32
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise AlphaEarthCampaignError("AlphaEarth probabilities must be finite and in [0,1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4):
        raise AlphaEarthCampaignError("AlphaEarth probability rows do not sum to one.")
    class_index = {str(name): index for index, name in enumerate(class_names)}
    labels = [_text(row, "label", "worldcover_label") for row in rows]
    missing = sorted(set(labels) - set(class_index))
    if missing:
        raise AlphaEarthCampaignError(f"Labels absent from probability mapping: {missing}")
    return probabilities, np.asarray([class_index[label] for label in labels], dtype=np.int64)


def _stratified_calibration_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_per_country: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_per_country < 2:
        raise ValueError("calibration_max_per_country must be at least two.")
    rng = np.random.default_rng(seed)
    by_country: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        country = _text(row, "country_iso3", "country")
        if not country:
            raise AlphaEarthCampaignError("Every AlphaEarth row requires verified country_iso3.")
        by_country.setdefault(country, []).append(dict(row))
    selected: list[dict[str, Any]] = []
    for country in sorted(by_country):
        items = by_country[country]
        indexes = np.arange(len(items))
        rng.shuffle(indexes)
        selected.extend(items[index] for index in indexes[:max_per_country])
    return selected


def _risk(row: Mapping[str, Any]) -> float:
    value = _text(row, "risk")
    if value:
        return float(value)
    return float(_text(row, "prediction") != _text(row, "label", "worldcover_label"))


def _formal_rows(rows: Sequence[Mapping[str, Any]], block_ids: Sequence[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, block_id in zip(rows, block_ids):
        sample_id = _text(row, "sample_id")
        country = _text(row, "country_iso3", "country")
        class_name = _text(row, "worldcover_class_name", "class_label", "label", "worldcover_label")
        region = _text(row, "region")
        if not sample_id or not country or not class_name:
            raise AlphaEarthCampaignError(
                "Every formal AlphaEarth row requires sample_id, verified country, and WorldCover class."
            )
        item = dict(row)
        item.update(
            {
                "sample_id": sample_id,
                "independent_unit_id": sample_id,
                "country": country,
                "country_iso3": country,
                "worldcover_class_name": class_name,
                "region": region,
                "country_class": f"{country}|{class_name}",
                "region_class": f"{region}|{class_name}" if region else "",
                "legacy_spatial_block_id": _text(row, "spatial_block_id"),
                "spatial_block_id": block_id,
                "latitude": _float(row, "lat", "latitude"),
                "longitude": _float(row, "lon", "longitude"),
            }
        )
        output.append(item)
    return output


def _sample_id_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(_text(row, "sample_id") for row in rows):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _calibration_signature(
    config: AlphaEarthCampaignConfig,
    *,
    source_hash: str,
    calibration_subset: Sequence[Mapping[str, Any]],
    protocol: BWERProtocol,
) -> str:
    payload = {
        "source_predictions_sha256": source_hash,
        "calibration_sample_id_hash": _sample_id_hash(calibration_subset),
        "protocol_hash": protocol.signature,
        "candidate_cell_km": list(config.candidate_cell_km),
        "calibration_max_per_country": config.calibration_max_per_country,
        "calibration_simulations": config.calibration_simulations,
        "calibration_bootstrap": config.calibration_bootstrap,
        "minimum_moderate_tail_power": config.minimum_moderate_tail_power,
        "coverage_tolerance": config.coverage_tolerance,
        "false_positive_tolerance": config.false_positive_tolerance,
        "seed": config.seed,
        "validity_gate": "range_adequacy_and_simulated_coverage_fpr",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_alphaearth_geobwer_campaign(config: AlphaEarthCampaignConfig) -> dict[str, Path]:
    hydrate_output(config.output_dir, config.persistent_output_dir)
    output = ensure_dir(config.output_dir)
    rows, probability_columns, class_names = _read_predictions(config.all_split_predictions)
    calibration_rows, test_rows = _split_rows(rows)
    calibration_probabilities, calibration_targets = _probabilities_targets(
        calibration_rows, probability_columns, class_names
    )
    test_probabilities, test_targets = _probabilities_targets(test_rows, probability_columns, class_names)
    base_protocol = BWERProtocol.from_mapping(load_yaml(config.protocol_path))
    source_hash = file_sha256(config.all_split_predictions)

    calibration_subset = _stratified_calibration_subset(
        calibration_rows,
        max_per_country=config.calibration_max_per_country,
        seed=config.seed,
    )
    block_calibration_path = output / "spatial_block_calibration.json"
    calibration_signature = _calibration_signature(
        config,
        source_hash=source_hash,
        calibration_subset=calibration_subset,
        protocol=base_protocol,
    )
    if block_calibration_path.exists():
        block_payload = json.loads(block_calibration_path.read_text(encoding="utf-8"))
        if block_payload.get("calibration_signature") != calibration_signature:
            raise AlphaEarthCampaignError(
                "Existing spatial calibration was produced by a different source/protocol. "
                "Use a new output directory; do not mix formal runs."
            )
        print(f"[alphaearth:geobwer] reusing verified spatial calibration {block_calibration_path}")
    else:
        print(
            f"[alphaearth:geobwer] calibrating spatial blocks on {len(calibration_subset)} validation rows "
            f"({len(calibration_rows)} total validation rows; test outcomes remain sealed)",
            flush=True,
        )
        block_calibration = calibrate_spatial_block_scale(
            [_risk(row) for row in calibration_subset],
            [_text(row, "country_iso3", "country") for row in calibration_subset],
            [_float(row, "lat", "latitude") for row in calibration_subset],
            [_float(row, "lon", "longitude") for row in calibration_subset],
            candidate_cell_km=config.candidate_cell_km,
            n_simulations=config.calibration_simulations,
            n_bootstrap=config.calibration_bootstrap,
            seed=config.seed,
            beta=base_protocol.beta,
            minimum_moderate_tail_power=config.minimum_moderate_tail_power,
            require_power_gate=False,
            coverage_tolerance=config.coverage_tolerance,
            false_positive_tolerance=config.false_positive_tolerance,
        )
        block_payload = {
                "schema": "geobwer.alphaearth.spatial_block_calibration.v2",
                "selection_data": "calibration_only",
                "validity_gate": "range_adequacy_and_simulated_coverage_fpr",
                "power_role": "reported_and_candidate_ranking_not_validity",
                **asdict(block_calibration),
                "validity": block_calibration.validity.value,
                "calibration_signature": calibration_signature,
                "calibration_rows_total": len(calibration_rows),
                "calibration_rows_simulation_subset": len(calibration_subset),
                "minimum_moderate_tail_power": config.minimum_moderate_tail_power,
                "coverage_tolerance": config.coverage_tolerance,
                "false_positive_tolerance": config.false_positive_tolerance,
        }
        block_calibration_path.write_text(
            json.dumps(
                block_payload,
            ensure_ascii=False,
            indent=2,
            ),
            encoding="utf-8",
        )
        persist_output(output, config.persistent_output_dir, label="alphaearth-spatial-calibration")
    if block_payload.get("validity") != Validity.VALID.value or block_payload.get("selected_cell_km") is None:
        raise AlphaEarthCampaignError(
            "No equal-area spatial block passed the pre-registered range/coverage/FPR gate; "
            "retain AlphaEarth inference as descriptive instead of tuning on test results."
        )
    cell_km = float(block_payload["selected_cell_km"])
    test_blocks = equal_area_block_ids(
        [_float(row, "lat", "latitude") for row in test_rows],
        [_float(row, "lon", "longitude") for row in test_rows],
        cell_km=cell_km,
    )
    sample_rows = _formal_rows(test_rows, test_blocks)
    metadata = dict(base_protocol.metadata)
    metadata.update(
        {
            "spatial_block_cell_km": str(cell_km),
            "spatial_block_selection": "calibration_only_range_coverage_fpr_gate_power_ranked",
            "spatial_block_calibration_sha256": file_sha256(block_calibration_path),
            "spatial_block_calibrated": "true",
        }
    )
    protocol = replace(
        base_protocol,
        metadata=tuple(sorted(metadata.items())),
        min_clusters_for_inference=base_protocol.min_clusters_per_slice,
        cluster_eligibility_calibration_signature=calibration_signature,
    )
    bundle: FormalOutputBundle = write_multiclass_bundle(
        output / "formal_outputs",
        sample_rows=sample_rows,
        probabilities=test_probabilities,
        targets=test_targets,
        class_names=class_names,
        dataset="alphaearth_worldcover_full",
        model=config.model_name,
        split="test",
        protocol=protocol,
        model_lineage={
            "model": "HistGradientBoostingClassifier",
            "representation": "Google_Satellite_Embedding_V1_Annual_AlphaEarth",
            "prediction_source": str(config.all_split_predictions),
            "prediction_source_sha256": source_hash,
            "adaptation_protocol": "existing_frozen_predictions_no_refit",
        },
        dataset_lineage={
            "dataset": "AlphaEarth_embeddings_x_ESA_WorldCover_v200",
            "split": "strict_legacy_spatial_block_hash",
            "reference_semantics": "map_product_agreement_not_human_ground_truth",
            "test_sample_count": len(sample_rows),
            "test_sample_id_hash": _sample_id_hash(test_rows),
        },
        independent_unit_column="independent_unit_id",
        split_role="evaluation",
    )
    calibration_path = output / "calibration_probabilities.npz"
    np.savez_compressed(
        calibration_path,
        probabilities=calibration_probabilities,
        targets=calibration_targets,
        class_names=np.asarray(class_names, dtype=str),
        sample_id=np.asarray([_text(row, "sample_id") for row in calibration_rows], dtype=str),
        latitude=np.asarray(
            [_float(row, "lat", "latitude") for row in calibration_rows],
            dtype=np.float64,
        ),
        longitude=np.asarray(
            [_float(row, "lon", "longitude") for row in calibration_rows],
            dtype=np.float64,
        ),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    calibration_manifest = output / "calibration_manifest.json"
    calibration_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.multiclass_calibration.v1",
                "split_role": "calibration",
                "test_rows_used": False,
                "probabilities_sha256": file_sha256(calibration_path),
                "source_predictions_sha256": source_hash,
                "sample_count": len(calibration_rows),
                "class_names": list(class_names),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    formal_rows = read_csv_rows(bundle.audit_table)
    raw_axes = tuple(
        column
        for column in ("country_iso3", "region", "worldcover_class_name", "country_class", "region_class")
        if all(str(row.get(column, "")).strip() for row in formal_rows)
    )
    raw_audit = audit_rows(
        formal_rows,
        group_columns=raw_axes,
        protocol=protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="spatial_block_id",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seed,
    )
    raw_artifacts = raw_audit.to_report(output / "geobwer_raw")
    standardized_audit = audit_rows(
        formal_rows,
        group_columns=("country_iso3",),
        protocol=protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="spatial_block_id",
        balance_column="worldcover_class_name",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seed,
    )
    standardized_artifacts = standardized_audit.to_report(output / "geobwer_standardized")
    uncertainty_artifacts = run_multiclass_uncertainty_suite(
        calibration_path,
        bundle.output_dir,
        output / "uncertainty_extensions",
        protocol=protocol,
        group_columns=raw_axes,
        calibration_manifest=calibration_manifest,
        alpha=config.conformal_alpha,
        n_bootstrap=config.audit_bootstrap,
        seed=config.seed,
        spatial_conformal_config=SpatialConformalConfig(),
    )
    run_manifest = output / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.alphaearth_campaign.v1",
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "source_predictions_sha256": source_hash,
                "formal_output_manifest": str(bundle.manifest),
                "calibration_manifest": str(calibration_manifest),
                "spatial_block_calibration": str(block_calibration_path),
                "raw_geobwer_artifacts": {key: str(value) for key, value in raw_artifacts.items()},
                "standardized_geobwer_artifacts": {key: str(value) for key, value in standardized_artifacts.items()},
                "uncertainty_artifacts": {key: str(value) for key, value in uncertainty_artifacts.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    persist_output(output, config.persistent_output_dir, label="alphaearth-formal-campaign-complete")
    return {
        "formal_audit_table": bundle.audit_table,
        "formal_output_manifest": bundle.manifest,
        "calibration_probabilities": calibration_path,
        "calibration_manifest": calibration_manifest,
        "spatial_block_calibration": block_calibration_path,
        "raw_geobwer_summary": raw_artifacts["summary"],
        "standardized_geobwer_summary": standardized_artifacts["summary"],
        "uncertainty_summary": uncertainty_artifacts["summary"],
        "run_manifest": run_manifest,
    }


__all__ = [
    "AlphaEarthCampaignConfig",
    "AlphaEarthCampaignError",
    "run_alphaearth_geobwer_campaign",
]
