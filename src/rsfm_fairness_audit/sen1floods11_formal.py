from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_inference import calibrate_spatial_block_scale, equal_area_block_ids
from rsfm_fairness_audit.bwer_protocol import BWERProtocol, Validity
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.formal_outputs import FormalOutputBundle, file_sha256, write_segmentation_bundle
from rsfm_fairness_audit.io import read_csv_rows
from rsfm_fairness_audit.terratorch_exports import write_probability_batch


class Sen1FormalizationError(RuntimeError):
    """Raised when a Sen1Floods11 formal result cannot be traced to geospatial evidence."""


def write_sen1_probability_export(
    output_dir: str | Path,
    *,
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    filenames: Sequence[Any],
    batch_size: int = 8,
    metadata: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Write a framework-neutral Sen1 probability export.

    The layout is intentionally identical to the TerraTorch prediction-writer
    contract so Prithvi and supervised baselines pass through the exact same
    parser, geolocation checks, spatial calibration and CRC implementation.
    Existing complete exports are validated and reused; partial exports fail
    rather than being silently mixed.
    """

    if not (
        len(probabilities) == len(targets) == len(filenames)
        and (metadata is None or len(metadata) == len(probabilities))
    ):
        raise Sen1FormalizationError(
            "probabilities, targets, filenames, and optional metadata must align."
        )
    if not probabilities:
        raise Sen1FormalizationError("Cannot write an empty Sen1 probability export.")
    output = Path(output_dir)
    manifest_path = output / "writer_manifest_rank_0.json"
    index_path = output / "index_parts" / "part-000000.jsonl"
    if manifest_path.exists() or index_path.exists():
        if not (manifest_path.exists() and index_path.exists()):
            raise Sen1FormalizationError(
                f"Partial probability export exists under {output}; use a new output directory."
            )
        existing = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(existing) != len(probabilities):
            raise Sen1FormalizationError(
                f"Existing probability export count differs: {len(existing)} vs {len(probabilities)}."
            )
        if all(_resolve_probability_artifact(output, row["probability_path"]).exists() for row in existing):
            return output
        raise Sen1FormalizationError(f"Existing probability export under {output} is incomplete.")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(probabilities), int(batch_size))):
        end = min(start + int(batch_size), len(probabilities))
        positive = np.stack(
            [np.asarray(value, dtype=np.float32).squeeze() for value in probabilities[start:end]]
        )
        if positive.ndim != 3:
            raise Sen1FormalizationError(
                f"Positive probability maps must stack as [N,H,W], got {positive.shape}."
            )
        if not np.all(np.isfinite(positive)) or np.any(positive < 0.0) or np.any(positive > 1.0):
            raise Sen1FormalizationError("Positive probability maps must be finite and in [0,1].")
        two_class = np.stack([1.0 - positive, positive], axis=1)
        batch_metadata = {
            key: [row.get(key, "") for row in (metadata or [{}] * len(probabilities))[start:end]]
            for key in ("event_id", "country", "region")
        }
        batch_records = write_probability_batch(
            output,
            outputs={
                "probabilities": two_class,
                "target": np.stack([np.asarray(value).squeeze() for value in targets[start:end]]),
                "filename": list(filenames[start:end]),
                **batch_metadata,
            },
            batch={},
            batch_idx=batch_index,
            dataloader_idx=0,
        )
        records.extend(batch_records)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.probability_export.v1",
                "writer": "framework_neutral",
                "sample_count": len(records),
                "index_part": str(index_path.relative_to(output)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def _read_index(export_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(export_dir)
    paths = sorted((root / "index_parts").glob("*.jsonl"))
    if not paths:
        raise Sen1FormalizationError(f"No TerraTorch index parts found under {root / 'index_parts'}.")
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise Sen1FormalizationError(f"Missing sample_id in {path}.")
            previous = by_id.get(sample_id)
            if previous is not None and previous != row:
                raise Sen1FormalizationError(f"Conflicting duplicate sample_id={sample_id} across prediction batches.")
            by_id[sample_id] = row
    if not by_id:
        raise Sen1FormalizationError("TerraTorch probability index is empty.")
    return [by_id[key] for key in sorted(by_id)]


def _filename_mapping(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    text = str(value or "").strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return {str(key): str(item) for key, item in parsed.items()}
    return {"image": text} if text else {}


def _source_path(filename: Any, data_root: str | Path | None) -> Path | None:
    mapping = _filename_mapping(filename)
    for preferred in ("S2L1C", "S1GRD", "S2", "S1", "image"):
        if preferred not in mapping:
            continue
        path = Path(mapping[preferred])
        if not path.is_absolute() and data_root is not None:
            path = Path(data_root) / path
        if path.exists():
            return path
    for value in mapping.values():
        path = Path(value)
        if not path.is_absolute() and data_root is not None:
            path = Path(data_root) / path
        if path.exists():
            return path
    return None


def _filename_candidate(filename: Any) -> Path | None:
    mapping = _filename_mapping(filename)
    for preferred in ("S2L1C", "S1GRD", "S2", "S1", "image"):
        if mapping.get(preferred):
            return Path(mapping[preferred])
    return Path(next(iter(mapping.values()))) if mapping else None


def _canonical_chip_id(path: Path | None, fallback: str) -> str:
    name = path.stem if path is not None else fallback
    for suffix in ("_S2Hand", "_S1Hand", "_LabelHand", "_S2", "_S1", "_Label"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _raster_centroid(path: Path) -> tuple[float, float]:
    try:
        import rasterio
        from rasterio.warp import transform
    except ImportError as exc:  # pragma: no cover - used in Colab
        raise Sen1FormalizationError("rasterio is required to recover verified GeoTIFF coordinates.") from exc
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise Sen1FormalizationError(f"GeoTIFF has no CRS: {path}")
        x = (dataset.bounds.left + dataset.bounds.right) / 2.0
        y = (dataset.bounds.bottom + dataset.bounds.top) / 2.0
        if str(dataset.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
            lon, lat = x, y
        else:
            longitude, latitude = transform(dataset.crs, "EPSG:4326", [x], [y])
            lon, lat = longitude[0], latitude[0]
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise Sen1FormalizationError(f"Invalid recovered coordinate for {path}: lat={lat}, lon={lon}.")
    return float(lat), float(lon)


def _metadata_lookup(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_csv_rows(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("chip_id") or "")
        if sample_id:
            output[sample_id] = dict(row)
    return output


def _resolve_probability_artifact(export_dir: str | Path, value: Any) -> Path:
    """Resolve portable v2 paths while retaining read compatibility with v1.

    New exports store paths relative to the export root.  For already-created
    v1 indexes, an absolute artifact is accepted when it still exists; if the
    export was moved from Colab to Drive, its basename is recovered from the
    local ``samples`` directory.  Relative paths may not escape the export.
    """

    root = Path(export_dir).resolve()
    raw = Path(str(value))
    if raw.is_absolute():
        if raw.exists():
            return raw
        migrated = root / "samples" / raw.name
        if migrated.exists():
            return migrated
        return raw
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Sen1FormalizationError(
            f"Probability artifact path escapes its export root: {value!r}."
        ) from exc
    return candidate


def _load_units(
    export_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    metadata_csv: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    metadata = _metadata_lookup(metadata_csv)
    rows: list[dict[str, Any]] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    seen_chip_ids: set[str] = set()
    aggregate_valid_pixel_count = 0
    for index_row in _read_index(export_dir):
        artifact_path = _resolve_probability_artifact(export_dir, index_row["probability_path"])
        if not artifact_path.exists():
            raise Sen1FormalizationError(f"Missing probability sample artifact: {artifact_path}")
        with np.load(artifact_path) as artifact:
            probability = np.asarray(artifact["probabilities"], dtype=np.float32)
            target = np.asarray(artifact["target"])
        if probability.ndim != 3 or probability.shape[0] != 2:
            raise Sen1FormalizationError(
                f"Expected two-class segmentation probabilities [2,H,W], got {probability.shape} for {artifact_path}."
            )
        if not np.allclose(probability.sum(axis=0), 1.0, atol=2e-4, rtol=2e-4):
            raise Sen1FormalizationError(f"Class probabilities do not sum to one for {artifact_path}.")
        target = np.squeeze(target)
        if target.shape != probability.shape[1:]:
            raise Sen1FormalizationError(
                f"Target/probability dimensions differ for {artifact_path}: {target.shape} vs {probability.shape[1:]}."
            )
        observed_target_values = set(np.unique(target).tolist())
        if not observed_target_values.issubset({-1, 0, 1}):
            raise Sen1FormalizationError(
                f"Target contains values outside the frozen Sen1Floods11 label contract "
                f"{{-1,0,1}} for {artifact_path}: {sorted(observed_target_values)}."
            )
        valid = np.isin(target, [0, 1])
        valid_pixel_count = int(np.sum(valid))
        aggregate_valid_pixel_count += valid_pixel_count

        source = _source_path(index_row.get("filename"), data_root)
        fallback = str(index_row["sample_id"])
        chip_id = _canonical_chip_id(source or _filename_candidate(index_row.get("filename")), fallback)
        if chip_id in seen_chip_ids:
            raise Sen1FormalizationError(f"Duplicate physical Sen1Floods11 chip after filename normalization: {chip_id}.")
        seen_chip_ids.add(chip_id)
        source_meta = metadata.get(chip_id, {})
        if source_meta.get("latitude") not in (None, "") and source_meta.get("longitude") not in (None, ""):
            latitude = float(source_meta["latitude"])
            longitude = float(source_meta["longitude"])
        elif source is not None:
            latitude, longitude = _raster_centroid(source)
        else:
            raise Sen1FormalizationError(
                f"Cannot recover coordinates for chip={chip_id}. Supply data_root or a metadata CSV with latitude/longitude."
            )
        event_id = str(source_meta.get("event_id") or source_meta.get("event") or chip_id.split("_", 1)[0])
        if not event_id or event_id == "to_verify":
            raise Sen1FormalizationError(f"Unverified event_id for chip={chip_id}.")
        rows.append(
            {
                "sample_id": chip_id,
                "independent_unit_id": chip_id,
                "scene_id": chip_id,
                "event_id": event_id,
                "latitude": latitude,
                "longitude": longitude,
                "country": source_meta.get("country", source_meta.get("ISO_CC", "")),
                "source_filename": str(source) if source is not None else str(index_row.get("filename", "")),
                "valid_pixel_count": valid_pixel_count,
                "label_support_status": (
                    "identified" if valid_pixel_count > 0 else "all_ignore"
                ),
            }
        )
        probabilities.append(probability[1])
        targets.append(target)
        valid_masks.append(valid)
    if aggregate_valid_pixel_count <= 0:
        raise Sen1FormalizationError(
            "Sen1Floods11 probability export contains no valid hand-labeled pixels "
            "across the complete split."
        )
    return rows, probabilities, targets, valid_masks


def load_sen1_probability_units(
    export_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    metadata_csv: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Load and validate one labeled TerraTorch export for downstream calibration.

    This public wrapper deliberately shares the exact physical-unit, geolocation,
    duplicate and mask checks used by formal test finalization. Calibration and
    evaluation therefore cannot drift through separate parsing implementations.
    """

    return _load_units(export_dir, data_root=data_root, metadata_csv=metadata_csv)


def _chip_risks(probabilities: Sequence[np.ndarray], targets: Sequence[np.ndarray], valid_masks: Sequence[np.ndarray]) -> list[float]:
    output: list[float] = []
    for probability, target, valid in zip(probabilities, targets, valid_masks):
        prediction = probability >= 0.5
        truth = target == 1
        tp = int(np.sum(valid & prediction & truth))
        fp = int(np.sum(valid & prediction & ~truth))
        fn = int(np.sum(valid & ~prediction & truth))
        union = tp + fp + fn
        output.append(1.0 - (tp / union if union else 1.0))
    return output


def _identified_units(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
    *,
    context: str,
) -> tuple[
    list[dict[str, Any]],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    dict[str, Any],
]:
    """Select units with an identifiable pixel risk without losing split lineage."""

    if not (
        len(rows) == len(probabilities) == len(targets) == len(valid_masks)
    ):
        raise Sen1FormalizationError(
            f"Sen1 unit arrays are misaligned during {context}."
        )
    keep = [
        index
        for index, valid in enumerate(valid_masks)
        if int(np.sum(np.asarray(valid, dtype=bool))) > 0
    ]
    if not keep:
        raise Sen1FormalizationError(
            f"Sen1Floods11 {context} contains no units with identifiable pixel risk."
        )
    keep_set = set(keep)
    excluded_ids = [
        str(rows[index]["sample_id"])
        for index in range(len(rows))
        if index not in keep_set
    ]
    support = {
        "source_split_sample_count": len(rows),
        "auditable_sample_count": len(keep),
        "all_ignore_sample_count": len(excluded_ids),
        "all_ignore_sample_ids": excluded_ids,
        "all_ignore_policy": (
            "preserved_in_source_probability_export_and_split_lineage;"
            "excluded_from_pixel_risk_estimand"
        ),
    }
    return (
        [dict(rows[index]) for index in keep],
        [np.asarray(probabilities[index]) for index in keep],
        [np.asarray(targets[index]) for index in keep],
        [np.asarray(valid_masks[index], dtype=bool) for index in keep],
        support,
    )


def _spatial_panel_signature(rows: Sequence[Mapping[str, Any]], risks: Sequence[float]) -> str:
    if len(rows) != len(risks):
        raise ValueError("rows and risks must align for spatial calibration provenance.")
    payload = sorted(
        (
            str(row["sample_id"]),
            str(row["event_id"]),
            float(row["latitude"]),
            float(row["longitude"]),
            float(risk),
        )
        for row, risk in zip(rows, risks)
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def calibrate_common_sen1_spatial_blocks(
    validation_exports: Mapping[str, str | Path],
    output_json: str | Path,
    *,
    data_root: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    candidate_cell_km: Sequence[float] = (25.0, 50.0, 100.0, 200.0, 400.0),
    confidence_level: float = 0.95,
    n_simulations: int = 200,
    n_bootstrap: int = 500,
    seed: int = 42,
    beta: float = 0.10,
    minimum_moderate_tail_power: float = 0.80,
) -> dict[str, Any]:
    """Choose one validation-only spatial scale shared by all TerraMind modes."""

    if not validation_exports:
        raise Sen1FormalizationError("At least one validation probability export is required.")
    loaded: dict[str, tuple[list[dict[str, Any]], list[float]]] = {}
    input_signatures: dict[str, str] = {}
    support_by_model: dict[str, dict[str, Any]] = {}
    for model_name, export_dir in sorted(validation_exports.items()):
        rows, probabilities, targets, valid = _load_units(
            export_dir, data_root=data_root, metadata_csv=metadata_csv
        )
        rows, probabilities, targets, valid, support = _identified_units(
            rows,
            probabilities,
            targets,
            valid,
            context=f"validation spatial calibration for model={model_name}",
        )
        risks = _chip_risks(probabilities, targets, valid)
        loaded[str(model_name)] = (rows, risks)
        input_signatures[str(model_name)] = _spatial_panel_signature(rows, risks)
        support_by_model[str(model_name)] = support
    signature_payload = {
        "validation_inputs": input_signatures,
        "validation_label_support": support_by_model,
        "candidate_cell_km": [float(value) for value in candidate_cell_km],
        "confidence_level": confidence_level,
        "n_simulations": n_simulations,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "beta": beta,
        "minimum_moderate_tail_power": minimum_moderate_tail_power,
        "validity_gate": "range_adequacy_and_simulated_coverage_fpr_for_every_model",
    }
    calibration_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = Path(output_json)
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if (
            previous.get("schema") != "geobwer.sen1floods11.common_spatial_block_calibration.v2"
            or previous.get("calibration_signature") != calibration_signature
        ):
            raise Sen1FormalizationError(
                "Existing common spatial calibration does not match the current validation exports/protocol. "
                "Use a new output directory; do not mix formal runs."
            )
        if previous.get("validity") != Validity.VALID.value or previous.get("all_models_passed") is not True:
            raise Sen1FormalizationError("The cached common spatial calibration is not valid.")
        print(f"[sen1:spatial-calibration] reusing verified calibration {output}")
        return previous
    reports: dict[str, Any] = {}
    passing_by_model: dict[str, set[float]] = {}
    records_by_model: dict[str, dict[float, Any]] = {}
    for model_name, (rows, risks) in loaded.items():
        calibration = calibrate_spatial_block_scale(
            risks,
            [row["event_id"] for row in rows],
            [row["latitude"] for row in rows],
            [row["longitude"] for row in rows],
            candidate_cell_km=candidate_cell_km,
            confidence_level=confidence_level,
            n_simulations=n_simulations,
            n_bootstrap=n_bootstrap,
            seed=seed,
            beta=beta,
            minimum_moderate_tail_power=minimum_moderate_tail_power,
            require_power_gate=False,
        )
        records = {record.cell_km: record for record in calibration.candidates}
        passing_by_model[str(model_name)] = {cell for cell, record in records.items() if record.passes}
        records_by_model[str(model_name)] = records
        reports[str(model_name)] = {
            "range_estimate": asdict(calibration.range_estimate),
            "candidates": [asdict(record) for record in calibration.candidates],
            "validity": calibration.validity.value,
        }
    common = set.intersection(*passing_by_model.values())
    if not common:
        raise Sen1FormalizationError(
            "No spatial block size passed coverage/FPR gates for every sensor mode. Keep Sen1 inference descriptive; "
            "do not choose a scale from test results."
        )
    ranked = sorted(
        common,
        key=lambda cell: (
            -min(records_by_model[model][cell].moderate_tail_power for model in records_by_model),
            -float(np.mean([records_by_model[model][cell].moderate_tail_power for model in records_by_model])),
            cell,
        ),
    )
    selected = float(ranked[0])
    payload = {
        "schema": "geobwer.sen1floods11.common_spatial_block_calibration.v2",
        "validity": Validity.VALID.value,
        "all_models_passed": True,
        "selection_data": "validation_only",
        "models": reports,
        "candidate_cell_km": [float(value) for value in candidate_cell_km],
        "selected_cell_km": selected,
        "validity_gate": "range_adequacy_and_simulated_coverage_fpr_for_every_model",
        "power_role": "reported_and_candidate_ranking_not_validity",
        "selection_rule": "pass_all_models_then_maximize_minimum_power_then_mean_power_then_smallest_cell",
        "confidence_level": confidence_level,
        "n_simulations": n_simulations,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "beta": beta,
        "minimum_moderate_tail_power": minimum_moderate_tail_power,
        "calibration_signature": calibration_signature,
        "validation_input_signatures": input_signatures,
        "validation_label_support": support_by_model,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def finalize_sen1floods11_segmentation(
    export_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    checkpoint_path: str | Path,
    pretraining_checkpoint_path: str | Path,
    pretraining_checkpoint_sha256: str,
    protocol_path: str | Path,
    block_calibration_path: str | Path,
    data_root: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    split: str = "test",
    sensor_mode: str,
    terratorch_version: str,
    model_selection_lineage: Mapping[str, Any] | None = None,
) -> FormalOutputBundle:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise Sen1FormalizationError(f"TerraMind checkpoint does not exist: {checkpoint}")
    pretraining_checkpoint = Path(pretraining_checkpoint_path)
    if not pretraining_checkpoint.exists():
        raise Sen1FormalizationError(f"TerraMind pretraining checkpoint does not exist: {pretraining_checkpoint}")
    observed_pretraining_sha256 = file_sha256(pretraining_checkpoint)
    if observed_pretraining_sha256 != str(pretraining_checkpoint_sha256).lower():
        raise Sen1FormalizationError(
            "TerraMind pretraining checkpoint changed between campaign preflight and formalization."
        )
    return finalize_sen1_probability_export(
        export_dir,
        output_dir,
        model_name=model_name,
        protocol_path=protocol_path,
        block_calibration_path=block_calibration_path,
        data_root=data_root,
        metadata_csv=metadata_csv,
        split=split,
        model_lineage={
            "model": model_name,
            "backbone": "terramind_v1_base",
            "pretrained": True,
            "sensor_mode": sensor_mode,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "pretraining_checkpoint_path": str(pretraining_checkpoint),
            "pretraining_checkpoint_sha256": observed_pretraining_sha256,
            "terratorch_version": terratorch_version,
            "probability_export": str(export_dir),
            **dict(model_selection_lineage or {}),
        },
        dataset_lineage={
            "dataset": "Sen1Floods11-v1.1-HandLabeled",
            "split": split,
            "sensor_mode": sensor_mode,
        },
    )


def finalize_sen1_probability_export(
    export_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    protocol_path: str | Path,
    block_calibration_path: str | Path,
    model_lineage: Mapping[str, Any],
    dataset_lineage: Mapping[str, Any],
    data_root: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    split: str = "test",
) -> FormalOutputBundle:
    """Formalize any Sen1 model through one shared GeoBWER contract."""

    source_rows, source_probabilities, source_targets, source_valid = _load_units(
        export_dir, data_root=data_root, metadata_csv=metadata_csv
    )
    rows, probabilities, targets, valid, label_support = _identified_units(
        source_rows,
        source_probabilities,
        source_targets,
        source_valid,
        context=f"{split} formalization",
    )
    calibration_path = Path(block_calibration_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("schema") != "geobwer.sen1floods11.common_spatial_block_calibration.v2":
        raise Sen1FormalizationError("Unrecognized Sen1 spatial-block calibration schema.")
    if calibration.get("selection_data") != "validation_only":
        raise Sen1FormalizationError("Spatial block calibration must be selected on validation data only.")
    if calibration.get("validity") != Validity.VALID.value or calibration.get("all_models_passed") is not True:
        raise Sen1FormalizationError("Spatial block calibration did not pass the all-model range/coverage/FPR gate.")
    cell_km = float(calibration["selected_cell_km"])
    block_ids = equal_area_block_ids(
        [row["latitude"] for row in rows], [row["longitude"] for row in rows], cell_km=cell_km
    )
    for row, block_id in zip(rows, block_ids):
        row["spatial_block_id"] = block_id

    base_protocol = BWERProtocol.from_mapping(load_yaml(protocol_path))
    metadata = dict(base_protocol.metadata)
    metadata.update(
        {
            "spatial_block_cell_km": str(cell_km),
            "spatial_block_selection": "validation_common_all_models",
            "spatial_block_calibration_sha256": file_sha256(calibration_path),
            "spatial_block_calibrated": "true",
            "small_cluster_calibrated": "true",
            "small_cluster_calibration_method": "validation_layout_range_coverage_fpr_monte_carlo_gate",
        }
    )
    protocol = replace(base_protocol, metadata=tuple(sorted(metadata.items())))
    resolved_model_lineage = dict(model_lineage)
    resolved_model_lineage.setdefault("model", model_name)
    resolved_model_lineage.setdefault("probability_export", str(export_dir))
    # The dataset signature is a cross-model comparison key. Sensor mode,
    # architecture and band profile describe the model input contract, not the
    # evaluated physical cohort, and must therefore never enter this signature.
    # Include the reference masks themselves so two exports cannot compare as
    # the same dataset merely because their sample identifiers happen to match.
    resolved_dataset_lineage = {
        key: value
        for key, value in dict(dataset_lineage).items()
        if key
        not in {
            "sensor_mode",
            "band_profile",
            "input_channels",
            "model",
            "architecture",
            "probability_export",
        }
    }
    reference_hash = hashlib.sha256()
    for row, target, valid_mask in sorted(
        zip(source_rows, source_targets, source_valid),
        key=lambda value: str(value[0]["sample_id"]),
    ):
        target_array = np.asarray(target)
        valid_array = np.asarray(valid_mask, dtype=np.uint8)
        reference_hash.update(str(row["sample_id"]).encode("utf-8"))
        reference_hash.update(str(target_array.shape).encode("ascii"))
        reference_hash.update(target_array.astype(np.int16, copy=False).tobytes(order="C"))
        reference_hash.update(valid_array.tobytes(order="C"))
    resolved_dataset_lineage.update(
        {
            "dataset": resolved_dataset_lineage.get(
                "dataset", "Sen1Floods11-v1.1-HandLabeled"
            ),
            "split": split,
            "sample_count": len(source_rows),
            "sample_ids": [row["sample_id"] for row in source_rows],
            "formal_audit_sample_ids": [row["sample_id"] for row in rows],
            "reference_targets_sha256": reference_hash.hexdigest(),
            "spatial_block_cell_km": cell_km,
            **label_support,
            "metadata_sha256": (
                file_sha256(Path(metadata_csv))
                if metadata_csv is not None and Path(metadata_csv).is_file()
                else ""
            ),
        }
    )
    return write_segmentation_bundle(
        output_dir,
        sample_rows=rows,
        positive_probability_maps=probabilities,
        target_masks=targets,
        valid_masks=valid,
        dataset="sen1floods11",
        model=model_name,
        split=split,
        protocol=protocol,
        model_lineage=resolved_model_lineage,
        dataset_lineage=resolved_dataset_lineage,
        independent_unit_column="independent_unit_id",
    )


__all__ = [
    "Sen1FormalizationError",
    "calibrate_common_sen1_spatial_blocks",
    "finalize_sen1_probability_export",
    "finalize_sen1floods11_segmentation",
    "load_sen1_probability_units",
    "write_sen1_probability_export",
]
