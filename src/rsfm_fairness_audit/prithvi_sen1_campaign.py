from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.prithvi import PrithviSen1Floods11TLAdapter
from rsfm_fairness_audit.adapters.sen1floods11 import Sen1Floods11DatasetAdapter
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output
from rsfm_fairness_audit.sen1floods11_formal import write_sen1_probability_export
from rsfm_fairness_audit.sen1_prithvi_mask_gate import (
    gate_prithvi_prepared_masks,
)
from rsfm_fairness_audit.terramind_sen1_config import read_sen1floods11_split_prefixes


class PrithviSen1CampaignError(RuntimeError):
    """Raised when the official Prithvi migration cannot preserve split identity."""


@dataclass(frozen=True)
class PrithviSen1CampaignConfig:
    prepared_data_root: Path
    bolivia_prepared_data_root: Path
    model_config: Path
    train_split: Path
    validation_split: Path
    test_split: Path
    bolivia_split: Path
    output_dir: Path
    persistent_output_dir: Path | None = None
    prepared_metadata_csv: Path | None = None
    bolivia_prepared_metadata_csv: Path | None = None
    batch_size: int = 1
    device: str = "auto"
    diagnostic_max_samples: int | None = None


def _canonical(value: Any) -> str:
    name = Path(str(value or "")).stem
    for suffix in (
        "_S2Hand",
        "_S1Hand",
        "_LabelHand",
        "_S2",
        "_S1",
        "_Label",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _row_prefix(row: Mapping[str, Any]) -> str:
    for key in (
        "sample_id",
        "chip_id",
        "source_s2_path",
        "source_image_path",
        "image_path",
        "chip_path",
    ):
        value = _canonical(row.get(key))
        if value:
            return value
    return ""


def match_prepared_rows_to_official_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_split: str | Path,
    validation_split: str | Path,
    test_split: str | Path,
) -> dict[str, list[int]]:
    by_prefix: dict[str, int] = {}
    for index, row in enumerate(rows):
        prefix = _row_prefix(row)
        if not prefix:
            raise PrithviSen1CampaignError(f"Prepared row {index} has no canonical sample prefix.")
        if prefix in by_prefix:
            raise PrithviSen1CampaignError(f"Duplicate prepared sample prefix: {prefix}")
        by_prefix[prefix] = index
    output: dict[str, list[int]] = {}
    split_paths = (
        ("train", train_split),
        ("validation", validation_split),
        ("test", test_split),
    )
    expected_counts = {"train": 252, "validation": 89, "test": 90}
    split_prefixes: dict[str, list[str]] = {}
    for split, path in split_paths:
        prefixes = read_sen1floods11_split_prefixes(path)
        split_prefixes[split] = prefixes
        if len(prefixes) != expected_counts[split]:
            raise PrithviSen1CampaignError(
                f"Official Prithvi {split} split must contain "
                f"{expected_counts[split]} samples, observed={len(prefixes)}."
            )
        missing = [prefix for prefix in prefixes if prefix not in by_prefix]
        if missing:
            raise PrithviSen1CampaignError(
                f"Prepared Prithvi data does not contain {split} members: {missing[:10]}"
            )
        output[split] = [by_prefix[prefix] for prefix in prefixes]
    split_names = tuple(output)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = set(output[left]) & set(output[right])
            if overlap:
                raise PrithviSen1CampaignError(
                    f"Prepared Prithvi {left} and {right} rows overlap."
                )
    selected = set().union(*(set(indices) for indices in output.values()))
    if len(rows) != 431 or selected != set(range(len(rows))):
        raise PrithviSen1CampaignError(
            "The immutable Prithvi core asset must contain exactly the 431 "
            "official train/validation/standard-test samples."
        )
    event = lambda prefix: str(prefix).split("_", 1)[0]
    core_events = {
        event(prefix)
        for prefixes in split_prefixes.values()
        for prefix in prefixes
    }
    test_events = {event(prefix) for prefix in split_prefixes["test"]}
    if (
        "Bolivia" in core_events
        or len(core_events) != 10
        or test_events != core_events
    ):
        raise PrithviSen1CampaignError(
            "The Prithvi core must contain 10 non-Bolivia events and standard "
            f"test must cover all 10: core={sorted(core_events)}, "
            f"test={sorted(test_events)}."
        )
    return output


def match_prepared_rows_to_bolivia_holdout(
    rows: Sequence[Mapping[str, Any]],
    *,
    bolivia_split: str | Path,
) -> list[int]:
    by_prefix: dict[str, int] = {}
    for index, row in enumerate(rows):
        prefix = _row_prefix(row)
        if not prefix:
            raise PrithviSen1CampaignError(
                f"Bolivia prepared row {index} has no canonical sample prefix."
            )
        if prefix in by_prefix:
            raise PrithviSen1CampaignError(
                f"Duplicate Bolivia prepared sample prefix: {prefix}"
            )
        by_prefix[prefix] = index
    prefixes = read_sen1floods11_split_prefixes(bolivia_split)
    if len(prefixes) != 15 or {prefix.split("_", 1)[0] for prefix in prefixes} != {
        "Bolivia"
    }:
        raise PrithviSen1CampaignError(
            "The independent Prithvi Bolivia split must contain exactly 15 Bolivia samples."
        )
    missing = [prefix for prefix in prefixes if prefix not in by_prefix]
    if missing:
        raise PrithviSen1CampaignError(
            f"Prepared Bolivia supplement is missing holdout members: {missing[:10]}"
        )
    extra = sorted(set(by_prefix) - set(prefixes))
    if extra:
        raise PrithviSen1CampaignError(
            "Prepared Bolivia supplement must remain a separate exact 15-chip "
            f"asset; unexpected rows={extra[:10]}."
        )
    return [by_prefix[prefix] for prefix in prefixes]


def _predict_indices(
    dataset: Sen1Floods11DatasetAdapter,
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    model: PrithviSen1Floods11TLAdapter,
    *,
    batch_size: int,
    capture_full_probabilities: bool = False,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[Any],
    list[dict[str, Any]],
    dict[str, Any],
    np.ndarray | None,
]:
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    filenames: list[Any] = []
    metadata: list[dict[str, Any]] = []
    captured: list[np.ndarray] = []
    maximum_probability_sum_error = 0.0
    for start in range(0, len(indices), batch_size):
        batch_indices = list(indices[start : start + batch_size])
        samples = [dataset.load_sample(index) for index in batch_indices]
        batch_rows = [dict(rows[index]) for index in batch_indices]
        prepared = model.preprocess({"samples": samples, "metadata": batch_rows})
        prediction = model.predict_segmentation(prepared)
        full_probabilities = np.asarray(prediction.get("probabilities"), dtype=np.float32)
        if full_probabilities.ndim != 4 or full_probabilities.shape[1] != 2:
            raise PrithviSen1CampaignError(
                f"Official Prithvi output must contain [B,2,H,W] probabilities, got {full_probabilities.shape}."
            )
        if not np.all(np.isfinite(full_probabilities)):
            raise PrithviSen1CampaignError("Official Prithvi probabilities contain NaN/Inf.")
        if np.min(full_probabilities) < -1e-6 or np.max(full_probabilities) > 1.0 + 1e-6:
            raise PrithviSen1CampaignError("Official Prithvi probabilities fall outside [0,1].")
        probability_sum_error = float(
            np.max(np.abs(np.sum(full_probabilities, axis=1) - 1.0))
        )
        maximum_probability_sum_error = max(
            maximum_probability_sum_error,
            probability_sum_error,
        )
        if probability_sum_error > 1e-5:
            raise PrithviSen1CampaignError(
                "Official Prithvi class probabilities do not sum to one; "
                f"maximum_error={probability_sum_error}."
            )
        masks = np.asarray(prepared["masks"])
        if masks.shape[0] != full_probabilities.shape[0]:
            raise PrithviSen1CampaignError("Prithvi target/probability batch sizes differ.")
        if tuple(masks.shape[-2:]) != tuple(full_probabilities.shape[-2:]):
            raise PrithviSen1CampaignError(
                "Prithvi target/probability spatial shapes differ: "
                f"targets={masks.shape}, probabilities={full_probabilities.shape}."
            )
        if capture_full_probabilities:
            captured.append(full_probabilities.copy())
        for local, row in enumerate(batch_rows):
            prefix = _row_prefix(row)
            probabilities.append(full_probabilities[local, 1])
            targets.append(np.asarray(masks[local]).squeeze())
            filenames.append(
                {
                    "S2L1C": str(
                        row.get("source_s2_path")
                        or row.get("source_image_path")
                        or row.get("image_path")
                        or prefix
                    )
                }
            )
            metadata.append(
                {
                    "event_id": str(
                        row.get("event_id")
                        or row.get("event")
                        or row.get("region")
                        or prefix.split("_", 1)[0]
                    ),
                    "country": str(row.get("country") or row.get("ISO_CC") or ""),
                    "region": str(row.get("region") or row.get("event_id") or ""),
                }
            )
    validation = {
        "row_count": len(probabilities),
        "full_probability_layout": "[B,2,H,W]",
        "probabilities_finite": True,
        "probabilities_in_unit_interval": True,
        "maximum_probability_sum_error": maximum_probability_sum_error,
        "target_shape_matches_probability_shape": True,
    }
    full_bundle = np.concatenate(captured, axis=0) if captured else None
    return probabilities, targets, filenames, metadata, validation, full_bundle


def run_prithvi_sen1_probability_campaign(
    config: PrithviSen1CampaignConfig,
) -> dict[str, Path]:
    """Re-export official Prithvi probability maps on the common split.

    This stage deliberately stops before GeoBWER. The all-model validation
    exports must first be used to freeze one common spatial block calibration.
    """

    hydrate_output(config.output_dir, config.persistent_output_dir)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dataset = Sen1Floods11DatasetAdapter(
        config.prepared_data_root,
        metadata_path=config.prepared_metadata_csv,
        split="all",
    )
    rows = dataset.load_metadata()
    bolivia_dataset = Sen1Floods11DatasetAdapter(
        config.bolivia_prepared_data_root,
        metadata_path=config.bolivia_prepared_metadata_csv,
        split="all",
    )
    bolivia_rows = bolivia_dataset.load_metadata()
    prepared_mask_gate = gate_prithvi_prepared_masks(
        core_root=config.prepared_data_root,
        core_metadata=config.prepared_metadata_csv,
        bolivia_root=config.bolivia_prepared_data_root,
        bolivia_metadata=config.bolivia_prepared_metadata_csv,
    )
    mask_gate_path = output / "pre_model_prepared_mask_gate.json"
    mask_gate_path.write_text(
        json.dumps(prepared_mask_gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    split_indices = match_prepared_rows_to_official_splits(
        rows,
        train_split=config.train_split,
        validation_split=config.validation_split,
        test_split=config.test_split,
    )
    bolivia_indices = match_prepared_rows_to_bolivia_holdout(
        bolivia_rows,
        bolivia_split=config.bolivia_split,
    )
    core_prefixes = {_row_prefix(row) for row in rows}
    holdout_prefixes = {
        _row_prefix(bolivia_rows[index]) for index in bolivia_indices
    }
    overlap = sorted(core_prefixes & holdout_prefixes)
    if overlap:
        raise PrithviSen1CampaignError(
            f"Core 431 prepared data overlaps the Bolivia supplement: {overlap[:10]}"
        )
    if len(core_prefixes | holdout_prefixes) != 446:
        raise PrithviSen1CampaignError(
            "Prithvi core plus Bolivia supplement must contain exactly 446 "
            "distinct hand-labeled samples."
        )
    split_indices["bolivia_holdout"] = bolivia_indices
    if config.diagnostic_max_samples:
        split_indices = {
            key: (
                values
                if key == "train"
                else values[: int(config.diagnostic_max_samples)]
            )
            for key, values in split_indices.items()
        }
    values = load_yaml(config.model_config)
    values["device"] = config.device
    values["batch_size"] = config.batch_size
    model = PrithviSen1Floods11TLAdapter.from_config(values)
    model.load_model()
    resolved_device = model._resolve_device()
    parameter_device = "not_reported"
    if hasattr(model.model, "parameters"):
        try:
            parameter_device = str(next(model.model.parameters()).device)
        except (StopIteration, TypeError):
            pass
    try:
        import torch

        gpu_name = (
            torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else "not_applicable"
        )
    except ImportError:  # pragma: no cover - load_model already requires torch
        gpu_name = "unavailable"
    print(
        f"[prithvi:sen1:device] resolved={resolved_device} gpu={gpu_name} "
        f"model_parameter_device={parameter_device}",
        flush=True,
    )
    exports: dict[str, Path] = {}
    split_runtime_validation: dict[str, dict[str, Any]] = {}
    diagnostic_full_probability_artifacts: dict[str, str] = {}
    for split in ("validation", "test", "bolivia_holdout"):
        active_dataset = bolivia_dataset if split == "bolivia_holdout" else dataset
        active_rows = bolivia_rows if split == "bolivia_holdout" else rows
        probabilities, targets, filenames, metadata, runtime_validation, full_bundle = _predict_indices(
            active_dataset,
            active_rows,
            split_indices[split],
            model,
            batch_size=config.batch_size,
            capture_full_probabilities=config.diagnostic_max_samples is not None,
        )
        split_runtime_validation[split] = runtime_validation
        if full_bundle is not None:
            diagnostic_dir = output / "diagnostic_probe"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            diagnostic_path = diagnostic_dir / f"{split}_full_probabilities.npz"
            sample_ids = np.asarray(
                [
                    _row_prefix(active_rows[index])
                    for index in split_indices[split]
                ],
                dtype=np.str_,
            )
            np.savez_compressed(
                diagnostic_path,
                probabilities=full_bundle.astype(np.float32),
                targets=np.stack(targets).astype(np.int16),
                sample_ids=sample_ids,
                split_role=np.asarray([split] * len(sample_ids), dtype=np.str_),
            )
            diagnostic_full_probability_artifacts[split] = str(diagnostic_path)
        exports[split] = write_sen1_probability_export(
            output / "probabilities" / split,
            probabilities=probabilities,
            targets=targets,
            filenames=filenames,
            metadata=metadata,
            batch_size=config.batch_size,
        )
        persist_output(
            exports[split],
            (
                config.persistent_output_dir / "probabilities" / split
                if config.persistent_output_dir
                else None
            ),
            label=f"prithvi-{split}-probabilities",
        )
    checkpoint_path = Path(str(getattr(model, "checkpoint_path", "") or ""))
    manifest = output / "campaign_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.prithvi_tl_probability_migration.v3",
                "formal_evidence": config.diagnostic_max_samples is None,
                "role": "task_specific_external_validity_reference",
                "model": model.protocol_model_name,
                "model_family": model.model_family,
                "adaptation_protocol": model.adaptation_protocol,
                "split_protocol": "official_252_89_90_plus_15_bolivia_holdout",
                "train_count": len(split_indices["train"]),
                "validation_count": len(split_indices["validation"]),
                "test_count": len(split_indices["test"]),
                "bolivia_holdout_count": len(split_indices["bolivia_holdout"]),
                "combined_evaluation_count": (
                    len(split_indices["test"])
                    + len(split_indices["bolivia_holdout"])
                ),
                "bolivia_holdout_used_for_training_or_calibration": False,
                "no_training_or_calibration_leakage": True,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": (
                    file_sha256(checkpoint_path) if checkpoint_path.is_file() else ""
                ),
                "model_load_diagnostics": getattr(model, "load_diagnostics", {}),
                "device_contract": getattr(model, "device_contract", {}),
                "inference_debug_records": getattr(model, "debug_records", []),
                "split_runtime_validation": split_runtime_validation,
                "pre_model_prepared_mask_gate": {
                    "path": str(mask_gate_path),
                    "sha256": file_sha256(mask_gate_path),
                    "schema": prepared_mask_gate["schema"],
                    "status": prepared_mask_gate["status"],
                    "model_loaded_by_gate": False,
                },
                "diagnostic_full_probability_artifacts": diagnostic_full_probability_artifacts,
                "probability_exports": {key: str(value) for key, value in exports.items()},
                "config": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    persist_output(output, config.persistent_output_dir, label="prithvi-probability-migration-complete")
    return {
        "validation_export": exports["validation"],
        "test_export": exports["test"],
        "bolivia_holdout_export": exports["bolivia_holdout"],
        "manifest": manifest,
    }


__all__ = [
    "PrithviSen1CampaignConfig",
    "PrithviSen1CampaignError",
    "match_prepared_rows_to_official_splits",
    "match_prepared_rows_to_bolivia_holdout",
    "run_prithvi_sen1_probability_campaign",
]
