from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SEN1_IMPUTATION_POLICY = "official_train_band_mean_normalized_zero"
SEN1_MODE_MODALITY_SLICES: dict[str, dict[str, slice]] = {
    "S1": {"S1": slice(0, 2)},
    "S2": {"S2": slice(0, 13)},
    "S1+S2": {"S1": slice(0, 2), "S2": slice(2, 15)},
}
EVALUATION_SPLIT_ROLES = {
    "test",
    "standard_test",
    "bolivia_holdout",
    "combined_held_out",
}
FORMAL_COMPLETE_MODALITY_MISSING_EXPECTATION = {
    "S1": {
        "train": {},
        "validation": {},
        "test": {"Paraguay_34417": ["S1"]},
        "bolivia_holdout": {},
    },
    "S2": {
        "train": {},
        "validation": {},
        "test": {},
        "bolivia_holdout": {},
    },
    "S1+S2": {
        "train": {},
        "validation": {},
        "test": {"Paraguay_34417": ["S1"]},
        "bolivia_holdout": {},
    },
}


class Sen1InputQualityError(RuntimeError):
    """Raised when a Sen1Floods11 input violates the frozen missing-data contract."""


def _modality_record(
    value: np.ndarray,
    *,
    modality: str,
    channel_offset: int,
) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3:
        raise Sen1InputQualityError(
            f"Expected {modality} input [C,H,W], got {array.shape}."
        )
    flat = array.reshape(array.shape[0], -1)
    finite = np.isfinite(flat)
    channels: list[dict[str, int]] = []
    for local_index in range(array.shape[0]):
        channel = flat[local_index]
        nan_count = int(np.isnan(channel).sum())
        posinf_count = int(np.isposinf(channel).sum())
        neginf_count = int(np.isneginf(channel).sum())
        channels.append(
            {
                "channel_index": int(channel_offset + local_index),
                "modality_channel_index": int(local_index),
                "nan_count": nan_count,
                "posinf_count": posinf_count,
                "neginf_count": neginf_count,
                "nonfinite_count": nan_count + posinf_count + neginf_count,
                "finite_count": int(np.isfinite(channel).sum()),
                "value_count": int(channel.size),
            }
        )
    nonfinite_count = int(sum(item["nonfinite_count"] for item in channels))
    finite_count = int(sum(item["finite_count"] for item in channels))
    fully_missing = finite_count == 0
    if fully_missing:
        status = "fully_missing"
    elif nonfinite_count:
        status = "partial_nonfinite_imputed"
    else:
        status = "available"
    return {
        "modality": str(modality),
        "availability_status": status,
        "fully_missing_modality": bool(fully_missing),
        "partial_nonfinite_imputed": bool(nonfinite_count and not fully_missing),
        "finite_count": finite_count,
        "nan_count": int(np.isnan(flat).sum()),
        "posinf_count": int(np.isposinf(flat).sum()),
        "neginf_count": int(np.isneginf(flat).sum()),
        "nonfinite_count": nonfinite_count,
        "value_count": int(flat.size),
        "pixel_count": int(flat.shape[1]),
        "jointly_finite_pixel_count": int(np.all(finite, axis=0).sum()),
        "channel_counts": channels,
    }


def audit_mode_input(
    image: np.ndarray,
    *,
    prefix: str,
    mode: str,
    split_role: str,
) -> dict[str, Any]:
    """Audit a concatenated U-Net input independently within each modality."""

    normalized_mode = str(mode).upper().replace(" ", "")
    if normalized_mode not in SEN1_MODE_MODALITY_SLICES:
        raise Sen1InputQualityError(f"Unsupported Sen1 sensor mode: {mode}.")
    value = np.asarray(image, dtype=np.float32)
    expected_channels = max(
        int(modality_slice.stop)
        for modality_slice in SEN1_MODE_MODALITY_SLICES[normalized_mode].values()
    )
    if value.ndim != 3 or value.shape[0] != expected_channels:
        raise Sen1InputQualityError(
            f"Expected {expected_channels} channels for {normalized_mode}, "
            f"got {value.shape} at prefix={prefix}."
        )
    modality_records = [
        _modality_record(
            value[modality_slice],
            modality=modality,
            channel_offset=int(modality_slice.start or 0),
        )
        for modality, modality_slice in SEN1_MODE_MODALITY_SLICES[
            normalized_mode
        ].items()
    ]
    fully_missing_modalities = [
        str(item["modality"])
        for item in modality_records
        if item["fully_missing_modality"]
    ]
    partial_modalities = [
        str(item["modality"])
        for item in modality_records
        if item["partial_nonfinite_imputed"]
    ]
    role = str(split_role)
    if fully_missing_modalities and role not in EVALUATION_SPLIT_ROLES:
        raise Sen1InputQualityError(
            "Complete required-modality absence is allowed only for frozen "
            "evaluation splits: "
            f"prefix={prefix}, mode={normalized_mode}, split_role={role}, "
            f"fully_missing_modalities={fully_missing_modalities}."
        )
    if fully_missing_modalities:
        availability_status = "fully_missing_modality"
    elif partial_modalities:
        availability_status = "partial_nonfinite_imputed"
    else:
        availability_status = "available"
    channel_counts = [
        dict(channel)
        for modality_record in modality_records
        for channel in modality_record["channel_counts"]
    ]
    finite = np.isfinite(value.reshape(value.shape[0], -1))
    imputed_value_count = int(
        sum(item["nonfinite_count"] for item in modality_records)
    )
    return {
        "prefix": str(prefix),
        "sample_id": str(prefix),
        "sensor_mode": normalized_mode,
        "split_role": role,
        "availability_status": availability_status,
        "fully_missing_modality": bool(fully_missing_modalities),
        "fully_missing_modalities": fully_missing_modalities,
        "partial_nonfinite_imputed": bool(partial_modalities),
        "partial_nonfinite_modalities": partial_modalities,
        "imputation_policy": SEN1_IMPUTATION_POLICY,
        "modalities": modality_records,
        # Retain the pre-v0.4.27 fields for downstream readers while making
        # clear that the joint count is descriptive, not an availability gate.
        "channel_counts": channel_counts,
        "jointly_finite_pixel_count": int(np.all(finite, axis=0).sum()),
        "pixel_count": int(finite.shape[1]),
        "imputed_value_count": imputed_value_count,
        "imputed_fraction": float(imputed_value_count / value.size),
    }


def normalize_mode_input(
    image: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    prefix: str,
    mode: str,
    split_role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mean-impute each non-finite value and apply frozen band normalization."""

    raw = np.asarray(image, dtype=np.float32)
    quality = audit_mode_input(
        raw,
        prefix=prefix,
        mode=mode,
        split_role=split_role,
    )
    channel_count = int(raw.shape[0])
    frozen_mean = np.asarray(mean, dtype=np.float32).reshape(channel_count, 1, 1)
    frozen_std = np.maximum(
        np.asarray(std, dtype=np.float32).reshape(channel_count, 1, 1),
        1e-6,
    )
    imputed = np.where(np.isfinite(raw), raw, frozen_mean)
    normalized = (imputed - frozen_mean) / frozen_std
    if not np.all(np.isfinite(normalized)):
        raise Sen1InputQualityError(
            "Input remains non-finite after official-train mean imputation: "
            f"prefix={prefix}, mode={mode}, split_role={split_role}."
        )
    return np.asarray(normalized, dtype=np.float32), quality


def input_quality_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    channel_count = max(
        int(value.stop)
        for value in SEN1_MODE_MODALITY_SLICES[
            str(mode).upper().replace(" ", "")
        ].values()
    )
    channel_totals = [
        {
            "channel_index": channel_index,
            "nan_count": 0,
            "posinf_count": 0,
            "neginf_count": 0,
            "nonfinite_count": 0,
            "finite_count": 0,
            "value_count": 0,
        }
        for channel_index in range(channel_count)
    ]
    aggregate_imputed_value_count = 0
    samples_with_imputation = 0
    maximum_imputed_fraction = 0.0
    fully_missing_by_sample: dict[str, list[str]] = {}
    partial_by_sample: dict[str, list[str]] = {}
    modality_totals: dict[str, dict[str, int]] = {}
    for record in records:
        imputed = int(record["imputed_value_count"])
        aggregate_imputed_value_count += imputed
        samples_with_imputation += int(imputed > 0)
        maximum_imputed_fraction = max(
            maximum_imputed_fraction,
            float(record["imputed_fraction"]),
        )
        sample_id = str(record.get("sample_id", record["prefix"]))
        fully_missing = list(map(str, record.get("fully_missing_modalities", [])))
        partial = list(map(str, record.get("partial_nonfinite_modalities", [])))
        if fully_missing:
            fully_missing_by_sample[sample_id] = fully_missing
        if partial:
            partial_by_sample[sample_id] = partial
        for modality in record.get("modalities", []):
            target = modality_totals.setdefault(
                str(modality["modality"]),
                {
                    "sample_count": 0,
                    "fully_missing_modality_count": 0,
                    "partial_nonfinite_sample_count": 0,
                    "nonfinite_count": 0,
                },
            )
            target["sample_count"] += 1
            target["fully_missing_modality_count"] += int(
                modality["fully_missing_modality"]
            )
            target["partial_nonfinite_sample_count"] += int(
                modality["partial_nonfinite_imputed"]
            )
            target["nonfinite_count"] += int(modality["nonfinite_count"])
        for channel in record["channel_counts"]:
            target = channel_totals[int(channel["channel_index"])]
            for key in (
                "nan_count",
                "posinf_count",
                "neginf_count",
                "nonfinite_count",
                "finite_count",
                "value_count",
            ):
                target[key] += int(channel[key])
    return {
        "sample_count": len(records),
        "samples_with_imputation": samples_with_imputation,
        "aggregate_imputed_value_count": aggregate_imputed_value_count,
        "maximum_imputed_fraction": maximum_imputed_fraction,
        "fully_missing_modality_count": int(
            sum(len(value) for value in fully_missing_by_sample.values())
        ),
        "fully_missing_sample_count": len(fully_missing_by_sample),
        "fully_missing_sample_ids": sorted(fully_missing_by_sample),
        "fully_missing_modalities_by_sample": {
            key: fully_missing_by_sample[key]
            for key in sorted(fully_missing_by_sample)
        },
        "partial_nonfinite_sample_count": len(partial_by_sample),
        "partial_nonfinite_sample_ids": sorted(partial_by_sample),
        "partial_nonfinite_modalities_by_sample": {
            key: partial_by_sample[key] for key in sorted(partial_by_sample)
        },
        "modality_totals": modality_totals,
        "channel_totals": channel_totals,
    }


def normalize_named_modalities(
    arrays: Mapping[str, np.ndarray],
    *,
    means: Mapping[str, Sequence[float]],
    stds: Mapping[str, Sequence[float]],
    prefix: str,
    split_role: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Normalize TerraMind modality arrays using the same frozen-mean policy."""

    ordered_modalities = [str(key) for key in arrays]
    if not ordered_modalities:
        raise Sen1InputQualityError("At least one modality is required.")
    canonical = {"S1GRD": "S1", "S2L1C": "S2"}
    mode = (
        "S1+S2"
        if set(ordered_modalities) == {"S1GRD", "S2L1C"}
        else canonical[ordered_modalities[0]]
    )
    normalized: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    channel_offset = 0
    for key in ordered_modalities:
        raw = np.asarray(arrays[key], dtype=np.float32)
        modality = canonical.get(key, key)
        record = _modality_record(
            raw,
            modality=modality,
            channel_offset=channel_offset,
        )
        records.append(record)
        channel_offset += int(raw.shape[0])
        frozen_mean = np.asarray(means[key], dtype=np.float32).reshape(
            raw.shape[0], 1, 1
        )
        frozen_std = np.maximum(
            np.asarray(stds[key], dtype=np.float32).reshape(raw.shape[0], 1, 1),
            1e-6,
        )
        output = (
            np.where(np.isfinite(raw), raw, frozen_mean) - frozen_mean
        ) / frozen_std
        if not np.all(np.isfinite(output)):
            raise Sen1InputQualityError(
                f"TerraMind input remains non-finite: prefix={prefix}, modality={key}."
            )
        normalized[key] = np.asarray(output, dtype=np.float32)
    fully_missing = [
        str(item["modality"]) for item in records if item["fully_missing_modality"]
    ]
    if fully_missing and str(split_role) not in EVALUATION_SPLIT_ROLES:
        raise Sen1InputQualityError(
            "Complete required-modality absence is allowed only for frozen "
            "evaluation splits: "
            f"prefix={prefix}, mode={mode}, split_role={split_role}, "
            f"fully_missing_modalities={fully_missing}."
        )
    partial = [
        str(item["modality"])
        for item in records
        if item["partial_nonfinite_imputed"]
    ]
    channel_counts = [
        dict(channel) for record in records for channel in record["channel_counts"]
    ]
    imputed_count = int(sum(item["nonfinite_count"] for item in records))
    total_values = int(sum(item["value_count"] for item in records))
    quality = {
        "prefix": str(prefix),
        "sample_id": str(prefix),
        "sensor_mode": mode,
        "split_role": str(split_role),
        "availability_status": (
            "fully_missing_modality"
            if fully_missing
            else "partial_nonfinite_imputed"
            if partial
            else "available"
        ),
        "fully_missing_modality": bool(fully_missing),
        "fully_missing_modalities": fully_missing,
        "partial_nonfinite_imputed": bool(partial),
        "partial_nonfinite_modalities": partial,
        "imputation_policy": SEN1_IMPUTATION_POLICY,
        "modalities": records,
        "channel_counts": channel_counts,
        "jointly_finite_pixel_count": None,
        "pixel_count": int(next(iter(arrays.values())).shape[-2] * next(iter(arrays.values())).shape[-1]),
        "imputed_value_count": imputed_count,
        "imputed_fraction": float(imputed_count / total_values),
    }
    return normalized, quality


__all__ = [
    "EVALUATION_SPLIT_ROLES",
    "FORMAL_COMPLETE_MODALITY_MISSING_EXPECTATION",
    "SEN1_IMPUTATION_POLICY",
    "SEN1_MODE_MODALITY_SLICES",
    "Sen1InputQualityError",
    "audit_mode_input",
    "input_quality_summary",
    "normalize_mode_input",
    "normalize_named_modalities",
]
