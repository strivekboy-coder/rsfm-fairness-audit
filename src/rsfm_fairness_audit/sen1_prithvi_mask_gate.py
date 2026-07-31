from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.formal_outputs import file_sha256


class Sen1PrithviMaskGateError(RuntimeError):
    """Raised when a prepared Sen1Floods11 mask violates its frozen contract."""


EXPECTED_MASK_SHAPE = (224, 224)
ALLOWED_MASK_VALUES = frozenset({-1, 0, 1})


def validate_prepared_mask(
    value: Any,
    *,
    stage: str,
    expected_shape: tuple[int, int] = EXPECTED_MASK_SHAPE,
) -> np.ndarray:
    """Return a canonical integer mask after strict semantic validation."""

    array = np.asarray(value)
    if tuple(array.shape) != tuple(expected_shape):
        raise Sen1PrithviMaskGateError(
            f"{stage}: mask shape must be {expected_shape}, observed={array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise Sen1PrithviMaskGateError(f"{stage}: mask must be finite numeric data.")
    if not np.all(array == np.round(array)):
        raise Sen1PrithviMaskGateError(f"{stage}: mask must be integer-valued.")
    values = {int(item) for item in np.unique(array).tolist()}
    if not values.issubset(ALLOWED_MASK_VALUES):
        raise Sen1PrithviMaskGateError(
            f"{stage}: mask values must be a subset of [-1,0,1], observed={sorted(values)}."
        )
    return np.asarray(array, dtype=np.int16)


def validate_source_label(value: Any, *, stage: str) -> np.ndarray:
    """Validate native-resolution LabelHand semantics before nearest resizing."""

    array = np.asarray(value)
    if array.ndim != 2 or not all(int(size) > 0 for size in array.shape):
        raise Sen1PrithviMaskGateError(
            f"{stage}: source LabelHand must be a non-empty 2-D array, observed={array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise Sen1PrithviMaskGateError(f"{stage}: source LabelHand must be finite.")
    if not np.all(array == np.round(array)):
        raise Sen1PrithviMaskGateError(
            f"{stage}: source LabelHand must be integer-valued."
        )
    values = {int(item) for item in np.unique(array).tolist()}
    if not values.issubset(ALLOWED_MASK_VALUES):
        raise Sen1PrithviMaskGateError(
            f"{stage}: source LabelHand contains invalid values {sorted(values)}."
        )
    return array


def read_mask_npz(
    path: str | Path,
    *,
    expected_shape: tuple[int, int] = EXPECTED_MASK_SHAPE,
) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise Sen1PrithviMaskGateError(f"Prepared mask NPZ is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if "mask" not in bundle.files:
                raise Sen1PrithviMaskGateError(
                    f"Prepared mask NPZ lacks the explicit 'mask' key: {path}"
                )
            mask = np.asarray(bundle["mask"])
    except Sen1PrithviMaskGateError:
        raise
    except Exception as exc:
        raise Sen1PrithviMaskGateError(
            f"Cannot read prepared mask NPZ: {path}"
        ) from exc
    return validate_prepared_mask(
        mask, stage=f"prepared_npz[{path}]", expected_shape=expected_shape
    )


def write_verified_mask_npz(
    path: str | Path,
    mask: Any,
    *,
    expected_shape: tuple[int, int] = EXPECTED_MASK_SHAPE,
) -> None:
    """Write a new mask NPZ and prove exact round-trip equality."""

    path = Path(path)
    if path.exists():
        raise Sen1PrithviMaskGateError(
            f"Refusing to overwrite prepared mask NPZ: {path}"
        )
    before = validate_prepared_mask(
        mask, stage=f"pre_write[{path}]", expected_shape=expected_shape
    )
    np.savez_compressed(path, mask=before)
    after = read_mask_npz(path, expected_shape=expected_shape)
    if not np.array_equal(before, after):
        raise Sen1PrithviMaskGateError(
            f"Prepared mask changed across NPZ round-trip: {path}"
        )


def _metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise Sen1PrithviMaskGateError(f"Prepared metadata is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise Sen1PrithviMaskGateError(f"Prepared metadata is empty: {path}")
    return rows


def _canonical(value: Any) -> str:
    name = Path(str(value or "")).stem
    for suffix in ("_S2Hand", "_S1Hand", "_LabelHand", "_S2", "_S1", "_Label"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _row_id(row: Mapping[str, Any]) -> str:
    for key in ("sample_id", "chip_id", "source_s2_path", "chip_path"):
        value = _canonical(row.get(key))
        if value:
            return value
    return ""


def _sequence_sha(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def audit_prepared_mask_root(
    root: str | Path,
    *,
    metadata_path: str | Path | None = None,
    expected_count: int,
    role: str,
    expected_shape: tuple[int, int] = EXPECTED_MASK_SHAPE,
) -> dict[str, Any]:
    root = Path(root).resolve()
    metadata = Path(metadata_path).resolve() if metadata_path else root / "metadata.csv"
    rows = _metadata(metadata)
    if len(rows) != expected_count:
        raise Sen1PrithviMaskGateError(
            f"{role}: expected {expected_count} metadata rows, observed={len(rows)}."
        )
    sample_ids: list[str] = []
    records: list[dict[str, Any]] = []
    target_digest = hashlib.sha256()
    for row in rows:
        sample_id = _row_id(row)
        relative = str(row.get("mask_path") or "").strip()
        if not sample_id or not relative:
            raise Sen1PrithviMaskGateError(
                f"{role}: metadata row lacks sample_id or mask_path."
            )
        path = Path(relative)
        if not path.is_absolute():
            path = root / path
        mask = read_mask_npz(path, expected_shape=expected_shape)
        sample_ids.append(sample_id)
        contiguous = np.ascontiguousarray(mask, dtype=np.int16)
        target_digest.update(sample_id.encode("utf-8"))
        target_digest.update(b"\0")
        target_digest.update(contiguous.tobytes())
        records.append(
            {
                "sample_id": sample_id,
                "event": sample_id.split("_", 1)[0],
                "mask_path": str(path),
                "mask_sha256": file_sha256(path),
                "mask_shape": list(contiguous.shape),
                "mask_unique_values": sorted(
                    int(item) for item in np.unique(contiguous).tolist()
                ),
            }
        )
    if len(set(sample_ids)) != expected_count:
        raise Sen1PrithviMaskGateError(
            f"{role}: prepared sample IDs are empty or duplicated."
        )
    return {
        "role": role,
        "root": str(root),
        "metadata_path": str(metadata),
        "metadata_sha256": file_sha256(metadata),
        "sample_count": expected_count,
        "sample_order_sha256": _sequence_sha(sample_ids),
        "sample_set_sha256": _sequence_sha(sorted(sample_ids)),
        "target_sha256": target_digest.hexdigest(),
        "events": sorted({item["event"] for item in records}),
        "records": records,
    }


def gate_prithvi_prepared_masks(
    *,
    core_root: str | Path,
    bolivia_root: str | Path,
    core_metadata: str | Path | None = None,
    bolivia_metadata: str | Path | None = None,
    core_count: int = 431,
    bolivia_count: int = 15,
) -> dict[str, Any]:
    """Hard-gate all 431+15 masks before any Prithvi model is loaded."""

    core = audit_prepared_mask_root(
        core_root,
        metadata_path=core_metadata,
        expected_count=core_count,
        role="core_431",
    )
    bolivia = audit_prepared_mask_root(
        bolivia_root,
        metadata_path=bolivia_metadata,
        expected_count=bolivia_count,
        role="bolivia_15",
    )
    core_ids = {item["sample_id"] for item in core["records"]}
    bolivia_ids = {item["sample_id"] for item in bolivia["records"]}
    if core_ids & bolivia_ids:
        raise Sen1PrithviMaskGateError(
            "Core and Bolivia prepared mask assets overlap."
        )
    if "Bolivia" in core["events"] or bolivia["events"] != ["Bolivia"]:
        raise Sen1PrithviMaskGateError(
            "Prepared mask event contract must be 10 non-Bolivia core events "
            "plus a Bolivia-only holdout."
        )
    if len(core["events"]) != 10 or len(core_ids | bolivia_ids) != core_count + bolivia_count:
        raise Sen1PrithviMaskGateError(
            "Prepared mask assets do not form the frozen 446-chip, 11-event universe."
        )
    return {
        "schema": "geobwer.sen1floods11.prithvi_prepared_mask_gate.v1",
        "status": "pass",
        "mask_contract": {
            "shape": list(EXPECTED_MASK_SHAPE),
            "finite": True,
            "integer": True,
            "allowed_values": [-1, 0, 1],
            "npz_required_key": "mask",
        },
        "core": core,
        "bolivia": bolivia,
        "combined_sample_count": core_count + bolivia_count,
        "split_overlap_count": 0,
        "model_loaded_by_gate": False,
        "blocking_errors": [],
    }


__all__ = [
    "EXPECTED_MASK_SHAPE",
    "Sen1PrithviMaskGateError",
    "audit_prepared_mask_root",
    "gate_prithvi_prepared_masks",
    "read_mask_npz",
    "validate_prepared_mask",
    "validate_source_label",
    "write_verified_mask_npz",
]
