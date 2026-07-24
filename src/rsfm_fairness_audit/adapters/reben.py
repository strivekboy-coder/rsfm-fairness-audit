from __future__ import annotations

import atexit
from contextlib import contextmanager
import importlib
import inspect
import io
import json
import os
import pickle
import re
import threading
import traceback
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter


class RebenDatasetError(RuntimeError):
    """Raised when official BigEarthNet v2.0 / reBEN loading is unavailable or inconsistent."""


REBEN_CONFIGILM_IMAGE_SIZES = {
    ("croma", "S1"): (2, 120, 120),
    ("croma", "S2"): (12, 120, 120),
    ("croma", "S1+S2"): (12, 120, 120),
    ("bifold_resnet101", "S1"): (2, 120, 120),
    ("bifold_resnet101", "S2"): (10, 120, 120),
    ("bifold_resnet101", "S1+S2"): (12, 120, 120),
}

S2_12_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S2_10_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
S1_BANDS = ("VV", "VH")
REBEN_TARGET_SIZE = (120, 120)
REBEN_CLASS_NAMES = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)
REBEN_CLASS_TO_INDEX = {name: index for index, name in enumerate(REBEN_CLASS_NAMES)}
_LMDB_ENV_CACHE: dict[tuple[int, str], Any] = {}
_LMDB_PAYLOAD_FORMAT_CACHE: dict[tuple[int, str], str] = {}
_LMDB_CACHE_LOCK = threading.RLock()


def reben_spatial_lineage(patch_id: Any) -> dict[str, str]:
    """Derive official acquisition/tile identifiers from a reBEN v2 patch id."""

    value = str(patch_id or "").strip()
    tile_match = re.search(r"(?:^|_)(T\d{2}[A-Z]{3})(?:_|$)", value.upper())
    tile = tile_match.group(1) if tile_match else ""
    scene = re.sub(r"_\d{2}_\d{2}$", "", value)
    return {
        "independent_unit_id": value,
        "source_scene_id": scene,
        "source_tile_id": tile,
        "spatial_block_id": tile,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _lmdb_cache_key(path: Path) -> tuple[int, str]:
    return os.getpid(), str(path.resolve())


def _open_lmdb_new(path: Path) -> Any:
    try:
        import lmdb
    except ImportError as exc:
        raise RebenDatasetError("The lmdb package is required for reBEN LMDB loading.") from exc
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=512,
        subdir=path.is_dir(),
    )


def _open_lmdb(path: Path) -> Any:
    """Return the sole long-lived LMDB environment for this path and process."""

    key = _lmdb_cache_key(path)
    with _LMDB_CACHE_LOCK:
        cached = _LMDB_ENV_CACHE.get(key)
        if cached is not None:
            return cached
        env = _open_lmdb_new(path)
        _LMDB_ENV_CACHE[key] = env
        return env


@contextmanager
def _probe_lmdb_environment(path: Path) -> Iterator[Any]:
    """Borrow a cached environment or use one short-lived pre-dataset probe.

    python-lmdb rejects opening the same path twice in one process, especially
    when the open flags differ. Holding the lifecycle lock through the short
    probe prevents a dataset environment from being created concurrently. Once
    a dataset owns the cached environment, every probe borrows that exact
    object and never closes it.
    """

    key = _lmdb_cache_key(path)
    with _LMDB_CACHE_LOCK:
        cached = _LMDB_ENV_CACHE.get(key)
        if cached is not None:
            yield cached
            return
        env = _open_lmdb_new(path)
        try:
            yield env
        finally:
            env.close()


def _close_lmdb_environments() -> None:
    """Close process-owned environments and clear derived payload metadata."""

    with _LMDB_CACHE_LOCK:
        environments = list(_LMDB_ENV_CACHE.values())
        _LMDB_ENV_CACHE.clear()
        _LMDB_PAYLOAD_FORMAT_CACHE.clear()
    seen: set[int] = set()
    for env in environments:
        identity = id(env)
        if identity in seen:
            continue
        seen.add(identity)
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _reset_lmdb_state_after_fork() -> None:
    # A child process must not reuse an Environment inherited from its parent.
    _close_lmdb_environments()


atexit.register(_close_lmdb_environments)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lmdb_state_after_fork)


def _load_safetensors_lmdb_value(value: bytes) -> dict[str, np.ndarray]:
    try:
        from safetensors.numpy import load as load_safetensors
    except ImportError as exc:
        raise RebenDatasetError("safetensors is required for this reBEN LMDB payload format.") from exc
    try:
        return {key: np.asarray(array) for key, array in load_safetensors(value).items()}
    except Exception as exc:
        raise RebenDatasetError(f"Could not decode LMDB value as safetensors: {type(exc).__name__}: {exc}") from exc


def detect_lmdb_payload_format(lmdb_path: str | Path) -> str:
    path = Path(lmdb_path)
    if not path.exists():
        return "missing"
    key = _lmdb_cache_key(path)
    with _LMDB_CACHE_LOCK:
        cached = _LMDB_PAYLOAD_FORMAT_CACHE.get(key)
    if cached is not None:
        return cached

    detected = "unknown"
    with _probe_lmdb_environment(path) as env:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            if not cursor.first():
                detected = "empty"
            else:
                _, value = cursor.item()
                try:
                    pickle.loads(value)
                    detected = "pickle_patch_interface"
                except Exception:
                    try:
                        _load_safetensors_lmdb_value(value)
                        detected = "safetensors"
                    except Exception:
                        detected = "unknown"
    with _LMDB_CACHE_LOCK:
        _LMDB_PAYLOAD_FORMAT_CACHE[key] = detected
    return detected


def resolve_reben_root_dir(images_lmdb: str | Path) -> tuple[Path, Path, list[str]]:
    """Resolve ConfigILM root_dir and BigEarthNetEncoded.lmdb path.

    ConfigILM BEN2DataSet expects root_dir to contain a child named
    BigEarthNetEncoded.lmdb. Some downloaded bundles contain a wrapper folder
    named BigEarthNetEncoded.lmdb that itself contains the actual LMDB folder.
    """
    path = Path(images_lmdb)
    notes: list[str] = []
    child = path / "BigEarthNetEncoded.lmdb"
    if child.exists():
        notes.append("input_path_contains_nested_BigEarthNetEncoded.lmdb; using input path as ConfigILM root_dir")
        return path, child, notes
    if path.name == "BigEarthNetEncoded.lmdb":
        notes.append("input_path_named_BigEarthNetEncoded.lmdb; using parent as ConfigILM root_dir")
        return path.parent, path, notes
    notes.append("input_path_treated_as_ConfigILM_root_dir")
    return path, child, notes


def _read_parquet_frame(path: Path) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RebenDatasetError("pandas + pyarrow are required for reBEN metadata compatibility preflight.") from exc
    return pd.read_parquet(path)


def _candidate_patch_id_columns(columns: Sequence[str]) -> list[str]:
    preferred = ["name", "patch_id", "patch_name", "s2_name", "s1_name"]
    return [column for column in preferred if column in columns]


def _normalise_split(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"validation", "valid"}:
        return "val"
    return text


def reben_labels_to_multihot(labels: Any) -> np.ndarray:
    if hasattr(labels, "tolist") and not isinstance(labels, (str, bytes)):
        labels = labels.tolist()
    if isinstance(labels, str):
        text = labels.strip()
        try:
            labels = json.loads(text.replace("'", '"'))
        except Exception:
            labels = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if labels is None:
        labels = []
    if not isinstance(labels, (list, tuple, set)):
        labels = [labels]
    values = list(labels)
    if len(values) == 19 and all(str(value).strip() in {"0", "1", "0.0", "1.0"} for value in values):
        return np.asarray(values, dtype=np.int64)
    multi_hot = np.zeros(19, dtype=np.int64)
    for value in values:
        if isinstance(value, (int, np.integer)):
            index = int(value)
        else:
            text = str(value).strip()
            if text in REBEN_CLASS_TO_INDEX:
                index = REBEN_CLASS_TO_INDEX[text]
            elif text.isdigit():
                index = int(text)
            else:
                raise RebenDatasetError(f"Unknown reBEN label {text!r}; expected one of the official 19 class names.")
        if index < 0 or index >= 19:
            raise RebenDatasetError(f"reBEN label index out of range: {index}")
        multi_hot[index] = 1
    return multi_hot


def prepare_configilm_compatible_metadata(
    metadata_parquet: str | Path,
    root_dir: str | Path,
    *,
    patch_id_column: str | None = None,
) -> dict[str, Any]:
    """Generate non-destructive ConfigILM compatibility files from official metadata."""
    metadata_path = Path(metadata_parquet)
    root = Path(root_dir)
    frame = _read_parquet_frame(metadata_path)
    columns = [str(column) for column in frame.columns]
    candidates = _candidate_patch_id_columns(columns)
    if patch_id_column:
        candidates = [patch_id_column] + [column for column in candidates if column != patch_id_column]
    if not candidates:
        raise RebenDatasetError(
            "Cannot generate ConfigILM compatibility files because metadata has no patch id column. "
            f"Observed columns: {columns}"
        )
    label_column = "labels" if "labels" in columns else "label" if "label" in columns else "class_labels" if "class_labels" in columns else ""
    if not label_column:
        raise RebenDatasetError(f"Cannot generate ConfigILM labels parquet; no labels column in metadata columns: {columns}")
    split_column = "split" if "split" in columns else ""
    selected = candidates[0]
    names = frame[selected].astype(str)
    labels_frame = frame[[label_column]].copy()
    labels_frame.insert(0, "name", names)
    labels_path = root / "labels_configilm_compat.parquet"
    labels_frame.to_parquet(labels_path, index=False)
    split_files: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    if split_column:
        split_values = frame[split_column].map(_normalise_split)
        for split in ["train", "val", "test"]:
            values = names[split_values == split]
            path = root / f"{split}.csv"
            values.to_csv(path, index=False, header=False)
            split_files[split] = str(path)
            split_counts[split] = int(len(values))
    else:
        path = root / "all.csv"
        names.to_csv(path, index=False, header=False)
        split_files["all"] = str(path)
        split_counts["all"] = int(len(names))
    return {
        "metadata_parquet": str(metadata_path),
        "metadata_columns": columns,
        "patch_id_column": selected,
        "candidate_patch_id_columns": candidates,
        "label_column": label_column,
        "split_column": split_column,
        "labels_compat_parquet": str(labels_path),
        "split_csv_files": split_files,
        "split_counts": split_counts,
        "csv_format": "single_column_patch_names_headerless",
    }


def prepare_lmdb_safetensors_metadata(
    metadata_parquet: str | Path,
    *,
    split: str,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    frame = _read_parquet_frame(Path(metadata_parquet))
    columns = [str(column) for column in frame.columns]
    if "split" in columns:
        split_values = frame["split"].map(_normalise_split)
        frame = frame[split_values == _normalise_split(split)]
    if max_samples is not None:
        frame = frame.head(int(max_samples))
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        item = row.to_dict()
        sample_id = item.get("patch_id") or item.get("s2v1_name") or item.get("s2_name") or item.get("s1_name")
        if not sample_id:
            continue
        item["sample_id"] = str(sample_id)
        item["patch_id"] = str(item.get("patch_id", sample_id))
        item.update(reben_spatial_lineage(item["patch_id"]))
        item["split"] = _normalise_split(item.get("split", split))
        item["country"] = str(item.get("country", ""))
        rows.append(item)
    return rows


class LmdbSafetensorsRebenDatasetAdapter(DatasetAdapter):
    """reBEN adapter for LMDB values stored as safetensors bytes."""

    valid_sensor_modes = {"S1", "S2", "S1+S2"}

    def __init__(
        self,
        images_lmdb: str | Path,
        metadata_parquet: str | Path,
        metadata_snow_cloud_parquet: str | Path | None = None,
        *,
        split: str,
        sensor_mode: str,
        max_samples: int | None = None,
        channel_profile: str = "croma",
    ) -> None:
        root_dir, lmdb_path, notes = resolve_reben_root_dir(images_lmdb)
        self.root_dir = root_dir
        self.images_lmdb = lmdb_path
        self.metadata_parquet = Path(metadata_parquet)
        self.metadata_snow_cloud_parquet = Path(metadata_snow_cloud_parquet) if metadata_snow_cloud_parquet else None
        self.split = split
        self.sensor_mode = sensor_mode
        self.max_samples = None if max_samples in (None, 0) else int(max_samples)
        self.channel_profile = channel_profile
        self.root_resolution_notes = notes
        self._rows: list[dict[str, Any]] | None = None
        self._env: Any | None = None
        if sensor_mode not in self.valid_sensor_modes:
            raise ValueError(f"sensor_mode must be one of {sorted(self.valid_sensor_modes)}, got {sensor_mode!r}.")

    def _metadata_rows(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = prepare_lmdb_safetensors_metadata(self.metadata_parquet, split=self.split, max_samples=self.max_samples)
        return self._rows

    def _lmdb_env(self) -> Any:
        if self._env is None:
            if not self.images_lmdb.exists():
                raise RebenDatasetError(f"LMDB path does not exist: {self.images_lmdb}")
            self._env = _open_lmdb(self.images_lmdb)
        return self._env

    def load_metadata(self) -> list[dict[str, Any]]:
        return [self._normalize_metadata(index, row) for index, row in enumerate(self._metadata_rows())]

    def _normalize_metadata(self, index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        sample_id = item.get("sample_id") or item.get("patch_id") or item.get("s2v1_name") or item.get("s1_name") or f"reben_{self.split}_{index:08d}"
        item["sample_id"] = str(sample_id)
        item["patch_id"] = str(item.get("patch_id", sample_id))
        item.update(reben_spatial_lineage(item["patch_id"]))
        item["split"] = str(item.get("split", self.split))
        item["country"] = str(item.get("country", ""))
        item["sensor_mode"] = self.sensor_mode
        seasonal_snow = item.get("contains_seasonal_snow", item.get("seasonal_snow", ""))
        cloud_shadow = item.get("contains_cloud_or_shadow", item.get("cloud_or_shadow", ""))
        if str(seasonal_snow).lower() in {"true", "1", "yes"} or str(cloud_shadow).lower() in {"true", "1", "yes"}:
            item["cloud_snow_shadow"] = "cloud_snow_shadow"
        elif seasonal_snow != "" or cloud_shadow != "":
            item["cloud_snow_shadow"] = "clear"
        return item

    def _load_key(self, key: str) -> dict[str, np.ndarray] | None:
        if not key:
            return None
        env = self._lmdb_env()
        with env.begin(write=False) as txn:
            value = txn.get(str(key).encode("utf-8"))
        if value is None:
            return None
        return _load_safetensors_lmdb_value(value)

    def _load_first_available_key(self, keys: Sequence[str], *, expected_bands: Sequence[str]) -> tuple[dict[str, np.ndarray], str]:
        errors: list[str] = []
        for key in keys:
            if not key:
                continue
            payload = self._load_key(str(key))
            if payload is None:
                errors.append(f"{key}: missing")
                continue
            missing = [band for band in expected_bands if band not in payload]
            if missing:
                errors.append(f"{key}: missing_bands={missing}; available={sorted(payload)}")
                continue
            return payload, str(key)
        raise RebenDatasetError(f"No LMDB key contained expected bands {list(expected_bands)}. Attempts: {' | '.join(errors)}")

    @staticmethod
    def _resize_band(array: np.ndarray, target_shape: tuple[int, int] = REBEN_TARGET_SIZE) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)
        if arr.shape == target_shape:
            return arr
        if arr.ndim != 2:
            raise RebenDatasetError(f"Expected 2D band array before resize, got shape {arr.shape}.")
        try:
            import cv2

            return cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        except ImportError:
            try:
                import torch
                import torch.nn.functional as F
            except ImportError as exc:
                raise RebenDatasetError("Resizing multi-resolution reBEN bands requires cv2 or torch.") from exc
            tensor = torch.as_tensor(arr[None, None, :, :], dtype=torch.float32)
            resized = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
            return resized[0, 0].cpu().numpy().astype(np.float32)

    @staticmethod
    def _stack_bands(payload: Mapping[str, np.ndarray], bands: Sequence[str]) -> np.ndarray:
        missing = [band for band in bands if band not in payload]
        if missing:
            raise RebenDatasetError(f"LMDB payload missing bands: {missing}; available={sorted(payload)}")
        return np.stack([LmdbSafetensorsRebenDatasetAdapter._resize_band(np.asarray(payload[band], dtype=np.float32)) for band in bands], axis=0)

    def load_sample(self, index: int) -> Mapping[str, Any]:
        row = self._metadata_rows()[index]
        metadata = self._normalize_metadata(index, row)
        label_array = reben_labels_to_multihot(row.get("labels", []))
        if label_array.shape[-1] != 19:
            raise RebenDatasetError(f"Expected 19-label vector in metadata labels, got shape {label_array.shape}.")
        s1_keys = [str(row.get("s1_name", "")), str(row.get("patch_id", "")), str(row.get("sample_id", ""))]
        s2_keys = [
            str(row.get("s2v1_name", "")),
            str(row.get("s2_name", "")),
            str(row.get("patch_id", "")),
            str(row.get("sample_id", "")),
            str(row.get("s1_name", "")),
        ]
        if self.sensor_mode == "S1":
            s1_payload, s1_key = self._load_first_available_key(s1_keys, expected_bands=S1_BANDS)
            image_payload: Any = self._stack_bands(s1_payload, S1_BANDS)
            metadata["lmdb_s1_key"] = s1_key
        elif self.sensor_mode == "S2":
            bands = S2_10_BANDS if self.channel_profile == "bifold_resnet101" else S2_12_BANDS
            s2_payload, s2_key = self._load_first_available_key(s2_keys, expected_bands=bands)
            image_payload = self._stack_bands(s2_payload, bands)
            metadata["lmdb_s2_key"] = s2_key
        else:
            s2_bands = S2_10_BANDS if self.channel_profile == "bifold_resnet101" else S2_12_BANDS
            s1_payload, s1_key = self._load_first_available_key(s1_keys, expected_bands=S1_BANDS)
            s2_payload, s2_key = self._load_first_available_key(s2_keys, expected_bands=s2_bands)
            metadata["lmdb_s1_key"] = s1_key
            metadata["lmdb_s2_key"] = s2_key
            if self.channel_profile == "bifold_resnet101":
                image_payload = np.concatenate([self._stack_bands(s1_payload, S1_BANDS), self._stack_bands(s2_payload, s2_bands)], axis=0)
            else:
                image_payload = {"S1": self._stack_bands(s1_payload, S1_BANDS), "S2": self._stack_bands(s2_payload, s2_bands)}
        metadata["label_vector"] = label_array.astype(int).tolist()
        return {"image": image_payload, "metadata": metadata}

    def get_labels(self, index: int) -> int:
        raise RebenDatasetError("reBEN is multi-label; use get_label_vector().")

    def get_label_vector(self, index: int) -> np.ndarray:
        return np.asarray(self.load_sample(index)["metadata"]["label_vector"], dtype=np.int64)

    def get_region(self, index: int) -> str:
        return str(self.load_sample(index)["metadata"].get("country", ""))

    def get_sensor(self, index: int) -> str:
        return self.sensor_mode

    def get_group_keys(self, index: int) -> dict[str, str]:
        row = self.load_sample(index)["metadata"]
        return {"country": str(row.get("country", "")), "sensor_mode": self.sensor_mode}

    def loader_info(self) -> dict[str, Any]:
        return {
            "loader": "LMDB+safetensors",
            "payload_format": "safetensors",
            "root_dir": str(self.root_dir),
            "images_lmdb": str(self.images_lmdb),
            "metadata_parquet": str(self.metadata_parquet),
            "split": self.split,
            "sensor_mode": self.sensor_mode,
            "channel_profile": self.channel_profile,
            "root_resolution_notes": self.root_resolution_notes,
        }


def inspect_lmdb_payload(lmdb_path: str | Path, split_csv_path: str | Path | None = None) -> dict[str, Any]:
    """Inspect LMDB structure without scanning or loading the full database."""
    path = Path(lmdb_path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "data_mdb_exists": (path / "data.mdb").exists(),
        "lock_mdb_exists": (path / "lock.mdb").exists(),
        "data_mdb_size_bytes": (path / "data.mdb").stat().st_size if (path / "data.mdb").exists() else 0,
        "split_csv_path": str(split_csv_path or ""),
        "first_split_patch_id": "",
        "lmdb_stat": {},
        "first_key": "",
        "first_value_len": 0,
        "first_value_prefix_hex": "",
        "first_value_prefix_ascii": "",
        "pickle_load_status": "not_attempted",
        "numpy_load_status": "not_attempted",
        "error": "",
    }
    if split_csv_path and Path(split_csv_path).exists():
        try:
            with Path(split_csv_path).open("r", encoding="utf-8") as handle:
                result["first_split_patch_id"] = handle.readline().strip().split(",")[0]
        except Exception as exc:
            result["first_split_patch_id_error"] = f"{type(exc).__name__}: {exc}"
    if not path.exists():
        result["error"] = "LMDB path does not exist."
        return result
    try:
        import lmdb
    except ImportError as exc:
        result["error"] = f"lmdb package unavailable: {exc}"
        return result
    try:
        with _probe_lmdb_environment(path) as env:
            with env.begin(write=False) as txn:
                result["lmdb_stat"] = dict(txn.stat())
                cursor = txn.cursor()
                if cursor.first():
                    key, value = cursor.item()
                    result["first_key"] = key.decode("utf-8", errors="replace")
                    result["first_value_len"] = len(value)
                    prefix = value[:32]
                    result["first_value_prefix_hex"] = prefix.hex()
                    result["first_value_prefix_ascii"] = prefix.decode("utf-8", errors="replace")
                    try:
                        pickle.loads(value)
                        result["pickle_load_status"] = "ok"
                    except Exception as exc:
                        result["pickle_load_status"] = f"failed: {type(exc).__name__}: {exc}"
                    try:
                        np.load(io.BytesIO(value), allow_pickle=False)
                        result["numpy_load_status"] = "ok"
                    except Exception as exc:
                        result["numpy_load_status"] = f"failed: {type(exc).__name__}: {exc}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def instantiate_configilm_ben2_smoke(
    *,
    root_dir: str | Path,
    labels_parquet: str | Path,
    split: str = "train",
    img_size: tuple[int, int, int] = (12, 120, 120),
    max_img_idx: int = 2,
) -> dict[str, Any]:
    dataset_class, class_info = import_configilm_reben_dataset_class()
    kwargs = {
        "root_dir": Path(root_dir),
        "split": split,
        "max_img_idx": max_img_idx,
        "img_size": img_size,
        "return_patchname": True,
        "new_label_file": Path(labels_parquet),
    }
    dataset = dataset_class(**kwargs)
    length = len(dataset)
    if length < 1:
        raise RebenDatasetError(f"BEN2DataSet instantiated but length is {length}; check LMDB keys and split CSV patch ids.")
    item = dataset[0]
    if not isinstance(item, (list, tuple)):
        raise RebenDatasetError(f"BEN2DataSet[0] returned {type(item).__name__}, expected tuple/list.")
    item_length = len(item)
    image = item[0]
    label = item[1] if item_length >= 2 else None
    patch_name = item[2] if item_length >= 3 else ""
    image_array = _to_numpy(image)
    label_array = _to_numpy(label)
    return {
        "status": "ok",
        "dataset_class": class_info,
        "constructor_kwargs": {key: str(value) for key, value in kwargs.items()},
        "dataset_length": int(length),
        "item_tuple_length": int(item_length),
        "image_shape": list(image_array.shape),
        "label_shape": list(label_array.shape),
        "patch_name": str(patch_name),
    }


def run_configilm_reben_preflight(
    *,
    images_lmdb: str | Path,
    metadata_parquet: str | Path,
    metadata_snow_cloud_parquet: str | Path | None = None,
    output_dir: str | Path | None = None,
    split: str = "train",
    img_size: tuple[int, int, int] = (12, 120, 120),
) -> dict[str, Any]:
    root_dir, lmdb_path, root_notes = resolve_reben_root_dir(images_lmdb)
    result: dict[str, Any] = {
        "status": "unknown",
        "root_dir": str(root_dir),
        "lmdb_path": str(lmdb_path),
        "root_resolution_notes": root_notes,
        "lmdb_exists": lmdb_path.exists(),
        "metadata_parquet": str(metadata_parquet),
        "metadata_snow_cloud_parquet": str(metadata_snow_cloud_parquet or ""),
        "metadata_snow_cloud_exists": bool(metadata_snow_cloud_parquet and Path(metadata_snow_cloud_parquet).exists()),
        "ben2_signature": "",
        "compatibility_files": {},
        "lmdb_payload_inspection": {},
        "dataset_smoke": {},
        "error": "",
        "traceback": "",
    }
    try:
        dataset_class, class_info = import_configilm_reben_dataset_class()
        result["dataset_class"] = class_info
        result["ben2_signature"] = str(inspect.signature(dataset_class.__init__))
        if not lmdb_path.exists():
            raise RebenDatasetError(f"Expected ConfigILM LMDB at {lmdb_path}")
        compat = prepare_configilm_compatible_metadata(metadata_parquet, root_dir)
        result["compatibility_files"] = compat
        result["lmdb_payload_inspection"] = inspect_lmdb_payload(lmdb_path, compat.get("split_csv_files", {}).get(split))
        smoke = instantiate_configilm_ben2_smoke(
            root_dir=root_dir,
            labels_parquet=compat["labels_compat_parquet"],
            split=split,
            img_size=img_size,
            max_img_idx=2,
        )
        result["dataset_smoke"] = smoke
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        if result.get("compatibility_files") and not result.get("lmdb_payload_inspection"):
            compat = result["compatibility_files"]
            if isinstance(compat, Mapping):
                result["lmdb_payload_inspection"] = inspect_lmdb_payload(lmdb_path, compat.get("split_csv_files", {}).get(split, None))
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "reben_configilm_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        lines = [
            "# reBEN ConfigILM Preflight",
            "",
            f"Status: {result['status']}",
            f"root_dir: `{result['root_dir']}`",
            f"lmdb_path: `{result['lmdb_path']}`",
            f"BEN2DataSet signature: `{result.get('ben2_signature', '')}`",
            "",
            "## Root Resolution",
            "",
            *[f"- {note}" for note in root_notes],
        ]
        payload_format = ""
        lmdb_info = result.get("lmdb_payload_inspection", {})
        if isinstance(lmdb_info, Mapping):
            if str(lmdb_info.get("pickle_load_status", "")).startswith("ok"):
                payload_format = "pickle_patch_interface"
            elif str(lmdb_info.get("numpy_load_status", "")).startswith("ok"):
                payload_format = "numpy"
            else:
                try:
                    if detect_lmdb_payload_format(lmdb_path) == "safetensors":
                        payload_format = "safetensors"
                except Exception:
                    payload_format = ""
        if result["status"] == "ok":
            smoke = result["dataset_smoke"]
            lines.extend(
                [
                    "",
                    "## Dataset Instantiation Smoke",
                    "",
                    f"- dataset_length: {smoke.get('dataset_length')}",
                    f"- item_tuple_length: {smoke.get('item_tuple_length')}",
                    f"- image_shape: {smoke.get('image_shape')}",
                    f"- label_shape: {smoke.get('label_shape')}",
                    f"- patch_name: {smoke.get('patch_name')}",
                ]
            )
        else:
            lines.extend(["", "## Error", "", str(result.get("error", ""))])
            lmdb_info = result.get("lmdb_payload_inspection", {})
            if isinstance(lmdb_info, Mapping) and lmdb_info:
                lines.extend(
                    [
                        "",
                        "## LMDB Payload Inspection",
                        "",
                        f"- data.mdb size bytes: {lmdb_info.get('data_mdb_size_bytes', '')}",
                        f"- first split patch id: `{lmdb_info.get('first_split_patch_id', '')}`",
                        f"- first LMDB key: `{lmdb_info.get('first_key', '')}`",
                        f"- first value prefix ascii: `{lmdb_info.get('first_value_prefix_ascii', '')}`",
                        f"- pickle load status: {lmdb_info.get('pickle_load_status', '')}",
                        f"- numpy load status: {lmdb_info.get('numpy_load_status', '')}",
                    ]
                )
            if result.get("traceback"):
                lines.extend(["", "## Traceback", "", "```text", str(result.get("traceback", "")), "```"])
        if payload_format == "safetensors":
            lines.extend(
                [
                    "",
                    "## Payload Format Note",
                    "",
                    "This LMDB stores safetensors payloads rather than ConfigILM/BigEarthNetEncoder pickle patch-interface objects.",
                    "ConfigILM BEN2DataSet cannot read this payload with `pickle.loads`; use the repo's LMDB+safetensors reBEN adapter path.",
                ]
            )
        (out / "reben_configilm_preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def import_configilm_reben_dataset_class() -> tuple[type[Any], dict[str, str]]:
    """Import the official ConfigILM reBEN Dataset class with version-compatible aliases."""
    errors: list[str] = []
    candidates = [
        ("configilm.extra.DataSets.BEN2_DataSet", "BEN2DataSet"),
        ("configilm.extra.DataSets.BEN2_DataSet", "BEN2_DataSet"),
        ("configilm.extra.DataSets.BEN2_DataSet", "BENv2DataSet"),
        ("configilm.extra.DataSets.BENv2_DataSet", "BENv2DataSet"),
        ("configilm.extra.DataSets.BENv2_DataSet", "BENv2_DataSet"),
    ]
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: import failed: {type(exc).__name__}: {exc}")
            continue
        dataset_class = getattr(module, class_name, None)
        if dataset_class is None:
            errors.append(f"{module_name}: missing class {class_name}")
            continue
        return dataset_class, {"module": module_name, "class": class_name, "qualified_name": f"{module_name}.{class_name}"}
    raise RebenDatasetError(
        "Could not import the ConfigILM BigEarthNet v2/reBEN dataset class. "
        "Expected BEN2_DataSet/BEN2DataSet in ConfigILM 0.4.x. Import attempts: " + " | ".join(errors)
    )


def check_reben_configilm_dependency_chain() -> dict[str, Any]:
    """Check imports that commonly fail before the ConfigILM reBEN loader is usable."""
    modules = [
        "appdirs",
        "fastcore",
        "fastcore.dispatch",
        "bigearthnet_common",
        "bigearthnet_patch_interface",
        "configilm",
    ]
    rows: list[dict[str, Any]] = []
    ok = True
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = str(getattr(module, "__version__", "unknown"))
            rows.append({"module": module_name, "status": "ok", "version": version, "message": ""})
        except Exception as exc:
            ok = False
            rows.append({"module": module_name, "status": "failed", "version": "", "message": f"{type(exc).__name__}: {exc}"})
    dataset_info: dict[str, str] | None = None
    if ok:
        try:
            _, dataset_info = import_configilm_reben_dataset_class()
            rows.append({"module": dataset_info["module"], "status": "ok", "version": "", "message": f"class={dataset_info['class']}"})
        except Exception as exc:
            ok = False
            rows.append({"module": "configilm.extra.DataSets.BEN2_DataSet", "status": "failed", "version": "", "message": f"{type(exc).__name__}: {exc}"})
    install_command = (
        "pip install -U --no-deps appdirs configilm bigearthnet_patch_interface bigearthnet_common "
        "&& pip install --force-reinstall 'fastcore==1.5.29'"
    )
    return {
        "status": "ok" if ok else "failed",
        "checks": rows,
        "dataset_class": dataset_info or {},
        "install_command": install_command,
        "notes": [
            "Do not reinstall torch/CUDA for this compatibility fix.",
            "fastcore==1.5.29 preserves fastcore.dispatch used by bigearthnet_common 2.8.x.",
            "fasttransform is not required by this project unless pulled in by the user's environment.",
        ],
    }


class ConfigILMRebenDatasetAdapter(DatasetAdapter):
    """Official ConfigILM-backed BigEarthNet v2.0 / reBEN adapter.

    This adapter is intentionally narrow: it delegates official LMDB/parquet
    indexing to ConfigILM's `BEN2_DataSet`, then exposes exactly the S1/S2
    arrays needed by the CROMA sensor-mode audit.
    """

    valid_sensor_modes = {"S1", "S2", "S1+S2"}

    def __init__(
        self,
        images_lmdb: str | Path,
        metadata_parquet: str | Path,
        metadata_snow_cloud_parquet: str | Path,
        *,
        split: str,
        sensor_mode: str,
        max_samples: int | None = None,
        channel_profile: str = "croma",
        dataset: Any | None = None,
    ) -> None:
        self.images_lmdb = Path(images_lmdb)
        self.metadata_parquet = Path(metadata_parquet)
        self.metadata_snow_cloud_parquet = Path(metadata_snow_cloud_parquet)
        self.split = split
        self.sensor_mode = sensor_mode
        self.max_samples = None if max_samples in (None, 0) else int(max_samples)
        self.channel_profile = channel_profile
        self._dataset = dataset
        self._s1_dataset: Any | None = None
        self._dataset_class_info: dict[str, str] = {}
        if sensor_mode not in self.valid_sensor_modes:
            raise ValueError(f"sensor_mode must be one of {sorted(self.valid_sensor_modes)}, got {sensor_mode!r}.")
        if channel_profile not in {"croma", "bifold_resnet101"}:
            raise ValueError("channel_profile must be 'croma' or 'bifold_resnet101'.")

    @property
    def data_dirs(self) -> dict[str, str]:
        root_dir, lmdb_path, _ = resolve_reben_root_dir(self.images_lmdb)
        return {
            "root_dir": str(root_dir),
            "images_lmdb": str(lmdb_path),
            "labels_parquet": str(self.metadata_parquet),
            "metadata_snow_cloud_parquet": str(self.metadata_snow_cloud_parquet),
        }

    @property
    def bands(self) -> int:
        if self.channel_profile == "bifold_resnet101":
            # BIFOLD v0.2.0 cards use S1=2, S2=10, all=12.
            return {"S1": 2, "S2": 10, "S1+S2": 12}[self.sensor_mode]
        # CROMA uses S1=2 and S2=12. For fusion, ConfigILM is instantiated
        # separately for the 2-channel SAR and 12-channel optical configs.
        return 2 if self.sensor_mode == "S1" else 12

    @property
    def img_size(self) -> tuple[int, int, int]:
        return REBEN_CONFIGILM_IMAGE_SIZES[(self.channel_profile, self.sensor_mode)]

    def _load_official_dataset(self) -> Any:
        if self._dataset is not None:
            return self._dataset
        for path in [self.images_lmdb, self.metadata_parquet, self.metadata_snow_cloud_parquet]:
            if not path.exists():
                raise RebenDatasetError(f"Required reBEN path does not exist: {path}")
        dataset_class, info = import_configilm_reben_dataset_class()
        self._dataset_class_info = dict(info)
        root_dir, _, _ = resolve_reben_root_dir(self.images_lmdb)
        compat_labels = root_dir / "labels_configilm_compat.parquet"
        label_file = compat_labels if compat_labels.exists() else self.metadata_parquet
        signature = inspect.signature(dataset_class.__init__)
        if "data_dirs" in signature.parameters:
            kwargs = {
                "data_dirs": self.data_dirs,
                "split": self.split,
                "bands": self.bands,
            }
            if self.max_samples is not None:
                kwargs["max_len"] = self.max_samples
            try:
                self._dataset = dataset_class(**kwargs)
            except TypeError:
                kwargs.pop("bands", None)
                self._dataset = dataset_class(**kwargs)
            return self._dataset
        kwargs = {
            "root_dir": root_dir,
            "split": self.split,
            "img_size": self.img_size,
            "return_patchname": True,
            "new_label_file": label_file,
        }
        if self.max_samples is not None:
            kwargs["max_img_idx"] = self.max_samples
        try:
            self._dataset = dataset_class(**kwargs)
        except TypeError:
            kwargs.pop("new_label_file", None)
            self._dataset = dataset_class(**kwargs)
        return self._dataset

    def _load_s1_dataset_for_fusion(self) -> Any:
        if self._s1_dataset is not None:
            return self._s1_dataset
        if self.sensor_mode != "S1+S2" or self.channel_profile != "croma":
            return self._load_official_dataset()
        if self._dataset is not None:
            return self._dataset
        for path in [self.images_lmdb, self.metadata_parquet, self.metadata_snow_cloud_parquet]:
            if not path.exists():
                raise RebenDatasetError(f"Required reBEN path does not exist: {path}")
        dataset_class, info = import_configilm_reben_dataset_class()
        self._dataset_class_info = dict(info)
        root_dir, _, _ = resolve_reben_root_dir(self.images_lmdb)
        compat_labels = root_dir / "labels_configilm_compat.parquet"
        label_file = compat_labels if compat_labels.exists() else self.metadata_parquet
        self._s1_dataset = dataset_class(
            root_dir=root_dir,
            split=self.split,
            img_size=(2, 120, 120),
            return_patchname=True,
            new_label_file=label_file,
            max_img_idx=self.max_samples,
        )
        return self._s1_dataset

    def loader_info(self) -> dict[str, Any]:
        return {
            "loader": "ConfigILM",
            "dataset_class": dict(self._dataset_class_info),
            "data_dirs": self.data_dirs,
            "split": self.split,
            "sensor_mode": self.sensor_mode,
            "channel_profile": self.channel_profile,
            "bands": self.bands,
            "img_size": self.img_size,
        }

    def _metadata_for_index(self, index: int) -> dict[str, Any]:
        dataset = self._load_official_dataset()
        for attr in ("metadata", "meta", "df", "data_df", "filtered_patches"):
            value = getattr(dataset, attr, None)
            if value is None:
                continue
            try:
                row = value.iloc[index].to_dict()
                if isinstance(row, dict):
                    return dict(row)
            except Exception:
                pass
            try:
                row = value[index]
                if isinstance(row, Mapping):
                    return dict(row)
            except Exception:
                pass
        return {"sample_id": f"reben_{self.split}_{index:08d}", "split": self.split}

    def load_metadata(self) -> list[dict[str, Any]]:
        dataset = self._load_official_dataset()
        count = len(dataset)
        if self.max_samples is not None:
            count = min(count, self.max_samples)
        return [self._normalize_metadata(index, self._metadata_for_index(index)) for index in range(count)]

    def _normalize_metadata(self, index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        sample_id = item.get("sample_id") or item.get("patch_id") or item.get("s2_name") or item.get("name") or f"reben_{self.split}_{index:08d}"
        item["sample_id"] = str(sample_id)
        item["patch_id"] = str(item.get("patch_id", sample_id))
        item["split"] = str(item.get("split", self.split))
        item["country"] = str(item.get("country", ""))
        item["sensor_mode"] = self.sensor_mode
        seasonal_snow = item.get("contains_seasonal_snow", item.get("seasonal_snow", ""))
        cloud_shadow = item.get("contains_cloud_or_shadow", item.get("cloud_or_shadow", ""))
        if str(seasonal_snow).lower() in {"true", "1", "yes"} or str(cloud_shadow).lower() in {"true", "1", "yes"}:
            item["cloud_snow_shadow"] = "cloud_snow_shadow"
        elif seasonal_snow != "" or cloud_shadow != "":
            item["cloud_snow_shadow"] = "clear"
        return item

    def load_sample(self, index: int) -> Mapping[str, Any]:
        dataset = self._load_official_dataset()
        item = dataset[index]
        if not isinstance(item, (list, tuple)):
            raise RebenDatasetError(f"Expected BEN2DataSet item tuple, got {type(item).__name__}.")
        if len(item) == 3:
            image, label, patch_name = item
        elif len(item) == 2:
            image, label = item
            patch_name = ""
        else:
            raise RebenDatasetError(f"Expected BEN2DataSet item of length 2 or 3, got {len(item)}.")
        image_array = _to_numpy(image).astype(np.float32)
        label_array = _to_numpy(label).astype(np.float32)
        if image_array.ndim != 3:
            raise RebenDatasetError(f"Expected ConfigILM image shape [C,H,W], got {image_array.shape}.")
        if label_array.shape[-1] != 19:
            raise RebenDatasetError(f"Expected reBEN 19-label vector, got shape {label_array.shape}.")
        if self.sensor_mode == "S1":
            if image_array.shape[0] >= 14:
                image_array = image_array[:2]
            if image_array.shape[0] != 2:
                raise RebenDatasetError(f"Expected S1 2-channel image, got {image_array.shape}.")
            image_payload: Any = image_array
        elif self.channel_profile == "bifold_resnet101":
            expected = 10 if self.sensor_mode == "S2" else 12
            if image_array.shape[0] != expected:
                raise RebenDatasetError(f"Expected BIFOLD {self.sensor_mode} {expected}-channel image, got {image_array.shape}.")
            image_payload = image_array
        elif self.sensor_mode == "S2":
            if image_array.shape[0] == 14:
                image_array = image_array[2:14]
            if image_array.shape[0] != 12:
                raise RebenDatasetError(f"CROMA S2 path requires ConfigILM 12-channel S2 data, got {image_array.shape}.")
            image_payload = image_array
        else:
            if image_array.shape[0] == 14:
                image_array = image_array[2:14]
            if image_array.shape[0] != 12:
                raise RebenDatasetError(f"CROMA fusion optical branch requires 12-channel S2 data, got {image_array.shape}.")
            s1_item = self._load_s1_dataset_for_fusion()[index]
            s1_image = s1_item[0] if isinstance(s1_item, (list, tuple)) else s1_item
            s1_array = _to_numpy(s1_image).astype(np.float32)
            if s1_array.shape[0] >= 14:
                s1_array = s1_array[:2]
            if s1_array.shape[0] != 2:
                raise RebenDatasetError(f"CROMA fusion SAR branch requires 2-channel S1 data, got {s1_array.shape}.")
            image_payload = {"S1": s1_array, "S2": image_array}
        metadata = self._normalize_metadata(index, self._metadata_for_index(index))
        if patch_name:
            metadata["sample_id"] = str(patch_name)
            metadata["patch_id"] = str(patch_name)
        metadata["label_vector"] = label_array.astype(int).tolist()
        return {"image": image_payload, "metadata": metadata}

    def get_labels(self, index: int) -> int:
        raise RebenDatasetError("reBEN is multi-label; use get_label_vector().")

    def get_label_vector(self, index: int) -> np.ndarray:
        return np.asarray(self.load_sample(index)["metadata"]["label_vector"], dtype=np.int64)

    def get_region(self, index: int) -> str:
        return str(self.load_sample(index)["metadata"].get("country", ""))

    def get_sensor(self, index: int) -> str:
        return self.sensor_mode

    def get_group_keys(self, index: int) -> dict[str, str]:
        row = self.load_sample(index)["metadata"]
        return {
            "country": str(row.get("country", "")),
            "sensor_mode": self.sensor_mode,
        }
