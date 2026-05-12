from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter


class BenGEDatasetError(RuntimeError):
    """Raised when a prepared BEN-GE subset cannot be read safely."""


def _parse_value(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return ast.literal_eval(text)
    return text


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise BenGEDatasetError(f"BEN-GE metadata path does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: _parse_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _primary_label(row: Mapping[str, Any]) -> int:
    if row.get("label") not in (None, ""):
        return int(row["label"])
    vector = row.get("label_vector")
    if isinstance(vector, str):
        vector = _parse_value(vector)
    if isinstance(vector, list):
        for index, value in enumerate(vector):
            if int(value) == 1:
                return index
        return 0
    raise BenGEDatasetError("BEN-GE row has no scalar label or usable label_vector.")


class BenGEDatasetAdapter(DatasetAdapter):
    """Adapter for prepared BEN-GE-800 paired Sentinel-1/Sentinel-2 subsets."""

    valid_sensor_modes = {"S1", "S2", "S1+S2"}

    def __init__(
        self,
        data_root: str | Path,
        metadata_path: str | Path | None = None,
        subset_size: int | None = None,
        split: str = "all",
        sensor_mode: str = "S1+S2",
        cache_metadata: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.subset_size = subset_size
        self.split = split
        self.sensor_mode = sensor_mode.upper()
        self.cache_metadata = cache_metadata
        self._metadata: list[dict[str, Any]] | None = None

        if self.sensor_mode not in self.valid_sensor_modes:
            raise ValueError(f"sensor_mode must be one of {sorted(self.valid_sensor_modes)}, got {sensor_mode!r}.")
        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError("subset_size must be positive when provided.")

    def load_metadata(self) -> list[dict[str, Any]]:
        if self.cache_metadata and self._metadata is not None:
            return list(self._metadata)
        if not self.data_root.exists():
            raise BenGEDatasetError(
                f"BEN-GE data_root does not exist: {self.data_root}. "
                "Run scripts/prepare_ben_ge_800_subset.py first."
            )
        metadata_path = self.metadata_path or self.data_root / "metadata.csv"
        rows = _read_csv(metadata_path)
        if self.split != "all":
            rows = [row for row in rows if str(row.get("split", "all")).lower() == self.split.lower()]
        rows = [self._normalize_row(row, index) for index, row in enumerate(rows)]
        rows = rows[: self.subset_size] if self.subset_size is not None else rows
        if not rows:
            raise BenGEDatasetError("No BEN-GE samples are available after filtering.")
        if self.cache_metadata:
            self._metadata = list(rows)
        return list(rows)

    def _normalize_row(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        item = dict(row)
        item["sample_id"] = str(item.get("sample_id") or item.get("patch_id") or f"ben-ge-{index:06d}")
        item["label"] = _primary_label(item)
        item["label_vector"] = _parse_value(item.get("label_vector"))
        item["label_names"] = _parse_value(item.get("label_names") or item.get("labels")) or []
        item["region"] = str(
            item.get("region")
            or item.get("country")
            or item.get("climatezone")
            or item.get("fallback_group")
            or "to_verify"
        )
        item["fallback_group"] = str(item.get("fallback_group") or item["region"])
        item["sensor"] = self.sensor_mode
        item["task"] = "ben_ge_800_land_cover_classification"
        item["latitude"] = item.get("latitude") or item.get("lat")
        item["longitude"] = item.get("longitude") or item.get("lon")
        return item

    def load_sample(self, index: int) -> Mapping[str, Any]:
        row = self.load_metadata()[index]
        if self.sensor_mode == "S1":
            image: Any = self._load_array(row, ["s1_path"])
        elif self.sensor_mode == "S2":
            image = self._load_array(row, ["s2_path"])
        else:
            image = {
                "S1": self._load_array(row, ["s1_path"]),
                "S2": self._load_array(row, ["s2_path"]),
            }
        return {"image": image, "metadata": row}

    def _load_array(self, row: Mapping[str, Any], keys: list[str]) -> np.ndarray:
        path_value = next((row.get(key) for key in keys if row.get(key)), None)
        if path_value is None:
            raise BenGEDatasetError(
                f"Sample {row.get('sample_id')} is missing a path for sensor_mode={self.sensor_mode}. "
                f"Expected one of: {', '.join(keys)}."
            )
        path = Path(str(path_value))
        if not path.is_absolute():
            path = self.data_root / path
        if not path.exists():
            raise BenGEDatasetError(f"Referenced BEN-GE chip does not exist for {row.get('sample_id')}: {path}")
        if path.suffix.lower() == ".npy":
            return np.load(path).astype(np.float32)
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            key = "image" if "image" in data else data.files[0]
            return data[key].astype(np.float32)
        raise BenGEDatasetError(f"Unsupported BEN-GE chip format: {path}. Expected .npy or .npz.")

    def get_labels(self, index: int) -> int:
        return int(self.load_metadata()[index]["label"])

    def get_region(self, index: int) -> str:
        return str(self.load_metadata()[index]["region"])

    def get_sensor(self, index: int) -> str:
        return str(self.load_metadata()[index]["sensor"])

    def get_group_keys(self, index: int) -> dict[str, str]:
        row = self.load_metadata()[index]
        return {
            "region": str(row["region"]),
            "sensor": str(row["sensor"]),
            "task": str(row["task"]),
            "region_class": f"{row['region']}::class_{row['label']}",
            "sensor_class": f"{row['sensor']}::class_{row['label']}",
        }
