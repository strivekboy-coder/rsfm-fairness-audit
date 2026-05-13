from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter


class Sen1Floods11DatasetError(RuntimeError):
    """Raised when a prepared Sen1Floods11 subset cannot be read safely."""


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
        raise Sen1Floods11DatasetError(f"Sen1Floods11 metadata path does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: _parse_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


class Sen1Floods11DatasetAdapter(DatasetAdapter):
    """Adapter for prepared Prithvi-compatible Sen1Floods11 S2/QC subsets."""

    def __init__(
        self,
        data_root: str | Path,
        metadata_path: str | Path | None = None,
        subset_size: int | None = None,
        split: str = "all",
        cache_metadata: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.subset_size = subset_size
        self.split = split
        self.cache_metadata = cache_metadata
        self._metadata: list[dict[str, Any]] | None = None
        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError("subset_size must be positive when provided.")

    def load_metadata(self) -> list[dict[str, Any]]:
        if self.cache_metadata and self._metadata is not None:
            return list(self._metadata)
        if not self.data_root.exists():
            raise Sen1Floods11DatasetError(
                f"Sen1Floods11 data_root does not exist: {self.data_root}. "
                "Run scripts/prepare_sen1floods11_subset.py first."
            )
        rows = _read_csv(self.metadata_path or self.data_root / "metadata.csv")
        if self.split != "all":
            rows = [row for row in rows if str(row.get("split", "all")).lower() == self.split.lower()]
        rows = [self._normalize_row(row, index) for index, row in enumerate(rows)]
        rows = rows[: self.subset_size] if self.subset_size is not None else rows
        if not rows:
            raise Sen1Floods11DatasetError("No Sen1Floods11 samples are available after filtering.")
        if self.cache_metadata:
            self._metadata = list(rows)
        return list(rows)

    def _normalize_row(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        item = dict(row)
        item["sample_id"] = str(item.get("sample_id") or item.get("chip_id") or f"sen1floods11-{index:06d}")
        item["label"] = int(item.get("label") or 0)
        item["region"] = str(item.get("region") or item.get("location") or item.get("ISO_CC") or item.get("event_id") or "to_verify")
        item["fallback_group"] = str(item.get("fallback_group") or item["region"])
        item["event"] = str(item.get("event") or item.get("event_id") or item["region"])
        item["sensor"] = "S2"
        item["task"] = "sen1floods11_flood_segmentation"
        item["latitude"] = item.get("latitude") or item.get("lat")
        item["longitude"] = item.get("longitude") or item.get("lon")
        return item

    def load_sample(self, index: int) -> Mapping[str, Any]:
        row = self.load_metadata()[index]
        image = self._load_array(row, "chip_path")
        mask = self._load_array(row, "mask_path")
        return {"image": image, "mask": mask, "metadata": row}

    def _load_array(self, row: Mapping[str, Any], key: str) -> np.ndarray:
        value = row.get(key)
        if not value:
            raise Sen1Floods11DatasetError(f"Sample {row.get('sample_id')} is missing {key}.")
        path = Path(str(value))
        if not path.is_absolute():
            path = self.data_root / path
        if not path.exists():
            raise Sen1Floods11DatasetError(f"Referenced file does not exist for {row.get('sample_id')}: {path}")
        if path.suffix.lower() == ".npy":
            return np.load(path).astype(np.float32)
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            key_name = "image" if "image" in data else "mask" if "mask" in data else data.files[0]
            return data[key_name].astype(np.float32)
        raise Sen1Floods11DatasetError(f"Unsupported prepared file format: {path}")

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
            "event": str(row["event"]),
            "region_class": f"{row['region']}::class_{row['label']}",
            "sensor_class": f"{row['sensor']}::class_{row['label']}",
        }
