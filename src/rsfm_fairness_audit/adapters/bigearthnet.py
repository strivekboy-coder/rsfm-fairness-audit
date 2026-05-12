from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter


class BigEarthNetDatasetError(RuntimeError):
    """Raised when a BigEarthNet subset cannot be read safely."""


def _parse_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return ast.literal_eval(text)
    return text


def _read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if not path.exists():
        raise BigEarthNetDatasetError(f"Metadata path does not exist: {path}")
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [{key: _parse_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        if not isinstance(data, list):
            raise BigEarthNetDatasetError(f"Expected a list of sample records in JSON metadata: {path}")
        return [dict(row) for row in data]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if suffix == ".parquet":
        raise BigEarthNetDatasetError(
            "BigEarthNet v2 official metadata is often distributed as metadata.parquet, "
            "but parquet loading is not enabled in this dependency-light smoke adapter. "
            "Provide a CSV/JSON/JSONL subset manifest with verified file paths, labels, and region fields."
        )
    raise BigEarthNetDatasetError(f"Unsupported metadata format '{suffix}'. Use CSV, JSON, or JSONL for smoke runs.")


def _primary_label(row: Mapping[str, Any]) -> int:
    if "label" in row and row["label"] is not None:
        return int(row["label"])
    vector = row.get("label_vector")
    if isinstance(vector, str):
        vector = _parse_value(vector)
    if isinstance(vector, list):
        for index, value in enumerate(vector):
            if int(value) == 1:
                return index
        return 0
    labels = row.get("labels")
    if isinstance(labels, str):
        labels = _parse_value(labels)
    if isinstance(labels, list) and labels:
        first = labels[0]
        if isinstance(first, int):
            return first
    raise BigEarthNetDatasetError(
        "BigEarthNet metadata row has no scalar 'label' or usable 'label_vector'. "
        "For the smoke probe, provide a deterministic primary label column."
    )


class BigEarthNetDatasetAdapter(DatasetAdapter):
    """Subset-first BigEarthNet v2 adapter for local smoke experiments.

    This adapter intentionally avoids guessing the full BigEarthNet directory
    layout. For Milestone 3, provide verified metadata as CSV/JSON/JSONL with
    relative or absolute paths to preconverted `.npy`/`.npz` chips.
    """

    valid_splits = {"train", "val", "test", "all"}
    valid_sensor_modes = {"S1", "S2", "S1+S2"}

    def __init__(
        self,
        data_root: str | Path,
        metadata_path: str | Path | None = None,
        subset_size: int | None = None,
        subset_manifest_path: str | Path | None = None,
        split: str = "all",
        sensor_mode: str = "S2",
        cache_metadata: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.subset_size = subset_size
        self.subset_manifest_path = Path(subset_manifest_path) if subset_manifest_path else None
        self.split = split
        self.sensor_mode = sensor_mode.upper()
        self.cache_metadata = cache_metadata
        self._metadata: list[dict[str, Any]] | None = None

        if self.split not in self.valid_splits:
            raise ValueError(f"split must be one of {sorted(self.valid_splits)}, got {split!r}.")
        if self.sensor_mode not in self.valid_sensor_modes:
            raise ValueError(f"sensor_mode must be one of {sorted(self.valid_sensor_modes)}, got {sensor_mode!r}.")
        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError("subset_size must be positive when provided.")

    def load_metadata(self) -> list[dict[str, Any]]:
        if self.cache_metadata and self._metadata is not None:
            return list(self._metadata)

        rows = self._load_source_rows()
        rows = self._filter_split(rows)
        rows = self._apply_subset_manifest(rows)
        rows = self._normalize_rows(rows)
        rows = rows[: self.subset_size] if self.subset_size is not None else rows

        if not rows:
            raise BigEarthNetDatasetError(
                "No BigEarthNet samples are available after split/subset filtering. "
                "Check split, subset_size, and subset_manifest_path."
            )

        if self.cache_metadata:
            self._metadata = list(rows)
        return list(rows)

    def _load_source_rows(self) -> list[dict[str, Any]]:
        if not self.data_root.exists():
            raise BigEarthNetDatasetError(
                f"BigEarthNet data_root does not exist: {self.data_root}. "
                "Expected a local subset root containing files referenced by a CSV/JSON/JSONL metadata manifest. "
                "Do not point this smoke adapter at a remote URL or an unverified full dataset layout."
            )
        if self.metadata_path is not None:
            return _read_table(self.metadata_path)
        if self.subset_manifest_path is not None:
            return _read_table(self.subset_manifest_path)

        candidates = [
            self.data_root / "metadata.csv",
            self.data_root / "metadata.json",
            self.data_root / "metadata.jsonl",
            self.data_root / "metadata.parquet",
        ]
        for candidate in candidates:
            if candidate.exists():
                return _read_table(candidate)
        raise BigEarthNetDatasetError(
            "No BigEarthNet metadata found. Expected one of metadata.csv, metadata.json, "
            "metadata.jsonl, or metadata.parquet under data_root, or pass --metadata-path / "
            "--subset-manifest-path. For a smoke subset, create a manifest with sample_id, label "
            "or label_vector, region/country if verified, sensor, and s1_path/s2_path columns."
        )

    def _filter_split(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.split == "all":
            return rows
        return [row for row in rows if str(row.get("split", "all")).lower() == self.split]

    def _apply_subset_manifest(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.subset_manifest_path is None or self.metadata_path is None:
            return rows
        manifest_rows = _read_table(self.subset_manifest_path)
        manifest_ids = {str(row.get("sample_id", row.get("id", ""))) for row in manifest_rows}
        manifest_ids.discard("")
        if not manifest_ids:
            return manifest_rows
        return [row for row in rows if str(row.get("sample_id", row.get("id", ""))) in manifest_ids]

    def _normalize_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for index, row in enumerate(rows):
            item = dict(row)
            item["sample_id"] = str(item.get("sample_id") or item.get("id") or f"bigearthnet-{index:06d}")
            item["label"] = _primary_label(item)
            item["label_vector"] = _parse_value(item.get("label_vector"))
            item["label_names"] = _parse_value(item.get("label_names") or item.get("labels")) or []
            item["region"] = str(
                item.get("region")
                or item.get("fallback_group")
                or item.get("country")
                or item.get("tile")
                or item.get("split")
                or "to_verify"
            )
            item["country"] = item.get("country") or ("to_verify" if item["region"] == "to_verify" else item["region"])
            item["sensor"] = self.sensor_mode
            item["task"] = "bigearthnet_land_cover_classification"
            item["latitude"] = item.get("latitude") or item.get("lat")
            item["longitude"] = item.get("longitude") or item.get("lon")
            normalized.append(item)
        return normalized

    def load_sample(self, index: int) -> Mapping[str, Any]:
        row = self.load_metadata()[index]
        if self.sensor_mode == "S2":
            image = self._load_array(row, ["s2_path", "chip_path", "s2_file", "file_path", "path"])
        elif self.sensor_mode == "S1":
            image = self._load_array(row, ["s1_path", "s1_file", "file_path", "path"])
        else:
            image = {
                "S1": self._load_array(row, ["s1_path", "s1_file"]),
                "S2": self._load_array(row, ["s2_path", "s2_file"]),
            }
        return {"image": image, "metadata": row}

    def _load_array(self, row: Mapping[str, Any], keys: list[str]) -> np.ndarray:
        path_value = next((row.get(key) for key in keys if row.get(key)), None)
        if path_value is None:
            raise BigEarthNetDatasetError(
                f"Sample {row.get('sample_id')} is missing a path for sensor_mode={self.sensor_mode}. "
                f"Expected one of: {', '.join(keys)}."
            )
        path = Path(str(path_value))
        if not path.is_absolute():
            path = self.data_root / path
        if not path.exists():
            raise BigEarthNetDatasetError(f"Referenced sample file does not exist for {row.get('sample_id')}: {path}")
        if path.suffix.lower() == ".npy":
            return np.load(path).astype(np.float32)
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            key = "image" if "image" in data else data.files[0]
            return data[key].astype(np.float32)
        raise BigEarthNetDatasetError(
            f"Unsupported sample file format '{path.suffix}' for {path}. "
            "Milestone 3 smoke runs support preconverted .npy/.npz chips. "
            "GeoTIFF loading must be added only after verifying the official BigEarthNet v2 layout and dependencies."
        )

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
