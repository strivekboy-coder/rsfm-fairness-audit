from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter


class RebenDatasetError(RuntimeError):
    """Raised when official BigEarthNet v2.0 / reBEN loading is unavailable or inconsistent."""


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ConfigILMRebenDatasetAdapter(DatasetAdapter):
    """Official ConfigILM-backed BigEarthNet v2.0 / reBEN adapter.

    This adapter is intentionally narrow: it delegates official LMDB/parquet
    indexing to ConfigILM's `BENv2DataSet`, then exposes exactly the S1/S2
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
        if sensor_mode not in self.valid_sensor_modes:
            raise ValueError(f"sensor_mode must be one of {sorted(self.valid_sensor_modes)}, got {sensor_mode!r}.")
        if channel_profile not in {"croma", "bifold_resnet101"}:
            raise ValueError("channel_profile must be 'croma' or 'bifold_resnet101'.")

    @property
    def data_dirs(self) -> dict[str, str]:
        return {
            "images_lmdb": str(self.images_lmdb),
            "metadata_parquet": str(self.metadata_parquet),
            "metadata_snow_cloud_parquet": str(self.metadata_snow_cloud_parquet),
        }

    @property
    def bands(self) -> int:
        if self.channel_profile == "bifold_resnet101":
            # BIFOLD v0.2.0 cards use S1=2, S2=10, all=12.
            return {"S1": 2, "S2": 10, "S1+S2": 12}[self.sensor_mode]
        # CROMA uses S1=2, S2=12, both=14. ConfigILM exposes S2=12 only as
        # part of the 14-channel S1+S2+60m-original profile, so S2-only loads
        # 14 and drops the first two SAR channels below.
        return 2 if self.sensor_mode == "S1" else 14

    def _load_official_dataset(self) -> Any:
        if self._dataset is not None:
            return self._dataset
        for path in [self.images_lmdb, self.metadata_parquet, self.metadata_snow_cloud_parquet]:
            if not path.exists():
                raise RebenDatasetError(f"Required reBEN path does not exist: {path}")
        try:
            from configilm.extra.DataSets import BENv2_DataSet
        except ImportError as exc:
            raise RebenDatasetError(
                "Official reBEN loading requires ConfigILM. Install/use ConfigILM in Colab; "
                "this project will not silently substitute the old BigEarthNet smoke adapter."
            ) from exc
        kwargs = {
            "data_dirs": self.data_dirs,
            "split": self.split,
            "bands": self.bands,
        }
        if self.max_samples is not None:
            kwargs["max_len"] = self.max_samples
        try:
            self._dataset = BENv2_DataSet.BENv2DataSet(**kwargs)
        except TypeError:
            kwargs.pop("bands", None)
            self._dataset = BENv2_DataSet.BENv2DataSet(**kwargs)
        return self._dataset

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
        image, label = dataset[index]
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
            if image_array.shape[0] < 14:
                raise RebenDatasetError("CROMA S2 path requires ConfigILM 14-channel S1+S2 data so it can drop the first two S1 channels.")
            image_payload = image_array[2:14]
        else:
            if image_array.shape[0] < 14:
                raise RebenDatasetError("CROMA both mode requires 14-channel ConfigILM S1+S2 data.")
            image_payload = {"S1": image_array[:2], "S2": image_array[2:14]}
        metadata = self._normalize_metadata(index, self._metadata_for_index(index))
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
