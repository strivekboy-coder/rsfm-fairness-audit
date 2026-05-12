from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter, ModelAdapter


@dataclass(frozen=True)
class DummyDatasetConfig:
    num_samples: int = 240
    bands: int = 8
    height: int = 16
    width: int = 16
    num_classes: int = 4
    seed: int = 7


class DummyEODataset(DatasetAdapter):
    """Synthetic EO dataset with intentionally severe intersectional imbalance."""

    regions = ("europe", "africa", "south_america", "southeast_asia")
    sensors = ("sentinel-2", "sentinel-1", "landsat")

    def __init__(self, config: DummyDatasetConfig | None = None) -> None:
        self.config = config or DummyDatasetConfig()
        self._metadata = self._build_metadata()

    def _build_metadata(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.config.seed)
        region_probs = np.array([0.58, 0.22, 0.14, 0.06])
        sensor_probs_by_region = {
            "europe": np.array([0.82, 0.14, 0.04]),
            "africa": np.array([0.42, 0.08, 0.50]),
            "south_america": np.array([0.30, 0.58, 0.12]),
            "southeast_asia": np.array([0.12, 0.78, 0.10]),
        }
        class_probs_by_region = {
            "europe": np.array([0.70, 0.18, 0.09, 0.03]),
            "africa": np.array([0.13, 0.60, 0.20, 0.07]),
            "south_america": np.array([0.10, 0.13, 0.67, 0.10]),
            "southeast_asia": np.array([0.05, 0.10, 0.20, 0.65]),
        }

        rows: list[dict[str, Any]] = []
        for index in range(self.config.num_samples):
            region = str(rng.choice(self.regions, p=region_probs))
            sensor = str(rng.choice(self.sensors, p=sensor_probs_by_region[region]))
            label = int(rng.choice(self.config.num_classes, p=class_probs_by_region[region]))
            rows.append(
                {
                    "sample_id": f"dummy-{index:04d}",
                    "index": index,
                    "region": region,
                    "sensor": sensor,
                    "label": label,
                    "task": "land_cover_classification",
                }
            )
        return rows

    def load_metadata(self) -> list[dict[str, Any]]:
        return list(self._metadata)

    def load_sample(self, index: int) -> Mapping[str, Any]:
        row = self._metadata[index]
        seed = self.config.seed * 10_000 + index
        rng = np.random.default_rng(seed)
        image = rng.normal(0, 0.35, size=(self.config.bands, self.config.height, self.config.width))

        label = int(row["label"])
        region_index = self.regions.index(str(row["region"]))
        sensor_index = self.sensors.index(str(row["sensor"]))
        image += label * 0.38
        image[:2] += region_index * 0.20
        image[2:4] += sensor_index * 0.18

        # Intentionally make one small deployment slice hard: minority region,
        # radar-heavy samples, and class 3 have weaker class signal.
        if row["region"] == "southeast_asia":
            image += rng.normal(0, 0.55, size=image.shape)
            image[4:] -= label * 0.30
        if row["sensor"] == "sentinel-1":
            image += rng.normal(0, 0.25, size=image.shape)

        return {"image": image.astype(np.float32), "metadata": row}

    def get_labels(self, index: int) -> int:
        return int(self._metadata[index]["label"])

    def get_region(self, index: int) -> str:
        return str(self._metadata[index]["region"])

    def get_sensor(self, index: int) -> str:
        return str(self._metadata[index]["sensor"])

    def get_group_keys(self, index: int) -> dict[str, str]:
        row = self._metadata[index]
        return {
            "region": str(row["region"]),
            "sensor": str(row["sensor"]),
            "task": str(row["task"]),
            "region_class": f"{row['region']}::class_{row['label']}",
            "sensor_class": f"{row['sensor']}::class_{row['label']}",
        }


class DummyModelAdapter(ModelAdapter):
    """Deterministic embedding extractor for smoke-testing the audit pipeline."""

    def __init__(self, embedding_dim: int = 16, seed: int = 17) -> None:
        self.embedding_dim = embedding_dim
        self.seed = seed
        self._projection: np.ndarray | None = None

    def load_model(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._projection = rng.normal(0, 1, size=(32, self.embedding_dim)).astype(np.float32)

    def preprocess(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        images = np.stack([sample["image"] for sample in batch["samples"]]).astype(np.float32)
        return {"images": images, "metadata": batch["metadata"]}

    def extract_embeddings(self, batch: Mapping[str, Any]) -> np.ndarray:
        if self._projection is None:
            raise RuntimeError("load_model() must be called before extract_embeddings().")
        images = batch["images"]
        means = images.mean(axis=(2, 3))
        stds = images.std(axis=(2, 3))
        mins = images.min(axis=(2, 3))
        maxs = images.max(axis=(2, 3))
        features = np.concatenate([means, stds, mins, maxs], axis=1)
        embeddings = features @ self._projection
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)

    def get_supported_modalities(self) -> Sequence[str]:
        return ("sentinel-2", "sentinel-1", "landsat")
