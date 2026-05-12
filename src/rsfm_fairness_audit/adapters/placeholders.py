from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import ModelAdapter


class PlaceholderModelAdapter(ModelAdapter):
    """Explicit placeholder for real RSFM integrations."""

    def __init__(self, model_name: str, supported_modalities: Sequence[str]) -> None:
        self.model_name = model_name
        self.supported_modalities = tuple(supported_modalities)

    def load_model(self) -> None:
        raise NotImplementedError(
            f"{self.model_name} is registered but not implemented in Milestone 1. "
            "Use official model cards, repositories, and papers before adding checkpoints."
        )

    def preprocess(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    def extract_embeddings(self, batch: Mapping[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def get_supported_modalities(self) -> Sequence[str]:
        return self.supported_modalities


def build_placeholder_adapters() -> dict[str, PlaceholderModelAdapter]:
    return {
        "dofa": PlaceholderModelAdapter("DOFA", ("sentinel-2", "sentinel-1", "landsat")),
        "croma": PlaceholderModelAdapter("CROMA", ("sentinel-2", "sentinel-1")),
        "prithvi-eo-2.0": PlaceholderModelAdapter("Prithvi-EO-2.0", ("sentinel-2", "landsat")),
        "clay": PlaceholderModelAdapter("Clay", ("sentinel-2", "sentinel-1", "landsat")),
        "anysat": PlaceholderModelAdapter("AnySat", ("sentinel-2", "sentinel-1")),
    }
