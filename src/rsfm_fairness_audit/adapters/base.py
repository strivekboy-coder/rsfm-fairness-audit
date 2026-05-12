from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import numpy as np


class ModelAdapter(ABC):
    """Common interface for Remote Sensing Foundation Model adapters."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and preprocessing state."""

    @abstractmethod
    def preprocess(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Convert a raw sample batch into model-ready inputs."""

    @abstractmethod
    def extract_embeddings(self, batch: Mapping[str, Any]) -> np.ndarray:
        """Return one embedding vector per sample."""

    @abstractmethod
    def get_supported_modalities(self) -> Sequence[str]:
        """Return supported sensor or modality names."""


class DatasetAdapter(ABC):
    """Common interface for remote-sensing dataset adapters."""

    @abstractmethod
    def load_metadata(self) -> list[dict[str, Any]]:
        """Return metadata rows for all samples."""

    @abstractmethod
    def load_sample(self, index: int) -> Mapping[str, Any]:
        """Return one raw sample."""

    @abstractmethod
    def get_labels(self, index: int) -> int:
        """Return target label for a sample."""

    @abstractmethod
    def get_region(self, index: int) -> str:
        """Return geographic grouping key."""

    @abstractmethod
    def get_sensor(self, index: int) -> str:
        """Return sensor or modality grouping key."""

    @abstractmethod
    def get_group_keys(self, index: int) -> dict[str, str]:
        """Return all slicing keys for a sample."""
