from __future__ import annotations

import numpy as np


class NearestCentroidProbe:
    """Tiny deterministic classifier for CPU smoke tests."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "NearestCentroidProbe":
        classes = np.unique(labels)
        centroids = []
        for cls in classes:
            centroids.append(embeddings[labels == cls].mean(axis=0))
        self.classes_ = classes
        self.centroids_ = np.vstack(centroids)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.centroids_ is None:
            raise RuntimeError("Probe must be fitted before predict().")
        distances = ((embeddings[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(distances, axis=1)]
