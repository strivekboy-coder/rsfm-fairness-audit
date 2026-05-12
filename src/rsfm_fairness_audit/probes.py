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


class KNNProbe:
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.embeddings: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "KNNProbe":
        self.embeddings = embeddings
        self.labels = labels
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.embeddings is None or self.labels is None:
            raise RuntimeError("Probe must be fitted before predict().")
        distances = ((embeddings[:, None, :] - self.embeddings[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argsort(distances, axis=1)[:, : self.k]
        predictions = []
        for row in nearest:
            labels, counts = np.unique(self.labels[row], return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions)


class LinearRidgeProbe:
    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "LinearRidgeProbe":
        self.classes_ = np.unique(labels)
        targets = np.zeros((len(labels), len(self.classes_)), dtype=np.float32)
        for column, cls in enumerate(self.classes_):
            targets[:, column] = labels == cls
        x = np.c_[embeddings, np.ones(len(embeddings))]
        regularizer = self.alpha * np.eye(x.shape[1], dtype=np.float32)
        self.weights_ = np.linalg.solve(x.T @ x + regularizer, x.T @ targets)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.weights_ is None:
            raise RuntimeError("Probe must be fitted before predict().")
        x = np.c_[embeddings, np.ones(len(embeddings))]
        scores = x @ self.weights_
        return self.classes_[np.argmax(scores, axis=1)]


def evaluate_probe_suite(embeddings: np.ndarray, labels: np.ndarray) -> list[dict[str, float | str]]:
    probes = {
        "nearest_centroid": NearestCentroidProbe(),
        "knn_k3": KNNProbe(k=min(3, len(labels))),
        "linear_ridge": LinearRidgeProbe(),
        "multilabel_one_vs_rest_ridge": LinearRidgeProbe(),
    }
    rows = []
    for name, probe in probes.items():
        predictions = probe.fit(embeddings, labels).predict(embeddings)
        rows.append({"probe": name, "accuracy": float(np.mean(predictions == labels))})
    return rows
