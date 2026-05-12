from __future__ import annotations

from typing import Any

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter, ModelAdapter


def extract_embeddings(
    dataset: DatasetAdapter,
    model: ModelAdapter,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    metadata = dataset.load_metadata()
    embeddings: list[np.ndarray] = []
    labels: list[int] = []

    model.load_model()
    for start in range(0, len(metadata), batch_size):
        indices = list(range(start, min(start + batch_size, len(metadata))))
        samples = [dataset.load_sample(index) for index in indices]
        batch = model.preprocess({"samples": samples, "metadata": [metadata[index] for index in indices]})
        embeddings.append(model.extract_embeddings(batch))
        labels.extend(dataset.get_labels(index) for index in indices)

    return np.vstack(embeddings), np.asarray(labels, dtype=np.int64), metadata
