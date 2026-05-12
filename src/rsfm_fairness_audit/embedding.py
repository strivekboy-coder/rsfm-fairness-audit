from __future__ import annotations

from typing import Any
from pathlib import Path
import json

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


def extract_embeddings_to_chunks(
    dataset: DatasetAdapter,
    model: ModelAdapter,
    output_dir: str | Path,
    batch_size: int = 32,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    metadata = dataset.load_metadata()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels: list[int] = []
    chunk_paths: list[Path] = []

    model.load_model()
    for chunk_start in range(0, len(metadata), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(metadata))
        chunk_embeddings: list[np.ndarray] = []
        chunk_labels: list[int] = []
        for start in range(chunk_start, chunk_end, batch_size):
            indices = list(range(start, min(start + batch_size, chunk_end)))
            samples = [dataset.load_sample(index) for index in indices]
            batch = model.preprocess({"samples": samples, "metadata": [metadata[index] for index in indices]})
            chunk_embeddings.append(model.extract_embeddings(batch))
            chunk_labels.extend(dataset.get_labels(index) for index in indices)
            del samples, batch
        embeddings = np.vstack(chunk_embeddings)
        labels_array = np.asarray(chunk_labels, dtype=np.int64)
        chunk_metadata = metadata[chunk_start:chunk_end]
        chunk_path = output / f"chunk_{len(chunk_paths):05d}.npz"
        np.savez_compressed(
            chunk_path,
            embeddings=embeddings,
            labels=labels_array,
            metadata_json=np.asarray([json.dumps(row, ensure_ascii=True) for row in chunk_metadata]),
        )
        chunk_paths.append(chunk_path)
        labels.extend(chunk_labels)
        print(f"[info] Wrote embedding chunk {chunk_path} ({chunk_end}/{len(metadata)})")
        del chunk_embeddings, embeddings, labels_array

    merged_embeddings = []
    merged_labels = []
    merged_metadata: list[dict[str, Any]] = []
    for chunk_path in chunk_paths:
        data = np.load(chunk_path, allow_pickle=False)
        merged_embeddings.append(data["embeddings"])
        merged_labels.append(data["labels"])
        merged_metadata.extend(json.loads(str(item)) for item in data["metadata_json"])
    return np.vstack(merged_embeddings), np.concatenate(merged_labels), merged_metadata
