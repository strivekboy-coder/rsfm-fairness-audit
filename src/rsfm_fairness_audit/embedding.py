from __future__ import annotations

from typing import Any
from pathlib import Path
import json

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter, ModelAdapter
from rsfm_fairness_audit.memory import log_memory, release_memory


def extract_embeddings(
    dataset: DatasetAdapter,
    model: ModelAdapter,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    log_memory("dataset loading:start")
    metadata = dataset.load_metadata()
    log_memory("dataset loading:done")
    embeddings: list[np.ndarray] = []
    labels: list[int] = []

    log_memory("model loading:start")
    model.load_model()
    log_memory("model loading:done")
    for start in range(0, len(metadata), batch_size):
        indices = list(range(start, min(start + batch_size, len(metadata))))
        samples = [dataset.load_sample(index) for index in indices]
        batch = model.preprocess({"samples": samples, "metadata": [metadata[index] for index in indices]})
        embeddings.append(model.extract_embeddings(batch))
        labels.extend(dataset.get_labels(index) for index in indices)
        release_memory()

    log_memory("embedding aggregation:start")
    return np.vstack(embeddings), np.asarray(labels, dtype=np.int64), metadata


def extract_embeddings_to_chunks(
    dataset: DatasetAdapter,
    model: ModelAdapter,
    output_dir: str | Path,
    batch_size: int = 32,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    log_memory("dataset loading:start")
    metadata = dataset.load_metadata()
    log_memory("dataset loading:done")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels: list[int] = []
    chunk_paths: list[Path] = []

    log_memory("model loading:start")
    model.load_model()
    log_memory("model loading:done")
    log_memory("embedding extraction:start")
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
            release_memory()
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
        log_memory(f"embedding chunk written:{chunk_end}/{len(metadata)}")
        del chunk_embeddings, embeddings, labels_array
        release_memory()

    log_memory("chunk merge:start")
    if not chunk_paths:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), []
    shapes: list[tuple[int, ...]] = []
    label_shapes: list[tuple[int, ...]] = []
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as data:
            shapes.append(tuple(data["embeddings"].shape))
            label_shapes.append(tuple(data["labels"].shape))
    feature_dims = {shape[1:] for shape in shapes}
    if len(feature_dims) != 1:
        raise ValueError(f"Embedding chunks have inconsistent feature shapes: {sorted(feature_dims)}")
    total_rows = sum(shape[0] for shape in shapes)
    feature_shape = shapes[0][1:]
    merged_embeddings = np.empty((total_rows, *feature_shape), dtype=np.float32)
    merged_labels = np.empty((sum(shape[0] for shape in label_shapes),), dtype=np.int64)
    merged_metadata: list[dict[str, Any]] = []
    offset = 0
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as data:
            emb = np.asarray(data["embeddings"], dtype=np.float32)
            lab = np.asarray(data["labels"], dtype=np.int64)
            end = offset + emb.shape[0]
            merged_embeddings[offset:end] = emb
            merged_labels[offset:end] = lab
            merged_metadata.extend(json.loads(str(item)) for item in data["metadata_json"])
            offset = end
        release_memory()
    log_memory("chunk merge:done")
    return merged_embeddings, merged_labels, merged_metadata
