from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from shutil import copyfileobj
from typing import Any

import numpy as np


SOURCE_DATASET = "lc-col/bigearthnet"
TRAIN_CSV = "bigearthnet_hdf5_train.csv"
TRAIN_SHARD_GZ = "bigearthnet_train_p0.hdf5.gz"
TRAIN_SHARD = "bigearthnet_train_p0.hdf5"


def _download_file(filename: str, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: python -m pip install huggingface_hub") from exc
    path = hf_hub_download(
        repo_id=SOURCE_DATASET,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(path)


def _ensure_hdf5(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = _download_file(TRAIN_SHARD_GZ, cache_dir)
    hdf5_path = cache_dir / TRAIN_SHARD
    if hdf5_path.exists() and hdf5_path.stat().st_size > 0:
        return hdf5_path
    print(f"[info] Decompressing {gz_path.name} -> {hdf5_path}")
    with gzip.open(gz_path, "rb") as src, hdf5_path.open("wb") as dst:
        copyfileobj(src, dst)
    return hdf5_path


def _read_train_index(csv_path: Path, shard_name: str) -> list[dict[str, Any]]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("s2_hdf5_file") == shard_name:
                rows.append(row)
    return rows


def _inspect_hdf5(path: Path) -> list[str]:
    import h5py

    keys = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            shape = getattr(obj, "shape", "")
            dtype = getattr(obj, "dtype", "")
            keys.append(f"{name} shape={shape} dtype={dtype}")

        handle.visititems(visitor)
    print("[info] HDF5 keys:")
    for key in keys:
        print(f"  {key}")
    return keys


def _numeric_dataset_candidates(handle: Any, min_rows: int) -> list[tuple[str, Any]]:
    import h5py

    candidates = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and hasattr(obj, "shape") and obj.shape:
            if obj.shape[0] >= min_rows and np.issubdtype(obj.dtype, np.number):
                candidates.append((name, obj))

    handle.visititems(visitor)
    return candidates


def _choose_image_dataset(handle: Any, max_samples: int) -> tuple[str, Any]:
    candidates = []
    for name, dataset in _numeric_dataset_candidates(handle, max_samples):
        if len(dataset.shape) >= 3:
            score = 0
            lname = name.lower()
            if any(token in lname for token in ["image", "s2", "sentinel", "data", "bands", "patch"]):
                score += 10
            score += len(dataset.shape)
            candidates.append((score, name, dataset))
    if not candidates:
        available = _inspect_hdf5(Path(handle.filename))
        raise RuntimeError("Could not find real image arrays in HDF5 shard. Available keys:\n" + "\n".join(available))
    _, name, dataset = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    print(f"[info] Selected image dataset: {name} shape={dataset.shape} dtype={dataset.dtype}")
    return name, dataset


def _choose_label_dataset(handle: Any, max_samples: int) -> tuple[str, Any] | None:
    candidates = []
    for name, dataset in _numeric_dataset_candidates(handle, max_samples):
        lname = name.lower()
        if any(token in lname for token in ["label", "target", "class", "y"]):
            candidates.append((name, dataset))
    if candidates:
        name, dataset = candidates[0]
        print(f"[info] Selected label dataset: {name} shape={dataset.shape} dtype={dataset.dtype}")
        return name, dataset
    return None


def _to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3 and array.shape[0] > 64 and array.shape[-1] <= 64:
        array = np.moveaxis(array, -1, 0)
    elif array.ndim != 3:
        array = np.squeeze(array)
        if array.ndim != 3:
            raise ValueError(f"Expected one image chip with 2D/3D shape, got {array.shape}")
        if array.shape[0] > 64 and array.shape[-1] <= 64:
            array = np.moveaxis(array, -1, 0)
    return array.astype(np.float32)


def _label_fields(label_dataset: Any | None, index: int) -> tuple[int, str, str]:
    if label_dataset is None:
        raise RuntimeError(
            "HDF5 shard contains real image arrays but no label dataset was found. "
            "Available keys were printed above; cannot run fairness/probe evaluation without labels."
        )
    raw = np.asarray(label_dataset[index])
    if raw.ndim == 0:
        label = int(raw.item())
        vector = ""
        names = json.dumps([label])
    else:
        flat = raw.reshape(-1)
        vector_values = [int(value) for value in flat.tolist()]
        label = next((i for i, value in enumerate(vector_values) if value == 1), int(np.argmax(flat)))
        vector = json.dumps(vector_values)
        names = json.dumps(vector_values)
    return label, vector, names


def convert_lccol_subset(output_dir: Path, max_samples: int = 64, seed: int = 42, cache_dir: Path | None = None) -> Path:
    if max_samples > 512:
        raise ValueError("This first lc-col smoke downloader supports at most --max-samples 512.")
    cache_dir = cache_dir or (output_dir / "_hf_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    chip_dir = output_dir / "chips"
    chip_dir.mkdir(exist_ok=True)

    csv_path = _download_file(TRAIN_CSV, cache_dir)
    hdf5_path = _ensure_hdf5(cache_dir)
    index_rows = _read_train_index(csv_path, TRAIN_SHARD)
    if not index_rows:
        raise RuntimeError(f"No rows in {TRAIN_CSV} point to {TRAIN_SHARD}.")

    import h5py

    metadata_rows = []
    with h5py.File(hdf5_path, "r") as handle:
        _inspect_hdf5(hdf5_path)
        _, image_dataset = _choose_image_dataset(handle, max_samples)
        label_choice = _choose_label_dataset(handle, max_samples)
        if label_choice is None:
            _inspect_hdf5(hdf5_path)
            raise RuntimeError("Could not find labels in lc-col HDF5 shard; refusing to create unlabeled fake metadata.")
        _, label_dataset = label_choice

        for out_index, row in enumerate(index_rows[:max_samples]):
            source_index = int(row["index"])
            image = _to_chw(np.asarray(image_dataset[source_index]))
            label, label_vector, labels = _label_fields(label_dataset, source_index)
            sample_id = str(row.get("s2_folder") or f"lc_col_train_p0_{source_index:06d}")
            chip_path = chip_dir / f"{sample_id}_s2.npz"
            np.savez_compressed(chip_path, image=image)
            fallback_group = "lc_col_train_p0"
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "chip_path": str(chip_path.relative_to(output_dir)),
                    "s2_path": str(chip_path.relative_to(output_dir)),
                    "label": label,
                    "label_vector": label_vector,
                    "labels": labels,
                    "label_names": labels,
                    "sensor": "S2",
                    "region": fallback_group,
                    "fallback_group": fallback_group,
                    "source_dataset": SOURCE_DATASET,
                    "source_shard": TRAIN_SHARD,
                    "source_index": source_index,
                }
            )
            if (out_index + 1) % 16 == 0 or out_index + 1 == max_samples:
                print(f"[info] Converted {out_index + 1}/{max_samples} real S2 chips")

    metadata_path = output_dir / "metadata.csv"
    columns = [
        "sample_id",
        "chip_path",
        "s2_path",
        "label",
        "label_vector",
        "labels",
        "label_names",
        "sensor",
        "region",
        "fallback_group",
        "source_dataset",
        "source_shard",
        "source_index",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"[info] Wrote {metadata_path}")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one lc-col BigEarthNet HDF5 shard and convert real S2 chips.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/bigearthnet_lccol_subset"))
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    convert_lccol_subset(args.output_dir, max_samples=args.max_samples, seed=args.seed, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
