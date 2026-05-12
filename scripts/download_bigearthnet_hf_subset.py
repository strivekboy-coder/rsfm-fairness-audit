from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_ORDER = [
    "GFM-Bench/BigEarthNet",
    "lc-col/bigearthnet",
    "hackelle/BigEarthNetV2-LMDB",
]

S2_KEYS = ["s2", "S2", "sentinel2", "sentinel_2", "sentinel-2", "optical", "image", "images", "bands", "array", "chip", "patch", "x"]
S1_KEYS = ["s1", "S1", "sentinel1", "sentinel_1", "sentinel-1", "sar", "radar", "vv_vh"]
LABEL_KEYS = ["label", "labels", "class", "classes", "target", "targets", "y", "land_cover", "multilabel"]
REGION_KEYS = ["region", "country", "country_name", "tile", "location", "continent", "fallback_group"]
LAT_KEYS = ["latitude", "lat", "center_lat", "centroid_lat"]
LON_KEYS = ["longitude", "lon", "lng", "center_lon", "centroid_lon"]


class SourceExhausted(RuntimeError):
    pass


def _source_sequence(requested: str) -> list[str]:
    if requested in {"auto", "all"}:
        return list(SOURCE_ORDER)
    return [requested] + [source for source in SOURCE_ORDER if source != requested]


def _import_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "This script requires the optional Hugging Face datasets package. "
            "Install it in Colab with: python -m pip install datasets huggingface_hub pillow"
        ) from exc
    return load_dataset


def _try_load_dataset(source: str, split: str, streaming: bool) -> Iterable[dict[str, Any]]:
    load_dataset = _import_datasets()
    errors = []
    candidate_splits = [split]
    for fallback_split in ["train", "all_data", "validation", "test"]:
        if fallback_split not in candidate_splits:
            candidate_splits.append(fallback_split)
    attempts = []
    for candidate_split in candidate_splits:
        attempts.extend(
            [
                {"path": source, "split": candidate_split, "streaming": streaming},
                {"path": source, "split": candidate_split, "streaming": streaming, "trust_remote_code": True},
            ]
        )
    for kwargs in attempts:
        try:
            return load_dataset(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on remote HF state
            errors.append(f"{kwargs}: {exc}")
    raise SourceExhausted(f"Could not load {source} through datasets.load_dataset. Attempts: {' | '.join(errors)}")


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "__array__") and not isinstance(value, (str, bytes)):
        array = np.asarray(value)
    elif hasattr(value, "convert") and hasattr(value, "size"):
        array = np.asarray(value)
    elif isinstance(value, list):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if array.ndim < 2 or not np.issubdtype(array.dtype, np.number):
        return None
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3:
        if array.shape[0] <= 32:
            pass
        elif array.shape[-1] <= 32:
            array = np.moveaxis(array, -1, 0)
        else:
            return None
    else:
        array = np.squeeze(array)
        if array.ndim != 3:
            return None
        if array.shape[0] > 32 and array.shape[-1] <= 32:
            array = np.moveaxis(array, -1, 0)
    return array.astype(np.float32)


def _find_array_in_candidates(row: dict[str, Any], keys: list[str]) -> np.ndarray | None:
    for key in keys:
        if key in row:
            array = _as_numpy(row[key])
            if array is not None:
                return array
            if isinstance(row[key], dict):
                nested = _find_array_in_candidates(row[key], keys)
                if nested is not None:
                    return nested
    return None


def _find_any_image_array(value: Any, skip_keys: set[str] | None = None) -> np.ndarray | None:
    skip_keys = skip_keys or set()
    array = _as_numpy(value)
    if array is not None and array.ndim == 3:
        return array
    if isinstance(value, dict):
        for key, item in value.items():
            if key in skip_keys:
                continue
            found = _find_any_image_array(item, skip_keys)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_any_image_array(item, skip_keys)
            if found is not None:
                return found
    return None


def _extract_image(row: dict[str, Any], sensor_mode: str) -> np.ndarray | dict[str, np.ndarray] | None:
    sensor_mode = sensor_mode.upper()
    if sensor_mode == "S2":
        found = _find_array_in_candidates(row, S2_KEYS)
        return found if found is not None else _find_any_image_array(row, set(LABEL_KEYS))
    if sensor_mode == "S1":
        return _find_array_in_candidates(row, S1_KEYS)
    s1 = _find_array_in_candidates(row, S1_KEYS)
    s2 = _find_array_in_candidates(row, S2_KEYS)
    if s1 is not None and s2 is not None:
        return {"S1": s1, "S2": s2}
    return None


def _extract_labels(row: dict[str, Any]) -> tuple[int, str, str]:
    for key in LABEL_KEYS:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            labels = list(value)
            if not labels:
                continue
            if all(isinstance(item, (int, float, np.integer, np.floating, bool)) for item in labels):
                vector = [int(item) for item in labels]
                primary = next((idx for idx, item in enumerate(vector) if int(item) == 1), int(vector[0]))
                return int(primary), json.dumps(vector), json.dumps(labels)
            return 0, "", json.dumps([str(item) for item in labels])
        if isinstance(value, (int, float, np.integer, np.floating)):
            return int(value), "", json.dumps([int(value)])
        if isinstance(value, str) and value.strip():
            return 0, "", json.dumps([value.strip()])
    raise ValueError("Row has image data but no recognizable label field.")


def _first_present(row: dict[str, Any], keys: list[str], default: str = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _save_chip(output_dir: Path, sample_id: str, sensor_mode: str, image: np.ndarray | dict[str, np.ndarray]) -> tuple[str, str, str]:
    chip_dir = output_dir / "chips"
    chip_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, dict):
        s1_path = chip_dir / f"{sample_id}_s1.npz"
        s2_path = chip_dir / f"{sample_id}_s2.npz"
        np.savez_compressed(s1_path, image=image["S1"])
        np.savez_compressed(s2_path, image=image["S2"])
        return "", str(s1_path.relative_to(output_dir)), str(s2_path.relative_to(output_dir))
    chip_path = chip_dir / f"{sample_id}_{sensor_mode.lower()}.npz"
    np.savez_compressed(chip_path, image=image)
    if sensor_mode.upper() == "S1":
        return str(chip_path.relative_to(output_dir)), str(chip_path.relative_to(output_dir)), ""
    return str(chip_path.relative_to(output_dir)), "", str(chip_path.relative_to(output_dir))


def _write_metadata(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    columns = [
        "sample_id",
        "chip_path",
        "s1_path",
        "s2_path",
        "label",
        "label_vector",
        "labels",
        "label_names",
        "sensor",
        "region",
        "fallback_group",
        "latitude",
        "longitude",
        "source_dataset",
        "source_index",
        "split",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])
    return metadata_path


def download_subset(
    source: str,
    output_dir: Path,
    max_samples: int,
    sensor_mode: str,
    seed: int,
    split: str = "train",
    streaming: bool = True,
) -> Path:
    rng = random.Random(seed)
    errors = []
    for candidate_source in _source_sequence(source):
        rows: list[dict[str, Any]] = []
        try:
            dataset = _try_load_dataset(candidate_source, split=split, streaming=streaming)
            iterable = iter(dataset)
            skipped_no_image = 0
            skipped_no_label = 0
            for source_index, row in enumerate(iterable):
                image = _extract_image(row, sensor_mode)
                if image is None:
                    skipped_no_image += 1
                    if skipped_no_image >= 32 and not rows:
                        raise SourceExhausted(
                            f"{candidate_source} appears to expose metadata only or unsupported chip fields; "
                            "no real image arrays/chips were found in the first rows."
                        )
                    continue
                try:
                    label, label_vector, label_names = _extract_labels(row)
                except ValueError:
                    skipped_no_label += 1
                    continue
                sample_id = str(row.get("sample_id") or row.get("id") or row.get("patch_id") or f"{candidate_source.replace('/', '_')}_{source_index:06d}")
                chip_path, s1_path, s2_path = _save_chip(output_dir, sample_id, sensor_mode, image)
                fallback_group = str(_first_present(row, REGION_KEYS, default="to_verify"))
                rows.append(
                    {
                        "sample_id": sample_id,
                        "chip_path": chip_path,
                        "s1_path": s1_path,
                        "s2_path": s2_path,
                        "label": label,
                        "label_vector": label_vector,
                        "labels": label_names,
                        "label_names": label_names,
                        "sensor": sensor_mode.upper(),
                        "region": fallback_group,
                        "fallback_group": fallback_group,
                        "latitude": _first_present(row, LAT_KEYS, default=""),
                        "longitude": _first_present(row, LON_KEYS, default=""),
                        "source_dataset": candidate_source,
                        "source_index": source_index,
                        "split": split,
                    }
                )
                print(f"[info] {candidate_source}: converted {len(rows)}/{max_samples} samples")
                if len(rows) >= max_samples:
                    metadata_path = _write_metadata(output_dir, rows)
                    print(f"[info] Wrote {metadata_path}")
                    return metadata_path
            raise SourceExhausted(
                f"{candidate_source} ended before {max_samples} usable real image samples were found "
                f"(converted={len(rows)}, skipped_no_image={skipped_no_image}, skipped_no_label={skipped_no_label})."
            )
        except Exception as exc:
            errors.append(f"{candidate_source}: {exc}")
            print(f"[warn] {candidate_source} failed: {exc}")
            continue
    raise RuntimeError("No requested Hugging Face BigEarthNet source yielded real image chips. " + " | ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a tiny real BigEarthNet-compatible subset from Hugging Face.")
    parser.add_argument("--source", default="auto", help="HF dataset repo or 'auto'.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--sensor-mode", choices=["S1", "S2", "S1+S2"], default="S2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    parser.add_argument("--no-streaming", action="store_true", help="Use non-streaming datasets.load_dataset.")
    args = parser.parse_args()

    download_subset(
        source=args.source,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        sensor_mode=args.sensor_mode,
        seed=args.seed,
        split=args.split,
        streaming=not args.no_streaming,
    )


if __name__ == "__main__":
    main()
