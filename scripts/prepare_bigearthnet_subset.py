from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from shutil import copy2
from typing import Any

import numpy as np


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        if not isinstance(data, list):
            raise ValueError("JSON metadata must contain a list or {'samples': [...]}.")
        return [dict(row) for row in data]
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError("This subset helper reads CSV, JSON, or JSONL metadata. Convert official parquet metadata first.")


def _load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "image" if "image" in data else data.files[0]
        return data[key].astype(np.float32)
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("rasterio is required to convert GeoTIFF chips. Install it in a geospatial environment.") from exc
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)
    raise ValueError(f"Unsupported image format for conversion: {path}")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def prepare_subset(
    source_root: Path,
    metadata_path: Path,
    output_root: Path,
    subset_size: int,
    sensor_mode: str,
    copy_only: bool,
) -> Path:
    rows = _read_rows(metadata_path)
    selected = rows[:subset_size]
    output_root.mkdir(parents=True, exist_ok=True)
    chip_dir = output_root / "chips"
    chip_dir.mkdir(exist_ok=True)

    output_rows = []
    sensor_key = "s2_path" if sensor_mode.upper() == "S2" else "s1_path"
    for index, row in enumerate(selected):
        if not row.get(sensor_key):
            raise ValueError(f"Row {index} is missing {sensor_key}; cannot prepare sensor_mode={sensor_mode}.")
        source_path = _resolve(source_root, str(row[sensor_key]))
        sample_id = str(row.get("sample_id") or row.get("id") or f"BEN-{index:06d}")
        if copy_only and source_path.suffix.lower() in {".npy", ".npz"}:
            target = chip_dir / source_path.name
            copy2(source_path, target)
        else:
            target = chip_dir / f"{sample_id}_{sensor_mode.lower()}.npy"
            np.save(target, _load_image(source_path))

        output_row = {
            "sample_id": sample_id,
            "label": row.get("label"),
            "label_vector": row.get("label_vector"),
            "label_names": row.get("label_names") or row.get("labels"),
            "country": row.get("country") or "to_verify",
            "region": row.get("region") or row.get("country") or "to_verify",
            "sensor": sensor_mode.upper(),
            "split": row.get("split") or "all",
            "latitude": row.get("latitude") or row.get("lat") or "",
            "longitude": row.get("longitude") or row.get("lon") or "",
            sensor_key: str(target.relative_to(output_root)),
        }
        output_rows.append(output_row)

    manifest_path = output_root / "metadata.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small BigEarthNet-style subset manifest for smoke runs.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--sensor-mode", choices=["S1", "S2"], default="S2")
    parser.add_argument("--copy-only", action="store_true", help="Copy existing .npy/.npz chips instead of rewriting them.")
    args = parser.parse_args()

    manifest = prepare_subset(
        source_root=args.source_root,
        metadata_path=args.metadata_path,
        output_root=args.output_root,
        subset_size=args.subset_size,
        sensor_mode=args.sensor_mode,
        copy_only=args.copy_only,
    )
    print(f"Wrote prepared subset manifest: {manifest}")


if __name__ == "__main__":
    main()
