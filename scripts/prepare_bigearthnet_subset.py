from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from shutil import copy2
from typing import Any, Iterable

import numpy as np


SUPPORTED_SUBSET_SIZES = {32, 500, 1000, 5000, 10000}


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
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading parquet metadata requires pandas plus pyarrow or fastparquet.") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError("Metadata must be CSV, JSON, JSONL, NDJSON, or parquet.")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _read_existing_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows selected; refusing to write an empty BigEarthNet subset manifest.")
    columns = [
        "sample_id",
        "label",
        "label_vector",
        "label_names",
        "country",
        "region",
        "sensor",
        "split",
        "latitude",
        "longitude",
        "s1_path",
        "s2_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _stringify(row.get(column)) for column in columns})


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


def _resolve(root: Path, value: Any) -> Path | None:
    if value in (None, "", "nan"):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _parse_label_vector(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.tolist()]
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.split(";") if part.strip()]
    if isinstance(parsed, list):
        return [int(float(v)) for v in parsed]
    return None


def _primary_class(row: dict[str, Any], label_column: str, label_vector_column: str) -> str:
    vector = _parse_label_vector(row.get(label_vector_column))
    if vector is not None:
        for index, value in enumerate(vector):
            if value == 1:
                return str(index)
        return "0"
    value = row.get(label_column)
    if value in (None, ""):
        return "to_verify"
    return str(value)


def _group_key(row: dict[str, Any], stratify_by: str, region_column: str, label_column: str, label_vector_column: str) -> str:
    region = str(row.get(region_column) or row.get("country") or "to_verify")
    label = _primary_class(row, label_column, label_vector_column)
    if stratify_by == "region":
        return region
    if stratify_by == "class":
        return label
    if stratify_by == "region_class":
        return f"{region}::{label}"
    return "all"


def _stratified_select(
    rows: list[dict[str, Any]],
    subset_size: int,
    stratify_by: str,
    seed: int,
    region_column: str,
    label_column: str,
    label_vector_column: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row, stratify_by, region_column, label_column, label_vector_column), []).append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    selected: list[dict[str, Any]] = []
    group_names = sorted(groups)
    cursor = 0
    while len(selected) < subset_size and any(groups.values()):
        group_name = group_names[cursor % len(group_names)]
        if groups[group_name]:
            selected.append(groups[group_name].pop())
        cursor += 1
    return selected


def _target_path(chip_dir: Path, sample_id: str, sensor: str, chip_format: str) -> Path:
    return chip_dir / f"{sample_id}_{sensor.lower()}.{chip_format}"


def _materialize_chip(
    source_path: Path,
    target_path: Path,
    chip_format: str,
    copy_existing: bool,
    overwrite: bool,
) -> None:
    if target_path.exists() and not overwrite:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_existing and source_path.suffix.lower() in {".npy", ".npz"} and source_path.suffix.lower().lstrip(".") == chip_format:
        copy2(source_path, target_path)
        return
    image = _load_image(source_path)
    if chip_format == "npy":
        np.save(target_path, image)
    elif chip_format == "npz":
        np.savez_compressed(target_path, image=image)
    else:
        raise ValueError("chip_format must be 'npy' or 'npz'.")


def _require_columns(rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
    available = set(rows[0].keys()) if rows else set()
    missing = [column for column in columns if column and column not in available]
    if missing:
        raise ValueError(f"Metadata is missing required column(s): {missing}. Available columns: {sorted(available)}")


def prepare_subset(
    source_root: Path,
    metadata_path: Path,
    output_root: Path,
    subset_size: int,
    sensor_mode: str,
    seed: int = 13,
    stratify_by: str = "region_class",
    split: str = "all",
    sample_id_column: str = "sample_id",
    label_column: str = "label",
    label_vector_column: str = "label_vector",
    label_names_column: str = "label_names",
    region_column: str = "region",
    country_column: str = "country",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    split_column: str = "split",
    s1_path_column: str = "s1_path",
    s2_path_column: str = "s2_path",
    chip_format: str = "npy",
    copy_existing: bool = False,
    overwrite: bool = False,
    progress_every: int = 100,
) -> Path:
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    if subset_size not in SUPPORTED_SUBSET_SIZES:
        print(f"[warn] subset_size={subset_size} is allowed, but recommended sizes are {sorted(SUPPORTED_SUBSET_SIZES)}.")
    sensor_mode = sensor_mode.upper()
    if sensor_mode not in {"S1", "S2", "S1+S2"}:
        raise ValueError("sensor_mode must be S1, S2, or S1+S2.")

    rows = _read_rows(metadata_path)
    if not rows:
        raise ValueError(f"No metadata rows found in {metadata_path}.")
    path_columns = []
    if sensor_mode in {"S1", "S1+S2"}:
        path_columns.append(s1_path_column)
    if sensor_mode in {"S2", "S1+S2"}:
        path_columns.append(s2_path_column)
    _require_columns(rows, [sample_id_column, *path_columns])

    if split != "all":
        rows = [row for row in rows if str(row.get(split_column, "all")).lower() == split.lower()]
    if not rows:
        raise ValueError(f"No rows remain after split={split!r} filtering.")

    selected = _stratified_select(
        rows,
        subset_size=subset_size,
        stratify_by=stratify_by,
        seed=seed,
        region_column=region_column,
        label_column=label_column,
        label_vector_column=label_vector_column,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    chip_dir = output_root / "chips"
    chip_dir.mkdir(exist_ok=True)
    manifest_path = output_root / "metadata.csv"
    prepared_rows_by_id = {row["sample_id"]: row for row in _read_existing_manifest(manifest_path)}
    output_rows: list[dict[str, Any]] = []

    print(f"[info] Preparing {len(selected)} real BigEarthNet rows from {metadata_path}")
    print(f"[info] sensor_mode={sensor_mode} stratify_by={stratify_by} seed={seed} output={output_root}")

    for index, row in enumerate(selected, start=1):
        sample_id = str(row.get(sample_id_column) or row.get("id") or "").strip()
        if not sample_id:
            raise ValueError(f"Selected row {index} has no sample id in column {sample_id_column!r}.")

        manifest_row = {
            "sample_id": sample_id,
            "label": row.get(label_column),
            "label_vector": row.get(label_vector_column),
            "label_names": row.get(label_names_column) or row.get("labels"),
            "country": row.get(country_column) or "to_verify",
            "region": row.get(region_column) or row.get(country_column) or "to_verify",
            "sensor": sensor_mode,
            "split": row.get(split_column) or "all",
            "latitude": row.get(lat_column) or row.get("lat") or "",
            "longitude": row.get(lon_column) or row.get("lon") or "",
            "s1_path": "",
            "s2_path": "",
        }

        if sensor_mode in {"S1", "S1+S2"}:
            source = _resolve(source_root, row.get(s1_path_column))
            if source is None or not source.exists():
                raise FileNotFoundError(f"Missing S1 source for sample {sample_id}: {source}")
            target = _target_path(chip_dir, sample_id, "s1", chip_format)
            _materialize_chip(source, target, chip_format, copy_existing, overwrite)
            manifest_row["s1_path"] = str(target.relative_to(output_root))

        if sensor_mode in {"S2", "S1+S2"}:
            source = _resolve(source_root, row.get(s2_path_column))
            if source is None or not source.exists():
                raise FileNotFoundError(f"Missing S2 source for sample {sample_id}: {source}")
            target = _target_path(chip_dir, sample_id, "s2", chip_format)
            _materialize_chip(source, target, chip_format, copy_existing, overwrite)
            manifest_row["s2_path"] = str(target.relative_to(output_root))

        if sample_id in prepared_rows_by_id and not overwrite:
            manifest_row.update({key: value for key, value in prepared_rows_by_id[sample_id].items() if value})

        output_rows.append(manifest_row)
        if index == len(selected) or index % progress_every == 0:
            print(f"[info] Prepared {index}/{len(selected)} samples")

    _write_manifest(manifest_path, output_rows)
    print(f"[info] Wrote adapter-compatible manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a real BigEarthNet subset for DOFA fairness smoke runs.")
    parser.add_argument("--source-root", type=Path, required=True, help="Root containing the chip paths referenced by metadata.")
    parser.add_argument("--metadata-path", type=Path, required=True, help="CSV/JSON/JSONL/parquet metadata table.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=500, help="Recommended: 500, 1000, 5000, or 10000.")
    parser.add_argument("--sensor-mode", choices=["S1", "S2", "S1+S2"], default="S2")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--stratify-by", choices=["none", "region", "class", "region_class"], default="region_class")
    parser.add_argument("--split", default="all")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--label-vector-column", default="label_vector")
    parser.add_argument("--label-names-column", default="label_names")
    parser.add_argument("--region-column", default="region")
    parser.add_argument("--country-column", default="country")
    parser.add_argument("--lat-column", default="latitude")
    parser.add_argument("--lon-column", default="longitude")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--s1-path-column", default="s1_path")
    parser.add_argument("--s2-path-column", default="s2_path")
    parser.add_argument("--chip-format", choices=["npy", "npz"], default="npy")
    parser.add_argument("--copy-existing", action="store_true", help="Copy existing .npy/.npz chips when possible.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute chips even when targets already exist.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    manifest = prepare_subset(
        source_root=args.source_root,
        metadata_path=args.metadata_path,
        output_root=args.output_root,
        subset_size=args.subset_size,
        sensor_mode=args.sensor_mode,
        seed=args.seed,
        stratify_by=args.stratify_by,
        split=args.split,
        sample_id_column=args.sample_id_column,
        label_column=args.label_column,
        label_vector_column=args.label_vector_column,
        label_names_column=args.label_names_column,
        region_column=args.region_column,
        country_column=args.country_column,
        lat_column=args.lat_column,
        lon_column=args.lon_column,
        split_column=args.split_column,
        s1_path_column=args.s1_path_column,
        s2_path_column=args.s2_path_column,
        chip_format=args.chip_format,
        copy_existing=args.copy_existing,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(f"Wrote prepared subset manifest: {manifest}")


if __name__ == "__main__":
    main()
