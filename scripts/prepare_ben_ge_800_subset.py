from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


ZENODO_URL = "https://zenodo.org/records/12941231/files/ben-ge-800.tar.gz?download=1"
ARCHIVE_NAME = "ben-ge-800.tar.gz"
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]
S1_BANDS = ["VV", "VH"]


def _download_archive(cache_dir: Path, url: str = ZENODO_URL) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / ARCHIVE_NAME
    if archive_path.exists() and archive_path.stat().st_size > 0:
        print(f"[info] Using cached BEN-GE-800 archive: {archive_path}")
        return archive_path
    print(f"[info] Downloading BEN-GE-800 archive from Zenodo: {url}")
    urllib.request.urlretrieve(url, archive_path)
    return archive_path


def _safe_extract_tar(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".ben_ge_800_extracted"
    if marker.exists():
        print(f"[info] Using cached extracted BEN-GE-800 directory: {extract_dir}")
        return
    print(f"[info] Extracting {archive_path} -> {extract_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        root = extract_dir.resolve()
        for member in tar.getmembers():
            target = (extract_dir / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"Refusing unsafe tar member path: {member.name}")
        tar.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")


def _find_dataset_root(extract_dir: Path) -> Path:
    for candidate in [extract_dir, *extract_dir.iterdir()]:
        if candidate.is_dir() and (candidate / "ben-ge-800_meta.csv").exists():
            return candidate
    raise FileNotFoundError(f"Could not find ben-ge-800_meta.csv under {extract_dir}")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels") or data.get("labels_metadata") or data.get("label")
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return []


def _read_tif(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Preparing BEN-GE-800 requires rasterio. In Colab, run: python -m pip install rasterio") from exc
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _stack_s1(dataset_root: Path, patch_id_s1: str) -> np.ndarray:
    folder = dataset_root / "sentinel-1" / patch_id_s1
    bands = []
    for band in S1_BANDS:
        path = folder / f"{patch_id_s1}_{band}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing BEN-GE S1 band file: {path}")
        bands.append(_read_tif(path))
    return np.stack(bands).astype(np.float32)


def _stack_s2(dataset_root: Path, patch_id: str) -> np.ndarray:
    folder = dataset_root / "sentinel-2" / patch_id
    bands = []
    for band in S2_BANDS:
        path = folder / f"{patch_id}_{band}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing BEN-GE S2 band file: {path}")
        bands.append(_read_tif(path))
    return np.stack(bands).astype(np.float32)


def _label_vocab(rows: list[dict[str, Any]], dataset_root: Path) -> list[str]:
    labels = set()
    for row in rows:
        patch_id = str(row["patch_id"])
        label_path = dataset_root / "sentinel-2" / patch_id / f"{patch_id}_labels_metadata.json"
        labels.update(_read_labels(label_path))
    return sorted(labels)


def _label_fields(row: dict[str, Any], dataset_root: Path, vocab: list[str]) -> tuple[int, str, str]:
    patch_id = str(row["patch_id"])
    label_path = dataset_root / "sentinel-2" / patch_id / f"{patch_id}_labels_metadata.json"
    labels = _read_labels(label_path)
    if not labels:
        climate = int(float(row.get("climatezone") or 0))
        return climate, "", json.dumps([f"climatezone_{climate}"], ensure_ascii=True)
    label_set = set(labels)
    vector = [1 if label in label_set else 0 for label in vocab]
    primary = next((index for index, value in enumerate(vector) if value), 0)
    return primary, json.dumps(vector, ensure_ascii=True), json.dumps(labels, ensure_ascii=True)


def _balanced_select(rows: list[dict[str, Any]], subset_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get("climatezone") or "to_verify")
        groups.setdefault(group, []).append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    selected: list[dict[str, Any]] = []
    names = sorted(groups)
    cursor = 0
    while len(selected) < subset_size and any(groups.values()):
        name = names[cursor % len(names)]
        if groups[name]:
            selected.append(groups[name].pop())
        cursor += 1
    return selected


def prepare_ben_ge_800_subset(
    output_dir: Path,
    max_samples: int = 64,
    seed: int = 42,
    cache_dir: Path = Path("data/_cache/ben_ge_800"),
    source_dir: Path | None = None,
) -> Path:
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if source_dir is None:
        archive_path = _download_archive(cache_dir)
        extract_dir = cache_dir / "extracted"
        _safe_extract_tar(archive_path, extract_dir)
        dataset_root = _find_dataset_root(extract_dir)
    else:
        dataset_root = source_dir
    rows = _read_rows(dataset_root / "ben-ge-800_meta.csv")
    selected = _balanced_select(rows, min(max_samples, len(rows)), seed)
    vocab = _label_vocab(selected, dataset_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    chip_dir = output_dir / "chips"
    chip_dir.mkdir(exist_ok=True)
    metadata_rows = []
    for index, row in enumerate(selected, start=1):
        patch_id = str(row["patch_id"])
        patch_id_s1 = str(row["patch_id_s1"])
        s1_path = chip_dir / f"{patch_id}_s1.npz"
        s2_path = chip_dir / f"{patch_id}_s2.npz"
        if not s1_path.exists():
            np.savez_compressed(s1_path, image=_stack_s1(dataset_root, patch_id_s1))
        if not s2_path.exists():
            np.savez_compressed(s2_path, image=_stack_s2(dataset_root, patch_id))
        label, label_vector, label_names = _label_fields(row, dataset_root, vocab)
        climatezone = str(row.get("climatezone") or "to_verify")
        metadata_rows.append(
            {
                "sample_id": patch_id,
                "patch_id": patch_id,
                "patch_id_s1": patch_id_s1,
                "s1_path": str(s1_path.relative_to(output_dir)),
                "s2_path": str(s2_path.relative_to(output_dir)),
                "label": label,
                "label_vector": label_vector,
                "label_names": label_names,
                "region": f"climatezone_{climatezone}",
                "fallback_group": f"climatezone_{climatezone}",
                "climatezone": climatezone,
                "latitude": row.get("lat", ""),
                "longitude": row.get("lon", ""),
                "sensor": "S1+S2",
                "source_dataset": "ben-ge-800",
                "band_profile_s1": "sentinel1_vv_vh",
                "band_profile_s2": "sentinel2_12_croma",
                "split": "all",
            }
        )
        if index % 16 == 0 or index == len(selected):
            print(f"[info] Prepared {index}/{len(selected)} BEN-GE-800 paired samples")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"[info] Wrote {metadata_path}")
    print(f"[summary] samples={len(metadata_rows)} labels={len(vocab)} source={dataset_root}")
    print(f"[summary] output={output_dir} s1_s2_chips={len(list(chip_dir.glob('*.npz')))}")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare a BEN-GE-800 paired S1/S2 subset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ben_ge_800_subset"))
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/_cache/ben_ge_800"))
    parser.add_argument("--source-dir", type=Path, help="Optional already-extracted BEN-GE-800 root.")
    args = parser.parse_args()
    prepare_ben_ge_800_subset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        cache_dir=args.cache_dir,
        source_dir=args.source_dir,
    )


if __name__ == "__main__":
    main()
