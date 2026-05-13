from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


GCS_ROOT = "gs://sen1floods11"
PRITHVI_BAND_INDICES = [1, 2, 3, 4, 5, 6]  # B02-B07 from 13-band Sen1Floods11 S2 GeoTIFFs.
TARGET_SIZE = 224
WATER_PRESENT_THRESHOLD = 0.01


def _resize_2d(array: np.ndarray, target_size: int, nearest: bool = False) -> np.ndarray:
    if array.shape == (target_size, target_size):
        return array.astype(np.float32)
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        y_idx = np.linspace(0, array.shape[0] - 1, target_size).round().astype(np.int64)
        x_idx = np.linspace(0, array.shape[1] - 1, target_size).round().astype(np.int64)
        return array[np.ix_(y_idx, x_idx)].astype(np.float32)
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None, :, :]
    mode = "nearest" if nearest else "bilinear"
    kwargs = {} if nearest else {"align_corners": False}
    resized = F.interpolate(tensor, size=(target_size, target_size), mode=mode, **kwargs)
    return resized[0, 0].cpu().numpy().astype(np.float32)


def _read_tif(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Preparing Sen1Floods11 requires rasterio. Install requirements-prithvi.txt first.") from exc
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def _prithvi_chip_from_s2(path: Path, target_size: int = TARGET_SIZE) -> np.ndarray:
    image = _read_tif(path)
    if image.shape[0] < 13:
        raise RuntimeError(f"Expected a 13-band Sen1Floods11 S2 GeoTIFF, got shape {image.shape} for {path}.")
    bands = [_resize_2d(image[index], target_size) for index in PRITHVI_BAND_INDICES]
    single_frame = np.stack(bands).astype(np.float32)
    return np.repeat(single_frame[None, :, :, :], 4, axis=0)


def _mask_from_qc(path: Path, target_size: int = TARGET_SIZE) -> np.ndarray:
    mask = _read_tif(path)[0]
    return _resize_2d(mask, target_size, nearest=True).astype(np.int16)


def _water_label(mask: np.ndarray) -> tuple[int, float]:
    valid = mask >= 0
    if not np.any(valid):
        return 0, 0.0
    fraction = float(np.mean(mask[valid] == 1))
    return int(fraction >= WATER_PRESENT_THRESHOLD), fraction


def _sample_id(path: Path) -> str:
    name = path.name
    for suffix in ["_S2Hand.tif", "_S2.tif", "_S2Weak.tif"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _event_from_sample(sample_id: str) -> str:
    return sample_id.split("_", 1)[0] if "_" in sample_id else "to_verify"


def _list_gcs_s2_files(gcs_root: str) -> list[str]:
    patterns = [
        f"{gcs_root}/**/*S2Hand.tif",
        f"{gcs_root}/**/*_S2.tif",
    ]
    found: list[str] = []
    for pattern in patterns:
        try:
            output = subprocess.check_output(["gsutil", "ls", "-r", pattern], text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        found.extend(line.strip() for line in output.splitlines() if line.strip().endswith(".tif"))
    unique = sorted(dict.fromkeys(found))
    if not unique:
        raise RuntimeError(
            "Could not list Sen1Floods11 S2 files from GCS. Install gsutil in Colab or pass --source-root with local files."
        )
    return unique


def _download_pair(s2_uri: str, cache_dir: Path) -> tuple[Path, Path]:
    sample = _sample_id(Path(s2_uri))
    qc_uri = s2_uri.replace("S2Hand.tif", "QC.tif").replace("_S2.tif", "_QC.tif")
    s2_path = cache_dir / "raw" / Path(s2_uri).name
    qc_path = cache_dir / "raw" / Path(qc_uri).name
    s2_path.parent.mkdir(parents=True, exist_ok=True)
    for uri, path in [(s2_uri, s2_path), (qc_uri, qc_path)]:
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"[info] Downloading {uri}")
        subprocess.check_call(["gsutil", "cp", uri, str(path)])
    if not qc_path.exists():
        raise FileNotFoundError(f"Missing QC mask for {sample}: {qc_uri}")
    return s2_path, qc_path


def _local_pairs(source_root: Path) -> list[tuple[Path, Path]]:
    s2_files = sorted(source_root.rglob("*S2Hand.tif")) + sorted(source_root.rglob("*_S2.tif"))
    pairs = []
    for s2_path in s2_files:
        qc_name = s2_path.name.replace("S2Hand.tif", "QC.tif").replace("_S2.tif", "_QC.tif")
        qc_path = s2_path.with_name(qc_name)
        if qc_path.exists():
            pairs.append((s2_path, qc_path))
    if not pairs:
        raise RuntimeError(f"No local Sen1Floods11 S2/QC pairs found under {source_root}.")
    return pairs


def prepare_sen1floods11_subset(
    output_dir: Path,
    max_samples: int = 64,
    source_root: Path | None = None,
    cache_dir: Path = Path("data/_cache/sen1floods11"),
    gcs_root: str = GCS_ROOT,
    target_size: int = TARGET_SIZE,
) -> Path:
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if source_root is not None:
        pairs = _local_pairs(source_root)[:max_samples]
    else:
        uris = _list_gcs_s2_files(gcs_root)[:max_samples]
        pairs = [_download_pair(uri, cache_dir) for uri in uris]

    output_dir.mkdir(parents=True, exist_ok=True)
    chip_dir = output_dir / "chips"
    mask_dir = output_dir / "masks"
    chip_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (s2_path, qc_path) in enumerate(pairs, start=1):
        sample_id = _sample_id(s2_path)
        chip = _prithvi_chip_from_s2(s2_path, target_size=target_size)
        mask = _mask_from_qc(qc_path, target_size=target_size)
        if chip.shape != (4, 6, target_size, target_size):
            raise RuntimeError(f"Prepared Prithvi chip has wrong shape for {sample_id}: {chip.shape}.")
        if mask.shape != (target_size, target_size):
            raise RuntimeError(f"Prepared QC mask has wrong shape for {sample_id}: {mask.shape}.")
        label, water_fraction = _water_label(mask)
        chip_path = chip_dir / f"{sample_id}_prithvi_s2.npz"
        mask_path = mask_dir / f"{sample_id}_qc.npz"
        np.savez_compressed(chip_path, image=chip)
        np.savez_compressed(mask_path, mask=mask)
        event = _event_from_sample(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "chip_path": str(chip_path.relative_to(output_dir)),
                "mask_path": str(mask_path.relative_to(output_dir)),
                "label": label,
                "water_fraction": water_fraction,
                "region": event,
                "event": event,
                "fallback_group": event,
                "sensor": "S2",
                "source_dataset": "sen1floods11",
                "band_profile": "prithvi_hls_6band_4frame_compat",
                "split": "all",
            }
        )
        if index % 16 == 0 or index == len(pairs):
            print(f"[info] Prepared {index}/{len(pairs)} Sen1Floods11 samples")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] Wrote {metadata_path} with {len(rows)} samples")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Prithvi-compatible Sen1Floods11 S2/QC subset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sen1floods11_prithvi_subset64"))
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--source-root", type=Path, help="Optional local root containing S2/QC GeoTIFF pairs.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/_cache/sen1floods11"))
    parser.add_argument("--gcs-root", default=GCS_ROOT)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    args = parser.parse_args()
    prepare_sen1floods11_subset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        source_root=args.source_root,
        cache_dir=args.cache_dir,
        gcs_root=args.gcs_root,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()
