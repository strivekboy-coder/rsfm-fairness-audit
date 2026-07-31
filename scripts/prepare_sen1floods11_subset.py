from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rsfm_fairness_audit.sen1_prithvi_mask_gate import (
    validate_prepared_mask,
    validate_source_label,
    write_verified_mask_npz,
)


GCS_ROOT = "gs://sen1floods11"
PRITHVI_BAND_INDICES = [1, 2, 3, 4, 5, 6]  # B02-B07 from 13-band Sen1Floods11 S2 GeoTIFFs.
BAND_PROFILES = {
    "prithvi_hls_6band_4frame_compat": {
        "indices": [1, 2, 3, 4, 5, 6],
        "names": ["B02", "B03", "B04", "B05", "B06", "B07"],
        "frames": 4,
    },
    "prithvi_tl_sen1floods11": {
        # Official HF inference default input_indices=[1,2,3,8,11,12] for S2L1C.
        "indices": [1, 2, 3, 8, 11, 12],
        "names": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        "frames": 1,
    },
}
TARGET_SIZE = 224
WATER_PRESENT_THRESHOLD = 0.01


def _log(message: str) -> None:
    print(message, flush=True)


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


def _prithvi_chip_from_s2(
    path: Path,
    target_size: int = TARGET_SIZE,
    band_indices: list[int] | None = None,
    frames: int = 4,
) -> np.ndarray:
    image = _read_tif(path)
    if image.shape[0] < 13:
        raise RuntimeError(f"Expected a 13-band Sen1Floods11 S2 GeoTIFF, got shape {image.shape} for {path}.")
    indices = band_indices or PRITHVI_BAND_INDICES
    bands = [_resize_2d(image[index], target_size) for index in indices]
    single_frame = np.stack(bands).astype(np.float32)
    return np.repeat(single_frame[None, :, :, :], frames, axis=0)


def _mask_from_label(path: Path, target_size: int = TARGET_SIZE) -> np.ndarray:
    source = validate_source_label(
        _read_tif(path)[0], stage=f"label_read[{path}]"
    )
    resized = _resize_2d(source, target_size, nearest=True)
    return validate_prepared_mask(
        resized,
        stage=f"nearest_resize[{path}]",
        expected_shape=(target_size, target_size),
    )


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
        f"{gcs_root}/v1.1/data/flood_events/HandLabeled/S2Hand/*_S2Hand.tif",
        f"{gcs_root}/data/flood_events/HandLabeled/S2Hand/*_S2Hand.tif",
        f"{gcs_root}/**/*S2Hand.tif",
        f"{gcs_root}/**/*_S2.tif",
    ]
    found: list[str] = []
    for pattern in patterns:
        _log(f"[stage] Listing S2 candidates: {pattern}")
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


def _list_gcs_files(patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        _log(f"[stage] Listing GCS files: {pattern}")
        try:
            command = ["gsutil", "ls", "-r", pattern] if "**" in pattern else ["gsutil", "ls", pattern]
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        found.extend(line.strip() for line in output.splitlines() if line.strip().endswith(".tif"))
    return sorted(dict.fromkeys(found))


def _manifest_label_key(uri: str) -> str:
    name = Path(uri).name
    for suffix in ["_S2Hand.tif", "_LabelHand.tif", "_QC.tif", "_S2.tif", "_Label.tif"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(uri).stem


def _manifest_cache_path(cache_dir: Path, gcs_root: str) -> Path:
    root_key = gcs_root.replace("gs://", "").replace("/", "_")
    return cache_dir / f"{root_key}_hand_labeled_manifest.csv"


def _read_pair_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _log(f"[stage] Loaded cached Sen1Floods11 pair manifest: {path} ({len(rows)} pairs)")
    return rows


def _write_pair_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "event", "s2_uri", "label_uri"])
        writer.writeheader()
        writer.writerows(rows)
    _log(f"[stage] Cached Sen1Floods11 pair manifest: {path} ({len(rows)} pairs)")


def _build_gcs_pair_manifest(gcs_root: str, cache_dir: Path, manifest_path: Path | None = None, refresh: bool = False) -> list[dict[str, str]]:
    manifest = manifest_path or _manifest_cache_path(cache_dir, gcs_root)
    if not refresh:
        cached = _read_pair_manifest(manifest)
        if cached:
            return cached
    start = time.time()
    _log("[stage] Building Sen1Floods11 hand-labeled pair manifest from bulk GCS listings")
    s2_uris = _list_gcs_files(
        [
            f"{gcs_root}/v1.1/data/flood_events/HandLabeled/S2Hand/*_S2Hand.tif",
            f"{gcs_root}/data/flood_events/HandLabeled/S2Hand/*_S2Hand.tif",
        ]
    )
    label_uris = _list_gcs_files(
        [
            f"{gcs_root}/v1.1/data/flood_events/HandLabeled/LabelHand/*_LabelHand.tif",
            f"{gcs_root}/data/flood_events/HandLabeled/LabelHand/*_LabelHand.tif",
        ]
    )
    if not s2_uris or not label_uris:
        _log("[warn] Targeted hand-labeled listing did not find both S2 and labels; falling back to recursive listing.")
        s2_uris = _list_gcs_s2_files(gcs_root)
        label_uris = _list_gcs_files([f"{gcs_root}/**/*_LabelHand.tif", f"{gcs_root}/**/*_QC.tif"])
    label_by_key = {_manifest_label_key(uri): uri for uri in label_uris}
    rows: list[dict[str, str]] = []
    missing = 0
    for s2_uri in s2_uris:
        sample_id = _sample_id(Path(s2_uri))
        label_uri = label_by_key.get(sample_id)
        if not label_uri:
            missing += 1
            continue
        rows.append({"sample_id": sample_id, "event": _event_from_sample(sample_id), "s2_uri": s2_uri, "label_uri": label_uri})
    rows = sorted(rows, key=lambda row: row["sample_id"])
    if not rows:
        raise RuntimeError("No paired Sen1Floods11 hand-labeled S2/LabelHand files were found in the GCS listings.")
    _log(f"[stage] Built manifest with {len(rows)} pairs; missing labels for {missing} S2 files; elapsed={time.time() - start:.1f}s")
    _write_pair_manifest(manifest, rows)
    return rows


def _filter_uris_by_event(uris: Iterable[str], event_filters: Iterable[str] | None) -> list[str]:
    filters = [value.lower() for value in (event_filters or []) if value]
    if not filters:
        return list(uris)
    output = []
    for uri in uris:
        event = _event_from_sample(_sample_id(Path(uri))).lower()
        if event in filters:
            output.append(uri)
    return output


def _label_uri_candidates(s2_uri: str) -> list[str]:
    """Return official v1.1 LabelHand candidates before legacy QC fallbacks."""
    candidates = []
    if "S2Hand" in s2_uri:
        candidates.append(s2_uri.replace("/S2Hand/", "/LabelHand/").replace("_S2Hand.tif", "_LabelHand.tif"))
        candidates.append(s2_uri.replace("_S2Hand.tif", "_LabelHand.tif"))
        candidates.append(s2_uri.replace("_S2Hand.tif", "_QC.tif"))
        candidates.append(s2_uri.replace("S2Hand.tif", "QC.tif"))
    if "_S2.tif" in s2_uri:
        candidates.append(s2_uri.replace("/S2/", "/Label/").replace("_S2.tif", "_Label.tif"))
        candidates.append(s2_uri.replace("_S2.tif", "_QC.tif"))
    return list(dict.fromkeys(candidates))


def _gsutil_ls_exists(uri: str) -> bool:
    try:
        subprocess.check_output(["gsutil", "ls", uri], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def _resolve_gcs_pair(s2_uri: str) -> tuple[str, str]:
    for label_uri in _label_uri_candidates(s2_uri):
        if _gsutil_ls_exists(label_uri):
            return s2_uri, label_uri
    raise FileNotFoundError(
        f"Missing hand label for {_sample_id(Path(s2_uri))}. "
        "Official v1.1 labels usually live in LabelHand/ with *_LabelHand.tif names."
    )


def _download_many(uris: list[str], raw_dir: Path) -> None:
    missing = [uri for uri in uris if not ((raw_dir / Path(uri).name).exists() and (raw_dir / Path(uri).name).stat().st_size > 0)]
    if not missing:
        _log("[info] All selected Sen1Floods11 GeoTIFFs are already cached.")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[info] Batch downloading {len(missing)} GeoTIFFs with gsutil -m cp -I")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        for uri in missing:
            handle.write(uri + "\n")
        manifest_path = Path(handle.name)
    try:
        with manifest_path.open("r", encoding="utf-8") as stdin:
            subprocess.check_call(["gsutil", "-m", "cp", "-I", str(raw_dir)], stdin=stdin)
    finally:
        manifest_path.unlink(missing_ok=True)


def _gsutil_cp_if_available(uri: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    try:
        subprocess.check_call(["gsutil", "cp", uri, str(path)])
    except (subprocess.CalledProcessError, FileNotFoundError):
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
        return False
    return path.exists() and path.stat().st_size > 0


def _download_pair(s2_uri: str, cache_dir: Path) -> tuple[Path, Path]:
    sample = _sample_id(Path(s2_uri))
    s2_path = cache_dir / "raw" / Path(s2_uri).name
    s2_path.parent.mkdir(parents=True, exist_ok=True)
    if not (s2_path.exists() and s2_path.stat().st_size > 0):
        print(f"[info] Downloading S2 {s2_uri}")
        if not _gsutil_cp_if_available(s2_uri, s2_path):
            raise FileNotFoundError(f"Could not download S2 chip: {s2_uri}")

    tried = []
    for label_uri in _label_uri_candidates(s2_uri):
        tried.append(label_uri)
        label_path = cache_dir / "raw" / Path(label_uri).name
        if label_path.exists() and label_path.stat().st_size > 0:
            return s2_path, label_path
        print(f"[info] Trying label {label_uri}")
        if _gsutil_cp_if_available(label_uri, label_path):
            return s2_path, label_path

    raise FileNotFoundError(
        f"Missing hand label for {sample}. Tried: {', '.join(tried)}. "
        "Official v1.1 labels usually live in LabelHand/ with *_LabelHand.tif names."
    )


def _select_and_download_gcs_pairs(
    uris: list[str],
    cache_dir: Path,
    max_samples: int,
    candidate_limit: int,
    parallel_download: bool = True,
) -> tuple[list[tuple[Path, Path]], int]:
    selected: list[tuple[str, str]] = []
    failures = 0
    _log(f"[stage] Resolving S2/label pairs from {min(len(uris), candidate_limit)} candidates")
    for s2_uri in uris[:candidate_limit]:
        if len(selected) >= max_samples:
            break
        try:
            selected.append(_resolve_gcs_pair(s2_uri))
        except FileNotFoundError as exc:
            failures += 1
            print(f"[warn] Skipping unavailable pair: {exc}")
    if not selected:
        return [], failures

    raw_dir = cache_dir / "raw"
    if parallel_download:
        try:
            _download_many([uri for pair in selected for uri in pair], raw_dir)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[warn] Batch download failed; falling back to per-pair download: {exc}")

    pairs: list[tuple[Path, Path]] = []
    for s2_uri, label_uri in selected:
        s2_path = raw_dir / Path(s2_uri).name
        label_path = raw_dir / Path(label_uri).name
        if s2_path.exists() and s2_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
            pairs.append((s2_path, label_path))
            continue
        try:
            pairs.append(_download_pair(s2_uri, cache_dir))
        except FileNotFoundError as exc:
            failures += 1
            print(f"[warn] Skipping failed fallback pair: {exc}")
    return pairs, failures


def _local_pairs(source_root: Path) -> list[tuple[Path, Path]]:
    s2_files = sorted(source_root.rglob("*S2Hand.tif")) + sorted(source_root.rglob("*_S2.tif"))
    pairs = []
    for s2_path in s2_files:
        label_names = [
            s2_path.name.replace("_S2Hand.tif", "_LabelHand.tif"),
            s2_path.name.replace("S2Hand.tif", "QC.tif"),
            s2_path.name.replace("_S2.tif", "_QC.tif"),
        ]
        candidate_paths = [s2_path.with_name(name) for name in label_names]
        if "S2Hand" in str(s2_path):
            candidate_paths.append(Path(str(s2_path).replace("S2Hand", "LabelHand").replace("_S2Hand.tif", "_LabelHand.tif")))
        label_path = next((path for path in candidate_paths if path.exists()), None)
        if label_path is not None:
            pairs.append((s2_path, label_path))
    if not pairs:
        raise RuntimeError(f"No local Sen1Floods11 S2/label pairs found under {source_root}.")
    return pairs


def prepare_sen1floods11_subset(
    output_dir: Path,
    max_samples: int = 64,
    source_root: Path | None = None,
    cache_dir: Path = Path("data/_cache/sen1floods11"),
    gcs_root: str = GCS_ROOT,
    target_size: int = TARGET_SIZE,
    event_filter: list[str] | None = None,
    candidate_limit: int = 1000,
    parallel_download: bool = True,
    band_profile: str = "prithvi_hls_6band_4frame_compat",
) -> Path:
    if output_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing prepared output directory: {output_dir}"
        )
    if max_samples < 0:
        raise ValueError("--max-samples must be non-negative; use 0 for all selected candidates.")
    start_time = time.time()
    if band_profile not in BAND_PROFILES:
        raise ValueError(f"Unknown band_profile={band_profile!r}. Allowed: {sorted(BAND_PROFILES)}")
    profile = BAND_PROFILES[band_profile]
    band_indices = list(profile["indices"])
    band_names = list(profile["names"])
    frames = int(profile["frames"])
    _log(
        f"[stage] Preparing Sen1Floods11 subset output_dir={output_dir} max_samples={max_samples or 'all'} "
        f"candidate_limit={candidate_limit} source_root={source_root or 'GCS'} band_profile={band_profile}"
    )
    if source_root is not None:
        local_pairs = _local_pairs(source_root)
        if event_filter:
            filters = {value.casefold() for value in event_filter}
            local_pairs = [
                pair
                for pair in local_pairs
                if _event_from_sample(_sample_id(pair[0])).casefold() in filters
            ]
            _log(
                f"[stage] Local event filters {sorted(filters)} retained "
                f"{len(local_pairs)} pairs"
            )
        pairs = local_pairs if max_samples == 0 else local_pairs[:max_samples]
    else:
        manifest_rows = _build_gcs_pair_manifest(gcs_root, cache_dir)
        if event_filter:
            filters = {value.lower() for value in event_filter}
            manifest_rows = [row for row in manifest_rows if row["event"].lower() in filters]
            _log(f"[stage] Event filters {sorted(filters)} retained {len(manifest_rows)} pairs")
        limited_rows = manifest_rows[:candidate_limit]
        selected_rows = limited_rows if max_samples == 0 else limited_rows[:max_samples]
        _log(f"[stage] Selected {len(selected_rows)} paired samples from {len(manifest_rows)} manifest pairs")
        raw_dir = cache_dir / "raw"
        if parallel_download:
            try:
                _download_many([uri for row in selected_rows for uri in [row["s2_uri"], row["label_uri"]]], raw_dir)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                _log(f"[warn] Batch download failed; falling back to per-pair download: {exc}")
        pairs = []
        failures = 0
        for row in selected_rows:
            s2_path = raw_dir / Path(row["s2_uri"]).name
            label_path = raw_dir / Path(row["label_uri"]).name
            if s2_path.exists() and s2_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                pairs.append((s2_path, label_path))
                continue
            try:
                pairs.append(_download_pair(row["s2_uri"], cache_dir))
            except FileNotFoundError as exc:
                failures += 1
                _log(f"[warn] Skipping failed fallback pair: {exc}")
        if not pairs:
            filters = ", ".join(event_filter or []) or "none"
            raise RuntimeError(
                "No valid Sen1Floods11 S2/label pairs were prepared. "
                f"Checked up to {min(len(uris), candidate_limit)} candidates with event_filter={filters}. "
                "The official bucket may be unavailable, or this event has missing labels. "
                "Try increasing --candidate-limit, using --event-filter for a different event, or passing --source-root "
                "pointing at a local rsync/HF mirror of the official files."
            )
        if failures:
            _log(f"[summary] Skipped {failures} unavailable S2/label candidates before preparing {len(pairs)} pairs.")

    output_dir.mkdir(parents=True, exist_ok=False)
    chip_dir = output_dir / "chips"
    mask_dir = output_dir / "masks"
    chip_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    _log(f"[stage] Converting {len(pairs)} GeoTIFF pairs to Prithvi-ready NPZ chips")
    for index, (s2_path, label_path) in enumerate(pairs, start=1):
        sample_id = _sample_id(s2_path)
        chip = _prithvi_chip_from_s2(s2_path, target_size=target_size, band_indices=band_indices, frames=frames)
        mask = _mask_from_label(label_path, target_size=target_size)
        if chip.shape != (frames, 6, target_size, target_size):
            raise RuntimeError(f"Prepared Prithvi chip has wrong shape for {sample_id}: {chip.shape}.")
        mask = validate_prepared_mask(
            mask,
            stage=f"pre_write[{sample_id}]",
            expected_shape=(target_size, target_size),
        )
        label, water_fraction = _water_label(mask)
        chip_path = chip_dir / f"{sample_id}_prithvi_s2.npz"
        mask_path = mask_dir / f"{sample_id}_qc.npz"
        np.savez_compressed(chip_path, image=chip)
        write_verified_mask_npz(
            mask_path,
            mask,
            expected_shape=(target_size, target_size),
        )
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
                "band_profile": band_profile,
                "band_indices": json.dumps(band_indices),
                "band_names": json.dumps(band_names),
                "split": "all",
            }
        )
        if index % 16 == 0 or index == len(pairs):
            _log(f"[info] Prepared {index}/{len(pairs)} Sen1Floods11 samples")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _log(f"[summary] Wrote {metadata_path} with {len(rows)} samples; total_elapsed={time.time() - start_time:.1f}s")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Prithvi-compatible Sen1Floods11 S2/label subset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sen1floods11_prithvi_subset64"))
    parser.add_argument("--max-samples", type=int, default=64, help="Number of pairs to prepare; use 0 for all selected candidates.")
    parser.add_argument("--source-root", type=Path, help="Optional local root containing S2/label GeoTIFF pairs.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/_cache/sen1floods11"))
    parser.add_argument("--gcs-root", default=GCS_ROOT)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--event-filter", action="append", help="Optional event/country filter, e.g. India or Pakistan. Repeatable.")
    parser.add_argument("--candidate-limit", type=int, default=1000, help="Maximum listed S2 candidates to inspect while searching for valid S2/label pairs.")
    parser.add_argument("--no-parallel-download", action="store_true", help="Disable gsutil -m cp batch download and use per-pair fallback only.")
    parser.add_argument("--refresh-manifest", action="store_true", help="Refresh cached GCS S2/LabelHand pair manifest before preparing data.")
    parser.add_argument(
        "--band-profile",
        choices=sorted(BAND_PROFILES),
        default="prithvi_hls_6band_4frame_compat",
        help="Prepared band profile. Use prithvi_tl_sen1floods11 for the official TL segmentation checkpoint.",
    )
    args = parser.parse_args()
    if args.refresh_manifest and args.source_root is None:
        _build_gcs_pair_manifest(args.gcs_root, args.cache_dir, refresh=True)
    prepare_sen1floods11_subset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        source_root=args.source_root,
        cache_dir=args.cache_dir,
        gcs_root=args.gcs_root,
        target_size=args.target_size,
        event_filter=args.event_filter,
        candidate_limit=args.candidate_limit,
        parallel_download=not args.no_parallel_download,
        band_profile=args.band_profile,
    )


if __name__ == "__main__":
    main()
