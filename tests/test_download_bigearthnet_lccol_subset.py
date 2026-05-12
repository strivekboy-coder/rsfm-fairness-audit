from __future__ import annotations

import csv
import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.adapters.bigearthnet import BigEarthNetDatasetAdapter

h5py = pytest.importorskip("h5py")


def _load_lccol_module():
    script_path = Path("scripts/download_bigearthnet_lccol_subset.py")
    spec = importlib.util.spec_from_file_location("download_bigearthnet_lccol_subset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tiny_lccol_files(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "bigearthnet_hdf5_train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["s2_folder", "s2_hdf5_file", "index"])
        writer.writeheader()
        for index in range(8):
            writer.writerow({"s2_folder": f"BEN-TINY-{index:03d}", "s2_hdf5_file": "bigearthnet_train_p0.hdf5", "index": index})

    hdf5_path = root / "bigearthnet_train_p0.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("images", data=np.ones((8, 9, 6, 6), dtype=np.float32))
        labels = np.zeros((8, 3), dtype=np.int64)
        labels[np.arange(8), np.arange(8) % 3] = 1
        handle.create_dataset("labels", data=labels)

    gz_path = root / "bigearthnet_train_p0.hdf5.gz"
    with hdf5_path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())
    return csv_path, gz_path


def test_lccol_converter_uses_real_hdf5_arrays(monkeypatch) -> None:
    module = _load_lccol_module()
    source = Path("outputs/test_lccol_source")
    csv_path, gz_path = _write_tiny_lccol_files(source)

    def fake_download(filename: str, cache_dir: Path) -> Path:
        return csv_path if filename.endswith(".csv") else gz_path

    monkeypatch.setattr(module, "_download_file", fake_download)
    output = Path("outputs/test_lccol_subset")

    metadata_path = module.convert_lccol_subset(output, max_samples=4, cache_dir=output / "_cache")

    rows = list(csv.DictReader(metadata_path.open(newline="", encoding="utf-8")))
    assert len(rows) == 4
    assert rows[0]["source_dataset"] == "lc-col/bigearthnet"
    assert rows[0]["source_shard"] == "bigearthnet_train_p0.hdf5"
    assert rows[0]["band_profile"] == "sentinel2_12_lccol"
    assert (output / rows[0]["chip_path"]).exists()

    adapter = BigEarthNetDatasetAdapter(output, subset_size=2, sensor_mode="S2")
    assert adapter.load_sample(0)["image"].shape == (9, 6, 6)


def test_lccol_converter_reuses_decompressed_cache(monkeypatch) -> None:
    module = _load_lccol_module()
    source = Path("outputs/test_lccol_cache_source")
    csv_path, gz_path = _write_tiny_lccol_files(source)
    cache = Path("outputs/test_lccol_cache")
    cache.mkdir(parents=True, exist_ok=True)
    cached_hdf5 = cache / "bigearthnet_train_p0.hdf5"
    with gzip.open(gz_path, "rb") as src, cached_hdf5.open("wb") as dst:
        dst.write(src.read())

    calls = []

    def fake_download(filename: str, cache_dir: Path) -> Path:
        calls.append(filename)
        return csv_path if filename.endswith(".csv") else gz_path

    monkeypatch.setattr(module, "_download_file", fake_download)
    module.convert_lccol_subset(Path("outputs/test_lccol_cache_subset"), max_samples=2, cache_dir=cache)

    assert module.TRAIN_CSV in calls
    assert module.TRAIN_SHARD_GZ not in calls


def test_lccol_download_caches_csv_to_stable_path(monkeypatch) -> None:
    pytest.importorskip("huggingface_hub")
    module = _load_lccol_module()
    root = Path("outputs/test_lccol_download_cache_csv")
    hub_cache_file = root / "hub" / "snapshots" / "abc" / module.TRAIN_CSV
    hub_cache_file.parent.mkdir(parents=True, exist_ok=True)
    hub_cache_file.write_text("s2_folder,s2_hdf5_file,index\n", encoding="utf-8")

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: str(hub_cache_file),
    )

    cache_dir = root / "cache"
    path = module._download_file(module.TRAIN_CSV, cache_dir)

    assert path == cache_dir / module.TRAIN_CSV
    assert path.read_text(encoding="utf-8").startswith("s2_folder")


def test_lccol_to_chw_handles_64_pixel_hwc_chips() -> None:
    module = _load_lccol_module()
    image = np.zeros((64, 64, 12), dtype=np.float32)

    converted = module._to_chw(image)

    assert converted.shape == (12, 64, 64)


def test_lccol_converter_fails_without_image_arrays(monkeypatch) -> None:
    module = _load_lccol_module()
    source = Path("outputs/test_lccol_no_image_source")
    source.mkdir(parents=True, exist_ok=True)
    csv_path = source / "bigearthnet_hdf5_train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["s2_folder", "s2_hdf5_file", "index"])
        writer.writeheader()
        writer.writerow({"s2_folder": "BEN-TINY-000", "s2_hdf5_file": "bigearthnet_train_p0.hdf5", "index": 0})
    hdf5_path = source / "bigearthnet_train_p0.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("labels", data=np.ones((1, 3), dtype=np.int64))
    gz_path = source / "bigearthnet_train_p0.hdf5.gz"
    with hdf5_path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())

    monkeypatch.setattr(module, "_download_file", lambda filename, cache_dir: csv_path if filename.endswith(".csv") else gz_path)

    try:
        module.convert_lccol_subset(Path("outputs/test_lccol_no_image_subset"), max_samples=1)
    except RuntimeError as exc:
        assert "Could not find real image arrays" in str(exc)
    else:
        raise AssertionError("Expected converter to fail when HDF5 contains no image arrays.")
