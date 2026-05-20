from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.cli import main
from rsfm_fairness_audit.fmow_sentinel_preflight import (
    FmowPreflightConfig,
    derive_latitude_band,
    derive_season,
    run_fmow_sentinel_preflight,
)
from rsfm_fairness_audit.io import read_csv_rows, write_csv


def _write_metadata(path: Path) -> None:
    rows = [
        {
            "image_id": "img001",
            "category": "airport",
            "location_id": "loc_a",
            "image_path": "chips/img001.tif",
            "split": "train",
            "timestamp": "2020-01-15T10:00:00",
            "latitude": "42.1",
            "longitude": "-71.2",
        },
        {
            "image_id": "img002",
            "category": "airport",
            "location_id": "loc_b",
            "image_path": "chips/img002.tif",
            "split": "train",
            "timestamp": "2020-07-02",
            "latitude": "-12.5",
            "longitude": "34.1",
        },
        {
            "image_id": "img003",
            "category": "port",
            "location_id": "loc_a",
            "image_path": "chips/img003.tif",
            "split": "val",
            "timestamp": "2021-10-11",
            "latitude": "64.0",
            "longitude": "12.2",
        },
        {
            "image_id": "img004",
            "category": "port",
            "location_id": "loc_c",
            "image_path": "chips/img004.tif",
            "split": "val",
            "timestamp": "",
            "latitude": "",
            "longitude": "",
        },
    ]
    write_csv(path, rows)


def test_fmow_derivations_are_stable() -> None:
    assert derive_season(1) == "DJF"
    assert derive_season(7) == "JJA"
    assert derive_latitude_band(42.0) == "north_mid_latitude"
    assert derive_latitude_band(-12.5) == "south_tropics"
    assert derive_latitude_band(64.0) == "north_high_latitude"


def test_fmow_preflight_metadata_only_outputs_schema_and_subset() -> None:
    root = Path("outputs") / f"test_fmow_preflight_{uuid.uuid4().hex}"
    metadata = root / "metadata.csv"
    root.mkdir(parents=True)
    _write_metadata(metadata)
    out = root / "preflight"
    artifacts = run_fmow_sentinel_preflight(
        FmowPreflightConfig(
            metadata_csvs=(metadata,),
            output_dir=out,
            metadata_only=True,
            subset_max_per_split=2,
            seed=7,
            min_support=1,
        )
    )
    for key in [
        "metadata_inventory",
        "slice_support_summary",
        "subset_manifest",
        "audit_table_schema",
        "warnings",
        "run_metadata",
    ]:
        assert artifacts[key].exists(), key
    inventory = read_csv_rows(out / "fmow_metadata_inventory.csv")
    country = next(row for row in inventory if row["canonical_field"] == "country")
    assert country["status"] == "missing"
    subset = read_csv_rows(out / "subset_manifest.csv")
    assert len(subset) == 4
    assert subset[0]["dataset"] == "fmow_sentinel"
    assert subset[0]["task"] == "scene_classification"
    assert "latitude_band" in subset[0]
    schema = (out / "audit_table_schema_fmow_sentinel.md").read_text(encoding="utf-8")
    assert "s2_13band_image_only" in schema
    report = (out / "fmow_preflight_report.md").read_text(encoding="utf-8")
    assert "Slice Recommendations" in report
    shutil.rmtree(root, ignore_errors=True)


def test_fmow_preflight_subset_is_deterministic() -> None:
    root = Path("outputs") / f"test_fmow_subset_seed_{uuid.uuid4().hex}"
    metadata = root / "metadata.csv"
    root.mkdir(parents=True)
    _write_metadata(metadata)
    out1 = root / "run1"
    out2 = root / "run2"
    config = dict(metadata_csvs=(metadata,), metadata_only=True, subset_max_per_split=1, seed=11, min_support=1)
    run_fmow_sentinel_preflight(FmowPreflightConfig(output_dir=out1, **config))
    run_fmow_sentinel_preflight(FmowPreflightConfig(output_dir=out2, **config))
    subset1 = read_csv_rows(out1 / "subset_manifest.csv")
    subset2 = read_csv_rows(out2 / "subset_manifest.csv")
    assert [row["image_id"] for row in subset1] == [row["image_id"] for row in subset2]
    shutil.rmtree(root, ignore_errors=True)


def test_fmow_preflight_cli_metadata_only(monkeypatch) -> None:
    root = Path("outputs") / f"test_fmow_cli_{uuid.uuid4().hex}"
    metadata = root / "metadata.csv"
    root.mkdir(parents=True)
    _write_metadata(metadata)
    out = root / "preflight"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "preflight-fmow-sentinel",
            "--metadata-csv",
            str(metadata),
            "--output-dir",
            str(out),
            "--metadata-only",
            "--subset-max-per-split",
            "2",
            "--seed",
            "5",
        ],
    )
    main()
    assert (out / "fmow_metadata_inventory.csv").exists()
    assert (out / "subset_manifest.csv").exists()
    shutil.rmtree(root, ignore_errors=True)


def test_fmow_raster_inspection_with_synthetic_13band_tif_if_available() -> None:
    tifffile = pytest.importorskip("tifffile")
    root = Path("outputs") / f"test_fmow_raster_{uuid.uuid4().hex}"
    chips = root / "chips"
    chips.mkdir(parents=True)
    metadata = root / "metadata.csv"
    image = np.stack([np.full((8, 8), band, dtype=np.uint16) for band in range(13)], axis=0)
    tifffile.imwrite(chips / "img001.tif", image)
    write_csv(
        metadata,
        [
            {
                "image_id": "img001",
                "category": "airport",
                "location_id": "loc_a",
                "image_path": "chips/img001.tif",
                "split": "train",
                "timestamp": "2020-01-15",
                "latitude": "42.1",
                "longitude": "-71.2",
            }
        ],
    )
    out = root / "preflight"
    run_fmow_sentinel_preflight(
        FmowPreflightConfig(
            metadata_csvs=(metadata,),
            output_dir=out,
            data_root=root,
            inspect_rasters=True,
            raster_sample_size=1,
            subset_max_per_split=1,
            min_support=1,
        )
    )
    stats = read_csv_rows(out / "band_statistics_sample.csv")
    assert len(stats) == 13
    shapes = read_csv_rows(out / "image_shape_summary.csv")
    assert any(row["shape"] == "bands=13" for row in shapes)
    shutil.rmtree(root, ignore_errors=True)
