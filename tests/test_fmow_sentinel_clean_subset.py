from __future__ import annotations

import shutil
import tarfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.fmow_sentinel_classification import load_fmow_sentinel_image
from scripts.prepare_fmow_sentinel_clean_subset import archive_path_for_row, build_parser, prepare_subset


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def test_fmow_archive_path_generation_matches_official_rule() -> None:
    row = {"split": "train", "category": "airport", "location_id": "0001", "image_id": "0002"}
    assert archive_path_for_row(row) == "fmow-sentinel/train/airport/airport_0001/airport_0001_0002.tif"


def test_prepare_fmow_clean_subset_extracts_and_validates_selected_tifs() -> None:
    tifffile = pytest.importorskip("tifffile")
    root = Path("outputs") / f"test_fmow_clean_subset_{uuid.uuid4().hex}"
    source = root / "source"
    source.mkdir(parents=True)
    valid_rel = Path("fmow-sentinel/train/airport/airport_0001/airport_0001_0002.tif")
    invalid_rel = Path("fmow-sentinel/val/port/port_0003/port_0003_0004.tif")
    (source / valid_rel).parent.mkdir(parents=True)
    (source / invalid_rel).parent.mkdir(parents=True)
    tifffile.imwrite(source / valid_rel, np.ones((13, 4, 4), dtype=np.uint16))
    tifffile.imwrite(source / invalid_rel, np.ones((3, 4, 4), dtype=np.uint16))
    archive = root / "fmow-sentinel.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source / "fmow-sentinel", arcname="fmow-sentinel")
    metadata = root / "metadata.csv"
    write_csv(
        metadata,
        [
            {
                "sample_id": "a",
                "split": "train",
                "category": "airport",
                "location_id": "0001",
                "image_id": "0002",
                "country": "CountryA",
                "region": "RegionA",
                "un_region": "UN_A",
                "continent": "ContinentA",
                "season": "DJF",
                "latitude_band": "north_mid_latitude",
            },
            {
                "sample_id": "b",
                "split": "val",
                "category": "port",
                "location_id": "0003",
                "image_id": "0004",
                "country": "CountryB",
                "region": "RegionB",
                "un_region": "UN_B",
                "continent": "ContinentB",
                "season": "JJA",
                "latitude_band": "south_tropics",
            },
        ],
    )
    out = root / "prepared"
    args = build_parser().parse_args(
        [
            "--archive",
            str(archive),
            "--metadata-csv",
            str(metadata),
            "--output-dir",
            str(out),
            "--max-samples-per-split",
            "2",
            "--stratify-field",
            "country",
        ]
    )
    prepare_subset(args)
    clean = read_csv_rows(out / "clean_subset_manifest.csv")
    validation = read_csv_rows(out / "raster_validation_report.csv")
    assert len(clean) == 1
    assert clean[0]["archive_path"] == valid_rel.as_posix()
    loaded = load_fmow_sentinel_image(clean[0], data_root=out, image_size=4)
    assert loaded.shape == (13, 4, 4)
    assert len(validation) == 2
    assert any(row["valid"] == "False" and "expected 13 bands" in row["reason"] for row in validation)
    assert (out / "include_list.txt").read_text(encoding="utf-8").count("fmow-sentinel/") == 2
    _cleanup(root)
