from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from scripts.analyze_fmow_patch_size_diagnostics import analyze_manifest


def test_fmow_patch_size_diagnostics_reports_category_variation() -> None:
    tifffile = pytest.importorskip("tifffile")
    root = Path("outputs") / f"test_fmow_patch_sizes_{uuid.uuid4().hex}"
    data_root = root / "data"
    try:
        airport = data_root / "fmow-sentinel/train/airport/airport_1/airport_1_1.tif"
        solar = data_root / "fmow-sentinel/val/solar_farm/solar_farm_2/solar_farm_2_1.tif"
        airport.parent.mkdir(parents=True)
        solar.parent.mkdir(parents=True)
        tifffile.imwrite(airport, np.ones((13, 50, 40), dtype=np.uint16))
        tifffile.imwrite(solar, np.ones((13, 10, 8), dtype=np.uint16))
        manifest = root / "manifest.csv"
        write_csv(
            manifest,
            [
                {
                    "sample_id": "a",
                    "image_id": "1",
                    "image_path": "fmow-sentinel/train/airport/airport_1/airport_1_1.tif",
                    "category": "airport",
                    "split": "train",
                    "country": "AAA",
                    "region": "RegionA",
                },
                {
                    "sample_id": "s",
                    "image_id": "2",
                    "image_path": "fmow-sentinel/val/solar_farm/solar_farm_2/solar_farm_2_1.tif",
                    "category": "solar_farm",
                    "split": "val",
                    "country": "BBB",
                    "region": "RegionB",
                },
            ],
        )
        out = root / "out"
        analyze_manifest(manifest, data_root, out, progress_every=0)
        per_sample = read_csv_rows(out / "patch_size_per_sample.csv")
        by_category = {row["category"]: row for row in read_csv_rows(out / "patch_size_by_category.csv")}
        assert len(per_sample) == 2
        assert by_category["airport"]["area_median"] == "2000.0"
        assert by_category["solar_farm"]["area_median"] == "80.0"
        report = (out / "patch_size_diagnostic_report.md").read_text(encoding="utf-8")
        assert "not a model experiment" in report
        assert "Resize" not in report
        assert "Resizing normalizes" in report
    finally:
        shutil.rmtree(root, ignore_errors=True)
