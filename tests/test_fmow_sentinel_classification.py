from __future__ import annotations

import gc
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from rsfm_fairness_audit.cli import main
from rsfm_fairness_audit.adapters.dofa import DOFAAdapter
from rsfm_fairness_audit.fmow_sentinel_classification import (
    FmowClassificationConfig,
    compare_fmow_runs,
    load_fmow_sentinel_image,
    run_fmow_geography_bwer,
    run_fmow_sentinel_classification,
    FmowBwerConfig,
)
from rsfm_fairness_audit.io import read_csv_rows, write_csv


def _cleanup(path: Path) -> None:
    gc.collect()
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.1)
    shutil.rmtree(path, ignore_errors=True)


def _write_fmow_fixture(root: Path, n_train_per_class: int = 12, n_val_per_class_country: int = 10) -> Path:
    chips = root / "chips"
    chips.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    index = 0
    for split, repeats in [("train", n_train_per_class), ("val", n_val_per_class_country)]:
        countries = ["CountryA", "CountryB"] if split == "val" else ["CountryA"]
        for country in countries:
            for category, value in [("airport", 10.0), ("port", 100.0)]:
                for local in range(repeats):
                    image_id = f"{split}_{country}_{category}_{local}"
                    chip = np.full((13, 8, 8), value, dtype=np.float32)
                    chip += np.arange(13, dtype=np.float32)[:, None, None]
                    np.save(chips / f"{image_id}.npy", chip)
                    rows.append(
                        {
                            "sample_id": image_id,
                            "image_id": image_id,
                            "image_path": f"chips/{image_id}.npy",
                            "category": category,
                            "split": split,
                            "timestamp": "2020-01-15",
                            "location_id": f"{country}_{local}",
                            "latitude": "42.0" if country == "CountryA" else "-12.0",
                            "longitude": "8.0",
                            "country": country,
                            "continent": "ContinentA" if country == "CountryA" else "ContinentB",
                            "un_region": "RegionA" if country == "CountryA" else "RegionB",
                            "region": "SubregionA" if country == "CountryA" else "SubregionB",
                            "latitude_band": "north_mid_latitude" if country == "CountryA" else "south_tropics",
                            "metadata_provenance": "location_level_geography_enrichment",
                        }
                    )
                    index += 1
    metadata = root / "metadata.csv"
    write_csv(metadata, rows)
    return metadata


def test_fmow_13band_loader_accepts_channels_first_npy() -> None:
    root = Path("outputs") / f"test_fmow_loader_{uuid.uuid4().hex}"
    chips = root / "chips"
    chips.mkdir(parents=True)
    np.save(chips / "img.npy", np.ones((13, 8, 8), dtype=np.float32))
    chip = load_fmow_sentinel_image({"image_path": "chips/img.npy"}, data_root=root, image_size=4)
    assert chip.shape == (13, 4, 4)
    _cleanup(root)


def test_dofa_fmow_config_is_13band_and_download_is_explicit() -> None:
    adapter = DOFAAdapter.from_config_file("configs/models/dofa_fmow_sentinel.yaml")
    assert adapter.band_profile == "sentinel2_13band_fmow"
    assert adapter.expected_bands == 13
    assert adapter.wavelengths is not None
    assert len(adapter.wavelengths) == 13
    assert adapter.image_size == 224
    assert adapter.allow_torch_hub_download is False


def test_fmow_supervised_stats_run_writes_bwer_compatible_outputs() -> None:
    root = Path("outputs") / f"test_fmow_cls_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    metadata = _write_fmow_fixture(root)
    out = root / "run"
    artifacts = run_fmow_sentinel_classification(
        FmowClassificationConfig(
            metadata_csv=metadata,
            data_root=root,
            output_dir=out,
            image_size=8,
            run_bwer=True,
        )
    )
    assert artifacts["predictions"].exists()
    assert artifacts["audit_table"].exists()
    assert (out / "bwer" / "bwer_summary.csv").exists()
    audit = read_csv_rows(out / "audit_table.csv")
    assert audit[0]["dataset"] == "fmow_sentinel"
    assert audit[0]["input_mode"] == "s2_13band_image_only"
    assert audit[0]["adaptation_protocol"] == "supervised_baseline"
    assert audit[0]["split_protocol"] == "official_split"
    assert audit[0]["country"] in {"CountryA", "CountryB"}
    bwer = read_csv_rows(out / "bwer" / "bwer_summary.csv")
    assert any(row["slice_variable"] == "country" for row in bwer)
    _cleanup(root)


def test_fmow_geography_bwer_cli_and_compare_two_runs(monkeypatch) -> None:
    root = Path("outputs") / f"test_fmow_cli_compare_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    metadata = _write_fmow_fixture(root)
    out = root / "run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "run-fmow-sentinel-classification",
            "--metadata-csv",
            str(metadata),
            "--data-root",
            str(root),
            "--output-dir",
            str(out),
            "--image-size",
            "8",
        ],
    )
    main()
    run_fmow_geography_bwer(FmowBwerConfig(input_dir=out, output_dir=out / "bwer"))
    other = root / "run_copy"
    shutil.copytree(out, other)
    artifacts = compare_fmow_runs({"supervised": out, "copy": other}, root / "comparison")
    assert artifacts["comparison_summary"].exists()
    summary = read_csv_rows(root / "comparison" / "comparison_summary.csv")
    assert len(summary) == 2
    assert "raw_bwer_country" in summary[0]
    _cleanup(root)


def test_fmow_max_samples_is_applied_after_train_eval_split() -> None:
    root = Path("outputs") / f"test_fmow_max_samples_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    metadata = _write_fmow_fixture(root, n_train_per_class=4, n_val_per_class_country=4)
    out = root / "run"
    run_fmow_sentinel_classification(
        FmowClassificationConfig(
            metadata_csv=metadata,
            data_root=root,
            output_dir=out,
            image_size=8,
            max_samples=2,
        )
    )
    run_metadata = (out / "run_metadata.json").read_text(encoding="utf-8")
    assert '"train_rows_readable": 2' in run_metadata
    assert '"eval_rows_readable": 2' in run_metadata
    _cleanup(root)
