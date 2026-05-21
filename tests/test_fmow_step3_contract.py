from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from rsfm_fairness_audit.cli import main
from rsfm_fairness_audit.fmow_step3_contract import (
    FmowStep3PackageConfig,
    FmowStep3ValidationConfig,
    package_fmow_step3_handoff,
    validate_fmow_step3_results,
)
from rsfm_fairness_audit.io import read_csv_rows, write_csv


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _write_valid_run(root: Path) -> Path:
    run = root / "fmow_step3_fixture"
    bwer = run / "bwer"
    data = run / "data"
    bwer.mkdir(parents=True)
    data.mkdir(parents=True)
    prediction_rows = [
        {
            "sample_id": "s1",
            "image_id": "i1",
            "image_path": "fmow-sentinel/val/airport/airport_1/airport_1_i1.tif",
            "dataset": "fmow_sentinel",
            "task": "scene_classification",
            "split": "val",
            "label": "airport",
            "category": "airport",
            "prediction": "airport",
            "correct": 1.0,
            "risk": 0.0,
            "model_family": "supervised_stats",
            "model_variant": "band_stats_nearest_centroid",
            "input_mode": "s2_13band_image_only",
            "adaptation_protocol": "supervised_baseline",
            "split_protocol": "official_split",
            "eval_scope": "val",
            "resolution": 96,
            "band_profile": "sentinel2_13band_fmow",
            "timestamp": "2020-01-01",
            "year": 2020,
            "month": 1,
            "season": "DJF",
            "location_id": "loc1",
            "latitude": 10.0,
            "longitude": 20.0,
            "country": "CountryA",
            "continent": "ContinentA",
            "un_region": "RegionA",
            "region": "SubregionA",
            "latitude_band": "north_tropics",
        },
        {
            "sample_id": "s2",
            "image_id": "i2",
            "image_path": "fmow-sentinel/val/port/port_2/port_2_i2.tif",
            "dataset": "fmow_sentinel",
            "task": "scene_classification",
            "split": "val",
            "label": "port",
            "category": "port",
            "prediction": "airport",
            "correct": 0.0,
            "risk": 1.0,
            "model_family": "supervised_stats",
            "model_variant": "band_stats_nearest_centroid",
            "input_mode": "s2_13band_image_only",
            "adaptation_protocol": "supervised_baseline",
            "split_protocol": "official_split",
            "eval_scope": "val",
            "resolution": 96,
            "band_profile": "sentinel2_13band_fmow",
            "timestamp": "2020-07-01",
            "year": 2020,
            "month": 7,
            "season": "JJA",
            "location_id": "loc2",
            "latitude": -10.0,
            "longitude": 30.0,
            "country": "CountryB",
            "continent": "ContinentB",
            "un_region": "RegionB",
            "region": "SubregionB",
            "latitude_band": "south_tropics",
        },
    ]
    write_csv(run / "predictions.csv", prediction_rows)
    audit_rows = [dict(row, unit_id=row["sample_id"], y_true=row["label"], y_pred=row["prediction"], score=row["correct"]) for row in prediction_rows]
    write_csv(run / "audit_table.csv", audit_rows)
    write_csv(data / "clean_subset_manifest.csv", prediction_rows)
    write_csv(
        bwer / "bwer_summary.csv",
        [
            {
                "dataset": "fmow_sentinel",
                "model": "supervised_stats_fmow_sentinel",
                "task": "scene_classification",
                "slice_variable": "country",
                "balance_variable": "",
                "bwer": 0.2,
                "n_slices_valid": 2,
                "tail_slices": "CountryB",
            },
            {
                "dataset": "fmow_sentinel",
                "model": "supervised_stats_fmow_sentinel",
                "task": "scene_classification",
                "slice_variable": "country",
                "balance_variable": "class_label",
                "bwer": 0.1,
                "n_slices_valid": 2,
                "tail_slices": "CountryB",
            },
        ],
    )
    write_csv(
        bwer / "bwer_by_slice.csv",
        [
            {"slice_variable": "country", "slice_value": "CountryA", "balanced_risk": 0.0},
            {"slice_variable": "country", "slice_value": "CountryB", "balanced_risk": 1.0},
        ],
    )
    (bwer / "warnings.json").write_text(json.dumps({"warnings": []}), encoding="utf-8")
    (run / "run_metadata.json").write_text(
        json.dumps({"model": "supervised_stats_fmow_sentinel", "split_protocol": "official_split", "seed": 42}),
        encoding="utf-8",
    )
    (run / "dummy.tif").write_bytes(b"not packaged by default")
    return run


def test_fmow_step3_validation_and_handoff_package() -> None:
    root = Path("outputs") / f"test_fmow_step3_contract_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    run = _write_valid_run(root)
    artifacts = validate_fmow_step3_results(FmowStep3ValidationConfig(run_dir=run, full_archive_downloaded_locally=True))
    assert artifacts["prediction_table_validation_json"].exists()
    assert artifacts["bwer_output_validation_json"].exists()
    assert artifacts["archive_manifest"].exists()
    assert artifacts["handoff_checklist"].exists()
    validation = json.loads((run / "prediction_table_validation.json").read_text(encoding="utf-8"))
    assert validation["passed"] is True
    packaged = package_fmow_step3_handoff(FmowStep3PackageConfig(run_dir=run))
    assert packaged["handoff_zip"].exists()
    import zipfile

    with zipfile.ZipFile(packaged["handoff_zip"]) as zf:
        names = zf.namelist()
    assert any(name.endswith("predictions.csv") for name in names)
    assert not any(name.endswith("dummy.tif") for name in names)
    _cleanup(root)


def test_fmow_step3_validator_detects_missing_columns_and_protocols() -> None:
    root = Path("outputs") / f"test_fmow_step3_bad_{uuid.uuid4().hex}"
    run = root / "run"
    bwer = run / "bwer"
    bwer.mkdir(parents=True)
    write_csv(run / "predictions.csv", [{"sample_id": "s1", "dataset": "wrong", "task": "scene_classification"}])
    write_csv(bwer / "bwer_summary.csv", [{"slice_variable": "country", "balance_variable": "country", "bwer": 0.0}])
    write_csv(bwer / "bwer_by_slice.csv", [{"slice_variable": "country", "slice_value": "A"}])
    validate_fmow_step3_results(FmowStep3ValidationConfig(run_dir=run))
    prediction = json.loads((run / "prediction_table_validation.json").read_text(encoding="utf-8"))
    bwer_validation = json.loads((run / "bwer_output_validation.json").read_text(encoding="utf-8"))
    assert prediction["passed"] is False
    assert bwer_validation["passed"] is False
    assert prediction["missing_required_columns"]
    assert bwer_validation["invalid_balance_rows"]
    _cleanup(root)


def test_fmow_step3_validation_cli(monkeypatch) -> None:
    root = Path("outputs") / f"test_fmow_step3_cli_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    run = _write_valid_run(root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "validate-fmow-step3-results",
            "--run-dir",
            str(run),
            "--full-archive-downloaded-locally",
            "true",
        ],
    )
    main()
    assert (run / "handoff_checklist.md").exists()
    _cleanup(root)
