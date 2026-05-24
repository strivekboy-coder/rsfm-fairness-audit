from __future__ import annotations

import zipfile
from pathlib import Path
from uuid import uuid4

from scripts.run_baseline_closure_sanity import (
    FINAL_DATASET_ZIP_NAME,
    FINAL_MANIFEST_NAME,
    _check_prepared_dataset_zip,
)


def _archive_path() -> Path:
    root = Path("outputs") / f"test_baseline_closure_sanity_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root / FINAL_DATASET_ZIP_NAME


def test_prepared_dataset_zip_accepts_final_manifest_glob() -> None:
    archive = _archive_path()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"dataset/{FINAL_MANIFEST_NAME}", "sample_id,split\ns1,train\n")
    result = _check_prepared_dataset_zip(archive)
    assert result["status"] == "pass"
    assert FINAL_MANIFEST_NAME in result["note"]


def test_prepared_dataset_zip_prioritizes_v3_manifest() -> None:
    archive = _archive_path()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dataset/final_clean_subset_manifest_legacy.csv", "sample_id,split\ns0,train\n")
        zf.writestr(f"dataset/{FINAL_MANIFEST_NAME}", "sample_id,split\ns1,train\n")
    result = _check_prepared_dataset_zip(archive)
    assert result["status"] == "pass"
    assert f"selected_manifest=dataset/{FINAL_MANIFEST_NAME}" in result["note"]
