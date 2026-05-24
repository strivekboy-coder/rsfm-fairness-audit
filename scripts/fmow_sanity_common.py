from __future__ import annotations

import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable


FINAL_MANIFEST_NAME = "final_clean_subset_manifest_30k_location_disjoint_v3_merged.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def find_manifest(root: Path) -> Path:
    candidates = sorted(root.rglob("final_clean_subset_manifest*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No final_clean_subset_manifest*.csv found under {root}")
    candidates = sorted(candidates, key=lambda path: (path.name != FINAL_MANIFEST_NAME, str(path)))
    return candidates[0]


def ensure_extracted_dataset(prepared_dataset_zip: Path | None, extract_dir: Path, manifest: Path | None) -> Path:
    if manifest is not None:
        return manifest
    if extract_dir.exists():
        try:
            return find_manifest(extract_dir)
        except FileNotFoundError:
            pass
    if prepared_dataset_zip is None:
        raise FileNotFoundError("Provide --manifest or --prepared-dataset-zip.")
    print(f"[extract] extracting {prepared_dataset_zip} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(prepared_dataset_zip) as archive:
        archive.extractall(extract_dir)
    return find_manifest(extract_dir)


def run_command(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def python_module_cmd(module: str, args: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *args]


def read_csv_from_zip(zip_path: Path, suffix: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if not matches:
            return []
        with archive.open(sorted(matches)[0]) as handle:
            text = handle.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def first_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return rows[0] if rows else {}

