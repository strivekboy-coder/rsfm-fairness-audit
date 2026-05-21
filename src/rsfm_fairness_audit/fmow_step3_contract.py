from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.bwer import is_invalid_balance_variable
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows


PREDICTION_REQUIRED_COLUMNS = {
    "sample_id",
    "image_id",
    "image_path",
    "dataset",
    "task",
    "split",
    "prediction",
    "correct",
    "risk",
    "model_family",
    "model_variant",
    "input_mode",
    "adaptation_protocol",
    "split_protocol",
    "eval_scope",
    "resolution",
    "band_profile",
    "location_id",
    "country",
    "continent",
    "un_region",
    "region",
    "latitude_band",
}

PREDICTION_OPTIONAL_CONTEXT_COLUMNS = {"timestamp", "year", "month", "season", "latitude", "longitude"}
VALID_BWER_SLICES = {
    "country",
    "continent",
    "un_region",
    "region",
    "latitude_band",
    "season",
    "category",
    "class_label",
    "country__class_label",
    "region__class_label",
    "season__class_label",
    "latitude_band__class_label",
    "country__category",
    "region__category",
    "season__category",
    "latitude_band__category",
}
KNOWN_ALIAS_PAIRS = {("country", "region"), ("region", "country")}
SMALL_FILE_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}
IMAGE_SUFFIXES = {".tif", ".tiff", ".npy", ".npz", ".h5", ".hdf5"}


@dataclass(frozen=True)
class FmowStep3ValidationConfig:
    run_dir: Path
    output_dir: Path | None = None
    run_name: str | None = None
    archive_source_url: str = "https://stacks.stanford.edu/file/druid:vg497cb6002/fmow-sentinel.tar.gz"
    full_archive_downloaded_locally: bool | None = None
    full_extraction_avoided: bool = True
    streaming_partial_extraction_excluded: bool = True


@dataclass(frozen=True)
class FmowStep3PackageConfig:
    run_dir: Path
    output_zip: Path | None = None
    include_rasters: bool = False


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).lower() in {"nan", "none", "null"}


def _as_float(value: Any) -> float | None:
    try:
        if _is_missing(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _find_files(run_dir: Path, names: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for name in names:
        found.extend(path for path in run_dir.rglob(name) if path.is_file())
    return sorted(set(found))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_markdown(path: Path, title: str, rows: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *rows]) + "\n", encoding="utf-8")
    return path


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "root"


def validate_prediction_table(path: Path, output_dir: Path, output_stem: str = "prediction_table_validation") -> dict[str, Any]:
    rows = read_csv_rows(path)
    columns = set(rows[0]) if rows else set()
    blocking: list[str] = []
    cautions: list[str] = []
    missing = sorted(PREDICTION_REQUIRED_COLUMNS - columns)
    if missing:
        blocking.append(f"Missing required columns: {', '.join(missing)}")
    if "label" not in columns and "category" not in columns:
        blocking.append("Missing label/category column.")
    optional_present = sorted(PREDICTION_OPTIONAL_CONTEXT_COLUMNS & columns)
    for column in PREDICTION_OPTIONAL_CONTEXT_COLUMNS - columns:
        cautions.append(f"Optional context column not present: {column}")
    duplicate_count = len(rows) - len({row.get("sample_id", "") for row in rows})
    if duplicate_count:
        blocking.append(f"Duplicate sample_id rows detected: {duplicate_count}")
    bad_dataset = sum(1 for row in rows if row.get("dataset") != "fmow_sentinel")
    bad_task = sum(1 for row in rows if row.get("task") != "scene_classification")
    bad_input = sum(1 for row in rows if "image_only" not in str(row.get("input_mode", "")))
    if bad_dataset:
        blocking.append(f"Rows with dataset != fmow_sentinel: {bad_dataset}")
    if bad_task:
        blocking.append(f"Rows with task != scene_classification: {bad_task}")
    if bad_input:
        blocking.append(f"Rows not marked image-only input_mode: {bad_input}")
    missing_protocol = sum(1 for row in rows if _is_missing(row.get("adaptation_protocol")) or _is_missing(row.get("split_protocol")))
    if missing_protocol:
        blocking.append(f"Rows with missing adaptation_protocol or split_protocol: {missing_protocol}")
    inconsistent = 0
    for row in rows:
        correct = _as_float(row.get("correct"))
        risk = _as_float(row.get("risk"))
        if correct is not None and risk is not None and abs((1.0 - correct) - risk) > 1e-6:
            inconsistent += 1
    if inconsistent:
        blocking.append(f"Rows where risk != 1 - correct: {inconsistent}")
    geo_columns = ["country", "continent", "un_region", "region", "latitude_band", "location_id"]
    geo_missing = {
        column: (sum(1 for row in rows if _is_missing(row.get(column))) / len(rows) if rows else 1.0)
        for column in geo_columns
        if column in columns
    }
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[str(row.get("split", ""))] = split_counts.get(str(row.get("split", "")), 0) + 1
    if len(split_counts) <= 1:
        cautions.append("Prediction table contains one split only; confirm this matches eval_scope.")
    payload = {
        "path": str(path),
        "row_count": len(rows),
        "column_count": len(columns),
        "missing_required_columns": missing,
        "optional_context_columns_present": optional_present,
        "duplicate_sample_id_count": duplicate_count,
        "bad_dataset_rows": bad_dataset,
        "bad_task_rows": bad_task,
        "bad_input_mode_rows": bad_input,
        "missing_protocol_label_rows": missing_protocol,
        "correct_risk_inconsistent_rows": inconsistent,
        "geography_missing_ratios": geo_missing,
        "split_counts": split_counts,
        "geography_metadata_policy": "audit_slicing_reporting_only_not_model_input",
        "blocking_warnings": blocking,
        "caution_warnings": cautions,
        "passed": not blocking,
    }
    _write_json(output_dir / f"{output_stem}.json", payload)
    lines = [
        f"- prediction table: `{path}`",
        f"- rows: {len(rows)}",
        f"- passed: `{payload['passed']}`",
        f"- blocking warnings: {len(blocking)}",
        f"- caution warnings: {len(cautions)}",
        "",
        "## Blocking Warnings",
        *(f"- {warning}" for warning in blocking or ["none"]),
        "",
        "## Caution Warnings",
        *(f"- {warning}" for warning in cautions or ["none"]),
        "",
        "## Geography Missing Ratios",
        *(f"- {key}: {value:.6f}" for key, value in geo_missing.items()),
    ]
    _write_markdown(output_dir / f"{output_stem}.md", "fMoW Step 3 Prediction Table Validation", lines)
    return payload


def validate_bwer_output(bwer_dir: Path, output_dir: Path, audit_table: Path | None = None) -> dict[str, Any]:
    blocking: list[str] = []
    cautions: list[str] = []
    summary_path = bwer_dir / "bwer_summary.csv"
    by_slice_path = bwer_dir / "bwer_by_slice.csv"
    if not summary_path.exists():
        blocking.append("Missing bwer_summary.csv.")
        rows: list[dict[str, str]] = []
    else:
        rows = read_csv_rows(summary_path)
    if not by_slice_path.exists():
        cautions.append("Missing bwer_by_slice.csv.")
    raw_entries = [row for row in rows if _is_missing(row.get("balance_variable"))]
    standardised_entries = [row for row in rows if not _is_missing(row.get("balance_variable"))]
    if not raw_entries:
        blocking.append("No Raw-BWER rows detected in bwer_summary.csv.")
    if not standardised_entries:
        cautions.append("No Standardised-BWER rows detected in bwer_summary.csv.")
    invalid_slices = sorted({row.get("slice_variable", "") for row in rows if row.get("slice_variable", "") not in VALID_BWER_SLICES})
    if invalid_slices:
        cautions.append(f"Unexpected slice variables: {', '.join(invalid_slices)}")
    missing_support = sum(1 for row in rows if _is_missing(row.get("n_slices_valid")) and _is_missing(row.get("valid_slice_count")))
    if missing_support:
        cautions.append(f"BWER rows missing valid slice count fields: {missing_support}")
    invalid_balance_rows: list[str] = []
    audit_rows = read_csv_rows(audit_table) if audit_table and audit_table.exists() else []
    for row in rows:
        g = str(row.get("slice_variable", ""))
        z = str(row.get("balance_variable", ""))
        if not z:
            continue
        if g == z or (g, z) in KNOWN_ALIAS_PAIRS:
            invalid_balance_rows.append(f"{g}|{z}")
        elif audit_rows:
            invalid, reason = is_invalid_balance_variable(audit_rows, g, z)
            if invalid:
                invalid_balance_rows.append(f"{g}|{z}: {reason}")
    if invalid_balance_rows:
        blocking.append(f"Invalid balance variable rows detected: {'; '.join(invalid_balance_rows)}")
    warnings_paths = [str(path) for path in bwer_dir.rglob("warnings.json")]
    payload = {
        "bwer_dir": str(bwer_dir),
        "bwer_summary_exists": summary_path.exists(),
        "bwer_by_slice_exists": by_slice_path.exists(),
        "summary_row_count": len(rows),
        "raw_bwer_row_count": len(raw_entries),
        "standardised_bwer_row_count": len(standardised_entries),
        "invalid_balance_rows": invalid_balance_rows,
        "warnings_files": warnings_paths,
        "blocking_warnings": blocking,
        "caution_warnings": cautions,
        "passed": not blocking,
    }
    _write_json(output_dir / "bwer_output_validation.json", payload)
    lines = [
        f"- BWER directory: `{bwer_dir}`",
        f"- passed: `{payload['passed']}`",
        f"- raw rows: {len(raw_entries)}",
        f"- standardised rows: {len(standardised_entries)}",
        "",
        "## Blocking Warnings",
        *(f"- {warning}" for warning in blocking or ["none"]),
        "",
        "## Caution Warnings",
        *(f"- {warning}" for warning in cautions or ["none"]),
    ]
    _write_markdown(output_dir / "bwer_output_validation.md", "fMoW Step 3 BWER Output Validation", lines)
    return payload


def _collect_file_records(run_dir: Path, include_rasters: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES and not include_rasters:
            continue
        if suffix not in SMALL_FILE_SUFFIXES and suffix not in IMAGE_SUFFIXES:
            continue
        records.append(
            {
                "relative_path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def collect_provenance(
    run_dir: Path,
    output_dir: Path,
    validation: Mapping[str, Any],
    config: FmowStep3ValidationConfig,
) -> dict[str, Any]:
    run_metadata_paths = _find_files(run_dir, ["run_metadata.json"])
    warning_paths = _find_files(run_dir, ["warnings.json"])
    metadata_blobs = [_read_json(path) for path in run_metadata_paths]
    labels: dict[str, Any] = {}
    for blob in metadata_blobs:
        for key in ["model", "model_family", "model_variant", "adaptation_protocol", "split_protocol", "band_profile", "resolution", "seed"]:
            if key in blob and key not in labels:
                labels[key] = blob[key]
    payload = {
        "run_name": config.run_name or run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path.cwd()),
        "run_dir": str(run_dir),
        "archive_source_url": config.archive_source_url,
        "full_archive_downloaded_locally": config.full_archive_downloaded_locally,
        "full_extraction_avoided": config.full_extraction_avoided,
        "streaming_partial_extraction_excluded_from_formal_results": config.streaming_partial_extraction_excluded,
        "extraction_strategy": "local_tar_target_path_extraction_no_full_extract",
        "model_protocol_labels": labels,
        "important_files": _collect_file_records(run_dir, include_rasters=False),
        "run_metadata_files": [str(path.relative_to(run_dir)) for path in run_metadata_paths],
        "warnings_files": [str(path.relative_to(run_dir)) for path in warning_paths],
        "validation_summary": validation,
    }
    _write_json(output_dir / "archive_manifest.json", payload)
    lines = [
        f"- run_name: `{payload['run_name']}`",
        f"- git_commit: `{payload['git_commit'] or 'unavailable'}`",
        f"- extraction_strategy: `{payload['extraction_strategy']}`",
        f"- full_extraction_avoided: `{payload['full_extraction_avoided']}`",
        f"- streaming_partial_extraction_excluded: `{payload['streaming_partial_extraction_excluded_from_formal_results']}`",
        f"- tracked small files: {len(payload['important_files'])}",
        "",
        "Geography metadata is recorded for audit slicing and reporting only, not as model input.",
    ]
    _write_markdown(output_dir / "provenance_report.md", "fMoW Step 3 Provenance Report", lines)
    return payload


def _find_first_existing(run_dir: Path, candidates: Sequence[str]) -> Path | None:
    for candidate in candidates:
        path = run_dir / candidate
        if path.exists():
            return path
    for candidate in candidates:
        found = sorted(run_dir.rglob(candidate))
        if found:
            return found[0]
    return None


def generate_handoff_checklist(output_dir: Path, validation: Mapping[str, Any]) -> Path:
    required = [
        "archive_manifest.json",
        "provenance_report.md",
        "prediction_table_validation.json",
        "prediction_table_validation.md",
        "bwer_output_validation.json",
        "bwer_output_validation.md",
    ]
    present = [name for name in required if (output_dir / name).exists()]
    missing = [name for name in required if not (output_dir / name).exists()]
    prediction_ok = bool(validation.get("prediction", {}).get("passed"))
    bwer_ok = bool(validation.get("bwer", {}).get("passed"))
    blocking = list(validation.get("prediction", {}).get("blocking_warnings", [])) + list(validation.get("bwer", {}).get("blocking_warnings", []))
    cautions = list(validation.get("prediction", {}).get("caution_warnings", [])) + list(validation.get("bwer", {}).get("caution_warnings", []))
    ready = prediction_ok and bwer_ok and not missing
    lines = [
        "## Required Files Present",
        *(f"- {name}" for name in present or ["none"]),
        "",
        "## Required Files Missing",
        *(f"- {name}" for name in missing or ["none"]),
        "",
        "## Blocking Warnings",
        *(f"- {warning}" for warning in blocking or ["none"]),
        "",
        "## Caution Warnings",
        *(f"- {warning}" for warning in cautions or ["none"]),
        "",
        "## Readiness",
        f"- prediction table ready: `{prediction_ok}`",
        f"- BWER outputs complete: `{bwer_ok}`",
        f"- model comparison ready: `{prediction_ok and bwer_ok}`",
        f"- ready for scientific interpretation: `{ready}`",
        "",
        "This checklist reports readiness only. It does not infer scientific findings.",
    ]
    return _write_markdown(output_dir / "handoff_checklist.md", "fMoW Step 3 Handoff Checklist", lines)


def validate_fmow_step3_results(config: FmowStep3ValidationConfig) -> dict[str, Path]:
    run_dir = config.run_dir
    output_dir = ensure_dir(config.output_dir or run_dir)
    prediction_paths = sorted(run_dir.rglob("predictions.csv"))
    if not prediction_paths:
        raise FileNotFoundError(f"No predictions.csv found under {run_dir}")
    bwer_dir = _find_first_existing(run_dir, ["bwer_summary.csv"])
    if bwer_dir is None:
        raise FileNotFoundError(f"No bwer_summary.csv found under {run_dir}")
    bwer_dir = bwer_dir.parent
    audit_table = _find_first_existing(run_dir, ["audit_table.csv"])
    prediction_validations = []
    for path in prediction_paths:
        relative_parent = path.parent.relative_to(run_dir).as_posix()
        stem = "prediction_table_validation" if len(prediction_paths) == 1 else f"prediction_table_validation_{_safe_name(relative_parent)}"
        prediction_validations.append(validate_prediction_table(path, output_dir, stem))
    if len(prediction_validations) == 1:
        prediction_validation = prediction_validations[0]
        # Keep canonical filenames for the single-run path.
        if not (output_dir / "prediction_table_validation.json").exists():
            _write_json(output_dir / "prediction_table_validation.json", prediction_validation)
    else:
        blocking = [warning for item in prediction_validations for warning in item.get("blocking_warnings", [])]
        cautions = [warning for item in prediction_validations for warning in item.get("caution_warnings", [])]
        prediction_validation = {
            "table_count": len(prediction_validations),
            "tables": prediction_validations,
            "blocking_warnings": blocking,
            "caution_warnings": cautions,
            "passed": all(bool(item.get("passed")) for item in prediction_validations),
        }
        _write_json(output_dir / "prediction_table_validation.json", prediction_validation)
        _write_markdown(
            output_dir / "prediction_table_validation.md",
            "fMoW Step 3 Prediction Table Validation",
            [
                f"- prediction tables: {len(prediction_validations)}",
                f"- passed: `{prediction_validation['passed']}`",
                "",
                "## Tables",
                *(f"- `{item['path']}` passed=`{item['passed']}`" for item in prediction_validations),
            ],
        )
    bwer_validation = validate_bwer_output(bwer_dir, output_dir, audit_table)
    validation = {"prediction": prediction_validation, "bwer": bwer_validation}
    collect_provenance(run_dir, output_dir, validation, config)
    generate_handoff_checklist(output_dir, validation)
    return {
        "prediction_table_validation_json": output_dir / "prediction_table_validation.json",
        "prediction_table_validation_md": output_dir / "prediction_table_validation.md",
        "bwer_output_validation_json": output_dir / "bwer_output_validation.json",
        "bwer_output_validation_md": output_dir / "bwer_output_validation.md",
        "archive_manifest": output_dir / "archive_manifest.json",
        "provenance_report": output_dir / "provenance_report.md",
        "handoff_checklist": output_dir / "handoff_checklist.md",
    }


def package_fmow_step3_handoff(config: FmowStep3PackageConfig) -> dict[str, Path]:
    run_dir = config.run_dir
    output_zip = config.output_zip or run_dir.parent / f"fmow_step3_{run_dir.name}_handoff.zip"
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    records = _collect_file_records(run_dir, include_rasters=config.include_rasters)
    manifest = {
        "run_dir": str(run_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "include_rasters": config.include_rasters,
        "files": records,
    }
    package_manifest = run_dir / "handoff_package_manifest.json"
    _write_json(package_manifest, manifest)
    records = _collect_file_records(run_dir, include_rasters=config.include_rasters)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in records:
            relative = Path(record["relative_path"])
            if not config.include_rasters and relative.suffix.lower() in IMAGE_SUFFIXES:
                continue
            archive.write(run_dir / relative, arcname=f"{run_dir.name}/{relative.as_posix()}")
    return {"handoff_zip": output_zip, "handoff_package_manifest": package_manifest}
