from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.fmow_geography import apply_country_region_mapping, read_country_region_map
from rsfm_fairness_audit.fmow_sentinel_preflight import (
    FIELD_SYNONYMS,
    _canonical_columns,
    _is_missing,
    _parse_timestamp,
    derive_latitude_band,
)
from rsfm_fairness_audit.io import ensure_dir, write_csv


GEOGRAPHY_FIELDS = ("latitude", "longitude", "country", "region", "continent", "un_region")
DERIVED_FIELDS = ("year", "month", "season", "latitude_band")
JOIN_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("category", "location_id", "image_id"),
    ("location_id", "image_id"),
    ("category", "location_id"),
    ("image_id",),
    ("location_id",),
)
KNOWN_SPLITS = {"train", "val", "valid", "validation", "test"}


@dataclass(frozen=True)
class FmowMetadataEnrichmentConfig:
    satmae_csvs: tuple[Path, ...]
    output_dir: Path
    external_metadata_csvs: tuple[Path, ...] = ()
    join_key: str = "auto"
    infer_split_from_filename: bool = True
    country_region_map: Path | None = None


def _norm_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _read_rows(paths: Sequence[Path], source_kind: str, infer_split: bool = False) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    columns: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            for column in fieldnames:
                if column not in columns:
                    columns.append(column)
            inferred_split = _infer_split(path) if infer_split else ""
            for row in reader:
                item = dict(row)
                item["_source_csv"] = str(path)
                item["_source_kind"] = source_kind
                if infer_split and inferred_split and _missing_canonical_value(item, _canonical_columns(fieldnames), "split"):
                    item["_inferred_split"] = inferred_split
                rows.append(item)
    extra_columns = ["_source_csv", "_source_kind", "_inferred_split"]
    for column in extra_columns:
        if column not in columns:
            columns.append(column)
    for row in rows:
        for column in columns:
            row.setdefault(column, "")
    return rows, columns


def _infer_split(path: Path) -> str:
    stem = path.stem.lower()
    return stem if stem in KNOWN_SPLITS else ""


def _missing_canonical_value(row: Mapping[str, Any], mapping: Mapping[str, str], field: str) -> bool:
    column = mapping.get(field)
    return not column or _is_missing(row.get(column))


def _canonical_value(row: Mapping[str, Any], mapping: Mapping[str, str], field: str) -> str:
    if field == "split" and _is_missing(row.get(mapping.get("split", ""))) and not _is_missing(row.get("_inferred_split")):
        return _norm_text(row.get("_inferred_split"))
    column = mapping.get(field)
    return _norm_text(row.get(column)) if column else ""


def _composite_sample_id(category: str, location_id: str, image_id: str, index: int) -> str:
    parts = [part for part in (category, location_id, image_id) if part]
    return "_".join(parts) if parts else f"fmow-{index:09d}"


def _key_for(row: Mapping[str, Any], mapping: Mapping[str, str], fields: Sequence[str]) -> tuple[str, ...] | None:
    values = tuple(_canonical_value(row, mapping, field) for field in fields)
    if any(_is_missing(value) for value in values):
        return None
    return values


def _parse_join_key(value: str) -> tuple[str, ...] | None:
    if value == "auto":
        return None
    fields = tuple(part.strip() for part in value.split("+") if part.strip())
    allowed = {"category", "location_id", "image_id"}
    if not fields or any(field not in allowed for field in fields):
        raise ValueError("--join-key must be auto or a + separated subset of category, location_id, image_id.")
    return fields


def _best_join_candidate(
    satmae_rows: Sequence[Mapping[str, Any]],
    satmae_mapping: Mapping[str, str],
    external_rows: Sequence[Mapping[str, Any]],
    external_mapping: Mapping[str, str],
    requested: tuple[str, ...] | None,
) -> tuple[tuple[str, ...] | None, dict[tuple[str, ...], Mapping[str, Any]], dict[str, Any]]:
    candidates = (requested,) if requested else JOIN_CANDIDATES
    best: tuple[str, ...] | None = None
    best_lookup: dict[tuple[str, ...], Mapping[str, Any]] = {}
    best_matches = -1
    best_duplicates = 0
    for candidate in candidates:
        if candidate is None:
            continue
        lookup: dict[tuple[str, ...], Mapping[str, Any]] = {}
        duplicate_counts: Counter[tuple[str, ...]] = Counter()
        for row in external_rows:
            key = _key_for(row, external_mapping, candidate)
            if key is None:
                continue
            duplicate_counts[key] += 1
            lookup.setdefault(key, row)
        matches = 0
        for row in satmae_rows:
            key = _key_for(row, satmae_mapping, candidate)
            if key is not None and key in lookup:
                matches += 1
        duplicates = sum(1 for count in duplicate_counts.values() if count > 1)
        if requested:
            best = candidate
            best_lookup = lookup
            best_matches = matches
            best_duplicates = duplicates
            break
        if matches > best_matches:
            best = candidate
            best_lookup = lookup
            best_matches = matches
            best_duplicates = duplicates
    diagnostics = {
        "join_key": "+".join(best) if best else "",
        "matched_rows": max(best_matches, 0),
        "external_lookup_rows": len(best_lookup),
        "duplicate_external_keys": best_duplicates,
    }
    return best, best_lookup, diagnostics


def _external_field(row: Mapping[str, Any] | None, mapping: Mapping[str, str], field: str) -> str:
    if row is None:
        return ""
    return _canonical_value(row, mapping, field)


def _enrich(
    satmae_rows: Sequence[Mapping[str, Any]],
    satmae_columns: Sequence[str],
    external_rows: Sequence[Mapping[str, Any]],
    external_columns: Sequence[str],
    join_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    satmae_mapping = _canonical_columns(satmae_columns)
    external_mapping = _canonical_columns(external_columns)
    requested = _parse_join_key(join_key)
    selected_key, lookup, diagnostics = _best_join_candidate(
        satmae_rows, satmae_mapping, external_rows, external_mapping, requested
    )
    enriched: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not external_rows:
        warnings.append("No external fMoW/GPS/geography metadata CSV was provided; country/region/coordinate fields are not derived.")
    elif not selected_key or diagnostics["matched_rows"] == 0:
        warnings.append("External metadata was provided, but no SatMAE rows matched the available join keys.")
    if diagnostics.get("duplicate_external_keys", 0):
        warnings.append(f"External metadata has {diagnostics['duplicate_external_keys']} duplicate join keys; first row per key was used.")

    for index, row in enumerate(satmae_rows):
        category = _canonical_value(row, satmae_mapping, "category")
        location_id = _canonical_value(row, satmae_mapping, "location_id")
        image_id = _canonical_value(row, satmae_mapping, "image_id") or f"fmow-{index:09d}"
        key = _key_for(row, satmae_mapping, selected_key) if selected_key else None
        matched = lookup.get(key) if key is not None else None
        timestamp = _canonical_value(row, satmae_mapping, "timestamp") or _external_field(matched, external_mapping, "timestamp")
        year, month, season = _parse_timestamp(timestamp)
        split = _canonical_value(row, satmae_mapping, "split") or _external_field(matched, external_mapping, "split") or "all"
        item: dict[str, Any] = {
            "sample_id": _composite_sample_id(category, location_id, image_id, index),
            "image_id": image_id,
            "category": category,
            "label": category,
            "location_id": location_id,
            "timestamp": timestamp,
            "year": year,
            "month": month,
            "season": season,
            "split": split,
            "image_path": _canonical_value(row, satmae_mapping, "image_path") or _external_field(matched, external_mapping, "image_path"),
            "dataset": "fmow_sentinel",
            "task": "scene_classification",
            "input_mode": "s2_13band_image_only",
            "band_profile": "sentinel2_13band_fmow",
            "split_protocol": "official_split",
            "metadata_source_csv": row.get("_source_csv", ""),
            "external_metadata_source_csv": matched.get("_source_csv", "") if matched else "",
            "join_key": "+".join(selected_key) if selected_key else "",
            "join_key_value": "|".join(key) if key is not None else "",
            "join_status": "matched" if matched else ("not_attempted_no_external_metadata" if not external_rows else "unmatched"),
            "metadata_provenance": "satmae_csv",
        }
        provenance_parts = ["satmae_csv"]
        for field in GEOGRAPHY_FIELDS:
            direct = _canonical_value(row, satmae_mapping, field)
            external = _external_field(matched, external_mapping, field)
            value = direct or external
            item[field] = value
            if direct:
                item[f"{field}_provenance"] = "satmae_csv"
                provenance_parts.append(f"{field}:satmae_csv")
            elif external:
                item[f"{field}_provenance"] = "external_metadata_csv"
                provenance_parts.append(f"{field}:external_metadata_csv")
            else:
                item[f"{field}_provenance"] = "unavailable"
        item["latitude_band"] = derive_latitude_band(item["latitude"])
        item["latitude_band_provenance"] = "derived_from_latitude" if item["latitude_band"] else "unavailable"
        if item["latitude_band"]:
            provenance_parts.append("latitude_band:derived_from_latitude")
        item["metadata_provenance"] = ";".join(dict.fromkeys(provenance_parts))
        if item["location_id"] and not item["country"]:
            item["geography_warning"] = "location_id_available_but_not_country"
        else:
            item["geography_warning"] = ""
        if not matched:
            failures.append(
                {
                    "sample_id": item["sample_id"],
                    "image_id": image_id,
                    "category": category,
                    "location_id": location_id,
                    "attempted_join_key": item["join_key"],
                    "attempted_join_key_value": item["join_key_value"],
                    "reason": item["join_status"],
                    "source_csv": item["metadata_source_csv"],
                }
            )
        elif not any(item[field] for field in GEOGRAPHY_FIELDS):
            failures.append(
                {
                    "sample_id": item["sample_id"],
                    "image_id": image_id,
                    "category": category,
                    "location_id": location_id,
                    "attempted_join_key": item["join_key"],
                    "attempted_join_key_value": item["join_key_value"],
                    "reason": "matched_but_no_geography_fields_available",
                    "source_csv": item["metadata_source_csv"],
                }
            )
        enriched.append(item)
    diagnostics.update(
        {
            "satmae_rows": len(satmae_rows),
            "external_rows": len(external_rows),
            "join_rate": diagnostics["matched_rows"] / len(satmae_rows) if satmae_rows else 0.0,
        }
    )
    return enriched, failures, warnings, diagnostics


def _coverage_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    output: list[dict[str, Any]] = []
    for field in [*GEOGRAPHY_FIELDS, *DERIVED_FIELDS, "image_path", "split"]:
        non_missing = sum(1 for row in rows if not _is_missing(row.get(field)))
        provenance_counts: Counter[str] = Counter()
        prov_field = f"{field}_provenance"
        for row in rows:
            provenance = _norm_text(row.get(prov_field))
            if provenance:
                provenance_counts[provenance] += 1
        output.append(
            {
                "field": field,
                "non_missing_count": non_missing,
                "missing_count": total - non_missing,
                "coverage": non_missing / total if total else 0.0,
                "provenance_counts": ";".join(f"{key}:{value}" for key, value in sorted(provenance_counts.items())),
                "ready_for_formal_geography_bwer": field in {"country", "region", "continent", "un_region", "latitude_band"} and non_missing == total and total > 0,
            }
        )
    return output


def _write_join_report(
    path: Path,
    config: FmowMetadataEnrichmentConfig,
    diagnostics: Mapping[str, Any],
    coverage: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
) -> None:
    coverage_by_field = {str(row["field"]): row for row in coverage}
    lines = [
        "# fMoW-Sentinel Metadata Join Report",
        "",
        "This enrichment step joins SatMAE fMoW-Sentinel CSV rows with optional original fMoW, GPS, or external geography CSV metadata.",
        "It does not train models, download imagery, geocode coordinates, or fabricate geography fields.",
        "",
        "## Inputs",
        "",
        f"- SatMAE CSVs: {', '.join(str(path) for path in config.satmae_csvs)}",
        f"- External metadata CSVs: {', '.join(str(path) for path in config.external_metadata_csvs) if config.external_metadata_csvs else 'none'}",
        f"- country-region map: {config.country_region_map if config.country_region_map else 'none'}",
        f"- requested join key: {config.join_key}",
        f"- selected join key: {diagnostics.get('join_key', '') or 'none'}",
        f"- SatMAE rows: {diagnostics.get('satmae_rows', 0)}",
        f"- external rows: {diagnostics.get('external_rows', 0)}",
        f"- matched rows: {diagnostics.get('matched_rows', 0)}",
        f"- join rate: {float(diagnostics.get('join_rate', 0.0)):.4f}",
        "",
        "## Geography Coverage",
        "",
    ]
    for field in GEOGRAPHY_FIELDS + ("latitude_band",):
        row = coverage_by_field.get(field, {})
        lines.append(f"- {field}: coverage={float(row.get('coverage', 0.0)):.4f}, non_missing={row.get('non_missing_count', 0)}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `location_id` is a scene/location identifier and must not be interpreted as country.",
            "- Country, region, continent, and UN region are copied only when supplied by source or external metadata.",
            "- `latitude_band` is derived only from available latitude.",
            "- Geography metadata is for audit slicing and reporting, not model input.",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fmow_sentinel_metadata_enrichment(config: FmowMetadataEnrichmentConfig) -> dict[str, Path]:
    output = ensure_dir(config.output_dir)
    satmae_rows, satmae_columns = _read_rows(config.satmae_csvs, "satmae_csv", infer_split=config.infer_split_from_filename)
    external_rows, external_columns = _read_rows(config.external_metadata_csvs, "external_metadata_csv") if config.external_metadata_csvs else ([], [])
    enriched, failures, warnings, diagnostics = _enrich(
        satmae_rows,
        satmae_columns,
        external_rows,
        external_columns,
        config.join_key,
    )
    country_map, map_warnings = read_country_region_map(config.country_region_map)
    mapping_stats: dict[str, int] = {}
    mapping_apply_warnings: list[str] = []
    if country_map:
        enriched, mapping_apply_warnings, mapping_stats = apply_country_region_mapping(enriched, country_map)
    warnings.extend(map_warnings)
    warnings.extend(mapping_apply_warnings)
    if any(row.get("location_id") for row in enriched) and not any(row.get("country") for row in enriched):
        warnings.append("location_id is available, but country is missing; do not interpret location_id as country.")
    if not any(row.get("latitude") and row.get("longitude") for row in enriched):
        warnings.append("No complete latitude/longitude pairs are available after enrichment; latitude_band and coordinate-derived geography remain unavailable.")
    if not any(row.get("country") for row in enriched):
        warnings.append("Country/region geography BWER is not formal-ready until original fMoW/GPS/geography metadata with coordinates or country fields is supplied.")
    coverage = _coverage_rows(enriched)
    artifacts = {
        "enriched_metadata": output / "fmow_enriched_metadata.csv",
        "join_report": output / "fmow_metadata_join_report.md",
        "geography_coverage_summary": output / "fmow_geography_coverage_summary.csv",
        "join_failures": output / "fmow_join_failures.csv",
        "warnings": output / "warnings.json",
        "run_metadata": output / "run_metadata.json",
    }
    write_csv(artifacts["enriched_metadata"], enriched)
    write_csv(artifacts["geography_coverage_summary"], coverage)
    write_csv(artifacts["join_failures"], failures)
    artifacts["warnings"].write_text(json.dumps({"warnings": warnings}, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "dataset": "fmow_sentinel",
        "task": "scene_classification",
        "workflow": "metadata_enrichment",
        "satmae_csvs": [str(path) for path in config.satmae_csvs],
        "external_metadata_csvs": [str(path) for path in config.external_metadata_csvs],
        "country_region_map": str(config.country_region_map) if config.country_region_map else "",
        "country_region_mapping_stats": mapping_stats,
        "join_key_requested": config.join_key,
        **diagnostics,
        "outputs": {name: str(path) for name, path in artifacts.items()},
    }
    artifacts["run_metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    _write_join_report(artifacts["join_report"], config, diagnostics, coverage, warnings)
    return artifacts
