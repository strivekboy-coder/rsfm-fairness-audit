from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.fmow_geography import apply_country_region_mapping, read_country_region_map
from rsfm_fairness_audit.io import ensure_dir, write_csv


FIELD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "category": ("category", "class", "label", "target", "class_name"),
    "location_id": ("location_id", "location", "loc_id", "location_uuid"),
    "timestamp": ("timestamp", "datetime", "date", "acquisition_date", "image_timestamp", "captured_at"),
    "image_id": ("image_id", "id", "uuid", "image_uuid", "img_id"),
    "image_path": ("image_path", "path", "file_path", "tif_path", "raster_path", "filename"),
    "split": ("split", "official_split", "partition"),
    "latitude": ("latitude", "lat", "gps_lat", "center_lat", "utm_lat"),
    "longitude": ("longitude", "lon", "lng", "gps_lon", "center_lon", "utm_lon"),
    "country": ("country", "country_name", "iso_country"),
    "region": ("region", "admin_region", "geo_region"),
    "continent": ("continent",),
    "un_region": ("un_region", "unregion", "world_region"),
}

REQUIRED_FUTURE_AUDIT_COLUMNS = [
    "sample_id",
    "image_id",
    "image_path",
    "dataset",
    "task",
    "split",
    "label",
    "category",
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
    "timestamp",
    "year",
    "month",
    "season",
    "location_id",
    "latitude",
    "longitude",
    "country",
    "region",
    "continent",
    "un_region",
    "latitude_band",
    "metadata_provenance",
]


@dataclass(frozen=True)
class FmowPreflightConfig:
    metadata_csvs: tuple[Path, ...]
    output_dir: Path
    data_root: Path | None = None
    split_protocol: str = "official_split"
    filter_splits: tuple[str, ...] = ()
    subset_max_per_split: int = 5000
    seed: int = 42
    metadata_only: bool = False
    inspect_rasters: bool = False
    raster_sample_size: int = 256
    min_support: int = 20
    country_region_map: Path | None = None


def _norm(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_columns(columns: Sequence[str]) -> dict[str, str]:
    normalized = {_norm(column): column for column in columns}
    mapping: dict[str, str] = {}
    for canonical, synonyms in FIELD_SYNONYMS.items():
        for synonym in synonyms:
            if _norm(synonym) in normalized:
                mapping[canonical] = normalized[_norm(synonym)]
                break
    return mapping


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).strip().lower() in {"nan", "none", "null"}


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_rows(paths: Sequence[Path]) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    rows: list[dict[str, str]] = []
    columns: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for column in reader.fieldnames:
                    if column not in columns:
                        columns.append(column)
            for row in reader:
                item = dict(row)
                item["_source_csv"] = str(path)
                rows.append(item)
    all_columns = list(columns) + ["_source_csv"]
    for row in rows:
        for column in all_columns:
            row.setdefault(column, "")
    return rows, columns, _canonical_columns(columns)


def derive_season(month: int | float | str | None) -> str:
    try:
        value = int(month)  # noqa: PLW2901
    except (TypeError, ValueError):
        return ""
    if value in {12, 1, 2}:
        return "DJF"
    if value in {3, 4, 5}:
        return "MAM"
    if value in {6, 7, 8}:
        return "JJA"
    if value in {9, 10, 11}:
        return "SON"
    return ""


def derive_latitude_band(latitude: Any) -> str:
    lat = _parse_float(latitude)
    if math.isnan(lat) or lat < -90 or lat > 90:
        return ""
    if lat < -60:
        return "south_high_latitude"
    if lat < -30:
        return "south_mid_latitude"
    if lat < 0:
        return "south_tropics"
    if lat < 30:
        return "north_tropics"
    if lat < 60:
        return "north_mid_latitude"
    return "north_high_latitude"


def _parse_timestamp(value: Any) -> tuple[str, str, str]:
    if _is_missing(value):
        return "", "", ""
    text = str(value).strip()
    candidates = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
    ]
    cleaned = text.replace("Z", "")
    for fmt in candidates:
        try:
            dt = datetime.strptime(cleaned[: len(datetime.now().strftime(fmt))], fmt)
            return str(dt.year), str(dt.month), derive_season(dt.month)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(cleaned)
        return str(dt.year), str(dt.month), derive_season(dt.month)
    except ValueError:
        return "", "", ""


def _value(row: Mapping[str, str], mapping: Mapping[str, str], field: str) -> str:
    column = mapping.get(field)
    return str(row.get(column, "")) if column else ""


def _enrich_rows(rows: Sequence[Mapping[str, str]], mapping: Mapping[str, str]) -> list[dict[str, Any]]:
    enriched = []
    for index, row in enumerate(rows):
        item = dict(row)
        image_id = _value(row, mapping, "image_id") or f"fmow-{index:09d}"
        category = _value(row, mapping, "category")
        timestamp = _value(row, mapping, "timestamp")
        year, month, season = _parse_timestamp(timestamp)
        latitude = _value(row, mapping, "latitude")
        location_id = _value(row, mapping, "location_id")
        item.update(
            {
                "sample_id": image_id,
                "image_id": image_id,
                "image_path": _value(row, mapping, "image_path"),
                "dataset": "fmow_sentinel",
                "task": "scene_classification",
                "category": category,
                "label": category,
                "split": _value(row, mapping, "split") or "all",
                "timestamp": timestamp,
                "year": year,
                "month": month,
                "season": season,
                "latitude": latitude,
                "longitude": _value(row, mapping, "longitude"),
                "latitude_band": derive_latitude_band(latitude),
                "location_id": location_id,
                "country": _value(row, mapping, "country"),
                "region": _value(row, mapping, "region"),
                "continent": _value(row, mapping, "continent"),
                "un_region": _value(row, mapping, "un_region"),
                "input_mode": "s2_13band_image_only",
                "band_profile": "sentinel2_13band_fmow",
                "split_protocol": "official_split",
                "metadata_provenance": "csv_direct_plus_timestamp_latitude_derivations",
            }
        )
        if item["location_id"] and not item["country"]:
            item["geography_warning"] = "location_id_available_but_not_country"
        enriched.append(item)
    return enriched


def _inventory_rows(rows: Sequence[Mapping[str, str]], columns: Sequence[str], mapping: Mapping[str, str]) -> list[dict[str, Any]]:
    output = []
    total = len(rows)
    for field in FIELD_SYNONYMS:
        column = mapping.get(field, "")
        present = bool(column)
        non_missing = sum(1 for row in rows if present and not _is_missing(row.get(column)))
        status = "present" if present else "missing"
        if field in {"year", "month", "season", "latitude_band"}:
            continue
        output.append(
            {
                "canonical_field": field,
                "matched_column": column,
                "status": status,
                "non_missing_count": non_missing,
                "missing_ratio": 1.0 - (non_missing / total) if total else 1.0,
                "notes": _field_note(field, present),
            }
        )
    for derived in ["year", "month", "season", "latitude_band"]:
        source = "timestamp" if derived in {"year", "month", "season"} else "latitude"
        source_present = source in mapping
        output.append(
            {
                "canonical_field": derived,
                "matched_column": f"derived_from_{source}" if source_present else "",
                "status": "derivable" if source_present else "missing",
                "non_missing_count": "",
                "missing_ratio": "",
                "notes": f"Derived from {source} when parseable." if source_present else f"Requires {source}.",
            }
        )
    output.append(
        {
            "canonical_field": "__all_columns__",
            "matched_column": ";".join(columns),
            "status": "observed",
            "non_missing_count": total,
            "missing_ratio": 0.0,
            "notes": f"{len(columns)} input columns observed.",
        }
    )
    return output


def _update_inventory_for_country_region_map(inventory: Sequence[Mapping[str, Any]], enriched_rows: Sequence[Mapping[str, Any]], map_path: Path | None) -> list[dict[str, Any]]:
    if map_path is None:
        return [dict(row) for row in inventory]
    output: list[dict[str, Any]] = []
    total = len(enriched_rows)
    for row in inventory:
        item = dict(row)
        field = str(item.get("canonical_field", ""))
        if field in {"continent", "un_region", "region"}:
            non_missing = sum(1 for enriched in enriched_rows if not _is_missing(enriched.get(field)))
            if non_missing:
                item.update(
                    {
                        "matched_column": "country_region_map",
                        "status": "derived",
                        "non_missing_count": non_missing,
                        "missing_ratio": 1.0 - (non_missing / total) if total else 1.0,
                        "notes": f"Derived from verified country mapping table: {map_path}",
                    }
                )
        output.append(item)
    return output


def _field_note(field: str, present: bool) -> str:
    if present:
        return "Available from supplied CSV."
    if field in {"country", "region", "continent", "un_region"}:
        return "Missing; requires original fMoW metadata join, coordinates plus boundary resources, or supplied geography table."
    if field in {"latitude", "longitude"}:
        return "Missing; latitude_band and geography derivation unavailable."
    return "Missing from supplied CSV."


def _support_for_candidate(rows: Sequence[Mapping[str, Any]], candidate: str, min_support: int) -> dict[str, Any]:
    parts = candidate.split(" x ")
    total = len(rows)
    missing = 0
    counts: Counter[tuple[str, ...]] = Counter()
    class_sets: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = []
        missing_row = False
        for part in parts:
            value = str(row.get(part, "") or "")
            if _is_missing(value):
                missing_row = True
            key.append(value)
        if missing_row:
            missing += 1
            continue
        key_tuple = tuple(key)
        counts[key_tuple] += 1
        if row.get("category"):
            class_sets[key_tuple].add(str(row.get("category")))
    supports = list(counts.values())
    filtered_supports = [count for count in supports if count >= min_support]
    valid_slices = len(supports)
    missing_ratio = missing / total if total else 1.0
    min_count = min(supports) if supports else 0
    median_count = float(np.median(supports)) if supports else 0.0
    max_count = max(supports) if supports else 0
    class_coverage = [len(classes) for classes in class_sets.values()]
    recommendation, reason = _recommend_candidate(candidate, valid_slices, missing_ratio, min_count, median_count, min_support, bool(supports))
    return {
        "candidate_slice": candidate,
        "valid_slice_count": valid_slices,
        "missing_field_ratio": missing_ratio,
        "min_support": min_count,
        "median_support": median_count,
        "max_support": max_count,
        "min_class_coverage_per_slice": min(class_coverage) if class_coverage else "",
        "median_class_coverage_per_slice": float(np.median(class_coverage)) if class_coverage else "",
        "max_class_coverage_per_slice": max(class_coverage) if class_coverage else "",
        "recommendation": recommendation,
        "reason": reason,
        "support_filtered_valid_slice_count": len(filtered_supports),
        "support_filtered_sample_count": sum(filtered_supports),
        "support_filtered_min_support": min(filtered_supports) if filtered_supports else 0,
        "support_filtered_recommendation": _support_filtered_recommendation(
            candidate,
            valid_slices,
            len(filtered_supports),
            missing_ratio,
            median_count,
            min_support,
            bool(supports),
        ),
        "support_filtered_reason": _support_filtered_reason(candidate, len(filtered_supports), min_support, bool(supports)),
    }


def _recommend_candidate(candidate: str, valid_slices: int, missing_ratio: float, min_count: int, median_count: float, min_support: int, has_support: bool) -> tuple[str, str]:
    if not has_support:
        return "not-recommended", "Required field is missing or entirely empty."
    if missing_ratio > 0.5:
        return "not-recommended", "More than half of rows are missing this field."
    if valid_slices < 2:
        return "not-recommended", "Fewer than two valid slices."
    if min_count >= min_support and missing_ratio <= 0.05:
        return "formal-BWER-ready", "Low missingness and all slices meet minimum support."
    if median_count >= min_support:
        return "diagnostic-only", "Some slices are sparse or missingness is non-trivial."
    return "not-recommended", "Support is too sparse for reliable slice risk."


def _support_filtered_recommendation(
    candidate: str,
    valid_slices: int,
    filtered_slices: int,
    missing_ratio: float,
    median_count: float,
    min_support: int,
    has_support: bool,
) -> str:
    if not has_support or missing_ratio > 0.05 or filtered_slices < 2:
        return "not-recommended"
    if filtered_slices == valid_slices:
        return "same-as-primary"
    if candidate in {"country", "country x category", "region x category", "season x category"} and median_count >= min_support:
        return "support-filtered-formal-BWER-ready"
    return "diagnostic-only"


def _support_filtered_reason(candidate: str, filtered_slices: int, min_support: int, has_support: bool) -> str:
    if not has_support:
        return "No non-missing slices are available."
    if filtered_slices < 2:
        return f"Fewer than two slices meet min_support >= {min_support}."
    if candidate in {"country", "country x category", "region x category", "season x category"}:
        return f"{filtered_slices} slices meet min_support >= {min_support}; formal BWER is defensible only on this support-filtered subset."
    return f"{filtered_slices} slices meet min_support >= {min_support}."


def _support_rows(rows: Sequence[Mapping[str, Any]], min_support: int) -> list[dict[str, Any]]:
    candidates = [
        "category",
        "location_id",
        "country",
        "continent",
        "un_region",
        "region",
        "latitude_band",
        "season",
        "region x category",
        "country x category",
        "season x category",
    ]
    return [_support_for_candidate(rows, candidate, min_support) for candidate in candidates]


def _split_filter(rows: Sequence[dict[str, Any]], filter_splits: Sequence[str]) -> list[dict[str, Any]]:
    if not filter_splits:
        return list(rows)
    allowed = {str(split) for split in filter_splits}
    return [row for row in rows if str(row.get("split", "")) in allowed]


def _stratification_field(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    for field in ["country", "region", "continent", "un_region", "latitude_band"]:
        if any(not _is_missing(row.get(field)) for row in rows):
            return field, "geography_stratified"
    if any(not _is_missing(row.get("location_id")) for row in rows):
        return "location_id", "location_id_stratified_fallback"
    return "", "class_stratified_only"


def _subset_manifest(rows: Sequence[dict[str, Any]], max_per_split: int, seed: int) -> tuple[list[dict[str, Any]], str]:
    rng = random.Random(seed)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split", "all") or "all")].append(row)
    selected: list[dict[str, Any]] = []
    strat_field, strategy = _stratification_field(rows)
    for split, split_rows in sorted(by_split.items()):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in split_rows:
            key = (str(row.get("category", "") or ""), str(row.get(strat_field, "") or "") if strat_field else "")
            grouped[key].append(row)
        for group_rows in grouped.values():
            rng.shuffle(group_rows)
        picked = 0
        groups = list(grouped.values())
        rng.shuffle(groups)
        while picked < max_per_split and any(groups):
            next_groups = []
            for group_rows in groups:
                if picked >= max_per_split:
                    break
                if group_rows:
                    item = dict(group_rows.pop())
                    item["subset_strategy"] = strategy
                    item["subset_stratification_field"] = strat_field
                    selected.append(item)
                    picked += 1
                if group_rows:
                    next_groups.append(group_rows)
            groups = next_groups
    return selected, strategy


def _resolve_image_path(path_value: str, data_root: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute() or data_root is None:
        return path
    return data_root / path


def _read_raster(path: Path) -> np.ndarray:
    try:
        import rasterio  # type: ignore

        with rasterio.open(path) as src:
            return np.asarray(src.read())
    except ImportError:
        pass
    try:
        import tifffile  # type: ignore

        arr = np.asarray(tifffile.imread(path))
        if arr.ndim == 3 and arr.shape[0] in {1, 3, 4, 6, 8, 10, 12, 13, 16}:
            return arr
        if arr.ndim == 3 and arr.shape[-1] <= 32:
            arr = np.moveaxis(arr, -1, 0)
        return arr
    except ImportError as exc:
        raise RuntimeError("Raster inspection requires rasterio or tifffile.") from exc


def _inspect_rasters(rows: Sequence[Mapping[str, Any]], data_root: Path | None, sample_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    stats: list[dict[str, Any]] = []
    shape_counter: Counter[str] = Counter()
    band_counter: Counter[int] = Counter()
    dtype_counter: Counter[str] = Counter()
    warnings: list[str] = []
    failures = 0
    inspected = 0
    band_sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, float("inf"), float("-inf"), 0.0])
    for row in rows[:sample_size]:
        image_path = str(row.get("image_path", "") or "")
        if not image_path:
            failures += 1
            continue
        path = _resolve_image_path(image_path, data_root)
        if not path.exists():
            failures += 1
            continue
        try:
            arr = _read_raster(path)
        except Exception:
            failures += 1
            continue
        inspected += 1
        shape_counter["x".join(str(value) for value in arr.shape)] += 1
        dtype_counter[str(arr.dtype)] += 1
        bands = int(arr.shape[0]) if arr.ndim == 3 else 1
        band_counter[bands] += 1
        if bands != 13:
            warnings.append(f"Raster {path} has {bands} bands, expected 13.")
        arr_float = arr.astype(np.float64, copy=False)
        if arr.ndim == 2:
            arr_float = arr_float[None, :, :]
        for band_index in range(arr_float.shape[0]):
            band = arr_float[band_index]
            valid = np.isfinite(band)
            values = band[valid]
            if values.size == 0:
                continue
            entry = band_sums[band_index + 1]
            entry[0] += float(values.sum())
            entry[1] += float((values**2).sum())
            entry[2] = min(entry[2], float(values.min()))
            entry[3] = max(entry[3], float(values.max()))
            entry[4] += float(values.size)
    for band_index, (sum_value, sum_sq, min_value, max_value, count) in sorted(band_sums.items()):
        mean = sum_value / count if count else float("nan")
        variance = max(0.0, (sum_sq / count) - (mean**2)) if count else float("nan")
        stats.append(
            {
                "band": band_index,
                "sample_count": inspected,
                "valid_pixel_count": int(count),
                "min": min_value,
                "max": max_value,
                "mean": mean,
                "std": math.sqrt(variance) if not math.isnan(variance) else float("nan"),
            }
        )
    shape_rows = [{"shape": shape, "count": count} for shape, count in sorted(shape_counter.items())]
    shape_rows.extend({"shape": f"bands={bands}", "count": count} for bands, count in sorted(band_counter.items()))
    shape_rows.extend({"shape": f"dtype={dtype}", "count": count} for dtype, count in sorted(dtype_counter.items()))
    shape_rows.append({"shape": "path_or_read_failures", "count": failures})
    if len(shape_counter) > 3:
        warnings.append("Raster shapes are highly variable in the inspected sample.")
    if not stats:
        warnings.append("No rasters were successfully inspected.")
    return stats, shape_rows, warnings


def _write_missing_fields_report(path: Path, inventory: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# fMoW-Sentinel Metadata Missing Fields Report", ""]
    for row in inventory:
        if row["canonical_field"] == "__all_columns__":
            continue
        lines.append(f"- {row['canonical_field']}: {row['status']} ({row['notes']})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preflight_report(path: Path, support_rows: Sequence[Mapping[str, Any]], subset_strategy: str, warnings: Sequence[str]) -> None:
    lines = [
        "# fMoW-Sentinel Preflight Report",
        "",
        "This preflight establishes metadata, slice-support, subset-manifest, raster-inspection, and audit-table foundations only. It does not train models or run inference.",
        "",
        f"- subset_strategy: {subset_strategy}",
        "",
        "## Slice Recommendations",
        "",
        "| candidate | valid slices | missing ratio | min | median | max | recommendation | support-filtered recommendation | reason |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in support_rows:
        lines.append(
            f"| {row['candidate_slice']} | {row['valid_slice_count']} | {float(row['missing_field_ratio']):.3f} | {row['min_support']} | {row['median_support']} | {row['max_support']} | {row['recommendation']} | {row['support_filtered_recommendation']} | {row['reason']} |"
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in sorted(set(warnings)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_subset_report(path: Path, subset_rows: Sequence[Mapping[str, Any]], strategy: str) -> None:
    counts = Counter(str(row.get("split", "all")) for row in subset_rows)
    lines = ["# fMoW-Sentinel Subset Support Report", "", f"- subset_strategy: {strategy}", f"- subset_size: {len(subset_rows)}", ""]
    lines.extend(f"- {split}: {count}" for split, count in sorted(counts.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_raster_report(path: Path, inspect_requested: bool, warnings: Sequence[str]) -> None:
    lines = [
        "# fMoW-Sentinel Raster Loading Report",
        "",
        f"- raster_inspection_requested: {inspect_requested}",
        "- reader_order: rasterio, then tifffile",
        "- no PIL/ImageNet/RGB assumptions are used.",
        "",
    ]
    if not inspect_requested:
        lines.append("Raster inspection was skipped in metadata-only mode or because `--inspect-rasters` was not passed.")
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in sorted(set(warnings)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_schema_report(path: Path) -> None:
    lines = [
        "# fMoW-Sentinel Future Audit Table Schema",
        "",
        "Future fMoW-Sentinel predictions should be normalized into BWER-compatible scene-classification rows.",
        "",
        "- dataset: `fmow_sentinel`",
        "- task: `scene_classification`",
        "- input_mode: `s2_13band_image_only`",
        "- band_profile: `sentinel2_13band_fmow`",
        "- split_protocol examples: `official_split`, `location_split`, `region_split`, `time_split`, `custom_stratified_subset`",
        "- risk: sample-level 0/1 error for classification",
        "",
        "## Columns",
        "",
    ]
    lines.extend(f"- `{column}`" for column in REQUIRED_FUTURE_AUDIT_COLUMNS)
    lines.extend(
        [
            "",
            "Geography metadata is for audit slicing and reporting. It should not be used as model input unless a separate metadata-aware protocol is explicitly declared.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fmow_sentinel_preflight(config: FmowPreflightConfig) -> dict[str, Path]:
    output = ensure_dir(config.output_dir)
    rows, columns, mapping = _read_rows(config.metadata_csvs)
    enriched = _enrich_rows(rows, mapping)
    country_map, map_warnings = read_country_region_map(config.country_region_map)
    mapping_stats: dict[str, int] = {}
    mapping_apply_warnings: list[str] = []
    if country_map:
        enriched, mapping_apply_warnings, mapping_stats = apply_country_region_mapping(enriched, country_map)
    filtered = _split_filter(enriched, config.filter_splits)
    subset_rows, subset_strategy = _subset_manifest(filtered, config.subset_max_per_split, config.seed)
    inventory = _update_inventory_for_country_region_map(_inventory_rows(rows, columns, mapping), enriched, config.country_region_map)
    support_rows = _support_rows(enriched, config.min_support)
    warnings = []
    warnings.extend(map_warnings)
    warnings.extend(mapping_apply_warnings)
    if "country" not in mapping and "location_id" in mapping:
        warnings.append("location_id is available, but country is missing; do not interpret location_id as country.")
    if "latitude" in mapping and "longitude" in mapping and "country" not in mapping:
        warnings.append("Coordinates are available, but country/region derivation requires external boundary resources and is not performed here.")
    if any(row.get("country") for row in enriched) and not any(row.get("continent") or row.get("un_region") for row in enriched):
        warnings.append("Country is available, but continent/UN region are missing; pass --country-region-map with a verified mapping table to derive them.")
    artifacts = {
        "metadata_inventory": output / "fmow_metadata_inventory.csv",
        "missing_fields_report": output / "fmow_missing_fields_report.md",
        "slice_support_summary": output / "fmow_slice_support_summary.csv",
        "slice_support_recommendations": output / "fmow_slice_support_recommendations.csv",
        "preflight_report": output / "fmow_preflight_report.md",
        "subset_metadata": output / "subset_metadata.csv",
        "subset_manifest": output / "subset_manifest.csv",
        "subset_support_report": output / "subset_support_report.md",
        "band_statistics_sample": output / "band_statistics_sample.csv",
        "image_shape_summary": output / "image_shape_summary.csv",
        "raster_loading_report": output / "raster_loading_report.md",
        "audit_table_schema": output / "audit_table_schema_fmow_sentinel.md",
        "warnings": output / "warnings.json",
        "run_metadata": output / "run_metadata.json",
    }
    write_csv(artifacts["metadata_inventory"], inventory)
    write_csv(artifacts["slice_support_summary"], support_rows)
    write_csv(artifacts["slice_support_recommendations"], support_rows)
    write_csv(artifacts["subset_metadata"], subset_rows)
    write_csv(artifacts["subset_manifest"], subset_rows)
    _write_missing_fields_report(artifacts["missing_fields_report"], inventory)
    _write_subset_report(artifacts["subset_support_report"], subset_rows, subset_strategy)
    raster_warnings: list[str] = []
    if config.inspect_rasters and not config.metadata_only:
        stats, shape_rows, raster_warnings = _inspect_rasters(subset_rows, config.data_root, config.raster_sample_size)
        write_csv(artifacts["band_statistics_sample"], stats)
        write_csv(artifacts["image_shape_summary"], shape_rows)
    else:
        write_csv(artifacts["band_statistics_sample"], [])
        write_csv(artifacts["image_shape_summary"], [])
    warnings.extend(raster_warnings)
    _write_raster_report(artifacts["raster_loading_report"], config.inspect_rasters and not config.metadata_only, raster_warnings)
    _write_preflight_report(artifacts["preflight_report"], support_rows, subset_strategy, warnings)
    _write_schema_report(artifacts["audit_table_schema"])
    artifacts["warnings"].write_text(json.dumps({"warnings": sorted(set(warnings))}, indent=2), encoding="utf-8")
    metadata = {
        "dataset": "fmow_sentinel",
        "task": "scene_classification",
        "input_mode": "s2_13band_image_only",
        "band_profile": "sentinel2_13band_fmow",
        "split_protocol": config.split_protocol,
        "metadata_csvs": [str(path) for path in config.metadata_csvs],
        "country_region_map": str(config.country_region_map) if config.country_region_map else "",
        "country_region_mapping_stats": mapping_stats,
        "row_count": len(rows),
        "subset_row_count": len(subset_rows),
        "subset_strategy": subset_strategy,
        "filter_splits": list(config.filter_splits),
        "metadata_only": config.metadata_only,
        "inspect_rasters": config.inspect_rasters,
    }
    artifacts["run_metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts
