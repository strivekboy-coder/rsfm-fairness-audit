from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence


COUNTRY_FIELD_SYNONYMS = {
    "country": ("country", "country_name", "iso_country", "name"),
    "continent": ("continent", "continent_name"),
    "un_region": ("un_region", "unregion", "world_region", "un_geoscheme_region"),
    "region": ("region", "admin_region", "geo_region", "subregion", "un_subregion"),
}


def _norm_column(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _norm_country(value: Any) -> str:
    return str(value or "").strip().casefold()


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).strip().lower() in {"nan", "none", "null"}


def _canonical_columns(columns: Sequence[str]) -> dict[str, str]:
    normalized = {_norm_column(column): column for column in columns}
    mapping: dict[str, str] = {}
    for canonical, synonyms in COUNTRY_FIELD_SYNONYMS.items():
        for synonym in synonyms:
            if _norm_column(synonym) in normalized:
                mapping[canonical] = normalized[_norm_column(synonym)]
                break
    return mapping


def read_country_region_map(path: str | Path | None) -> tuple[dict[str, dict[str, str]], list[str]]:
    if path is None:
        return {}, []
    warnings: list[str] = []
    country_map: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        mapping = _canonical_columns(columns)
        if "country" not in mapping:
            warnings.append("Country-region map has no country column; expected country/country_name/iso_country/name.")
            return {}, warnings
        for index, row in enumerate(reader, start=2):
            country = str(row.get(mapping["country"], "") or "").strip()
            if not country:
                warnings.append(f"Country-region map row {index} has empty country; row skipped.")
                continue
            key = _norm_country(country)
            values = {"country": country}
            for field in ("continent", "un_region", "region"):
                column = mapping.get(field)
                values[field] = str(row.get(column, "") or "").strip() if column else ""
            if key in country_map:
                warnings.append(f"Country-region map contains duplicate country '{country}'; first mapping was kept.")
                continue
            country_map[key] = values
    return country_map, warnings


def apply_country_region_mapping(
    rows: Sequence[Mapping[str, Any]],
    country_map: Mapping[str, Mapping[str, str]],
    provenance_label: str = "country_region_map",
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    warnings: list[str] = []
    stats = {
        "rows_with_country": 0,
        "mapped_rows": 0,
        "unmapped_country_rows": 0,
        "filled_continent": 0,
        "filled_un_region": 0,
        "filled_region": 0,
    }
    output: list[dict[str, Any]] = []
    missing_countries: set[str] = set()
    for row in rows:
        item = dict(row)
        for field in ("continent", "un_region", "region"):
            item.setdefault(field, "")
            item.setdefault(f"{field}_provenance", "")
        country = str(item.get("country", "") or "").strip()
        if not country:
            output.append(item)
            continue
        stats["rows_with_country"] += 1
        mapped = country_map.get(_norm_country(country))
        if not mapped:
            stats["unmapped_country_rows"] += 1
            missing_countries.add(country)
            output.append(item)
            continue
        stats["mapped_rows"] += 1
        for field in ("continent", "un_region", "region"):
            mapped_value = str(mapped.get(field, "") or "").strip()
            if mapped_value and _is_missing(item.get(field)):
                item[field] = mapped_value
                item[f"{field}_provenance"] = provenance_label
                stats[f"filled_{field}"] += 1
        output.append(item)
    if missing_countries:
        preview = "; ".join(sorted(missing_countries)[:10])
        suffix = "..." if len(missing_countries) > 10 else ""
        warnings.append(f"Country-region map did not cover {len(missing_countries)} countries: {preview}{suffix}")
    return output, warnings, stats
