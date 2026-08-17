from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.geographic_risk_association import DW_TO_WORLDCOVER
from rsfm_fairness_audit.geographic_risk_atlas import LATITUDE_FIELDS, LONGITUDE_FIELDS, _iter_csv
from rsfm_fairness_audit.fmow_geographic_identity import (
    FMOW_GEOGRAPHIC_SITE_COUNT,
    FMOW_GEOGRAPHY_FIELDS,
    FMOW_POLYGON_SPAN_LIMIT_M,
    validate_fmow_geographic_unit,
)
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


SCHEMA = "geobwer.geographic_covariates.official_gee.v3"
FMOW_METADATA_FIELDS = FMOW_GEOGRAPHY_FIELDS + (
    "coordinate_source", "country", "country_code", "continent", "region",
    "geography_match_count", "geography_source",
)
PROBABILITY_BANDS = (
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice",
)
WORLDCOVER_CLASSES = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}
PRODUCTS = {
    "ghsl_urbanization": {
        "asset": "JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020", "band": "smod_code",
        "epoch": 2020, "native_resolution_m": 1000,
        "url": "https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_SMOD_V2-0",
    },
    "population_density": {
        "asset": "JRC/GHSL/P2023A/GHS_POP/2020", "band": "population_count",
        "epoch": 2020, "native_resolution_m": 100,
        "native_cell_area_km2": 0.01, "sampling_scale_m": 100,
        "transform": "population_count_per_native_100m_cell / 0.01_km2",
        "url": "https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_POP",
    },
    "nightlights": {
        "asset": "NOAA/VIIRS/DNB/ANNUAL_V21", "band": "average_masked",
        "epoch": 2020, "native_resolution_m": 463.83,
        "url": "https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_ANNUAL_V21",
    },
    "dynamic_world": {
        "asset": "GOOGLE/DYNAMICWORLD/V1", "epoch": 2021,
        "native_resolution_m": 10,
        "temporal_label": "mode", "temporal_confidence": "mean_per_observation_top1_probability",
        "url": "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1",
    },
    "fmow_admin_geography": {
        "asset": "USDOS/LSIB_SIMPLE/2017",
        "country_name_property": "country_na", "country_code_property": "country_co",
        "region_property": "wld_rgn",
        "join": "exact_point_in_polygon_at_raw_polygon_centroid",
        "url": "https://developers.google.com/earth-engine/datasets/catalog/USDOS_LSIB_SIMPLE_2017",
    },
}
VARIABLES_BY_TASK = {
    "AlphaEarth": (
        "land_cover_heterogeneity", "reference_confidence", "reference_disagreement",
        "ghsl_urbanization", "population_density", "nightlights",
    ),
    "fMoW": ("ghsl_urbanization", "population_density", "nightlights"),
}


def _first(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    return next((row.get(field) for field in fields if row.get(field) not in (None, "")), None)


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_hash(task: str, source_hash: str, target_hash: str) -> str:
    payload = json.dumps(
        {"schema": SCHEMA, "task": task, "products": PRODUCTS,
         "source_hash": source_hash, "target_hash": target_hash},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coordinate(row: Mapping[str, Any]) -> tuple[float, float]:
    latitude = _float(_first(row, LATITUDE_FIELDS))
    longitude = _float(_first(row, LONGITUDE_FIELDS))
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return math.nan, math.nan
    return latitude, longitude


def build_alphaearth_sampling_rows(
    sample_csv: str | Path, risk_csv: str | Path, *, split: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use frozen sample coordinates, retaining only atlas risk units."""
    target = read_csv_rows(risk_csv)
    units = {str(row["spatial_unit"]) for row in target}
    rows, seen = [], set()
    source_columns: set[str] = set()
    for row in _iter_csv(Path(sample_csv)):
        source_columns.update(row)
        if row.get("split") not in (None, "", split):
            continue
        unit = str(row.get("spatial_block_id", ""))
        if unit not in units:
            continue
        latitude, longitude = _coordinate(row)
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            continue
        key = str(row.get("sample_id") or f"{unit}|{latitude:.8f}|{longitude:.8f}")
        if key in seen:
            continue
        seen.add(key)
        worldcover = _first(row, ("worldcover_label", "label"))
        rows.append({
            "row_id": key, "spatial_unit": unit, "latitude": latitude,
            "longitude": longitude, "source_worldcover_label": worldcover or "",
        })
    present = {str(row["spatial_unit"]) for row in rows}
    return rows, {
        "task": "AlphaEarth", "source": str(sample_csv), "risk_source": str(risk_csv),
        "source_columns": sorted(source_columns), "sampling_row_count": len(rows),
        "target_unit_count": len(units), "represented_unit_count": len(present),
        "unit_coverage": len(present) / len(units) if units else 0.0,
        "sampling_rule": "frozen_test_sample_points_grouped_by_exact_spatial_block",
    }


def build_fmow_sampling_rows(
    risk_csv: str | Path, *, expected_site_count: int | None = FMOW_GEOGRAPHIC_SITE_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use one raw-polygon-derived coordinate per audited fMoW sequence."""
    rows, invalid = [], 0
    for row in read_csv_rows(risk_csv):
        latitude, longitude = _coordinate(row)
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            invalid += 1
            continue
        unit = validate_fmow_geographic_unit(row)
        span_m = _float(row.get("polygon_centroid_span_m"))
        if not math.isfinite(span_m) or not span_m < FMOW_POLYGON_SPAN_LIMIT_M:
            raise ValueError(f"Invalid polygon centroid span for {unit}: {span_m}")
        if row.get("coordinate_source") != "original_polygon_wkt_centroid":
            raise ValueError(f"fMoW coordinates are not raw-polygon-derived for {unit}")
        rows.append({
            "row_id": unit, "spatial_unit": unit, "latitude": latitude,
            "longitude": longitude, "source_worldcover_label": "",
            **{field: row.get(field, "") for field in FMOW_METADATA_FIELDS},
        })
    if expected_site_count is not None and len(rows) != expected_site_count:
        raise ValueError(
            f"fMoW covariate sampling requires exactly {expected_site_count} audited sites; "
            f"observed {len(rows)}"
        )
    return rows, {
        "task": "fMoW", "risk_source": str(risk_csv), "sampling_row_count": len(rows),
        "target_unit_count": len(rows) + invalid, "represented_unit_count": len(rows),
        "unit_coverage": len(rows) / (len(rows) + invalid) if rows or invalid else 0.0,
        "invalid_coordinate_count": invalid,
        "sampling_rule": "one_raw_polygon_derived_centroid_per_exact_original_sequence",
        "spatial_unit_contract": "split_original|category|location_id_equivalent_to_archive_parent",
        "site_collision_count": 0,
        "maximum_polygon_centroid_span_m": max((_float(row.get("polygon_centroid_span_m")) for row in rows), default=math.nan),
    }


def _official_image_stack(ee: Any, *, include_dynamic_world: bool) -> tuple[Any, Any]:
    smod = ee.Image(PRODUCTS["ghsl_urbanization"]["asset"]).select("smod_code")
    smod = smod.updateMask(smod.gt(10)).rename("ghsl_urbanization")
    population = ee.Image(PRODUCTS["population_density"]["asset"]).select("population_count")
    density = population.divide(
        PRODUCTS["population_density"]["native_cell_area_km2"]
    ).rename("population_density")
    nightlights = (
        ee.ImageCollection(PRODUCTS["nightlights"]["asset"])
        .filterDate("2020-01-01", "2021-01-01").first()
        .select("average_masked").rename("nightlights")
    )
    stack = smod.addBands(nightlights)
    if include_dynamic_world:
        dynamic = ee.ImageCollection(PRODUCTS["dynamic_world"]["asset"]).filterDate(
            "2021-01-01", "2022-01-01"
        )
        label = dynamic.select("label").mode().rename("dynamic_world_label")

        def top_probability(image: Any) -> Any:
            return image.select(list(PROBABILITY_BANDS)).reduce(ee.Reducer.max()).rename("dw_top1")

        confidence = dynamic.map(top_probability).mean().rename("reference_confidence")
        stack = stack.addBands(label).addBands(confidence)
    # Preserve a feature when one product is masked; sentinels become blanks client-side.
    return stack.unmask(-9999), density.unmask(-9999)


def _continent_from_lsib_region(region: Any) -> str:
    value = str(region or "").strip()
    normalized = value.casefold()
    if any(token in normalized for token in ("caribbean", "central america")):
        return "North America"
    if "middle east" in normalized:
        return "Asia"
    for token, continent in (
        ("africa", "Africa"), ("asia", "Asia"), ("europe", "Europe"),
        ("north america", "North America"), ("south america", "South America"),
        ("oceania", "Oceania"), ("australia", "Oceania"),
        ("pacific", "Oceania"), ("antarctica", "Antarctica"),
    ):
        if token in normalized:
            return continent
    return ""


def _attach_fmow_admin_geography(ee: Any, collection: Any) -> Any:
    boundaries = ee.FeatureCollection(PRODUCTS["fmow_admin_geography"]["asset"])

    def attach(feature: Any) -> Any:
        matches = boundaries.filterBounds(feature.geometry())
        count = matches.size()
        first = ee.Feature(matches.first())
        return feature.set({
            "geography_match_count": count,
            "country": ee.Algorithms.If(count.eq(1), first.get("country_na"), ""),
            "country_code": ee.Algorithms.If(count.eq(1), first.get("country_co"), ""),
            "region": ee.Algorithms.If(count.eq(1), first.get("wld_rgn"), ""),
            "geography_source": PRODUCTS["fmow_admin_geography"]["asset"],
        })

    return collection.map(attach)


def extract_official_covariates_gee(
    sampling_rows: Sequence[Mapping[str, Any]], *, include_dynamic_world: bool,
    batch_size: int = 400, max_retries: int = 3, ee_module: Any | None = None,
) -> list[dict[str, Any]]:
    """Sample official GEE products in bounded, logged batches."""
    if ee_module is None:
        import ee as ee_module  # type: ignore[import-not-found]
    ee = ee_module
    stack, population_density = _official_image_stack(
        ee, include_dynamic_world=include_dynamic_world,
    )
    output: list[dict[str, Any]] = []
    total = len(sampling_rows)
    for start in range(0, total, batch_size):
        batch = sampling_rows[start:start + batch_size]
        features = [
            ee.Feature(ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])]), {
                "row_id": str(row["row_id"]), "spatial_unit": str(row["spatial_unit"]),
                "latitude": float(row["latitude"]), "longitude": float(row["longitude"]),
                "source_worldcover_label": str(row.get("source_worldcover_label", "")),
                **{field: str(row.get(field, "")) for field in FMOW_METADATA_FIELDS},
            }) for row in batch
        ]
        feature_collection = ee.FeatureCollection(features)
        is_fmow = bool(batch and batch[0].get("fmow_geographic_site_id"))
        if is_fmow:
            feature_collection = _attach_fmow_admin_geography(ee, feature_collection)
        sampled = stack.sampleRegions(
            collection=feature_collection, scale=10, geometries=False, tileScale=4,
        )
        population_sampled = population_density.sampleRegions(
            collection=feature_collection,
            scale=PRODUCTS["population_density"]["sampling_scale_m"],
            geometries=False, tileScale=4,
        )
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                payload = sampled.getInfo()
                population_payload = population_sampled.getInfo()
                break
            except Exception as exc:  # network/service errors are retried, then surfaced
                last_error = exc
                print(f"[covariates] batch {start // batch_size + 1} attempt {attempt} failed: {exc}", flush=True)
                if attempt < max_retries:
                    time.sleep(min(30, 2 ** attempt))
        else:
            raise RuntimeError(f"Earth Engine sampling failed at rows {start}:{start + len(batch)}") from last_error
        population_by_row = {
            str(feature.get("properties", {}).get("row_id")): feature.get("properties", {}).get("population_density")
            for feature in population_payload.get("features", [])
        }
        for feature in payload.get("features", []):
            item = dict(feature.get("properties", {}))
            item["population_density"] = population_by_row.get(str(item.get("row_id")), "")
            for field in ("ghsl_urbanization", "population_density", "nightlights",
                          "dynamic_world_label", "reference_confidence"):
                value = _float(item.get(field))
                item[field] = "" if not math.isfinite(value) or value <= -9990 else value
            if is_fmow:
                item["continent"] = _continent_from_lsib_region(item.get("region"))
            output.append(item)
        print(f"[covariates] sampled {min(start + len(batch), total)}/{total}", flush=True)
    return output


def aggregate_covariates(
    task: str, sampled_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sampled_rows:
        grouped[str(row["spatial_unit"])].append(row)
    output = []
    for unit, rows in sorted(grouped.items()):
        latitude = np.asarray([_float(row.get("latitude")) for row in rows], dtype=float)
        longitude = np.radians([_float(row.get("longitude")) for row in rows])
        item: dict[str, Any] = {
            "spatial_unit": unit, "latitude": float(np.nanmean(latitude)),
            "longitude": float(math.degrees(math.atan2(np.nanmean(np.sin(longitude)), np.nanmean(np.cos(longitude))))),
            "covariate_support": len(rows),
        }
        if task == "fMoW":
            for field in FMOW_METADATA_FIELDS:
                values = {str(row.get(field, "") or "") for row in rows}
                if len(values) > 1:
                    raise ValueError(f"Metadata drift for {unit}/{field}: {sorted(values)}")
                item[field] = next(iter(values), "")
            match_count = int(float(item.get("geography_match_count") or 0))
            country = str(item.get("country", "") or "").strip()
            region = str(item.get("region", "") or "").strip()
            continent = _continent_from_lsib_region(region)
            reliable = match_count == 1 and bool(country and region and continent)
            item["country"] = country if reliable else "unknown"
            item["country_code"] = str(item.get("country_code", "") or "").strip() if reliable else "unknown"
            item["region"] = region if reliable else "unknown"
            item["continent"] = continent if reliable else "unknown"
            item["admin_geography_reliable"] = int(reliable)
        for variable in ("ghsl_urbanization", "population_density", "nightlights", "reference_confidence"):
            values = [_float(row.get(variable)) for row in rows]
            clean = [value for value in values if math.isfinite(value)]
            if clean:
                # SMOD is an ordered class code, so preserve the modal official class.
                item[variable] = (
                    float(Counter(int(round(value)) for value in clean).most_common(1)[0][0])
                    if variable == "ghsl_urbanization" else float(np.mean(clean))
                )
            else:
                item[variable] = ""
            item[f"{variable}_support"] = len(clean)
        if task == "AlphaEarth":
            labels = []
            disagreements = []
            for row in rows:
                wc_value = _float(row.get("source_worldcover_label"))
                dw_value = _float(row.get("dynamic_world_label"))
                if math.isfinite(wc_value) and int(wc_value) in WORLDCOVER_CLASSES:
                    labels.append(str(int(wc_value)))
                if (math.isfinite(wc_value) and int(wc_value) in WORLDCOVER_CLASSES
                        and math.isfinite(dw_value) and 0 <= int(dw_value) <= 8):
                    disagreements.append(float(DW_TO_WORLDCOVER.get(str(int(dw_value)), "") != str(int(wc_value))))
            counts = np.asarray(list(Counter(labels).values()), dtype=float)
            if counts.size:
                probabilities = counts / counts.sum()
                item["land_cover_heterogeneity"] = float(-np.sum(probabilities * np.log(probabilities)) / np.log(11.0))
            else:
                item["land_cover_heterogeneity"] = ""
            item["reference_disagreement"] = float(np.mean(disagreements)) if disagreements else ""
            item["land_cover_heterogeneity_support"] = len(labels)
            item["reference_disagreement_support"] = len(disagreements)
        output.append(item)
    return output


def covariate_qa(task: str, rows: Sequence[Mapping[str, Any]], target_units: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    units = [str(row["spatial_unit"]) for row in rows]
    checks.append({
        "task": task, "check": "unique_spatial_unit", "status": "pass" if len(units) == len(set(units)) else "fail",
        "observed": len(set(units)), "expected": len(units),
    })
    if task == "fMoW":
        exact_geography = sum(int(float(row.get("admin_geography_reliable") or 0)) == 1 for row in rows)
        checks.extend([
            {
                "task": task, "check": "exact_key_covariate_coverage",
                "status": "pass" if len(units) == target_units and len(set(units)) == target_units else "fail",
                "observed": len(set(units)), "expected": target_units,
            },
            {
                "task": task, "check": "point_in_admin_polygon_no_ocean_or_cross_continent",
                "status": "pass" if exact_geography == len(rows) == target_units else "partial",
                "observed": exact_geography, "expected": target_units,
                "unmatched_site_count": len(rows) - exact_geography,
                "interpretation": "unmatched sites remain in global associations and are excluded only from administrative summaries",
            },
            {
                "task": task, "check": "polygon_centroid_span_below_1m",
                "status": "pass" if all(_float(row.get("polygon_centroid_span_m")) < 1.0 for row in rows) else "fail",
                "observed": max((_float(row.get("polygon_centroid_span_m")) for row in rows), default=""),
                "expected": "<1.0m",
            },
        ])
    coverage = len(set(units)) / target_units if target_units else 0.0
    checks.append({
        "task": task, "check": "target_unit_alignment", "status": "pass" if coverage >= .8 else "fail",
        "observed": coverage, "expected": ">=0.80",
    })
    bounds = {
        "land_cover_heterogeneity": (0, 1), "reference_confidence": (0, 1),
        "reference_disagreement": (0, 1), "population_density": (0, math.inf),
        "nightlights": (0, math.inf),
    }
    registered = VARIABLES_BY_TASK[task]
    for variable in registered:
        values = [_float(row.get(variable)) for row in rows]
        clean = [value for value in values if math.isfinite(value)]
        variable_coverage = len(clean) / len(rows) if rows else 0.0
        valid = True
        if variable == "ghsl_urbanization":
            valid = all(int(round(value)) in {11, 12, 13, 21, 22, 23, 30} for value in clean)
        elif variable in bounds:
            low, high = bounds[variable]
            valid = all(low <= value <= high for value in clean)
        checks.append({
            "task": task, "check": f"variable:{variable}",
            "status": "pass" if clean and valid else "unavailable",
            "observed": variable_coverage, "expected": "range_valid; association_ready_if_coverage>=0.80",
            "valid_value_count": len(clean), "minimum": min(clean) if clean else "",
            "maximum": max(clean) if clean else "",
        })
    return checks


def admin_geography_coverage_qa(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unmatched = []
    exact = 0
    for row in rows:
        reliable = int(float(row.get("admin_geography_reliable") or 0)) == 1
        if reliable:
            exact += 1
            continue
        unmatched.append({
            "spatial_unit": row.get("spatial_unit", ""),
            "fmow_geographic_site_id": row.get("fmow_geographic_site_id", ""),
            "split_original": row.get("split_original", ""),
            "category": row.get("category", ""),
            "location_id": row.get("location_id", ""),
            "latitude": row.get("latitude", ""), "longitude": row.get("longitude", ""),
            "geography_match_count": row.get("geography_match_count", ""),
            "country": row.get("country", "unknown"),
            "continent": row.get("continent", "unknown"),
            "region": row.get("region", "unknown"),
            "reason": "no_unique_reliable_LSIB_match_or_unmapped_region_taxonomy",
        })
    total = len(rows)
    return {
        "task": "fMoW", "status": "complete",
        "exact_admin_match_count": exact,
        "unmatched_admin_site_count": len(unmatched),
        "total_site_count": total,
        "admin_match_coverage": exact / total if total else 0.0,
        "global_association_site_count": total,
        "administrative_summary_eligible_site_count": exact,
        "nearest_country_fallback": "forbidden",
    }, unmatched


def _read_cache(path: Path, protocol_hash: str, expected_rows: int) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    rows = read_csv_rows(path)
    if len(rows) != expected_rows or any(row.get("covariate_protocol_hash") != protocol_hash for row in rows):
        return None
    print(f"[covariates] reusing validated cache: {path}", flush=True)
    return rows


def _write_cache(path: Path, rows: Sequence[Mapping[str, Any]], protocol_hash: str) -> None:
    write_csv(path, [{**dict(row), "covariate_protocol_hash": protocol_hash} for row in rows])


def prepare_geographic_risk_covariates(
    *, atlas_dir: str | Path, alphaearth_sample_csv: str | Path,
    output_dir: str | Path, cache_dir: str | Path, batch_size: int = 400,
    ee_module: Any | None = None,
    expected_fmow_site_count: int = FMOW_GEOGRAPHIC_SITE_COUNT,
    reuse_cache_only: bool = False,
) -> dict[str, Any]:
    atlas, output, cache = Path(atlas_dir), ensure_dir(output_dir), ensure_dir(cache_dir)
    alpha_risk = atlas / "alphaearth_spatial_unit_risk.csv"
    fmow_risks = sorted(atlas.glob("fmow_*_spatial_unit_risk.csv"))
    if not alpha_risk.is_file() or not fmow_risks:
        raise FileNotFoundError(f"Atlas lacks canonical AlphaEarth/fMoW risk tables: {atlas}")
    # All fMoW model tables must describe the same frozen location universe.
    fmow_risk = fmow_risks[0]
    alpha_sampling, alpha_alignment = build_alphaearth_sampling_rows(alphaearth_sample_csv, alpha_risk)
    fmow_sampling, fmow_alignment = build_fmow_sampling_rows(
        fmow_risk, expected_site_count=expected_fmow_site_count,
    )
    fmow_units = {row["spatial_unit"] for row in fmow_sampling}
    model_unit_audit = []
    for path in fmow_risks:
        source_rows = read_csv_rows(path)
        source_by_unit = {str(row["spatial_unit"]): row for row in source_rows}
        canonical_by_unit = {str(row["spatial_unit"]): row for row in fmow_sampling}
        units = set(source_by_unit)
        coordinates = {
            str(row["spatial_unit"]): (_float(row.get("latitude")), _float(row.get("longitude")))
            for row in source_rows
        }
        canonical_coordinates = {
            str(row["spatial_unit"]): (float(row["latitude"]), float(row["longitude"]))
            for row in fmow_sampling
        }
        coordinate_mismatches = sum(
            unit not in coordinates or any(
                not (math.isfinite(a) and math.isfinite(b)) or abs(a - b) > 1e-8
                for a, b in zip(canonical_coordinates[unit], coordinates[unit])
            )
            for unit in fmow_units
        )
        metadata_mismatches = sum(
            unit not in source_by_unit or any(
                str(source_by_unit[unit].get(field, ""))
                != str(canonical_by_unit[unit].get(field, ""))
                for field in FMOW_GEOGRAPHY_FIELDS
            )
            for unit in fmow_units
        )
        model_unit_audit.append({
            "source": str(path), "unit_count": len(units),
            "matches_canonical": units == fmow_units and coordinate_mismatches == 0 and metadata_mismatches == 0,
            "coordinate_mismatch_count": coordinate_mismatches,
            "identity_metadata_mismatch_count": metadata_mismatches,
        })
        if units != fmow_units:
            raise ValueError(f"fMoW atlas model unit mismatch: {path}")
        if coordinate_mismatches:
            raise ValueError(f"fMoW atlas model coordinate mismatch: {path}")
        if metadata_mismatches:
            raise ValueError(f"fMoW atlas model identity metadata mismatch: {path}")
    sources = {
        "AlphaEarth": (alpha_sampling, alpha_alignment, True, Path(alphaearth_sample_csv), alpha_risk),
        "fMoW": (fmow_sampling, fmow_alignment, False, fmow_risk, fmow_risk),
    }
    artifacts, qa_rows, task_manifests = {}, [], []
    admin_qa: dict[str, Any] | None = None
    unmatched_admin_sites: list[dict[str, Any]] = []
    for task, (sampling, alignment, include_dw, source, risk) in sources.items():
        source_hash, target_hash = _sha256(source), _sha256(risk)
        protocol_hash = _protocol_hash(task, source_hash, target_hash)
        cache_path = cache / f"{task.lower()}_official_gee_samples.csv"
        extracted = _read_cache(cache_path, protocol_hash, len(sampling))
        cache_status = "reused"
        if extracted is None:
            if reuse_cache_only:
                raise RuntimeError(
                    f"Cache-only mode forbids Earth Engine sampling, but no valid cache matched "
                    f"protocol_hash={protocol_hash}: {cache_path}"
                )
            extracted = extract_official_covariates_gee(
                sampling, include_dynamic_world=include_dw, batch_size=batch_size, ee_module=ee_module,
            )
            _write_cache(cache_path, extracted, protocol_hash)
            cache_status = "created"
        expected_row_ids = {str(row["row_id"]) for row in sampling}
        observed_row_ids = {str(row.get("row_id", "")) for row in extracted}
        if len(extracted) != len(sampling) or observed_row_ids != expected_row_ids:
            raise ValueError(
                f"{task} GEE extraction violated exact-key coverage: "
                f"expected={len(expected_row_ids)}, observed={len(observed_row_ids)}, "
                f"missing={sorted(expected_row_ids - observed_row_ids)[:5]}, "
                f"extra={sorted(observed_row_ids - expected_row_ids)[:5]}"
            )
        canonical = aggregate_covariates(task, extracted)
        if task == "fMoW":
            admin_qa, unmatched_admin_sites = admin_geography_coverage_qa(canonical)
        canonical_path = output / ("alphaearth_covariates.csv" if task == "AlphaEarth" else "fmow_covariates.csv")
        write_csv(canonical_path, canonical)
        task_qa = covariate_qa(task, canonical, int(alignment["target_unit_count"]))
        hard_failures = [row for row in task_qa if row["status"] == "fail"]
        if hard_failures:
            raise ValueError(f"Geographic covariate hard QA failed: {hard_failures}")
        qa_rows.extend(task_qa)
        artifacts[task] = str(canonical_path)
        task_manifests.append({
            "task": task, "status": "complete", "canonical_csv": str(canonical_path),
            "cache_csv": str(cache_path), "cache_status": cache_status,
            "protocol_hash": protocol_hash, "source_sha256": source_hash,
            "target_risk_sha256": target_hash, "alignment": alignment,
            "row_count": len(canonical), "canonical_csv_sha256": _sha256(canonical_path),
            "qa": task_qa,
        })
    qa_path = output / "covariate_qa.csv"
    write_csv(qa_path, qa_rows)
    write_csv(output / "fmow_model_unit_alignment.csv", model_unit_audit)
    if admin_qa is not None:
        write_csv(output / "fmow_admin_geography_coverage_qa.csv", [admin_qa])
        write_csv(output / "fmow_unmatched_admin_sites.csv", unmatched_admin_sites)
    manifest = {
        "schema": SCHEMA, "status": "complete", "training": False, "gpu_required": False,
        "cache_policy": "reuse_only_fail_closed" if reuse_cache_only else "reuse_then_sample_if_missing",
        "extraction_backend": "Google Earth Engine official public catalog",
        "products": PRODUCTS, "epoch_policy": "fixed_before_association_results: GHSL/VIIRS=2020; DynamicWorld=2021",
        "spatial_matching": {
            "AlphaEarth": "frozen test sample coordinates aggregated by exact spatial_block",
            "fMoW": "one original-polygon-derived centroid per exact split_original|category|location_id",
            "buffer_selection": "none", "nearest_join": "not_used_in_canonical_CSVs",
        },
        "sampling_scales_m": {
            "DynamicWorld_and_point_stack": 10,
            "GHS_POP_population_density": PRODUCTS["population_density"]["sampling_scale_m"],
        },
        "scientific_boundary": {
            "confirmatory": ["AlphaEarth:land_cover_heterogeneity", "AlphaEarth:reference_confidence",
                              "AlphaEarth:reference_disagreement", "AlphaEarth:ghsl_urbanization",
                              "fMoW:ghsl_urbanization"],
            "exploratory_exposure": ["AlphaEarth:population_density", "AlphaEarth:nightlights",
                                     "fMoW:population_density", "fMoW:nightlights"],
            "selection_policy": "no outcome-informed variable, product, year, radius, or transformation selection",
            "reference_semantics": "Dynamic World confidence/disagreement are reference-map ambiguity diagnostics, not ground truth quality",
        },
        "fmow_spatial_unit_contract": (
            "split_original|category|location_id equivalent to raw archive parent; "
            "location_id, category|location_id, and canonical lat/lon predecessors are invalid"
        ),
        "fmow_expected_site_count": expected_fmow_site_count,
        "fmow_admin_geography_coverage": admin_qa,
        "fmow_unmatched_admin_site_count": len(unmatched_admin_sites),
        "admin_summary_policy": "reliable_exact_matches_only; unmatched retained in global associations",
        "tasks": task_manifests, "artifacts": artifacts,
        "qa_csv": str(qa_path), "fmow_model_unit_alignment": model_unit_audit,
    }
    (output / "geographic_covariate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return manifest
