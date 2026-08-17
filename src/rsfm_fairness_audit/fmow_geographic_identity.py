from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


FMOW_GEOGRAPHIC_SITE_FIELDS = ("split_original", "category", "location_id")
FMOW_GEOGRAPHIC_SITE_COUNT = 1480
FMOW_POLYGON_SPAN_LIMIT_M = 1.0
FMOW_GEOGRAPHY_FIELDS = (
    "fmow_geographic_site_id", "split_original", "category", "location_id",
    "archive_parent", "polygon_centroid_span_m",
)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def fmow_geographic_site_id(row: Mapping[str, Any]) -> str:
    values = [str(row.get(field, "") or "").strip() for field in FMOW_GEOGRAPHIC_SITE_FIELDS]
    if any(not value for value in values):
        raise ValueError(
            "fMoW geographic identity requires non-empty "
            "split_original, category, and location_id"
        )
    if any("|" in value for value in values):
        raise ValueError(f"fMoW geographic identity component contains '|': {values}")
    return "|".join(values)


def archive_parent(row: Mapping[str, Any]) -> str:
    raw = str(row.get("archive_path", "") or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("fMoW geographic identity requires original archive_path")
    parent = str(PurePosixPath(raw).parent)
    split_original = str(row.get("split_original", "") or "").strip()
    category = str(row.get("category", "") or "").strip()
    location_id = str(row.get("location_id", "") or "").strip()
    expected_suffix = f"/{split_original}/{category}/{category}_{location_id}"
    if not ("/" + parent.lstrip("/")).endswith(expected_suffix):
        raise ValueError(
            f"archive parent {parent!r} is not equivalent to the frozen geographic "
            f"identity suffix {expected_suffix!r}"
        )
    return parent


def polygon_centroid(wkt: Any) -> tuple[float, float]:
    """Return latitude, longitude from one original fMoW WKT polygon.

    fMoW-Sentinel formal tables contain simple lon/lat POLYGON WKT.  Parsing is
    deliberately strict: canonical lat/lon are never consulted as a fallback.
    """
    text = str(wkt or "").strip()
    if not text.upper().startswith("POLYGON"):
        raise ValueError("fMoW geographic identity requires simple POLYGON WKT")
    numbers = [float(value) for value in _NUMBER.findall(text)]
    if len(numbers) < 8 or len(numbers) % 2:
        raise ValueError(f"Malformed fMoW polygon WKT: {text[:120]!r}")
    points = list(zip(numbers[0::2], numbers[1::2]))
    if points[0] != points[-1]:
        points.append(points[0])
    twice_area = 0.0
    cx = 0.0
    cy = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(twice_area) < 1e-18:
        raise ValueError("Degenerate fMoW polygon has zero area")
    longitude = cx / (3.0 * twice_area)
    latitude = cy / (3.0 * twice_area)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError(f"Polygon centroid outside lon/lat bounds: {(latitude, longitude)}")
    return latitude, longitude


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371008.8 * math.asin(min(1.0, math.sqrt(value)))


def maximum_coordinate_span_m(points: Sequence[tuple[float, float]]) -> float:
    return max(
        (haversine_m(points[i], points[j]) for i in range(len(points)) for j in range(i + 1, len(points))),
        default=0.0,
    )


def mean_coordinate(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        raise ValueError("Cannot average an empty coordinate sequence")
    latitude = sum(point[0] for point in points) / len(points)
    sin_lon = sum(math.sin(math.radians(point[1])) for point in points)
    cos_lon = sum(math.cos(math.radians(point[1])) for point in points)
    longitude = math.degrees(math.atan2(sin_lon, cos_lon))
    return latitude, longitude


def validate_fmow_geographic_unit(row: Mapping[str, Any]) -> str:
    unit = str(row.get("spatial_unit", "") or "").strip()
    explicit = str(row.get("fmow_geographic_site_id", "") or "").strip()
    expected = fmow_geographic_site_id(row)
    if not unit or unit != explicit or unit != expected:
        raise ValueError(
            "Invalid fMoW geographic unit; expected spatial_unit="
            "fmow_geographic_site_id=split_original|category|location_id, "
            f"observed spatial_unit={unit!r}, explicit={explicit!r}, expected={expected!r}"
        )
    if row.get("site_id") not in (None, ""):
        raise ValueError("Legacy fMoW site_id field is forbidden in corrected spatial outputs")
    return unit
