"""Minimal GEE diagnostics for AlphaEarth / WorldCover pilot access.

Run in Colab. This script intentionally avoids long Drive exports and large
region sampling. It checks:

1. Earth Engine authentication/project access.
2. AlphaEarth collection availability for 2021.
3. AlphaEarth band names.
4. Single-point AlphaEarth sampling.
5. Single-point ESA WorldCover sampling.
6. Tiny Drive export of one feature without imagery.
"""

from __future__ import annotations


YEAR = 2021
POINT = [-73.9857, 40.7484]  # NYC, fixed smoke point
EXPORT_FOLDER = "rsfm_fairness_audit_alphaearth_pilot_v1"


def try_step(name, fn):
    print(f"\n=== {name} ===")
    try:
        value = fn()
        print(value)
        return value
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    import ee

    try_step("Authenticate / initialize", lambda: (ee.Authenticate(), ee.Initialize(project=None), "initialized")[-1])

    collection_id = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    alpha_ic = ee.ImageCollection(collection_id).filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
    try_step("AlphaEarth 2021 collection size", lambda: alpha_ic.size().getInfo())
    try_step("AlphaEarth first image properties", lambda: alpha_ic.first().toDictionary().getInfo())
    try_step("AlphaEarth first image band names", lambda: alpha_ic.first().bandNames().getInfo())

    point = ee.Geometry.Point(POINT)
    alpha_at_point = alpha_ic.filterBounds(point)
    try_step("AlphaEarth point-filtered collection size", lambda: alpha_at_point.size().getInfo())
    try_step("AlphaEarth point-filtered first image properties", lambda: alpha_at_point.first().toDictionary().getInfo())
    alpha_img = alpha_at_point.mosaic()
    try_step(
        "AlphaEarth single point sample, point-filtered mosaic",
        lambda: alpha_img.sample(region=point, scale=250, numPixels=1, geometries=False).first().toDictionary().getInfo(),
    )

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover_label")
    try_step(
        "WorldCover single point sample",
        lambda: worldcover.sample(region=point, scale=10, numPixels=1, geometries=False).first().toDictionary().getInfo(),
    )

    # If the two samples above work, test a tiny table export that should finish
    # quickly. This rules out Drive export permission issues.
    tiny = ee.FeatureCollection([ee.Feature(point, {"sample_id": "tiny_export_probe", "year": YEAR})])
    task = ee.batch.Export.table.toDrive(
        collection=tiny,
        description="alphaearth_tiny_export_probe",
        folder=EXPORT_FOLDER,
        fileNamePrefix="alphaearth_tiny_export_probe",
        fileFormat="CSV",
    )
    task.start()
    print("\n=== Tiny Drive export task started ===")
    print(task.status())


if __name__ == "__main__":
    main()
