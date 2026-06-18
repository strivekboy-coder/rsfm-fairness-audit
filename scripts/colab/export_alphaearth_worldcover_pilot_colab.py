"""Colab/GEE scaffold for AlphaEarth + ESA WorldCover pilot table export.

Run this in Google Colab after authenticating Earth Engine. It exports a small
table first, not image chips. The local audit script expects the exported CSV
to be copied to:

    outputs/alphaearth_gee_pilot_v1/alphaearth_worldcover_pilot_export.csv

This scaffold is intentionally conservative: it samples points, attaches
WorldCover labels, AlphaEarth embedding bands A00..A63, and optional Dynamic
World agreement/confidence fields if available. It does not train models.
"""

from __future__ import annotations


PROJECT_ROOT = "/content/drive/MyDrive/rsfm_fairness_audit"
EXPORT_FOLDER = "rsfm_fairness_audit_alphaearth_pilot_v1"
EXPORT_DESCRIPTION = "alphaearth_worldcover_pilot_export_2021_v1"
EXPORT_FILE_PREFIX = "alphaearth_worldcover_pilot_export"
YEAR = 2021
SAMPLES_PER_CLASS = 80
PILOT_SCALE_M = 100
SEED = 42


def main() -> None:
    import ee

    ee.Authenticate()
    ee.Initialize(project=None)

    embedding_bands = [f"A{i:02d}" for i in range(64)]
    alphaearth = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
        .mosaic()
        .select(embedding_bands)
    )
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover_label")

    # Optional diagnostic source. Dynamic World labels/probabilities are not
    # human truth; they only provide confidence/agreement diagnostics.
    dynamic_world = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
        .select(["label", "water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"])
    )
    dw_label = dynamic_world.select("label").mode().rename("dynamic_world_label")
    dw_confidence = dynamic_world.select(["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]).max().reduce(ee.Reducer.max()).rename("dynamic_world_confidence")

    # Use a small, diverse set of pilot countries. Replace this with a
    # project-specific ISO3 country polygon asset for a full run.
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
        ee.Filter.inList("country_co", ["US", "BR", "IN", "ZA", "AU", "FR", "ID", "MX"])
    )
    region = countries.geometry()

    stacked = alphaearth.addBands(worldcover).addBands(dw_label).addBands(dw_confidence)

    class_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    samples = stacked.stratifiedSample(
        numPoints=SAMPLES_PER_CLASS,
        classBand="worldcover_label",
        region=region,
        scale=PILOT_SCALE_M,
        seed=SEED,
        geometries=True,
        classValues=class_values,
        classPoints=[SAMPLES_PER_CLASS] * len(class_values),
    ).randomColumn("split_random", seed=SEED)

    worldcover_names = ee.Dictionary(
        {
            "10": "Tree cover",
            "20": "Shrubland",
            "30": "Grassland",
            "40": "Cropland",
            "50": "Built-up",
            "60": "Bare/sparse vegetation",
            "70": "Snow and ice",
            "80": "Permanent water bodies",
            "90": "Herbaceous wetland",
            "95": "Mangroves",
            "100": "Moss and lichen",
        }
    )
    iso2_to_iso3 = ee.Dictionary({"US": "USA", "BR": "BRA", "IN": "IND", "ZA": "ZAF", "AU": "AUS", "FR": "FRA", "ID": "IDN", "MX": "MEX"})

    # Country properties depend on the chosen boundary source. LSIB provides
    # two-letter codes. For full paper-grade runs, replace with an ISO3 country
    # polygon asset or join ISO3 after export.
    country_join = ee.Join.saveFirst("country_feature").apply(
        primary=samples,
        secondary=countries,
        condition=ee.Filter.intersects(leftField=".geo", rightField=".geo"),
    )

    def enrich(feature: ee.Feature) -> ee.Feature:
        geom = feature.geometry()
        coords = geom.coordinates()
        country_feature = ee.Feature(feature.get("country_feature"))
        country_code = ee.String(ee.Algorithms.If(country_feature, country_feature.get("country_co"), ""))
        country_iso3 = ee.String(iso2_to_iso3.get(country_code, country_code))
        label = ee.Number(feature.get("worldcover_label")).format()
        # Spatial block at 1-degree grid for pilot leakage-aware splitting.
        lon = ee.Number(coords.get(0))
        lat = ee.Number(coords.get(1))
        block_lon = lon.floor()
        block_lat = lat.floor()
        block = block_lon.format().cat("_").cat(block_lat.format())
        split = ee.Algorithms.If(ee.Number(feature.get("split_random")).gte(0.8), "test", "train")
        built_proxy = ee.Algorithms.If(ee.Number(feature.get("worldcover_label")).eq(50), "built_proxy", "non_built_proxy")
        return (
            feature.setGeometry(None)
            .set("sample_id", ee.String("ae_wc_").cat(ee.Number(feature.id()).format()))
            .set("lon", lon)
            .set("lat", lat)
            .set("year", YEAR)
            .set("country_iso3", country_iso3)
            .set("region", "")
            .set("income_group", "")
            .set("biome_or_ecoregion", "")
            .set("urban_rural_or_built_proxy", built_proxy)
            .set("spatial_block_id", block)
            .set("split", split)
            .set("worldcover_class_name", worldcover_names.get(label))
        )

    enriched = ee.FeatureCollection(country_join).map(enrich)
    selectors = [
        "sample_id",
        "lon",
        "lat",
        "year",
        "country_iso3",
        "region",
        "income_group",
        "biome_or_ecoregion",
        "urban_rural_or_built_proxy",
        "spatial_block_id",
        "split",
        "worldcover_label",
        "worldcover_class_name",
        "dynamic_world_label",
        "dynamic_world_confidence",
        *embedding_bands,
    ]
    task = ee.batch.Export.table.toDrive(
        collection=enriched,
        description=EXPORT_DESCRIPTION,
        folder=EXPORT_FOLDER,
        fileNamePrefix=EXPORT_FILE_PREFIX,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    print("Started Earth Engine export task.")
    print(f"Drive folder: {EXPORT_FOLDER}")
    print(f"Expected CSV prefix: {EXPORT_FILE_PREFIX}")
    print(f"After export, copy CSV to {PROJECT_ROOT}/outputs/alphaearth_gee_pilot_v1/alphaearth_worldcover_pilot_export.csv")


if __name__ == "__main__":
    main()
