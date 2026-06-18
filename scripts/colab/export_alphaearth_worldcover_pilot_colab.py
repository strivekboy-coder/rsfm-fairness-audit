"""Colab/GEE scaffold for AlphaEarth + ESA WorldCover pilot table export.

Run this in Google Colab after authenticating Earth Engine. It exports a small
table first, not image chips. The local audit script expects the exported CSV
to be copied to:

    outputs/alphaearth_gee_pilot_v1/alphaearth_worldcover_pilot_export.csv

This scaffold is intentionally conservative. By default it exports a fixed
point smoke table with WorldCover labels and AlphaEarth embedding bands A00..A63.
Avoid using large-region stratified sampling until the fixed-point export works.
Dynamic World is disabled by default because annual probability compositing can
make the smoke export much slower. It does not train models.
"""

from __future__ import annotations


PROJECT_ROOT = "/content/drive/MyDrive/rsfm_fairness_audit"
EXPORT_FOLDER = "rsfm_fairness_audit_alphaearth_pilot_v1"
EXPORT_DESCRIPTION = "alphaearth_worldcover_pilot_export_2021_v1"
EXPORT_FILE_PREFIX = "alphaearth_worldcover_pilot_export"
YEAR = 2021
SAMPLES_PER_CLASS = 15
PILOT_SCALE_M = 250
SEED = 42
INCLUDE_DYNAMIC_WORLD = False
EXPORT_MODE = "fixed_points_smoke"  # fixed_points_smoke first; stratified_country_pilot later.
WAIT_AND_CANCEL_AFTER_MINUTES = 20
POLL_SECONDS = 120
PILOT_COUNTRIES = [
    ("US", "USA"),
    ("BR", "BRA"),
    ("IN", "IND"),
]
SMOKE_POINTS = [
    # sample_id, lon, lat, ISO3, coarse region, built proxy hint
    ("usa_urban_nyc", -73.9857, 40.7484, "USA", "North America", "built_proxy"),
    ("usa_cropland_iowa", -93.6250, 42.0320, "USA", "North America", "non_built_proxy"),
    ("usa_forest_oregon", -121.7000, 44.0000, "USA", "North America", "non_built_proxy"),
    ("usa_water_michigan", -86.5000, 44.5000, "USA", "North America", "non_built_proxy"),
    ("bra_urban_sao_paulo", -46.6333, -23.5505, "BRA", "Latin America", "built_proxy"),
    ("bra_forest_amazon", -60.0250, -3.4653, "BRA", "Latin America", "non_built_proxy"),
    ("bra_cropland_mato_grosso", -55.0000, -13.0000, "BRA", "Latin America", "non_built_proxy"),
    ("bra_water_amazon", -58.5000, -3.2000, "BRA", "Latin America", "non_built_proxy"),
    ("ind_urban_delhi", 77.2090, 28.6139, "IND", "South Asia", "built_proxy"),
    ("ind_cropland_punjab", 75.8500, 30.9000, "IND", "South Asia", "non_built_proxy"),
    ("ind_forest_western_ghats", 75.5000, 12.3000, "IND", "South Asia", "non_built_proxy"),
    ("ind_water_chilika", 85.3500, 19.7000, "IND", "South Asia", "non_built_proxy"),
]


def main() -> None:
    import ee

    ee.Authenticate()
    ee.Initialize(project=None)

    embedding_bands = [f"A{i:02d}" for i in range(64)]
    alphaearth_ic = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover_label")

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

    def add_common_fields(feature: ee.Feature) -> ee.Feature:
        label = ee.Number(feature.get("worldcover_label")).format()
        lon = ee.Number(feature.get("lon"))
        lat = ee.Number(feature.get("lat"))
        block = lon.floor().format().cat("_").cat(lat.floor().format())
        built_proxy = ee.Algorithms.If(ee.Number(feature.get("worldcover_label")).eq(50), "built_proxy", feature.get("urban_rural_or_built_proxy"))
        return (
            feature.set("year", YEAR)
            .set("spatial_block_id", block)
            .set("worldcover_class_name", worldcover_names.get(label))
            .set("urban_rural_or_built_proxy", built_proxy)
        )

    def stack_for_region(region: ee.Geometry) -> ee.Image:
        alphaearth = alphaearth_ic.filterBounds(region).mosaic().select(embedding_bands)
        image = alphaearth.addBands(worldcover)
        if INCLUDE_DYNAMIC_WORLD:
            # Optional diagnostic source. Dynamic World labels/probabilities are
            # not human truth; they only provide confidence/agreement diagnostics.
            dynamic_world = (
                ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
                .filterBounds(region)
                .select(["label", "water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"])
            )
            dw_label = dynamic_world.select("label").mode().rename("dynamic_world_label")
            dw_confidence = dynamic_world.select(["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]).max().reduce(ee.Reducer.max()).rename("dynamic_world_confidence")
            image = image.addBands(dw_label).addBands(dw_confidence)
        return image

    def fixed_point_collection() -> ee.FeatureCollection:
        features = []
        for index, (sample_id, lon, lat, iso3, region, built_hint) in enumerate(SMOKE_POINTS):
            split = "test" if index % 5 == 0 else "train"
            features.append(
                ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {
                        "sample_id": sample_id,
                        "lon": lon,
                        "lat": lat,
                        "country_iso3": iso3,
                        "region": region,
                        "income_group": "",
                        "biome_or_ecoregion": "",
                        "urban_rural_or_built_proxy": built_hint,
                        "split": split,
                    },
                )
            )
        points = ee.FeatureCollection(features)
        sampled = stack_for_region(points.geometry()).sampleRegions(
            collection=points,
            properties=[
                "sample_id",
                "lon",
                "lat",
                "country_iso3",
                "region",
                "income_group",
                "biome_or_ecoregion",
                "urban_rural_or_built_proxy",
                "split",
            ],
            scale=PILOT_SCALE_M,
            geometries=False,
            tileScale=1,
        )
        return sampled.filter(ee.Filter.notNull(["A00", "worldcover_label"])).map(add_common_fields)

    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    class_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

    def enrich(feature: ee.Feature, iso3: str) -> ee.Feature:
        geom = feature.geometry()
        coords = geom.coordinates()
        iso3_value = ee.String(iso3)
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
            .set("country_iso3", iso3_value)
            .set("region", "")
            .set("income_group", "")
            .set("biome_or_ecoregion", "")
            .set("urban_rural_or_built_proxy", built_proxy)
            .set("spatial_block_id", block)
            .set("split", split)
            .set("worldcover_class_name", worldcover_names.get(label))
        )

    def sample_country(iso2: str, iso3: str) -> ee.FeatureCollection:
        country = countries.filter(ee.Filter.eq("country_co", iso2)).geometry()
        samples = stack_for_region(country).stratifiedSample(
            numPoints=SAMPLES_PER_CLASS,
            classBand="worldcover_label",
            region=country,
            scale=PILOT_SCALE_M,
            seed=SEED,
            geometries=True,
            classValues=class_values,
            classPoints=[SAMPLES_PER_CLASS] * len(class_values),
            tileScale=4,
        ).randomColumn("split_random", seed=SEED)
        return samples.map(lambda feature: enrich(feature, iso3))

    if EXPORT_MODE == "fixed_points_smoke":
        enriched = fixed_point_collection()
    elif EXPORT_MODE == "stratified_country_pilot":
        enriched = ee.FeatureCollection([])
        for iso2, iso3 in PILOT_COUNTRIES:
            enriched = enriched.merge(sample_country(iso2, iso3))
    else:
        raise ValueError(f"Unknown EXPORT_MODE={EXPORT_MODE!r}")
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
        *embedding_bands,
    ]
    if INCLUDE_DYNAMIC_WORLD:
        selectors.insert(13, "dynamic_world_label")
        selectors.insert(14, "dynamic_world_confidence")
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
    print(f"Export mode: {EXPORT_MODE}")
    print(f"Drive folder: {EXPORT_FOLDER}")
    print(f"Expected CSV prefix: {EXPORT_FILE_PREFIX}")
    print(f"After export, copy CSV to {PROJECT_ROOT}/outputs/alphaearth_gee_pilot_v1/alphaearth_worldcover_pilot_export.csv")
    if WAIT_AND_CANCEL_AFTER_MINUTES:
        import time

        deadline = time.time() + WAIT_AND_CANCEL_AFTER_MINUTES * 60
        while True:
            status = task.status()
            state = status.get("state", "UNKNOWN")
            print(f"GEE task state: {state}")
            if state in {"COMPLETED", "FAILED", "CANCELLED"}:
                print(status)
                break
            if time.time() >= deadline:
                print(f"Task exceeded {WAIT_AND_CANCEL_AFTER_MINUTES} minutes; cancelling.")
                task.cancel()
                print(task.status())
                break
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
