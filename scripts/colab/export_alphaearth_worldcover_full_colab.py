"""GEE/Colab scaffold for the AlphaEarth formal land-cover audit export.

This script starts sharded table exports only. It does not train models and it
does not export image chips. The local full audit expects either a merged CSV:

    outputs/alphaearth_gee_full_v1/alphaearth_worldcover_full_export.csv

or a shard manifest:

    outputs/alphaearth_gee_full_v1/alphaearth_worldcover_full_export_manifest.csv

The default shard size is conservative. Increase countries/classes only after
one shard completes and passes the local schema checker.
"""

from __future__ import annotations


PROJECT_ROOT = "/content/drive/MyDrive/rsfm_fairness_audit"
EE_PROJECT = "rsfm-fairness-audit"
EXPORT_FOLDER = "rsfm_fairness_audit_alphaearth_full_v1"
YEAR = 2021
SAMPLES_PER_CLASS_PER_COUNTRY = 100
PILOT_FIRST_N_SHARDS = None  # keep None for formal mode; set to e.g. 2 only for quota debugging.
PILOT_SCALE_M = 250
SEED = 42
INCLUDE_DYNAMIC_WORLD = False
WAIT_AND_CANCEL_AFTER_MINUTES = 360
POLL_SECONDS = 180

COUNTRIES = [
    ("United States", "USA", "North America", "High income"),
    ("Brazil", "BRA", "Latin America", "Upper middle income"),
    ("India", "IND", "South Asia", "Lower middle income"),
    ("South Africa", "ZAF", "Sub-Saharan Africa", "Upper middle income"),
    ("Australia", "AUS", "East Asia & Pacific", "High income"),
    ("France", "FRA", "Europe & Central Asia", "High income"),
    ("Indonesia", "IDN", "East Asia & Pacific", "Upper middle income"),
    ("Mexico", "MEX", "Latin America", "Upper middle income"),
]

CLASS_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
CLASS_NAMES = {
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


def main() -> None:
    import time
    import ee

    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

    embedding_bands = [f"A{i:02d}" for i in range(64)]
    alphaearth_ic = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover_label")
    countries_fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    worldcover_names = ee.Dictionary(CLASS_NAMES)

    def stack_for_region(region):
        image = alphaearth_ic.filterBounds(region).mosaic().select(embedding_bands).addBands(worldcover)
        if INCLUDE_DYNAMIC_WORLD:
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

    def enrich(feature, iso3, region_name, income_group):
        coords = feature.geometry().coordinates()
        lon = ee.Number(coords.get(0))
        lat = ee.Number(coords.get(1))
        label = ee.Number(feature.get("worldcover_label")).format()
        block = lon.multiply(2).floor().divide(2).format().cat("_").cat(lat.multiply(2).floor().divide(2).format())
        random = ee.Number(feature.get("split_random"))
        split = ee.Algorithms.If(random.gte(0.85), "test", ee.Algorithms.If(random.gte(0.70), "calibration", "train"))
        built_proxy = ee.Algorithms.If(ee.Number(feature.get("worldcover_label")).eq(50), "built_proxy", "non_built_proxy")
        return (
            feature.setGeometry(None)
            .set("sample_id", ee.String(iso3).cat("_").cat(label).cat("_").cat(ee.Number(feature.id()).format()))
            .set("lon", lon)
            .set("lat", lat)
            .set("year", YEAR)
            .set("country_iso3", iso3)
            .set("region", region_name)
            .set("income_group", income_group)
            .set("biome_or_ecoregion", "")
            .set("urban_rural_or_built_proxy", built_proxy)
            .set("spatial_block_id", block)
            .set("split", split)
            .set("worldcover_class_name", worldcover_names.get(label))
        )

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
    if not INCLUDE_DYNAMIC_WORLD:
        selectors = [item for item in selectors if item not in {"dynamic_world_label", "dynamic_world_confidence"}]

    tasks = []
    countries_to_export = COUNTRIES[:PILOT_FIRST_N_SHARDS] if PILOT_FIRST_N_SHARDS else COUNTRIES
    for shard_index, (country_name, iso3, region_name, income_group) in enumerate(countries_to_export, start=1):
        country_geom = countries_fc.filter(ee.Filter.eq("country_na", country_name)).geometry()
        sample = stack_for_region(country_geom).stratifiedSample(
            numPoints=SAMPLES_PER_CLASS_PER_COUNTRY,
            classBand="worldcover_label",
            region=country_geom,
            scale=PILOT_SCALE_M,
            seed=SEED + shard_index,
            geometries=True,
            classValues=CLASS_VALUES,
            classPoints=[SAMPLES_PER_CLASS_PER_COUNTRY] * len(CLASS_VALUES),
            tileScale=8,
        ).randomColumn("split_random", seed=SEED + shard_index)
        enriched = sample.map(lambda feature, iso3=iso3, region_name=region_name, income_group=income_group: enrich(feature, iso3, region_name, income_group))
        description = f"alphaearth_worldcover_full_{YEAR}_{iso3}_shard"
        task = ee.batch.Export.table.toDrive(
            collection=enriched,
            description=description,
            folder=EXPORT_FOLDER,
            fileNamePrefix=description,
            fileFormat="CSV",
            selectors=selectors,
        )
        task.start()
        tasks.append((iso3, task))
        print(f"Started shard {shard_index}/{len(countries_to_export)}: {iso3} -> {description}")

    deadline = time.time() + WAIT_AND_CANCEL_AFTER_MINUTES * 60
    while WAIT_AND_CANCEL_AFTER_MINUTES:
        states = {iso3: task.status().get("state", "UNKNOWN") for iso3, task in tasks}
        print(states)
        if all(state in {"COMPLETED", "FAILED", "CANCELLED"} for state in states.values()):
            for iso3, task in tasks:
                print(iso3, task.status())
            break
        if time.time() >= deadline:
            print(f"At least one shard exceeded {WAIT_AND_CANCEL_AFTER_MINUTES} minutes; cancelling unfinished shards.")
            for iso3, task in tasks:
                if task.status().get("state") not in {"COMPLETED", "FAILED", "CANCELLED"}:
                    task.cancel()
                    print(iso3, task.status())
            break
        time.sleep(POLL_SECONDS)

    print(f"Export folder: {EXPORT_FOLDER}")
    print(f"Copy completed CSV shards into {PROJECT_ROOT}/outputs/alphaearth_gee_full_v1/")
    print("Then create alphaearth_worldcover_full_export_manifest.csv with columns shard_id,path,status.")


if __name__ == "__main__":
    main()
