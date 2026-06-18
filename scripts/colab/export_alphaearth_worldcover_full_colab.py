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
    ("US", "USA", "North America", "High income"),
    ("BR", "BRA", "Latin America", "Upper middle income"),
    ("IN", "IND", "South Asia", "Lower middle income"),
    ("ZA", "ZAF", "Sub-Saharan Africa", "Upper middle income"),
    ("AU", "AUS", "East Asia & Pacific", "High income"),
    ("FR", "FRA", "Europe & Central Asia", "High income"),
    ("ID", "IDN", "East Asia & Pacific", "Upper middle income"),
    ("MX", "MEX", "Latin America", "Upper middle income"),
    ("EG", "EGY", "Middle East & North Africa", "Lower middle income"),
    ("NG", "NGA", "Sub-Saharan Africa", "Lower middle income"),
    ("CA", "CAN", "North America", "High income"),
    ("AR", "ARG", "Latin America", "Upper middle income"),
    ("CL", "CHL", "Latin America", "High income"),
    ("CO", "COL", "Latin America", "Upper middle income"),
    ("PE", "PER", "Latin America", "Upper middle income"),
    ("VE", "VEN", "Latin America", "Not classified"),
    ("BO", "BOL", "Latin America", "Lower middle income"),
    ("PY", "PRY", "Latin America", "Upper middle income"),
    ("UY", "URY", "Latin America", "High income"),
    ("GT", "GTM", "Latin America", "Upper middle income"),
    ("CU", "CUB", "Latin America", "Upper middle income"),
    ("DO", "DOM", "Latin America", "Upper middle income"),
    ("GB", "GBR", "Europe & Central Asia", "High income"),
    ("DE", "DEU", "Europe & Central Asia", "High income"),
    ("ES", "ESP", "Europe & Central Asia", "High income"),
    ("IT", "ITA", "Europe & Central Asia", "High income"),
    ("PL", "POL", "Europe & Central Asia", "High income"),
    ("SE", "SWE", "Europe & Central Asia", "High income"),
    ("NO", "NOR", "Europe & Central Asia", "High income"),
    ("FI", "FIN", "Europe & Central Asia", "High income"),
    ("TR", "TUR", "Europe & Central Asia", "Upper middle income"),
    ("UA", "UKR", "Europe & Central Asia", "Lower middle income"),
    ("RO", "ROU", "Europe & Central Asia", "High income"),
    ("GR", "GRC", "Europe & Central Asia", "High income"),
    ("PT", "PRT", "Europe & Central Asia", "High income"),
    ("NL", "NLD", "Europe & Central Asia", "High income"),
    ("BE", "BEL", "Europe & Central Asia", "High income"),
    ("CH", "CHE", "Europe & Central Asia", "High income"),
    ("AT", "AUT", "Europe & Central Asia", "High income"),
    ("CZ", "CZE", "Europe & Central Asia", "High income"),
    ("RU", "RUS", "Europe & Central Asia", "Upper middle income"),
    ("CN", "CHN", "East Asia & Pacific", "Upper middle income"),
    ("JP", "JPN", "East Asia & Pacific", "High income"),
    ("KR", "KOR", "East Asia & Pacific", "High income"),
    ("TH", "THA", "East Asia & Pacific", "Upper middle income"),
    ("VN", "VNM", "East Asia & Pacific", "Lower middle income"),
    ("PH", "PHL", "East Asia & Pacific", "Lower middle income"),
    ("MY", "MYS", "East Asia & Pacific", "Upper middle income"),
    ("MM", "MMR", "East Asia & Pacific", "Lower middle income"),
    ("KH", "KHM", "East Asia & Pacific", "Lower middle income"),
    ("LA", "LAO", "East Asia & Pacific", "Lower middle income"),
    ("NZ", "NZL", "East Asia & Pacific", "High income"),
    ("PG", "PNG", "East Asia & Pacific", "Lower middle income"),
    ("PK", "PAK", "South Asia", "Lower middle income"),
    ("BD", "BGD", "South Asia", "Lower middle income"),
    ("NP", "NPL", "South Asia", "Lower middle income"),
    ("LK", "LKA", "South Asia", "Lower middle income"),
    ("AF", "AFG", "South Asia", "Low income"),
    ("IR", "IRN", "Middle East & North Africa", "Lower middle income"),
    ("IQ", "IRQ", "Middle East & North Africa", "Upper middle income"),
    ("SA", "SAU", "Middle East & North Africa", "High income"),
    ("AE", "ARE", "Middle East & North Africa", "High income"),
    ("IL", "ISR", "Middle East & North Africa", "High income"),
    ("JO", "JOR", "Middle East & North Africa", "Lower middle income"),
    ("MA", "MAR", "Middle East & North Africa", "Lower middle income"),
    ("DZ", "DZA", "Middle East & North Africa", "Lower middle income"),
    ("TN", "TUN", "Middle East & North Africa", "Lower middle income"),
    ("LY", "LBY", "Middle East & North Africa", "Upper middle income"),
    ("ET", "ETH", "Sub-Saharan Africa", "Low income"),
    ("KE", "KEN", "Sub-Saharan Africa", "Lower middle income"),
    ("TZ", "TZA", "Sub-Saharan Africa", "Lower middle income"),
    ("UG", "UGA", "Sub-Saharan Africa", "Low income"),
    ("GH", "GHA", "Sub-Saharan Africa", "Lower middle income"),
    ("CI", "CIV", "Sub-Saharan Africa", "Lower middle income"),
    ("SN", "SEN", "Sub-Saharan Africa", "Lower middle income"),
    ("CM", "CMR", "Sub-Saharan Africa", "Lower middle income"),
    ("CD", "COD", "Sub-Saharan Africa", "Low income"),
    ("AO", "AGO", "Sub-Saharan Africa", "Lower middle income"),
    ("ZM", "ZMB", "Sub-Saharan Africa", "Lower middle income"),
    ("ZW", "ZWE", "Sub-Saharan Africa", "Lower middle income"),
    ("MZ", "MOZ", "Sub-Saharan Africa", "Low income"),
    ("MG", "MDG", "Sub-Saharan Africa", "Low income"),
    ("ML", "MLI", "Sub-Saharan Africa", "Low income"),
    ("NE", "NER", "Sub-Saharan Africa", "Low income"),
    ("SD", "SDN", "Sub-Saharan Africa", "Low income"),
    ("SS", "SSD", "Sub-Saharan Africa", "Low income"),
    ("BF", "BFA", "Sub-Saharan Africa", "Low income"),
    ("MW", "MWI", "Sub-Saharan Africa", "Low income"),
    ("KZ", "KAZ", "Europe & Central Asia", "Upper middle income"),
    ("UZ", "UZB", "Europe & Central Asia", "Lower middle income"),
    ("MN", "MNG", "East Asia & Pacific", "Lower middle income"),
    ("AZ", "AZE", "Europe & Central Asia", "Upper middle income"),
    ("GE", "GEO", "Europe & Central Asia", "Upper middle income"),
    ("AM", "ARM", "Europe & Central Asia", "Upper middle income"),
    ("KG", "KGZ", "Europe & Central Asia", "Lower middle income"),
    ("TJ", "TJK", "Europe & Central Asia", "Lower middle income"),
    ("TM", "TKM", "Europe & Central Asia", "Upper middle income"),
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
    for shard_index, (iso2, iso3, region_name, income_group) in enumerate(countries_to_export, start=1):
        country_geom = countries_fc.filter(ee.Filter.eq("country_co", iso2)).geometry()
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
