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
    ("Egypt", "EGY", "Middle East & North Africa", "Lower middle income"),
    ("Nigeria", "NGA", "Sub-Saharan Africa", "Lower middle income"),
    ("Canada", "CAN", "North America", "High income"),
    ("Argentina", "ARG", "Latin America", "Upper middle income"),
    ("Chile", "CHL", "Latin America", "High income"),
    ("Colombia", "COL", "Latin America", "Upper middle income"),
    ("Peru", "PER", "Latin America", "Upper middle income"),
    ("Bolivia", "BOL", "Latin America", "Lower middle income"),
    ("Paraguay", "PRY", "Latin America", "Upper middle income"),
    ("Uruguay", "URY", "Latin America", "High income"),
    ("Guatemala", "GTM", "Latin America", "Upper middle income"),
    ("Cuba", "CUB", "Latin America", "Upper middle income"),
    ("Dominican Republic", "DOM", "Latin America", "Upper middle income"),
    ("United Kingdom", "GBR", "Europe & Central Asia", "High income"),
    ("Germany", "DEU", "Europe & Central Asia", "High income"),
    ("Spain", "ESP", "Europe & Central Asia", "High income"),
    ("Italy", "ITA", "Europe & Central Asia", "High income"),
    ("Poland", "POL", "Europe & Central Asia", "High income"),
    ("Sweden", "SWE", "Europe & Central Asia", "High income"),
    ("Norway", "NOR", "Europe & Central Asia", "High income"),
    ("Finland", "FIN", "Europe & Central Asia", "High income"),
    ("Turkey", "TUR", "Europe & Central Asia", "Upper middle income"),
    ("Ukraine", "UKR", "Europe & Central Asia", "Lower middle income"),
    ("Romania", "ROU", "Europe & Central Asia", "High income"),
    ("Greece", "GRC", "Europe & Central Asia", "High income"),
    ("Portugal", "PRT", "Europe & Central Asia", "High income"),
    ("Netherlands", "NLD", "Europe & Central Asia", "High income"),
    ("Belgium", "BEL", "Europe & Central Asia", "High income"),
    ("Switzerland", "CHE", "Europe & Central Asia", "High income"),
    ("Austria", "AUT", "Europe & Central Asia", "High income"),
    ("Czech Republic", "CZE", "Europe & Central Asia", "High income"),
    ("Russia", "RUS", "Europe & Central Asia", "Upper middle income"),
    ("China", "CHN", "East Asia & Pacific", "Upper middle income"),
    ("Japan", "JPN", "East Asia & Pacific", "High income"),
    ("South Korea", "KOR", "East Asia & Pacific", "High income"),
    ("Thailand", "THA", "East Asia & Pacific", "Upper middle income"),
    ("Vietnam", "VNM", "East Asia & Pacific", "Lower middle income"),
    ("Philippines", "PHL", "East Asia & Pacific", "Lower middle income"),
    ("Malaysia", "MYS", "East Asia & Pacific", "Upper middle income"),
    ("Myanmar", "MMR", "East Asia & Pacific", "Lower middle income"),
    ("Cambodia", "KHM", "East Asia & Pacific", "Lower middle income"),
    ("Laos", "LAO", "East Asia & Pacific", "Lower middle income"),
    ("New Zealand", "NZL", "East Asia & Pacific", "High income"),
    ("Papua New Guinea", "PNG", "East Asia & Pacific", "Lower middle income"),
    ("Pakistan", "PAK", "South Asia", "Lower middle income"),
    ("Bangladesh", "BGD", "South Asia", "Lower middle income"),
    ("Nepal", "NPL", "South Asia", "Lower middle income"),
    ("Sri Lanka", "LKA", "South Asia", "Lower middle income"),
    ("Afghanistan", "AFG", "South Asia", "Low income"),
    ("Iran", "IRN", "Middle East & North Africa", "Lower middle income"),
    ("Iraq", "IRQ", "Middle East & North Africa", "Upper middle income"),
    ("Saudi Arabia", "SAU", "Middle East & North Africa", "High income"),
    ("United Arab Emirates", "ARE", "Middle East & North Africa", "High income"),
    ("Israel", "ISR", "Middle East & North Africa", "High income"),
    ("Jordan", "JOR", "Middle East & North Africa", "Lower middle income"),
    ("Morocco", "MAR", "Middle East & North Africa", "Lower middle income"),
    ("Algeria", "DZA", "Middle East & North Africa", "Lower middle income"),
    ("Tunisia", "TUN", "Middle East & North Africa", "Lower middle income"),
    ("Libya", "LBY", "Middle East & North Africa", "Upper middle income"),
    ("Ethiopia", "ETH", "Sub-Saharan Africa", "Low income"),
    ("Kenya", "KEN", "Sub-Saharan Africa", "Lower middle income"),
    ("Tanzania", "TZA", "Sub-Saharan Africa", "Lower middle income"),
    ("Uganda", "UGA", "Sub-Saharan Africa", "Low income"),
    ("Ghana", "GHA", "Sub-Saharan Africa", "Lower middle income"),
    ("Cote d'Ivoire", "CIV", "Sub-Saharan Africa", "Lower middle income"),
    ("Senegal", "SEN", "Sub-Saharan Africa", "Lower middle income"),
    ("Cameroon", "CMR", "Sub-Saharan Africa", "Lower middle income"),
    ("Democratic Republic of the Congo", "COD", "Sub-Saharan Africa", "Low income"),
    ("Angola", "AGO", "Sub-Saharan Africa", "Lower middle income"),
    ("Zambia", "ZMB", "Sub-Saharan Africa", "Lower middle income"),
    ("Zimbabwe", "ZWE", "Sub-Saharan Africa", "Lower middle income"),
    ("Mozambique", "MOZ", "Sub-Saharan Africa", "Low income"),
    ("Madagascar", "MDG", "Sub-Saharan Africa", "Low income"),
    ("Mali", "MLI", "Sub-Saharan Africa", "Low income"),
    ("Niger", "NER", "Sub-Saharan Africa", "Low income"),
    ("Sudan", "SDN", "Sub-Saharan Africa", "Low income"),
    ("South Sudan", "SSD", "Sub-Saharan Africa", "Low income"),
    ("Burkina Faso", "BFA", "Sub-Saharan Africa", "Low income"),
    ("Malawi", "MWI", "Sub-Saharan Africa", "Low income"),
    ("Kazakhstan", "KAZ", "Europe & Central Asia", "Upper middle income"),
    ("Uzbekistan", "UZB", "Europe & Central Asia", "Lower middle income"),
    ("Mongolia", "MNG", "East Asia & Pacific", "Lower middle income"),
    ("Azerbaijan", "AZE", "Europe & Central Asia", "Upper middle income"),
    ("Georgia", "GEO", "Europe & Central Asia", "Upper middle income"),
    ("Armenia", "ARM", "Europe & Central Asia", "Upper middle income"),
    ("Kyrgyzstan", "KGZ", "Europe & Central Asia", "Lower middle income"),
    ("Tajikistan", "TJK", "Europe & Central Asia", "Lower middle income"),
    ("Turkmenistan", "TKM", "Europe & Central Asia", "Upper middle income"),
    ("Ireland", "IRL", "Europe & Central Asia", "High income"),
    ("Denmark", "DNK", "Europe & Central Asia", "High income"),
    ("Hungary", "HUN", "Europe & Central Asia", "High income"),
    ("Bulgaria", "BGR", "Europe & Central Asia", "Upper middle income"),
    ("Serbia", "SRB", "Europe & Central Asia", "Upper middle income"),
    ("Croatia", "HRV", "Europe & Central Asia", "High income"),
    ("Slovakia", "SVK", "Europe & Central Asia", "High income"),
    ("Slovenia", "SVN", "Europe & Central Asia", "High income"),
    ("Rwanda", "RWA", "Sub-Saharan Africa", "Low income"),
    ("Botswana", "BWA", "Sub-Saharan Africa", "Upper middle income"),
    ("Namibia", "NAM", "Sub-Saharan Africa", "Upper middle income"),
    ("Qatar", "QAT", "Middle East & North Africa", "High income"),
    ("Oman", "OMN", "Middle East & North Africa", "High income"),
    ("Kuwait", "KWT", "Middle East & North Africa", "High income"),
    ("Ecuador", "ECU", "Latin America", "Upper middle income"),
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
            .set("sample_id", ee.String(iso3).cat("_").cat(label).cat("_").cat(ee.String(feature.id())))
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
