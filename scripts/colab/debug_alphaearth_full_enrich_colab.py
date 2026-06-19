"""Debug the formal AlphaEarth export enrich() map path without Drive export."""

from __future__ import annotations


EE_PROJECT = "rsfm-fairness-audit"
YEAR = 2021


def main() -> None:
    import ee

    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

    embedding_bands = [f"A{i:02d}" for i in range(64)]
    countries_fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    country_name, iso3, region_name, income_group = ("Italy", "ITA", "Europe & Central Asia", "High income")
    geom = countries_fc.filter(ee.Filter.eq("country_na", country_name)).geometry()
    alpha = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01").filterBounds(geom).mosaic().select(embedding_bands)
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover_label")
    worldcover_names = ee.Dictionary({"10": "Tree cover", "20": "Shrubland", "30": "Grassland", "40": "Cropland", "50": "Built-up", "60": "Bare/sparse vegetation", "70": "Snow and ice", "80": "Permanent water bodies", "90": "Herbaceous wetland", "95": "Mangroves", "100": "Moss and lichen"})
    sample = alpha.addBands(worldcover).stratifiedSample(
        numPoints=1,
        classBand="worldcover_label",
        region=geom,
        scale=250,
        seed=7,
        geometries=True,
        classValues=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        classPoints=[1] * 11,
        tileScale=8,
    ).randomColumn("split_random", seed=7)

    def enrich(feature):
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

    enriched = sample.map(enrich)
    print("enriched sample size:", enriched.size().getInfo())
    print(enriched.first().toDictionary().getInfo())


if __name__ == "__main__":
    main()
