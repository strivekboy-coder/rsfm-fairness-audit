"""Fast preflight for AlphaEarth formal country shards.

Run before launching long exports. This checks country boundary matching,
AlphaEarth tile coverage, and one-point sample availability per country without
starting Drive export tasks.
"""

from __future__ import annotations


EE_PROJECT = "rsfm-fairness-audit"
YEAR = 2021
COUNTRIES = [
    ("United States", "USA"),
    ("Brazil", "BRA"),
    ("India", "IND"),
    ("South Africa", "ZAF"),
    ("Australia", "AUS"),
    ("France", "FRA"),
    ("Indonesia", "IDN"),
    ("Mexico", "MEX"),
]


def main() -> None:
    import ee

    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

    countries_fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    alpha_ic = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover_label")

    print("country_name,iso3,lsib_matches,alpha_tiles,worldcover_sample,alpha_sample_status")
    for country_name, iso3 in COUNTRIES:
        fc = countries_fc.filter(ee.Filter.eq("country_na", country_name))
        match_count = fc.size().getInfo()
        if match_count == 0:
            print(f"{country_name},{iso3},0,0,,missing_country_boundary")
            continue
        geom = fc.geometry()
        alpha_count = alpha_ic.filterBounds(geom).size().getInfo()
        centroid = geom.centroid(maxError=1000)
        wc = worldcover.sample(region=centroid, scale=10, numPixels=1, geometries=False).first()
        wc_value = None if wc.getInfo() is None else wc.toDictionary().getInfo().get("worldcover_label")
        alpha_sample = (
            alpha_ic.filterBounds(centroid)
            .mosaic()
            .select(["A00"])
            .sample(region=centroid, scale=250, numPixels=1, geometries=False)
            .first()
        )
        alpha_status = "ok" if alpha_sample.getInfo() is not None else "null_at_centroid"
        print(f"{country_name},{iso3},{match_count},{alpha_count},{wc_value},{alpha_status}")


if __name__ == "__main__":
    main()
