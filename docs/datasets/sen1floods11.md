# Sen1Floods11 Dataset Note

## Official Sources

- Official repository: https://github.com/cloudtostreet/Sen1Floods11
- Paper: http://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html

## Verified Facts

The official repository describes Sen1Floods11 as a georeferenced dataset for training and testing flood algorithms for Sentinel-1. It is hosted in a Google Cloud Storage bucket and includes data, splits, and a STAC catalog.

Official documentation lists Sentinel-1 imagery with VV and VH bands, Sentinel-2 Level-1C imagery with 13 bands, and hand-labeled water masks. Files are projected to WGS84 at 10 m ground resolution.

## Fairness Use

This dataset is well suited for flood-event, country, orbit, sensor, and label-source fairness. The first subset should use hand-labeled chips only, grouped by flood event and ISO country.

## Open Items

- Current official license or terms in the GCS bucket: to_verify.
- Exact current train/validation/test split files in v1.1: to_verify.
- Mapping from event metadata to biome/ecoregion slices: to_verify.
