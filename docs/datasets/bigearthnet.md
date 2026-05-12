# BigEarthNet v2.0 Dataset Note

## Official Sources

- Homepage: https://bigearth.net/
- Description PDF: https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf
- Zenodo record: https://zenodo.org/records/10891137

## Verified Facts

BigEarthNet v2.0 contains 549,488 paired Sentinel-1 and Sentinel-2 image patches. The official description states that Sentinel-2 tiles were selected over 10 European countries and that Sentinel-1 patches were prepared for the same patch set. The dataset is licensed under CDLA-Permissive-1.0.

BigEarthNet v2.0 includes patch-level multi-label land-cover labels and pixel-level reference maps, making it useful for both scene and pixel-level tasks.

## Fairness Use

This is the first real dataset target. Start with a metadata-filtered subset that balances country, sensor availability, and label frequency. Region slicing should begin with country/tile-level slices before attempting broader region groupings.

## Open Items

- Exact metadata parquet columns needed for country, coordinates, season, climate zone, and split: to_verify.
- Minimal subset download path that avoids full S1/S2 download: to_verify.
- Official split file structure and loader implementation details: to_verify.
