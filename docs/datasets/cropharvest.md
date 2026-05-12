# CropHarvest Dataset Note

## Official Sources

- Official repository: https://github.com/nasaharvest/cropharvest
- NeurIPS Datasets and Benchmarks paper: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/54229abfcfa5649e7003b83dd4755294-Abstract-round2.html
- Zenodo record: https://zenodo.org/records/10251170

## Verified Facts

CropHarvest is an open-source remote sensing dataset for agriculture. The official repository states that it aggregates agricultural land-use datasets and remote-sensing products. It includes 95,186 datapoints, 33,205 multiclass labels, and 70,213 labels paired with Sentinel-2, Sentinel-1, SRTM DEM, and ERA5 climatology data.

## Fairness Use

CropHarvest is a strong candidate for temporal and agricultural fairness. Start with binary crop/non-crop classification because multiclass labels cover only part of the dataset. Use `labels.geojson` for geographic slicing.

## Open Items

- Exact license for the latest Zenodo record: to_verify if not visible in metadata export.
- Country/admin/year fields available directly versus requiring geometry joins: to_verify.
- Recommended benchmark splits from the package API: to_verify before adapter implementation.
