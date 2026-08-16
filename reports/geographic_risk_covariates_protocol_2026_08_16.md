# Geographic risk covariates: frozen extraction protocol

## Scope

This stage prepares external covariates for the existing geographic risk atlas and reruns only the CPU association postprocess. It does not train a model, alter the frozen GeoBWER protocol, or start experiments #8/#9.

## Fixed products and epochs

| Variable | Official public asset | Epoch | Canonical value |
|---|---|---:|---|
| GHSL urbanization | `JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020` | 2020 | Official L2 `smod_code`; water/no-data is missing |
| Population density | `JRC/GHSL/P2023A/GHS_POP/2020` | 2020 | `population_count / pixel_area_km2` |
| Nightlights | `NOAA/VIIRS/DNB/ANNUAL_V21` | 2020 | `average_masked` radiance |
| Dynamic World reference label | `GOOGLE/DYNAMICWORLD/V1` | 2021 | Temporal modal class |
| Dynamic World reference confidence | `GOOGLE/DYNAMICWORLD/V1` | 2021 | Temporal mean of each observation's top-1 class probability |

Official catalog records are linked in the generated manifest. Product, epoch, band, transform, and spatial matching are fixed before reading association results.

## Spatial contract

- AlphaEarth: sample official rasters at frozen test-sample coordinates, retain exact atlas `spatial_block` units, then aggregate within each block. WorldCover heterogeneity uses frozen WorldCover sample labels. WorldCover–Dynamic World disagreement compares the frozen WorldCover label with the 2021 Dynamic World modal class at the same point.
- fMoW: sample at the frozen atlas representative coordinate for each exact location/site unit. All fMoW model atlas tables must have identical unit sets and coordinates.
- No outcome-selected buffer radius or nearest-neighbour substitution is used in the canonical CSVs.

## Analysis boundary

Confirmatory variables remain AlphaEarth land-cover heterogeneity, reference confidence, reference disagreement and GHSL urbanization, plus fMoW GHSL urbanization. Population density and nightlights remain exploratory exposures. Dynamic World variables describe reference-map ambiguity and are not treated as human ground truth quality.

The rerun reports partial Spearman effect sizes and spatial-cluster bootstrap intervals under the already frozen association protocol. Missing or low-coverage variables remain unavailable; they are not replaced with selected proxies.

## Persistent outputs

- `covariates/geographic_risk_v1/alphaearth_covariates.csv`
- `covariates/geographic_risk_v1/fmow_covariates.csv`
- `covariates/geographic_risk_v1/geographic_covariate_manifest.json`
- `covariates/geographic_risk_v1/covariate_qa.csv`
- `cache/geographic_risk_covariates_v1/*_official_gee_samples.csv`
- `outputs/geobwer_final_v3/geographic_risk_association_v1_2/`

The cache key includes the extraction protocol, product registry, input hash, and target risk-table hash. A rerun reuses the cache only when all of these remain identical.
