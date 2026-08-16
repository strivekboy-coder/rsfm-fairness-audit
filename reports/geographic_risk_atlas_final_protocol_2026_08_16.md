# Geographic risk atlas final protocol (2026-08-16)

## Frozen scope

This is CPU-only postprocessing of frozen audit tables. It does not retrain a model, rerun inference, change GeoBWER, download raster products, or start #8/#9. AlphaEarth and fMoW use observed coordinates/spatial units. reBEN remains a country × label paired-burden reproduction; no socioeconomic regression is permitted for its 10 countries.

## Atlas finality

- fMoW ResNet50 is aggregated across seeds 101/202/303 after within-seed location aggregation. The three inputs must have exactly the same unit universe, support, and coordinates.
- DOFAv2 remains the existing single-run localization asset and is labelled as such.
- reBEN uses constrained layout, dynamic figure dimensions, anchored label rotation, and vector/raster export.
- Every PNG/PDF is subject to fail-closed visual QA: existence/size, PNG dimensions, and non-white content. Manual synthetic-data review covers map overlay, effect plot, and burden heatmap layouts.

## Association preregistration

AlphaEarth confirmatory variables are land-cover heterogeneity, Dynamic World reference confidence, WorldCover–Dynamic World disagreement, and GHSL urbanization. fMoW confirmatory matching is limited to GHSL because scene categories are not a land-cover product and it has no equivalent reference-map confidence/disagreement. Population density and nightlights are exploratory exposures for both coordinate-bearing tasks.

The primary reported statistic is partial Spearman rho after controlling for latitude, sine/cosine longitude, and log unit support. Spatial sensitivity uses a fixed 15-degree latitude grid with latitude-adjusted longitude widths and a 500-replicate cluster bootstrap. Effect plots use fixed quartiles. A variable must cover at least 80% and 20 units. Missing fields are reported as unavailable and never replaced with convenient proxies.

“Confirmatory” here means variables and estimands were fixed before viewing these association results; it does not convert observational association into causality. We report effect sizes and spatial-cluster bootstrap confidence intervals without dichotomous p-value claims.

## Optional external covariate contract

Small, already-extracted CSVs may be placed at:

- `rsfm_fairness_audit/covariates/geographic_risk_v1/alphaearth_covariates.csv`
- `rsfm_fairness_audit/covariates/geographic_risk_v1/fmow_covariates.csv`

Each file must contain `spatial_unit` or real `latitude`/`longitude`, plus one or more registered fields: `ghsl_urbanization`, `population_density`, `nightlights`. Coordinate matching is capped at 50 km and audited. If the files are absent, the run still completes, marks those variables unavailable, and performs only covariates already present in frozen assets.
