# Geographic risk atlas final protocol (2026-08-16)

## Frozen scope

This is CPU-only postprocessing of frozen audit tables. It does not retrain a model, rerun inference, change GeoBWER, download raster products, or start #8/#9. AlphaEarth and fMoW use observed coordinates/spatial units. reBEN remains a country × label paired-burden reproduction; no socioeconomic regression is permitted for its 10 countries.

## Atlas finality

- fMoW ResNet50 seeds are discovered only from direct `seed_*` children that contain both the canonical `formal_outputs/formal_audit_table.csv` and its `formal_output_manifest.json`. Drive inspection on 2026-08-16 found seeds 42/73/101; all three manifests report the same protocol hash (`f51e772730ce48a8a203e9768112419e396e96a7fd88bb299342896456a31a16`), dataset metadata hash, and geography-contract hash. The atlas aggregates those real seeds after within-seed location aggregation; it does not assume 101/202/303 or fabricate missing seeds. Inputs must have exactly the same unit universe, support, and coordinates. If a future canonical root contains only one valid seed, the atlas completes but explicitly marks ResNet50 as `single_seed_descriptive` and does not present across-seed uncertainty.
- DOFAv2 remains the existing single-run localization asset and is labelled as such.
- reBEN uses constrained layout, dynamic figure dimensions, anchored label rotation, and vector/raster export.
- reBEN retains the TerraMind country × label burden matrix and adds a country-level TerraMind-versus-CROMA paired Δrisk view from each model's frozen `paired_shift_country_deltas.csv`. The comparison requires the same three seeds, countries, support, and `mean_labelwise_binary_error` definition; it does not recompute predictions or alter the paired protocol.
- Every PNG/PDF is subject to fail-closed visual QA: existence/size, PNG dimensions, and non-white content. Manual synthetic-data review covers map overlay, effect plot, and burden heatmap layouts.

All Atlas figures share the `atlas_*` filename prefix, DejaVu Sans typography, uppercase panel labels, left-aligned task/view headers, consistent colorbar geometry, frameless legends, and the same semantic palette: viridis for risk, magma for tail excess, cividis for across-seed variability, PuOr centred at zero for signed burden, and Okabe–Ito colors plus distinct markers for model identity.

The enhanced artifact is written to `geographic_risk_atlas_v2_1` rather than mixed into the prior v2 directory. This is an output/visual-version increment only; the scientific estimands and frozen inputs are unchanged.

## Association preregistration

AlphaEarth confirmatory variables are land-cover heterogeneity, Dynamic World reference confidence, WorldCover–Dynamic World disagreement, and GHSL urbanization. fMoW confirmatory matching is limited to GHSL because scene categories are not a land-cover product and it has no equivalent reference-map confidence/disagreement. Population density and nightlights are exploratory exposures for both coordinate-bearing tasks.

The primary reported statistic is partial Spearman rho after controlling for latitude, sine/cosine longitude, and log unit support. Spatial sensitivity uses a fixed 15-degree latitude grid with latitude-adjusted longitude widths and a 500-replicate cluster bootstrap. Effect plots use fixed quartiles. A variable must cover at least 80% and 20 units. Missing fields are reported as unavailable and never replaced with convenient proxies.

“Confirmatory” here means variables and estimands were fixed before viewing these association results; it does not convert observational association into causality. We report effect sizes and spatial-cluster bootstrap confidence intervals without dichotomous p-value claims.

## Optional external covariate contract

Small, already-extracted CSVs may be placed at:

- `rsfm_fairness_audit/covariates/geographic_risk_v1/alphaearth_covariates.csv`
- `rsfm_fairness_audit/covariates/geographic_risk_v1/fmow_covariates.csv`

Each file must contain `spatial_unit` or real `latitude`/`longitude`, plus one or more registered fields: `ghsl_urbanization`, `population_density`, `nightlights`. Coordinate matching is capped at 50 km and audited. If the files are absent, the run still completes, marks those variables unavailable, and performs only covariates already present in frozen assets.
