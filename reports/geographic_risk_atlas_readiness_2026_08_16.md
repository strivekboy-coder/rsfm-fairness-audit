# Geographic risk atlas readiness: AlphaEarth + fMoW + reBEN

Date: 2026-08-16

## Decision

The three-task atlas is feasible from frozen assets and is scientifically useful as a localization layer. It should use one visual system but three honest spatial representations rather than forcing every task into a country choropleth.

| Task | Existing usable geography | Recommended view now | Status | Claim level |
|---|---|---|---|---|
| AlphaEarth | `lat/lon`, `spatial_block_id`, country, region, biome/ecoregion, built proxy, WorldCover and Dynamic World fields in the frozen test metadata/predictions | observed spatial-block centroids coloured by mean risk and tail excess | Ready from canonical v2 frozen table; no new model run | Exploratory/descriptive map-label agreement |
| fMoW-Sentinel | latitude/longitude plus location/site identity in the clean location-disjoint metadata and frozen audit assets | observed site/location-unit risk maps, separately for DOFAv2 and ResNet-50; paired delta only on verified common support | Ready if canonical prediction/audit CSV is supplied; exact field audit remains a runtime gate | Descriptive scene-classification risk; paired differences only after ID alignment |
| reBEN | country, label, source tile and paired sample identity; current package does not justify precise sample coordinates | country × label S1-OOD minus S2-ID burden matrix with seed recurrence/support | Ready from per-seed ID/OOD label audit tables | Descriptive paired burden localization |

## Visual and statistical contract

- Mean risk uses `viridis`; tail excess uses `magma`; signed OOD−ID/model deltas use zero-centred `PuOr_r`. All figures are 300-dpi PNG plus PDF with editable text.
- Coordinate panels show observed points or verified spatial-unit centroids only, with no spatial interpolation and no invented coordinates.
- The atlas fixes three views in advance: mean unit risk, unit q90 tail excess, and paired signed burden. Extra variable fishing is excluded from the primary atlas.
- The unit-q90 tail map is exploratory and is not relabelled as formal GeoBWER certification. Formal inference would require a separate registered spatial/cluster procedure and multiplicity control.
- External covariates (nightlights, population, urbanisation, elevation, climate) remain a later preregistered mechanism analysis. None are downloaded by this implementation.

## Implementation

`scripts/analysis/build_geographic_risk_atlas.py` is CPU-only and reads frozen CSV/audit outputs. It writes aggregated evidence tables, readiness metadata, a manifest, and unified figures. Missing coordinate/risk fields fail closed. reBEN aggregation streams the large label audit CSVs and computes ID/OOD differences within seed before averaging.
