# Geographic risk atlas readiness: AlphaEarth + fMoW + reBEN

Date: 2026-08-16

## Decision

The three-task atlas is feasible from frozen assets and is scientifically useful as a localization layer. It should use one visual system but three honest spatial representations rather than forcing every task into a country choropleth.

| Task | Existing usable geography | Recommended view now | Status | Claim level |
|---|---|---|---|---|
| AlphaEarth | `formal_outputs/formal_audit_table.csv` contains sample-level `latitude`, `longitude`, `spatial_block_id`, and `risk` for the 24,030-row frozen test panel | observed spatial-block centroids coloured by mean risk and tail excess | Ready from canonical v2 formal table; no new model run | Exploratory/descriptive map-label agreement |
| fMoW-Sentinel | latitude/longitude plus location/site identity in the clean location-disjoint metadata and frozen audit assets | observed site/location-unit risk maps, separately for DOFAv2 and ResNet-50; paired delta only on verified common support | Ready if canonical prediction/audit CSV is supplied; exact field audit remains a runtime gate | Descriptive scene-classification risk; paired differences only after ID alignment |
| reBEN | country, label, source tile and paired sample identity; current package does not justify precise sample coordinates | country × label S1-OOD minus S2-ID burden matrix with seed recurrence/support | Ready from per-seed ID/OOD label audit tables | Descriptive paired burden localization |

## Visual and statistical contract

- Mean risk uses `viridis`; tail excess uses `magma`; signed OOD−ID/model deltas use zero-centred `PuOr_r`. All figures are 300-dpi PNG plus PDF with editable text.
- Coordinate panels show observed points or verified spatial-unit centroids only, with no spatial interpolation and no invented coordinates.
- The atlas fixes three views in advance: mean unit risk, unit q90 tail excess, and paired signed burden. Extra variable fishing is excluded from the primary atlas.
- The unit-q90 tail map is exploratory and is not relabelled as formal GeoBWER certification. Formal inference would require a separate registered spatial/cluster procedure and multiplicity control.
- External covariates (nightlights, population, urbanisation, elevation, climate) remain a later preregistered mechanism analysis. None are downloaded by this implementation.

## AlphaEarth Drive reconciliation

The canonical `alphaearth_geobwer_spatial_v2` root does not contain a prediction CSV at its top level. The three files under `geobwer_raw/` are not atlas coordinate sources: `geobwer_by_group.csv` has `axis, group, risk, support, cluster_support, selected_tail_mass, protocol_hash`, while `geobwer_profile.csv` and `geobwer_summary.csv` contain profile/axis-level metrics. None contains real latitude and longitude. The valid source is `formal_outputs/formal_audit_table.csv`; its formal manifest records 24,030 test rows, `spatial_block_id` as the cluster field, and sample-level risk, while the frozen upgrade pipeline preserves the verified latitude and longitude metadata. Automatic discovery now selects this file only after header validation and records the ignored aggregate tables in the atlas manifest.

## Implementation

`scripts/analysis/build_geographic_risk_atlas.py` is CPU-only and reads frozen CSV/audit outputs. It writes aggregated evidence tables, readiness metadata, a manifest, and unified figures. Missing coordinate/risk fields fail closed. reBEN aggregation streams the large label audit CSVs and computes ID/OOD differences within seed before averaging.
