# fMoW-Sentinel / fMoW-S2 Preflight Foundation

This step builds the data and audit foundation for a later global-geography
BWER audit. It does not train models, run inference, or define a new benchmark
suite.

Known dataset target:

- dataset label: `fmow_sentinel`
- task label: `scene_classification`
- input mode: `s2_13band_image_only`
- band profile: `sentinel2_13band_fmow`
- common benchmark framing: 62-class Sentinel-2 scene classification
- expected scale: roughly 882,779 images across train/val/test

Do not assume country, region, season, or latitude are already present in one
clean CSV. The preflight reports which fields are supplied, derivable, missing,
or require future metadata joins.

## CLI

Metadata enrichment from SatMAE CSV plus optional original fMoW/GPS/geography
metadata. This is needed only when rebuilding the final metadata package from
source inputs:

```bash
python -m rsfm_fairness_audit.cli enrich-fmow-sentinel-metadata \
  --satmae-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/train.csv \
  --satmae-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/val.csv \
  --external-metadata-csv path/to/original_fmow_or_gps_metadata.csv \
  --country-region-map /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_country_region_map_full_v1.csv \
  --output-dir /content/outputs/fmow_sentinel_metadata_enrichment/satmae_with_geo
```

If no external metadata is available, run the same command without
`--external-metadata-csv`. The output will keep latitude, longitude, country,
region, continent, and UN region blank and will report that SatMAE CSVs alone
are not sufficient for formal country/region geography BWER.

Metadata-only mode:

```bash
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel \
  --metadata-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_geography_final_v1.csv \
  --country-region-map /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_country_region_map_full_v1.csv \
  --output-dir outputs/fmow_sentinel_preflight/enriched_geography_final_v1 \
  --metadata-only
```

Preflight with subset generation and optional raster inspection:

```bash
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel \
  --metadata-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_geography_final_v1.csv \
  --country-region-map /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_country_region_map_full_v1.csv \
  --output-dir outputs/fmow_sentinel_preflight/enriched_geography_final_v1_raster_sample \
  --split official_split \
  --subset-max-per-split 5000 \
  --inspect-rasters \
  --raster-sample-size 256 \
  --data-root path/to/fmow_sentinel_root \
  --seed 42
```

Use repeated `--metadata-csv` for separate train/val/test CSVs. Use repeated
`--filter-split` to restrict subset generation to specific official split
values.

## Outputs

The preflight writes:

- `fmow_metadata_inventory.csv`
- `fmow_missing_fields_report.md`
- `fmow_slice_support_summary.csv`
- `fmow_slice_support_recommendations.csv`
- `fmow_preflight_report.md`
- `subset_metadata.csv`
- `subset_manifest.csv`
- `subset_support_report.md`
- `band_statistics_sample.csv`
- `image_shape_summary.csv`
- `raster_loading_report.md`
- `audit_table_schema_fmow_sentinel.md`
- `warnings.json`
- `run_metadata.json`

The enrichment step writes:

- `fmow_enriched_metadata.csv`
- `fmow_metadata_join_report.md`
- `fmow_geography_coverage_summary.csv`
- `fmow_join_failures.csv`
- `warnings.json`
- `run_metadata.json`

## Metadata Derivations

If source columns are available, the workflow derives:

- `year`, `month`, and `season` from timestamp.
- `latitude_band` from latitude.

It does not derive country, continent, or UN region unless those fields are
already supplied. Coordinate-to-country joins require external boundary or
geocoder resources and should be recorded as a separate metadata provenance
step.

If only `location_id` is available, keep it as `location_id`; do not reinterpret
it as country.

The SatMAE fMoW-Sentinel CSVs observed so far contain `category`,
`location_id`, `timestamp`, and `image_id`. Those fields support class and
season preflight, but they do not by themselves support formal country,
continent, UN-region, or latitude-band BWER. Provide original fMoW metadata,
GPS/coordinate metadata, or a verified external geography table keyed by
`category + location_id + image_id`, `location_id + image_id`, or `location_id`
to enrich geography fields. Any joined field is marked with provenance in
`fmow_enriched_metadata.csv`.

The final confirmed enrichment protocol uses original fMoW metadata as the
external geography source. Image-level join on
`category + location_id + image_id` was incomplete, while location-level join on
`category + location_id` reached 100%. The final metadata therefore uses
location-level attributes:

- `country`: majority mode per `category + location_id`.
- `latitude` / `longitude`: median polygon centroid per
  `category + location_id`.
- `continent` / `un_region` / `region`: mapped from `country` using
  `country_region_map_full.csv`.

These geography fields are for audit slicing and reporting only. They are not
model inputs.

If enriched metadata already has a verified `country` field, pass
`--country-region-map` to either enrichment or preflight to fill `continent`,
`un_region`, and optional `region` from a small verified mapping table. The map
should contain at least `country`, and may contain `continent`, `un_region`, and
`region`. Rows not covered by the map are left blank and reported in warnings;
the workflow does not geocode or invent missing regions.

For high-cardinality geography such as country, the preflight distinguishes
full-slice formal readiness from support-filtered formal readiness. If
`country` has low missingness but some countries fall below `--min-support`,
the primary recommendation remains `diagnostic-only`, while
`support_filtered_recommendation` can mark the subset of countries meeting the
support threshold as `support-filtered-formal-BWER-ready`.

## Current Enriched Geography Preflight Result

Recorded: 2026-05-21.

The current final fMoW-Sentinel geography preflight artifact is:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/fmow_sentinel_preflight/enriched_geography_final_v1.zip
```

The final reproducible metadata package lives in:

```text
/content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/
```

Recommended internal directory name:

```text
/content/outputs/fmow_sentinel_preflight/enriched_geography_final_v1
```

The directory
`/content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final_backup_20260520_233834/`
is a temporary safety backup. It is not part of the canonical final structure
and can be deleted after the Step 3 fMoW-Sentinel model prototype has run
successfully from the `metadata/final/` files.

Confirmed preflight facts:

- `country` missing ratio = 0.
- `latitude` and `longitude` are present.
- `continent`, `un_region`, and `region` missing ratio = 0.0186.
- `country`, `latitude_band`, `continent`, `un_region`, `region`, `category`,
  and `season` are usable geography/audit fields according to the generated
  support recommendations.
- Remaining unmapped / dirty geography codes are `ambiguous_country`, `ANT`,
  `CA-`, and `KO-`.

These cases are intentionally left unmapped and reported as known dirty or
legacy geography metadata. Do not silently map them to a country, continent, or
UN region.

Canonical final metadata filenames:

```text
fmow_sentinel_enriched_geography_final_v1.csv
fmow_sentinel_location_join_summary_final_v1.txt
fmow_sentinel_location_country_conflicts_final_v1.csv
fmow_sentinel_location_geography_summary_final_v1.csv
fmow_country_region_map_full_v1.csv
fmow_country_region_mapping_report_final_v1.md
fmow_country_region_unmapped_final_v1.csv
```

Recommended Step 3 / rerun command using the final metadata package:

```bash
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel \
  --metadata-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_geography_final_v1.csv \
  --country-region-map /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_country_region_map_full_v1.csv \
  --output-dir /content/outputs/fmow_sentinel_preflight/enriched_geography_final_v1 \
  --metadata-only \
  --subset-max-per-split 5000 \
  --seed 42
```

Do not use the older loose copies under
`outputs/fmow_sentinel_preflight/` as canonical metadata inputs. They were
intermediate convenience copies; `cache/fmow_sentinel/metadata/final/` is the
reproducible metadata source of record.

## Clean Subset Extraction From Official Archive

Step 3 model prototypes should not fully extract the official archive. The
official archive is large and contains roughly 882k Sentinel-2 TIFF files.
Download or place the archive on local Colab storage, for example:

```text
/content/fmow-sentinel.tar.gz
```

The official member path convention is:

```text
fmow-sentinel/<split>/<category>/<category>_<location_id>/<category>_<location_id>_<image_id>.tif
```

Use the clean subset preparation script to select target paths from metadata
first, extract only those archive members, and validate every extracted TIFF:

```bash
python scripts/prepare_fmow_sentinel_clean_subset.py \
  --archive /content/fmow-sentinel.tar.gz \
  --metadata-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_sample_manifest_final_v1.csv \
  --output-dir /content/data/fmow_sentinel_clean_subset_v1 \
  --split train \
  --split val \
  --max-samples-per-split 5000 \
  --stratify-field category \
  --stratify-field country \
  --stratify-field region \
  --stratify-field latitude_band \
  --seed 42
```

If the canonical sample-level manifest is missing, the same script can rebuild
it from SatMAE train/val CSVs plus the final location-level geography metadata:

```bash
python scripts/prepare_fmow_sentinel_clean_subset.py \
  --archive /content/fmow-sentinel.tar.gz \
  --satmae-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/train.csv \
  --satmae-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/val.csv \
  --location-geography-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_geography_final_v1.csv \
  --output-dir /content/data/fmow_sentinel_clean_subset_v1 \
  --split train \
  --split val \
  --max-samples-per-split 5000 \
  --seed 42
```

This fallback still uses location-level geography enrichment. It must not be
described as an image-level exact geography join.

The script writes:

- `target_paths.csv`
- `include_list.txt`
- `clean_subset_manifest.csv`
- `support_summary.csv`
- `extraction_summary.csv`
- `raster_validation_report.csv`
- `warnings.json`

Only rasters that are readable, non-empty, and contain 13 Sentinel-2 bands are
included in `clean_subset_manifest.csv`. Missing members, corrupt files, and
band-count mismatches are recorded in `warnings.json` and
`raster_validation_report.csv`.

### Support-Aware Augmentation

If an initial clean subset is valid but too weak for some planned audit slices,
augment it instead of discarding it. This preserves all existing valid samples
and selects additional target paths from the full metadata while excluding
already selected samples.

Example: grow a valid 10k subset to a 30k final Step 3 audit subset with a
15k/15k train/val target:

```bash
python scripts/prepare_fmow_sentinel_clean_subset.py \
  --archive /content/fmow-sentinel.tar.gz \
  --metadata-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_sample_manifest_final_v1.csv \
  --augment-existing-manifest /content/data/fmow_sentinel_clean_subset_v1/clean_subset_manifest.csv \
  --output-dir /content/data/fmow_sentinel_clean_subset_v1 \
  --split train \
  --split val \
  --target-total 30000 \
  --target-train 15000 \
  --target-val 15000 \
  --seed 42
```

The augmentation scorer prioritizes weak support in:

- `season`
- `latitude_band`
- `un_region`
- `region`
- `country`
- `category x region`
- `country x category`

Augmentation outputs:

- `augmented_clean_subset_manifest.csv`
- `augmentation_target_paths.csv`
- `augmentation_support_before.csv`
- `augmentation_support_after.csv`
- `augmentation_summary.json`
- `raster_validation_report_augmented.csv`
- `warnings_augmented.json`

The finalized Step 3 dataset is:

```text
fmow_sentinel_clean_subset_30k_location_disjoint_v2
```

Expected Colab manifest:

```text
/content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv
```

Drive archive:

```text
/content/drive/MyDrive/rsfm_fairness_audit/fmow_sentinel_clean_subset_30k_location_disjoint_v2.zip
```

The full official fMoW-Sentinel tarball is also cached in Drive for future
reprocessing:

```text
/content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/fmow-sentinel.tar.gz
```

This cache copy is for reproducibility and future clean-subset extraction. Do
not fully extract it into Drive, package it into run handoff zips, or commit it
to the repository.

Use this final manifest for ResNet-50 and DOFA Step 3 runs. Do not use the
earlier 10k subset or any non-location-disjoint manifest for the main Step 3
experiments. Do not package the raw full archive or a fully extracted fMoW tree
into the repository.

Remaining low-support slices are recorded in support tables and should be
reported as support limitations, not silently treated as balanced.

### Final 30k Location-Disjoint Dataset Record

Source archive:

- official fMoW-Sentinel `fmow-sentinel.tar.gz` from Stanford PURL.

Extraction strategy:

- full tarball downloaded to Colab local `/content`;
- archive was not fully extracted;
- support-aware clean subset extracted from the local tar;
- earlier streaming partial-extraction experiments are excluded from formal
  data.

Split protocol:

- `split_protocol = location_disjoint`
- location-disjoint 70/30 split
- group key: `category + location_id`
- original source split preserved as `split_original`
- final `split` column contains `train` / `val`
- train/val location overlap: 0
- `sample_id` was regenerated to be unique in the final manifest

Final dataset facts:

| field | value |
| --- | ---: |
| total rows | 30000 |
| train rows | 21046 |
| val rows | 8954 |
| categories | 62 |
| val category coverage | all 62 categories |
| minimum val category support | 33 |
| country coverage | 195 countries |
| country missing ratio | 0 |
| continent / un_region / region missing ratio | 0.024 |
| season missing ratio | 0 |
| latitude_band missing ratio | 0 |
| val countries with >=20 samples | 145 |
| val countries with >=30 samples | 105 |

`continent`, `un_region`, `region`, `season`, and `latitude_band` all have
validation support. Geography dirty country codes are retained in the manifest
for provenance but should not be used for formal region-level BWER when mapped
geography is missing:

- `ambiguous_country`
- `ANT`
- `KO-`
- `CA-`

Formal analysis guidance for this dataset:

- Primary formal slices: `continent`, `un_region`, `region`,
  `latitude_band`, `season`, and `category`.
- Country-level BWER should use support thresholds such as validation support
  `>=20` or `>=30`.
- `country x category` is diagnostic-only unless support-threshold filtered.
- `region x category`, `season x category`, and
  `latitude_band x category` may be used for standardisation where preflight
  support allows; otherwise report them as diagnostics or sensitivity checks.

## Slice Support

Candidate slices:

- category/class
- location_id
- country
- continent / UN region
- latitude_band
- season
- region x category
- country x category
- season x category

Recommendations are labeled:

- `formal-BWER-ready`
- `diagnostic-only`
- `not-recommended`

These are support and missingness checks only. Formal BWER still requires
future model predictions normalized into the fMoW-Sentinel audit table schema.

## Raster Inspection

Raster inspection is optional and samples only the subset manifest. It uses
`rasterio` first, then `tifffile`, and never assumes PIL/ImageNet/RGB input.

The report checks:

- image shape distribution
- band count distribution
- dtype distribution
- per-band min/max/mean/std
- path/read failures
- warnings for band counts other than 13
- warnings for highly variable shapes

If raster dependencies or image files are unavailable, use metadata-only mode.

## Future Audit Table

Future prediction rows should include sample-level 0/1 classification risk:

```text
sample_id, image_id, image_path, dataset=fmow_sentinel,
task=scene_classification, split, category/label, prediction, correct,
risk, model_family, model_variant, input_mode=s2_13band_image_only,
adaptation_protocol, split_protocol, eval_scope, resolution,
band_profile=sentinel2_13band_fmow, timestamp/year/month/season when available,
location_id, latitude/longitude when available, geography fields when supplied
or explicitly joined.
```

Geography metadata is for slicing and reporting, not model input, unless a
separate metadata-aware protocol is explicitly declared.

## Step 3 Core Prototype

The Step 3 core prototype uses the final enriched metadata package from Step 2
and runs image-only fMoW-Sentinel scene classification. It does not rebuild
metadata enrichment, and it does not feed geography metadata into the model.
The model sees only 13-band Sentinel-2 imagery; geography fields are carried
forward for support diagnostics and BWER reporting.

Expected final Step 3 dataset input:

```text
/content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv
```

This manifest already includes final `train` / `val` split labels,
`split_original`, unique `sample_id` values, raster paths, and the final
location-level geography enrichment fields.

Raster inspection before model work:

```bash
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel \
  --metadata-csv /content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv \
  --country-region-map /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_country_region_map_full_v1.csv \
  --output-dir /content/outputs/fmow_sentinel_preflight/step3_raster_sample \
  --data-root /content/data/fmow_sentinel_clean_subset_30k_v2 \
  --inspect-rasters \
  --raster-sample-size 256 \
  --split location_disjoint \
  --seed 42
```

Lightweight supervised image-only prototype:

```bash
python -m rsfm_fairness_audit.cli run-fmow-sentinel-classification \
  --metadata-csv /content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv \
  --data-root /content/data/fmow_sentinel_clean_subset_30k_v2 \
  --output-dir /content/outputs/fmow_sentinel_supervised_stats_val \
  --model supervised_stats \
  --train-split train \
  --eval-split val \
  --split-protocol location_disjoint \
  --image-size 96 \
  --run-bwer
```

The supervised prototype uses 13-band image statistics and a nearest-centroid
classifier. It is a small supervised baseline for checking the geography BWER
pipeline, not a SOTA classifier.

Paper-grade supervised ResNet-50 baseline:

```bash
python -m rsfm_fairness_audit.cli run-fmow-sentinel-classification \
  --metadata-csv /content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv \
  --data-root /content/data/fmow_sentinel_clean_subset_30k_v2 \
  --output-dir /content/outputs/fmow_sentinel_resnet50_30k_location_disjoint \
  --model resnet50 \
  --train-split train \
  --eval-split val \
  --split-protocol location_disjoint \
  --eval-scope val \
  --image-size 96 \
  --batch-size 32 \
  --epochs 20 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --run-bwer
```

The ResNet-50 path uses torchvision's ResNet-50 with `weights=None`, replaces
the first convolution with a 13-channel convolution, and trains with
cross-entropy on Sentinel-2 13-band image tensors. It computes per-band
mean/std from the training split only and writes `norm_stats.json` for reuse.
Do not use ImageNet RGB normalization. The model input is imagery only:
`country`, `region`, coordinates, timestamp, season, and other geography fields
are copied to prediction/audit outputs solely for support diagnostics and BWER
reporting.

Minimal DOFA frozen-encoder candidate:

```bash
python -m rsfm_fairness_audit.cli run-fmow-sentinel-classification \
  --metadata-csv /content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv \
  --data-root /content/data/fmow_sentinel_clean_subset_30k_v2 \
  --output-dir /content/outputs/fmow_sentinel_dofa_frozen_probe_val \
  --model dofa \
  --model-config configs/models/dofa_fmow_sentinel.yaml \
  --train-split train \
  --eval-split val \
  --split-protocol location_disjoint \
  --image-size 224 \
  --batch-size 16 \
  --allow-torch-hub-download \
  --run-bwer
```

The DOFA command uses the existing DOFA adapter with
`band_profile=sentinel2_13band_fmow`. Torch Hub download is disabled in the
config by default; pass `--allow-torch-hub-download` only in a Colab/runtime
where downloading the official weights is intended.

Post-hoc geography BWER can be rerun without model inference:

```bash
python -m rsfm_fairness_audit.cli run-fmow-geography-bwer \
  --input-dir /content/outputs/fmow_sentinel_supervised_stats_val \
  --output-dir /content/outputs/fmow_sentinel_supervised_stats_val/bwer
```

Compare the supervised prototype and DOFA frozen-probe run:

```bash
python -m rsfm_fairness_audit.cli compare-fmow-runs \
  --run supervised=/content/outputs/fmow_sentinel_supervised_stats_val \
  --run dofa=/content/outputs/fmow_sentinel_dofa_frozen_probe_val \
  --output-dir /content/outputs/comparisons/fmow_sentinel_supervised_vs_dofa
```

The comparison reports aggregate accuracy, Raw-BWER(country), and
class-standardised country BWER where the completed run outputs support it.
Use these together: high average accuracy does not by itself imply low
geography tail risk.

## Step 3 Result Contract And Handoff

Recommended result layout:

```text
outputs/fmow_sentinel_step3/<run_name>/
  data/
    final_clean_subset_manifest_30k_location_disjoint_v2.csv
    subset_support_summary.csv
    raster_validation_report.csv
    warnings.json
  supervised_baseline/
    predictions.csv
    metrics_summary.csv
    run_metadata.json
  dofa_frozen_probe/
    predictions.csv
    metrics_summary.csv
    run_metadata.json
  bwer/
    bwer_summary.csv
    bwer_by_slice.csv
    alpha_sensitivity.csv
    support_sensitivity.csv
    warnings.json
  comparison/
    comparison_summary.csv
    average_vs_bwer.csv
    report.md
  archive_manifest.json
  handoff_checklist.md
```

For single-model prototype runs, the existing run-level layout
`<run_dir>/predictions.csv`, `<run_dir>/audit_table.csv`, and
`<run_dir>/bwer/` is also accepted by the validators.

The data contract for Step 3 formal runs is
`fmow_sentinel_clean_subset_30k_location_disjoint_v2`: 30,000 rows,
`split_protocol=location_disjoint`, final `split=train/val`,
`split_original` preserved, unique `sample_id`, and zero train/val overlap for
the `category + location_id` grouping key.

Validate a completed run directory:

```bash
python -m rsfm_fairness_audit.cli validate-fmow-step3-results \
  --run-dir /content/outputs/fmow_sentinel_supervised_stats_val \
  --full-archive-downloaded-locally true \
  --full-extraction-avoided true \
  --streaming-partial-extraction-excluded true
```

This writes:

- `prediction_table_validation.json`
- `prediction_table_validation.md`
- `bwer_output_validation.json`
- `bwer_output_validation.md`
- `archive_manifest.json`
- `provenance_report.md`
- `handoff_checklist.md`

Package the handoff zip without raster imagery:

```bash
python -m rsfm_fairness_audit.cli package-fmow-step3-handoff \
  --run-dir /content/outputs/fmow_sentinel_supervised_stats_val \
  --output-zip /content/drive/MyDrive/rsfm_fairness_audit/outputs/fmow_step3_supervised_stats_handoff.zip
```

The handoff package includes reports, manifests, prediction tables, BWER
outputs, metadata summaries, warnings, and checksums. It excludes `.tif`,
`.npy`, `.npz`, HDF5, and other raster/array payloads by default. Use
`--include-rasters` only for a deliberately small artifact where packaging
images is intended.

Current download and data handling protocol:

- Download the full official archive to Colab local storage such as
  `/content/data` or `/content`.
- The archived source tarball may be cached for future reprocessing at
  `/content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/fmow-sentinel.tar.gz`.
- Use `aria2c` with multi-connection resume when downloading in Colab.
- Do not fully extract the archive. Full extraction creates many small TIFF
  files and increases disk, inode, and I/O failure risk.
- Extract the clean subset from the local tarball using metadata-derived target
  paths and validate every raster.
- Save clean subset manifests, validation reports, prediction tables, BWER
  outputs, provenance, and handoff zips back to Drive.
- Do not use earlier streaming partial-extraction experiments as formal data.

Files to preserve for reproducibility:

- final dataset archive:
  `/content/drive/MyDrive/rsfm_fairness_audit/fmow_sentinel_clean_subset_30k_location_disjoint_v2.zip`
- final manifest:
  `final_clean_subset_manifest_30k_location_disjoint_v2.csv`
- final enriched metadata
- country-region map
- `target_paths.csv`
- `include_list.txt`
- `clean_subset_manifest.csv`
- `raster_validation_report.csv`
- prediction tables
- BWER outputs
- `archive_manifest.json`
- handoff zip
