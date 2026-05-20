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

Metadata-only mode:

```bash
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel \
  --metadata-csv path/to/fmow_sentinel.csv \
  --output-dir outputs/fmow_sentinel_preflight/run1 \
  --metadata-only
```

Preflight with subset generation and optional raster inspection:

```bash
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel \
  --metadata-csv path/to/fmow_sentinel.csv \
  --output-dir outputs/fmow_sentinel_preflight/run1 \
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
