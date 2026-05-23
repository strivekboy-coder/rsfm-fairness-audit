# fMoW-Sentinel Step 3 Analysis Plan

This document pre-registers the lightweight Step 3 result contract and
analysis scope. It is not a results report and should not be edited to add
scientific conclusions.

## Task

- Dataset: `fmow_sentinel`
- Task: Sentinel-2 scene classification
- Input: Sentinel-2 13-band image-only raster input
- Input mode: `s2_13band_image_only`
- Band profile: `sentinel2_13band_fmow`
- Formal Step 3 subset:
  `fmow_sentinel_clean_subset_30k_location_disjoint_v2`
- Manifest:
  `/content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv`
- Drive archive:
  `/content/drive/MyDrive/rsfm_fairness_audit/prepared_zips/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged.zip`
- Final Step 3 result bundle:
  `/content/drive/MyDrive/rsfm_fairness_audit/outputs/final_step3/fmow_step3_final_bundle_30k_location_disjoint.zip`

Geography metadata is used only for audit slicing, support diagnostics, BWER
reporting, and comparison reports. Country, region, latitude, longitude,
timestamp, and location identifiers are not model inputs unless a separate
metadata-aware protocol is explicitly implemented and labeled.

## Dataset Protocol

The final Step 3 subset is a support-aware clean subset extracted from the
official Stanford PURL `fmow-sentinel.tar.gz`. The full tarball is downloaded
to Colab local `/content` storage and is not fully extracted. Earlier streaming
partial-extraction experiments are excluded from formal data.

The split protocol is `location_disjoint`: final `split` values are
`train` / `val`, `split_original` preserves the source split, and the group key
is `category + location_id`. Train/val location overlap is zero. `sample_id`
was regenerated to be unique in the final manifest.

Recorded dataset facts:

- total rows: 30000
- train rows: 21046
- val rows: 8954
- categories: 62
- validation category coverage: all 62 categories
- minimum validation category support: 33
- country coverage: 195 countries
- country missing ratio: 0
- continent / un_region / region missing ratio: 0.024
- season missing ratio: 0
- latitude_band missing ratio: 0
- validation countries with >=20 samples: 145
- validation countries with >=30 samples: 105
- dirty country codes retained for provenance:
  `ambiguous_country`, `ANT`, `KO-`, `CA-`

## Primary Metrics

- Accuracy
- Balanced accuracy, if available
- Macro-F1, if available
- Raw-BWER over geography slices
- Class-standardised geography BWER where support allows

## Primary Slices

- `country`
- `continent`
- `un_region`
- `region`
- `latitude_band`
- `season`
- `category`
- `region x category`
- `country x category`, if support allows

For the finalized 30k location-disjoint subset, primary formal slices are
`continent`, `un_region`, `region`, `latitude_band`, `season`, and `category`.
Country-level BWER should use support thresholds such as validation support
`>=20` or `>=30`. `country x category` is diagnostic-only unless
support-threshold filtered. `region x category`, `season x category`, and
`latitude_band x category` may be used for standardisation where preflight
support allows; otherwise report them as diagnostics or sensitivity checks.

## Ranking Mismatch

A ranking mismatch is present when model ordering by aggregate score differs
from model ordering by Raw-BWER or class-standardised geography BWER. Report
aggregate performance and BWER together; BWER is a tail-risk dispersion
diagnostic, not a replacement for accuracy.

## Support Limitations

Report a slice as support-limited when:

- required geography fields are missing at nontrivial rates;
- slice support is below configured BWER thresholds;
- class-standardised event/country x class cells are sparse or missing;
- missing-balance policy materially changes the result.

Support-limited rows can be diagnostic-only and should not be used as formal
paper-grade claims.

## Non-Claims

Do not claim:

- causal geography bias;
- demographic fairness;
- image-level exact geography join;
- metadata-aware model behavior;
- global production-scale performance from subset prototypes;
- scientific conclusions from validation/packaging checks alone.

## Formal vs Diagnostic Reporting

Formal BWER candidates require adequate support, valid slice variables, valid
balance variables, and documented protocol labels. Low-support geography x
class outputs, sparse country cells, or runs with incomplete provenance should
be marked diagnostic-only until the corresponding support and validation checks
pass.

## Final Bundle Contract

The final Step 3 bundle is a handoff archive, not a raw-data archive. It should
contain run outputs, comparison outputs, metadata/provenance reports, and BWER
tables, but not the full fMoW-Sentinel tarball or a fully extracted raster tree.

Current final bundle:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/final_step3/fmow_step3_final_bundle_30k_location_disjoint.zip
```

Expected contents:

- `final_step3/resnet50_30k_location_disjoint_patched_metadata.zip`
- `final_step3/dofa_scaled10000_30k_location_disjoint.zip`
- `final_step3/comparison_resnet50_vs_dofa_scaled10000.zip`

The ResNet-50 archive includes patched metadata with reconstructed
`class_mapping`. The DOFA archive must be the scaled `input_scale = 10000`
linear-probe run; unscaled DOFA outputs are debug artifacts and should not be
included as formal comparison results.
