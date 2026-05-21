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

Geography metadata is used only for audit slicing, support diagnostics, BWER
reporting, and comparison reports. Country, region, latitude, longitude,
timestamp, and location identifiers are not model inputs unless a separate
metadata-aware protocol is explicitly implemented and labeled.

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
