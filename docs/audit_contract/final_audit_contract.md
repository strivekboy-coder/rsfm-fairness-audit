# Final Audit Contract

This document defines the minimum prediction and audit-table contract for
post-hoc BWER, robustness, selective-risk, and future conformal selective BWER
analysis. It is a framework contract, not a new experiment protocol.

## Scope

The contract applies to completed and future RSFM fairness-audit outputs:

- single-label classification;
- multi-label classification;
- segmentation;
- GEE / AlphaEarth-style tabular embedding experiments.

The contract is intentionally output-oriented. BWER operates on normalized audit
tables and task-specific risk primitives, not on raw pixels or raw embeddings.

## Shared Required Fields

All task outputs should preserve:

| Field | Requirement |
|---|---|
| `sample_id` | Stable unique row identifier. For segmentation, use chip/sample row identifiers and preserve event aggregation identifiers separately. |
| `split` | Evaluation split such as `train`, `val`, `test`, `calibration`, or `all`. |
| `dataset` | Dataset identifier. |
| `task_type` | One of `single_label_classification`, `multi_label_classification`, `segmentation`, or `tabular_embedding_classification`. |
| `model` / `model_family` | Model identifier sufficient for cross-run comparison. |
| `protocol` / `split_protocol` | Protocol label, including location-disjoint, official split, random split, or diagnostic status. |
| `y_true` | Ground-truth target where the task is sample- or label-row based. |
| `y_pred` | Predicted class/label/mask-derived prediction where applicable. |
| `risk` or task risk primitive | Row-level risk or enough primitives to reconstruct task-specific risk. |
| slice variables | Deployment grouping columns such as `event_id`, `country`, `region`, `latitude_band`, `season`, `class_label`, `sensor_mode`. |
| support variables | Sample support for classification; pixel/count support for segmentation. |
| provenance | Source artifact, checkpoint/source label, band profile, input scale, image size, manifest hash where available. |

## Task-Specific Fields

### Single-Label Classification

Required:

- `sample_id`, `split`, `dataset`, `task_type`, `model`, `protocol`;
- `y_true`, `y_pred`;
- one of `confidence`, `probability`, `prob_true`, `max_probability`, `logit`, or serialized probability/logit columns;
- `risk` or enough fields to derive `1[y_pred != y_true]`;
- `class_label` or equivalent balance/class variable;
- deployment slice variables and sample-level support.

Selective and conformal analysis require real score/probability/logit exports.
Thresholded predictions alone support Raw/Standardised BWER, but not calibrated
selective or conformal selective risk.

### Multi-Label Classification

Required:

- sample identifier plus `label` / `class_label` for sample-label rows;
- `y_true`, `y_pred`;
- label probability/logit or confidence score;
- `risk_bce` or BCE primitives for probability-aware risk;
- threshold and binary-error fields if binary risk is reported;
- slice variables, label support, and provenance.

BCE risk and binary error answer different questions and must remain separate.
Do not collapse them into one universal score.

### Segmentation

Required:

- chip/sample identifier and aggregation unit such as `event_id`;
- split, dataset, model, protocol, and task labels;
- `TP`, `FP`, `FN`, and preferably `TN`, or equivalent pixel-count primitives;
- `valid_pixel_count`;
- `positive_pixel_count` / positive support;
- `predicted_positive_pixel_count`;
- task risk primitive such as `1 - micro_iou` or fields to reconstruct it;
- mask/preprocessing provenance, band profile, resolution, and adapter/checkpoint source.

Segmentation support is pixel-level. Do not apply segmentation
`min_positive_support` thresholds to chip-level classification sanity outputs.

### GEE / AlphaEarth-Style Tabular Embedding Experiments

Required before formal audit:

- `sample_id`, split, and calibration indicator;
- coordinates, timestamp/year, and spatial provenance;
- land-cover/class labels;
- embedding source/checkpoint and feature protocol;
- prediction score/probability/logit;
- deployment slice variables and support variables;
- model and manifest provenance.

These fields are required for social-spatial interpretation and future
conformal selective BWER. This contract does not authorize starting new
AlphaEarth/GEE experiments before existing outputs are audited.

## Analysis Support Matrix

| Analysis | Minimum fields |
|---|---|
| Raw-BWER | Slice variable, task-specific risk or reconstructable risk, support fields. |
| Standardised-BWER | Raw-BWER fields plus valid balance variable and slice x balance support. |
| Stabilised-BWER | Raw/standardised fields plus enough support information to apply shrinkage or support-aware stabilization. |
| Bootstrap CI | Unit identifiers and row-level or slice-level resampling primitives. |
| Alpha sensitivity | Valid slice-level risk table or BWER summary recomputable for multiple tail fractions. |
| Support threshold sensitivity | Support fields and recomputable slice risks. |
| Missing-policy sensitivity | Balance-variable missingness and slice x balance cells. |
| BWER vs traditional subgroup metrics | Slice-level risks and support, or documented BWER summaries with explicit caveats. |
| Selective risk | Confidence/probability/logit and risk primitive. |
| Future conformal selective BWER | Selective-risk fields plus calibration split or calibration indicator. |
| Social-spatial interpretation | Coordinates, timestamp/year, and verified spatial metadata. |

## Missing-Field Status Vocabulary

Use these statuses in reports:

- `supports_posthoc_bwer_robustness`;
- `supports_selective_risk`;
- `requires_probability_or_logit_export`;
- `requires_calibration_split_for_conformal`;
- `requires_inference_rerun_if_checkpoint_exists`;
- `posthoc_only`;
- `diagnostic_only`;
- `missing_artifact`;
- `not_recoverable_from_current_outputs`.

Avoid vague states such as "maybe broken." Reports should explain which fields
are missing and whether the fix is post-hoc analysis, prediction export,
inference rerun, or full retraining.

## Cache and Provenance Keys

Cache keys must include protocol-changing parameters:

- `input_scale`;
- `image_size` / prepared resolution;
- `band_profile`;
- checkpoint/source;
- manifest hash;
- split protocol;
- task adapter and preprocessing profile.

Generated robustness outputs must use new directories. Do not overwrite formal
experiment artifacts while debugging contract coverage.
