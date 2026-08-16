# Optimization 1–7 scientific result review

Date: 2026-08-16

Review mode: result validation and reproducibility reconciliation

Training performed in this review: none

## Material Passport

- reBEN label-budget Drive folder: `1RjsLnJuioor9Ky5cQlmSXRV8fVucrk7z`
- reBEN TerraMind paired-shift Drive folder: `1VIb54N9PmIwZgNFxr4a4Ewa5vN0BmctE`
- Label-budget audit: pass, 15/15 seed-budget rows, nested independent-unit selection, fixed validation/test, and no test-set selection.
- Paired-shift audit: pass under the experiment-time schema, with formal preflight, 119,825 paired test samples, identical paired ordering/targets/metadata, the same S2-trained head within each seed, and no test-set selection.
- Current-code reconciliation: both formal experiments are complete. The final evidence freeze remains pending only until the saved probabilities are processed by the newer no-retraining probability diagnostic gate.

## Label-budget validity

The experiment is valid as a descriptive nested-budget sensitivity study. It uses the same fixed validation and test panels at every budget and seed, chooses thresholds on validation, and preserves nested training subsets. It does not establish a population-level learning-curve law or statistical significance from three seeds.

From 5% to 100% labelled training units, the three-seed means change as follows:

| Metric | 5% | 100% | Change |
|---|---:|---:|---:|
| Macro AP | 0.6616 | 0.6808 | +0.0192 |
| Macro F1 | 0.6205 | 0.6350 | +0.0144 |
| Mean geographic risk | 0.1036 | 0.0978 | -0.0058 |
| Tail geographic risk | 0.1494 | 0.1380 | -0.0114 |
| GeoBWER | 0.0458 | 0.0403 | -0.0056 |

The strongest label-budget finding is early saturation, not a claim that more labels are useless. The 50% point has macro AP 0.6783 versus 0.6808 at 100%, while its macro F1 is slightly higher and its mean risk, tail risk, and GeoBWER are all lower. This is a useful empirical cost–performance–fairness trade-off.

The fairness path is not monotonic. GeoBWER is worse at 25% than at 5% on the seed mean, and the 100% point is worse than 50%. The 5%→100% GeoBWER endpoint improves for seeds 73 and 101 but worsens by 0.0034 for seed 42. Therefore the defensible conclusion is that aggregate performance improves consistently while geographic tail disparity is seed- and subset-sensitive.

Two limitations matter. First, 5% still contains 11,894 training samples, so this is label-efficiency sensitivity rather than few-shot learning. Second, `selected_independent_units == selected_samples` at every point, indicating that the current independent-unit key is effectively sample-level in this cache. This is acceptable if one chip is the declared independent unit, but it must not be described as a multi-sample location-group budget. The legacy output also lacks selected-label-coverage columns; this is already disclosed by the audit warning.

## Paired S2 ID → S1 OOD validity

The experiment is internally strong for its stated estimand: compatibility of one S2-trained frozen linear head with paired S2 and S1 TerraMind representations. It is not a general causal estimate of sensor shift and does not establish EarthShift-style effective robustness. The large effect cannot be attributed solely to geography or labels because each S1 observation is paired to the same S2 test sample and target.

Three-seed mean changes are:

| Metric | Result |
|---|---:|
| Macro AP drop | 0.4882 |
| Macro F1 drop | 0.5325 |
| Mean risk increase | 0.3088 |
| Tail risk increase | 0.3555 |
| GeoBWER increase | 0.0467 |
| BCE risk increase | approximately 3.12 |

All three seeds have positive mean-risk, tail-risk, and GeoBWER changes. Tail degradation exceeds mean degradation in every seed by 0.0314–0.0570. The strongest scientific finding is therefore not merely catastrophic average degradation: the sensor transfer also amplifies geographic tail burden.

All ten reported countries worsen in all three seeds. Belgium and Ireland carry the largest recurrent country-level increases, although Belgium has materially less support than the largest countries. The pattern is not driven only by tiny label classes: several high-support labels, including arable land, pastures, broad-leaved forest, coniferous forest, and marine waters, reach OOD F1 risk 1.0. Marine waters is especially striking because its ID risk is approximately 0.014 but its OOD risk is 1.0 across all seeds.

## Why OOD risk approximately 1 is not yet “representation collapse”

The reported label risk is `1 − F1` at an S2-validation-locked threshold. A value of 1 therefore says F1 is zero; it does not identify why. It can result from all-negative predictions, all-positive predictions with no true-positive overlap, a large calibration/score offset across the locked threshold, lost class ranking, or genuinely contracted/non-separating frozen representations.

The existing saved probability arrays and targets are sufficient to distinguish these operational cases without training:

- Stable AUROC/AP with a large predicted-positive-rate change and many threshold crossings supports a threshold/score-shift-dominant interpretation.
- Large AUROC/AP loss plus reduced positive-versus-negative score separation and score-variance contraction supports a `representation_collapse_signature`.
- Mixed changes should remain labelled mixed; probability diagnostics alone cannot causally assign the failure to the encoder.

The current repository implements these diagnostics and additionally records locked-threshold F1, TP/FP/FN/TN, all-negative/all-positive flags, score margins, probability transport, and crossing direction. The Drive result predates this postprocessor, so the scientific answer must remain unresolved until `--postprocess-only` materializes the diagnostic tables from the already saved arrays.

## Is a second model needed?

Yes, as a targeted confirmatory replication, but it should not block reporting this TerraMind case study if the claim is model-specific. CROMA is the preferred second model because the Drive already contains aligned S1 and S2 frozen caches and CROMA is explicitly designed around SAR–optical representation learning. It is the most diagnostic comparator for whether the failure is TerraMind-specific feature-space misalignment or a broader limitation of unchanged-head cross-sensor transfer. A conventional ResNet is less clean because modality/input construction differences introduce a larger protocol confound.

Do not launch CROMA before the current probability diagnostic is read. If the result is threshold-shift-dominant, a second model tests calibration-transfer robustness; if rank/separation collapses, it tests cross-modal representation alignment. One CROMA replication is sufficient; a broad second-model panel is not justified at this stage.

## Statistical and inferential audit

Confidence: **SOLID** for experiment completion and the direction/magnitude of descriptive paired degradation; **CAUTION** for generalization beyond TerraMind and for mechanism attribution.

All 11 experiment-agent fallacy checks were considered. No aggregate-to-country Simpson reversal is visible because every country delta is positive. No individual-level or causal claim is made from group summaries. Base rates remain relevant for per-label AP/F1 and are retained through positive support. There is no pre/post regression-to-mean design or survivor filtering. All 19 labels are retained, avoiding selective look-elsewhere reporting, but any future inferential label tests require multiplicity control. The protocol was frozen before this review, limiting forking-path risk. Correlation/causation and reverse-causality claims are not made.

One reproducibility caution remains: the cache contract reports source-tile overlap across train/validation/test while independent-unit overlap is zero. This was disclosed rather than hidden. It does not invalidate the paired S1/S2 within-test contrast, but manuscript claims about out-of-location generalization should follow the independent-unit split definition and disclose the coarser source-tile overlap.

## Finality decision

- Formal experiment completion for items 1–7: complete.
- Current evidence-package finality: `pending_no_retraining_probability_diagnostics`.
- Remaining action: run the current paired-shift postprocessor only; no training, threshold selection, embedding extraction, or items 8–17 activity is required.
