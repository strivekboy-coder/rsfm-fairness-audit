# Optimization 1–7 final scientific result review

Date: 2026-08-16

Review mode: Drive result validation, scientific interpretation, and final evidence freeze

Training or experiment execution in this review: none

## Material Passport

- reBEN label-budget Drive folder: `1RjsLnJuioor9Ky5cQlmSXRV8fVucrk7z`
- reBEN TerraMind paired-shift Drive folder: `1VIb54N9PmIwZgNFxr4a4Ewa5vN0BmctE`
- Label-budget audit: `pass`; 15/15 seed-budget rows, nested selection, fixed validation/test panels, validation-locked thresholds, and no test-set selection.
- TerraMind paired-shift audit: `pass`; 119,825 paired test samples, identical sample ordering/targets/metadata, the same S2-trained head within seed, and no test-set selection.
- CROMA confirmatory paired-shift audit: `pass`; the same S2-ID → S1-OOD estimand, seed labels, paired panel, locked-head logic, and 19-label universe passed the registered gates.
- Paired probability diagnostic audit: `pass`; both source-completeness and diagnostic-completeness gates are true.
- Final evidence status: `complete`; `finality=true` for items 1–7. Items 8–17 were not started.

## Main-text-level findings

### Label-budget sensitivity

The valid headline is early saturation with non-monotonic geographic fairness, not “more labels are useless.” From 5% to 100% labelled training units, three-seed mean macro AP increases by 0.0192 and macro F1 by 0.0144, while mean risk, tail risk, and GeoBWER decrease by 0.0058, 0.0114, and 0.0056 respectively. The 50% point is already close to 100% macro AP and is slightly better on macro F1 and all three fairness summaries. GeoBWER does not improve monotonically, and seed 42 worsens from 5% to 100%. Thus aggregate predictive performance saturates early, whereas tail disparity remains sensitive to the sampled training subset and seed.

### Paired S2 ID → S1 OOD shift

The unchanged S2-trained TerraMind head suffers a large paired S1 degradation: mean risk increases by 0.3088, tail risk by 0.3555, and GeoBWER by 0.0467. All three deltas are positive for all three seeds. Tail degradation exceeds mean degradation in every seed, so the shift both lowers average performance and disproportionately increases geographic tail burden.

The probability diagnostics materially strengthen the mechanism interpretation. Mean label AUROC drops by 0.376–0.394 across seeds and mean AP by 0.484–0.493; no seed-label case is classified as threshold-shift-dominant. At the modal label level, 11/19 labels show a representation-collapse signature and 8/19 show mixed/partial degradation; 0/19 are threshold-only and 0/19 are stable. Therefore OOD F1 risk near one cannot be explained as a mere locked-threshold artifact. The evidence supports pervasive rank/separation failure, often combined with large score displacement.

The most revealing finding is that the same sensor shift produces opposing failure directions hidden by macro averages. Urban fabric, industrial/commercial units, and inland wetlands move toward near-all-positive prediction, whereas arable land, pastures, complex cultivation, broad-leaved forest, coniferous forest, and marine waters move toward all- or nearly-all-negative prediction. This is a stronger scientific result than “performance drops”: cross-sensor incompatibility redistributes errors by semantic class and changes which geographic slices carry the burden.

`representation_collapse_signature` remains an operational frozen-head diagnosis based on AUROC/AP, score separation, variance/contraction, and threshold behavior. It is not causal proof that the TerraMind encoder itself collapsed.

## Labelwise scientific interpretation

Values are three-seed means. `ΔPPR` is the S1-OOD minus S2-ID predicted-positive rate; crossing is the paired locked-threshold crossing rate.

| Label | ΔAUROC | ΔAP | ΔPPR | Crossing | OOD F1 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Urban fabric | -0.352 | -0.670 | +0.901 | 0.901 | 0.193 | Collapse signature plus near-all-positive saturation; both ranking and operating point fail. |
| Industrial/commercial | -0.304 | -0.417 | +0.962 | 0.962 | 0.034 | Mixed failure dominated by extreme upward score shift, but AUROC/AP loss rules out pure threshold shift. |
| Arable land | -0.318 | -0.424 | -0.431 | 0.431 | 0.000 | Collapse signature with all-negative behavior; high support makes this main-text evidence. |
| Permanent crops | -0.482 | -0.517 | +0.200 | 0.271 | 0.038 | Mixed rank failure and upward threshold movement; not a calibration-only case. |
| Pastures | -0.550 | -0.670 | -0.212 | 0.212 | 0.000 | Strong collapse signature and all-negative prediction. |
| Complex cultivation | -0.275 | -0.395 | -0.243 | 0.243 | 0.000 | Moderate rank loss plus downward score shift yields all-negative prediction. |
| Agriculture with natural vegetation | -0.235 | -0.375 | +0.302 | 0.448 | 0.440 | One of the better-retained labels; mixed degradation with substantial threshold churn. |
| Agro-forestry | -0.656 | -0.788 | +0.227 | 0.392 | 0.026 | Catastrophic rank reversal/separation loss despite an upward score shift; strongest AUROC collapse. |
| Broad-leaved forest | -0.352 | -0.513 | -0.328 | 0.328 | 0.000 | Collapse signature with all-negative prediction. |
| Coniferous forest | -0.469 | -0.631 | -0.335 | 0.335 | 0.000 | Collapse signature with all-negative prediction and severe separation loss. |
| Mixed forest | -0.383 | -0.507 | +0.065 | 0.464 | 0.419 | Mixed degradation; some F1 remains, but almost half of paired samples cross the threshold. |
| Natural grassland/sparse vegetation | -0.320 | -0.361 | -0.012 | 0.012 | 0.000 | Collapse signature with almost no threshold crossings: direct evidence against a threshold-only explanation. |
| Moors/heathland/sclerophyllous | -0.632 | -0.545 | -0.017 | 0.043 | 0.006 | Collapse signature in two seeds and severe rank loss with little threshold movement. |
| Transitional woodland/shrub | -0.246 | -0.351 | +0.203 | 0.442 | 0.441 | Relatively retained mixed case; score shift and ranking loss both matter. |
| Beaches/dunes/sands | -0.275 | -0.037 | -0.001 | 0.001 | 0.000 | Collapse signature, but only 152 positives; retain as appendix/descriptive evidence. |
| Inland wetlands | -0.434 | -0.515 | +0.921 | 0.921 | 0.076 | Mixed failure with near-all-positive saturation and substantial rank loss. |
| Coastal wetlands | -0.481 | -0.318 | -0.002 | 0.002 | 0.000 | Collapse signature, but only 117 positives; appendix-only. |
| Inland waters | -0.151 | -0.420 | +0.722 | 0.728 | 0.275 | Best retained AUROC (0.819 OOD), yet large upward shift breaks the locked operating point; mixed, not threshold-only. |
| Marine waters | -0.420 | -0.821 | -0.099 | 0.099 | 0.000 | Clean high-support collapse example: near-perfect ID performance becomes all-negative OOD with the largest separation loss. |

## Appendix-level findings

- Seed-level category counts strengthen reproducibility: collapse/mixed counts are 10/9, 12/7, and 13/6 for seeds 42, 73, and 101; threshold-only and stable counts are zero in every seed.
- All ten reported countries worsen in every seed. Country burden tables are valuable localization evidence, but individual country ordering remains descriptive without uncertainty intervals and multiplicity control.
- Full per-label confusion counts, score transport, probability Wasserstein distance, score margins, crossing direction, and probability plots belong in the appendix and evidence package.
- Beaches/dunes/sands and coastal wetlands should not support headline label claims because their positive supports are 152 and 117 respectively.
- Label-budget seed trajectories and the non-monotonic GeoBWER path should be shown in the appendix even if the main text emphasizes the 5%, 50%, and 100% trade-off.

## Remaining limitations

1. The paired estimand is cross-modal compatibility of an unchanged S2-trained head with paired S1 representations. It is not conventional retrained S1 robustness, a causal sensor effect, or EarthShift effective robustness.
2. Mechanism evidence is diagnostic rather than causal: frozen-head rank/separation failure can arise from feature-space misalignment, encoder preprocessing incompatibility, or other representation changes.
3. The paired result now has same-task replication in CROMA, but only one task and one paired sensor shift. This supports model-dependent failure geometry for this shift, not universal cross-model or cross-task generality.
4. Label and country comparisons are descriptive. With only three seeds, do not make formal label-order or country-order significance claims; multiplicity correction would be required for inferential slice testing.
5. The 5% label budget still contains 11,894 samples and the stored independent-unit key is effectively sample-level. This is label-efficiency sensitivity, not few-shot or multi-chip location-budget evidence.
6. Source-tile overlap exists across train/validation/test while declared independent-unit overlap is zero. This does not invalidate the paired within-test contrast, but out-of-location claims must follow and disclose the independent-unit definition.

## Confirmatory CROMA cross-model update

CROMA confirms a large common rank/separation failure but a different tail and label geometry. Its three-seed means are Δmean=0.2686, Δtail=0.2727, ΔGeoBWER=0.0041, ΔAUROC=-0.3790, and ΔAP=-0.4440. TerraMind gives Δmean=0.3088, Δtail=0.3555, ΔGeoBWER=0.0467, ΔAUROC=-0.3861, and ΔAP=-0.4882. Thus AUROC degradation is nearly the same, but CROMA has materially less AP/F1 loss and almost no systematic excess tail acceleration. One CROMA seed has negative ΔGeoBWER; all TerraMind seeds are positive.

The label diagnostics also differ. CROMA has 19/19 modal mixed/partial signatures and no modal collapse label, whereas TerraMind has 11/19 collapse and 8/19 mixed. Neither model has threshold-only or stable labels. Several score-transport directions reverse: urban fabric is approximately -0.047 in CROMA versus +0.901 in TerraMind; pastures +0.328 versus -0.212; natural grass/sparse vegetation +0.430 versus -0.012; and marine waters +0.411 versus -0.099. These observations support the bounded statement that the same paired sensor shift has different **frozen-head failure geometry** across models: a shared rank-collapse phenomenon, but model-specific operating-point displacement and geographic tail allocation. They do not identify encoder causality.

## Revised extension decision

Item #8 adaptation is now the higher-priority mechanistic experiment. Both models lose approximately 0.38 mean AUROC, while neither shows a threshold-only signature; threshold recalibration alone is therefore unlikely to recover the failure. A locked adaptation ladder—threshold recalibration, frozen-encoder S1 head refit, then representation adaptation only if needed—would quantify recoverability without confusing mechanisms.

Item #9 model×task 2×2 remains conditional and moves below #8. The completed two-model, one-task comparison already supports model-dependent geometry for reBEN S2→S1. A second task is valuable only if the manuscript claims that model×shift geometry generalizes across tasks, or if review requires that interaction. It should not precede the cheaper and more diagnostic adaptation ladder.

## Finality decision

- Items 1–7 formal experiment completion: `complete`.
- Probability-diagnostic gate: `pass`.
- Final evidence status: `complete`.
- `finality=true`.
- Items 8–17: not started.
