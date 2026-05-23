# Step 0 Closure Status

Recorded: 2026-05-24.

This document closes the current Sen1Floods11 and fMoW-Sentinel baseline phase
as a reproducible, protocol-aware evidence package. It does not add a new model
matrix, new dataset family, or new mainline experiment.

## Closure Judgment

Step 0 is closed for the current paper foundation:

- Sen1Floods11 supports a native segmentation case study of disaster-event
  tail risk.
- fMoW-Sentinel supports a modest-performance geography-tail-risk and
  model-ranking-divergence case study on a 30k location-disjoint audit subset.
- DOFA unscaled outputs are invalid protocol artifacts and are excluded from
  formal comparison.

The remaining items are not blockers for Step 0 claims, but they must remain
visible as limitations: DOFA pooling/fine-tuning recipe is unresolved, final
per-band raster statistics should be referenced from the archived preflight
outputs when writing the paper, and tiny-overfit/random-split fMoW checks are
diagnostic-only sanity checks rather than current main results.

## Sen1Floods11 Closure

| item | status | evidence | closure note |
| --- | --- | --- | --- |
| native segmentation protocol | done | `docs/datasets/sen1floods11.md`; `docs/experiments/scientific_findings.md` | Formal path is native pixel-level segmentation, not chip-level classification sanity. |
| event aggregation | done | `docs/reproduction/bwer_v2_posthoc.md`; `docs/experiments/scientific_findings.md` | Main segmentation risk uses event-level aggregated TP/FP/FN/TN and valid-pixel counts. |
| BWER v2 | done | `docs/reproduction/bwer_v2_posthoc.md`; `docs/experiments/scientific_findings.md` | Post-hoc BWER v2 writes derived balance variables and sensitivity tables. |
| standardised BWER | done | `docs/experiments/scientific_findings.md` | Primary derived balance: `flood_extent_bin`; secondary: `invalid_pixel_ratio_bin`. |
| support threshold / missing policy | done | `docs/reproduction/bwer_v2_posthoc.md`; `src/rsfm_fairness_audit/slice_support.py`; `src/rsfm_fairness_audit/bwer_v2.py` | `renormalize` is the default missing-balance policy; overlap/invalidate are sensitivity checks. |
| formal vs debug distinction | done | `docs/experiments/scientific_findings.md`; `docs/reproduction/sen1floods11_closure.md` | Non-TL threshold and failed all-background TL attempts are debug artifacts, not formal results. |
| closure comparison | done | `docs/reproduction/sen1floods11_closure.md`; `docs/experiments/scientific_findings.md` | Four-run closure is protocol-aware, not a pure same-split architecture comparison. |
| LOEO | partial but sufficient | `docs/reproduction/sen1floods11_closure.md`; `docs/experiments/scientific_findings.md` | Vanilla U-Net LOEO is complete; S2 ResNet34-U-Net LOEO is not required for current Step 0 claims. |

Claim guardrail: Sen1Floods11 supports event-level deployment-tail-risk evidence
for this case study. It must not be generalized to all disaster segmentation,
all flood mapping, or all remote-sensing disaster settings without additional
datasets and protocols.

## fMoW-Sentinel Closure

| item | status | evidence | closure note |
| --- | --- | --- | --- |
| clean subset | done | `docs/datasets/fmow_sentinel.md`; `docs/experiments/fmow_step3_analysis_plan.md` | Final prepared archive: `prepared_zips/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged.zip`. |
| split policy | done | `docs/experiments/fmow_step3_analysis_plan.md` | `split_protocol=location_disjoint`; group key is `category + location_id`. |
| leakage check | done | `docs/experiments/scientific_findings.md`; `docs/datasets/fmow_sentinel.md` | Recorded train/val group overlap is zero. |
| support record | done | `docs/datasets/fmow_sentinel.md`; `docs/experiments/fmow_step3_analysis_plan.md` | Primary formal slices: continent, un_region, region, latitude_band, season, category. Country requires support thresholds. |
| ResNet-50 baseline | done | `docs/experiments/scientific_findings.md`; `docs/experiments/fmow_step3_scientific_findings.md` | `resnet50_13band_from_scratch`, supervised baseline. |
| DOFA scaled baseline | done | `docs/experiments/fmow_step3_scientific_findings.md`; `configs/models/dofa_fmow_sentinel.yaml` | `input_scale=10000`, frozen encoder linear probe. |
| ResNet vs DOFA comparison | done | `docs/experiments/fmow_step3_scientific_findings.md` | Aggregate ranking and geography-BWER ranking differ for some slices. |
| final bundle | done | `docs/experiments/fmow_step3_scientific_findings.md`; `docs/experiments/fmow_step3_analysis_plan.md` | Final bundle: `outputs/final_step3/fmow_step3_final_bundle_30k_location_disjoint.zip`. |

Final bundle contents:

- `final_step3/resnet50_30k_location_disjoint_patched_metadata.zip`
- `final_step3/dofa_scaled10000_30k_location_disjoint.zip`
- `final_step3/comparison_resnet50_vs_dofa_scaled10000.zip`

Claim guardrail: fMoW-Sentinel currently supports modest-performance
geography-tail-risk and model-ranking-divergence evidence. It should not be
written as a high-performance RSFM failure story, a full fMoW-Sentinel benchmark
result, or a causal geography fairness claim.

## DOFA Protocol Guardrails

| item | status | evidence | closure note |
| --- | --- | --- | --- |
| author-confirmed input scale requirement | done | `docs/experiments/fmow_step3_scientific_findings.md` | Inputs should be normalized to `[0,1]` or `[-1,1]`. |
| formal fMoW scaling | done | `configs/models/dofa_fmow_sentinel.yaml`; `src/rsfm_fairness_audit/adapters/dofa.py` | `input_scale=10000` applies `x = x / input_scale` before DOFA normalization and embedding extraction. |
| unscaled DOFA status | done | `docs/experiments/fmow_step3_scientific_findings.md` | Unscaled DOFA is invalid/debug/protocol artifact, not a formal result. |
| band order vs wavelength | partial but documented | `src/rsfm_fairness_audit/band_profiles.py`; `docs/experiments/fmow_step3_scientific_findings.md` | Band order itself is not critical; wavelength-band correspondence is critical. Current profile records the expected 13-band correspondence. |
| embedding cache key | done | `src/rsfm_fairness_audit/fmow_sentinel_classification.py` | Cache key includes `input_scale`, `image_size`, `band_profile`, checkpoint/source, embedding layer, row hash, and manifest path. |
| pooling/fine-tuning recipe | open | `docs/experiments/fmow_step3_scientific_findings.md` | Pooling and fine-tuning remain open until author examples or clearer guidance are released. |

## Step 0 Sanity Checks

| check | status | evidence/output | interpretation |
| --- | --- | --- | --- |
| location-disjoint leakage | done | `docs/experiments/scientific_findings.md`; `docs/datasets/fmow_sentinel.md` | Recorded train/val overlap for `category + location_id` is zero. |
| class/label mapping | done | `docs/experiments/fmow_step3_scientific_findings.md`; final ResNet archive | ResNet metadata was patched with reconstructed `class_mapping`; DOFA run metadata records class mapping. |
| embedding cache key includes input scale | done | `src/rsfm_fairness_audit/fmow_sentinel_classification.py`; `docs/results/step0_closure_sanity/step0_sanity_report.md` | Prevents accidental reuse of unscaled embeddings for scaled DOFA. |
| band profile and wavelength list | done as code/protocol check | `src/rsfm_fairness_audit/band_profiles.py`; `docs/results/step0_closure_sanity/step0_sanity_report.md` | Expected 13-band profile is recorded. Actual TIFF channel order should still be cited from final raster inspection artifacts in paper text. |
| per-band statistics / input range | partial | `docs/datasets/fmow_sentinel.md` names `band_statistics_sample.csv`; final artifact path should be cited from preflight zip | The workflow supports this. The final paper should cite the exact archived stats file if used. |
| DOFA pooling ablation | formally open | `docs/results/step0_closure_sanity/step0_sanity_report.md` | Current adapter flattens multi-dimensional `forward_features`; CLS/mean pooling are not exposed as formal outputs in the current completed run. |
| tiny overfit test | formal optional diagnostic | `docs/results/step0_closure_sanity/step0_sanity_report.md` | Not required for current modest-performance claims; run only if diagnosing training bugs. |
| random split sanity | formal optional diagnostic | `docs/results/step0_closure_sanity/step0_sanity_report.md` | Not a main result; useful only to illustrate location-disjoint difficulty if needed. |

## Claims Guardrails

- Do not claim DOFA is universally fairer.
- Do not claim fMoW high accuracy hides risk; current fMoW models are modest
  accuracy.
- Do not write fMoW as an RSFM high-performance failure story.
- Do not treat unscaled DOFA as a formal failed result.
- Do not treat country x class / country x category as formal unless
  support-threshold filtered.
- Do not generalize Sen1Floods11 event-level tail risk to all disaster
  segmentation or all flood settings.
- Do not mix BigEarthNet/reBEN, DOFA fine-tuning, or new model families into
  Step 0.

## Step 0 Exit Criteria

Step 0 is closed when the following are true:

- formal result records exist for Sen1Floods11 and fMoW-Sentinel;
- final Drive artifact paths are documented;
- debug artifacts are explicitly excluded from formal claims;
- known sanity checks are either completed, implemented as lightweight
  artifact checks, or recorded as open optional diagnostics;
- claims are bounded to the evidence actually produced.

Current status: closed for Step 1 drafting, with the limitations above carried
forward.
