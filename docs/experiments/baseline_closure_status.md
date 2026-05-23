# Baseline Closure Status

Recorded: 2026-05-24.

This document closes the current Sen1Floods11 and fMoW-Sentinel baseline phase
as a reproducible, protocol-aware evidence package. It does not add a new model
matrix, new dataset family, or new mainline experiment.

## Closure Judgment

The baseline closure package is closed for the current paper foundation:

- Sen1Floods11 supports a native segmentation case study of disaster-event
  tail risk.
- fMoW-Sentinel supports a modest-performance geography-tail-risk and
  model-ranking-divergence case study on a 30k location-disjoint audit subset.
- DOFA unscaled outputs are invalid protocol artifacts and are excluded from
  formal comparison.

The remaining items are not blockers for the current claims, but they must
remain visible as limitations: DOFA pooling/fine-tuning recipe is unresolved,
final per-band raster statistics should be referenced from the archived
preflight outputs when writing the paper, and tiny-overfit/random-split fMoW
checks are awaiting Colab artifact runs and should remain diagnostic-only.

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
| LOEO | partial but sufficient | `docs/reproduction/sen1floods11_closure.md`; `docs/experiments/scientific_findings.md` | Vanilla U-Net LOEO is complete; S2 ResNet34-U-Net LOEO is not required for current claims. |

Claim guardrail: Sen1Floods11 supports event-level deployment-tail-risk evidence
for this case study. It must not be generalized to all disaster segmentation,
all flood mapping, or all remote-sensing disaster settings without additional
datasets and protocols.

## fMoW-Sentinel Closure

| item | status | evidence | closure note |
| --- | --- | --- | --- |
| clean subset | done | `docs/datasets/fmow_sentinel.md`; `docs/experiments/fmow_step3_analysis_plan.md` | Final self-contained prepared archive: `prepared_zips/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged.zip`. |
| split policy | done | `docs/experiments/fmow_step3_analysis_plan.md` | `split_protocol=location_disjoint`; group key is `category + location_id`. |
| leakage check | done | `docs/experiments/scientific_findings.md`; `docs/datasets/fmow_sentinel.md` | Recorded train/val group overlap is zero. |
| support record | done | `docs/datasets/fmow_sentinel.md`; `docs/experiments/fmow_step3_analysis_plan.md` | Primary formal slices: continent, un_region, region, latitude_band, season, category. Country requires support thresholds. |
| ResNet-50 baseline | done | `docs/experiments/scientific_findings.md`; `docs/experiments/fmow_step3_scientific_findings.md` | Formal artifact is `final_step3/resnet50_30k_location_disjoint_patched_metadata.zip`. Do not use unpatched ResNet outputs as the provenance source. |
| DOFA scaled baseline | done | `docs/experiments/fmow_step3_scientific_findings.md`; `configs/models/dofa_fmow_sentinel.yaml` | Formal artifact is `final_step3/dofa_scaled10000_30k_location_disjoint.zip`. Unscaled DOFA is invalid/debug only. |
| ResNet vs DOFA comparison | done | `docs/experiments/fmow_step3_scientific_findings.md` | Formal artifact is `final_step3/comparison_resnet50_vs_dofa_scaled10000.zip`. |
| final bundle | done | `docs/experiments/fmow_step3_scientific_findings.md`; `docs/experiments/fmow_step3_analysis_plan.md` | Final bundle: `outputs/final_step3/fmow_step3_final_bundle_30k_location_disjoint.zip`. |

Final fMoW dataset and artifact paths:

```text
/content/drive/MyDrive/rsfm_fairness_audit/prepared_zips/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged.zip
/content/drive/MyDrive/rsfm_fairness_audit/outputs/final_step3/resnet50_30k_location_disjoint_patched_metadata.zip
/content/drive/MyDrive/rsfm_fairness_audit/outputs/final_step3/dofa_scaled10000_30k_location_disjoint.zip
/content/drive/MyDrive/rsfm_fairness_audit/outputs/final_step3/comparison_resnet50_vs_dofa_scaled10000.zip
/content/drive/MyDrive/rsfm_fairness_audit/outputs/final_step3/fmow_step3_final_bundle_30k_location_disjoint.zip
```

`v3_merged` is the final self-contained clean 30k location-disjoint dataset
archive. Earlier `v1` and `v2` archives are not formal reproduction data:
`v1` is an initial subset, and `v2` existed before the old 10k image-path issue
was fixed by merging the missing image tree.

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
| pooling/fine-tuning recipe | awaiting_colab_run | `docs/experiments/fmow_step3_scientific_findings.md` | Pooling and fine-tuning remain open until author examples or a cached-embedding ablation is deliberately run. |

## Baseline Closure Sanity Checks

| check | status | evidence/output | interpretation |
| --- | --- | --- | --- |
| location-disjoint leakage | done | `docs/experiments/scientific_findings.md`; `docs/datasets/fmow_sentinel.md` | Recorded train/val overlap for `category + location_id` is zero. |
| class/label mapping | done | `docs/experiments/fmow_step3_scientific_findings.md`; final ResNet archive | ResNet metadata was patched with reconstructed `class_mapping`; DOFA run metadata records class mapping. |
| embedding cache key includes input scale | done | `src/rsfm_fairness_audit/fmow_sentinel_classification.py`; `docs/results/baseline_closure_sanity/baseline_sanity_report.md` | Prevents accidental reuse of unscaled embeddings for scaled DOFA. |
| band profile and wavelength list | done as code/protocol check | `src/rsfm_fairness_audit/band_profiles.py`; `docs/results/baseline_closure_sanity/baseline_sanity_report.md` | Expected 13-band profile is recorded. Actual TIFF channel order should still be cited from final raster inspection artifacts in paper text. |
| final artifact verification | awaiting_colab_run | `docs/results/baseline_closure_sanity/baseline_sanity_report.md` | The sanity runner accepts explicit prepared/resnet/dofa/comparison/final-bundle zip paths so Colab can verify it is reading final artifacts. |
| per-band statistics / input range | partial | `docs/datasets/fmow_sentinel.md` names `band_statistics_sample.csv`; final artifact path should be cited from preflight zip | The workflow supports this. The final paper should cite the exact archived stats file if used. |
| DOFA pooling ablation | awaiting_colab_run | `docs/results/baseline_closure_sanity/baseline_sanity_report.md` | Current adapter flattens multi-dimensional `forward_features`; CLS/mean pooling are not exposed as formal outputs in the current completed run. |
| tiny overfit test | awaiting_colab_run | `docs/results/baseline_closure_sanity/baseline_sanity_report.md` | Diagnostic only; do not present as a main result. |
| random split sanity | awaiting_colab_run | `docs/results/baseline_closure_sanity/baseline_sanity_report.md` | Diagnostic only; do not present as a main result. |

## Claims Guardrails

- Do not claim DOFA is universally fairer.
- Do not claim fMoW high accuracy hides risk; current fMoW models are modest
  accuracy.
- Do not write fMoW as an RSFM high-performance failure story.
- Do not treat unscaled DOFA as a formal failed result.
- Do not use `v1`, pre-merge `v2`, handoff-only, unpatched, or debug outputs as
  formal fMoW artifacts.
- Do not treat country x class / country x category as formal unless
  support-threshold filtered.
- Do not generalize Sen1Floods11 event-level tail risk to all disaster
  segmentation or all flood settings.
- Do not mix BigEarthNet/reBEN, DOFA fine-tuning, or new model families into
  this closure package.

## Exit Criteria

Baseline closure is complete when the following are true:

- formal result records exist for Sen1Floods11 and fMoW-Sentinel;
- final Drive artifact paths are documented and point to `v3_merged`,
  `patched_metadata`, `scaled10000`, and the formal comparison zip;
- debug artifacts are explicitly excluded from formal claims;
- known sanity checks are either completed, implemented as lightweight artifact
  checks, or recorded as awaiting Colab diagnostic runs;
- claims are bounded to the evidence actually produced.

Current status: closed for Step 1 drafting after Colab verifies final artifact
paths with the baseline sanity runner.
