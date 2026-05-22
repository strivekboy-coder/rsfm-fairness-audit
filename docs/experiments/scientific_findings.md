# Scientific Findings Log

This file records high-level paper-relevant interpretations from completed pilot experiments. It intentionally excludes raw outputs, full logs, and large result artifacts.

## BEN-GE-800 CROMA Sensor-Mode Pilot

BEN-GE-800 validates the CROMA paired SAR/optical/both pipeline and the BWER audit framework. However, due to severe climatezone x class_label sparsity and strong sensitivity to missing-balance policy, it should be treated as a sensor-mode pilot and support-diagnostics case rather than a main paper-grade climatezone fairness result.

## Full Sen1Floods11 Prithvi Classification BWER Audit

Recorded: 2026-05-14.

The full hand-labeled Sen1Floods11 Prithvi chip-level classification sanity audit successfully produced valid event-level BWER results after the task-aware taxonomy fix.

Metadata/preflight:
- 446 prepared samples.
- 11 event_id slices.
- 9 event slices with n >= 20.
- BWER(event_id), BWER(event_id | class_label), and BWER(event_id | flood_label) were all preflight-recommended and formal-BWER-runnable.
- The task-aware taxonomy correctly used sen1floods11_classification, avoiding the previous segmentation min_positive_support=1000 issue.

Main results:
- Raw BWER(event_id) = 0.2848, mean_risk = 0.3581, tail_risk = 0.6429, worst_slice = Pakistan, CI = [0.0617, 0.2961].
- Balanced BWER(event_id | class_label) = 0.2894, mean_risk = 0.3683, tail_risk = 0.6578, worst_slice = Pakistan, best_slice = India, CI = [0.1131, 0.3110].
- Balanced BWER(event_id | flood_label) produced the same result as class_label, suggesting the classification labels align with flood_label in this audit.

Interpretation:
This provides paper-prep evidence that average performance can hide event-level deployment tail risk in RSFM flood mapping evaluation. The signal persists after class/flood balancing, with Pakistan consistently identified as the worst-tail event.

Limitations:
This is still chip-level classification sanity evidence, not final pixel-level segmentation fairness. It should motivate segmentation-level and cross-model follow-up rather than be treated as the final main result.

## Sen1Floods11 Native Segmentation Readiness Diagnostics

Recorded: 2026-05-19.

The native Sen1Floods11 segmentation audit pipeline now prepares the hand-labeled Sentinel-2/LabelHand path and writes pixel-count-based event metrics for BWER. A full prepared run confirmed 446 chips and 11 `event_id` slices, matching the official hand-labeled Sen1Floods11 scale used by the Prithvi TL Sen1Floods11 model card.

Step A diagnostics on a 64-chip validation subset found that the prepared inputs are internally consistent:
- Band order is `B02,B03,B04,B05,B06,B07`.
- Label values are the expected `-1`, `0`, and `1`.
- Mask and prediction grids are aligned at the prepared resolution.
- A small number of zero-valid chips can appear and should be tracked, but event-level aggregation from valid pixels remains well-defined.

The very low IoU from the frozen non-TL Prithvi threshold head is therefore not evidence that the dataset loader, label polarity, or mask alignment is broken. It is evidence that the transparent readiness head is not a competitive flood segmentation model.

Step B diagnostic baselines on the same 64-chip subset strengthen this interpretation:
- `mean_threshold_high_positive`, the original frozen threshold convention, produced overall micro IoU around 0.028.
- The inverted threshold improved but still behaved like a diagnostic polarity test, not a defensible model.
- NDWI-like baselines using B03 against B06/B07 produced much higher overall micro IoU, with B03/B06 around 0.63 and B03/B07 around 0.50 on the validation subset.

Interpretation:
The prepared Sen1Floods11 native segmentation data path is plausible, and the official task-adapted segmentation checkpoint is the correct next formal experiment. Diagnostic NDWI-like baselines should be retained for sanity checks and appendix-level troubleshooting only; they should not be presented as the main RSFM fairness result.

Current protocol decision:
- `prithvi` remains the frozen non-TL smoke/diagnostic path and is labeled `frozen_encoder_lightweight_head`.
- `prithvi_tl_sen1floods11` is the formal Prithvi Sen1Floods11 route and is labeled `task_adapted_decoder` with `training_budget=official_sen1floods11_finetune`.

Remaining limitations:
The current prepared data cache may contain 224x224 resized chips for Colab-friendly validation. For final paper-grade reporting, the official task-adapted decoder should be run with a deliberately chosen prepared resolution and the report should state that resolution explicitly.

Update after the first official TL full run:
An initial 446-chip TL run produced all-background predictions and must not be interpreted scientifically. The failure exposed a band-profile mismatch: the older prepared NPZ cache used the non-TL compatibility bands `B02-B07`, while the official Sen1Floods11 TL checkpoint expects Sentinel-2 indices `[1,2,3,8,11,12]` corresponding to `BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`. The pipeline now has a dedicated `prithvi_tl_sen1floods11` preparation profile, rejects incompatible cached prepared data for the TL adapter, and writes model/probability debug diagnostics before any future full-run interpretation.

Additional adapter debugging:
The second all-background failure occurred after the correct TL band profile was in place. The model was loaded from the official Hugging Face checkpoint and the logits were shaped correctly, but water probabilities remained near zero. The root cause was incomplete reproduction of the official inference preprocessing: the adapter had skipped the TerraTorch datamodule `test_transform` and `aug` path after fixing a time/channel layout bug. Restoring the official preprocessing and 512x512 sliding-window inference resolved the failure.

## Official Prithvi TL Sen1Floods11 Native Segmentation Full Run

Recorded: 2026-05-19.

Protocol:
- Model: `prithvi_tl_sen1floods11`.
- Model family: Prithvi.
- Dataset/task: Sen1Floods11 native pixel-level flood segmentation.
- Prepared data: 446 hand-labeled Sentinel-2 chips, 11 events, 512x512 prepared resolution.
- Band profile: `BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`, corresponding to Sentinel-2 indices `[1,2,3,8,11,12]`.
- Adaptation protocol: `task_adapted_decoder`.
- Training budget label: `official_sen1floods11_finetune`.
- Split protocol label: `standard_split`.

Main aggregate result:
- Overall pixel-count micro IoU across all chips: 0.8052.
- Overall pixel-count micro Dice: 0.8921.
- Total valid pixels: 100,983,314.
- Total ground-truth positive pixels: 10,705,605.
- Total predicted positive pixels: 10,172,293.
- Total TP/FP/FN/TN: 9,312,469 / 859,824 / 1,393,136 / 89,417,885.

Event-level micro IoU:
- Pakistan: 0.6550.
- Bolivia: 0.6766.
- Ghana: 0.7303.
- Somalia: 0.7389.
- India: 0.7404.
- Paraguay: 0.7724.
- USA: 0.7871.
- Spain: 0.8354.
- Sri-Lanka: 0.8709.
- Nigeria: 0.8932.
- Mekong: 0.9159.

BWER(event_id):
- BWER = 0.1175.
- Mean risk = 0.2167.
- Tail risk = 0.3342.
- Worst slice = Pakistan, risk = 0.3450.
- Best slice = Mekong, risk = 0.0841.
- Tail slices = Pakistan; Bolivia.

Interpretation:
The official task-adapted Prithvi decoder produces strong average native segmentation performance, but event-level deployment risk remains heterogeneous. Pakistan and Bolivia form the event-tail under the current raw event-level BWER audit. This is a meaningful paper-grade event fairness signal, unlike the earlier chip-level classification sanity run or the frozen non-TL threshold diagnostics.

Quality notes:
- Five chips have zero valid pixels and should remain documented as data QC edge cases.
- Fifty-three chips have zero predicted positive pixels, while fifty-two chips have zero ground-truth positive pixels; this does not indicate a global all-background failure.
- `event_id` is an operational disaster-event slice, not a causal country fairness attribute.
- `support_diagnostics.csv` was empty in this run; the primary evidence tables are `segmentation_metrics.csv`, `event_segmentation_metrics.csv`, `bwer_summary.csv`, `warnings.json`, report, and figures.

## BWER-Audit v2 Standardised Sen1Floods11 Analysis

Recorded: 2026-05-19.

The final fused result zip now includes a `bwer_v2/` post-hoc analysis folder
with real derived balance variables, not only raw event-level BWER. The key
tables are populated:
- `derived_balance_variables.csv`: 446 chip rows.
- `standardised_bwer.csv`: 12 rows.
- `reference_weight_sensitivity.csv`: 4 rows.
- `missing_policy_sensitivity.csv`: 6 rows.
- `bwer_v2_summary.csv`: 3 rows.

Raw result:
- Raw-BWER(event_id) = 0.1175.
- Mean risk = 0.2167.
- Tail risk = 0.3342.
- Worst slice = Pakistan.
- Best slice = Mekong.
- Tail slices = Pakistan; Bolivia.

Primary standardised result:
- Standardised-BWER(event_id | flood_extent_bin) = 0.1566.
- Mean risk = 0.4502.
- Tail risk = 0.6067.
- Worst slice = Bolivia.
- Tail slices = Bolivia; Pakistan.
- Missing cell count = 0.
- Valid balance bins = 3.

Interpretation:
After standardising over measured flood-extent composition, event-level tail
risk persists and even increases. Bolivia and Pakistan remain the tail slices,
although the worst event changes from Pakistan under raw BWER to Bolivia after
flood-extent standardisation. This suggests that the high-risk tail is not
fully explained by different flood-extent composition alone. The appropriate
paper wording is: the tail-risk signal "persists after standardising over
measured flood extent composition." Do not claim that it cannot be explained by
any confounder.

Secondary invalid/no-data composition result:
- Standardised-BWER(event_id | invalid_pixel_ratio_bin) = 0.1545 under
  renormalize/invalidate-style handling.
- Worst slice = Bolivia.
- Tail slices = Bolivia; Pakistan.
- Missing cell count = 1.

This supports a similar tail-risk interpretation, but it is secondary because
the result is more missing-policy-sensitive. Under overlap policy, the
invalid-ratio standardisation changes to BWER = 0.1082 with tail slices
Pakistan; USA. Use invalid-pixel-ratio standardisation as robustness evidence,
not the primary claim.

Sensitivity notes:
- `flood_extent_bin` is stable across reference weighting: uniform and empirical
  both give BWER = 0.1566 with tail slices Bolivia; Pakistan.
- `flood_extent_bin` is stable across overlap, renormalize, and invalidate
  missing policies because missing_cell_count = 0.
- `invalid_pixel_ratio_bin` is less stable because missing_cell_count = 1 and
  overlap changes the tail slices.

Reporting caution:
The standardised mean risk over flood extent is much higher than raw mean risk
(0.4502 vs 0.2167). This does not mean the model's aggregate IoU changed.
Standardised-BWER is a composition-standardised tail-risk diagnostic under a
reference composition, not a direct replacement for raw aggregate performance.

## U-Net Supervised Baseline and Prithvi Comparison

Recorded: 2026-05-20.

The first completed U-Net supervised baseline establishes a useful Protocol C
comparison point for Step 3. It uses `adaptation_protocol=supervised_baseline`
and `split_protocol=random_chip_split`, so it is a deployment-practice
baseline and not event-held-out generalization.

Comparison against the official Prithvi TL run:
- Prithvi TL aggregate IoU/Dice: 0.8052 / 0.8921.
- U-Net aggregate IoU/Dice: 0.7511 / 0.8579.
- Prithvi Raw-BWER(event_id): 0.1175.
- U-Net Raw-BWER(event_id): 0.3189.
- Prithvi Standardised-BWER(event_id | flood_extent_bin): 0.1566.
- U-Net Standardised-BWER(event_id | flood_extent_bin): 0.2295.
- Both models identify Pakistan as the worst event and Bolivia; Pakistan as
  tail events.

Interpretation:
The U-Net baseline is competitive enough in aggregate segmentation quality to
be a meaningful comparator, but its event-level Raw-BWER is substantially worse
than Prithvi TL. This strengthens the central average-vs-tail-risk story:
aggregate IoU/Dice and deployment-relevant event-tail risk can rank models
differently in practical importance. The shared Bolivia/Pakistan tail also
suggests these events are robust stress cases across both the official
task-adapted Prithvi decoder and a classical supervised U-Net baseline.

Protocol caution:
This comparison is protocol-aware, not architecture-only. Prithvi TL is an
official task-adapted checkpoint evaluated on the available full set, whereas
U-Net is trained as a supervised baseline under random chip split and evaluated
on its test split. Do not claim event-held-out generalization from the U-Net
result.

## Sen1Floods11 Closure Core Confirmed Facts

Recorded: 2026-05-20.

Sen1Floods11 Closure Core Package completed four native segmentation outputs:
- Prithvi TL.
- Vanilla U-Net.
- MNDWI spectral diagnostic baseline.
- S2 ResNet34-U-Net.

Aggregate IoU ranking:
- prithvi_tl > s2_resnet34_unet > vanilla_unet > spectral_mndwi.

Raw-BWER ranking:
- prithvi_tl < spectral_mndwi < s2_resnet34_unet < vanilla_unet.

Standardised-BWER ranking:
- spectral_mndwi < prithvi_tl < s2_resnet34_unet < vanilla_unet.

Average-vs-BWER ranking reversal was observed in the protocol-aware closure
comparison.

Persistent tail events across all four runs were not established.

Tail events by run:
- Prithvi TL: Bolivia; Pakistan.
- Vanilla U-Net: Bolivia; Pakistan.
- MNDWI spectral: India; Paraguay.
- S2 ResNet34-U-Net: Bolivia; Ghana.

S2 ResNet34-U-Net summary:
- aggregate_iou = 0.789684.
- aggregate_dice = 0.882484.
- raw_bwer_event_id = 0.177334.
- standardised_bwer_event_id_flood_extent_bin = 0.209538.
- worst_event = Ghana.
- best_event = Mekong.
- tail_events = Bolivia; Ghana.

Important caveat:
Prithvi TL and MNDWI use `standard_split/all`, while Vanilla U-Net and S2
ResNet34-U-Net use `random_chip_split/test`. Therefore the closure result is
protocol-aware and should not be described as pure architecture-only or
same-split comparison.

## Sen1Floods11 Closure Interpretation Guardrails

Recorded: 2026-05-21.

Protocol-matched check:
- The protocol-matched post-hoc check reached `exact_chip_level_match=True`
  with `matched_chip_count=89`.
- The observed average-vs-BWER ranking reversal persisted under exact
  evaluation-chip matching.
- Interpretation: evaluation-subset mismatch is not the sole explanation for
  the reversal. This does not rule out adaptation protocol, model capacity,
  thresholding, or training-protocol differences.

MNDWI diagnostic baseline:
- MNDWI has the lowest average segmentation performance in the closure set but
  the lowest Standardised-BWER.
- This should not be interpreted as MNDWI being the best segmentation model.
  It shows that BWER measures event-risk dispersion and tail concentration,
  not absolute segmentation quality.
- MNDWI should be reported as a low-capacity physics baseline and
  interpretability reference. Aggregate score and BWER must be reported
  together.

S2 ResNet34-U-Net / AlbuNet-style baseline:
- Relative to vanilla U-Net, S2 ResNet34-U-Net improves aggregate IoU and
  reduces Raw-BWER.
- Its event-level tail risk does not disappear.
- S2 ResNet34-U-Net reduces raw tail disparity relative to vanilla U-Net, but
  its Standardised-BWER remains higher than Prithvi TL and MNDWI.
- This helps address the concern that event-tail findings only occur for a weak
  vanilla U-Net baseline.

Tail-event identity:
- Persistent tail events across all four runs were not established.
- Some events recur across learned models, but no universal tail set appears
  across all protocols.
- Bolivia appears in Prithvi TL, vanilla U-Net, and S2 ResNet34-U-Net tails;
  Pakistan appears in Prithvi TL and vanilla U-Net tails; MNDWI has a different
  India/Paraguay spectral-rule failure profile.
- Interpretation: Sen1Floods11 contains both shared learned-model stress cases
  and model/protocol-specific event failure modes.

Selective Risk:
- Current outputs support only chip-level confidence-retention diagnostics when
  confidence summaries are available.
- These diagnostics are not pixel-level selective segmentation risk.
- Runs without confidence, logit, probability, or usable score fields should be
  marked selective-risk unavailable rather than assigned fabricated values.

## Vanilla U-Net Leave-One-Event-Out Result

Recorded: 2026-05-20.

Vanilla U-Net leave-one-event-out evaluation was completed for all 11
Sen1Floods11 held-out events:
- Bolivia.
- Ghana.
- India.
- Mekong.
- Nigeria.
- Pakistan.
- Paraguay.
- Somalia.
- Spain.
- Sri-Lanka.
- USA.

LOEO event-level IoU range:
- Best event: Mekong, micro IoU = 0.9336, risk = 0.0664.
- Worst event: Somalia, micro IoU = 0.6361, risk = 0.3639.

LOEO Raw-BWER(event_id):
- Raw-BWER = 0.1291.
- tail_slices = Somalia; Ghana.
- worst_slice = Somalia.
- best_slice = Mekong.

LOEO Standardised-BWER(event_id | flood_extent_bin):
- Standardised-BWER = 0.1531.
- tail_slices = Somalia; Pakistan.

Confirmed factual conclusion:
Event-level tail risk remains present under vanilla U-Net leave-one-event-out
evaluation, while the tail-event identities differ from the random-chip-split
vanilla U-Net result.

## fMoW-Sentinel Enriched Geography Preflight

Recorded: 2026-05-21.

The fMoW-Sentinel metadata enrichment and geography preflight foundation
completed for the location-level majority enriched metadata. SatMAE
fMoW-Sentinel train/val CSVs alone contain `category`, `location_id`,
`image_id`, and `timestamp`, but do not contain country, latitude, longitude,
region, continent, or UN-region fields.

Confirmed enrichment protocol:
- External fMoW metadata was parsed and joined back to the SatMAE metadata.
- Image-level join on `category + location_id + image_id` was incomplete.
- Location-level join on `category + location_id` reached 100%.
- `country` is resolved by majority mode per `category + location_id`.
- `latitude` and `longitude` are resolved by median polygon centroid per
  `category + location_id`.
- `continent`, `un_region`, and `region` are resolved from
  `country_region_map_full.csv`.

Confirmed metadata coverage:
- `country` missing ratio = 0.
- `latitude` and `longitude` are present.
- `continent`, `un_region`, and `region` missing ratio = 0.0186.
- `country`, `latitude_band`, `continent`, `un_region`, `region`, `category`,
  and `season` are usable geography/audit fields according to the generated
  support recommendations.

Known remaining unmapped / dirty geography codes:
- `ambiguous_country`: 14,304 rows.
- `ANT`: 293 rows.
- `CA-` / `KO-`: 256 rows.

Interpretation:
These remaining cases are treated as known dirty or legacy geography metadata
cases and are not silently mapped. The Step 2 fMoW-Sentinel geography audit
foundation is ready for future prediction-table generation and formal BWER
evaluation over supported geography slices. Geography metadata remains an audit
slicing/reporting field and is not a model input.

Final artifacts:
- final reproducible metadata directory:
  `cache/fmow_sentinel/metadata/final/`.
- output directory: `outputs/fmow_sentinel_preflight/enriched_geography_final_v1`.
- archive:
  `/content/drive/MyDrive/rsfm_fairness_audit/outputs/fmow_sentinel_preflight/enriched_geography_final_v1.zip`.
- temporary safety backup:
  `cache/fmow_sentinel/metadata/final_backup_20260520_233834/`; keep until
  the Step 3 fMoW-Sentinel prototype runs successfully from
  `cache/fmow_sentinel/metadata/final/`.

## fMoW-Sentinel Step 3 ResNet-50 Experiment Record

Recorded: 2026-05-22.

Experiment:
- Dataset: `fmow_sentinel_clean_subset_30k_location_disjoint_v2`.
- Task: 62-class scene classification.
- Input: Sentinel-2 13-band image-only.
- Model: ResNet-50, 13-band first convolution, trained from scratch.
- Adaptation protocol: `supervised_baseline`.
- Split protocol: `location_disjoint`.
- Train / validation rows: train = 21046, val = 8954.
- Location leakage check: `category + location_id` overlap between train and
  validation = 0.
- Geography metadata was used only for audit slicing/reporting, not as model
  input.

Aggregate validation metrics:
- accuracy = 0.2000.
- balanced_accuracy = 0.1841.
- macro_f1 = 0.1725.
- top5_accuracy = 0.4518.

Training behavior:
- Training loss decreased from approximately 3.8464 at epoch 1 to
  approximately 0.1872 by epoch 14.
- Validation accuracy plateaued around 0.18-0.20.
- Best observed validation accuracy in the shown log was approximately 0.2000.
- This run is recorded as the first full fMoW-Sentinel 30k
  location-disjoint ResNet-50 supervised baseline result.

Raw-BWER results:

| slice | BWER | tail_risk | mean_risk | best_slice | best_slice_risk | worst_slice | worst_slice_risk | n_slices_valid | support note |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| continent | 0.056653 | 0.871941 | 0.815287 | Asia | 0.747496 | Africa | 0.871941 | 5 |  |
| un_region | 0.056653 | 0.871941 | 0.815287 | Asia | 0.747496 | Africa | 0.871941 | 5 |  |
| region | 0.139434 | 0.948864 | 0.809429 | Central America | 0.628272 | Melanesia | 1.0 | 20 |  |
| latitude_band | 0.096197 | 0.923850 | 0.827653 | lat_10_20 | 0.746247 | lat_60_70 | 0.928058 | 13 |  |
| season | 0.038336 | 0.855556 | 0.817219 | Autumn | 0.785540 | Summer | 0.855556 | 4 |  |
| country | 0.173612 | 0.998011 | 0.824399 | ANT | 0.125 | UGA | 1.0 | 145 | min_samples_per_slice = 20 |

Class-standardised BWER results:

| slice | balance | BWER | tail_risk | mean_risk | best_slice | best_slice_risk | worst_slice | worst_slice_risk | n_slices_valid | warning |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| continent | class_label | 0.039066 | 0.883193 | 0.844126 | Asia | 0.821358 | Africa | 0.883193 | 5 | Some continent x class_label cells are missing, especially for Oceania. |
| un_region | class_label | 0.039066 | 0.883193 | 0.844126 | Asia | 0.821358 | Africa | 0.883193 | 5 | Some un_region x class_label cells are missing, especially for Oceania. |
| region | class_label | 0.102101 | 0.963204 | 0.861103 | Eastern Asia | 0.810753 | Melanesia | 1.0 | 20 | Many region x class_label cells are missing; use support diagnostics when interpreting. |
| latitude_band | class_label | 0.102722 | 0.951230 | 0.848508 | lat_-30_-20 | 0.794963 | lat_60_70 | 0.964959 | 13 | Some latitude_band x class_label cells are missing, especially in extreme latitude bands. |
| season | class_label | 0.029463 | 0.862754 | 0.833290 | Autumn | 0.809773 | Summer | 0.862754 | 4 | Winter missing class_label level interchange. |
| country | class_label | 0.142348 | 0.998814 | 0.856465 | KO- | 0.418182 | UGA | 1.0 | 145 | country x class_label has extensive missing cells; treat as diagnostic/support-filtered only. |

Formal-analysis notes:
- Primary formal geography slices for this run: `continent`, `un_region`,
  `region`, `latitude_band`, and `season`.
- Country-level results use support threshold `min_samples_per_slice = 20`.
- `country x class_label` is diagnostic only unless additional support
  filtering is applied.
- Missing class-label cells are expected in intersectional standardisation
  because the 30k subset has limited support for fine-grained slice x class
  combinations.
- Do not state causal fairness or country-level discrimination from these
  results.
- Do not claim these are full fMoW-Sentinel benchmark results; they are results
  on the support-aware 30k location-disjoint audit subset.
