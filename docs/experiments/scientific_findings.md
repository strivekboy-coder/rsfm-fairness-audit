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
