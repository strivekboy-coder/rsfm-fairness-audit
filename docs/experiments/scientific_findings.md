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
