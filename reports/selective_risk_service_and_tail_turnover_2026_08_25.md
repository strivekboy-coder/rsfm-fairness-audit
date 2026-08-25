# Selective risk–service and shift–adaptation burden analysis

Date: 2026-08-25
Status: paper-facing descriptive analysis of frozen canonical outputs; no model rerun and no frozen-output mutation.

## Scientific decision

Both proposed directions are supported and worth retaining. The first is a strong, compact extension of the selective-prediction result. The second provides a useful dynamic synthesis of paired sensor shift and Experiment 8, provided that label-level `1 − F1` and prevalence-weighted slice-risk contributions remain explicitly distinguished.

## 1. Selective prediction: empirical geographic risk–service frontier

The analysis uses the complete registered panel: 2 models × 3 target coverages × 3 slice axes (country, region and class) = 18 cells. It does not select a favorable coverage after inspecting results.

- 17/18 cells have negative tail-minus-non-tail retained coverage.
- The only exception is DOFAv2 at 90% target coverage on the class axis: +0.447 percentage points.
- All 18 cells have higher estimated risk among rejected than retained examples.
- All 18 cells reduce remaining risk relative to the unselective baseline.

Country-axis tail service gaps are:

| Model | 70% coverage | 80% coverage | 90% coverage |
|---|---:|---:|---:|
| DOFAv2 | −8.47 pp | −7.72 pp | −4.82 pp |
| ResNet50 | −10.54 pp | −7.98 pp | −6.18 pp |

The retained-risk reduction is larger under stronger abstention. At 70% target coverage it is 0.0261 for DOFAv2 and 0.0396 for ResNet50 on the country axis; the rejected-minus-retained risk gaps are 0.1038 and 0.1520, respectively. Thus selective prediction removes genuinely harder examples, but the resulting service reduction falls disproportionately on already high-risk countries.

Recommended paper wording:

> Across the complete 18-cell selective panel, abstention reduced remaining prediction risk, but 17/18 tail-versus-non-tail retention contrasts were negative. On the country axis, high-risk countries retained 4.8–8.5 pp fewer DOFAv2 predictions and 6.2–10.5 pp fewer ResNet50 predictions across 70–90% target coverage. Selective reliability gains therefore coincided with a geographically unequal service burden.

This is an empirical risk–service frontier, not a causal or formally optimized Pareto frontier.

## 2. Paired shift → adaptation: tail turnover and burden migration

The paired-shift and Experiment 8 artifacts share the frozen TerraMind lineage, test support, seeds and unchanged-head Stage A. Their label-level `1 − F1` values agree and can therefore be joined to Stage C without test tuning. The code nevertheless keeps two estimands separate:

- label trajectory: `1 − F1`, useful for an interpretable label-specific example;
- tail turnover: the frozen prevalence-weighted Hamming slice-risk contribution used by the Experiment 8 audit.

The clearest label trajectory is `Marine waters`:

| Stage | Mean `1 − F1` |
|---|---:|
| S2 ID | 0.0137 |
| unchanged-head S1 shifted | 1.0000 |
| frozen-encoder S1-head Stage C | 0.0481 |

`Beaches, dunes, sands` and `Coastal wetlands` were already difficult at ID (0.9048 and 0.7278), reached 1.0 after shift, and remained difficult after Stage C (0.9690 and 0.8561). Stage C therefore repairs the shift-created Marine failure without uniformly resolving pre-existing ceiling labels.

Tie-aware fixed-universe turnover gives the following supported-slice counts:

| Axis | Supported units | Shift-created tail | Shift-tail exits after C | New C-tail | Persistent shifted tail after C |
|---|---:|---:|---:|---:|---:|
| country | 10 | 1 | 1 | 1 | 0 |
| class label | 19 | 2 | 2 | 2 | 0 |
| country × label | 125 | 20 | 20 | 18 | 0 |

The new residual burden is localized rather than random. For example, `Switzerland × Complex cultivation patterns` changes from 0.2006 under the shifted Stage A head to 0.6217 after Stage C, while `Switzerland × Broad-leaved forest` changes from 0.1609 to 0.5011. These are three-seed, support-qualified descriptive patterns.

Recommended paper wording:

> Adaptation did not simply scale down the shifted tail. In the fixed, tie-aware slice universe, all recurrent shifted-tail units exited the Stage C tail, while new residual-tail units appeared on each axis, including 18 supported country×label cells. The identities of the deployment groups bearing risk therefore changed during recovery, even when aggregate performance substantially recovered.

This supports burden localization and redistribution. It does not identify a causal encoder mechanism, and CROMA has paired-shift evidence but no matching Stage C adaptation ladder.

## Generated artifacts

- `25_selective_risk_service_frontier.csv`: complete 18-cell frontier and verification fields.
- `26_shift_adaptation_tail_turnover_seed_level.csv`: full per-seed fixed-universe lineage.
- `27_shift_adaptation_tail_turnover.csv`: support-qualified three-seed turnover summary.
- `F6_selective_risk_service_frontier.{png,pdf}`: complete selective result.
- `F7_shift_adaptation_tail_turnover.{png,pdf}`: label recovery plus country×label burden migration.

All generated outputs remain under the ignored analysis-output directory; this report, the reproducible builder and regression tests are the committed paper-facing provenance.
