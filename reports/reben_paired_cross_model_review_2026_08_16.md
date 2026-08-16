# reBEN paired S2→S1 cross-model scientific review

Date: 2026-08-16
Execution in this review: Drive artifact read and CPU-only interpretation; no retraining or new experiment.

## Result

The evidence supports the bounded claim that **the same paired shift has different failure geometry across models**. It does not support the stronger causal claim that one encoder “collapses” and the other does not.

| Three-seed mean | TerraMind | CROMA | Scientific contrast |
|---|---:|---:|---|
| Δ mean risk | 0.3088 | 0.2686 | Both degrade; CROMA is 0.0402 lower. |
| Δ tail risk | 0.3555 | 0.2727 | TerraMind allocates substantially more additional burden to the tail. |
| Δ GeoBWER | 0.0467 | 0.0041 | Strong TerraMind tail acceleration; CROMA is near zero on average. |
| Tail acceleration | 0.0467 | 0.0041 | Equal to ΔGeoBWER under the locked definition. |
| Mean Δ AUROC | -0.3861 | -0.3790 | Nearly identical rank/separation degradation. |
| Mean Δ AP | -0.4882 | -0.4440 | Both severe; CROMA retains more positive-ranking quality. |
| Mean Δ locked F1 | -0.5325 | -0.4004 | TerraMind operating-point failure is larger. |
| Modal label signatures | 11 collapse, 8 mixed | 0 collapse, 19 mixed | Different operational diagnostic geometry. |

All formal result audits pass. Seed directions are reproducible for mean/tail degradation. TerraMind has positive ΔGeoBWER in all three seeds; CROMA has two positive seeds and one levelling-down seed (−0.0048), so its excess tail acceleration is weaker and not directionally unanimous.

## Mechanistic interpretation

The common mechanism-level observation is a severe loss of label ranking under cross-sensor transfer: mean ΔAUROC differs by only about 0.007 between models. CROMA is therefore not sensor-invariant in this same-head test. The model-dependent part is how that loss is transported through probability space and allocated geographically.

TerraMind exhibits more probability-distribution displacement, more locked-threshold F1 loss, and a 0.0467 rise in GeoBWER. CROMA exhibits comparably severe AUROC loss but only a 0.0041 GeoBWER rise. Several labels reverse score-transport direction across models: urban fabric (CROMA ≈−0.047; TerraMind ≈+0.901), pastures (+0.328; −0.212), natural grass/sparse vegetation (+0.430; −0.012), and marine waters (+0.411; −0.099). Consequently, a macro performance drop alone cannot describe who bears the errors or whether the tail accelerates.

No label is threshold-shift-dominant in either model. The near-one OOD risks therefore cannot be rescued conceptually by saying only that the locked threshold moved out of calibration. CROMA's 19 mixed signatures mean rank loss, score movement, and threshold effects coexist without satisfying the stricter contraction-based collapse rule.

## Claim boundary

The valid unit of comparison is the deployed frozen-head pipeline under one task and one paired S2→S1 shift. Diagnostic labels are operational signatures derived from probability ranks, separation, contraction, and crossings. They are not causal localization to the encoder, preprocessor, or head. Label and country ordering remains descriptive with three seeds.

## Priority decision

1. **#8 adaptation: raise to the next mechanistic priority, but do not start yet.** Use a preregistered ladder: locked-threshold recalibration; S1 head refit with frozen encoder; representation adaptation only if the head refit is insufficient. The first step estimates calibration recoverability; the second separates head mismatch from deeper representation mismatch. The observed AUROC loss predicts that recalibration alone will be insufficient.
2. **#9 model×task 2×2: keep conditional and below #8.** The two-model reBEN result already establishes model-dependent geometry for this shift. A second task is needed only for a cross-task interaction/generalization claim or a reviewer request; otherwise its extra task and protocol interactions cost more than the mechanistic information it adds.

## Evidence status

- Cross-model sensitivity: `pass`.
- Same-shift different-failure-geometry claim: `supported_with_scope_boundary`.
- Items 1–7 finality: unchanged, `true`.
- #8/#9 execution: not started.
