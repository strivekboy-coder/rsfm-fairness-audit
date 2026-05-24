# fMoW-Sentinel Step 3 Scientific Findings

This document records confirmed fMoW-Sentinel Step 3 experiment facts,
protocol notes, and debugging outcomes. It should not contain causal fairness
claims or full-benchmark claims unless separately justified.

## Protocol Note: DOFA Input Scaling

Recorded: 2026-05-22. Updated: 2026-05-23.

An initial DOFA frozen-probe run used identity normalization on raw
fMoW-Sentinel reflectance-like TIFF values. This run is treated as
invalid/diagnostic because the frozen DOFA encoder received inputs outside the
expected reflectance scale.

Author-confirmed protocol guidance: DOFA inputs should be normalized to
`[0, 1]` or `[-1, 1]`. Scaling raw fMoW-Sentinel reflectance-like values by
`1 / 10000` is consistent with that requirement. The unscaled DOFA run is
therefore an invalid/inappropriate protocol for formal comparison.

Band order is not intrinsically important for DOFA as long as the
wavelength-band correspondence is correct. The critical requirement is that
each input channel is paired with the correct wavelength value.

After applying `input_scale = 10000`, DOFA embedding separability recovered:
validation accuracy improved from 10.9% to 17.8%, and macro-F1 improved from
6.3% to 16.9%. Embedding diagnostics showed no NaN/Inf values, non-collapsed
feature variance, and reasonable embedding norms.

The scaled DOFA run is the formal interpretable DOFA frozen linear-probe
result. The unscaled run should not be used for scientific model comparison or
BWER conclusions. Pooling and fine-tuning recipes remain open until the DOFA
authors release task examples or clearer protocol guidance.

## Step 3 ResNet-50 and DOFA Result Record

Recorded: 2026-05-22.

### Final Archive

The final Step 3 bundle is archived at:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/final_step3/fmow_step3_final_bundle_30k_location_disjoint.zip
```

Archive size: 385.37 MB.

Bundle contents:

- `final_step3/resnet50_30k_location_disjoint_patched_metadata.zip`:
  ResNet-50 13-band supervised baseline outputs. This archive includes patched
  `run_metadata.json` with reconstructed `class_mapping`.
- `final_step3/dofa_scaled10000_30k_location_disjoint.zip`: DOFA ViT-B frozen
  encoder plus linear-probe outputs using `input_scale = 10000`. This is the
  valid DOFA run.
- `final_step3/comparison_resnet50_vs_dofa_scaled10000.zip`: ResNet-50 vs
  scaled-DOFA comparison outputs, including `comparison_summary.csv`,
  `average_vs_bwer.csv`, `geography_slice_comparison.csv`, and
  `comparison_report.md`.

### Experiment

- Dataset: fMoW-Sentinel 30k support-aware clean subset.
- Split: location-disjoint.
- Group definition: `category + location_id`.
- Train rows: 21046.
- Val rows: 8954.
- Task: 62-class scene classification.
- Input: Sentinel-2 13-band image-only.
- Geography metadata was used only for audit slicing and reporting, not model
  input.

### Models

ResNet-50 13-band supervised baseline:

- `model_variant`: `resnet50_13band_from_scratch`.
- `adaptation_protocol`: `supervised_baseline`.
- `image_size`: 96.
- Training: AdamW cross-entropy, 20 epochs.
- Accuracy: 0.200022.
- Balanced accuracy: 0.184051.
- Macro-F1: 0.172450.
- Top-5 accuracy: 0.451753.

DOFA ViT-B frozen encoder with linear probe:

- `model_variant`: `dofa_vit_base`.
- `adaptation_protocol`: `frozen_encoder_linear_probe`.
- `image_size`: 224.
- `input_scale`: 10000.
- Accuracy: 0.177686.
- Balanced accuracy: 0.179074.
- Macro-F1: 0.168659.
- Top-5 accuracy: not available (`NaN`).

### DOFA Preprocessing Finding

- The unscaled DOFA diagnostic run used raw fMoW-Sentinel reflectance-like TIFF
  values directly.
- The unscaled DOFA diagnostic result had accuracy about 0.1094 and macro-F1
  about 0.0631.
- After applying `input_scale = 10000`, DOFA improved to accuracy 0.177686 and
  macro-F1 0.168659.
- This indicates that frozen RSFM evaluation is highly sensitive to input
  scaling and preprocessing protocol.
- The unscaled DOFA run is an invalid/debug artifact, not a main result.

### Model Comparison Facts

- ResNet-50 aggregate accuracy is higher than scaled DOFA:
  0.200022 vs 0.177686.
- ResNet-50 macro-F1 is slightly higher than scaled DOFA:
  0.172450 vs 0.168659.
- Scaled DOFA has lower country-level Raw-BWER:
  ResNet-50 0.173612 vs DOFA 0.161415.
- Scaled DOFA has lower country x class standardised BWER:
  ResNet-50 0.142348 vs DOFA 0.126978.
- Scaled DOFA has lower region Raw-BWER:
  ResNet-50 0.139434 vs DOFA 0.117179.
- Scaled DOFA has lower region x class standardised BWER:
  ResNet-50 0.102101 vs DOFA 0.086553.
- Scaled DOFA has lower latitude_band x class standardised BWER:
  ResNet-50 0.102722 vs DOFA 0.086541.
- ResNet-50 has lower continent/un_region Raw-BWER and class-standardised BWER:
  continent Raw-BWER is 0.056653 for ResNet-50 vs 0.068857 for DOFA;
  continent x class BWER is 0.039066 for ResNet-50 vs 0.053048 for DOFA.
- Season BWER is lower for DOFA:
  season Raw-BWER is 0.020915 and season x class BWER is 0.024367. Compare
  against ResNet-50 season rows from the saved BWER table where needed.

### Careful Interpretation

- Do not claim DOFA is universally fairer.
- Do not claim high aggregate accuracy hides risk based only on ResNet-50,
  because ResNet-50 accuracy is modest on this subset.
- Supported observation: model ranking by aggregate accuracy differs from model
  ranking by some geography-tail-risk metrics.
- Supported observation: class-standardised geography BWER persists, especially
  at region and latitude-band levels.
- Country x class remains diagnostic due to sparse support; country-level formal
  analysis should use support thresholds such as val >= 20 or val >= 30.
- Stronger formal slices: continent, un_region, region, latitude_band, season,
  and category.

### Local Data and Model Handling Notes

- The initial 30k v2 manifest/directory was incomplete because old 10k rows
  still pointed to v1 image paths. The formal dataset was fixed by merging v1
  images into the v2 tree and creating `fixed_manifest_30k_merged.csv`.
- ResNet-50 `run_metadata.json` initially missed `class_mapping` even though
  `audit_table.csv` contained all 62 classes. The archived ResNet-50 output uses
  patched metadata with `class_mapping` reconstructed from `audit_table.csv`.
- Country x class and country x category analyses remain diagnostic or
  support-threshold filtered because country-class support is sparse.

## Baseline Closure Sanity Results

Recorded: 2026-05-24.

The baseline closure sanity Colab runs were completed and archived under:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/baseline_closure_sanity/
```

Archived files:

- `fmow_random_split_resnet50_sanity.zip`.
- `fmow_tiny_overfit_resnet50_sanity.zip`.
- `fmow_dofa_pooling_ablation_sanity.zip`.
- `fmow_dofa_random_split_sanity.zip`.
- `fmow_baseline_closure_sanity_bundle.zip`.

Only the 16-epoch random-split result is the final saved random-split sanity
record. The earlier 8-epoch random-split run is not the final sanity output and
should not be documented as such.

### Random Split ResNet-50 Sanity

Path:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/baseline_closure_sanity/fmow_random_split_resnet50_sanity.zip
```

Source Colab output directory:

```text
/content/outputs/baseline_closure_sanity/random_split_resnet50_16epoch
```

Protocol:

- Dataset: `fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged.zip`.
- Split protocol: `random_split_sanity`.
- Model: ResNet-50 13-band from scratch.
- Train/eval rows: 21000 / 9000.
- Epochs: 16.
- Checkpoint selection: best validation accuracy.
- Best epoch: 14 / 16.
- This is a sanity contrast only, not the formal deployment protocol.

Metrics:

- Accuracy: 0.7118888888888889.
- Balanced accuracy: 0.6672910350320261.
- Macro-F1: 0.678352105923704.
- Top-5 accuracy: 0.8354444444444444.

Random split accuracy is much higher than the formal location-disjoint ResNet
result, whose accuracy is approximately 0.20. This supports the protocol record
that the location-disjoint evaluation is substantially harder and is necessary
for cross-location generalization evaluation.

Even after random split accuracy rises strongly, geography tail risk does not
disappear. From the BWER summary:

- Country Raw-BWER: approximately 0.2517475898.
- Country | class standardised BWER: approximately 0.2577851581.

This is diagnostic evidence that average performance improvement does not
automatically eliminate deployment slice risk. It must not be treated as the
formal deployment protocol.

### Tiny Overfit ResNet-50 Sanity

Path:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/baseline_closure_sanity/fmow_tiny_overfit_resnet50_sanity.zip
```

Source Colab output directory:

```text
/content/outputs/baseline_closure_sanity/tiny_overfit_resnet50
```

Protocol:

- Tiny repeated train/eval subset.
- Classes: 4.
- Samples per class: 8.
- Epochs: 40.
- Diagnostic only, not a scientific result.

Metrics:

- Final accuracy: 0.96875.
- Final macro-F1: 0.9686274509803923.

This sanity check records that the ResNet training loop, label mapping, and loss
plumbing can overfit a tiny subset. This reduces concern that low
location-disjoint performance is caused by broken labels or broken training
logic.

### DOFA Random Split Sanity

Path:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/baseline_closure_sanity/fmow_dofa_random_split_sanity.zip
```

Protocol:

- Dataset: same `v3_merged` clean 30k subset.
- Split: same `random_split_manifest.csv` as the final ResNet-50 16-epoch
  random split sanity run.
- Model: DOFA ViT-base frozen encoder plus linear probe.
- `input_scale`: 10000.
- Probe epochs: 200.
- Diagnostic only, not the formal deployment protocol.

Metrics:

- Accuracy: 0.38433333333333336.
- Balanced accuracy: 0.3845599647290616.
- Macro-F1: 0.3798401872360772.

BWER:

- Country Raw-BWER: approximately 0.2051891816.
- Country | class standardised BWER: approximately 0.1769901752.

The DOFA random split result records that scaled DOFA also improves
substantially under random sample-level splitting compared with the formal
location-disjoint result. It remains a diagnostic sanity contrast and should
not be used as a formal deployment benchmark.

### DOFA Pooling Ablation Sanity

Path:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/baseline_closure_sanity/fmow_dofa_pooling_ablation_sanity.zip
```

Source Colab output directory:

```text
/content/outputs/baseline_closure_sanity/dofa_pooling_ablation
```

Protocol:

- Model: DOFA frozen encoder plus linear probe.
- `input_scale`: 10000.
- Split protocol: `location_disjoint`.
- Compared pooling modes: `flatten` and `mean_tokens`.
- CLS is unavailable in the current adapter/output contract; no CLS result is
  fabricated.

Comparison result:

- `flatten` and `mean_tokens` produced identical results.
- Accuracy: 0.17768595041322313.
- Raw-BWER(country): 0.16141538857738702.
- Standardised-BWER(country | class): 0.1269780950367737.

Inspection result:

- Cache keys differ: true.
- Embeddings are identical across train/eval: true.
- Diagnosis: the pooling parameter reached run metadata and cache keys, but
  cached embeddings are already 2D and identical. The current DOFA
  adapter/output path exposes a pooled 768-dimensional representation rather
  than token/spatial features, so there is no token dimension for `mean_tokens`
  to change.

This is a protocol sanity finding, not a model-performance finding. Under the
current DOFA adapter, pooling choice does not affect the formal result because
the adapter exposes an already-pooled representation. CLS is unavailable and
should not be claimed.

### Baseline Closure Status

The main baseline closure sanity checks are now completed:

- Artifact sanity verification passed.
- 30k random split sanity completed.
- DOFA random split sanity completed.
- Tiny overfit sanity completed.
- DOFA pooling inspection completed.

Do not document the 8-epoch random split result as final. Do not create
additional `v1`/`v2`/`v3` names for these sanity outputs. Do not treat random
split as the formal deployment protocol. Do not treat DOFA pooling ablation as a
performance improvement or failure. Do not rerun the formal location-disjoint
ResNet/DOFA main experiments for this record.
