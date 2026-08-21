# Experiments 8 and 9: Colab runbook

These commands assume the repository is `/content/rsfm-fairness-audit`, Drive is
mounted, the pinned model assets have already been prepared, and the real
datasets have been extracted to `/content`.  They never write into the frozen
canonical result directories.

## Experiment 8: reBEN S2 to S1 adaptation ladder

Copy the already-computed TerraMind embeddings to the local runtime.  This is
I/O staging, not a baseline rerun.

```bash
cd /content/rsfm-fairness-audit
mkdir -p /content/exp8_cache/s2 /content/exp8_cache/s1
cp -a /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_full_panel/terramind/s2/shared_embedding_cache/. /content/exp8_cache/s2/
cp -a /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_full_panel/terramind/s1/shared_embedding_cache/. /content/exp8_cache/s1/
```

Run cache/baseline preflight, then B and C. Stage A is read only; D is not run.

```bash
python scripts/colab/run_experiment8_reben_adaptation_colab.py \
  --s2-cache-root /content/exp8_cache/s2 \
  --s1-cache-root /content/exp8_cache/s1 \
  --frozen-baseline-root /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_terramind_paired_shift_v1 \
  --output-dir /content/experiment8_reben_adaptation_v1 \
  --seeds 42,73,101 \
  --device cuda \
  --preflight-only

python scripts/colab/run_experiment8_reben_adaptation_colab.py \
  --s2-cache-root /content/exp8_cache/s2 \
  --s1-cache-root /content/exp8_cache/s1 \
  --frozen-baseline-root /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_terramind_paired_shift_v1 \
  --output-dir /content/experiment8_reben_adaptation_v1 \
  --seeds 42,73,101 \
  --epochs 100 \
  --batch-size 512 \
  --device cuda

mkdir -p /content/drive/MyDrive/rsfm_fairness_audit/outputs/experiment8_reben_adaptation_v1
cp -a /content/experiment8_reben_adaptation_v1/. /content/drive/MyDrive/rsfm_fairness_audit/outputs/experiment8_reben_adaptation_v1/
```

Do not run a LoRA job unless `stage_d_gate.json` says
`D_eligible_for_consideration`. Even then, inspect the per-seed validation-only
criteria before defining a D protocol.

## Experiment 9a: missing DOFAv2 x reBEN cell

This uses the official raw-band LMDB and the native DOFA 9-band order. The
shared embeddings are extracted once and reused by all three probe seeds.

```bash
cd /content/rsfm-fairness-audit
python scripts/colab/run_experiment9_model_task_generalization_colab.py dofa-reben \
  --lmdb-root /content/data/reben/BigEarthNetEncoded.lmdb \
  --metadata-parquet /content/data/reben/metadata.parquet \
  --model-config configs/models/dofav2_fmow_sentinel.yaml \
  --dofa-repo /content/rsfm_model_repos/dofa \
  --checkpoint /content/rsfm_model_assets/dofav2_vit_base_e150.pth \
  --output-dir /content/experiment9_dofav2_reben_v1 \
  --persistent-output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/experiment9_dofav2_reben_v1 \
  --seeds 42,73,101 \
  --device cuda
```

If the real LMDB path has not yet been verified in the current runtime, first
append `--diagnostic-max-samples 32`, use a separate diagnostic output path,
and inspect `diagnostic_manifest.json`. Never reuse that diagnostic directory
for the formal run.

## Experiment 9b: missing TerraMind x fMoW cell

Extract the existing 13-band prepared archive before running. TerraMind selects
the 12 S2L2A bands by removing B10; DOFA's frozen existing cell remains 9-band.
This makes the matrix a model-pipeline comparison, not a backbone-only causal
ablation.

```bash
cd /content/rsfm-fairness-audit
python scripts/colab/run_experiment9_model_task_generalization_colab.py terramind-fmow \
  --metadata-csv /content/fmow_formal_split_v1/fmow_formal_manifest_train_calibration_test.csv \
  --data-root /content/data/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged \
  --checkpoint /content/rsfm_model_assets/TerraMind_v1_base.pt \
  --output-dir /content/experiment9_terramind_fmow_v1 \
  --persistent-output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/experiment9_terramind_fmow_v1 \
  --seeds 42,73,101 \
  --device cuda
```

For a one-batch real-data smoke, add `--diagnostic-max-per-split 32` and use a
separate diagnostic output directory. The diagnostic run is not evidence.

## Experiment 9c: CPU analysis after both missing cells finish

```bash
cd /content/rsfm-fairness-audit
python scripts/colab/run_experiment9_model_task_generalization_colab.py analyze \
  --dofa-fmow-root /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/fmow_dofav2_geo_clean_v1 \
  --terramind-fmow-root /content/drive/MyDrive/rsfm_fairness_audit/outputs/experiment9_terramind_fmow_v1 \
  --dofa-reben-root /content/drive/MyDrive/rsfm_fairness_audit/outputs/experiment9_dofav2_reben_v1 \
  --terramind-reben-root /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_full_panel/terramind/s2 \
  --output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/experiment9_model_task_analysis_v1
```

The analyzer fails closed unless the two models within each task share test
sample IDs, seeds, and RiskSpec signature. It reports task-wise ranks,
standardised within-task contrasts, M/T/D consistency, and a descriptive
model-by-task interaction. It does not average raw GeoBWER across tasks.

## Required completion checks

Experiment 8:

- `experiment8_manifest.json` says the frozen A root was not modified;
- `adaptation_stage_metrics.csv` contains A/B/C for three seeds and both
  validation-decision and test roles;
- `adaptation_recovery.csv` contains AUROC/AP/F1/M/T/D recovery;
- `adaptation_slice_patterns.csv` contains label, country and country x label;
- `adaptation_no_harm_slices.csv` contains harms relative to shifted A;
- `adaptation_mean_tail_consistency.csv` flags ordinary recovery without tail
  recovery and levelling-down;
- `stage_d_gate.json` is validation-only and explicitly says whether D is
  eligible.

Experiment 9:

- each missing cell has three seed outputs, formal audit tables and GeoBWER
  summaries;
- embedding cache manifests record checkpoint, preprocessing, band profile,
  image size, metadata hash and row hash;
- no row was silently dropped because of an unreadable chip;
- `experiment9_manifest.json` passes same-support, same-seed and same-RiskSpec
  checks within both tasks;
- the interaction table is marked descriptive and no output averages raw
  cross-task GeoBWER.
