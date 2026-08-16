# Optimization items 1–7

This workflow implements only the approved optimization items 1–7. Items 8–17 are not launched.

## CPU post-processing

```bash
python scripts/analysis/run_optimization_1_7.py \
  --output-dir outputs/optimization_1_7_v1
```

The command freezes source hashes, compares GeoBWER with weighted SD and robust dispersion metrics, writes the complete slice table, renders ordered slice plots, builds the TerraMind cross-task table, and creates the available interaction atlas.

Low-support slices stay in `full_slice_distribution.csv` with `presentation_role=descriptive_low_support`; they are not silently deleted. Only rows satisfying the frozen support rule enter the primary metric comparator.

## reBEN label-budget runner

The cache root must contain `train/`, `val/`, and `test/`. Every split needs `embeddings.npy`, `labels.npy`, and `metadata.jsonl` from the frozen TerraMind S2 extraction. Metadata must include `sample_id`, `country`, `source_tile_id`, and `independent_unit_id`.

```bash
python scripts/run_reben_label_budget_colab.py \
  --cache-root /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_full_panel/terramind/s2/shared_embedding_cache \
  --output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_label_budget_v1 \
  --device cuda
```

Run the same command with `--preflight-only` first. It performs no training.

The runner freezes one independent-unit order per seed, uses nested 5/10/25/50/100% subsets, keeps validation/test unchanged, and locks per-label thresholds on S2 validation only. Existing embeddings are memory-mapped and selected by saved row indices; pixels are not decoded again and cumulative embedding copies are not written. Probe hyperparameters match the frozen 27-run panel (100 epochs, learning rate 0.01, weight decay 0.0001, batch size 512).

If training was started from an earlier checkout, pull the current code after it finishes and run only:

```bash
python scripts/run_reben_label_budget_colab.py \
  --output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_label_budget_v1 \
  --postprocess-only
```

This does not retrain. It checks the seed×budget grid, nested independent-unit selections, fixed test support, finite primary metrics, and endpoint contrasts, then writes PNG/PDF sensitivity curves and a result audit.

## reBEN paired S2-ID to S1-OOD runner

Both cache roots must use the same TerraMind checkpoint and preprocessing protocol. Their test caches must contain exactly the same sample IDs, targets, countries, source tiles, and independent units, and their embedding dimensions must match.

```bash
python scripts/run_reben_terramind_paired_shift_colab.py \
  --s2-cache-root /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_full_panel/terramind/s2/shared_embedding_cache \
  --s1-cache-root /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_full_panel/terramind/s1/shared_embedding_cache \
  --output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_terramind_s2_to_s1_paired_shift_v1 \
  --device cuda
```

Run the same command with `--preflight-only` first. Training must not start unless it reports `status=formal_ready`.

The formal preflight requires all six split-level embedding-cache lineage manifests. It checks the same TerraMind checkpoint/encoder protocol, the expected S2 versus S1 profiles, identical paired test sample IDs/targets/country/source-tile/independent-unit metadata, binary labels, and matching embedding dimensions. Missing lineage is blocking.

The runner uses seeds 42/73/101 by default. Within each seed, the head is trained only on S2 train, thresholds are selected only on S2 validation, and that unchanged head is evaluated on paired S2 and S1 test embeddings. Probe hyperparameters match the frozen 27-run panel (100 epochs, learning rate 0.01, weight decay 0.0001, batch size 512). The output deliberately uses “OOD degradation”; it does not claim EarthShift effective robustness.

The completed runner automatically writes per-label and per-country paired risk changes, seed aggregates, levelling-down and tail-acceleration diagnostics, PNG/PDF figures, and `paired_shift_result_audit.json`. To audit a completed directory without training:

```bash
python scripts/run_reben_terramind_paired_shift_colab.py \
  --output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_terramind_s2_to_s1_paired_shift_v1 \
  --postprocess-only
```

## Final evidence freeze for items 1–7

After both item 6 and item 7 audits pass:

```bash
python scripts/analysis/finalize_optimization_1_7.py \
  --base-result-dir outputs/optimization_1_7_v1 \
  --label-budget-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_label_budget_v1 \
  --paired-shift-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/reben_terramind_s2_to_s1_paired_shift_v1 \
  --output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/optimization_1_7_final_evidence_v1
```

Without both passing result audits, the finalizer fails rather than labelling incomplete evidence as final. `--allow-pending` may be used only to write a readiness manifest with `finality=false`.

## Scope boundary

Items 8–17 remain explicitly not started. This workflow does not launch their formal experiments, training, data engineering, preflight, result audits, or figures. Existing protocol notes or lightweight runner skeletons are left untouched and are not promoted to empirical evidence.
