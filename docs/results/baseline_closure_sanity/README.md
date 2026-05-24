# Baseline Closure Sanity Outputs

This directory holds lightweight readiness records for the fMoW/Sen1Floods11
baseline closure package. These files are not scientific results by themselves.

Current local files:

- `baseline_sanity_report.md`
- `baseline_sanity_checks.json`

The baseline closure sanity Colab outputs have been completed and archived in
Google Drive:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/baseline_closure_sanity/
```

Archived files:

- `fmow_random_split_resnet50_sanity.zip`
- `fmow_tiny_overfit_resnet50_sanity.zip`
- `fmow_dofa_pooling_ablation_sanity.zip`
- `fmow_baseline_closure_sanity_bundle.zip`

Only the 16-epoch random-split result should be treated as the final saved
random-split sanity record. Do not keep or document the earlier 8-epoch run as
the final random-split sanity output.

Completed diagnostic checks:

- random split sanity:
  `fmow_random_split_resnet50_sanity.zip`
- optional DOFA random split sanity runner:
  `scripts/run_fmow_dofa_random_split_sanity_colab.py`
- tiny overfit sanity:
  `fmow_tiny_overfit_resnet50_sanity.zip`
- DOFA pooling ablation and cache inspection:
  `fmow_dofa_pooling_ablation_sanity.zip`

The artifact checker should be rerun in Colab with explicit final paths so the
report records the actual files it read. The final prepared dataset zip may
contain any `final_clean_subset_manifest*.csv`; when several are present, the
checker prioritizes `final_clean_subset_manifest_30k_location_disjoint_v3_merged.csv`.

Suggested Colab run order:

1. Rerun `scripts/run_baseline_closure_sanity.py` with explicit final zips.
2. Run `scripts/run_fmow_random_split_sanity_colab.py`.
3. Run `scripts/run_fmow_tiny_overfit_sanity_colab.py`.
4. Run `scripts/run_fmow_dofa_pooling_ablation_colab.py`.
5. If flatten and mean-token metrics are identical, run
   `scripts/inspect_fmow_dofa_pooling_ablation.py` to compare cached embedding
   shapes, hashes, and max absolute differences without rerunning DOFA.
6. Optional: run `scripts/run_fmow_dofa_random_split_sanity_colab.py` to reuse
   the final ResNet 16-epoch `random_split_manifest.csv` for scaled DOFA.

These checks are diagnostic closure evidence only. The random-split result is
not the formal deployment protocol, and the DOFA pooling ablation is not a
performance finding.
