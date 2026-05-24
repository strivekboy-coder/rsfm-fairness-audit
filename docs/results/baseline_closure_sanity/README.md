# Baseline Closure Sanity Outputs

This directory holds lightweight readiness records for the fMoW/Sen1Floods11
baseline closure package. These files are not scientific results by themselves.

Current local files:

- `baseline_sanity_report.md`
- `baseline_sanity_checks.json`

The following checks are implemented but remain `awaiting_colab_run` until real
Colab outputs are generated and copied back into the evidence package:

- random split sanity:
  `scripts/run_fmow_random_split_sanity_colab.py`
- tiny overfit sanity:
  `scripts/run_fmow_tiny_overfit_sanity_colab.py`
- DOFA pooling ablation:
  `scripts/run_fmow_dofa_pooling_ablation_colab.py`

The artifact checker should be rerun in Colab with explicit final paths so the
report records the actual files it read. The final prepared dataset zip may
contain any `final_clean_subset_manifest*.csv`; when several are present, the
checker prioritizes `final_clean_subset_manifest_30k_location_disjoint_v3_merged.csv`.

Suggested Colab run order:

1. Rerun `scripts/run_baseline_closure_sanity.py` with explicit final zips.
2. Run `scripts/run_fmow_random_split_sanity_colab.py`.
3. Run `scripts/run_fmow_tiny_overfit_sanity_colab.py`.
4. Run `scripts/run_fmow_dofa_pooling_ablation_colab.py`.

Each runner writes its own report and metadata JSON under the selected
`/content/outputs/baseline_closure_sanity/...` directory. Copy those outputs
back into the evidence package only after the Colab run completes.
