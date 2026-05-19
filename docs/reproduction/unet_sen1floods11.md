# U-Net Sen1Floods11 Supervised Baseline

This baseline is Protocol C: a fully supervised classical U-Net for native
Sen1Floods11 flood segmentation. It is not a foundation model. It is intended
as a deployment-practice baseline for average-vs-BWER comparisons and should
not be compared to Prithvi TL without noting adaptation-protocol differences.

The U-Net path uses the same prepared 512 Sen1Floods11 chips used by the
Prithvi TL run when available:

```text
/content/drive/MyDrive/rsfm_fairness_audit/prepared_zips/sen1floods11_prithvi_tl_official_full_512.zip
```

The prepared data are read-only input. The script extracts them to `/content`
for training speed and does not rerun data preparation or modify the prepared
zip.

## Protocol

- `model_family = unet`
- `model_variant = unet_sen1floods11_s2_512`
- `adaptation_protocol = supervised_baseline`
- default `split_protocol = random_chip_split`
- optional `split_protocol = event_held_out`
- `resolution = 512`
- `input_mode = S2`
- label mapping: `0=background;1=water_flood;-1=ignore`
- loss: masked BCE-with-logits plus soft Dice loss
- primary segmentation metrics: event-level micro IoU/Dice/F1, precision,
  recall, TP/FP/FN/TN, valid-pixel support, and positive-pixel support

Random chip split is useful as the first reproducible supervised baseline, but
it is not event-held-out generalization. Use `event_held_out` before making
generalization claims about unseen disaster events.

## CLI Smoke Run

Use this for a quick local or Colab check after extracting prepared data:

```bash
python -m rsfm_fairness_audit.cli run-unet-sen1floods11 \
  --data-root /content/data/sen1floods11_tl_official_full_512 \
  --output-dir /content/outputs/unet_sen1floods11_smoke64 \
  --max-samples 64 \
  --epochs 1 \
  --batch-size 2 \
  --learning-rate 1e-3 \
  --split-protocol random_chip_split \
  --eval-split test \
  --run-bwer-v2
```

## Full Baseline

The recommended final supervised baseline uses a simple but stronger training
budget than the first 8-epoch smoke run: up to 50 epochs, best-checkpoint
selection by validation IoU, ReduceLROnPlateau scheduling, and early stopping
with patience 10. This gives the U-Net a fair chance while staying reproducible
on Colab L4/A100.

```bash
python -m rsfm_fairness_audit.cli run-unet-sen1floods11 \
  --data-root /content/data/sen1floods11_tl_official_full_512 \
  --output-dir /content/outputs/unet_sen1floods11_full_512 \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --early-stopping-patience 10 \
  --split-protocol random_chip_split \
  --eval-split test \
  --run-bwer-v2
```

For an event-held-out diagnostic:

```bash
python -m rsfm_fairness_audit.cli run-unet-sen1floods11 \
  --data-root /content/data/sen1floods11_tl_official_full_512 \
  --output-dir /content/outputs/unet_sen1floods11_event_held_out_512 \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --early-stopping-patience 10 \
  --split-protocol event_held_out \
  --held-out-event Pakistan \
  --held-out-event Bolivia \
  --eval-split test \
  --run-bwer-v2
```

## Post-hoc BWER v2

If the first run did not pass `--run-bwer-v2`, run:

```bash
python -m rsfm_fairness_audit.cli run-bwer-v2 \
  --input-dir /content/outputs/unet_sen1floods11_full_512 \
  --output-dir /content/outputs/unet_sen1floods11_full_512/bwer_v2
```

BWER-Audit v2 derives `flood_extent_bin` from chip-level positive-pixel support
and computes Raw-BWER, Standardised-BWER, stabilised BWER, alpha/support
sensitivity, missing-policy sensitivity, reference-weight sensitivity,
leave-one-slice-out diagnostics, and post-hoc bootstrap CI.

## Colab One-Script Workflow

Use this focused Colab cell to clone/update the repo, install project
dependencies, and run the helper:

```python
from pathlib import Path
import os, shutil, subprocess
from google.colab import drive

REPO_URL = "https://github.com/strivekboy-coder/rsfm-fairness-audit.git"
PROJECT_DIR = Path("/content/rsfm-fairness-audit")

drive.mount("/content/drive")
if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)
subprocess.run(["git", "clone", REPO_URL, str(PROJECT_DIR)], check=True)
os.chdir(PROJECT_DIR)
subprocess.run(["python", "-m", "pip", "install", "-e", ".[unet]"], check=True)
subprocess.run([
    "python", "scripts/run_unet_sen1floods11_colab.py",
    "--epochs", "50",
    "--batch-size", "4",
    "--learning-rate", "1e-3",
    "--early-stopping-patience", "10",
    "--split-protocol", "random_chip_split",
    "--eval-split", "test",
    "--force",
], check=True)
```

The helper script performs the reproducible workflow after the repo is present:

1. Mount or assume Google Drive.
2. Read the existing prepared 512 zip.
3. Extract it to `/content/data`.
4. Train/evaluate U-Net.
5. Run BWER-Audit v2.
6. Write one fused output zip containing original outputs plus `bwer_v2/`.

```bash
python scripts/run_unet_sen1floods11_colab.py \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --early-stopping-patience 10 \
  --split-protocol random_chip_split \
  --eval-split test \
  --force
```

Expected final output zip:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/unet_sen1floods11_full_512.zip
```

That zip should contain:

```text
unet_sen1floods11_full_512/
  segmentation_metrics.csv
  event_segmentation_metrics.csv
  audit_table.csv
  bwer_summary.csv
  bwer_by_slice.csv
  support_diagnostics.csv
  warnings.json
  report.md
  model_debug.json
  run_metadata.json
  bwer_v2/
    bwer_v2_summary.csv
    standardised_bwer.csv
    event_failure_analysis.csv
    bwer_audit_report.md
    figures/
```

## Standalone Prithvi vs U-Net Comparison

Comparison outputs are stored separately from single-model result zips so they
can later include more runs, such as event-held-out U-Net, NDWI, DOFA, or CROMA
variants.

Explicit CLI:

```bash
python -m rsfm_fairness_audit.cli compare-runs \
  --dataset sen1floods11 \
  --run prithvi=/content/outputs/prithvi_tl_sen1floods11_official_full_512 \
  --run unet=/content/outputs/unet_sen1floods11_full_512 \
  --output-dir /content/outputs/comparisons/sen1floods11_prithvi_vs_unet_512
```

Colab helper:

```bash
python scripts/run_sen1floods11_comparison_colab.py --force
```

Expected standalone comparison zip:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/comparisons/sen1floods11_prithvi_vs_unet_512.zip
```

The comparison writes `comparison_summary.csv`, `average_vs_bwer.csv`,
`event_level_comparison.csv`, figures, and `comparison_report.md`. The report
is protocol-aware: Prithvi is the official task-adapted checkpoint; U-Net is a
`supervised_baseline` under `random_chip_split` test evaluation unless a
different split is explicitly recorded.

## Interpretation

U-Net BWER outputs are deployment slice-risk diagnostics. In Sen1Floods11,
`event_id` is an operational disaster-event slice, not a causal country
fairness attribute. If the baseline uses `random_chip_split`, state that event
leakage is possible and do not claim event-held-out generalization.

Step 3 can compare Prithvi TL and U-Net using both aggregate segmentation
performance and BWER v2 tail-risk outputs. Ranking reversals should be reported
only after filtering by adaptation protocol and split protocol.
