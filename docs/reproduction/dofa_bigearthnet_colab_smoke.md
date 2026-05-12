# Colab Smoke Run: DOFA + BigEarthNet-Style Subset

This guide runs the first real-model smoke experiment on Google Colab. It does
not download full BigEarthNet v2.0 and does not download DOFA weights unless you
explicitly choose torch.hub mode or provide the checkpoint yourself.

## 1. Start Colab

Use a GPU runtime:

```text
Runtime -> Change runtime type -> T4 GPU or better
```

## 2. Clone Your Repo

Replace the URL with your GitHub repository after pushing:

```bash
git clone https://github.com/<your-user>/rsfm-fairness-audit.git
cd rsfm-fairness-audit
```

## 3. Install Requirements

Base package:

```bash
python -m pip install -e .
python -m pip install PyYAML pytest
```

Optional DOFA runtime:

```bash
python -m pip install -r requirements-dofa.txt
```

Colab usually already has PyTorch installed. If you need a specific CUDA wheel,
use the official PyTorch install selector rather than guessing a version.

## 4. Configure DOFA

Edit `configs/models/dofa.yaml`.

### Option A: Local Official Repo + Explicit Checkpoint

Clone the official DOFA implementation:

```bash
git clone https://github.com/zhu-xlab/DOFA.git /content/DOFA
```

Provide the official checkpoint yourself, for example by uploading it to Drive
or downloading it manually from an official source:

- Hugging Face file: https://huggingface.co/earthflow/DOFA/blob/main/DOFA_ViT_base_e100.pth
- Zenodo record: https://zenodo.org/records/11002557

Then set:

```yaml
repo_path: /content/DOFA
checkpoint_path: /content/drive/MyDrive/<path>/DOFA_ViT_base_e100.pth
allow_torch_hub_download: false
device: auto
```

### Option B: Torch Hub

If you are comfortable letting the first run download the official checkpoint:

```yaml
torch_hub_repo: zhu-xlab/DOFA
model_variant: vit_base_dofa
allow_torch_hub_download: true
device: auto
```

This may download an approximately 448 MB checkpoint. The project default is
`false` so this never happens accidentally.

## 5. Prepare a Tiny BigEarthNet-Compatible Subset

The adapter expects a prepared local subset:

```text
prepared_bigearthnet_subset/
  metadata.csv
  chips/
    BEN-000001_s2.npy
```

See `docs/datasets/bigearthnet_subset_setup.md` for the manifest schema.

If you already have chips and a CSV/JSON/JSONL manifest with `s2_path`:

```bash
python scripts/prepare_bigearthnet_subset.py \
  --source-root <source_root> \
  --metadata-path <source_metadata.csv> \
  --output-root /content/prepared_bigearthnet_subset \
  --subset-size 32 \
  --sensor-mode S2
```

If country or coordinates are not verified from official metadata, leave them
blank or use `to_verify`. Do not infer geographic metadata from filenames.

## 6. Run Preflight

```bash
python -m rsfm_fairness_audit.cli check-real \
  --model dofa \
  --dataset bigearthnet \
  --model-config configs/models/dofa.yaml \
  --data-root /content/prepared_bigearthnet_subset
```

Fix any `[FAIL]` lines before running inference. `[WARN]` lines are usually
acceptable when they explain optional dependencies or CPU-only fallback.

## 7. Run Smoke Test

```bash
python -m rsfm_fairness_audit.cli run-real \
  --dataset bigearthnet \
  --model dofa \
  --data-root /content/prepared_bigearthnet_subset \
  --model-config configs/models/dofa.yaml \
  --subset-size 32 \
  --output-dir outputs/runs/dofa_bigearthnet_real_smoke
```

## 8. Run Sanity Test

After the 32-sample smoke run succeeds:

```bash
python -m rsfm_fairness_audit.cli run-real \
  --dataset bigearthnet \
  --model dofa \
  --data-root /content/prepared_bigearthnet_subset \
  --model-config configs/models/dofa.yaml \
  --subset-size 500 \
  --output-dir outputs/runs/dofa_bigearthnet_real_sanity_500
```

For a larger sanity run, use `--subset-size 1000`.

## 9. Inspect Outputs

Expected files:

- `embeddings.npz`
- `predictions.csv`
- `fairness_matrix_region.csv`
- `fairness_matrix_sensor.csv`
- `raw_vs_balanced_gap.csv`
- `average_vs_worst.png`
- `sensor_fairness_heatmap.png`
- `representation_shift.png`
- `fairness_map.png`, only if verified coordinates exist
- `report.md`

If coordinates are missing, map generation is skipped and the report states why.

## 10. Download Outputs

```bash
zip -r dofa_bigearthnet_outputs.zip outputs/runs/dofa_bigearthnet_real_smoke
```

In Colab:

```python
from google.colab import files
files.download("dofa_bigearthnet_outputs.zip")
```

## Common Errors

`DOFA is not configured for real inference`

: Set `repo_path` plus `checkpoint_path`, or explicitly set
  `allow_torch_hub_download: true`.

`Configured DOFA checkpoint_path does not exist`

: Upload or mount the checkpoint and update `configs/models/dofa.yaml`.

`First chip has N bands but DOFA config expects 9`

: Your prepared subset band count/order does not match the config. Verify the
  band order and update `expected_bands`, `wavelength_list`, and normalization
  constants only from official sources.

`No BigEarthNet metadata found`

: Put `metadata.csv` under `data_root` or pass `--metadata-path`.

`CUDA is not available`

: Switch Colab runtime to GPU. Tiny CPU smoke runs may work, but they are slow.
