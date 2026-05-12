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

## 5. Obtain Official BigEarthNet Metadata And Data

Use official BigEarthNet v2.0 sources:

- Homepage: https://bigearth.net/
- Zenodo record: https://zenodo.org/records/10891137
- Description PDF: https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf

BigEarthNet v2.0 is large. Do not download the full archive into a temporary
Colab runtime unless you know the storage budget is enough. Prefer one of these
workflows:

1. Download official metadata first, inspect columns, and prepare a small
   subset from already available Sentinel chips.
2. Store a manageable real subset in Google Drive, then mount Drive in Colab.
3. Use Colab only for model inference after preparing real `.npy`/`.npz` chips
   elsewhere.

The official dataset is licensed under CDLA-Permissive-1.0. Keep the source
metadata and any data access notes with your experiment outputs.

## 6. Prepare A Real lc-col BigEarthNet Smoke Subset

For the first real-data smoke result, use the third-party
`lc-col/bigearthnet` HDF5 export. This is not official BigEarthNet v2.0, but it
contains real Sentinel-2 BigEarthNet chips and is acceptable for a first
pipeline smoke test.

This command downloads exactly one train HDF5 shard from Hugging Face, prints
the HDF5 keys, extracts the first 64 real Sentinel-2 samples, and writes the
adapter format:

```bash
python scripts/download_bigearthnet_lccol_subset.py \
  --output-dir data/bigearthnet_lccol_subset \
  --max-samples 64 \
  --seed 42
```

Expected output:

```text
data/bigearthnet_lccol_subset/
  metadata.csv
  chips/*.npz
```

`lc-col/bigearthnet` does not provide verified country/latitude/longitude in
this export. The pipeline therefore uses `fallback_group=lc_col_train_p0` and
labels the generated `fairness_map.png` as fallback grouping, not geography.

## 7. Run Preflight

```bash
python -m rsfm_fairness_audit.cli check-real \
  --model dofa \
  --dataset bigearthnet \
  --model-config configs/models/dofa.yaml \
  --data-root data/bigearthnet_lccol_subset
```

Fix any `[FAIL]` lines before running inference. `[WARN]` lines are usually
acceptable when they explain optional dependencies or CPU-only fallback.

## 8. Run Smoke Test

```bash
python -m rsfm_fairness_audit.cli run-real \
  --dataset bigearthnet \
  --model dofa \
  --dataset-root data/bigearthnet_lccol_subset \
  --model-config configs/models/dofa.yaml \
  --max-samples 64 \
  --output-dir outputs/dofa_bigearthnet_lccol64
```

## 9. Run Sanity And Phase 1 Runs

After the 64-sample smoke run succeeds, reuse the same downloaded HDF5 cache and
prepare a larger converted subset:

```bash
python scripts/download_bigearthnet_lccol_subset.py \
  --output-dir data/bigearthnet_lccol_subset512 \
  --max-samples 512 \
  --seed 42

python -m rsfm_fairness_audit.cli run-real \
  --dataset bigearthnet \
  --model dofa \
  --dataset-root data/bigearthnet_lccol_subset512 \
  --config configs/models/dofa.yaml \
  --max-samples 512 \
  --output-dir outputs/dofa_bigearthnet_lccol512
```

For the completed Phase 1 run:

```bash
python scripts/download_bigearthnet_lccol_subset.py \
  --output-dir data/bigearthnet_lccol_subset5000 \
  --max-samples 5000 \
  --seed 42

python -m rsfm_fairness_audit.cli run-real \
  --dataset bigearthnet \
  --model dofa \
  --dataset-root data/bigearthnet_lccol_subset5000 \
  --config configs/models/dofa.yaml \
  --max-samples 5000 \
  --output-dir outputs/dofa_bigearthnet_lccol5000
```

The downloader reuses Hugging Face cache files when available and does not
re-download the shard unnecessarily.

## 10. Inspect Outputs

Expected files:

- `embeddings.npz`
- `predictions.csv`
- `fairness_matrix_region.csv`
- `fairness_matrix_sensor.csv`
- `raw_vs_balanced_gap.csv`
- `tables/classwise_metrics.csv`
- `tables/probe_comparison.csv`
- `average_vs_worst.png`
- `figures/average_vs_worst_group.png`
- `figures/raw_vs_balanced_gap.png`
- `sensor_fairness_heatmap.png`
- `representation_shift.png`
- `figures/fairness_map.png`, a fallback-group visualization for lc-col
- `report.md`

For lc-col, `fairness_map.png` is explicitly a fallback-group visualization,
not a real geographic map.

## 11. Download Outputs

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

`Metadata is missing required column(s)`

: Pass explicit `--sample-id-column`, `--s1-path-column`, `--s2-path-column`,
  `--label-column`, `--label-vector-column`, or `--region-column` values that
  match your official metadata export.

`Missing S2 source for sample ...`

: The metadata row points to a chip path that does not exist under
  `--source-root`. Fix the path column or source root.

`CUDA is not available`

: Switch Colab runtime to GPU. Tiny CPU smoke runs may work, but they are slow.
