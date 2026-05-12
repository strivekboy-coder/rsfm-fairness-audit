# Colab Phase 1 Run: DOFA + lc-col BigEarthNet Subset

This guide runs the Phase 1 real-data experiment on Google Colab with DOFA and
real Sentinel-2 chips from `lc-col/bigearthnet`. It does not download full
BigEarthNet v2.0. In the current Colab path, DOFA uses torch.hub mode and may
download the official checkpoint on the first run.

## 1. Start Colab

Use a GPU runtime:

```text
Runtime -> Change runtime type -> T4 GPU or better
```

## 2. Clone Your Repo

Use your GitHub repository URL:

```bash
git clone https://github.com/strivekboy-coder/rsfm-fairness-audit.git
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

Use the current torch.hub path:

```yaml
torch_hub_repo: zhu-xlab/DOFA
model_variant: vit_base_dofa
repo_path: null
checkpoint_path: null
band_profile: sentinel2_12_lccol
expected_bands: 12
allow_torch_hub_download: true
device: auto
```

This may download the official DOFA checkpoint on the first run. Do not clone
the DOFA repo manually for this Colab workflow unless you are deliberately
testing the local-repo path.

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
  --cache-dir data/_cache/lc_col_bigearthnet \
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

## 9. Run The 5000-Sample Phase 1 Run

After the 64-sample smoke run succeeds, reuse the same downloaded HDF5 cache and
prepare the completed Phase 1 subset:

```bash
python scripts/download_bigearthnet_lccol_subset.py \
  --output-dir data/bigearthnet_lccol_subset5000 \
  --cache-dir data/_cache/lc_col_bigearthnet \
  --max-samples 5000 \
  --seed 42

python -m rsfm_fairness_audit.cli run-real \
  --dataset bigearthnet \
  --model dofa \
  --dataset-root data/bigearthnet_lccol_subset5000 \
  --config configs/models/dofa.yaml \
  --max-samples 5000 \
  --output-dir outputs/dofa_bigearthnet_lccol5000 \
  --chunk-size 256 \
  --streaming-embeddings true
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

## 11. Package Final Outputs

Package only the final 5000-sample report artifacts. Do not include
`embeddings.npz`, `predictions.csv`, `embedding_chunks/`, `data/`, or the HDF5
cache in the final zip.

In Colab:

```python
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from google.colab import files

final_run = Path("outputs/dofa_bigearthnet_lccol5000")
zip_path = Path("dofa_bigearthnet_phase1_results.zip")
root_artifacts = [
    "report.md",
    "fairness_summary.csv",
    "fairness_matrix_region.csv",
    "fairness_matrix_sensor.csv",
    "fairness_matrix_task.csv",
    "raw_vs_balanced_gap.csv",
    "classwise_metrics.csv",
    "probe_comparison.csv",
]
files_to_zip = [final_run / name for name in root_artifacts if (final_run / name).exists()]
for folder_name in ["tables", "figures"]:
    folder = final_run / folder_name
    if folder.exists():
        files_to_zip.extend(sorted(path for path in folder.rglob("*") if path.is_file()))

with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
    for path in files_to_zip:
        archive.write(path, path.relative_to(final_run).as_posix())
files.download(str(zip_path))
```

Optional cleanup after the zip has downloaded successfully:

```bash
rm -rf data/bigearthnet_lccol_subset*
rm -rf data/_cache/lc_col_bigearthnet
rm -rf outputs/dofa_bigearthnet_lccol64
rm -rf outputs/dofa_bigearthnet_lccol512
rm -rf outputs/dofa_bigearthnet_lccol5000/embedding_chunks
rm -f outputs/dofa_bigearthnet_lccol5000/embeddings.npz
rm -f outputs/dofa_bigearthnet_lccol5000/predictions.csv
```

## Common Errors

`DOFA is not configured for real inference`

: Set `repo_path` plus `checkpoint_path`, or explicitly set
  `allow_torch_hub_download: true`.

`Configured DOFA checkpoint_path does not exist`

: Upload or mount the checkpoint and update `configs/models/dofa.yaml`.

`First chip has N bands but DOFA config expects 12`

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
