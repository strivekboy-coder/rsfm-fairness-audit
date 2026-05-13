# rsfm-fairness-audit

Research-grade fairness auditing framework for Remote Sensing Foundation Models.

The first milestone is intentionally small and CPU-only: a fully runnable dummy
pipeline with synthetic multi-band imagery, severe region/class/sensor
imbalance, deterministic embeddings, balanced sampling, fairness metrics, CSV
outputs, and static figures. Real model and dataset adapters are added through
the same interfaces without changing the evaluation pipeline.

## Smoke Run

```powershell
python -m pip install -e .
python -m rsfm_fairness_audit.cli run-dummy --output-dir outputs/dummy_smoke
python -m pytest
```

Generated artifacts include fairness matrices, raw-vs-balanced gap tables,
sensor heatmaps, average-vs-worst scatter plots, representation shift plots,
and a static Markdown report.

## BigEarthNet + DOFA Runs

### A. Mocked Real Pipeline

Use a prepared BigEarthNet-style subset with `.npy`/`.npz` chips and manifest
metadata. Tests use an injected mock model; the CLI path expects a configured
real model unless `--allow-torch-hub-download` or `--model-config` points to a
usable DOFA setup.

```powershell
python -m rsfm_fairness_audit.cli run-real `
  --dataset bigearthnet `
  --model dofa `
  --data-root <mock_subset_path> `
  --subset-size 32 `
  --output-dir outputs/runs/dofa_bigearthnet_mock
```

### B. Official DOFA + Prepared BigEarthNet Smoke

First prepare a subset manifest as described in
[bigearthnet_subset_setup.md](D:/Codex/rsfm-fairness-audit/docs/datasets/bigearthnet_subset_setup.md).
Then fill `repo_path` and `checkpoint_path` in
[dofa.yaml](D:/Codex/rsfm-fairness-audit/configs/models/dofa.yaml), or set
`allow_torch_hub_download: true` explicitly.

```powershell
python -m rsfm_fairness_audit.cli run-real `
  --dataset bigearthnet `
  --model dofa `
  --data-root <prepared_subset_path> `
  --model-config configs/models/dofa.yaml `
  --subset-size 32 `
  --output-dir outputs/runs/dofa_bigearthnet_real_smoke
```

### C. Medium Sanity Run

```powershell
python -m rsfm_fairness_audit.cli run-real `
  --dataset bigearthnet `
  --model dofa `
  --data-root <prepared_subset_path> `
  --model-config configs/models/dofa.yaml `
  --subset-size 1000 `
  --output-dir outputs/runs/dofa_bigearthnet_real_sanity
```

The current Phase 1 Colab config uses `allow_torch_hub_download: true`, so the
first real DOFA run may download the official DOFA checkpoint. Large datasets
are still not downloaded automatically; the lc-col subset downloader fetches one
explicit HDF5 shard only when you run the script. Optional DOFA runtime
dependencies are listed in `requirements-dofa.txt`.

## Running Real DOFA Smoke Test On Colab

For the first real DOFA + BigEarthNet-style subset run, use the Colab-first
guide and notebook:

- [Colab smoke guide](D:/Codex/rsfm-fairness-audit/docs/reproduction/dofa_bigearthnet_colab_smoke.md)
- [Colab notebook template](D:/Codex/rsfm-fairness-audit/notebooks/dofa_bigearthnet_smoke_colab.ipynb)

Start with:

```powershell
python -m rsfm_fairness_audit.cli check-real `
  --model dofa `
  --dataset bigearthnet `
  --model-config configs/models/dofa.yaml `
  --data-root <prepared_subset_path>
```

## Phase 2A CROMA On lc-col BigEarthNet S2

CROMA is integrated for optical-only Phase 2A runs on the same real
lc-col/BigEarthNet Sentinel-2 subset used for DOFA. Install optional
dependencies separately:

```powershell
python -m pip install -r requirements-croma.txt
```

Configure [croma.yaml](D:/Codex/rsfm-fairness-audit/configs/models/croma.yaml)
with `repo_path` or `source_file_path` pointing to the official
`antofuller/CROMA` implementation. The checkpoint download path is restricted
to official `antofuller/CROMA` weights.

```powershell
python -m rsfm_fairness_audit.cli run-real `
  --dataset bigearthnet `
  --model croma `
  --data-root data/bigearthnet_lccol_subset `
  --model-config configs/models/croma.yaml `
  --subset-size 64 `
  --output-dir outputs/croma_bigearthnet_lccol64
```

For the 5000-sample Phase 2A run, reuse the lc-col HDF5 cache and chunk
embedding extraction:

```powershell
python -m rsfm_fairness_audit.cli run-real `
  --dataset bigearthnet `
  --model croma `
  --data-root data/bigearthnet_lccol_subset5000 `
  --model-config configs/models/croma.yaml `
  --subset-size 5000 `
  --output-dir outputs/croma_bigearthnet_lccol5000 `
  --chunk-size 256 `
  --streaming-embeddings true
```

Compare completed DOFA and CROMA runs:

```powershell
python -m rsfm_fairness_audit.cli compare-runs `
  --dataset bigearthnet `
  --run dofa=outputs/dofa_bigearthnet_lccol5000 `
  --run croma=outputs/croma_bigearthnet_lccol5000 `
  --output-dir outputs/dofa_vs_croma_lccol5000
```

True CROMA sensor fairness is Phase 2B and requires a verified aligned S1/S2
dataset. The lc-col smoke path is S2-only and must not be used to claim
SAR-vs-optical fairness.

## Phase 2B CROMA Sensor Fairness On BEN-GE-800

Phase 2B uses BEN-GE-800, a lightweight paired Sentinel-1/Sentinel-2 subset,
to compare CROMA SAR-only, optical-only, and S1+S2 fusion modes. The Colab
workflow is:

- [CROMA BEN-GE-800 sensor fairness notebook](D:/Codex/rsfm-fairness-audit/notebooks/croma_benge800_sensor_fairness_colab.ipynb)

The default 64-sample BEN-GE-800 run is a smoke validation only. Extreme
worst-group or gap values should not be interpreted as paper-grade fairness
conclusions.

Core commands:

```powershell
python scripts/prepare_ben_ge_800_subset.py `
  --output-dir data/ben_ge_800_subset64 `
  --max-samples 64 `
  --seed 42

python -m rsfm_fairness_audit.cli run-real `
  --dataset ben_ge `
  --model croma `
  --data-root data/ben_ge_800_subset64 `
  --sensor-mode S1+S2 `
  --model-config configs/models/croma_both.yaml `
  --output-dir outputs/croma_benge800_both64 `
  --max-samples 64

python -m rsfm_fairness_audit.cli compare-sensor-modes `
  --dataset ben_ge `
  --run sar=outputs/croma_benge800_sar64 `
  --run optical=outputs/croma_benge800_optical64 `
  --run both=outputs/croma_benge800_both64 `
  --output-dir outputs/comparisons/croma_benge800_sensor64
```

## Prithvi-EO-2.0 On Sen1Floods11

The third model path uses `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`
non-TL with Sen1Floods11. It runs both chip-level classification sanity and
a lightweight segmentation fairness smoke. The 64-sample results are smoke
validation only, not paper-grade flood mapping conclusions.

- [Prithvi Sen1Floods11 notebook](D:/Codex/rsfm-fairness-audit/notebooks/prithvi_sen1floods11_colab.ipynb)

```powershell
python scripts/prepare_sen1floods11_subset.py `
  --output-dir data/sen1floods11_prithvi_subset64 `
  --max-samples 64

python -m rsfm_fairness_audit.cli run-real `
  --dataset sen1floods11 `
  --model prithvi `
  --data-root data/sen1floods11_prithvi_subset64 `
  --model-config configs/models/prithvi.yaml `
  --output-dir outputs/prithvi_sen1floods11_class64 `
  --max-samples 64

python -m rsfm_fairness_audit.cli run-segmentation-real `
  --dataset sen1floods11 `
  --model prithvi `
  --data-root data/sen1floods11_prithvi_subset64 `
  --model-config configs/models/prithvi.yaml `
  --output-dir outputs/prithvi_sen1floods11_seg64 `
  --max-samples 64
```
