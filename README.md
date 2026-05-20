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

## fMoW-Sentinel Global Geography Preflight

This is the Step 2 foundation for a later fMoW-Sentinel global geography BWER
audit. It inventories CSV metadata, derives safe timestamp/latitude fields,
checks slice support, builds a deterministic subset manifest, optionally
inspects a small raster sample, and writes the future audit-table schema. It
does not train models or run inference.

```powershell
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel `
  --metadata-csv path/to/fmow_sentinel.csv `
  --output-dir outputs/fmow_sentinel_preflight/run1 `
  --metadata-only
```

With optional raster inspection:

```powershell
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel `
  --metadata-csv path/to/fmow_sentinel.csv `
  --output-dir outputs/fmow_sentinel_preflight/run1 `
  --split official_split `
  --subset-max-per-split 5000 `
  --inspect-rasters `
  --raster-sample-size 256 `
  --data-root path/to/fmow_sentinel_root `
  --seed 42
```

Outputs include `fmow_metadata_inventory.csv`,
`fmow_slice_support_recommendations.csv`, `subset_manifest.csv`,
`band_statistics_sample.csv`, `raster_loading_report.md`, and
`audit_table_schema_fmow_sentinel.md`. See
[fmow_sentinel.md](D:/Codex/rsfm-fairness-audit/docs/datasets/fmow_sentinel.md).

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
native pixel-level segmentation audit readiness. Chip-level classification is a
sanity audit; native pixel-level Sen1Floods11 segmentation is the paper-grade
disaster/event fairness path. Small 64-sample results are smoke validation only,
not paper-grade flood mapping conclusions.
For Colab, keep `numpy<2.1` because the preinstalled `numba` stack is not
compatible with newer NumPy releases. The Prithvi notebook also upgrades
TerraTorch and uses `terratorch_prithvi_eo_v2_300`, the current TerraTorch
registry name for the 300M non-TL backbone. Sen1Floods11 preparation resolves
valid `S2Hand`/`LabelHand` pairs first, then uses `gsutil -m cp` for batched
cache downloads.

- [Prithvi Sen1Floods11 smoke notebook](D:/Codex/rsfm-fairness-audit/notebooks/prithvi_sen1floods11_colab.ipynb)
- [Native Sen1Floods11 segmentation audit notebook](D:/Codex/rsfm-fairness-audit/notebooks/sen1floods11_native_segmentation_audit_colab.ipynb)

```powershell
python scripts/prepare_sen1floods11_subset.py `
  --output-dir data/sen1floods11_prithvi_subset64 `
  --max-samples 64 `
  --candidate-limit 1000

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

## U-Net Sen1Floods11 Supervised Baseline

The U-Net path is Protocol C: a fully supervised classical baseline for native
Sen1Floods11 flood segmentation. It uses `adaptation_protocol =
supervised_baseline`, records `split_protocol`, ignores `-1` label pixels, and
writes the same event-level BWER-compatible segmentation outputs as Prithvi.
Random chip split is the default baseline split and must not be described as
event-held-out generalization.

```powershell
python -m rsfm_fairness_audit.cli run-unet-sen1floods11 `
  --data-root data/sen1floods11_tl_official_full_512 `
  --output-dir outputs/unet_sen1floods11_full_512 `
  --epochs 50 `
  --batch-size 4 `
  --learning-rate 1e-3 `
  --early-stopping-patience 10 `
  --split-protocol random_chip_split `
  --eval-split test `
  --run-bwer-v2
```

For Colab, reuse the prepared 512 zip and produce one fused output archive:

```powershell
python scripts/run_unet_sen1floods11_colab.py `
  --epochs 50 `
  --batch-size 4 `
  --learning-rate 1e-3 `
  --early-stopping-patience 10 `
  --force
```

The expected final archive is
`/content/drive/MyDrive/rsfm_fairness_audit/outputs/unet_sen1floods11_full_512.zip`.
It contains the original U-Net outputs plus `bwer_v2/`. See
[unet_sen1floods11.md](D:/Codex/rsfm-fairness-audit/docs/reproduction/unet_sen1floods11.md).

For the stronger U-Net-family closure baseline:

```powershell
python scripts/run_unet_sen1floods11_colab.py `
  --architecture s2_resnet34_unet `
  --epochs 50 `
  --batch-size 4 `
  --learning-rate 1e-3 `
  --output-dir /content/outputs/s2_resnet34_unet_sen1floods11_full_512 `
  --output-zip /content/drive/MyDrive/rsfm_fairness_audit/outputs/s2_resnet34_unet_sen1floods11_full_512.zip `
  --force
```

For the diagnostic spectral baseline:

```powershell
python scripts/run_spectral_sen1floods11_colab.py `
  --index mndwi `
  --threshold 0.0 `
  --threshold-policy fixed `
  --force
```

Standalone Sen1Floods11 Prithvi-vs-U-Net comparison:

```powershell
python -m rsfm_fairness_audit.cli compare-runs `
  --dataset sen1floods11 `
  --run prithvi=outputs/prithvi_tl_sen1floods11_official_full_512 `
  --run unet=outputs/unet_sen1floods11_full_512 `
  --output-dir outputs/comparisons/sen1floods11_prithvi_vs_unet_512
```

Four-run closure comparison, after Prithvi TL, vanilla U-Net, spectral MNDWI,
and S2 ResNet34-U-Net output zips exist:

```powershell
python scripts/run_sen1floods11_closure_colab.py --force
```

The closure package writes
`outputs/comparisons/sen1floods11_closure/` with `closure_report.md`,
`closure_comparison_summary.csv`, `closure_average_vs_bwer.csv`,
`closure_event_level_comparison.csv`, and `closure_tail_event_overlap.csv`.
See [sen1floods11_closure.md](D:/Codex/rsfm-fairness-audit/docs/reproduction/sen1floods11_closure.md).

Advanced closure post-hoc checks:

```powershell
python scripts/run_sen1floods11_advanced_closure_colab.py --force
```

This writes protocol-matched chip-intersection diagnostics and Selective Risk
availability/results under `outputs/comparisons/`. LOEO supervised-baseline
runs are separate because they train one model per held-out event:

```powershell
python scripts/run_unet_sen1floods11_loeo_colab.py `
  --architecture vanilla_unet `
  --epochs 50 `
  --batch-size 4 `
  --force
```

## BWER Slice Fairness Audit

The paper-grade audit layer adds BWER: Balanced Worst-slice Excess Risk. It
is a support-aware, composition-standardised, CVaR-style tail-risk statistic for
deployment-relevant remote sensing slices. It operates on a normalized tabular
audit table, so it can consume existing classification predictions,
segmentation metric rows, or pre-aggregated score tables without rewriting
model adapters. For segmentation, formal BWER should use event/slice risk from
aggregated TP/FP/FN/TN and valid-pixel support, not chip-level macro IoU alone.

```powershell
python -m rsfm_fairness_audit.cli evaluate-bwer `
  --audit-table outputs/audit_table.csv `
  --dataset dummy `
  --model dummy `
  --task classification `
  --output-dir outputs/audit/dummy `
  --missing-balance-policy renormalize `
  --bootstrap 200
```

For existing outputs, build the audit table and evaluate in one step:

```powershell
python -m rsfm_fairness_audit.cli run-audit `
  --predictions outputs/run/predictions.csv `
  --dataset bigearthnet_lccol `
  --model dofa `
  --task classification `
  --output-dir outputs/audit/dofa_bigearthnet_lccol `
  --bootstrap 200
```

Audit outputs include `audit_table.csv`, `bwer_summary.csv`,
`bwer_by_slice.csv`, `bootstrap_ci.csv`, `warnings.json`, publication-oriented
figures, and `report.md`. BWER reports deployment-relevant slice risk; it does
not claim causal bias. In Sen1Floods11 reports, `event_id` is an operational
disaster-event slice, not a causal country fairness attribute.

Balanced BWER supports explicit missing balance-level policies:
`renormalize` keeps the current behavior, `invalidate` excludes slices missing
required balance levels, and `overlap` restricts balancing to levels present in
all valid slices. Each run writes `support_diagnostics.csv`.
