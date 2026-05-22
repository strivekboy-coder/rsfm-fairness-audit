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

If you are rebuilding from the SatMAE fMoW-Sentinel `train.csv` / `val.csv`,
run metadata enrichment first. SatMAE CSVs alone typically contain `category`,
`location_id`, `timestamp`, and `image_id`; they do not by themselves provide
country/region geography slices.

The current final reproducible fMoW-Sentinel metadata package is expected under
Google Drive as:

```text
/content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/
```

The final preflight evidence archive is:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/fmow_sentinel_preflight/enriched_geography_final_v1.zip
```

```powershell
python -m rsfm_fairness_audit.cli enrich-fmow-sentinel-metadata `
  --satmae-csv path/to/train.csv `
  --satmae-csv path/to/val.csv `
  --external-metadata-csv path/to/original_fmow_or_gps_metadata.csv `
  --country-region-map /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_country_region_map_full_v1.csv `
  --output-dir outputs/fmow_sentinel_metadata_enrichment/run1
```

Omit `--external-metadata-csv` when no verified external geography table is
available; the command will write a limitation report instead of fabricating
country, region, or coordinates.
If a verified country mapping is available, `--country-region-map` fills
`continent`, `un_region`, and optional `region` from the supplied country
field. Country slices with sparse small countries are reported as
support-filtered formal candidates rather than blindly formal-ready.

```powershell
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel `
  --metadata-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_geography_final_v1.csv `
  --country-region-map /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_country_region_map_full_v1.csv `
  --output-dir outputs/fmow_sentinel_preflight/enriched_geography_final_v1 `
  --metadata-only
```

With optional raster inspection:

```powershell
python -m rsfm_fairness_audit.cli preflight-fmow-sentinel `
  --metadata-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_geography_final_v1.csv `
  --country-region-map /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_country_region_map_full_v1.csv `
  --output-dir outputs/fmow_sentinel_preflight/enriched_geography_final_v1_raster_sample `
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

## fMoW-Sentinel Step 3 Image-Only Geography BWER Prototype

Step 3 starts from the final enriched metadata package produced in Step 2. It
does not redo metadata enrichment and does not feed geography metadata into the
model. Geography fields are carried only into prediction tables, support
diagnostics, and BWER reporting.

The finalized Step 3 dataset/protocol record is
`fmow_sentinel_clean_subset_30k_location_disjoint_v2`. It was extracted from
the official Stanford PURL `fmow-sentinel.tar.gz` after downloading the full
tarball to Colab local `/content` storage, without fully extracting the archive.
Earlier streaming partial-extraction experiments are excluded from formal data.

Canonical Colab paths:

```text
/content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv
/content/drive/MyDrive/rsfm_fairness_audit/fmow_sentinel_clean_subset_30k_location_disjoint_v2.zip
```

The final manifest has 30,000 rows, a location-disjoint 70/30 split
(`group = category + location_id`), `split_original` preserving the source
split, and final `split` values of `train` / `val`. Location overlap between
train and val is zero. It contains 21,046 train rows and 8,954 validation rows,
all 62 categories in validation with minimum validation category support of 33,
195 countries with country missing ratio 0, and continent / UN region / region
missing ratio 0.024. Do not use the earlier 10k subset or a non-location-
disjoint manifest for main Step 3 experiments.

Primary formal slices for this dataset are `continent`, `un_region`, `region`,
`latitude_band`, `season`, and `category`. Country-level BWER should use
validation support thresholds such as `>=20` or `>=30`; `country x category`
is diagnostic unless support-threshold filtered.

Prepare a clean subset from the official local archive without extracting all
TIFF files when rebuilding from source:

```powershell
python scripts/prepare_fmow_sentinel_clean_subset.py `
  --archive /content/fmow-sentinel.tar.gz `
  --metadata-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_sample_manifest_final_v1.csv `
  --output-dir /content/data/fmow_sentinel_clean_subset_v1 `
  --split train `
  --split val `
  --max-samples-per-split 5000 `
  --stratify-field category `
  --stratify-field country `
  --stratify-field region `
  --stratify-field latitude_band `
  --seed 42
```

If the first valid subset needs stronger audit-slice support, augment it
without discarding existing valid samples:

```powershell
python scripts/prepare_fmow_sentinel_clean_subset.py `
  --archive /content/fmow-sentinel.tar.gz `
  --metadata-csv /content/drive/MyDrive/rsfm_fairness_audit/cache/fmow_sentinel/metadata/final/fmow_sentinel_enriched_sample_manifest_final_v1.csv `
  --augment-existing-manifest /content/data/fmow_sentinel_clean_subset_v1/clean_subset_manifest.csv `
  --output-dir /content/data/fmow_sentinel_clean_subset_v1 `
  --split train `
  --split val `
  --target-total 30000 `
  --target-train 15000 `
  --target-val 15000 `
  --seed 42
```

Lightweight supervised image-only sanity/debug baseline:

```powershell
python -m rsfm_fairness_audit.cli run-fmow-sentinel-classification `
  --metadata-csv /content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv `
  --data-root /content/data/fmow_sentinel_clean_subset_30k_v2 `
  --output-dir /content/outputs/fmow_sentinel_supervised_stats_val `
  --model supervised_stats `
  --train-split train `
  --eval-split val `
  --split-protocol location_disjoint `
  --image-size 96 `
  --run-bwer
```

Formal DOFA ViT-B frozen-backbone linear probe:

```powershell
python -m rsfm_fairness_audit.cli run-fmow-sentinel-classification `
  --metadata-csv /content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv `
  --data-root /content/data/fmow_sentinel_clean_subset_30k_v2 `
  --output-dir /content/outputs/fmow_sentinel_dofa_vitb_linear_probe_30k_location_disjoint `
  --model dofa `
  --model-config configs/models/dofa_fmow_sentinel.yaml `
  --probe linear `
  --dofa-input-scale 10000 `
  --train-split train `
  --eval-split val `
  --split-protocol location_disjoint `
  --image-size 224 `
  --batch-size 16 `
  --allow-torch-hub-download `
  --run-bwer
```

The DOFA path extracts frozen ViT-B embeddings once, caches train/val
embeddings under the run output, trains a linear classifier on train
embeddings, and writes confidence/max-probability from classifier softmax.
Nearest-centroid probing is kept only as an optional sanity mode. fMoW-Sentinel
TIFF values are raw reflectance-like values, so the formal DOFA config scales
inputs by `input_scale=10000` before DOFA normalization/embedding extraction.

Paper-grade supervised ResNet-50 baseline on the clean 30k
location-disjoint subset:

```powershell
python -m rsfm_fairness_audit.cli run-fmow-sentinel-classification `
  --metadata-csv /content/data/fmow_sentinel_clean_subset_30k_v2/final_clean_subset_manifest_30k_location_disjoint_v2.csv `
  --data-root /content/data/fmow_sentinel_clean_subset_30k_v2 `
  --output-dir /content/outputs/fmow_sentinel_resnet50_30k_location_disjoint `
  --model resnet50 `
  --train-split train `
  --eval-split val `
  --split-protocol location_disjoint `
  --eval-scope val `
  --image-size 96 `
  --batch-size 32 `
  --epochs 20 `
  --learning-rate 1e-3 `
  --weight-decay 1e-4 `
  --num-workers 2 `
  --run-bwer
```

This ResNet-50 is trained from scratch with `weights=None`, a 13-channel first
convolution, train-split-only per-band normalization, and Sentinel-2 image
inputs only. Geography metadata is copied into prediction/audit rows for BWER
slicing but is not used as model input.

Post-hoc geography BWER can be rerun without model inference:

```powershell
python -m rsfm_fairness_audit.cli run-fmow-geography-bwer `
  --input-dir /content/outputs/fmow_sentinel_supervised_stats_val `
  --output-dir /content/outputs/fmow_sentinel_supervised_stats_val/bwer
```

Compare completed fMoW-Sentinel runs:

```powershell
python -m rsfm_fairness_audit.cli compare-fmow-runs `
  --run supervised=/content/outputs/fmow_sentinel_supervised_stats_val `
  --run dofa=/content/outputs/fmow_sentinel_dofa_vitb_linear_probe_30k_location_disjoint `
  --output-dir /content/outputs/comparisons/fmow_sentinel_supervised_vs_dofa
```

Validate and package a completed Step 3 run without including raster imagery:

```powershell
python -m rsfm_fairness_audit.cli validate-fmow-step3-results `
  --run-dir /content/outputs/fmow_sentinel_supervised_stats_val `
  --full-archive-downloaded-locally true `
  --full-extraction-avoided true `
  --streaming-partial-extraction-excluded true

python -m rsfm_fairness_audit.cli package-fmow-step3-handoff `
  --run-dir /content/outputs/fmow_sentinel_supervised_stats_val `
  --output-zip /content/drive/MyDrive/rsfm_fairness_audit/outputs/fmow_step3_supervised_stats_handoff.zip
```

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
