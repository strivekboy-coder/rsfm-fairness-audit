# Prithvi-EO-2.0 Reproduction Note

## Official Sources

- Paper: https://arxiv.org/abs/2412.02732
- NASA Technical Reports Server: https://ntrs.nasa.gov/citations/20240015391
- Official release repository: https://github.com/NASA-IMPACT/Prithvi-EO-2.0
- Hugging Face 300M model: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
- Hugging Face config: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/main/config.json
- Sen1Floods11 official repo: https://github.com/cloudtostreet/Sen1Floods11

## Verified Facts

Prithvi-EO-2.0 is an IBM/NASA/Juelich EO foundation model family trained on global HLS time-series samples. This project supports two distinct Sen1Floods11 routes:

- `prithvi`: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` non-TL, frozen encoder, used for classification sanity checks and segmentation pipeline diagnostics.
- `prithvi_tl_sen1floods11`: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`, the official Sen1Floods11 flood segmentation fine-tune, used for formal native segmentation audit.

The official Hugging Face config records `img_size=224`, `num_frames=4`, `in_chans=6`, bands `B02, B03, B04, B05, B06, B07`, and the mean/std values copied into `configs/models/prithvi.yaml`.

Sen1Floods11 provides Sentinel-2 13-band GeoTIFFs and hand-label masks where `-1` is invalid, `0` is not water, and `1` is water. In the official v1.1 bucket, hand labels are stored under `LabelHand/` with names such as `*_LabelHand.tif`; `*_QC.tif` is treated only as a legacy fallback.

The official Sen1Floods11 TL model card states that the fine-tune segments Sentinel-2 flood extent from 446 labeled 512x512 chips across 11 flood events, using six bands: Blue, Green, Red, Narrow NIR, SWIR, and SWIR 2. Its TerraTorch loading path is `BACKBONE_REGISTRY.build("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11")`.

## Adapter And Data Plan

Use TerraTorch's official registry path for `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`. Current TerraTorch builds expose the 300M non-TL backbone as `terratorch_prithvi_eo_v2_300`, so `configs/models/prithvi.yaml` keeps the HF model id for provenance and uses `terratorch_model_name` for registry loading. The adapter preserves `[T, C, H, W]` structure and repeats single-timestamp Sen1Floods11 S2 chips to four frames as a compatibility shim, not as a true temporal experiment.

The classification sanity path uses a chip-level `water_present` label derived from valid-water fraction in the hand-label mask. It is not the main paper-grade disaster fairness experiment.

The native segmentation path ignores invalid mask pixels and writes per-chip TP/FP/FN/TN, event-level aggregated segmentation metrics, a normalized segmentation audit table, BWER support preflight outputs, and raw event-level BWER where support permits. Event metrics are micro IoU/Dice/F1/precision/recall computed from aggregated counts.

The non-TL Prithvi route is honestly labeled `frozen_encoder_lightweight_head` because it uses the frozen encoder with a lightweight threshold head, not a supervised flood decoder. Diagnostic runs showed that this head badly overpredicts water and produces very low IoU; NDWI-like diagnostic baselines perform much better on the validation subset, which supports the conclusion that the data/label path is plausible and the naive head is the weak link.

The official TL route is labeled `task_adapted_decoder` with `training_budget=official_sen1floods11_finetune`. Use `configs/models/prithvi_tl_sen1floods11.yaml` and `--model prithvi_tl_sen1floods11` for the formal Sen1Floods11 native segmentation path. To avoid TerraTorch registry-name drift across releases, the adapter downloads the official Hugging Face `config.yaml` and `Prithvi-EO-V2-300M-TL-Sen1Floods11.pt` files and loads them with `LightningInferenceModel.from_config`, matching the official inference script.

For the TL route, the prepared band profile must be `prithvi_tl_sen1floods11`, not the non-TL `B02-B07` compatibility profile. The TL profile selects Sentinel-2 source indices `[1,2,3,8,11,12]`, corresponding to `BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`.

The TL adapter must reproduce the official preprocessing path: scale source reflectance to `0-1`, run the TerraTorch datamodule `test_transform` and `aug`, restore the singleton time axis, and feed the model as `[B,C,T,H,W]`. It also uses 512x512 windowed inference to match the official inference path. Skipping the datamodule transform can produce highly confident all-background predictions even when the checkpoint and bands are otherwise correct.

Colab entrypoint: [prithvi_sen1floods11_colab.ipynb](D:/Codex/rsfm-fairness-audit/notebooks/prithvi_sen1floods11_colab.ipynb).

The preparation script scans official S2 candidates and keeps trying until it finds valid S2/label pairs. It resolves valid pairs first, then uses `gsutil -m cp -I` to batch-download missing GeoTIFFs into the cache; `--no-parallel-download` is available as a conservative fallback. If a specific flood event has missing labels in GCS, use `--event-filter India` or another event, or increase `--candidate-limit`. If GCS is unavailable, pass `--source-root` pointing at a local rsync/HF mirror with matching `S2Hand` and `LabelHand` files.

## Caveats

- The 64-sample run is a smoke validation only and is not paper-grade flood mapping evidence.
- The non-TL `prithvi` checkpoint is not a flood segmentation fine-tune.
- If TerraTorch does not expose dense token features in a stable output key, the non-TL segmentation path uses transparent spectral-feature fallback for mask/metric validation.
- The official TL model should be evaluated with an explicitly recorded prepared resolution. The Colab-friendly 224x224 preparation is useful for validation; final paper runs may choose 512x512 to match the official chip size if memory allows.
- Sen1Floods11 `event_id` is an operational disaster-event slice, not a causal country fairness attribute.

## Final Colab Artifact Names

Prepared data:

```text
/content/data/sen1floods11_tl_official_full_512
/content/drive/MyDrive/rsfm_fairness_audit/prepared_zips/sen1floods11_prithvi_tl_official_full_512.zip
```

Canonical final output:

```text
/content/outputs/prithvi_tl_sen1floods11_official_full_512
/content/drive/MyDrive/rsfm_fairness_audit/outputs/prithvi_tl_sen1floods11_official_full_512.zip
```

The canonical output zip is the public/final result artifact. It contains the
original segmentation audit outputs plus the post-hoc `bwer_v2/` directory.

Post-hoc BWER-Audit v2 enrichment:

```text
python -m rsfm_fairness_audit.cli run-bwer-v2 \
  --input-dir /content/outputs/prithvi_tl_sen1floods11_official_full_512 \
  --output-dir /content/outputs/prithvi_tl_sen1floods11_official_full_512/bwer_v2

/content/outputs/prithvi_tl_sen1floods11_official_full_512/bwer_v2
```

The BWER v2 step is post-hoc. It reads saved event-level segmentation metrics
and reports; it does not rerun model inference or re-prepare data. The final
published zip should be the fused output directory, not a separate BWER-only
archive.

Expected full-run checks:

```text
446 chip rows in segmentation_metrics.csv
11 event rows in event_segmentation_metrics.csv
nonzero predicted_positive_pixel_count overall
BWER(event_id) in bwer_summary.csv
warnings.json contains invalid-balance warnings for country|country and event_id|event
```
