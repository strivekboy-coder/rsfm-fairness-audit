# Prithvi-EO-2.0 Reproduction Note

## Official Sources

- Paper: https://arxiv.org/abs/2412.02732
- NASA Technical Reports Server: https://ntrs.nasa.gov/citations/20240015391
- Official release repository: https://github.com/NASA-IMPACT/Prithvi-EO-2.0
- Hugging Face 300M model: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
- Hugging Face config: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/main/config.json
- Sen1Floods11 official repo: https://github.com/cloudtostreet/Sen1Floods11

## Verified Facts

Prithvi-EO-2.0 is an IBM/NASA/Juelich EO foundation model family trained on global HLS time-series samples. This project uses `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` non-TL only, not the Sen1Floods11 fine-tuned model.

The official Hugging Face config records `img_size=224`, `num_frames=4`, `in_chans=6`, bands `B02, B03, B04, B05, B06, B07`, and the mean/std values copied into `configs/models/prithvi.yaml`.

Sen1Floods11 provides Sentinel-2 13-band GeoTIFFs and QC masks where `-1` is invalid, `0` is not water, and `1` is water.

## Adapter And Data Plan

Use TerraTorch's official registry path for `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`. The adapter preserves `[T, C, H, W]` structure and repeats single-timestamp Sen1Floods11 S2 chips to four frames as a compatibility shim, not as a true temporal experiment.

The classification sanity path uses a chip-level `water_present` label derived from valid-water fraction in the QC mask. The segmentation path is a smoke validation that ignores invalid QC pixels and reports group IoU/accuracy.

Colab entrypoint: [prithvi_sen1floods11_colab.ipynb](D:/Codex/rsfm-fairness-audit/notebooks/prithvi_sen1floods11_colab.ipynb).

## Caveats

- The 64-sample run is a smoke validation only and is not paper-grade flood mapping evidence.
- The selected Prithvi checkpoint is not a flood segmentation fine-tune.
- If TerraTorch does not expose dense token features in a stable output key, the segmentation smoke uses transparent spectral-feature fallback for mask/metric validation.
