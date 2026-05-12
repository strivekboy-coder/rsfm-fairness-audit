# Prithvi-EO-2.0 Reproduction Note

## Official Sources

- Paper: https://arxiv.org/abs/2412.02732
- NASA Technical Reports Server: https://ntrs.nasa.gov/citations/20240015391
- Official release repository: https://github.com/NASA-IMPACT/Prithvi-EO-2.0
- Hugging Face 300M model: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
- Hugging Face 300M TL model: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
- Demo Space index from paper page: https://huggingface.co/papers/2412.02732

## Verified Facts

Prithvi-EO-2.0 is an IBM/NASA/Jülich EO foundation model family. The official paper page states that it was trained on global time-series samples from NASA Harmonized Landsat Sentinel-2 at 30 m resolution. The Hugging Face cards list TerraTorch as the loading library and Apache-2.0 licensing.

The TL variants use date and geolocation metadata: year/day-of-year and center latitude/longitude.

## Adapter Plan

Use TerraTorch's official registry path for the selected HF model ID. Start with `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`, not a TL variant, unless temporal/location metadata is ready. The adapter must preserve temporal tensor structure and should not flatten time into channels.

## Open Items

- TerraTorch example path to use for the first adapter smoke test: to_verify.
- Exact HLS band order and normalization for the selected checkpoint: to_verify.
- Compatibility path from BigEarthNet v2 Sentinel-2 patches to HLS-trained Prithvi inputs: to_verify.
