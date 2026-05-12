# Clay Reproduction Note

## Official Sources

- Hugging Face model card: https://huggingface.co/made-with-clay/Clay
- Official code: https://github.com/Clay-foundation/model
- Documentation: https://clay-foundation.github.io/model/
- Quickstart: https://clay-foundation.github.io/model/getting-started/quickstart.html
- Basic use: https://clay-foundation.github.io/model/getting-started/basic_use.html

## Verified Facts

Clay is an open-source foundation model for Earth. The official documentation states that the model takes satellite imagery plus location and time information as input and outputs embeddings.

Clay v1.5 documentation lists supported sensors including Landsat C2 L1, Landsat C2 L2 SR, LINZ, MODIS, NAIP, Sentinel-1 RTC, and Sentinel-2 L2A. The model input dictionary includes `pixels`, `time`, `latlon`, `waves`, and `gsd`. Official docs allow zero tensors for missing time/location, but this must be treated carefully in fairness experiments.

## Adapter Plan

Use the official package and `ClayMAEModule.load_from_checkpoint` for the first implementation. The adapter should load `configs/metadata.yaml`, normalize with sensor-specific means/stds, pass wavelengths and GSD, and return encoder embeddings.

## Open Items

- Whether to use the HF Transformers `AutoModel` path or the Lightning checkpoint path for experiments: to_verify.
- Exact BigEarthNet v2 band mapping to Clay `sentinel-2-l2a` 10-band metadata: to_verify.
- Impact of zero-filled lat/lon or time on geographic fairness analysis: to_verify.
