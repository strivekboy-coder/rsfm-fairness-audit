# DOFA Reproduction Note

## Official Sources

- Paper: https://arxiv.org/abs/2403.15356
- Official code: https://github.com/zhu-xlab/DOFA
- Hugging Face model card: https://huggingface.co/earthflow/DOFA
- Demo notebook: https://github.com/zhu-xlab/DOFA/blob/main/demo.ipynb

## Verified Facts

DOFA is described by its official card as a unified multimodal foundation model for remote sensing and Earth observation. The card and repository state that it was pretrained with five EO modalities and can handle images with arbitrary channel counts when wavelengths are supplied.

The official examples cover Sentinel-1 SAR, Sentinel-2, and NAIP RGB. The repository shows a `torch.hub` loading route:

```python
torch.hub.load("zhu-xlab/DOFA", "vit_base_dofa", pretrained=True)
```

The official examples pass `wave_list` values to `forward_features` and `forward`.

The official `hubconf.py` defines `vit_base_dofa` and downloads
`DOFA_ViT_base_e100.pth` from `https://huggingface.co/earthflow/DOFA/resolve/main/DOFA_ViT_base_e100.pth`.
The Hugging Face file listing also exposes `DOFA_ViT_base_e100_full_weight.pth`,
but the minimal official torch.hub route uses `DOFA_ViT_base_e100.pth`.

Official README examples use:

- Sentinel-1: 2 channels, wavelengths `[5.405, 5.405]`
- Sentinel-2: 9 channels, wavelengths `[0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]`
- NAIP RGB: 3 channels, wavelengths `[0.665, 0.56, 0.49]`

The official demo preprocessing uses 224x224 random resized crop and sensor-specific normalization constants. For deterministic smoke runs, this project resizes prepared chips to 224x224 when configured and applies the official mean/std values without random augmentation.

The official `dofa_v1.py` implementation returns frozen representations from `forward_features(x, wave_list=...)`; this is the embedding layer used by the audit adapter.

## Adapter Plan

Use one of two explicit loading paths:

1. Local official repo path plus explicit local checkpoint path.
2. `torch.hub.load("zhu-xlab/DOFA", "vit_base_dofa", pretrained=True)` only when the user sets `allow_torch_hub_download: true`.

The real adapter accepts a sensor preset, validates band count, normalizes inputs using official constants, supplies wavelengths, resizes to configured image size, and returns `forward_features` embeddings.

## Open Items

- Whether to use `DOFA_ViT_base_e100.pth`, `DOFA_ViT_base_e100_full_weight.pth`, or a newer DOFAv2 file for final paper experiments: to_verify.
- Whether to use DOFA v1 or newer DOFA/DOFAv2 weights for the thesis baseline: to_verify.
- Sentinel-2 band mapping from BigEarthNet v2 to the official 9-channel DOFA demo order: to_verify for any downloaded real subset.
- CPU feasibility for tiny subset inference: to_verify.

## Project Config

The default project config is:

```text
configs/models/dofa.yaml
```

It leaves `repo_path`, `checkpoint_path`, and `allow_torch_hub_download` unset/false so no checkpoint is downloaded automatically.
