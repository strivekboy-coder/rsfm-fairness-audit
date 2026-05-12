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

## Adapter Plan

Start with `torch.hub` loading and a no-download test stub. The real adapter should accept a sensor preset, normalize inputs using official constants, supply wavelengths, and return `forward_features` embeddings. BigEarthNet v2 should be introduced only after band mapping is explicit.

## Open Items

- Exact checkpoint to pin for paper experiments: to_verify.
- Whether to use DOFA v1 or newer DOFA/DOFAv2 weights for the thesis baseline: to_verify.
- Sentinel-2 band mapping for BigEarthNet v2: to_verify.
- CPU feasibility for tiny subset inference: to_verify.
