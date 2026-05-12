# AnySat Reproduction Note

## Official Sources

- Paper: https://arxiv.org/abs/2412.14123
- Official code: https://github.com/gastruc/AnySat
- Hugging Face model card: https://huggingface.co/g-astruc/AnySat
- Project page: https://gastruc.github.io/anysat
- Demo notebook: https://github.com/gastruc/AnySat/blob/main/demo.ipynb

## Verified Facts

AnySat is a JEPA-based EO model designed for multiple resolutions, scales, and modalities. Official sources state that it trains on GeoPlex, a collection of five multimodal datasets spanning 11 sensors. The model card describes inputs ranging from 3 to 11 channels and resolutions from 0.2 m to 500 m.

The official quickstart uses:

```python
torch.hub.load("gastruc/anysat", "anysat", pretrained=True, flash_attn=False)
```

Time-series modalities require a companion `_dates` tensor containing day-of-year values from 0 to 364.

## Adapter Plan

Defer AnySat until after DOFA/CROMA. The adapter should validate modality tensor shapes, require date tensors for time series, and expose output modes: `tile`, `patch`, `dense`, and `all`.

## Open Items

- Full official list of 11 sensors and exact modality config names: to_verify.
- Normalization and preprocessing details from official config files: to_verify.
- CPU and non-flash-attention feasibility for subset inference: to_verify.
