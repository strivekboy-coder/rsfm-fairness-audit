# CROMA Reproduction Note

## Official Sources

- Paper: https://arxiv.org/abs/2311.00566
- NeurIPS page: https://papers.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html
- Official code: https://github.com/antofuller/CROMA
- Hugging Face weights: https://huggingface.co/antofuller/CROMA

## Verified Facts

CROMA learns radar, optical, and joint radar-optical representations from spatially aligned Sentinel-1 and Sentinel-2 inputs. The official README specifies Sentinel-1 as 2 channels and Sentinel-2 as 12 channels, with the cirrus band removed if necessary.

The official usage example uses `PretrainedCROMA` from `use_croma.py`, default image size `120x120`, and official weights `CROMA_base.pt` or `CROMA_large.pt`.

## Adapter Plan

The first adapter should wrap `PretrainedCROMA` with `CROMA_base.pt`, support `SAR_images`, `optical_images`, and joint inference, and expose `SAR_GAP`, `optical_GAP`, and `joint_GAP` as embedding choices.

## Open Items

- Exact BigEarthNet v2 S2 band conversion to CROMA's 12-channel expectation: to_verify.
- CPU-only feasibility for tiny subset inference: to_verify.
- Whether official HF or GitHub should be the canonical checkpoint source in experiments: to_verify.
