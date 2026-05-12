# SpaceNet7 Dataset Note

## Official Sources

- Challenge page: https://spacenet.ai/sn7-challenge/
- SpaceNet datasets page: https://spacenet.ai/datasets/
- Paper: https://arxiv.org/abs/2102.11958
- GeoBench-2 HF mirror/license attribution: https://huggingface.co/datasets/aialliance/spacenet7

## Verified Facts

SpaceNet7 is the SpaceNet Multi-Temporal Urban Development Challenge dataset. The official challenge page describes Planet satellite imagery mosaics with 24 monthly observations over 101 AOIs, more than 40,000 square kilometers of imagery, and more than 11 million building annotations.

The official challenge page states that the dataset is hosted as an AWS public dataset and licensed under CC BY-SA 4.0.

## Fairness Use

SpaceNet7 is optional for later milestones. It is useful for urban temporal change fairness but is not needed for the first BigEarthNet/DOFA integration.

## Open Items

- Exact AOI-to-country/continent metadata fields in the downloaded archive: to_verify.
- Whether to use original AWS release or GeoBench-2 mirror for experiments: to_verify.
- Fair temporal sampling strategy under cloud/unusable-data masks: to_verify.
