# CROMA Reproduction Note

## Official Sources

- Paper: https://arxiv.org/abs/2311.00566
- NeurIPS page: https://papers.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html
- Official code: https://github.com/antofuller/CROMA
- Hugging Face weights: https://huggingface.co/antofuller/CROMA
- Official usage file: https://github.com/antofuller/CROMA/blob/main/use_croma.py

## Verified Facts

CROMA learns radar, optical, and joint radar-optical representations from spatially aligned Sentinel-1 and Sentinel-2 inputs. The official README specifies Sentinel-1 as 2 channels and Sentinel-2 as 12 channels, with the cirrus band removed if necessary.

The official usage example uses `PretrainedCROMA` from `use_croma.py`, default image size `120x120`, and official weights `CROMA_base.pt` or `CROMA_large.pt`.

The official model wrapper supports `modality="optical"`, `modality="SAR"`, and `modality="both"`. It returns global pooled embeddings such as `optical_GAP`, `SAR_GAP`, and `joint_GAP`, plus patch-level encodings.

## Adapter Plan

Phase 2A wraps `PretrainedCROMA` with `CROMA_base.pt` in optical-only mode. This is compatible with the current lc-col BigEarthNet Sentinel-2 subset because that subset provides 12-channel real S2 chips.

The project config is [configs/models/croma.yaml](D:/Codex/rsfm-fairness-audit/configs/models/croma.yaml). It allows Hugging Face download only from `antofuller/CROMA` and only for `CROMA_base.pt` or `CROMA_large.pt`. The adapter still requires an explicit official source implementation path through `source_file_path` or `repo_path` because the checkpoint and model code are separate artifacts.

Minimal Colab setup:

```bash
pip install -e .
pip install -r requirements-croma.txt
git clone https://github.com/antofuller/CROMA /content/CROMA
python - <<'PY'
from pathlib import Path
import yaml
path = Path("configs/models/croma.yaml")
config = yaml.safe_load(path.read_text())
config["repo_path"] = "/content/CROMA"
config["source_file_path"] = None
config["checkpoint_path"] = None
config["allow_hf_download"] = True
path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

Quick check:

```bash
python scripts/download_bigearthnet_lccol_subset.py \
  --output-dir data/bigearthnet_lccol_subset \
  --cache-dir data/_cache/lc_col_bigearthnet \
  --max-samples 64 \
  --seed 42

python -m rsfm_fairness_audit.cli check-real \
  --dataset bigearthnet \
  --model croma \
  --model-config configs/models/croma.yaml \
  --data-root data/bigearthnet_lccol_subset

python -m rsfm_fairness_audit.cli run-real \
  --dataset bigearthnet \
  --model croma \
  --data-root data/bigearthnet_lccol_subset \
  --model-config configs/models/croma.yaml \
  --subset-size 64 \
  --output-dir outputs/croma_bigearthnet_lccol64
```

Main Phase 2A run:

```bash
python scripts/download_bigearthnet_lccol_subset.py \
  --output-dir data/bigearthnet_lccol_subset5000 \
  --cache-dir data/_cache/lc_col_bigearthnet \
  --max-samples 5000 \
  --seed 42

python -m rsfm_fairness_audit.cli run-real \
  --dataset bigearthnet \
  --model croma \
  --data-root data/bigearthnet_lccol_subset5000 \
  --model-config configs/models/croma.yaml \
  --subset-size 5000 \
  --output-dir outputs/croma_bigearthnet_lccol5000 \
  --chunk-size 256 \
  --streaming-embeddings true
```

DOFA vs CROMA comparison after both 5000 runs:

```bash
python -m rsfm_fairness_audit.cli compare-runs \
  --dataset bigearthnet \
  --run dofa=outputs/dofa_bigearthnet_lccol5000 \
  --run croma=outputs/croma_bigearthnet_lccol5000 \
  --output-dir outputs/dofa_vs_croma_lccol5000
```

## Phase 2B Boundary

Phase 2B is implemented with BEN-GE-800 as the immediate lightweight paired S1/S2 dataset. It runs three separate CROMA modes on the same paired sample set:

- `SAR`: Sentinel-1 VV/VH only, extracting `SAR_GAP`.
- `optical`: Sentinel-2 12-band only, extracting `optical_GAP`.
- `both`: paired Sentinel-1 + Sentinel-2 fusion, extracting `joint_GAP`.

The Colab notebook is [croma_benge800_sensor_fairness_colab.ipynb](D:/Codex/rsfm-fairness-audit/notebooks/croma_benge800_sensor_fairness_colab.ipynb). It downloads the 183 MB BEN-GE-800 archive from Zenodo, prepares 64 paired samples, runs the three modes, and writes a sensor-mode comparison report.

Phase 2A and Phase 2B must remain separate: lc-col BigEarthNet is S2-only and is not valid for SAR/optical sensor fairness; BEN-GE-800 provides paired S1/S2 samples for the first real sensor-conditioned audit.

## Open Items

- Exact BigEarthNet v2 S2 band conversion to CROMA's 12-channel expectation: to_verify.
- CPU-only feasibility for tiny subset inference: to_verify; Colab GPU remains recommended.
- For 50k/full-scale runs, current chunked extraction avoids OOM during model inference, but probe/metrics may still load merged embeddings into RAM. Future work should add out-of-core probe training and streaming group metric aggregation.
