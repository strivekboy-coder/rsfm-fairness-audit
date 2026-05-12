# rsfm-fairness-audit

Research-grade fairness auditing framework for Remote Sensing Foundation Models.

The first milestone is intentionally small and CPU-only: a fully runnable dummy
pipeline with synthetic multi-band imagery, severe region/class/sensor
imbalance, deterministic embeddings, balanced sampling, fairness metrics, CSV
outputs, and static figures. Real model and dataset adapters are added through
the same interfaces without changing the evaluation pipeline.

## Smoke Run

```powershell
python -m pip install -e .
python -m rsfm_fairness_audit.cli run-dummy --output-dir outputs/dummy_smoke
python -m pytest
```

Generated artifacts include fairness matrices, raw-vs-balanced gap tables,
sensor heatmaps, average-vs-worst scatter plots, representation shift plots,
and a static Markdown report.

## BigEarthNet + DOFA Runs

### A. Mocked Real Pipeline

Use a prepared BigEarthNet-style subset with `.npy`/`.npz` chips and manifest
metadata. Tests use an injected mock model; the CLI path expects a configured
real model unless `--allow-torch-hub-download` or `--model-config` points to a
usable DOFA setup.

```powershell
python -m rsfm_fairness_audit.cli run-real `
  --dataset bigearthnet `
  --model dofa `
  --data-root <mock_subset_path> `
  --subset-size 32 `
  --output-dir outputs/runs/dofa_bigearthnet_mock
```

### B. Official DOFA + Prepared BigEarthNet Smoke

First prepare a subset manifest as described in
[bigearthnet_subset_setup.md](D:/Codex/rsfm-fairness-audit/docs/datasets/bigearthnet_subset_setup.md).
Then fill `repo_path` and `checkpoint_path` in
[dofa.yaml](D:/Codex/rsfm-fairness-audit/configs/models/dofa.yaml), or set
`allow_torch_hub_download: true` explicitly.

```powershell
python -m rsfm_fairness_audit.cli run-real `
  --dataset bigearthnet `
  --model dofa `
  --data-root <prepared_subset_path> `
  --model-config configs/models/dofa.yaml `
  --subset-size 32 `
  --output-dir outputs/runs/dofa_bigearthnet_real_smoke
```

### C. Medium Sanity Run

```powershell
python -m rsfm_fairness_audit.cli run-real `
  --dataset bigearthnet `
  --model dofa `
  --data-root <prepared_subset_path> `
  --model-config configs/models/dofa.yaml `
  --subset-size 1000 `
  --output-dir outputs/runs/dofa_bigearthnet_real_sanity
```

No large checkpoint or dataset is downloaded automatically. Optional DOFA
runtime dependencies are listed in `requirements-dofa.txt`.

## Running Real DOFA Smoke Test On Colab

For the first real DOFA + BigEarthNet-style subset run, use the Colab-first
guide and notebook:

- [Colab smoke guide](D:/Codex/rsfm-fairness-audit/docs/reproduction/dofa_bigearthnet_colab_smoke.md)
- [Colab notebook template](D:/Codex/rsfm-fairness-audit/notebooks/dofa_bigearthnet_smoke_colab.ipynb)

Start with:

```powershell
python -m rsfm_fairness_audit.cli check-real `
  --model dofa `
  --dataset bigearthnet `
  --model-config configs/models/dofa.yaml `
  --data-root <prepared_subset_path>
```
