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
