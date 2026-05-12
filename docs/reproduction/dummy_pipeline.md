# Dummy Pipeline Reproduction

The dummy pipeline is synthetic and CPU-only.

```powershell
python -m rsfm_fairness_audit.cli run-dummy --output-dir outputs/dummy_smoke
```

It produces deterministic multi-band samples with artificial imbalance across:

- region
- class
- sensor
- region x class

The current probe is a nearest-centroid classifier over deterministic dummy
embeddings. It is intended for pipeline validation, not scientific claims.
