# Paper-facing supplementary analyses v1

This pipeline implements five post-hoc analyses over frozen predictions and
audit tables. It does not load a model, tune on test data, or alter canonical
outputs. New derivatives are written only below
`outputs/paper_supplementary_analyses_v1/`.

## Scientific scope

1. Sen1Floods11 event descriptors use five predeclared, physically motivated
   quantities. With 11 events, associations are descriptive and accompanied by
   leave-one-event-out ranges. Acquisition season is omitted unless a date is
   verifiable in source metadata.
2. reBEN is multi-label. The pipeline reports paired label-error transitions
   and conditional co-errors, not an exclusive-class confusion matrix.
3. AlphaEarth boundary distance is computed from ESA WorldCover 2021 and is a
   reference-product ambiguity diagnostic, not independent ground truth.
4. The flood mean-versus-tail example is retrospective and does not validate a
   new operational utility policy.
5. Cluster intervals are generated only where every requested slice has at
   least two independent clusters. Three-seed SD remains a separate estimate of
   training repeatability.

The machine-readable contract is
`configs/analysis/paper_supplementary_analyses_v1.json`.

## Colab preparation

Use a high-memory CPU runtime with at least 20 GB of free local disk. A GPU is
unnecessary. Mount Drive, clone or pull
the repository at the frozen code revision, and install only analysis packages:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!test -d rsfm-fairness-audit || git clone https://github.com/kaixinol/rsfm-fairness-audit.git
%cd /content/rsfm-fairness-audit
!git pull --ff-only origin master
!pip -q install -e . pandas pyarrow rasterio earthengine-api
```

## Exact execution order

### 1. Read-only asset preflight

```bash
python scripts/colab/run_paper_supplementary_analyses_v1_colab.py \
  --stage preflight \
  --repo /content/rsfm-fairness-audit \
  --project-root /content/drive/MyDrive/rsfm_fairness_audit
```

The preflight must find the canonical Sen1 19-route event metrics, source
contract, and validation-locked threshold profile on Drive, 446 Sen1 coordinate
rows, all three raw TIFF folders, the exact frozen set of 111 AlphaEarth country
shards, AlphaEarth evaluation predictions, and
three paired seeds for both TerraMind and CROMA. The Colab pipeline does not
depend on local thesis drafts or ignored optimization derivatives.

The two Sen1 sources are read from
`outputs/geobwer_final_v3/sen1_19model_descriptive_v2/event_level_metrics.csv`
and
`outputs/geobwer_final_v3/geobwer_evidence_rebuild_v060/sen1_validation_locked_threshold_v12/validation_locked_threshold_profile.csv`.
Preflight verifies the 19-route inventory, the complete 11-event replicated
panel, and all six validation-locked U-Net rows before analysis.

### 2. CPU analyses

```bash
python scripts/colab/run_paper_supplementary_analyses_v1_colab.py \
  --stage cpu \
  --repo /content/rsfm-fairness-audit \
  --project-root /content/drive/MyDrive/rsfm_fairness_audit \
  --n-boot 2000
```

This copies the Sen1 TIFFs and approximately 12 GB of paired reBEN audit CSVs
to `/content`, scans the latter in chunks, builds the retrospective flood
decision figure, and computes valid cluster intervals. The paired audits are
staged once and reused by the error-transition and interval analyses. Runtime
is dominated by Drive I/O. Do not run directly over mounted TIFFs if the copy
step has not finished.

### 3. Submit Earth Engine exports

```bash
python scripts/colab/run_paper_supplementary_analyses_v1_colab.py \
  --stage start-ee \
  --repo /content/rsfm-fairness-audit \
  --project-root /content/drive/MyDrive/rsfm_fairness_audit \
  --ee-project YOUR_EARTH_ENGINE_PROJECT
```

Authenticate when prompted. This submits one Sen1 DEM summary and one
AlphaEarth boundary-distance export per country. It creates the temporary Drive
folder `rsfm_fairness_audit_paper_supplement_ee_v1`. Wait until all tasks are
complete before continuing. The command is resume-safe: completed CSV exports
and currently active Earth Engine task descriptions are skipped rather than
submitted again.

`finish-ee` requires exactly one completed boundary export for each country in
the frozen 111-country shard inventory. A partial set such as 105/111,
duplicate exports, or unexpected country exports is a hard failure rather than
a partial merge.

### 4. Merge Earth Engine exports and finish associations

```bash
python scripts/colab/run_paper_supplementary_analyses_v1_colab.py \
  --stage finish-ee \
  --repo /content/rsfm-fairness-audit \
  --project-root /content/drive/MyDrive/rsfm_fairness_audit \
  --n-boot 2000
```

`finish-ee` runs the completion check automatically. It can also be repeated
independently without recomputation:

```bash
python scripts/colab/run_paper_supplementary_analyses_v1_colab.py \
  --stage verify \
  --repo /content/rsfm-fairness-audit \
  --project-root /content/drive/MyDrive/rsfm_fairness_audit
```

The run is complete only when `completion_manifest.json` reports
`status=complete` for all six components (the reBEN analysis has separate
TerraMind and CROMA components).

## Expected outputs

- `sen1_decision_scenario/`: 11-event paired table, M/T/D summary and PDF/SVG/PNG decision figure.
- `reben_shift_error/`: label, country-by-label and conditional co-error tables plus one PDF/SVG/PNG figure per model.
- `sen1_event_descriptors/`: chip/event descriptors, DEM merge, n=11 associations and a five-panel PDF/SVG/PNG figure.
- `alphaearth_boundary_distance/`: sampled distances, binned risk, partial association and a two-panel PDF/SVG/PNG diagnostic.
- `cluster_mtd_intervals/`: headline M/T/D interval table and forest figure plus explicit unavailable cases.
- `preflight.json`, `bootstrap_input_specs.json`, and manifests for provenance.

## Expected runtime

- `preflight`: usually below 2 minutes.
- `cpu`: approximately 1--3 hours on a high-memory Colab CPU runtime; Drive transfer of the paired reBEN tables is the main uncertainty.
- `start-ee`: normally below 10 minutes to submit jobs; the Earth Engine queue then runs asynchronously, commonly for 1--6 hours but potentially longer under quota pressure.
- `finish-ee`: approximately 15--45 minutes, dominated by the 150k-row AlphaEarth merge and spatial bootstrap.

No stage benefits materially from a GPU. If local scratch space is below 20 GB,
use a fresh Colab runtime before the CPU stage.

## Stop conditions

- A paired reBEN key/order mismatch is a hard failure.
- Fewer than 100 completed AlphaEarth boundary exports is a hard failure.
- A boundary-distance merge that does not retain the complete frozen
  AlphaEarth evaluation support is a hard failure.
- Unverified acquisition dates remain missing; they are never imputed.
- Missing cluster lineage or fewer than two independent clusters per slice
  produces an explicit unavailable result, not a fabricated confidence interval.
