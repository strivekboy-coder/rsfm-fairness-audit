# Post-hoc BWER-Audit v2

BWER-Audit v2 is a post-hoc analysis layer for completed audit runs. It reads
saved segmentation outputs and writes an enriched `bwer_v2/` directory. It does
not rerun model inference, re-prepare Sen1Floods11 data, or modify prepared
data zips.

The canonical public result artifact is a single fused output zip named:

```text
prithvi_tl_sen1floods11_official_full_512.zip
```

That zip should contain both the original audit outputs and the `bwer_v2/`
folder:

```text
prithvi_tl_sen1floods11_official_full_512/
  segmentation_metrics.csv
  event_segmentation_metrics.csv
  bwer_summary.csv
  ...
  bwer_v2/
    bwer_v2_summary.csv
    event_failure_analysis.csv
    bwer_audit_report.md
    figures/
    ...
```

## CLI

Run this after a successful native segmentation audit:

```bash
python -m rsfm_fairness_audit.cli run-bwer-v2 \
  --input-dir /content/outputs/prithvi_tl_sen1floods11_official_full_512 \
  --output-dir /content/outputs/prithvi_tl_sen1floods11_official_full_512/bwer_v2
```

Expected input files include `event_segmentation_metrics.csv` and, when
available, `segmentation_metrics.csv`, `audit_table.csv`,
`segmentation_audit_table.csv`, `bwer_summary.csv`, `bootstrap_ci.csv`,
`warnings.json`, `model_debug.json`, and `report.md`.

The formal Sen1Floods11 segmentation risk unit is the event-level row computed
from aggregated TP/FP/FN/TN and valid-pixel counts. Chip-level macro IoU remains
an auxiliary diagnostic.

## Outputs

The command writes:

- `bwer_v2_summary.csv`
- `alpha_sensitivity.csv`
- `support_sensitivity.csv`
- `reference_weight_sensitivity.csv`
- `missing_policy_sensitivity.csv`
- `stabilised_bwer.csv`
- `leave_one_slice_out.csv`
- `bootstrap_ci.csv`
- `event_failure_analysis.csv`
- `event_ranking.csv`
- `metric_primitives_report.md`
- `adaptation_protocol_report.md`
- `split_diagnostics_report.md`
- `bwer_audit_report.md`
- `figures/event_risk_ranking.png`
- `figures/alpha_sensitivity.png`

For the current official Prithvi TL Sen1Floods11 run, Standardised-BWER and
missing-policy sensitivity are expected to be `not_applicable` because no
meaningful non-proxy balance variable is present. Invalid balances such as
`event_id|event`, `event_id|event_id`, and `country|country` are not formal
standardisation variables.

## Colab unzip -> BWER v2 -> rezip helper

The helper script assumes the repository is already cloned and dependencies are
installed. It only unzips the completed output archive, runs post-hoc BWER v2,
and writes one canonical final archive.

Default Drive-backed path:

```python
!python scripts/run_bwer_v2_from_colab_zip.py
```

By default, the helper reads the Drive output zip and writes the fused final zip
to `/content/prithvi_tl_sen1floods11_official_full_512.zip`. This avoids Colab
Drive synchronization problems; download it from the Colab file panel or upload
it manually to Drive.

Manual-upload path that avoids Drive synchronization issues:

```python
!python scripts/run_bwer_v2_from_colab_zip.py \
  --no-mount-drive \
  --input-zip /content/prithvi_tl_sen1floods11_official_full_512.zip \
  --output-zip /content/prithvi_tl_sen1floods11_official_full_512.zip
```

The script injects `/content/rsfm-fairness-audit/src` into `PYTHONPATH` for the
subprocess, so it can run from a cloned repo even before package installation.

Equivalent explicit cell:

```python
from pathlib import Path
from google.colab import drive
import os, shutil, subprocess, zipfile

drive.mount("/content/drive")

PROJECT_DIR = Path("/content/rsfm-fairness-audit")
DRIVE_ROOT = Path("/content/drive/MyDrive/rsfm_fairness_audit")
INPUT_ZIP = DRIVE_ROOT / "outputs" / "prithvi_tl_sen1floods11_official_full_512.zip"
CONTENT_OUTPUTS = Path("/content/outputs")
RUN_DIR = CONTENT_OUTPUTS / "prithvi_tl_sen1floods11_official_full_512"
BWER_V2_DIR = RUN_DIR / "bwer_v2"
FINAL_ZIP = Path("/content/prithvi_tl_sen1floods11_official_full_512.zip")

assert INPUT_ZIP.exists(), f"Missing input zip: {INPUT_ZIP}"
CONTENT_OUTPUTS.mkdir(parents=True, exist_ok=True)
if RUN_DIR.exists():
    shutil.rmtree(RUN_DIR)

with zipfile.ZipFile(INPUT_ZIP) as zf:
    zf.extractall(CONTENT_OUTPUTS)

assert RUN_DIR.exists(), f"Expected unzipped run dir: {RUN_DIR}"

os.chdir(PROJECT_DIR)
subprocess.run([
    "python", "-m", "rsfm_fairness_audit.cli", "run-bwer-v2",
    "--input-dir", str(RUN_DIR),
    "--output-dir", str(BWER_V2_DIR),
], check=True)

if FINAL_ZIP.exists():
    FINAL_ZIP.unlink()
with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in RUN_DIR.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(CONTENT_OUTPUTS))

print("Input zip:", INPUT_ZIP)
print("BWER v2 dir:", BWER_V2_DIR)
print("Canonical final zip:", FINAL_ZIP)
```

Exact expected input zip:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/prithvi_tl_sen1floods11_official_full_512.zip
```

Exact canonical final output zip:

```text
/content/prithvi_tl_sen1floods11_official_full_512.zip
```

For long-term Drive storage, place that final zip at:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/prithvi_tl_sen1floods11_official_full_512.zip
```
