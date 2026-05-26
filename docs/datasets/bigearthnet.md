# BigEarthNet v2.0 Dataset Note

## Official Sources

- Homepage: https://bigearth.net/
- Description PDF: https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf
- Zenodo record: https://zenodo.org/records/10891137

## Verified Facts

BigEarthNet v2.0 contains 549,488 paired Sentinel-1 and Sentinel-2 image patches. The official description states that Sentinel-2 tiles were selected over 10 European countries and that Sentinel-1 patches were prepared for the same patch set. The dataset is licensed under CDLA-Permissive-1.0.

BigEarthNet v2.0 includes patch-level multi-label land-cover labels and pixel-level reference maps, making it useful for both scene and pixel-level tasks. The official metadata parquet includes `patch_id`, `labels`, `split`, `country`, `s1_name`, and cloud/snow fields such as `contains_seasonal_snow` and `contains_cloud_or_shadow`.

For the Step 1 reBEN/CROMA sensor-mode audit, the 19 class names follow the BigEarthNet v2.0 description PDF Table 1 19-class nomenclature. The runner treats this as a multi-label classification task and expands predictions to one row per sample x class before BWER.

## Fairness Use

Step 1 uses BigEarthNet v2.0 / reBEN for sensor-mode audit. Sensor mode is an experimental cross-run condition: S1-only, S2-only, and S1+S2. It is not a per-sample metadata slice.

Within each completed run, BWER slices are class, country, country | class, country x class diagnostic, and cloud/snow sensitivity when support permits. The primary multi-label risk primitive is label-wise BCE risk; thresholded label-wise binary error is secondary.

The Colab runner writes both primitives for each completed run: `risk_bce` as the primary BWER result and `risk_binary_error` as a secondary diagnostic. Selective risk uses probability confidence and is tied to the primary BCE prediction table.

## Current Implementation

The Colab workflow is:

1. Prepare or verify official resources:

```bash
python scripts/colab/prepare_reben_croma_sensor_audit_colab.py \
  --reben-root /content/data/reben \
  --lmdb-root /content/data/reben/BigEarthNetEncoded.lmdb \
  --metadata-parquet /content/data/reben/metadata.parquet \
  --metadata-snow-cloud-parquet /content/data/reben/metadata_for_patches_with_snow_cloud_or_shadow.parquet \
  --croma-repo /content/CROMA \
  --croma-checkpoint /content/checkpoints/CROMA_base.pt \
  --output-dir /content/outputs/reben_croma_sensor_mode_audit_prepare
```

This preparation script downloads/verifies the official CROMA repo/checkpoint and the official BigEarthNet v2 Zenodo metadata parquet files. It does not use the unofficial community LMDB mirror. If the reBEN LMDB is missing, it writes `blocked_report.md` with manual placement instructions.

2. Run the smoke or full audit runner:

```bash
python scripts/colab/run_reben_croma_sensor_mode_audit_colab.py \
  --lmdb-root /content/data/reben/BigEarthNetEncoded.lmdb \
  --metadata-parquet /content/data/reben/metadata.parquet \
  --metadata-snow-cloud-parquet /content/data/reben/metadata_for_patches_with_snow_cloud_or_shadow.parquet \
  --croma-checkpoint /content/checkpoints/CROMA_base.pt \
  --croma-repo /content/CROMA \
  --output-dir /content/outputs/reben_croma_sensor_mode_audit \
  --batch-size 64 \
  --run-croma \
  --run-bifold \
  --probe-epochs 100 \
  --package
```

The runner uses ConfigILM/reBEN for official LMDB/parquet loading. It refuses to silently substitute BigEarthNet v1, BEN-GE pilots, torchvision ResNet101, or single-label BWER.

## Open Items

- Real Colab execution with official LMDB/parquet, CROMA checkpoint/repo, ConfigILM, and official `reben_publication` code is still required.
- Final six-row result zip is not a local artifact until the Colab run completes and `reben_contract_validation.md` reports no missing artifacts.
