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

### 2026-05-27 CROMA smoke, full runs, and CUDA note

The CROMA-only smoke run has completed for `croma_s1`, `croma_s2`, and
`croma_s1_plus_s2` on a 256-sample cap. It produced the expected CROMA
per-mode outputs and package under the Colab smoke output directory.

The CROMA-only full sensor-mode audit is also completed for:

- `croma_s1`: S1-only.
- `croma_s2`: S2-only.
- `croma_s1_plus_s2`: S1+S2 fusion.

Completed Colab output directories:

```text
/content/outputs/reben_croma_sensor_mode_audit_croma_s1_full
/content/outputs/reben_croma_sensor_mode_audit_croma_s2_full
/content/outputs/reben_croma_sensor_mode_audit_croma_s1_plus_s2_full
/content/outputs/reben_croma_sensor_mode_audit_croma_comparison
```

Drive archive target:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/reben_croma_sensor_mode_audit/
```

The official BIFOLD ResNet101 path remains blocked until the public/authorized
`reben_publication.BigEarthNetv2_0_ImageClassifier` source is available. The
BIFOLD Hugging Face model repositories expose weights/configs but not the custom
model code required by the official model cards, and this project must not
replace that path with torchvision ResNet101.

The current LMDB source used for the completed CROMA full runs is
`hackelle/BigEarthNetV2-LMDB` from Hugging Face. It is an unofficial
preconverted safetensors-style LMDB, not the official ConfigILM pickle-LMDB.
The repo therefore uses a direct LMDB + safetensors loader for this source. This
is recorded as protocol-risk relative to an official ConfigILM-compatible LMDB
or an official raw-data-to-LMDB reproduction.

A CROMA GPU-device bug was found during the full-run attempt: the Colab runtime
had an A100 and `torch.cuda.is_available()` was true, but the subprocess still
used 0 MB GPU RAM because the runner did not pass `--device` into
`CROMAAdapter`. This is now fixed. The runner supports
`--device auto|cuda|cpu`, passes it into the adapter, and the adapter logs the
resolved device, GPU name, model parameter device, and input tensor devices at
the first forward pass. A healthy CUDA run should print a line like:

```text
[info] CROMA device: requested=auto resolved=cuda gpu=NVIDIA A100... model_parameter_device=cuda:0 input_tensor_devices={'SAR_images': 'cuda:0'}
```

If this line reports `resolved=cpu` or GPU memory remains at 0 MB during CROMA
embedding extraction, stop and inspect the device handoff before running the
full audit.

The Colab workflow is:

Before preparation, install the ConfigILM/reBEN dependency chain without reinstalling torch/CUDA:

```bash
pip install -U --no-deps appdirs configilm bigearthnet_patch_interface bigearthnet_common
pip install --force-reinstall 'fastcore==1.5.29'
```

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

The ConfigILM loader class used by the current Colab stack is `configilm.extra.DataSets.BEN2_DataSet.BEN2DataSet`. The adapter keeps fallback aliases, but reports the exact class used in `dataset_preflight.json` and per-run `run_metadata_*.json`. `bigearthnet_common` 2.8.x expects `fastcore.dispatch`, so pin `fastcore==1.5.29` if a newer fastcore removes that API. Do not reinstall torch/CUDA for this compatibility fix.

2. Run the smoke or full audit runner.

For long full-data CROMA runs, prefer one sensor mode per Colab cell so a later
mode failure does not waste the earlier mode's wall time. Example S1-only full
run:

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
  --croma-mode S1 \
  --device auto \
  --probe-epochs 100 \
  --package
```

Repeat with `--croma-mode S2` and `--croma-mode "S1+S2"` for the other two
CROMA rows. Omitting `--croma-mode` runs all three modes sequentially.

The runner uses ConfigILM/reBEN-compatible loading or the repo LMDB+safetensors
adapter when the LMDB payload is safetensors rather than ConfigILM pickle
payloads. It refuses to silently substitute BigEarthNet v1, BEN-GE pilots,
torchvision ResNet101, or single-label BWER.

## Open Items

- The final six-row CROMA+BIFOLD package cannot be completed until the official
  `reben_publication` source path is available. Until then, CROMA-only outputs
  are valid partial Step 1 evidence and BIFOLD is a documented blocked official
  reference path.
