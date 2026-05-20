# Sen1Floods11 Closure Core Package

This closure package keeps the case study narrow: Sen1Floods11 native flood
segmentation, event-level BWER-Audit v2, and protocol-aware comparison across
four completed runs. It is not a flood segmentation architecture zoo.

## Inputs

Use the existing prepared 512 Sen1Floods11 zip:

```text
/content/drive/MyDrive/rsfm_fairness_audit/prepared_zips/sen1floods11_prithvi_tl_official_full_512.zip
```

The prepared data zip is read-only. Helpers extract it to `/content/data` and
write result zips under:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/
```

## Spectral Baseline

The spectral baseline uses the same 6-band S2 protocol:

```text
BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2
```

It implements:

- `NDWI = (GREEN - NIR_NARROW) / (GREEN + NIR_NARROW + eps)`
- `MNDWI = (GREEN - SWIR_1) / (GREEN + SWIR_1 + eps)`
- `nir_darkness`, a diagnostic NIR-low rule

Fixed thresholds are primary for full-set evaluation. Validation-selected
thresholds are allowed only when a validation split exists. Thresholds selected
on evaluation labels must be run as `oracle_diagnostic` and excluded from
primary claims.

Smoke:

```bash
python scripts/run_spectral_sen1floods11_colab.py \
  --max-samples 64 \
  --output-dir /content/outputs/spectral_mndwi_sen1floods11_smoke64 \
  --output-zip /content/drive/MyDrive/rsfm_fairness_audit/outputs/spectral_mndwi_sen1floods11_smoke64.zip \
  --force
```

Full fixed-threshold MNDWI diagnostic:

```bash
python scripts/run_spectral_sen1floods11_colab.py \
  --index mndwi \
  --threshold 0.0 \
  --threshold-policy fixed \
  --eval-split all \
  --force
```

Expected output:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/spectral_mndwi_sen1floods11_full_512.zip
```

## S2 ResNet34-U-Net / AlbuNet-Style Baseline

This is a stronger U-Net-family supervised baseline, not a foundation model.

Metadata:

- `model_family = unet`
- `model_variant = s2_resnet34_unet`
- `display_name = S2 ResNet34-U-Net / AlbuNet-style baseline`
- `adaptation_protocol = supervised_baseline`
- `input_mode = s2_6band_image_only`
- default `split_protocol = random_chip_split`
- `resolution = 512`

If `--pretrained-encoder` is used, torchvision ResNet34 ImageNet weights are
loaded and the first convolution is adapted from 3 to 6 channels by copying RGB
weights into the first three channels and using the mean RGB filter for the
additional S2 channels. If this is used, report it in the protocol metadata.

Smoke:

```bash
python scripts/run_unet_sen1floods11_colab.py \
  --architecture s2_resnet34_unet \
  --max-samples 64 \
  --epochs 2 \
  --batch-size 2 \
  --output-dir /content/outputs/s2_resnet34_unet_sen1floods11_smoke64 \
  --output-zip /content/drive/MyDrive/rsfm_fairness_audit/outputs/s2_resnet34_unet_sen1floods11_smoke64.zip \
  --force
```

Full:

```bash
python scripts/run_unet_sen1floods11_colab.py \
  --architecture s2_resnet34_unet \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --early-stopping-patience 10 \
  --split-protocol random_chip_split \
  --eval-split test \
  --output-dir /content/outputs/s2_resnet34_unet_sen1floods11_full_512 \
  --output-zip /content/drive/MyDrive/rsfm_fairness_audit/outputs/s2_resnet34_unet_sen1floods11_full_512.zip \
  --force
```

## Closure Comparison

Once the four single-run output zips exist:

1. `prithvi_tl_sen1floods11_official_full_512.zip`
2. `unet_sen1floods11_full_512.zip`
3. `spectral_mndwi_sen1floods11_full_512.zip`
4. `s2_resnet34_unet_sen1floods11_full_512.zip`

run:

```bash
python scripts/run_sen1floods11_closure_colab.py --force
```

or directly:

```bash
python -m rsfm_fairness_audit.cli compare-runs \
  --dataset sen1floods11 \
  --run prithvi_tl=/content/outputs/prithvi_tl_sen1floods11_official_full_512 \
  --run vanilla_unet=/content/outputs/unet_sen1floods11_full_512 \
  --run spectral_mndwi=/content/outputs/spectral_mndwi_sen1floods11_full_512 \
  --run s2_resnet34_unet=/content/outputs/s2_resnet34_unet_sen1floods11_full_512 \
  --output-dir /content/outputs/comparisons/sen1floods11_closure \
  --closure
```

Expected output:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/comparisons/sen1floods11_closure.zip
```

Closure outputs:

- `closure_comparison_summary.csv`
- `closure_average_vs_bwer.csv`
- `closure_event_level_comparison.csv`
- `closure_tail_event_overlap.csv`
- `closure_report.md`
- comparison figures under `figures/`

The report identifies aggregate ranking, BWER ranking, average-vs-BWER ranking
reversal, persistent tail events, Bolivia/Pakistan persistence, spectral-rule
tail profile, and whether the stronger U-Net-family run reduces BWER.

## LOEO and Selective Risk Notes

LOEO is future work in this repo stage. The intended workflow is to hold out
one disaster event, train on all other events, evaluate the held-out event, and
write the same single-run segmentation and BWER v2 schema. Do not claim
event-held-out generalization from `random_chip_split` runs.

Selective Risk is also future work here. It requires saved probability, logit,
or confidence outputs. Current U-Net runs save chip-level confidence summaries
but not full probability maps; Prithvi TL confidence availability depends on the
completed run; spectral rules are deterministic and uncalibrated. If a run lacks
usable confidence fields, report Selective Risk as unavailable rather than
fabricating fixed-coverage results.
