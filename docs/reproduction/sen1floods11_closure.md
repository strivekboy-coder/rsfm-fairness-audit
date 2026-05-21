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

## Advanced Closure Checks

Protocol-matched post-hoc check:

```bash
python -m rsfm_fairness_audit.cli protocol-match-runs \
  --run prithvi_tl=/content/outputs/prithvi_tl_sen1floods11_official_full_512 \
  --run vanilla_unet=/content/outputs/unet_sen1floods11_full_512 \
  --run spectral_mndwi=/content/outputs/spectral_mndwi_sen1floods11_full_512 \
  --run s2_resnet34_unet=/content/outputs/s2_resnet34_unet_sen1floods11_full_512 \
  --output-dir /content/outputs/comparisons/sen1floods11_protocol_matched
```

This recomputes event metrics, Raw-BWER, and BWER v2 on the exact common chip
intersection when every run has chip-level `segmentation_metrics.csv` with
stable `sample_id`/`chip_id` identifiers. If exact matching is impossible, it
writes a limitation report rather than fabricating a same-split result.

Confirmed interpretation for the completed closure run: the exact chip-level
match succeeded with 89 matched chips, and the average-vs-BWER ranking reversal
remained. This rules out evaluation-subset mismatch as the sole explanation for
the reversal, but it does not remove adaptation-protocol, model-capacity,
thresholding, or training-protocol differences.

Selective Risk availability/post-hoc check:

```bash
python -m rsfm_fairness_audit.cli run-selective-risk \
  --run prithvi_tl=/content/outputs/prithvi_tl_sen1floods11_official_full_512 \
  --run vanilla_unet=/content/outputs/unet_sen1floods11_full_512 \
  --run spectral_mndwi=/content/outputs/spectral_mndwi_sen1floods11_full_512 \
  --run s2_resnet34_unet=/content/outputs/s2_resnet34_unet_sen1floods11_full_512 \
  --output-dir /content/outputs/comparisons/sen1floods11_selective_risk
```

When only chip-level confidence summaries are available, this is a
whole-chip-retention diagnostic, not pixel-level selective segmentation risk.
If no confidence/logit/probability fields are available, it writes
`selective_risk_availability.csv` and a limitation report without fake metrics.

MNDWI and other deterministic spectral baselines should be marked selective-risk
unavailable unless a defensible confidence or score field is explicitly saved.

Colab helper for both post-hoc checks:

```bash
python scripts/run_sen1floods11_advanced_closure_colab.py --force
```

Expected outputs:

```text
/content/drive/MyDrive/rsfm_fairness_audit/outputs/comparisons/sen1floods11_protocol_matched.zip
/content/drive/MyDrive/rsfm_fairness_audit/outputs/comparisons/sen1floods11_selective_risk.zip
```

## LOEO Workflow

LOEO is implemented for supervised baselines only. It holds out one disaster
event, trains on all remaining events, evaluates only on the held-out event,
and preserves `split_protocol=leave_one_event_out`.

Smoke:

```bash
python scripts/run_unet_sen1floods11_loeo_colab.py \
  --architecture vanilla_unet \
  --held-out-event Pakistan \
  --epochs 2 \
  --batch-size 2 \
  --force
```

Full vanilla U-Net LOEO:

```bash
python scripts/run_unet_sen1floods11_loeo_colab.py \
  --architecture vanilla_unet \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --early-stopping-patience 10 \
  --force
```

Full S2 ResNet34-U-Net LOEO:

```bash
python scripts/run_unet_sen1floods11_loeo_colab.py \
  --architecture s2_resnet34_unet \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --early-stopping-patience 10 \
  --aggregate-zip /content/drive/MyDrive/rsfm_fairness_audit/outputs/comparisons/sen1floods11_loeo_s2_resnet34_unet.zip \
  --force
```

The script is resumable: completed held-out-event directories are reused unless
`--force` is passed. Aggregation can be rerun with `--aggregate-only`.

LOEO aggregate outputs:

- `loeo_summary.csv` through BWER v2 summaries under `bwer_v2/`
- `loeo_event_level_metrics.csv`
- `loeo_bwer_summary.csv`
- `loeo_report.md`

Completed vanilla U-Net LOEO confirms that event-level tail risk persists under
unseen-event evaluation, while tail identities differ from the random-chip-split
vanilla U-Net result. Do not describe the current result as LOEO amplifying
tail risk or random-chip-split necessarily underestimating BWER; the completed
vanilla LOEO Raw-BWER is lower than the random-chip-split vanilla Raw-BWER.
Do not claim event-held-out generalization from `random_chip_split` runs.
