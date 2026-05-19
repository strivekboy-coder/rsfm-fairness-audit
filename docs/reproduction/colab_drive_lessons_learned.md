# Colab And Google Drive Lessons Learned

This note records operational lessons from the Sen1Floods11 native segmentation Colab runs. It is about reproducibility and artifact management, not scientific interpretation.

## Stable Storage Layout

Use one Google Drive project root:

```text
/content/drive/MyDrive/rsfm_fairness_audit/
  cache/
    sen1floods11/
      raw/
      sen1floods11_hand_labeled_manifest.csv
  prepared_zips/
  outputs/
```

Keep raw downloaded GeoTIFFs in `cache/<dataset>/raw/`. These are shared across model protocols and prepared resolutions, so do not delete them unless Drive space is genuinely constrained.

Keep prepared datasets as zip files in `prepared_zips/`. Prepared zips are model/profile/resolution specific and should be named explicitly, for example:

```text
prepared_zips/sen1floods11_prithvi_tl_official_full_512.zip
```

Keep run evidence zips in `outputs/`, named by model, dataset, scope, and resolution:

```text
outputs/prithvi_tl_sen1floods11_official_full_512.zip
```

Legacy roots such as `rsfm_fairness_audit_cache/` should not be used for current runs. If old artifacts are worth keeping, move them under a separate `rsfm_fairness_audit_legacy/` folder and do not mix them into the current workflow.

## What To Check Before A Full Run

Raw Sen1Floods11 cache should contain:

```text
892 total GeoTIFFs
446 *_S2Hand.tif
446 *_LabelHand.tif
```

Prepared full TL 512 data should contain:

```text
metadata.csv with 447 lines
446 chips/*.npz
446 masks/*.npz
```

Full output should contain:

```text
segmentation_metrics.csv with 447 lines
event_segmentation_metrics.csv with 12 lines
bwer_summary.csv
warnings.json
report.md
figures/
```

If an output directory exists but these CSV files are missing, it is a partial or failed run, not a valid result.

## Common Pitfalls

Colab `/content` is temporary. A prepared dataset can disappear after runtime reset even if raw data remains cached in Drive.

The Drive mount can show files inside the current notebook before they appear in the web UI. If the web UI does not show a freshly written zip, verify from Colab with `ls -lh` and `du -h`. If Drive becomes disconnected, download the zip from `/content` manually and upload it through the Google Drive web UI.

The preparation script writes prepared NPZ files to `/content/data`. It does not automatically create a Drive zip unless the notebook or user explicitly zips the prepared folder.

The CLI writes outputs to `/content/outputs`. It does not automatically persist them to Drive unless the caller zips the output folder.

Do not delete raw caches to solve a prepared-output problem. Delete stale prepared zips or bad output zips first.

## Recommended Sequence

1. Confirm raw cache exists and has 892 files.
2. Prepare `/content/data/<prepared_name>`.
3. Verify `metadata.csv`, chip count, and mask count.
4. Zip prepared data to Drive.
5. Run model/audit from `/content/data/<prepared_name>`.
6. Verify output CSV row counts.
7. Zip output to Drive.
8. Only after successful zips, optionally clean `/content`.

## Naming Used For The Final Sen1Floods11 TL Run

Prepared data:

```text
/content/data/sen1floods11_tl_official_full_512
/content/drive/MyDrive/rsfm_fairness_audit/prepared_zips/sen1floods11_prithvi_tl_official_full_512.zip
```

Output:

```text
/content/outputs/prithvi_tl_sen1floods11_official_full_512
/content/drive/MyDrive/rsfm_fairness_audit/outputs/prithvi_tl_sen1floods11_official_full_512.zip
```
