# BigEarthNet v2.0 Subset Setup

This project does not download the full BigEarthNet v2.0 archive automatically.
Milestone 3B expects a prepared local subset with a manifest and small chip
files.

## Adapter-Readable Format

Create a directory like this:

```text
prepared_bigearthnet_subset/
  metadata.csv
  chips/
    BEN-000001_s2.npy
    BEN-000002_s2.npy
```

`metadata.csv` should contain:

```csv
sample_id,label,label_vector,label_names,country,region,sensor,split,latitude,longitude,s2_path
BEN-000001,0,"[1,0]","[""forest""]",to_verify,to_verify,S2,train,,,chips/BEN-000001_s2.npy
```

Required columns for the current smoke adapter:

- `sample_id`
- `label` or `label_vector`
- `s2_path` for `sensor_mode=S2`, or `s1_path` for `sensor_mode=S1`

Recommended columns:

- `label_names`
- `country`
- `region`
- `sensor`
- `split`
- `latitude`
- `longitude`

If country, region, latitude, or longitude are not verified from official
metadata, use `to_verify` or leave coordinates blank. Do not infer them from
filenames.

## Preparing From Existing Chips

If you already have `.npy`, `.npz`, or GeoTIFF chips plus a CSV/JSON/JSONL table
with paths and labels:

```powershell
python scripts/prepare_bigearthnet_subset.py `
  --source-root <source_root> `
  --metadata-path <source_metadata.csv> `
  --output-root <prepared_subset_path> `
  --subset-size 32 `
  --sensor-mode S2
```

GeoTIFF conversion requires `rasterio` in a geospatial environment. `.npy` and
`.npz` chips work with the base smoke environment.

## DOFA Band Requirement

The default DOFA config expects the official DOFA Sentinel-2 demo order:

```text
9 channels with wavelengths:
0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19
```

If your BigEarthNet subset uses a different Sentinel-2 band order or channel
count, update `configs/models/dofa.yaml` with a verified `wavelength_list`,
`expected_bands`, and normalization constants before running real inference.

## Known Limits

- Official BigEarthNet v2.0 parquet loading is not implemented in the base
  adapter because the exact fields still need to be verified for this project.
- Full dataset layout discovery is intentionally not guessed.
- Coordinates and country fields are used only when present in the manifest.
