# BigEarthNet Scaling Strategy

This project uses BigEarthNet v2.0 as the first real dataset for DOFA fairness
auditing. The goal is not to jump straight to a leaderboard-sized run; it is to
increase data scale only after each layer of the pipeline is reproducible.

## Smoke Subset

Recommended size: 32 samples.

Purpose:

- prove the Colab environment can import the project
- prove DOFA config, checkpoint, and device settings are valid
- verify the prepared `metadata.csv` is compatible with `BigEarthNetDatasetAdapter`
- generate all required artifacts once

Use `--stratify-by region_class` when possible, but do not overinterpret
fairness metrics at this size.

## Sanity Subset

Recommended sizes: 500 and 1000 samples.

Purpose:

- check runtime and memory
- inspect whether region/class balancing is actually working
- verify that sensor and region groups have enough support for meaningful tables
- catch path, band-order, and preprocessing problems before larger runs

This is the right stage for Colab T4/L4 experiments.

## Paper-Scale Pilot

Recommended sizes: 5000 and 10000 samples.

Purpose:

- produce early paper figures with nontrivial group sizes
- compare raw vs balanced region gaps
- inspect worst-region stability across random seeds
- decide whether country, tile, eco-region, or broader geographic slices are usable

Run multiple seeds if storage and compute permit. Store manifests alongside
outputs so every figure can be reproduced.

## Large-Scale Or Full-Scale Future Runs

Full BigEarthNet v2.0 experiments should be treated as a later milestone.
External compute is recommended when:

- subset size exceeds 10000 samples
- both S1 and S2 chips are materialized
- multiple DOFA variants or seeds are evaluated
- GeoTIFF conversion dominates runtime
- Colab storage is not enough for prepared chips plus outputs

Use managed storage or a persistent VM. Keep the subset manifest, model config,
commit SHA, and checkpoint SHA/path with every run.

## Fairness Slice Readiness

Only use geographic slices that are verified from official metadata or a
documented geospatial join. If country/region/coordinates are unavailable, use
`to_verify` and skip map-based claims.

Minimum useful group sizes should be decided before paper figures. A practical
starting rule is to report any group with fewer than 20 samples as low-support
and exclude it from headline fairness conclusions until a larger subset is run.
