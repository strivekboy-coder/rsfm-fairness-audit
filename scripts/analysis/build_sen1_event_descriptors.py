"""Extract physically interpretable Sen1Floods11 event descriptors.

Raw S1/S2/label GeoTIFFs are read without modifying them. Acquisition season is
reported only when a verifiable date exists in a file tag or filename.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def locate(directory: Path, sample_id: str, suffix: str) -> Path:
    exact = directory / f"{sample_id}_{suffix}.tif"
    if exact.is_file():
        return exact
    matches = list(directory.glob(f"{sample_id}*{suffix}*.tif"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {suffix} TIFF for {sample_id}; found {len(matches)}")
    return matches[0]


def read_raster(path: Path):
    import rasterio
    with rasterio.open(path) as src:
        array = src.read()
        tags = src.tags()
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs
    return array, tags, nodata, bounds, crs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--s2-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from rsfm_fairness_audit.paper_supplementary import (
        infer_verified_datetime, robust_water_land_contrast, write_json,
    )
    from rasterio.warp import transform_bounds

    metadata = pd.read_csv(args.metadata)
    required = {"sample_id", "event_id", "latitude", "longitude"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata lacks {sorted(required - set(metadata.columns))}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in metadata.iterrows():
        sample_id = str(row.sample_id)
        s1_path = locate(args.s1_dir, sample_id, "S1Hand")
        s2_path = locate(args.s2_dir, sample_id, "S2Hand")
        label_path = locate(args.label_dir, sample_id, "LabelHand")
        label, label_tags, label_nodata, _, _ = read_raster(label_path)
        s1, s1_tags, s1_nodata, bounds, crs = read_raster(s1_path)
        s2, s2_tags, s2_nodata, _, _ = read_raster(s2_path)
        mask = label[0]
        valid = np.isfinite(mask) & np.isin(mask, [0, 1])
        flood = mask == 1
        status = "complete"
        if s1.shape[1:] != mask.shape or s2.shape[1:] != mask.shape:
            status = "shape_mismatch_contrasts_unavailable"
            s1_contrast = s2_contrast = float("nan")
        else:
            s1_valid = valid.copy()
            s2_valid = valid.copy()
            if s1_nodata is not None:
                s1_valid &= np.all(s1 != s1_nodata, axis=0)
            if s2_nodata is not None:
                s2_valid &= np.all(s2 != s2_nodata, axis=0)
            s1_contrast = robust_water_land_contrast(s1, flood, s1_valid)
            s2_contrast = robust_water_land_contrast(s2, flood, s2_valid)
        date, date_status = infer_verified_datetime({**label_tags, **s1_tags, **s2_tags}, s1_path.name)
        if crs:
            west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
        else:
            west = east = float(row.longitude)
            south = north = float(row.latitude)
        records.append({
            "sample_id": sample_id, "event_id": str(row.event_id),
            "latitude": float(row.latitude), "longitude": float(row.longitude),
            "descriptor_status": status,
            "valid_pixel_count": int(np.sum(valid)),
            "reference_flood_fraction": float(np.mean(flood[valid])) if np.any(valid) else np.nan,
            "s1_water_land_robust_contrast": s1_contrast,
            "s2_water_land_robust_contrast": s2_contrast,
            "acquisition_date": date, "acquisition_date_status": date_status,
            "bbox_west": west, "bbox_south": south, "bbox_east": east, "bbox_north": north,
        })
        if (index + 1) % 25 == 0 or index + 1 == len(metadata):
            print(f"[Sen1 descriptors] {index + 1}/{len(metadata)} chips", flush=True)
    chip = pd.DataFrame(records)
    chip.to_csv(args.output_dir / "sen1_chip_descriptors.csv", index=False)
    numeric = ["reference_flood_fraction", "s1_water_land_robust_contrast", "s2_water_land_robust_contrast"]
    event = chip.groupby("event_id", as_index=False)[numeric].agg(["mean", "median", "std", "count"])
    event.columns = ["event_id"] + [f"{left}_{right}" for left, right in event.columns.tolist()[1:]]
    dates = chip.groupby("event_id").acquisition_date.apply(lambda x: sorted({v for v in x if isinstance(v, str) and v})).reset_index(name="verified_dates")
    dates["verified_dates"] = dates.verified_dates.map(lambda x: "|".join(x))
    event = event.merge(dates, on="event_id", how="left")
    event.to_csv(args.output_dir / "sen1_event_descriptors_without_dem.csv", index=False)
    write_json(args.output_dir / "descriptor_manifest.json", {
        "schema": "geobwer.paper_supplement.sen1_descriptors.v1", "status": "awaiting_dem_merge",
        "sample_count": len(chip), "event_count": int(chip.event_id.nunique()),
        "verified_date_count": int((chip.acquisition_date_status == "verified_from_file_metadata_or_name").sum()),
        "season_policy": "unavailable unless verified from file metadata or filename",
        "contrast_definition": "mean absolute water-vs-land median difference divided by per-band IQR",
        "frozen_outputs_modified": False, "model_rerun": False,
    })
    print(json.dumps({"chips": len(chip), "events": int(chip.event_id.nunique()), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
