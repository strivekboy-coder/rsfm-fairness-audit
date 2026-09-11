"""Colab entrypoint for five frozen-output, paper-facing supplementary analyses.

No model is loaded or trained. Run ``preflight`` first, then ``cpu``, then
``start-ee``. After Earth Engine exports finish, run ``finish-ee``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def require(path: Path, label: str, *, directory: bool = False) -> Path:
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise FileNotFoundError(f"Missing {label}: {path}")
    print(f"[asset] {label}: {path}", flush=True)
    return path


def run(command: list[str], repo: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, env=env, check=True)


def manifest_status(path: Path, accepted: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        status = str(json.loads(path.read_text(encoding="utf-8")).get("status", ""))
    except (OSError, ValueError, TypeError):
        return False
    return status in accepted


def copy_tree_progress(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No TIFFs in {source}")
    for index, path in enumerate(files, 1):
        target = destination / path.name
        if not target.is_file() or target.stat().st_size != path.stat().st_size:
            shutil.copy2(path, target)
        if index % 25 == 0 or index == len(files):
            print(f"[copy] {source.name}: {index}/{len(files)}", flush=True)


def find_paired_roots(project: Path) -> dict[str, Path]:
    out = project / "outputs"
    candidates = {
        "TerraMind": [out / "geobwer_final_v3/reben_terramind_paired_shift_v1"],
        "CROMA": [out / "reben_croma_paired_shift_v1", out / "geobwer_final_v3/reben_croma_paired_shift_v1"],
    }
    found = {}
    for model, values in candidates.items():
        root = next((value for value in values if (value / "seed_42/id_label_audit.csv").is_file()), None)
        if root is None:
            raise FileNotFoundError(f"No canonical paired root for {model}: {values}")
        found[model] = root
    return found


def expected_alpha_country_shards(project: Path) -> dict[str, Path]:
    """Resolve the frozen 111-country shard inventory without guessing."""
    root = project / "outputs/alphaearth_gee_full_v2_150k"
    prefix = "alphaearth_worldcover_full_2021_"
    suffix = "_shard.csv"
    by_country: dict[str, Path] = {}
    for path in sorted(root.glob(f"{prefix}*{suffix}")):
        country = path.name[len(prefix):-len(suffix)]
        if not country or country in by_country:
            raise RuntimeError(f"Duplicate or invalid AlphaEarth country shard: {path}")
        by_country[country] = path
    if len(by_country) != 111:
        raise RuntimeError(
            f"Expected the frozen set of 111 AlphaEarth country shards below {root}; "
            f"found {len(by_country)}."
        )
    return by_country


def resolve_alpha_boundary_exports(root: Path, expected_countries: set[str]) -> list[Path]:
    """Require exactly one completed boundary export for every frozen country."""
    by_country: dict[str, list[Path]] = defaultdict(list)
    for country in sorted(expected_countries):
        by_country[country] = sorted(root.glob(f"alpha_boundary_{country}_v1*.csv"))
    missing = sorted(country for country, paths in by_country.items() if not paths)
    duplicates = {
        country: [str(path) for path in paths]
        for country, paths in by_country.items()
        if len(paths) > 1
    }
    all_candidates = sorted(root.glob("alpha_boundary_*_v1*.csv"))
    claimed = {path.resolve() for paths in by_country.values() for path in paths}
    unexpected = [str(path) for path in all_candidates if path.resolve() not in claimed]
    if missing or duplicates or unexpected:
        raise RuntimeError(
            "AlphaEarth boundary export set is not an exact one-per-country match: "
            f"expected={len(expected_countries)}, "
            f"completed={sum(bool(paths) for paths in by_country.values())}, "
            f"missing={missing}, duplicates={duplicates}, unexpected={unexpected}."
        )
    return [by_country[country][0] for country in sorted(expected_countries)]


def preflight(repo: Path, project: Path) -> dict[str, object]:
    output = project / "outputs"
    sen_cache = project / "cache/sen1floods11"
    sen_results = output / "geobwer_final_v3/sen1_19model_descriptive_v2"
    sen_locked = output / "geobwer_final_v3/geobwer_evidence_rebuild_v060/sen1_validation_locked_threshold_v12"
    assets = {
        "sen_event_metrics": require(sen_results / "event_level_metrics.csv", "canonical Sen1 event metrics"),
        "sen_source_contract": require(sen_results / "source_contract.json", "canonical Sen1 source contract"),
        "sen_locked_threshold_profile": require(sen_locked / "validation_locked_threshold_profile.csv", "canonical Sen1 validation-locked threshold profile"),
        "sen_metadata": require(sen_cache / "sen1_geospatial_metadata_446_v0426.csv", "Sen1 coordinates"),
        "sen_s1": require(sen_cache / "S1GRDHand", "Sen1 S1 GeoTIFFs", directory=True),
        "sen_s2": require(sen_cache / "S2L1CHand", "Sen1 S2 GeoTIFFs", directory=True),
        "sen_label": require(sen_cache / "LabelHand", "Sen1 label GeoTIFFs", directory=True),
        "alpha_predictions": require(output / "alphaearth_landcover_audit_full_v2_150k/alphaearth_full_eval_predictions.csv", "AlphaEarth eval predictions"),
        "alpha_shards": require(output / "alphaearth_gee_full_v2_150k", "AlphaEarth GEE shards", directory=True),
    }
    paired = find_paired_roots(project)
    for model, root in paired.items():
        for seed in (42, 73, 101):
            require(root / f"seed_{seed}/id_label_audit.csv", f"{model} seed {seed} ID label audit")
            require(root / f"seed_{seed}/ood_label_audit.csv", f"{model} seed {seed} shifted label audit")
    with Path(assets["sen_metadata"]).open("r", encoding="utf-8-sig", newline="") as handle:
        sen_metadata_rows = sum(1 for _ in csv.DictReader(handle))
    with Path(assets["sen_event_metrics"]).open("r", encoding="utf-8-sig", newline="") as handle:
        sen_event_rows = list(csv.DictReader(handle))
    sen_models = {row.get("model", "") for row in sen_event_rows}
    sen_events = {
        row.get("event_id", "") for row in sen_event_rows
        if row.get("split") == "combined_held_out"
        and row.get("comparison_role") == "same_grid_primary_panel"
    }
    sen_panel_rows = [
        row for row in sen_event_rows
        if row.get("split") == "combined_held_out"
        and row.get("comparison_role") == "same_grid_primary_panel"
        and row.get("family") in {"supervised_resnet34_unet", "terramind_v1_base"}
    ]
    if len(sen_models) != 19 or len(sen_events) != 11 or len(sen_panel_rows) != 198:
        raise RuntimeError(
            "Canonical Sen1 event panel is incomplete: "
            f"models={len(sen_models)}, events={len(sen_events)}, replicated_panel_rows={len(sen_panel_rows)}"
        )
    with Path(assets["sen_locked_threshold_profile"]).open("r", encoding="utf-8-sig", newline="") as handle:
        locked_rows = list(csv.DictReader(handle))
    locked_primary = [
        row for row in locked_rows
        if row.get("family") == "supervised_resnet34_unet"
        and row.get("mode") in {"s1_plus_s2", "s2"}
        and row.get("split") == "combined_held_out"
        and row.get("is_validation_selected_operating_point", "").lower() == "true"
    ]
    if len(locked_primary) != 6 or {row.get("seed") for row in locked_primary} != {"42", "73", "101"}:
        raise RuntimeError(
            "Canonical Sen1 validation-locked M/T/D panel is incomplete: "
            f"rows={len(locked_primary)}, seeds={sorted({row.get('seed') for row in locked_primary})}"
        )
    sen_tiff_counts = {
        key: len(list(Path(assets[key]).glob("*.tif")))
        for key in ("sen_s1", "sen_s2", "sen_label")
    }
    if sen_metadata_rows != 446:
        raise RuntimeError(f"Expected 446 Sen1 metadata rows; found {sen_metadata_rows}")
    if any(value != 446 for value in sen_tiff_counts.values()):
        raise RuntimeError(f"Expected 446 TIFFs in each Sen1 source folder; found {sen_tiff_counts}")
    alpha_shard_count = len(expected_alpha_country_shards(project))
    return {
        "status": "pass",
        "sen_metadata_rows": sen_metadata_rows,
        "sen_event_model_count": len(sen_models),
        "sen_event_count": len(sen_events),
        "sen_replicated_panel_rows": len(sen_panel_rows),
        "sen_locked_primary_rows": len(locked_primary),
        "sen_tiff_counts": sen_tiff_counts,
        "alpha_shard_count": alpha_shard_count,
        "paired_roots": {k: str(v) for k, v in paired.items()},
    }


def stage_paired_audits(project: Path, local_root: Path) -> dict[str, Path]:
    """Copy large paired audits once so repeated analyses do not stream from Drive."""
    staged: dict[str, Path] = {}
    for model, source_root in find_paired_roots(project).items():
        destination_root = local_root / model.lower()
        for seed in (42, 73, 101):
            destination = destination_root / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("id_label_audit.csv", "ood_label_audit.csv"):
                source = source_root / f"seed_{seed}" / name
                target = destination / name
                print(f"[stage paired] {model} seed {seed} {name} ({source.stat().st_size / 1e9:.2f} GB)", flush=True)
                if not target.is_file() or target.stat().st_size != source.stat().st_size:
                    shutil.copy2(source, target)
                print(f"[stage paired complete] {target}", flush=True)
        staged[model] = destination_root
    return staged


def build_bootstrap_specs(project: Path, output_dir: Path, paired_roots: dict[str, Path] | None = None) -> Path:
    output = project / "outputs"
    specs = []
    roots = [
        ("fmow", "DOFAv2", output / "geobwer_final_v3/fmow_dofav2_geo_clean_v1"),
        ("fmow", "ResNet50", output / "geobwer_final_v3/fmow_resnet50_common9_v1"),
        ("fmow", "TerraMind", output / "experiment9_terramind_fmow_v1"),
    ]
    for task, model, root in roots:
        candidates = [path for path in root.rglob("formal_audit_table.csv") if "uncertainty_extensions" not in path.parts]
        canonical = [path for path in candidates if path.parent.name == "formal_outputs"]
        for path in sorted(canonical):
            seed = next((part.replace("seed_", "") for part in path.parts if part.startswith("seed_")), "unspecified")
            seed_dir = path.parent.parent
            summary_candidates = [seed_dir / "geobwer/geobwer_summary.csv", seed_dir / "geobwer_raw/geobwer_summary.csv"]
            summary = next((candidate for candidate in summary_candidates if candidate.is_file()), None)
            specs.append({
                "task": task, "model": model, "condition": "primary", "seed": seed,
                "axis": "country", "balance_col": "class_label",
                "evidence_scope": "headline_class_standardised_country", "path": str(path),
                "canonical_summary_path": str(summary) if summary else "", "canonical_axis": "country",
            })
    for model, root in (paired_roots or find_paired_roots(project)).items():
        for seed in (42, 73, 101):
            for condition, filename in (("S2_ID", "id_label_audit.csv"), ("S1_shift", "ood_label_audit.csv")):
                specs.append({"task": "reben", "model": model, "condition": condition, "seed": seed, "axis": "country", "evidence_scope": "paired_country_hamming", "path": str(root / f"seed_{seed}" / filename)})
    path = output_dir / "bootstrap_input_specs.json"
    path.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    return path


def cpu_stage(repo: Path, project: Path, out: Path, n_boot: int) -> None:
    status = preflight(repo, project)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    decision_manifest = out / "sen1_decision_scenario/manifest.json"
    if manifest_status(decision_manifest, {"complete"}):
        print(f"[resume] decision scenario already complete: {decision_manifest}", flush=True)
    else:
        event_metrics = project / "outputs/geobwer_final_v3/sen1_19model_descriptive_v2/event_level_metrics.csv"
        threshold_profile = project / "outputs/geobwer_final_v3/geobwer_evidence_rebuild_v060/sen1_validation_locked_threshold_v12/validation_locked_threshold_profile.csv"
        run([sys.executable, str(repo / "scripts/analysis/build_sen1_mean_tail_decision_scenario.py"), "--repo", str(repo), "--event-metrics", str(event_metrics), "--threshold-profile", str(threshold_profile), "--output-dir", str(out / "sen1_decision_scenario")], repo)

    interval_manifest = out / "cluster_mtd_intervals/manifest.json"
    reben_complete = {
        model: manifest_status(out / "reben_shift_error" / model.lower() / "manifest.json", {"complete"})
        for model in find_paired_roots(project)
    }
    paired_needed = not all(reben_complete.values()) or not manifest_status(interval_manifest, {"complete_with_explicit_unavailable_cases"})
    paired_local = stage_paired_audits(project, Path("/content/geobwer_paper_supplement_inputs/reben_paired")) if paired_needed else {}
    for model, complete in reben_complete.items():
        if complete:
            print(f"[resume] reBEN shift-error analysis already complete: {model}", flush=True)
            continue
        run([sys.executable, str(repo / "scripts/analysis/build_reben_shift_error_structure.py"), "--repo", str(repo), "--paired-root", str(paired_local[model]), "--model", model, "--output-dir", str(out / "reben_shift_error" / model.lower())], repo)

    descriptor_manifest = out / "sen1_event_descriptors/descriptor_manifest.json"
    final_descriptor_manifest = out / "sen1_event_descriptors/manifest.json"
    descriptor_ready = manifest_status(descriptor_manifest, {"awaiting_dem_merge"}) or manifest_status(final_descriptor_manifest, {"complete_descriptive"})
    if descriptor_ready:
        print(f"[resume] Sen1 chip descriptors already complete: {descriptor_manifest}", flush=True)
    else:
        local = Path("/content/geobwer_paper_supplement_inputs/sen1floods11")
        sen_cache = project / "cache/sen1floods11"
        for name in ("S1GRDHand", "S2L1CHand", "LabelHand"):
            copy_tree_progress(sen_cache / name, local / name)
        shutil.copy2(sen_cache / "sen1_geospatial_metadata_446_v0426.csv", local / "metadata.csv")
        run([sys.executable, str(repo / "scripts/analysis/build_sen1_event_descriptors.py"), "--repo", str(repo), "--metadata", str(local / "metadata.csv"), "--s1-dir", str(local / "S1GRDHand"), "--s2-dir", str(local / "S2L1CHand"), "--label-dir", str(local / "LabelHand"), "--output-dir", str(out / "sen1_event_descriptors")], repo)

    if manifest_status(interval_manifest, {"complete_with_explicit_unavailable_cases"}):
        print(f"[resume] cluster M/T/D interval analysis already complete: {interval_manifest}", flush=True)
    else:
        spec = build_bootstrap_specs(project, out, paired_local)
        run([sys.executable, str(repo / "scripts/analysis/build_cluster_mtd_intervals.py"), "--repo", str(repo), "--spec-json", str(spec), "--output-dir", str(out / "cluster_mtd_intervals"), "--n-boot", str(n_boot)], repo)
    # Sen1 has one independent event cluster per event slice; this is recorded as
    # non-estimable for a within-slice cluster CI instead of forcing a number.
    with (out / "cluster_mtd_intervals/non_estimable_interval_scope.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "status", "reason", "allowed_summary"])
        writer.writeheader()
        writer.writerow({"task": "Sen1Floods11", "status": "insufficient_independent_clusters_within_event", "reason": "each deployment event is itself the independent cluster", "allowed_summary": "descriptive event-resampling sensitivity, not a within-event cluster CI"})
        writer.writerow({"task": "AlphaEarth-WorldCover", "status": "spatial_inference_gate_failed", "reason": "validation-only spatial scale certification failed in the frozen protocol", "allowed_summary": "point estimates and explicitly descriptive spatial diagnostics; no new formal headline CI"})
    (out / "cpu_stage_status.json").write_text(json.dumps({"status": "complete", "resume_safe": True, "n_boot": n_boot}, indent=2), encoding="utf-8")


def ee_initialize(project_name: str):
    import ee
    try:
        ee.Initialize(project=project_name)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_name)
    return ee


def start_ee(project: Path, out: Path, ee_project: str, export_folder: str) -> None:
    ee = ee_initialize(ee_project)
    tasks = []
    export_root = Path("/content/drive/MyDrive") / export_folder
    export_root.mkdir(parents=True, exist_ok=True)
    active_states = {"READY", "RUNNING"}
    active_descriptions = {
        str(item.config.get("description", ""))
        for item in ee.batch.Task.list()
        if str(item.status().get("state", "")) in active_states
    }
    chip = __import__("pandas").read_csv(out / "sen1_event_descriptors/sen1_chip_descriptors.csv")
    features = []
    for row in chip.itertuples():
        geometry = ee.Geometry.Rectangle([float(row.bbox_west), float(row.bbox_south), float(row.bbox_east), float(row.bbox_north)], proj="EPSG:4326", geodesic=False)
        features.append(ee.Feature(geometry, {"sample_id": str(row.sample_id), "event_id": str(row.event_id)}))
    dem_description = "geobwer_sen1_dem_v1"
    if list(export_root.glob("sen1_dem_descriptors_v1*.csv")):
        tasks.append({"kind": "sen1_dem", "description": dem_description, "status": "skipped_completed_export"})
    elif dem_description in active_descriptions:
        tasks.append({"kind": "sen1_dem", "description": dem_description, "status": "skipped_active_task"})
    else:
        elevation = ee.Image("USGS/SRTMGL1_003").rename("elevation")
        reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True).combine(ee.Reducer.minMax(), sharedInputs=True)
        terrain = elevation.reduceRegions(collection=ee.FeatureCollection(features), reducer=reducer, scale=30, tileScale=4)
        task = ee.batch.Export.table.toDrive(collection=terrain, description=dem_description, folder=export_folder, fileNamePrefix="sen1_dem_descriptors_v1", fileFormat="CSV")
        task.start(); tasks.append({"kind": "sen1_dem", "description": dem_description, "task_id": task.id, "status": "submitted"})

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    kernel = ee.Kernel.square(radius=1, units="pixels", normalize=False)
    edge = wc.neq(wc.focal_min(kernel=kernel)).Or(wc.neq(wc.focal_max(kernel=kernel))).selfMask()
    # The 512-pixel search window is part of the prespecified diagnostic.
    # Cap unmatched interiors at 5.12 km so the export records explicit
    # right-censoring instead of silently returning an implementation sentinel.
    distance = (
        edge.fastDistanceTransform(neighborhood=512, units="pixels", metric="squared_euclidean")
        .sqrt()
        .multiply(10)
        .min(5120)
        .rename("boundary_distance_m")
    )
    import pandas as pd
    for country, shard in expected_alpha_country_shards(project).items():
        description = f"geobwer_alpha_boundary_{country}_v1"
        if list(export_root.glob(f"alpha_boundary_{country}_v1*.csv")):
            tasks.append({"kind": "alpha_boundary", "country": country, "description": description, "status": "skipped_completed_export"})
            continue
        if description in active_descriptions:
            tasks.append({"kind": "alpha_boundary", "country": country, "description": description, "status": "skipped_active_task"})
            continue
        frame = pd.read_csv(shard, usecols=["sample_id", "lon", "lat"])
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {"sample_id": str(row.sample_id)}) for row in frame.itertuples()])
        sampled = distance.sampleRegions(collection=fc, scale=10, geometries=False, tileScale=4)
        task = ee.batch.Export.table.toDrive(collection=sampled, description=description, folder=export_folder, fileNamePrefix=f"alpha_boundary_{country}_v1", fileFormat="CSV")
        task.start(); tasks.append({"kind": "alpha_boundary", "country": country, "description": description, "task_id": task.id, "sample_count": len(frame), "status": "submitted"})
        print(f"[EE queued] {country}: {len(frame)} points", flush=True)
    submitted = sum(item.get("status") == "submitted" for item in tasks)
    (out / "earth_engine_task_manifest.json").write_text(json.dumps({"status": "submitted_or_resumed", "export_folder": export_folder, "submitted_count": submitted, "tasks": tasks}, indent=2), encoding="utf-8")
    print(f"Submitted {submitted} new Earth Engine tasks; retained {len(tasks) - submitted} completed/active tasks. Wait until all complete, then run --stage finish-ee.")


def finish_ee(repo: Path, project: Path, out: Path, export_folder: str, n_boot: int) -> None:
    import pandas as pd
    root = Path("/content/drive/MyDrive") / export_folder
    expected_countries = set(expected_alpha_country_shards(project))
    boundary_files = resolve_alpha_boundary_exports(root, expected_countries)
    boundary = pd.concat([pd.read_csv(path) for path in boundary_files], ignore_index=True)
    boundary_path = out / "alphaearth_boundary_distance/alphaearth_boundary_distance_samples.csv"
    boundary_path.parent.mkdir(parents=True, exist_ok=True); boundary.to_csv(boundary_path, index=False)
    predictions = project / "outputs/alphaearth_landcover_audit_full_v2_150k/alphaearth_full_eval_predictions.csv"
    run([sys.executable, str(repo / "scripts/analysis/build_alphaearth_boundary_distance_analysis.py"), "--repo", str(repo), "--predictions", str(predictions), "--boundary", str(boundary_path), "--output-dir", str(out / "alphaearth_boundary_distance"), "--n-boot", str(min(n_boot, 500))], repo)
    dem_files = sorted(root.glob("sen1_dem_descriptors_v1*.csv"))
    if not dem_files:
        raise RuntimeError("No Sen1 DEM export is available.")
    dem = pd.concat([pd.read_csv(path) for path in dem_files], ignore_index=True)
    if "sample_id" not in dem:
        raise RuntimeError("Sen1 DEM exports lack sample_id.")
    if dem["sample_id"].isna().any() or dem["sample_id"].astype(str).duplicated().any() or len(dem) != 446:
        raise RuntimeError(
            "Expected one complete DEM row for each of 446 Sen1 chips; "
            f"rows={len(dem)}, unique_sample_ids={dem['sample_id'].astype(str).nunique()}."
        )
    dem_path = out / "sen1_event_descriptors/sen1_dem_descriptors_merged.csv"
    dem_path.parent.mkdir(parents=True, exist_ok=True)
    dem.to_csv(dem_path, index=False)
    event_metrics = project / "outputs/geobwer_final_v3/sen1_19model_descriptive_v2/event_level_metrics.csv"
    run([sys.executable, str(repo / "scripts/analysis/finish_sen1_event_descriptor_analysis.py"), "--repo", str(repo), "--chip-descriptors", str(out / "sen1_event_descriptors/sen1_chip_descriptors.csv"), "--dem-export", str(dem_path), "--event-metrics", str(event_metrics), "--output-dir", str(out / "sen1_event_descriptors")], repo)
    (out / "final_stage_status.json").write_text(json.dumps({"status": "complete", "alpha_boundary_rows": len(boundary), "sen1_dem_rows": len(dem)}, indent=2), encoding="utf-8")
    verify_outputs(out)


def verify_outputs(out: Path) -> dict[str, object]:
    contracts = [
        ("sen1_decision_scenario", out / "sen1_decision_scenario/manifest.json", {"complete"}),
        ("reben_terramind_error_structure", out / "reben_shift_error/terramind/manifest.json", {"complete"}),
        ("reben_croma_error_structure", out / "reben_shift_error/croma/manifest.json", {"complete"}),
        ("sen1_event_descriptors", out / "sen1_event_descriptors/manifest.json", {"complete_descriptive"}),
        ("alphaearth_boundary_distance", out / "alphaearth_boundary_distance/manifest.json", {"complete"}),
        ("cluster_mtd_intervals", out / "cluster_mtd_intervals/manifest.json", {"complete_with_explicit_unavailable_cases"}),
    ]
    rows = []
    for name, path, accepted in contracts:
        complete = manifest_status(path, accepted)
        rows.append({"component": name, "manifest": str(path), "complete": complete})
    missing = [row["component"] for row in rows if not row["complete"]]
    payload = {
        "schema": "geobwer.paper_supplementary.completion.v1",
        "status": "complete" if not missing else "incomplete",
        "components": rows,
        "missing_components": missing,
        "frozen_outputs_modified": False,
        "model_rerun": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "completion_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if missing:
        raise RuntimeError(f"Paper supplementary analyses are incomplete: {missing}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "cpu", "start-ee", "finish-ee", "verify"], required=True)
    parser.add_argument("--repo", type=Path, default=Path("/content/rsfm-fairness-audit"))
    parser.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/rsfm_fairness_audit"))
    parser.add_argument("--ee-project", default="rsfm-fairness-audit")
    parser.add_argument("--ee-export-folder", default="rsfm_fairness_audit_paper_supplement_ee_v1")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    repo, project = args.repo.resolve(), args.project_root.resolve()
    out = project / "outputs/paper_supplementary_analyses_v1"
    if args.stage == "preflight":
        print(json.dumps(preflight(repo, project), indent=2))
    elif args.stage == "cpu":
        cpu_stage(repo, project, out, args.n_boot)
    elif args.stage == "start-ee":
        start_ee(project, out, args.ee_project, args.ee_export_folder)
    elif args.stage == "finish-ee":
        finish_ee(repo, project, out, args.ee_export_folder, args.n_boot)
    else:
        verify_outputs(out)


if __name__ == "__main__":
    main()
