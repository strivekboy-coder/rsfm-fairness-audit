from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.geographic_risk_atlas import (
    LATITUDE_FIELDS,
    LONGITUDE_FIELDS,
    UNIT_FIELDS,
    _iter_csv,
    _safe_stem,
    _save_figure,
    _style,
    plot_reben_burden,
    run_visual_qa,
)
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


SCHEMA = "geobwer.geographic_risk_association.preregistered.v1"
DW_TO_WORLDCOVER = {"0": "80", "1": "10", "2": "30", "3": "90", "4": "40", "5": "20", "6": "50", "7": "60", "8": "70"}
VARIABLES = {
    "land_cover_heterogeneity": {"role": "confirmatory", "tasks": {"AlphaEarth"}, "direction": "two_sided"},
    "reference_confidence": {"role": "confirmatory", "tasks": {"AlphaEarth"}, "direction": "two_sided"},
    "reference_disagreement": {"role": "confirmatory", "tasks": {"AlphaEarth"}, "direction": "two_sided"},
    "ghsl_urbanization": {"role": "confirmatory", "tasks": {"AlphaEarth", "fMoW"}, "direction": "two_sided"},
    "population_density": {"role": "exploratory_exposure", "tasks": {"AlphaEarth", "fMoW"}, "direction": "descriptive"},
    "nightlights": {"role": "exploratory_exposure", "tasks": {"AlphaEarth", "fMoW"}, "direction": "descriptive"},
}
EXTERNAL_ALIASES = {
    "land_cover_heterogeneity": ("land_cover_heterogeneity", "worldcover_heterogeneity"),
    "reference_confidence": ("reference_confidence", "dynamic_world_confidence", "dw_confidence"),
    "reference_disagreement": ("reference_disagreement", "worldcover_dynamic_world_disagreement", "map_disagreement"),
    "ghsl_urbanization": ("ghsl_urbanization", "ghsl_degree_urbanisation", "ghsl_smod", "degree_of_urbanisation"),
    "population_density": ("population_density", "population_per_km2", "pop_density"),
    "nightlights": ("nightlights", "nighttime_lights", "viirs_nightlights", "viirs_rad"),
}
REFERENCE_CONFIDENCE_FIELDS = ("dynamic_world_confidence", "reference_confidence", "dw_confidence")
REFERENCE_DISAGREEMENT_FIELDS = ("reference_disagreement", "worldcover_dynamic_world_disagreement", "map_disagreement")
LAND_COVER_FIELDS = ("worldcover_class_name", "worldcover_label")


def _first_field(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    return next((name for name in fields if row.get(name) not in (None, "")), None)


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _correlation(_rank(x), _rank(y))


def _partial_spearman(risk: np.ndarray, exposure: np.ndarray, controls: np.ndarray) -> float:
    y, x = _rank(risk), _rank(exposure)
    design = np.column_stack([np.ones(len(y)), controls])
    y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    return _correlation(y_resid, x_resid)


def _spatial_cell(latitude: float, longitude: float, degrees: float = 15.0) -> str:
    """Return a fixed grid cell; longitude width is fixed within a latitude band.

    The previous implementation derived ``lon_width`` from every observation's
    exact latitude and included that float in the key. Nearby observations then
    became separate clusters. Using the band centre preserves the registered
    high-latitude adjustment while making the partition deterministic.
    """
    latitude = min(max(latitude, -90.0), math.nextafter(90.0, -math.inf))
    longitude = min(max(longitude, -180.0), math.nextafter(180.0, -math.inf))
    lat_bin = math.floor((latitude + 90.0) / degrees)
    lat_centre = -90.0 + (lat_bin + .5) * degrees
    lon_width = min(60.0, degrees / max(.25, math.cos(math.radians(lat_centre))))
    lon_bin = math.floor((longitude + 180.0) / lon_width)
    return f"{lat_bin}:{lon_bin}"


def _maximum_spatial_cell_count(degrees: float = 15.0) -> int:
    latitude_bins = math.ceil(180.0 / degrees)
    total = 0
    for lat_bin in range(latitude_bins):
        lat_centre = -90.0 + (lat_bin + .5) * degrees
        lon_width = min(60.0, degrees / max(.25, math.cos(math.radians(lat_centre))))
        total += math.ceil(360.0 / lon_width)
    return total


def _spatial_cluster_qa(cells: Sequence[str], *, degrees: float = 15.0) -> dict[str, Any]:
    counts = Counter(cells)
    sizes = np.asarray(list(counts.values()), dtype=float)
    maximum = _maximum_spatial_cell_count(degrees)
    cluster_count = len(counts)
    return {
        "status": "pass" if 4 <= cluster_count <= maximum else "fail",
        "definition": "fixed_latitude_band_centre_adjusted_longitude_grid",
        "latitude_band_degrees": degrees,
        "maximum_possible_global_cluster_count": maximum,
        "cluster_count": cluster_count,
        "unit_count": len(cells),
        "mean_units_per_cluster": float(np.mean(sizes)) if sizes.size else math.nan,
        "median_units_per_cluster": float(np.median(sizes)) if sizes.size else math.nan,
        "maximum_units_per_cluster": int(np.max(sizes)) if sizes.size else 0,
        "singleton_cluster_count": int(np.sum(sizes == 1)) if sizes.size else 0,
        "singleton_cluster_fraction": float(np.mean(sizes == 1)) if sizes.size else math.nan,
    }


def _spatial_bootstrap_ci(
    risk: np.ndarray, exposure: np.ndarray, controls: np.ndarray,
    cells: Sequence[str], *, n_boot: int = 500, seed: int = 1907,
) -> tuple[float, float, int]:
    clusters = sorted(set(cells))
    if len(clusters) < 4:
        return math.nan, math.nan, len(clusters)
    members = {cell: np.flatnonzero(np.asarray(cells) == cell) for cell in clusters}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([members[str(cell)] for cell in sampled])
        value = _partial_spearman(risk[indices], exposure[indices], controls[indices])
        if math.isfinite(value):
            estimates.append(value)
    if len(estimates) < max(30, n_boot // 5):
        return math.nan, math.nan, len(clusters)
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975)), len(clusters)


def derive_alphaearth_covariates(sample_csv: str | Path, *, split: str = "test") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive only pre-registered, semantically valid AlphaEarth covariates."""
    grouped: dict[str, dict[str, Any]] = {}
    columns: set[str] = set()
    for index, row in enumerate(_iter_csv(Path(sample_csv))):
        columns.update(row)
        if row.get("split") not in (None, "", split):
            continue
        unit_field = _first_field(row, ("spatial_block_id",))
        lat_field, lon_field = _first_field(row, LATITUDE_FIELDS), _first_field(row, LONGITUDE_FIELDS)
        if not unit_field or not lat_field or not lon_field:
            continue
        lat, lon = _float(row[lat_field]), _float(row[lon_field])
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        unit = str(row[unit_field])
        item = grouped.setdefault(unit, {
            "latitude": [], "longitude_sin": [], "longitude_cos": [],
            "land_cover": Counter(), "reference_confidence": [],
            "reference_disagreement": [], "support": 0,
        })
        item["latitude"].append(lat)
        item["longitude_sin"].append(math.sin(math.radians(lon)))
        item["longitude_cos"].append(math.cos(math.radians(lon)))
        item["support"] += 1
        lc_field = _first_field(row, LAND_COVER_FIELDS)
        if lc_field:
            item["land_cover"][str(row[lc_field])] += 1
        conf_field = _first_field(row, REFERENCE_CONFIDENCE_FIELDS)
        confidence = _float(row.get(conf_field)) if conf_field else math.nan
        if math.isfinite(confidence):
            item["reference_confidence"].append(confidence)
        disagreement_field = _first_field(row, REFERENCE_DISAGREEMENT_FIELDS)
        disagreement = _float(row.get(disagreement_field)) if disagreement_field else math.nan
        if not math.isfinite(disagreement) and row.get("dynamic_world_label") not in (None, "") and row.get("worldcover_label") not in (None, ""):
            dw = str(int(float(row["dynamic_world_label"])))
            wc = str(int(float(row["worldcover_label"])))
            disagreement = float(DW_TO_WORLDCOVER.get(dw, "") != wc)
        if math.isfinite(disagreement):
            item["reference_disagreement"].append(disagreement)
    output = []
    for unit, item in sorted(grouped.items()):
        counts = np.asarray(list(item["land_cover"].values()), dtype=float)
        if counts.size > 0:
            probabilities = counts / counts.sum()
            entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(11.0))
        else:
            entropy = math.nan
        output.append({
            "spatial_unit": unit,
            "latitude": float(np.mean(item["latitude"])),
            "longitude": math.degrees(math.atan2(np.mean(item["longitude_sin"]), np.mean(item["longitude_cos"]))),
            "land_cover_heterogeneity": entropy,
            "reference_confidence": float(np.mean(item["reference_confidence"])) if item["reference_confidence"] else "",
            "reference_disagreement": float(np.mean(item["reference_disagreement"])) if item["reference_disagreement"] else "",
            "covariate_support": item["support"],
        })
    return output, {
        "source": str(sample_csv), "unit_count": len(output), "columns": sorted(columns),
        "land_cover_heterogeneity_definition": "Shannon_entropy_of_WorldCover_classes_within_spatial_block_divided_by_log_11",
        "reference_confidence_definition": "mean_Dynamic_World_top_class_probability_within_spatial_block",
        "reference_disagreement_definition": "mean_WorldCover_vs_Dynamic_World_class_mismatch_within_spatial_block",
        "reference_semantics": "map_product_agreement_or_ambiguity_not_human_ground_truth",
    }


def load_external_covariates(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv_rows(path)
    if not rows:
        return [], {"source": str(path), "status": "empty", "variables": []}
    unit_field = _first_field(rows[0], ("spatial_unit",) + UNIT_FIELDS)
    lat_field, lon_field = _first_field(rows[0], LATITUDE_FIELDS), _first_field(rows[0], LONGITUDE_FIELDS)
    if not unit_field and not (lat_field and lon_field):
        raise ValueError(f"External covariates need spatial_unit or latitude/longitude: {path}")
    variables = {name: _first_field(rows[0], aliases) for name, aliases in EXTERNAL_ALIASES.items()}
    output = []
    for row in rows:
        item: dict[str, Any] = {}
        if unit_field and row.get(unit_field) not in (None, ""):
            item["spatial_unit"] = str(row[unit_field])
        if lat_field and lon_field:
            item["latitude"], item["longitude"] = _float(row[lat_field]), _float(row[lon_field])
        for name, field in variables.items():
            if field:
                value = _float(row.get(field))
                item[name] = value if math.isfinite(value) else ""
        output.append(item)
    return output, {"source": str(path), "status": "ready", "variables": variables, "row_count": len(output)}


def _merge_by_unit_or_coordinate(
    target: list[dict[str, Any]], source: Sequence[Mapping[str, Any]], *, max_distance_km: float = 50.0,
) -> dict[str, Any]:
    by_unit = {str(row["spatial_unit"]): row for row in source if row.get("spatial_unit") not in (None, "")}
    coordinate_rows = [row for row in source if math.isfinite(_float(row.get("latitude"))) and math.isfinite(_float(row.get("longitude")))]
    matched, method, distances = 0, "spatial_unit", []
    source_xyz = None
    if coordinate_rows:
        source_lat = np.radians([float(row["latitude"]) for row in coordinate_rows])
        source_lon = np.radians([float(row["longitude"]) for row in coordinate_rows])
        source_xyz = np.column_stack([
            np.cos(source_lat) * np.cos(source_lon),
            np.cos(source_lat) * np.sin(source_lon),
            np.sin(source_lat),
        ])
    variable_names = set(EXTERNAL_ALIASES)
    for target_row in target:
        source_row = by_unit.get(str(target_row["spatial_unit"]))
        if source_row is None and source_xyz is not None:
            lat, lon = math.radians(float(target_row["latitude"])), math.radians(float(target_row["longitude"]))
            query = np.asarray([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
            dots = np.clip(source_xyz @ query, -1.0, 1.0)
            nearest = int(np.argmax(dots))
            distance_km = float(math.acos(float(dots[nearest])) * 6371.0088)
            if distance_km <= max_distance_km:
                source_row = coordinate_rows[nearest]
                distances.append(distance_km); method = "nearest_coordinate"
        if source_row is not None:
            matched += 1
            for name in variable_names:
                if source_row.get(name) not in (None, ""):
                    target_row[name] = source_row[name]
    return {
        "join_method": method, "matched_unit_count": matched, "target_unit_count": len(target),
        "coverage": matched / len(target) if target else 0.0,
        "max_match_distance_km": max(distances) if distances else 0.0,
        "registered_distance_cap_km": max_distance_km,
    }


def _analyse_variable(
    task: str, model: str, rows: Sequence[Mapping[str, Any]], variable: str,
    role: str, *, n_boot: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    clean = [row for row in rows if math.isfinite(_float(row.get(variable))) and math.isfinite(_float(row.get("mean_risk")))]
    if len(clean) < 20:
        return ({"task": task, "model": model, "variable": variable, "role": role,
                 "status": "unavailable_insufficient_units", "n": len(clean)}, [],
                {"task": task, "model": model, "variable": variable,
                 "status": "unavailable_insufficient_units"})
    risk = np.asarray([float(row["mean_risk"]) for row in clean])
    exposure = np.asarray([float(row[variable]) for row in clean])
    latitude = np.asarray([float(row["latitude"]) for row in clean])
    longitude = np.asarray([float(row["longitude"]) for row in clean])
    support = np.asarray([max(1.0, float(row.get("support", 1))) for row in clean])
    controls = np.column_stack([
        latitude / 90.0, np.sin(np.radians(longitude)), np.cos(np.radians(longitude)), np.log1p(support),
    ])
    cells = [_spatial_cell(lat, lon) for lat, lon in zip(latitude, longitude)]
    cluster_qa = {"task": task, "model": model, "variable": variable,
                  **_spatial_cluster_qa(cells)}
    ci_low, ci_high, cell_count = _spatial_bootstrap_ci(
        risk, exposure, controls, cells, n_boot=n_boot,
    )
    # Prespecified equal-frequency quartiles; ties are kept together where possible.
    quantiles = np.quantile(exposure, [0, .25, .5, .75, 1])
    bins = np.unique(quantiles)
    effect_rows: list[dict[str, Any]] = []
    if len(bins) >= 3:
        assignments = np.searchsorted(bins[1:-1], exposure, side="right")
        rng = np.random.default_rng(1907)
        for bin_index in sorted(set(assignments)):
            idx = np.flatnonzero(assignments == bin_index)
            boot = [float(np.mean(rng.choice(risk[idx], size=len(idx), replace=True))) for _ in range(n_boot)]
            effect_rows.append({
                "task": task, "model": model, "variable": variable,
                "bin": int(bin_index + 1), "n": len(idx),
                "exposure_min": float(np.min(exposure[idx])), "exposure_max": float(np.max(exposure[idx])),
                "mean_exposure": float(np.mean(exposure[idx])), "mean_risk": float(np.mean(risk[idx])),
                "risk_ci_low": float(np.quantile(boot, .025)), "risk_ci_high": float(np.quantile(boot, .975)),
            })
    return ({
        "task": task, "model": model, "variable": variable, "role": role,
        "status": "complete", "n": len(clean), "coverage": len(clean) / len(rows),
        "spearman_rho": _spearman(risk, exposure),
        "partial_spearman_rho": _partial_spearman(risk, exposure, controls),
        "spatial_cluster_bootstrap_ci_low": ci_low,
        "spatial_cluster_bootstrap_ci_high": ci_high,
        "spatial_cluster_count": cell_count, "spatial_cluster_width_degrees": 15,
        "spatial_cluster_definition": cluster_qa["definition"],
        "controls": "latitude+sin(longitude)+cos(longitude)+log1p(unit_support)",
        "inference_boundary": "association_not_causation",
    }, effect_rows, cluster_qa)


def _plot_overlay(rows: Sequence[Mapping[str, Any]], variable: str, title: str, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    clean = [row for row in rows if math.isfinite(_float(row.get(variable)))]
    lon = np.asarray([float(row["longitude"]) for row in clean])
    lat = np.asarray([float(row["latitude"]) for row in clean])
    covariate = np.asarray([float(row[variable]) for row in clean])
    risk = np.asarray([float(row["mean_risk"]) for row in clean])
    tail = risk >= np.quantile(risk, .9)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15), sharex=True, sharey=True, constrained_layout=True)
    first = axes[0].scatter(lon, lat, c=covariate, s=13, cmap="cividis", alpha=.8, linewidth=0, rasterized=True)
    axes[0].scatter(lon[tail], lat[tail], facecolors="none", edgecolors="#D55E00", s=35, linewidth=.7, label="Top-decile risk")
    axes[0].legend(frameon=False, loc="lower left")
    fig.colorbar(first, ax=axes[0], label=variable.replace("_", " "))
    second = axes[1].scatter(lon, lat, c=risk, s=13, cmap="viridis", vmin=0, vmax=1, alpha=.8, linewidth=0, rasterized=True)
    fig.colorbar(second, ax=axes[1], label="Mean risk")
    for ax, label in zip(axes, ("Covariate with tail-risk outline", "Observed unit risk")):
        ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude", title=label)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-60, -30, 0, 30, 60]); ax.grid(alpha=.18)
    fig.suptitle(title, fontweight="bold")
    paths = _save_figure(fig, output, _safe_stem(title.lower()) + "_overlay")
    plt.close(fig)
    return paths


def _plot_effect(effect_rows: Sequence[Mapping[str, Any]], title: str, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    rows = sorted(effect_rows, key=lambda row: int(row["bin"]))
    x = np.asarray([float(row["mean_exposure"]) for row in rows])
    y = np.asarray([float(row["mean_risk"]) for row in rows])
    low = np.asarray([float(row["risk_ci_low"]) for row in rows])
    high = np.asarray([float(row["risk_ci_high"]) for row in rows])
    fig, ax = plt.subplots(figsize=(4.7, 3.8), constrained_layout=True)
    ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="o-", color="#0072B2", capsize=3)
    for xi, yi, row in zip(x, y, rows):
        ax.annotate(f"n={row['n']}", (xi, yi), xytext=(4, 5), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Mean covariate value in preregistered quartile", ylabel="Mean unit risk", title=title)
    ax.grid(alpha=.18)
    paths = _save_figure(fig, output, _safe_stem(title.lower()) + "_effect")
    plt.close(fig)
    return paths


def build_geographic_risk_association(
    atlas_dir: str | Path, output_dir: str | Path, *,
    alphaearth_sample_csv: str | Path | None = None,
    fmow_sample_csvs: Mapping[str, str | Path] | None = None,
    alphaearth_external_csv: str | Path | None = None,
    fmow_external_csvs: Mapping[str, str | Path] | None = None,
    n_boot: int = 500,
) -> dict[str, Any]:
    atlas, output = Path(atlas_dir), ensure_dir(output_dir)
    assets: list[tuple[str, str, Path]] = []
    alpha_path = atlas / "alphaearth_spatial_unit_risk.csv"
    if alpha_path.is_file():
        assets.append(("AlphaEarth", "AlphaEarth", alpha_path))
    for path in sorted(atlas.glob("fmow_*_spatial_unit_risk.csv")):
        model = path.name.removeprefix("fmow_").removesuffix("_spatial_unit_risk.csv")
        assets.append(("fMoW", model, path))
    availability, results, effects, cluster_qa_rows, artifacts = [], [], [], [], []
    provenance: list[dict[str, Any]] = []
    for task, model, risk_path in assets:
        rows = [dict(row) for row in read_csv_rows(risk_path)]
        if task == "AlphaEarth" and alphaearth_sample_csv:
            derived, evidence = derive_alphaearth_covariates(alphaearth_sample_csv)
            provenance.append({"task": task, "model": model, "kind": "internal", **evidence})
            provenance.append({"task": task, "model": model, "kind": "internal_join", **_merge_by_unit_or_coordinate(rows, derived, max_distance_km=0.1)})
        if task == "fMoW" and (fmow_sample_csvs or {}).get(model):
            sample_path = Path((fmow_sample_csvs or {})[model])
            iterator = _iter_csv(sample_path)
            first = next(iterator, {})
            provenance.append({
                "task": task, "model": model, "kind": "internal_schema_audit",
                "source": str(sample_path), "columns": sorted(first),
                "same_variable_matchability": (
                    "GHSL/population/nightlights require coordinate-matched external covariates; "
                    "fMoW scene categories are not land-cover heterogeneity and no reference-map confidence/disagreement is inferred"
                ),
            })
        external_path = alphaearth_external_csv if task == "AlphaEarth" else (fmow_external_csvs or {}).get(model)
        if external_path:
            external, evidence = load_external_covariates(external_path)
            provenance.append({"task": task, "model": model, "kind": "external", **evidence})
            provenance.append({"task": task, "model": model, "kind": "external_join", **_merge_by_unit_or_coordinate(rows, external)})
        for variable, spec in VARIABLES.items():
            if task not in spec["tasks"]:
                availability.append({"task": task, "model": model, "variable": variable,
                                     "role": spec["role"], "status": "not_registered_for_task",
                                     "reason": "fMoW scene labels cannot substitute for land-cover/reference-map variables"})
                continue
            count = sum(math.isfinite(_float(row.get(variable))) for row in rows)
            coverage = count / len(rows) if rows else 0.0
            status = "ready" if count >= 20 and coverage >= .8 else "unavailable"
            reason = "" if status == "ready" else "missing_field_or_coverage_below_0.80_or_fewer_than_20_units"
            availability.append({"task": task, "model": model, "variable": variable,
                                 "role": spec["role"], "status": status, "unit_count": count,
                                 "total_units": len(rows), "coverage": coverage, "reason": reason})
            if status != "ready":
                continue
            result, effect, cluster_qa = _analyse_variable(
                task, model, rows, variable, str(spec["role"]), n_boot=n_boot,
            )
            results.append(result); effects.extend(effect); cluster_qa_rows.append(cluster_qa)
            prefix = task if model == task else f"{task} — {model}"
            title = f"{prefix} — {variable.replace('_', ' ')}"
            artifacts += [path.name for path in _plot_overlay(rows, variable, title, output)]
            if effect:
                artifacts += [path.name for path in _plot_effect(effect, title, output)]
    reben_path = atlas / "reben_country_label_burden.csv"
    reben_status: dict[str, Any] = {"status": "not_available"}
    if reben_path.is_file():
        reben_rows = read_csv_rows(reben_path)
        artifacts += [path.name for path in plot_reben_burden(reben_rows, output)]
        write_csv(output / "reben_country_label_burden_reproduced.csv", reben_rows)
        reben_status = {
            "status": "reproduced", "cell_count": len(reben_rows),
            "analysis_boundary": "burden_reproduction_only_no_socioeconomic_regression_for_10_countries",
        }
    write_csv(output / "covariate_availability.csv", availability)
    write_csv(output / "association_results.csv", results)
    write_csv(output / "association_effect_bins.csv", effects)
    write_csv(output / "spatial_cluster_qa.csv", cluster_qa_rows)
    failed_cluster_qa = [row for row in cluster_qa_rows if row.get("status") != "pass"]
    if failed_cluster_qa:
        raise ValueError(f"Fixed spatial-cluster QA failed: {failed_cluster_qa}")
    write_csv(output / "association_input_provenance.csv", provenance)
    qa = run_visual_qa([output / name for name in artifacts])
    write_csv(output / "visual_qa.csv", qa)
    if any(row["status"] != "pass" for row in qa):
        raise ValueError(f"Association visual QA failed: {[row for row in qa if row['status'] != 'pass']}")
    manifest = {
        "schema": SCHEMA, "status": "complete", "cpu_only": True, "training": False,
        "large_external_downloads": False, "n_boot": n_boot,
        "confirmatory_family": [name for name, spec in VARIABLES.items() if spec["role"] == "confirmatory"],
        "exploratory_family": [name for name, spec in VARIABLES.items() if spec["role"] != "confirmatory"],
        "primary_estimand": "partial_Spearman_controlling_latitude_longitude_and_log_support",
        "significance_policy": "effect_sizes_and_spatial_cluster_bootstrap_CI_no_dichotomous_p_value_claims",
        "spatial_sensitivity": "15_degree_fixed_latitude_bands_with_band_centre_adjusted_longitude_cluster_bootstrap_95CI",
        "spatial_cluster_qa_status": "pass",
        "spatial_cluster_qa": cluster_qa_rows,
        "minimum_readiness": {"coverage": .8, "units": 20},
        "availability": availability, "results": results, "reben": reben_status,
        "artifacts": artifacts, "visual_qa": qa,
        "claim_boundary": (
            "Confirmatory means variables and estimands were fixed before this association run; it does not imply causality. "
            "Population density and nightlights remain exploratory exposures. Missing variables are not replaced by proxies."
        ),
    }
    (output / "geographic_risk_association_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return manifest
