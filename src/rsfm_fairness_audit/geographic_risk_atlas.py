from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.io import ensure_dir, write_csv


SCHEMA = "geobwer.geographic_risk_atlas.v1"
LATITUDE_FIELDS = ("latitude", "lat", "center_lat", "centroid_lat")
LONGITUDE_FIELDS = ("longitude", "lon", "lng", "center_lon", "centroid_lon")
UNIT_FIELDS = ("spatial_block_id", "location_id", "site_id", "independent_unit_id", "sample_id")
ALPHAEARTH_AGGREGATE_TABLES = (
    "geobwer_by_group.csv", "geobwer_profile.csv", "geobwer_summary.csv",
)
ALPHAEARTH_SAMPLE_TABLES = (
    "formal_audit_table.csv", "alphaearth_full_eval_predictions.csv",
    "alphaearth_full_predictions.csv", "audit_table.csv",
)


def _first(row: Mapping[str, Any], fields: Sequence[str], *, required: bool = True) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    if required:
        raise ValueError(f"Missing all required fields {fields!r}")
    return ""


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def inspect_coordinate_asset(
    path: str | Path, *, required_unit_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    iterator = _iter_csv(path)
    try:
        first = next(iterator)
    except StopIteration:
        return {"path": str(path), "status": "empty", "columns": []}
    lat_field = next((name for name in LATITUDE_FIELDS if name in first), None)
    lon_field = next((name for name in LONGITUDE_FIELDS if name in first), None)
    accepted_units = tuple(required_unit_fields or UNIT_FIELDS)
    unit_field = next((name for name in accepted_units if name in first), None)
    risk_field = next((name for name in ("risk", "error", "correct") if name in first), None)
    ready = bool(lat_field and lon_field and risk_field and unit_field)
    return {
        "path": str(path), "status": "ready" if ready else "blocked_missing_fields",
        "columns": list(first), "latitude_field": lat_field, "longitude_field": lon_field,
        "unit_field": unit_field, "risk_field": risk_field,
        "required_unit_fields": list(accepted_units),
    }


def discover_alphaearth_atlas_asset(root: str | Path) -> tuple[Path, dict[str, Any]]:
    """Find a sample-level AlphaEarth coordinate/risk table and reject aggregates.

    The canonical v2 package stores its valid atlas source under
    ``formal_outputs/formal_audit_table.csv``. GeoBWER tables under
    ``geobwer_raw`` are slice/axis aggregates and cannot be mapped without
    inventing coordinates.
    """
    root = Path(root)
    preferred = root / "formal_outputs" / "formal_audit_table.csv"
    candidates = [preferred] if preferred.is_file() else []
    for name in ALPHAEARTH_SAMPLE_TABLES:
        candidates.extend(sorted(root.rglob(name)))
    seen: set[Path] = set()
    inspected: list[dict[str, Any]] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        status = inspect_coordinate_asset(
            candidate, required_unit_fields=("spatial_block_id",),
        )
        inspected.append(status)
        if status["status"] == "ready":
            ignored = [
                {
                    "path": str(path),
                    "reason": "aggregate GeoBWER evidence has no real coordinates or sample-level spatial_block/risk contract",
                }
                for name in ALPHAEARTH_AGGREGATE_TABLES
                for path in sorted(root.rglob(name))
            ]
            return candidate, {
                "discovery_root": str(root), "selected_path": str(candidate),
                "selection_rule": "validated real coordinates + spatial_block_id + sample risk",
                "inspected_candidates": inspected,
                "ignored_aggregate_tables": ignored,
            }
    aggregates = [
        str(path) for name in ALPHAEARTH_AGGREGATE_TABLES
        for path in sorted(root.rglob(name))
    ]
    raise ValueError(
        "No AlphaEarth sample-level atlas asset satisfies latitude/longitude + "
        f"spatial_block_id + risk below {root}. Inspected={inspected}; "
        f"aggregate_only_tables={aggregates}"
    )


def aggregate_coordinate_risk(
    path: str | Path, *, split: str | None = "test",
    required_unit_fields: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    readiness = inspect_coordinate_asset(path, required_unit_fields=required_unit_fields)
    if readiness["status"] != "ready":
        raise ValueError(f"Coordinate asset is not atlas-ready: {readiness}")
    lat_field, lon_field = readiness["latitude_field"], readiness["longitude_field"]
    unit_field = readiness["unit_field"] or "sample_id"
    risk_field = readiness["risk_field"]
    # Circular longitude averaging prevents a unit spanning the dateline from
    # being plotted near Greenwich.
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    for index, row in enumerate(_iter_csv(path)):
        if split and row.get("split") not in (None, "", split):
            continue
        try:
            lat, lon = float(row[lat_field]), float(row[lon_field])
            raw_risk = float(row[risk_field])
            risk = 1.0 - raw_risk if risk_field == "correct" else raw_risk
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (lat, lon, risk)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        unit = str(row.get(unit_field, "") or f"row_{index}")
        acc = sums[unit]
        acc[0] += lat
        acc[1] += math.sin(math.radians(lon))
        acc[2] += math.cos(math.radians(lon))
        acc[3] += risk
        acc[4] += 1
    rows = [
        {"spatial_unit": unit, "latitude": acc[0] / acc[4],
         "longitude": math.degrees(math.atan2(acc[1], acc[2])),
         "mean_risk": acc[3] / acc[4], "support": int(acc[4])}
        for unit, acc in sums.items() if acc[4] > 0
    ]
    if not rows:
        raise ValueError(f"No finite coordinate-risk rows found in {path}")
    q90 = float(np.quantile([row["mean_risk"] for row in rows], .9))
    for row in rows:
        row["tail_excess_over_unit_q90"] = max(0.0, float(row["mean_risk"]) - q90)
    readiness.update({"status": "ready", "usable_spatial_unit_count": len(rows), "unit_risk_q90": q90})
    return rows, readiness


def _seed_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("seed_"):
            return part.removeprefix("seed_")
    return path.parent.name


def _aggregate_country_label(path: Path) -> dict[tuple[str, str], tuple[float, int]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in _iter_csv(path):
        country = _first(row, ("country", "country_iso3", "country_code"))
        label = _first(row, ("class_label", "label", "label_name"))
        # Paired reBEN label audits use the registered 0-1 label error field.
        # Do not silently substitute risk_bce: that would change the atlas
        # estimand relative to the paired GeoBWER result.
        if row.get("risk_binary_error") not in (None, ""):
            risk = float(row["risk_binary_error"])
        elif row.get("risk") not in (None, ""):
            risk = float(row["risk"])
        elif row.get("label_risk") not in (None, ""):
            risk = float(row["label_risk"])
        elif row.get("error") not in (None, ""):
            risk = float(row["error"])
        elif row.get("correct") not in (None, ""):
            risk = 1.0 - float(row["correct"])
        else:
            raise ValueError(
                "reBEN label audit lacks registered binary-error risk fields; "
                f"available columns={sorted(row)}"
            )
        if math.isfinite(risk):
            values[(country, label)][0] += risk
            values[(country, label)][1] += 1
    return {key: (value[0] / value[1], int(value[1])) for key, value in values.items() if value[1]}


def aggregate_reben_country_label_burden(root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    id_files = sorted(root.rglob("id_label_audit.csv"))
    ood_files = sorted(root.rglob("ood_label_audit.csv"))
    id_by_seed = {_seed_from_path(path): path for path in id_files}
    ood_by_seed = {_seed_from_path(path): path for path in ood_files}
    seeds = sorted(set(id_by_seed) & set(ood_by_seed), key=lambda value: int(value) if value.isdigit() else value)
    if not seeds:
        raise ValueError("No matched seed_*/id_label_audit.csv and ood_label_audit.csv files found")
    accumulated: dict[tuple[str, str], list[float]] = defaultdict(list)
    supports: dict[tuple[str, str], list[int]] = defaultdict(list)
    for seed in seeds:
        id_values = _aggregate_country_label(id_by_seed[seed])
        ood_values = _aggregate_country_label(ood_by_seed[seed])
        if set(id_values) != set(ood_values):
            raise ValueError(f"Country-label support differs between ID and OOD for seed {seed}")
        for key in id_values:
            accumulated[key].append(ood_values[key][0] - id_values[key][0])
            supports[key].append(min(id_values[key][1], ood_values[key][1]))
    rows = [{
        "country": key[0], "class_label": key[1], "seed_count": len(values),
        "mean_delta_risk": float(np.mean(values)), "delta_risk_sd": float(np.std(values)),
        "positive_seed_count": int(sum(value > 0 for value in values)), "minimum_cell_support": min(supports[key]),
    } for key, values in sorted(accumulated.items())]
    return rows, {
        "path": str(root), "status": "ready", "representation": "country_x_label_burden_matrix",
        "seed_count": len(seeds), "country_count": len({row["country"] for row in rows}),
        "label_count": len({row["class_label"] for row in rows}), "cell_count": len(rows),
        "risk_contract": "risk_binary_error (registered 0-1 label error; correct fallback only)",
        "limitation": "country is an administrative slice; no within-country coordinates are inferred",
    }


def _style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_coordinate_atlas(rows: Sequence[Mapping[str, Any]], title: str, stem: str, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharex=True, sharey=True)
    lon = np.asarray([float(row["longitude"]) for row in rows])
    lat = np.asarray([float(row["latitude"]) for row in rows])
    risk = np.asarray([float(row["mean_risk"]) for row in rows])
    excess = np.asarray([float(row["tail_excess_over_unit_q90"]) for row in rows])
    size = np.clip(np.sqrt([float(row["support"]) for row in rows]) * 5, 8, 55)
    panels = ((risk, "viridis", "Mean risk"), (excess, "magma", "Tail excess above unit q90"))
    for ax, (color, cmap, label) in zip(axes, panels):
        artist = ax.scatter(
            lon, lat, c=color, s=size, cmap=cmap, vmin=0.0, vmax=1.0,
            alpha=.78, linewidth=0, rasterized=True,
        )
        ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude")
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-60, -30, 0, 30, 60])
        ax.grid(alpha=.18); ax.set_title(label)
        fig.colorbar(artist, ax=ax, shrink=.82)
    fig.suptitle(title + " — observed coordinate units (no interpolation)", fontweight="bold")
    fig.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"; fig.savefig(path, dpi=300, bbox_inches="tight"); paths.append(path)
    plt.close(fig)
    return paths


def plot_reben_burden(rows: Sequence[Mapping[str, Any]], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    countries = sorted({str(row["country"]) for row in rows})
    labels = sorted({str(row["class_label"]) for row in rows})
    lookup = {(str(row["country"]), str(row["class_label"])): float(row["mean_delta_risk"]) for row in rows}
    matrix = np.asarray([[lookup.get((country, label), np.nan) for label in labels] for country in countries])
    bound = max(.01, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(max(8.5, .43 * len(labels)), max(3.6, .34 * len(countries))))
    image = ax.imshow(matrix, cmap="PuOr_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    ax.set_yticks(range(len(countries)), countries)
    ax.set(xlabel="Label", ylabel="Country")
    ax.set_title("reBEN paired S2→S1 country × label burden", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Mean Δ risk (S1 OOD − S2 ID)")
    fig.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"reben_country_label_burden.{suffix}"; fig.savefig(path, dpi=300, bbox_inches="tight"); paths.append(path)
    plt.close(fig)
    return paths


def build_geographic_risk_atlas(
    output_dir: str | Path, *, alphaearth_csv: str | Path | None = None,
    alphaearth_root: str | Path | None = None,
    fmow_csvs: Mapping[str, str | Path] | None = None, reben_paired_dir: str | Path | None = None,
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    readiness: list[dict[str, Any]] = []
    artifacts: list[str] = []
    if alphaearth_csv and alphaearth_root:
        raise ValueError("Use only one of alphaearth_csv or alphaearth_root")
    alpha_discovery: dict[str, Any] | None = None
    if alphaearth_root:
        alphaearth_csv, alpha_discovery = discover_alphaearth_atlas_asset(alphaearth_root)
    if alphaearth_csv:
        rows, status = aggregate_coordinate_risk(
            alphaearth_csv, required_unit_fields=("spatial_block_id",),
        )
        status.update({"task": "AlphaEarth", "scientific_role": "exploratory_spatial_localization"})
        if alpha_discovery:
            status["automatic_discovery"] = alpha_discovery
        readiness.append(status); write_csv(output / "alphaearth_spatial_unit_risk.csv", rows)
        artifacts += [path.name for path in plot_coordinate_atlas(rows, "AlphaEarth", "alphaearth_coordinate_risk", output)]
    for name, path in (fmow_csvs or {}).items():
        rows, status = aggregate_coordinate_risk(path)
        status.update({"task": "fMoW", "model": name, "scientific_role": "exploratory_spatial_localization"})
        readiness.append(status); write_csv(output / f"fmow_{name}_spatial_unit_risk.csv", rows)
        artifacts += [item.name for item in plot_coordinate_atlas(rows, f"fMoW — {name}", f"fmow_{name}_coordinate_risk", output)]
    if reben_paired_dir:
        rows, status = aggregate_reben_country_label_burden(reben_paired_dir)
        status.update({"task": "reBEN", "scientific_role": "descriptive_paired_burden_localization"})
        readiness.append(status); write_csv(output / "reben_country_label_burden.csv", rows)
        artifacts += [path.name for path in plot_reben_burden(rows, output)]
    manifest = {
        "schema": SCHEMA, "status": "complete" if readiness else "no_inputs", "cpu_only": True,
        "external_downloads": False, "training": False, "readiness": readiness, "artifacts": artifacts,
        "visual_contract": {
            "coordinate_risk_cmap": "viridis", "tail_burden_cmap": "magma",
            "signed_delta_cmap": "PuOr_r centered at zero", "formats": ["png_300dpi", "pdf_vector_text"],
            "coordinate_policy": "observed points or spatial-unit centroids only; never infer missing geography",
        },
        "claim_boundary": "Atlas outputs localize descriptive risk; they do not create causal or multiplicity-adjusted geographic inference.",
    }
    (output / "geographic_risk_atlas_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output / "atlas_asset_readiness.csv", readiness)
    return manifest
