from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_core import compute_geobwer
from rsfm_fairness_audit.bwer_inference import (
    paired_bwer_comparison,
    simultaneous_group_risk_band,
)
from rsfm_fairness_audit.bwer_standardization import partial_bwer_bounds
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.reben_sensor_audit import compute_multilabel_metrics


FAMILIES = ("supervised_resnet50", "croma", "terramind")
MODES = ("s1", "s2", "s1_plus_s2")
SEEDS = (42, 73, 101)
DEFAULT_BETAS = (0.10, 0.20, 0.30)
DEFAULT_CLUSTER_THRESHOLDS = (2, 3, 5)


class RebenScientificPostprocessError(RuntimeError):
    """Raised when the frozen reBEN panel cannot support read-only analysis."""


@dataclass(frozen=True)
class RunKey:
    family: str
    mode: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.family}__{self.mode}__seed_{self.seed}"


@dataclass(frozen=True)
class RunArtifacts:
    key: RunKey
    run_dir: Path
    formal_table: Path
    probabilities: Path
    metrics_summary: Path
    geobwer_summary: Path
    uncertainty_summary: Path
    formal_manifest: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _find_panel_root(source_root: str | Path) -> Path:
    source = Path(source_root)
    candidates = [source]
    candidates.extend(path for path in source.rglob("*") if path.is_dir())
    matches = [
        candidate
        for candidate in candidates
        if all((candidate / family).is_dir() for family in FAMILIES)
    ]
    if not matches:
        raise RebenScientificPostprocessError(
            f"Could not find a panel root containing {FAMILIES} below {source}."
        )
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def discover_reben_panel(source_root: str | Path) -> tuple[Path, dict[RunKey, RunArtifacts]]:
    root = _find_panel_root(source_root)
    output: dict[RunKey, RunArtifacts] = {}
    missing: list[str] = []
    for family, mode, seed in itertools.product(FAMILIES, MODES, SEEDS):
        key = RunKey(family, mode, seed)
        run_dir = root / family / mode / f"seed_{seed}"
        formal = run_dir / "formal_outputs"
        artifact = RunArtifacts(
            key=key,
            run_dir=run_dir,
            formal_table=formal / "formal_audit_table.csv",
            probabilities=formal / "probabilities.npz",
            metrics_summary=run_dir / "metrics_summary.csv",
            geobwer_summary=run_dir / "geobwer" / "geobwer_summary.csv",
            uncertainty_summary=run_dir / "uncertainty_extensions" / "uncertainty_summary.csv",
            formal_manifest=formal / "formal_output_manifest.json",
        )
        required = (
            artifact.formal_table,
            artifact.probabilities,
            artifact.geobwer_summary,
            artifact.uncertainty_summary,
            artifact.formal_manifest,
        )
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing.extend(f"{key.run_id}: {path}" for path in absent)
        else:
            output[key] = artifact
    if missing:
        raise RebenScientificPostprocessError(
            "The frozen 27-run panel is incomplete:\n" + "\n".join(missing[:30])
        )
    return root, output


def _read_first_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _load_formal_columns(path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[Any]] = {
        "sample_id": [],
        "country": [],
        "cluster": [],
        "risk": [],
        "label_cardinality": [],
        "protocol_hash": [],
        "metric_version": [],
    }
    optional_values: dict[str, list[str]] = {}
    optional_names: list[str] | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        snow_cloud_names = [
            name
            for name in fieldnames
            if ("snow" in name.lower() or "cloud" in name.lower())
            and name not in {"probabilities_path"}
        ]
        optional_names = snow_cloud_names
        for name in snow_cloud_names:
            optional_values[name] = []
        for row in reader:
            columns["sample_id"].append(str(row.get("sample_id", "")))
            columns["country"].append(str(row.get("country", "")))
            columns["cluster"].append(str(row.get("source_tile_id", "")))
            columns["risk"].append(_float(row.get("risk")))
            columns["label_cardinality"].append(
                int(_float(row.get("label_cardinality"), 0.0))
            )
            columns["protocol_hash"].append(str(row.get("protocol_hash", "")))
            columns["metric_version"].append(str(row.get("metric_version", "")))
            for name in snow_cloud_names:
                optional_values[name].append(str(row.get(name, "")))
    if not columns["sample_id"]:
        raise RebenScientificPostprocessError(f"Empty formal audit table: {path}")
    if any(not value for value in columns["sample_id"]):
        raise RebenScientificPostprocessError(f"Missing sample_id in {path}")
    if len(set(columns["sample_id"])) != len(columns["sample_id"]):
        raise RebenScientificPostprocessError(f"Duplicate sample_id in {path}")
    if any(not value for value in columns["country"]):
        raise RebenScientificPostprocessError(f"Missing country in {path}")
    if any(not value for value in columns["cluster"]):
        raise RebenScientificPostprocessError(f"Missing source_tile_id in {path}")
    risks = np.asarray(columns["risk"], dtype=float)
    if np.any(~np.isfinite(risks)) or np.any((risks < 0.0) | (risks > 1.0)):
        raise RebenScientificPostprocessError(f"Invalid bounded risk in {path}")
    output = {
        "sample_id": np.asarray(columns["sample_id"], dtype=str),
        "country": np.asarray(columns["country"], dtype=str),
        "cluster": np.asarray(columns["cluster"], dtype=str),
        "risk": risks,
        "label_cardinality": np.asarray(columns["label_cardinality"], dtype=int),
        "protocol_hash": np.asarray(columns["protocol_hash"], dtype=str),
        "metric_version": np.asarray(columns["metric_version"], dtype=str),
    }
    for name in optional_names or ():
        output[f"optional::{name}"] = np.asarray(optional_values[name], dtype=str)
    return output


def _support(
    countries: np.ndarray,
    clusters: np.ndarray,
    *,
    min_units: int,
    min_clusters: int,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    valid: list[str] = []
    for country in sorted(set(countries.tolist())):
        mask = countries == country
        units = int(np.sum(mask))
        cluster_count = int(len(set(clusters[mask].tolist())))
        supported = units >= min_units and cluster_count >= min_clusters
        if supported:
            valid.append(country)
        rows.append(
            {
                "country": country,
                "sample_count": units,
                "cluster_count": cluster_count,
                "min_units": min_units,
                "min_clusters": min_clusters,
                "supported": supported,
            }
        )
    return tuple(valid), rows


def _group_risks(
    risks: np.ndarray,
    countries: np.ndarray,
    groups: Sequence[str],
) -> dict[str, float]:
    return {
        group: float(np.mean(risks[countries == group]))
        for group in groups
    }


def _cardinality_band(values: np.ndarray) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    output[values <= 0] = "zero"
    output[(values >= 1) & (values <= 2)] = "one_to_two"
    output[(values >= 3) & (values <= 4)] = "three_to_four"
    output[values >= 5] = "five_plus"
    return output.astype(str)


def _standardized_composition_geobwer(
    risks: np.ndarray,
    countries: np.ndarray,
    strata: np.ndarray,
    supported_groups: Sequence[str],
    *,
    beta: float,
) -> dict[str, Any]:
    levels = tuple(sorted(set(strata.tolist())))
    if len(levels) < 2:
        return {
            "validity": "not_available",
            "reason": "fewer_than_two_composition_levels",
        }
    supported_mask = np.isin(countries, np.asarray(supported_groups))
    target = {
        level: float(np.mean(strata[supported_mask] == level))
        for level in levels
    }
    standardized: dict[str, float] = {}
    missing: list[str] = []
    for country in supported_groups:
        value = 0.0
        for level in levels:
            mask = (countries == country) & (strata == level)
            if not np.any(mask):
                missing.append(f"{country}::{level}")
                continue
            value += target[level] * float(np.mean(risks[mask]))
        if not any(item.startswith(f"{country}::") for item in missing):
            standardized[country] = value
    if missing:
        return {
            "validity": "not_identified_missing_composition_cells",
            "reason": "strict_common_composition",
            "missing_cells": ";".join(missing),
            "target_weights": json.dumps(target, sort_keys=True),
        }
    point = compute_geobwer(standardized, beta)
    return {
        "validity": "valid",
        "reason": "",
        "standardized_mean_risk": point.mean_risk,
        "standardized_tail_risk": point.tail_risk,
        "standardized_geobwer": point.bwer,
        "target_weights": json.dumps(target, sort_keys=True),
        "missing_cells": "",
    }


def _find_snow_cloud_stratum(reference: Mapping[str, np.ndarray]) -> tuple[str, np.ndarray] | None:
    candidates: list[tuple[str, np.ndarray]] = []
    for key, values in reference.items():
        if not key.startswith("optional::"):
            continue
        nonempty = np.asarray([str(value).strip() for value in values], dtype=str)
        unique = {value for value in nonempty.tolist() if value}
        if 2 <= len(unique) <= 20:
            candidates.append((key.split("::", 1)[1], nonempty))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (len(set(item[1].tolist())), item[0]))
    return candidates[0]


def _paired_row(
    *,
    comparison_family: str,
    contrast: str,
    run_a: RunKey,
    run_b: RunKey,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    countries: np.ndarray,
    clusters: np.ndarray,
    supported: Sequence[str],
    beta: float,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mask = np.isin(countries, np.asarray(supported))
    result = paired_bwer_comparison(
        risk_a[mask],
        risk_b[mask],
        countries[mask],
        clusters[mask],
        model_a=run_a.run_id,
        model_b=run_b.run_id,
        beta=beta,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    point_a = compute_geobwer(_group_risks(risk_a, countries, supported), beta)
    point_b = compute_geobwer(_group_risks(risk_b, countries, supported), beta)
    row = {
        "comparison_family": comparison_family,
        "contrast": contrast,
        "model_a": run_a.run_id,
        "model_b": run_b.run_id,
        "delta_definition": "model_a_minus_model_b",
        "sample_mean_risk_delta": float(np.mean(risk_a[mask] - risk_b[mask])),
        "tail_risk_delta": point_a.tail_risk - point_b.tail_risk,
        "geobwer_delta": result.delta_bwer,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "direct_multiplier_ci_low": result.direct_multiplier_ci_low,
        "direct_multiplier_ci_high": result.direct_multiplier_ci_high,
        "confidence_level": result.confidence_level,
        "common_groups": ";".join(result.common_groups),
        "common_units": result.common_units,
        "cluster_count": result.cluster_count,
        "validity": result.validity.value,
        "interpretation_scope": "adapted_model_pipeline_under_common_evaluation_contract",
        "causal_backbone_attribution": False,
    }
    difference = risk_a[mask] - risk_b[mask]
    band = simultaneous_group_risk_band(
        difference,
        countries[mask],
        clusters[mask],
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed + 100_000,
        risk_bounds=(-1.0, 1.0),
    )
    estimates = dict(band.estimates)
    lower = dict(band.lower)
    upper = dict(band.upper)
    country_rows = [
        {
            "comparison_family": comparison_family,
            "contrast": contrast,
            "model_a": run_a.run_id,
            "model_b": run_b.run_id,
            "country": group,
            "paired_risk_difference": estimates[group],
            "simultaneous_ci_low": lower.get(group, ""),
            "simultaneous_ci_high": upper.get(group, ""),
            "confidence_level": confidence_level,
            "validity": band.validity.value,
        }
        for group in supported
    ]
    return row, country_rows


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    numeric = (
        "sample_mean_hamming_risk",
        "deployment_mean_risk",
        "tail_risk",
        "geobwer",
        "macro_ap",
        "micro_ap",
        "macro_f1",
        "micro_f1",
    )
    for family, mode in itertools.product(FAMILIES, MODES):
        selected = [
            row for row in rows
            if row["family"] == family and row["mode"] == mode
        ]
        result: dict[str, Any] = {
            "family": family,
            "mode": mode,
            "seed_count": len(selected),
        }
        for name in numeric:
            values = np.asarray([_float(row.get(name)) for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{name}_mean"] = float(np.mean(finite)) if len(finite) else ""
            result[f"{name}_sd"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if len(finite) else ""
            )
        output.append(result)
    return output


def run_reben_scientific_postprocess(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    beta: float = 0.10,
    betas: Sequence[float] = DEFAULT_BETAS,
    min_units: int = 20,
    cluster_thresholds: Sequence[int] = DEFAULT_CLUSTER_THRESHOLDS,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Path]:
    """Create a non-destructive scientific synthesis of the frozen reBEN panel."""

    panel_root, runs = discover_reben_panel(source_root)
    output = ensure_dir(output_dir)
    keys = sorted(runs, key=lambda value: (value.family, value.mode, value.seed))

    reference_key = keys[0]
    reference = _load_formal_columns(runs[reference_key].formal_table)
    sample_ids = reference["sample_id"]
    countries = reference["country"]
    clusters = reference["cluster"]
    fixed_countries = tuple(sorted(set(countries.tolist())))
    protocol_values = set(reference["protocol_hash"].tolist())
    metric_values = set(reference["metric_version"].tolist())
    if len(protocol_values) != 1 or len(metric_values) != 1:
        raise RebenScientificPostprocessError("Reference formal table has mixed protocol or metric versions.")
    protocol_hash = next(iter(protocol_values))
    metric_version = next(iter(metric_values))

    base_supported, support_rows = _support(
        countries,
        clusters,
        min_units=min_units,
        min_clusters=int(cluster_thresholds[0]),
    )
    if len(base_supported) < 2:
        raise RebenScientificPostprocessError("Fewer than two supported countries.")
    risks_by_run: dict[RunKey, np.ndarray] = {}
    metrics_rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    snow_cloud = _find_snow_cloud_stratum(reference)
    for run_index, key in enumerate(keys):
        artifact = runs[key]
        formal = reference if key == reference_key else _load_formal_columns(artifact.formal_table)
        if not np.array_equal(formal["sample_id"], sample_ids):
            raise RebenScientificPostprocessError(f"Sample order drift in {key.run_id}.")
        for invariant in ("country", "cluster", "label_cardinality"):
            if not np.array_equal(formal[invariant], reference[invariant]):
                raise RebenScientificPostprocessError(f"{invariant} drift in {key.run_id}.")
        if set(formal["protocol_hash"].tolist()) != {protocol_hash}:
            raise RebenScientificPostprocessError(f"Protocol drift in {key.run_id}.")
        if set(formal["metric_version"].tolist()) != {metric_version}:
            raise RebenScientificPostprocessError(f"Metric-version drift in {key.run_id}.")
        risks = formal["risk"]
        risks_by_run[key] = risks
        group_risks = _group_risks(risks, countries, base_supported)
        point = compute_geobwer(group_risks, beta)

        metrics = _read_first_row(artifact.metrics_summary) if artifact.metrics_summary.is_file() else {}
        if not metrics:
            with np.load(artifact.probabilities, allow_pickle=False) as archive:
                summary, _ = compute_multilabel_metrics(
                    archive["targets"],
                    archive["probabilities"],
                    archive["thresholds"],
                    archive["class_names"].astype(str).tolist(),
                )
                metrics = {str(name): str(value) for name, value in summary.items()}
        geobwer_existing = _read_first_row(artifact.geobwer_summary)
        metrics_rows.append(
            {
                "run_id": key.run_id,
                "family": key.family,
                "mode": key.mode,
                "seed": key.seed,
                "n_samples": len(sample_ids),
                "sample_mean_hamming_risk": float(np.mean(risks)),
                "deployment_mean_risk": point.mean_risk,
                "tail_risk": point.tail_risk,
                "geobwer": point.bwer,
                "worst_country": max(group_risks, key=group_risks.get),
                "supported_country_count": len(base_supported),
                "fixed_country_count": len(fixed_countries),
                "sample_coverage": float(np.mean(np.isin(countries, np.asarray(base_supported)))),
                "deployment_mass_coverage": len(base_supported) / len(fixed_countries),
                "macro_ap": _float(metrics.get("macro_ap")),
                "micro_ap": _float(metrics.get("micro_ap")),
                "macro_f1": _float(metrics.get("macro_f1")),
                "micro_f1": _float(metrics.get("micro_f1")),
                "existing_validity": geobwer_existing.get("validity", ""),
                "protocol_hash": protocol_hash,
                "metric_version": metric_version,
                "comparison_scope": "adapted_model_pipeline",
            }
        )
        for beta_value in betas:
            profile = compute_geobwer(group_risks, float(beta_value))
            beta_rows.append(
                {
                    "run_id": key.run_id,
                    "family": key.family,
                    "mode": key.mode,
                    "seed": key.seed,
                    "beta": float(beta_value),
                    "deployment_mean_risk": profile.mean_risk,
                    "tail_risk": profile.tail_risk,
                    "geobwer": profile.bwer,
                    "tail_effective_groups": profile.allocation.tail_effective_groups,
                }
            )
        lower_map = {
            country: group_risks[country] if country in group_risks else 0.0
            for country in fixed_countries
        }
        upper_map = {
            country: group_risks[country] if country in group_risks else 1.0
            for country in fixed_countries
        }
        partial = partial_bwer_bounds(lower_map, upper_map, beta=beta)
        partial_rows.append(
            {
                "run_id": key.run_id,
                "fixed_country_count": len(fixed_countries),
                "identified_country_count": len(base_supported),
                "excluded_countries": ";".join(sorted(set(fixed_countries) - set(base_supported))),
                "supported_universe_geobwer": point.bwer,
                "fixed_universe_partial_lower": partial.lower,
                "fixed_universe_partial_upper": partial.upper,
                "partial_validity": partial.validity.value,
                "sample_coverage": float(np.mean(np.isin(countries, np.asarray(base_supported)))),
                "deployment_mass_coverage": len(base_supported) / len(fixed_countries),
            }
        )
        for cluster_threshold in cluster_thresholds:
            supported, _ = _support(
                countries,
                clusters,
                min_units=min_units,
                min_clusters=int(cluster_threshold),
            )
            if len(supported) >= 2:
                sensitivity = compute_geobwer(
                    _group_risks(risks, countries, supported),
                    beta,
                )
                sensitivity_rows.append(
                    {
                        "run_id": key.run_id,
                        "min_units": min_units,
                        "min_clusters": int(cluster_threshold),
                        "supported_countries": ";".join(supported),
                        "supported_country_count": len(supported),
                        "sample_coverage": float(np.mean(np.isin(countries, np.asarray(supported)))),
                        "deployment_mass_coverage": len(supported) / len(fixed_countries),
                        "geobwer": sensitivity.bwer,
                        "tail_risk": sensitivity.tail_risk,
                        "validity": "supported_universe_descriptive",
                    }
                )
            else:
                sensitivity_rows.append(
                    {
                        "run_id": key.run_id,
                        "min_units": min_units,
                        "min_clusters": int(cluster_threshold),
                        "supported_countries": ";".join(supported),
                        "supported_country_count": len(supported),
                        "sample_coverage": float(np.mean(np.isin(countries, np.asarray(supported)))),
                        "deployment_mass_coverage": len(supported) / len(fixed_countries),
                        "geobwer": "",
                        "tail_risk": "",
                        "validity": "insufficient_slices",
                    }
                )
        cardinality = _standardized_composition_geobwer(
            risks,
            countries,
            _cardinality_band(reference["label_cardinality"]),
            base_supported,
            beta=beta,
        )
        composition_rows.append(
            {
                "run_id": key.run_id,
                "composition_axis": "label_cardinality_band",
                **cardinality,
            }
        )
        if snow_cloud is not None:
            snow_name, snow_values = snow_cloud
            snow_result = _standardized_composition_geobwer(
                risks,
                countries,
                snow_values,
                base_supported,
                beta=beta,
            )
            composition_rows.append(
                {
                    "run_id": key.run_id,
                    "composition_axis": snow_name,
                    **snow_result,
                }
            )
        else:
            composition_rows.append(
                {
                    "run_id": key.run_id,
                    "composition_axis": "snow_cloud",
                    "validity": "not_available",
                    "reason": "no_snow_cloud_field_in_formal_audit_table",
                }
            )
        for row in read_csv_rows(artifact.uncertainty_summary):
            uncertainty_rows.append(
                {
                    "run_id": key.run_id,
                    "family": key.family,
                    "mode": key.mode,
                    "seed": key.seed,
                    **row,
                }
            )
        source_records.append(
            {
                "run_id": key.run_id,
                "formal_manifest": str(artifact.formal_manifest.relative_to(panel_root)),
                "formal_manifest_sha256": _sha256(artifact.formal_manifest),
                "formal_table_rows": len(sample_ids),
                "probabilities_file_size": artifact.probabilities.stat().st_size,
            }
        )

    paired_rows: list[dict[str, Any]] = []
    paired_country_rows: list[dict[str, Any]] = []
    comparison_counter = 0
    adjusted_confidence = 1.0 - (1.0 - confidence_level) / 3.0
    for mode, seed_value in itertools.product(MODES, SEEDS):
        family_keys = [RunKey(family, mode, seed_value) for family in FAMILIES]
        for left, right in itertools.combinations(family_keys, 2):
            row, country_rows = _paired_row(
                comparison_family=f"cross_model::{mode}::seed_{seed_value}",
                contrast=f"{left.family}_minus_{right.family}",
                run_a=left,
                run_b=right,
                risk_a=risks_by_run[left],
                risk_b=risks_by_run[right],
                countries=countries,
                clusters=clusters,
                supported=base_supported,
                beta=beta,
                confidence_level=adjusted_confidence,
                n_bootstrap=n_bootstrap,
                seed=seed + comparison_counter,
            )
            paired_rows.append(row)
            paired_country_rows.extend(country_rows)
            comparison_counter += 1
    for family, seed_value in itertools.product(FAMILIES, SEEDS):
        mode_keys = [RunKey(family, mode, seed_value) for mode in MODES]
        for left, right in itertools.combinations(mode_keys, 2):
            row, country_rows = _paired_row(
                comparison_family=f"cross_modality::{family}::seed_{seed_value}",
                contrast=f"{left.mode}_minus_{right.mode}",
                run_a=left,
                run_b=right,
                risk_a=risks_by_run[left],
                risk_b=risks_by_run[right],
                countries=countries,
                clusters=clusters,
                supported=base_supported,
                beta=beta,
                confidence_level=adjusted_confidence,
                n_bootstrap=n_bootstrap,
                seed=seed + comparison_counter,
            )
            paired_rows.append(row)
            paired_country_rows.extend(country_rows)
            comparison_counter += 1

    aggregate_rows = _aggregate_seed_rows(metrics_rows)
    support_summary = {
        "schema": "geobwer.reben.support_universe.v1",
        "fixed_countries": list(fixed_countries),
        "supported_countries": list(base_supported),
        "excluded_countries": sorted(set(fixed_countries) - set(base_supported)),
        "sample_coverage": float(np.mean(np.isin(countries, np.asarray(base_supported)))),
        "equal_country_deployment_mass_coverage": len(base_supported) / len(fixed_countries),
        "min_units": min_units,
        "min_clusters": int(cluster_thresholds[0]),
        "support_rows": support_rows,
    }
    report_lines = [
        "# reBEN 27-run GeoBWER scientific postprocess",
        "",
        "## Comparison contract",
        "",
        "- Primary cross-model estimand: adapted model-pipeline reliability under one common evaluation contract.",
        "- Same test samples, country slices, source-tile clusters, bounded Hamming risk, beta, equal deployment measure, and inference design are enforced.",
        "- Training/adaptation recipes may differ; therefore model-pipeline comparisons are valid, but differences are not causal estimates of the pretrained backbone alone.",
        "",
        "## Universe and support",
        "",
        f"- Fixed countries: {len(fixed_countries)}",
        f"- Supported countries: {len(base_supported)}",
        f"- Sample coverage: {100 * support_summary['sample_coverage']:.2f}%",
        f"- Equal-country deployment-mass coverage: {100 * support_summary['equal_country_deployment_mass_coverage']:.2f}%",
        "",
        "The fixed-universe bounds, supported-universe point estimates, paired intervals, support frontier, beta profile, and composition sensitivity are reported separately.",
    ]
    paths = {
        "unified_metrics": output / "unified_27run_metrics.csv",
        "three_seed_summary": output / "three_seed_model_mode_summary.csv",
        "paired_comparisons": output / "paired_common_support_comparisons.csv",
        "paired_country_differences": output / "paired_country_risk_differences.csv",
        "partial_identification": output / "fixed_universe_partial_identification.csv",
        "support_sensitivity": output / "support_sensitivity.csv",
        "beta_profile": output / "beta_profile_summary.csv",
        "composition_sensitivity": output / "composition_sensitivity.csv",
        "uncertainty": output / "uncertainty_summary_27run.csv",
        "support_contract": output / "support_universe.json",
        "report": output / "scientific_postprocess_report.md",
        "manifest": output / "postprocess_manifest.json",
    }
    write_csv(paths["unified_metrics"], metrics_rows)
    write_csv(paths["three_seed_summary"], aggregate_rows)
    write_csv(paths["paired_comparisons"], paired_rows)
    write_csv(paths["paired_country_differences"], paired_country_rows)
    write_csv(paths["partial_identification"], partial_rows)
    write_csv(paths["support_sensitivity"], sensitivity_rows)
    write_csv(paths["beta_profile"], beta_rows)
    write_csv(paths["composition_sensitivity"], composition_rows)
    write_csv(paths["uncertainty"], uncertainty_rows)
    paths["support_contract"].write_text(
        json.dumps(_native(support_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": "geobwer.reben.scientific_postprocess.v1",
        "read_only_source": True,
        "panel_root": str(panel_root),
        "run_count": len(runs),
        "protocol_hash": protocol_hash,
        "metric_version": metric_version,
        "beta": beta,
        "beta_profile": list(map(float, betas)),
        "support_cluster_thresholds": list(map(int, cluster_thresholds)),
        "confidence_level": confidence_level,
        "paired_familywise_method": "bonferroni_within_each_three_contrast_family",
        "paired_confidence_level": adjusted_confidence,
        "n_bootstrap": n_bootstrap,
        "comparison_scope": "adapted_model_pipeline_under_common_evaluation_contract",
        "causal_backbone_attribution": False,
        "sources": source_records,
        "outputs": {
            name: {
                "path": path.name,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(_native(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "RebenScientificPostprocessError",
    "RunArtifacts",
    "RunKey",
    "discover_reben_panel",
    "run_reben_scientific_postprocess",
]
