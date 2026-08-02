from __future__ import annotations

import csv
import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import read_csv_rows
from rsfm_fairness_audit.sen1floods11_formal import (
    Sen1FormalizationError,
    _identified_units,
    _load_units,
)


class Sen119ModelDescriptiveError(RuntimeError):
    """Raised when the frozen 19-model descriptive panel is not comparable."""


MODES = ("S1", "S2", "S1+S2")
SEEDS = (42, 73, 101)
SPLITS = ("validation", "standard_test", "bolivia_holdout", "combined_held_out")
EXPECTED_SOURCE_COUNTS = {"validation": 89, "standard_test": 90, "bolivia_holdout": 15}
RISK_DEFINITION = "per_chip_one_minus_flood_iou_at_probability_0.5"


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    family: str
    mode: str
    seed: int | None
    run_root: Path
    spatial_grid_profile: str
    comparison_role: str

    def export(self, split: str) -> Path:
        source_split = "test" if split == "standard_test" else split
        return self.run_root / "probabilities" / source_split


def expected_model_specs(
    *, unet_root: str | Path, prithvi_root: str | Path, terramind_root: str | Path
) -> list[ModelSpec]:
    unet = Path(unet_root)
    prithvi = Path(prithvi_root)
    terramind = Path(terramind_root)
    specs: list[ModelSpec] = []
    for family, root, prefix in (
        ("supervised_resnet34_unet", unet, "resnet34_unet"),
        ("terramind_v1_base", terramind, "terramind_v1_base"),
    ):
        for mode in MODES:
            slug = mode.lower().replace("+", "_plus_")
            for seed in SEEDS:
                specs.append(
                    ModelSpec(
                        model_name=f"{prefix}_{slug}_seed_{seed}",
                        family=family,
                        mode=mode,
                        seed=seed,
                        run_root=root / slug / f"seed_{seed}",
                        spatial_grid_profile="official_handlabel_512",
                        comparison_role="same_grid_primary_panel",
                    )
                )
    specs.append(
        ModelSpec(
            model_name="prithvi_eo_v2_300_tl_s2",
            family="prithvi_eo_v2_300_tl",
            mode="S2",
            seed=None,
            run_root=prithvi,
            spatial_grid_profile="official_tl_nearest_mask_224",
            comparison_role="task_specific_external_resolution_reference",
        )
    )
    return specs


def _safe_roots(source_roots: Sequence[Path], output_dir: Path) -> None:
    output = output_dir.resolve()
    for source in source_roots:
        resolved = source.resolve()
        if output == resolved or output.is_relative_to(resolved) or resolved.is_relative_to(output):
            raise Sen119ModelDescriptiveError(
                f"Output and frozen source roots must not overlap: output={output}, source={resolved}."
            )


def _metadata_rows(path: str | Path, *, label: str) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise Sen119ModelDescriptiveError(f"{label} metadata CSV is missing: {source}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(source):
        sample_id = str(row.get("sample_id") or row.get("chip_id") or "").strip()
        if not sample_id:
            raise Sen119ModelDescriptiveError(
                f"{label} metadata row lacks sample_id/chip_id: {source}"
            )
        if sample_id in by_id:
            raise Sen119ModelDescriptiveError(
                f"Duplicate {label} metadata sample_id={sample_id}: {source}"
            )
        by_id[sample_id] = {str(key): value for key, value in row.items()}
    return by_id


def _merge_metadata(
    *,
    core_metadata_csv: str | Path,
    bolivia_metadata_csv: str | Path,
    geospatial_metadata_csv: str | Path,
    output: Path,
) -> Path:
    core = _metadata_rows(core_metadata_csv, label="core431")
    bolivia = _metadata_rows(bolivia_metadata_csv, label="Bolivia15")
    overlap = sorted(set(core) & set(bolivia))
    if overlap:
        raise Sen119ModelDescriptiveError(
            f"Core and Bolivia attribute metadata overlap: {overlap[:5]}."
        )
    attributes = {**core, **bolivia}
    coordinates = _metadata_rows(
        geospatial_metadata_csv, label="authoritative geospatial446"
    )
    if len(core) != 431 or len(bolivia) != 15 or len(attributes) != 446:
        raise Sen119ModelDescriptiveError(
            "Attribute metadata must contain exactly core431 + Bolivia15 unique chips: "
            f"core={len(core)}, bolivia={len(bolivia)}, union={len(attributes)}."
        )
    if len(coordinates) != 446 or set(coordinates) != set(attributes):
        missing = sorted(set(attributes) - set(coordinates))
        extra = sorted(set(coordinates) - set(attributes))
        raise Sen119ModelDescriptiveError(
            "Authoritative geospatial metadata must join one-to-one to all 446 chips: "
            f"rows={len(coordinates)}, missing={missing[:5]}, extra={extra[:5]}."
        )
    joined: dict[str, dict[str, Any]] = {}
    fieldnames: list[str] = []
    coordinate_aliases = {"latitude", "longitude", "lat", "lon", "lng"}
    for sample_id in sorted(attributes):
        attribute_row = {
            key: value
            for key, value in attributes[sample_id].items()
            if key.lower() not in coordinate_aliases and key not in {"chip_id"}
        }
        coordinate_row = coordinates[sample_id]
        try:
            latitude = float(coordinate_row["latitude"])
            longitude = float(coordinate_row["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Sen119ModelDescriptiveError(
                f"Invalid authoritative latitude/longitude for sample_id={sample_id}."
            ) from exc
        if not np.isfinite(latitude) or not np.isfinite(longitude):
            raise Sen119ModelDescriptiveError(
                f"Non-finite authoritative coordinate for sample_id={sample_id}."
            )
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            raise Sen119ModelDescriptiveError(
                f"Out-of-range authoritative coordinate for sample_id={sample_id}: "
                f"latitude={latitude}, longitude={longitude}."
            )
        normalized = {
            **attribute_row,
            "sample_id": sample_id,
            "latitude": latitude,
            "longitude": longitude,
            "coordinate_source": "sen1_geospatial_metadata_446_v0426",
        }
        joined[sample_id] = normalized
        for key in normalized:
            if key not in fieldnames:
                fieldnames.append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in sorted(joined):
            writer.writerow(joined[sample_id])
    return output


def _export_inventory(export: Path) -> dict[str, Any]:
    indexes = sorted((export / "index_parts").glob("*.jsonl"))
    if not indexes:
        raise Sen119ModelDescriptiveError(f"Missing probability index under {export}.")
    rows: list[dict[str, Any]] = []
    for index in indexes:
        for line in index.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise Sen119ModelDescriptiveError(f"Empty or duplicate sample IDs under {export}.")
    digest = hashlib.sha256()
    total_size = 0
    for row in rows:
        raw = Path(str(row.get("probability_path", "")))
        artifact = raw if raw.is_absolute() else export / raw
        if raw.is_absolute() and not artifact.is_file():
            artifact = export / "samples" / raw.name
        if not artifact.is_file():
            raise Sen119ModelDescriptiveError(f"Missing probability artifact: {artifact}")
        sha = file_sha256(artifact)
        size = int(artifact.stat().st_size)
        total_size += size
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\0")
    return {
        "row_count": len(rows),
        "unique_sample_count": len(set(ids)),
        "sample_id_sha256": hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest(),
        "index_sha256": {path.name: file_sha256(path) for path in indexes},
        "probability_artifact_count": len(rows),
        "probability_artifact_total_size_bytes": total_size,
        "sample_probability_sha256": digest.hexdigest(),
    }


def _loaded_split(export: Path, metadata_csv: Path, *, context: str) -> dict[str, Any]:
    rows, probabilities, targets, valid_masks = _load_units(export, metadata_csv=metadata_csv)
    rows, probabilities, targets, valid_masks, support = _identified_units(
        rows, probabilities, targets, valid_masks, context=context
    )
    return {
        "rows": rows,
        "probabilities": probabilities,
        "targets": targets,
        "valid_masks": valid_masks,
        "support": support,
    }


def _metric_summary(loaded: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = loaded["rows"]
    probabilities = loaded["probabilities"]
    targets = loaded["targets"]
    valid_masks = loaded["valid_masks"]
    risks: list[float] = []
    pred_fractions: list[float] = []
    truth_fractions: list[float] = []
    all_nonflood = 0
    all_flood = 0
    near_constant = 0
    probability_values: list[np.ndarray] = []
    event_values: dict[str, list[float]] = {}
    total_tp = total_fp = total_fn = 0
    for row, probability, target, valid in zip(rows, probabilities, targets, valid_masks):
        values = np.asarray(probability, dtype=np.float32)[valid]
        truth = np.asarray(target)[valid] == 1
        prediction = values >= 0.5
        tp = int(np.sum(prediction & truth))
        fp = int(np.sum(prediction & ~truth))
        fn = int(np.sum(~prediction & truth))
        union = tp + fp + fn
        risk = float(1.0 - (tp / union if union else 1.0))
        risks.append(risk)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        pred_fraction = float(np.mean(prediction))
        truth_fraction = float(np.mean(truth))
        pred_fractions.append(pred_fraction)
        truth_fractions.append(truth_fraction)
        all_nonflood += int(not np.any(prediction))
        all_flood += int(np.all(prediction))
        near_constant += int(float(np.max(values) - np.min(values)) <= 1e-6)
        probability_values.append(values)
        event_values.setdefault(str(row["event_id"]), []).append(risk)
    if not risks:
        raise Sen119ModelDescriptiveError("No auditable chip risks were produced.")
    pixels = np.concatenate(probability_values)
    quantiles = np.quantile(pixels, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    pooled_union = total_tp + total_fp + total_fn
    n = len(risks)
    summary = {
        "source_split_sample_count": int(loaded["support"]["source_split_sample_count"]),
        "auditable_sample_count": n,
        "all_ignore_sample_count": int(loaded["support"]["all_ignore_sample_count"]),
        "all_ignore_sample_ids": list(loaded["support"]["all_ignore_sample_ids"]),
        "mean_chip_iou_risk": float(np.mean(risks)),
        "mean_chip_flood_iou": float(1.0 - np.mean(risks)),
        "median_chip_iou_risk": float(np.median(risks)),
        "minimum_chip_iou_risk": float(np.min(risks)),
        "maximum_chip_iou_risk": float(np.max(risks)),
        "pooled_pixel_flood_iou": float(total_tp / pooled_union if pooled_union else 1.0),
        "mean_predicted_flood_fraction": float(np.mean(pred_fractions)),
        "mean_reference_flood_fraction": float(np.mean(truth_fractions)),
        "all_nonflood_prediction_chip_count": all_nonflood,
        "all_nonflood_prediction_chip_rate": float(all_nonflood / n),
        "all_flood_prediction_chip_count": all_flood,
        "all_flood_prediction_chip_rate": float(all_flood / n),
        "near_constant_probability_chip_count": near_constant,
        "near_constant_probability_chip_rate": float(near_constant / n),
        **{f"flood_probability_q{label}": float(value) for label, value in zip(
            ("01", "05", "25", "50", "75", "95", "99"), quantiles
        )},
    }
    events = [
        {
            "event_id": event,
            "auditable_sample_count": len(values),
            "mean_chip_iou_risk": float(np.mean(values)),
            "mean_chip_flood_iou": float(1.0 - np.mean(values)),
            "minimum_chip_iou_risk": float(np.min(values)),
            "maximum_chip_iou_risk": float(np.max(values)),
        }
        for event, values in sorted(event_values.items())
    ]
    return summary, events


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise Sen119ModelDescriptiveError(f"Refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _seed_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["family"]), str(row["mode"]), str(row["split"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (family, mode, split), values in sorted(groups.items()):
        risks = np.asarray([float(row["mean_chip_iou_risk"]) for row in values])
        pooled = np.asarray([float(row["pooled_pixel_flood_iou"]) for row in values])
        output.append({
            "family": family,
            "mode": mode,
            "split": split,
            "run_count": len(values),
            "seeds": ",".join(str(row["seed"]) for row in values),
            "mean_chip_iou_risk_mean": float(np.mean(risks)),
            "mean_chip_iou_risk_std": float(np.std(risks, ddof=1)) if len(risks) > 1 else 0.0,
            "mean_chip_iou_risk_min": float(np.min(risks)),
            "mean_chip_iou_risk_max": float(np.max(risks)),
            "mean_chip_flood_iou_mean": float(1.0 - np.mean(risks)),
            "pooled_pixel_flood_iou_mean": float(np.mean(pooled)),
            "pooled_pixel_flood_iou_std": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
        })
    return output


def _modality_rankings(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [
        row for row in rows
        if row["family"] in {"supervised_resnet34_unet", "terramind_v1_base"}
    ]
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in eligible:
        groups.setdefault((str(row["family"]), str(row["split"]), str(row["seed"])), []).append(row)
    rankings: list[dict[str, Any]] = []
    order_by_panel: dict[tuple[str, str], list[str]] = {}
    for (family, split, seed), values in sorted(groups.items()):
        if {str(row["mode"]) for row in values} != set(MODES):
            raise Sen119ModelDescriptiveError(
                f"Incomplete modality panel for family={family}, split={split}, seed={seed}."
            )
        ordered = sorted(values, key=lambda row: (float(row["mean_chip_iou_risk"]), str(row["mode"])))
        order_text = " < ".join(str(row["mode"]) for row in ordered)
        order_by_panel.setdefault((family, split), []).append(order_text)
        for rank, row in enumerate(ordered, start=1):
            rankings.append({
                "family": family,
                "split": split,
                "seed": seed,
                "mode": row["mode"],
                "mean_chip_iou_risk": row["mean_chip_iou_risk"],
                "risk_rank": rank,
                "modality_order": order_text,
            })
    stability = [
        {
            "family": family,
            "split": split,
            "seed_count": len(orders),
            "unique_modality_order_count": len(set(orders)),
            "modality_order_consistent_across_seeds": len(set(orders)) == 1,
            "observed_orders": " | ".join(orders),
        }
        for (family, split), orders in sorted(order_by_panel.items())
    ]
    return rankings, stability


def _completion_valid(output: Path) -> bool:
    path = output / "completion_contract.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("model_count") != 19:
        return False
    return all(
        (output / record["path"]).is_file()
        and file_sha256(output / record["path"]) == record["sha256"]
        for record in payload.get("artifacts", [])
    )


def run_sen1_19model_descriptive_postprocess(
    *,
    unet_root: str | Path,
    prithvi_root: str | Path,
    terramind_root: str | Path,
    core_metadata_csv: str | Path,
    bolivia_metadata_csv: str | Path,
    geospatial_metadata_csv: str | Path,
    audit_evidence: Sequence[str | Path],
    output_dir: str | Path,
    code_commit: str,
    package_version: str,
) -> Path:
    """Create one descriptive-only, cross-architecture Sen1 probability panel."""

    output = Path(output_dir)
    roots = [Path(unet_root), Path(prithvi_root), Path(terramind_root)]
    _safe_roots(roots, output)
    if _completion_valid(output):
        return output
    if output.exists() and any(output.iterdir()):
        raise Sen119ModelDescriptiveError(
            f"Non-empty incomplete output exists; use a new directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    merged_metadata = _merge_metadata(
        core_metadata_csv=core_metadata_csv,
        bolivia_metadata_csv=bolivia_metadata_csv,
        geospatial_metadata_csv=geospatial_metadata_csv,
        output=output / "official_446_metadata_binding.csv",
    )
    evidence_records = []
    for raw in audit_evidence:
        path = Path(raw)
        if not path.is_file():
            raise Sen119ModelDescriptiveError(f"Required frozen audit evidence is missing: {path}")
        evidence_records.append({"path": str(path), "sha256": file_sha256(path)})

    metrics: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    source_inventory: dict[str, Any] = {}
    specs = expected_model_specs(unet_root=roots[0], prithvi_root=roots[1], terramind_root=roots[2])
    if len(specs) != 19 or len({spec.model_name for spec in specs}) != 19:
        raise Sen119ModelDescriptiveError("The frozen panel must contain exactly 19 unique models.")
    for model_index, spec in enumerate(specs, start=1):
        print(f"[sen1:19model] model={model_index}/19 name={spec.model_name}", flush=True)
        loaded_by_split: dict[str, dict[str, Any]] = {}
        source_inventory[spec.model_name] = {}
        for split in ("validation", "standard_test", "bolivia_holdout"):
            export = spec.export(split)
            inventory = _export_inventory(export)
            expected = EXPECTED_SOURCE_COUNTS[split]
            if inventory["row_count"] != expected:
                raise Sen119ModelDescriptiveError(
                    f"{spec.model_name}/{split} has {inventory['row_count']} rows; expected={expected}."
                )
            source_inventory[spec.model_name][split] = inventory
            loaded_by_split[split] = _loaded_split(
                export, merged_metadata, context=f"19-model descriptive {spec.model_name}/{split}"
            )
        combined = {
            key: loaded_by_split["standard_test"][key] + loaded_by_split["bolivia_holdout"][key]
            for key in ("rows", "probabilities", "targets", "valid_masks")
        }
        combined["support"] = {
            "source_split_sample_count": 105,
            "auditable_sample_count": len(combined["rows"]),
            "all_ignore_sample_count": sum(
                int(loaded_by_split[name]["support"]["all_ignore_sample_count"])
                for name in ("standard_test", "bolivia_holdout")
            ),
            "all_ignore_sample_ids": sum(
                (list(loaded_by_split[name]["support"]["all_ignore_sample_ids"])
                 for name in ("standard_test", "bolivia_holdout")), []
            ),
        }
        loaded_by_split["combined_held_out"] = combined
        for split in SPLITS:
            summary, events = _metric_summary(loaded_by_split[split])
            base = {
                "model": spec.model_name,
                "family": spec.family,
                "mode": spec.mode,
                "seed": "not_applicable" if spec.seed is None else spec.seed,
                "split": split,
                "spatial_grid_profile": spec.spatial_grid_profile,
                "comparison_role": spec.comparison_role,
                "risk_definition": RISK_DEFINITION,
                "status": "descriptive_only",
            }
            metrics.append({**base, **summary})
            event_rows.extend({**base, **event} for event in events)
        del loaded_by_split, combined
        gc.collect()

    metric_path = output / "unified_19model_metrics.csv"
    event_path = output / "event_level_metrics.csv"
    seed_path = output / "three_seed_architecture_modality_summary.csv"
    ranking_path = output / "same_seed_modality_rankings.csv"
    ranking_stability_path = output / "modality_ranking_stability.csv"
    degeneracy_path = output / "prediction_degeneracy_diagnostics.csv"
    _write_csv(metric_path, metrics)
    _write_csv(event_path, event_rows)
    _write_csv(seed_path, _seed_summary(metrics))
    rankings, ranking_stability = _modality_rankings(metrics)
    _write_csv(ranking_path, rankings)
    _write_csv(ranking_stability_path, ranking_stability)
    diagnostic_fields = {
        "model", "family", "mode", "seed", "split", "source_split_sample_count",
        "auditable_sample_count", "all_ignore_sample_count", "mean_predicted_flood_fraction",
        "mean_reference_flood_fraction", "all_nonflood_prediction_chip_count",
        "all_nonflood_prediction_chip_rate", "all_flood_prediction_chip_count",
        "all_flood_prediction_chip_rate", "near_constant_probability_chip_count",
        "near_constant_probability_chip_rate", "flood_probability_q01", "flood_probability_q05",
        "flood_probability_q25", "flood_probability_q50", "flood_probability_q75",
        "flood_probability_q95", "flood_probability_q99",
    }
    _write_csv(degeneracy_path, [{key: row[key] for key in row if key in diagnostic_fields} for row in metrics])
    source_contract = output / "source_contract.json"
    source_contract.write_text(json.dumps({
        "schema": "geobwer.sen1floods11.19model_descriptive_source.v1",
        "status": "bound",
        "source_artifacts_formal": True,
        "derived_result_status": "descriptive_only",
        "model_count": 19,
        "source_roots": {"unet": str(roots[0]), "prithvi": str(roots[1]), "terramind": str(roots[2])},
        "model_registry": [
            {
                "model": spec.model_name,
                "family": spec.family,
                "mode": spec.mode,
                "seed": spec.seed,
                "spatial_grid_profile": spec.spatial_grid_profile,
                "comparison_role": spec.comparison_role,
            }
            for spec in specs
        ],
        "official_metadata_sha256": file_sha256(merged_metadata),
        "metadata_sources": {
            "coordinate_authority": {
                "path": str(Path(geospatial_metadata_csv)),
                "sha256": file_sha256(geospatial_metadata_csv),
                "role": "exclusive_latitude_longitude_source",
            },
            "core431_attributes": {
                "path": str(Path(core_metadata_csv)),
                "sha256": file_sha256(core_metadata_csv),
                "role": "event_split_and_noncoordinate_attributes_only",
            },
            "bolivia15_attributes": {
                "path": str(Path(bolivia_metadata_csv)),
                "sha256": file_sha256(bolivia_metadata_csv),
                "role": "event_split_and_noncoordinate_attributes_only",
            },
        },
        "audit_evidence": evidence_records,
        "probability_inventory": source_inventory,
        "risk_definition": RISK_DEFINITION,
        "threshold": 0.5,
        "test_or_bolivia_used_for_selection": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report = output / "scientific_interpretation_report.md"
    report.write_text(
        "# Sen1Floods11 19-model unified descriptive panel\n\n"
        "All 19 runs are recomputed from immutable full probability exports with the same "
        "per-chip flood-IoU risk at threshold 0.5. The panel reports validation, standard "
        "test, Bolivia holdout, and their 105-chip combined held-out population.\n\n"
        "This is descriptive evidence only. The pre-registered common spatial calibration "
        "failed; no inferential GeoBWER, bootstrap significance, or model-panel fairness "
        "claim is produced here. Pooled-pixel IoU is retained only as a secondary diagnostic "
        "and must not be confused with mean per-chip IoU. U-Net and TerraMind share the "
        "official 512 grid and form the primary same-grid comparison panel. Prithvi uses "
        "the frozen official-TL 224 nearest-mask grid and is therefore a task-specific "
        "external-resolution reference, not an exact pixel-grid peer.\n",
        encoding="utf-8",
    )
    manifest = output / "postprocess_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "geobwer.sen1floods11.19model_descriptive_postprocess.v1",
        "status": "descriptive_only_complete",
        "formal_evidence": False,
        "source_artifacts_formal": True,
        "code_commit": str(code_commit),
        "package_version": str(package_version),
        "model_count": 19,
        "metric_row_count": len(metrics),
        "event_row_count": len(event_rows),
        "risk_definition": RISK_DEFINITION,
        "inferential_geobwer_run": False,
        "bootstrap_run": False,
        "source_contract_sha256": file_sha256(source_contract),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_paths = [
        metric_path,
        event_path,
        seed_path,
        ranking_path,
        ranking_stability_path,
        degeneracy_path,
        source_contract,
        report,
        manifest,
        merged_metadata,
    ]
    completion = output / "completion_contract.json"
    completion.write_text(json.dumps({
        "schema": "geobwer.sen1floods11.19model_descriptive_completion.v1",
        "status": "complete",
        "formal_evidence": False,
        "model_count": 19,
        "code_commit": str(code_commit),
        "package_version": str(package_version),
        "artifacts": [
            {"path": path.relative_to(output).as_posix(), "sha256": file_sha256(path), "size_bytes": int(path.stat().st_size)}
            for path in artifact_paths
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


__all__ = [
    "Sen119ModelDescriptiveError",
    "ModelSpec",
    "expected_model_specs",
    "run_sen1_19model_descriptive_postprocess",
]
