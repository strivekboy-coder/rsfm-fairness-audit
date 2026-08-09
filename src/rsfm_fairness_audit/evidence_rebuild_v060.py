from __future__ import annotations

import csv
import base64
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit import __version__
from rsfm_fairness_audit.bwer_core import compute_geobwer
from rsfm_fairness_audit.bwer_inference import paired_simultaneous_risk_boxes
from rsfm_fairness_audit.bwer_inference import calibrate_spatial_block_scale, equal_area_block_ids
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.geobwer import audit
from rsfm_fairness_audit.geobwer_certification import paired_risk_triple_from_boxes
from rsfm_fairness_audit.risk_spec import RiskSpec
from rsfm_fairness_audit.cluster_uncertainty import cluster_max_lac, cluster_hoeffding_crc_lac


EVIDENCE_REBUILD_SCHEMA = "geobwer.evidence_rebuild.v0.6"
DEFAULT_BETAS = (0.05, 0.10, 0.20, 0.30)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _new_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned evidence output: {output}")
    output.mkdir(parents=True)
    return output


def _source_contract(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "read_only": True}
        for path in paths
    ]


def _load_b64_npz(path: str | Path) -> dict[str, np.ndarray]:
    encoded = "".join(Path(path).read_text(encoding="ascii").splitlines())
    with np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def seal_evidence_output(output_dir: str | Path) -> Path:
    """Write an immutable completion contract for a new derived-evidence directory."""
    output = Path(output_dir)
    contract = output / "completion_contract.json"
    if contract.exists():
        raise FileExistsError(f"Completion contract already exists: {contract}")
    artifacts = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        artifacts.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)})
    write_json(contract, {
        "schema": "geobwer.evidence_rebuild.completion.v1",
        "status": "complete", "package_version": __version__,
        "model_training_or_inference": False, "artifact_count": len(artifacts),
        "artifacts": artifacts,
    })
    return contract


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in output:
            raise ValueError("Paired formal tables require unique, non-empty sample_id values.")
        output[sample_id] = row
    return output


def _aligned_pair(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    cluster_column: str,
) -> tuple[list[float], list[float], list[str], list[str], list[str]]:
    left, right = _index(rows_a), _index(rows_b)
    if set(left) != set(right):
        raise ValueError("Paired model tables do not have identical sample support.")
    ids = sorted(left)
    loss_a: list[float] = []
    loss_b: list[float] = []
    groups: list[str] = []
    clusters: list[str] = []
    for sample_id in ids:
        a, b = left[sample_id], right[sample_id]
        group_a, group_b = str(a.get(axis, "")), str(b.get(axis, ""))
        cluster_a, cluster_b = str(a.get(cluster_column, "")), str(b.get(cluster_column, ""))
        if not group_a or group_a != group_b or not cluster_a or cluster_a != cluster_b:
            raise ValueError(f"Pair identity mismatch for sample={sample_id}, axis={axis}.")
        loss_a.append(float(a["risk"]))
        loss_b.append(float(b["risk"]))
        groups.append(group_a)
        clusters.append(cluster_a)
    return loss_a, loss_b, groups, clusters, ids


def run_fmow_same_seed_paired_v12(
    *,
    dofa_tables: Mapping[int, str | Path],
    resnet_tables: Mapping[int, str | Path],
    output_dir: str | Path,
    axes: Sequence[str] = (
        "country", "region", "un_region", "continent", "class_label",
        "country_class", "region_class",
    ),
    cluster_column: str = "site_id",
    beta: float = 0.10,
    min_clusters: int = 75,
    n_bootstrap: int = 2000,
) -> dict[str, Path]:
    output = _new_output(output_dir)
    if set(dofa_tables) != set(resnet_tables) or not dofa_tables:
        raise ValueError("DOFA and ResNet must expose the same non-empty aligned seed labels.")
    result_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    sources = [*dofa_tables.values(), *resnet_tables.values()]
    for seed in sorted(dofa_tables):
        rows_a = read_csv(dofa_tables[seed])
        rows_b = read_csv(resnet_tables[seed])
        for axis in axes:
            loss_a, loss_b, groups, clusters, ids = _aligned_pair(
                rows_a, rows_b, axis=axis, cluster_column=cluster_column
            )
            group_names = sorted(set(groups))
            cluster_counts = {
                group: len({cluster for value, cluster in zip(groups, clusters) if value == group})
                for group in group_names
            }
            eligible = [group for group in group_names if cluster_counts[group] >= min_clusters]
            point_a = {
                group: mean(value for value, label in zip(loss_a, groups) if label == group)
                for group in group_names
            }
            point_b = {
                group: mean(value for value, label in zip(loss_b, groups) if label == group)
                for group in group_names
            }
            lower_a = {group: 0.0 for group in group_names}
            upper_a = {group: 1.0 for group in group_names}
            lower_b = {group: 0.0 for group in group_names}
            upper_b = {group: 1.0 for group in group_names}
            critical = float("nan")
            # A disparity band needs at least two inferentially eligible
            # groups.  One eligible group cannot identify a tail contrast.
            inferential_groups = eligible if len(eligible) >= 2 else []
            if inferential_groups:
                keep = [index for index, group in enumerate(groups) if group in set(inferential_groups)]
                boxes = paired_simultaneous_risk_boxes(
                    [loss_a[index] for index in keep],
                    [loss_b[index] for index in keep],
                    [groups[index] for index in keep],
                    [clusters[index] for index in keep],
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    min_clusters_per_group=min_clusters,
                )
                if boxes.validity.value != "valid":
                    raise RuntimeError("Prequalified paired groups did not produce a valid joint band.")
                lower_a.update(dict(boxes.lower_a)); upper_a.update(dict(boxes.upper_a))
                lower_b.update(dict(boxes.lower_b)); upper_b.update(dict(boxes.upper_b))
                critical = boxes.critical_value
            triple = paired_risk_triple_from_boxes(
                lower_a, upper_a, lower_b, upper_b,
                beta=beta, point_a=point_a, point_b=point_b,
            )
            point_a_card = compute_geobwer(point_a, beta)
            point_b_card = compute_geobwer(point_b, beta)
            status = "formal_confirmed" if len(eligible) == len(group_names) else "formal_partial"
            result_rows.append({
                "seed": seed, "axis": axis, "model_a": "DOFAv2", "model_b": "ResNet50_common9",
                "beta": beta, "common_sample_count": len(ids), "fixed_group_count": len(group_names),
                "eligible_group_count": len(eligible), "inferential_group_count": len(inferential_groups),
                "ineligible_group_count": len(group_names)-len(eligible),
                "min_clusters": min_clusters, "critical_value": critical, "evidence_status": status,
                "dofa_mean_risk": point_a_card.mean_risk, "dofa_tail_risk": point_a_card.tail_risk,
                "dofa_geobwer": point_a_card.bwer, "resnet_mean_risk": point_b_card.mean_risk,
                "resnet_tail_risk": point_b_card.tail_risk, "resnet_geobwer": point_b_card.bwer,
                "delta_mean": triple.delta_mean.point, "delta_mean_low": triple.delta_mean.lower,
                "delta_mean_high": triple.delta_mean.upper, "delta_tail": triple.delta_tail.point,
                "delta_tail_low": triple.delta_tail.lower, "delta_tail_high": triple.delta_tail.upper,
                "delta_geobwer": triple.delta_bwer.point, "delta_geobwer_low": triple.delta_bwer.lower,
                "delta_geobwer_high": triple.delta_bwer.upper,
                "no_harm_decision": triple.no_harm_decision.value,
                "estimand_note": "same sample/site support; aligned seed labels are reporting alignment, not shared training randomness",
            })
            for group in group_names:
                support_rows.append({"seed": seed, "axis": axis, "group": group,
                    "paired_cluster_count": cluster_counts[group], "eligible": group in eligible})
    results_path = output / "same_seed_paired_certification_v12.csv"
    support_path = output / "paired_cluster_eligibility.csv"
    write_csv(results_path, result_rows); write_csv(support_path, support_rows)
    summary_rows = []
    for axis in axes:
        selected = [row for row in result_rows if row["axis"] == axis]
        values = [float(row["delta_geobwer"]) for row in selected]
        summary_rows.append({"axis": axis, "seed_count": len(values), "delta_geobwer_mean": mean(values),
            "delta_geobwer_min": min(values), "delta_geobwer_max": max(values),
            "direction_consistent": all(value >= 0 for value in values) or all(value <= 0 for value in values),
            "all_no_harm_certified": all(row["no_harm_decision"] == "certified_no_harm_improvement" for row in selected),
            "evidence_status": "formal_confirmed" if all(row["evidence_status"] == "formal_confirmed" for row in selected) else "formal_partial"})
    summary_path = output / "three_seed_paired_summary_v12.csv"; write_csv(summary_path, summary_rows)
    manifest = {"schema": EVIDENCE_REBUILD_SCHEMA, "task": "fmow_sentinel", "stage": "same_seed_paired_1.2",
        "package_version": __version__, "model_training_or_inference": False, "test_used_for_selection": False,
        "sources": _source_contract(sources), "beta": beta, "min_clusters": min_clusters,
        "n_bootstrap": n_bootstrap, "old_evidence_preserved": True}
    manifest_path = output / "postprocess_manifest.json"; write_json(manifest_path, manifest)
    return {"results": results_path, "support": support_path, "summary": summary_path, "manifest": manifest_path}


def run_reben_fixed_universe_v12(
    *, unified_metrics: str | Path, support_universe: str | Path,
    output_dir: str | Path, beta: float = 0.10, min_clusters: int = 75,
) -> dict[str, Path]:
    output = _new_output(output_dir)
    metrics = read_csv(unified_metrics)
    support = json.loads(Path(support_universe).read_text(encoding="utf-8"))
    countries = [str(value) for value in support["fixed_countries"]]
    counts = {str(row["country"]): int(row["cluster_count"]) for row in support["support_rows"]}
    eligible = [country for country in countries if counts[country] >= min_clusters]
    # With no inferentially eligible group, the fixed-universe RiskSpec box is
    # [0,1]^G.  Its exact GeoBWER range is computed rather than hand-coded.
    lower = {country: 0.0 for country in countries}; upper = {country: 1.0 for country in countries}
    maximum = compute_geobwer({countries[0]: 1.0, **{country: 0.0 for country in countries[1:]}}, beta).bwer
    rows = []
    for row in metrics:
        rows.append({"run_id": row["run_id"], "family": row["family"], "mode": row["mode"], "seed": row["seed"],
            "mean_risk": row["deployment_mean_risk"], "tail_risk": row["tail_risk"], "geobwer": row["geobwer"],
            "fixed_country_count": len(countries), "eligible_country_count": len(eligible),
            "eligible_countries": ";".join(eligible), "cluster_threshold": min_clusters,
            "partial_geobwer_lower": 0.0, "partial_geobwer_upper": maximum,
            "evidence_status": "formal_partial", "certification_interpretation": "point estimate retained; fixed-universe inferential set is uninformative because every country has fewer than the calibrated cluster threshold"})
    result_path = output / "fixed_universe_certification_v12.csv"; write_csv(result_path, rows)
    support_path = output / "country_cluster_eligibility_v12.csv"
    write_csv(support_path, [{"country": country, "cluster_count": counts[country], "eligible": country in eligible, "threshold": min_clusters} for country in countries])
    manifest_path = output / "postprocess_manifest.json"
    write_json(manifest_path, {"schema": EVIDENCE_REBUILD_SCHEMA, "task": "reben", "stage": "fixed_universe_partial_id_1.2",
        "package_version": __version__, "model_training_or_inference": False, "test_used_for_selection": False,
        "sources": _source_contract([unified_metrics, support_universe]), "risk_bounds": [0.0,1.0],
        "beta": beta, "min_clusters": min_clusters, "fixed_countries": countries, "eligible_countries": eligible,
        "old_evidence_preserved": True})
    return {"results": result_path, "support": support_path, "manifest": manifest_path}


def run_sen1_event_geobwer(
    *, event_metrics: str | Path, output_dir: str | Path,
    betas: Sequence[float] = DEFAULT_BETAS,
) -> dict[str, Path]:
    output = _new_output(output_dir)
    rows = read_csv(event_metrics)
    keys = sorted({(row["model"], row["split"]) for row in rows})
    profile: list[dict[str, Any]] = []
    allocation: list[dict[str, Any]] = []
    for model, split in keys:
        selected = [row for row in rows if row["model"] == model and row["split"] == split]
        risks = {row["event_id"]: float(row["mean_chip_iou_risk"]) for row in selected}
        if len(risks) < 2:
            continue
        for beta in betas:
            point = compute_geobwer(risks, float(beta))
            profile.append({"model": model, "split": split, "beta": beta,
                "event_count": len(risks), "mean_risk": point.mean_risk, "tail_risk": point.tail_risk,
                "event_geobwer": point.bwer, "tail_effective_groups": point.allocation.tail_effective_groups,
                "max_tail_atom_share": point.allocation.max_tail_atom_share,
                "tail_regime": point.allocation.tail_regime,
                "boundary_risk": point.allocation.boundary_risk, "evidence_status": "descriptive_only",
                "spatial_inference_valid": False})
            for event, mass in point.allocation.conditional_tail_weights:
                if mass > 0:
                    allocation.append({"model": model, "split": split, "beta": beta,
                        "event_id": event, "tail_weight": mass, "event_risk": risks[event]})
    profile_path = output / "event_geobwer_beta_profile.csv"; write_csv(profile_path, profile)
    allocation_path = output / "event_tail_allocation.csv"; write_csv(allocation_path, allocation)
    manifest_path = output / "postprocess_manifest.json"
    write_json(manifest_path, {"schema": EVIDENCE_REBUILD_SCHEMA, "task": "sen1floods11",
        "stage": "event_geobwer_beta_tail_regime", "package_version": __version__,
        "risk_estimand": "equal chips within event, then equal deployment mass across observed held-out events",
        "spatial_inference_valid": False, "evidence_status": "descriptive_only",
        "model_training_or_inference": False, "sources": _source_contract([event_metrics]),
        "betas": list(map(float, betas)), "old_evidence_preserved": True})
    return {"profile": profile_path, "allocation": allocation_path, "manifest": manifest_path}


def run_alphaearth_validation_spatial_recalibration(
    *,
    calibration_bundle_b64: str | Path,
    test_bundle_b64: str | Path,
    eval_metadata: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    candidate_cell_km: Sequence[float] = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0),
    calibration_max_per_country: int = 50,
    n_simulations: int = 200,
    calibration_bootstrap: int = 500,
    audit_bootstrap: int = 2000,
    min_clusters: int = 75,
    seed: int = 42,
) -> dict[str, Path]:
    """Recalibrate AlphaEarth spatial inference using validation outcomes only.

    Split identity, class order, targets and probabilities come from the
    already-audited v2 bundles.  Original GEE shards provide coordinates and
    geography only; their older ``split`` field is never reused.
    """

    output = _new_output(output_dir)
    metadata = _index(read_csv(eval_metadata))
    calibration_bundle = _load_b64_npz(calibration_bundle_b64)
    test_bundle = _load_b64_npz(test_bundle_b64)
    required = {"sample_id", "probabilities", "targets", "class_names"}
    if not required.issubset(calibration_bundle) or not required.issubset(test_bundle):
        raise ValueError("AlphaEarth probability bundles are missing required arrays.")
    class_names = tuple(map(str, calibration_bundle["class_names"].tolist()))
    if class_names != tuple(map(str, test_bundle["class_names"].tolist())):
        raise ValueError("AlphaEarth calibration/test class order differs.")
    calibration_ids = list(map(str, calibration_bundle["sample_id"].tolist()))
    test_ids = list(map(str, test_bundle["sample_id"].tolist()))
    if len(set(calibration_ids)) != len(calibration_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("AlphaEarth bundle sample IDs are not unique.")
    if set(calibration_ids) & set(test_ids):
        raise ValueError("AlphaEarth calibration/test bundle IDs overlap.")
    if set(calibration_ids) | set(test_ids) != set(metadata):
        raise ValueError("AlphaEarth bundle union does not match frozen compact metadata.")
    class_index = {name: index for index, name in enumerate(class_names)}
    def bundle_rows(bundle: Mapping[str, np.ndarray], ids: Sequence[str], split: str) -> list[dict[str, Any]]:
        probabilities = np.asarray(bundle["probabilities"], dtype=float)
        targets = np.asarray(bundle["targets"], dtype=int)
        if probabilities.shape != (len(ids), len(class_names)) or targets.shape != (len(ids),):
            raise ValueError("AlphaEarth bundle shape mismatch.")
        if not np.all(np.isfinite(probabilities)) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4):
            raise ValueError("AlphaEarth bundle probabilities are invalid.")
        rows: list[dict[str, Any]] = []
        for index, sample_id in enumerate(ids):
            meta = metadata[sample_id]
            if "latitude" in bundle and "longitude" in bundle:
                if not math.isclose(float(bundle["latitude"][index]), float(meta["lat"]), abs_tol=1e-9) or not math.isclose(
                    float(bundle["longitude"][index]), float(meta["lon"]), abs_tol=1e-9
                ):
                    raise ValueError(f"Frozen coordinate mismatch for sample={sample_id}.")
            target_code = str(int(float(meta["worldcover_label"])))
            if target_code not in class_index or class_index[target_code] != int(targets[index]):
                raise ValueError(f"WorldCover target mismatch for sample={sample_id}.")
            rows.append({**meta, "sample_id": sample_id, "split": split,
                "risk": float(int(np.argmax(probabilities[index]) != targets[index]))})
        return rows
    calibration = bundle_rows(calibration_bundle, calibration_ids, "calibration")
    test = bundle_rows(test_bundle, test_ids, "test")
    if (len(calibration), len(test)) != (22818, 24030):
        raise ValueError("Frozen AlphaEarth calibration/test counts must be 22,818/24,030.")
    if {row["spatial_block_id"] for row in calibration} & {row["spatial_block_id"] for row in test}:
        raise ValueError("Frozen AlphaEarth calibration/test spatial blocks overlap.")

    by_country: dict[str, list[dict[str, Any]]] = {}
    for row in calibration:
        by_country.setdefault(str(row["country_iso3"]), []).append(row)
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for country in sorted(by_country):
        items = by_country[country]
        indexes = np.arange(len(items)); rng.shuffle(indexes)
        selected.extend(items[int(index)] for index in indexes[:calibration_max_per_country])
    signature_payload = {
        "source_sha": [sha256_file(calibration_bundle_b64), sha256_file(test_bundle_b64), sha256_file(eval_metadata)],
        "calibration_ids": hashlib.sha256("\n".join(sorted(row["sample_id"] for row in selected)).encode()).hexdigest(),
        "candidates": list(map(float, candidate_cell_km)), "n_simulations": n_simulations,
        "calibration_bootstrap": calibration_bootstrap, "seed": seed,
        "gate": "one_sided_coverage_lower_fpr_upper_v1",
    }
    calibration_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    block = calibrate_spatial_block_scale(
        [row["risk"] for row in selected],
        [row["country_iso3"] for row in selected],
        [float(row["lat"]) for row in selected],
        [float(row["lon"]) for row in selected],
        candidate_cell_km=candidate_cell_km,
        n_simulations=n_simulations,
        n_bootstrap=calibration_bootstrap,
        seed=seed,
        beta=0.10,
        require_power_gate=False,
        coverage_tolerance=0.02,
        false_positive_tolerance=0.01,
    )
    calibration_path = output / "validation_only_spatial_calibration.json"
    write_json(calibration_path, {
        "schema": "geobwer.alphaearth.validation_only_spatial_calibration.v3",
        "selection_data": "calibration_only", "test_rows_used_for_selection": False,
        "calibration_rows_total": len(calibration), "calibration_rows_simulation_subset": len(selected),
        "calibration_signature": calibration_signature, "gate": "one_sided_coverage_lower_fpr_upper_v1",
        **asdict(block), "validity": block.validity.value,
    })
    risk_cards_path = output / "certification_v12_risk_cards.csv"
    risk_rows: list[dict[str, Any]] = []
    if block.selected_cell_km is not None and block.validity.value == "valid":
        block_ids = equal_area_block_ids(
            [float(row["lat"]) for row in test], [float(row["lon"]) for row in test],
            cell_km=float(block.selected_cell_km),
        )
        base = BWERProtocol.from_mapping(load_yaml(protocol_path))
        risk_spec = RiskSpec(
            name="map_disagreement", lower_bound=0.0, upper_bound=1.0, unit="sample",
            aggregation="mean_within_registered_slice",
            reference="ESA_WorldCover_reference_map_agreement",
            threshold_source="frozen_AlphaEarth_argmax",
            task_adapter="multiclass",
        )
        protocol = replace(
            base,
            certification_version="geobwer_certification_1.2",
            min_clusters_for_inference=min_clusters,
            cluster_eligibility_calibration_signature=calibration_signature,
            risk_spec=risk_spec,
            metadata=tuple(sorted({**dict(base.metadata),
                "spatial_block_cell_km": str(block.selected_cell_km),
                "spatial_block_calibrated": "true",
                "spatial_block_selection": "validation_only_one_sided_gate"}.items())),
        )
        axes = ("country_iso3", "region", "worldcover_class_name", "country_class",
            "region_class", "income_group", "urban_rural_or_built_proxy")
        groups: dict[str, list[str]] = {axis: [] for axis in axes}
        for row in test:
            country, region, label = str(row["country_iso3"]), str(row["region"]), str(row["worldcover_class_name"])
            values = {"country_iso3": country, "region": region, "worldcover_class_name": label,
                "country_class": f"{country}|{label}", "region_class": f"{region}|{label}",
                "income_group": str(row["income_group"]),
                "urban_rural_or_built_proxy": str(row["urban_rural_or_built_proxy"])}
            for axis in axes: groups[axis].append(values[axis])
        audited = audit(
            loss=[row["risk"] for row in test], groups=groups,
            unit_id=[row["sample_id"] for row in test], spatial_block_id=block_ids,
            protocol=protocol, formal=True, n_bootstrap=audit_bootstrap, seed=seed,
        )
        for axis in audited.axes:
            row = axis.to_summary_row(); row["spatial_cell_km"] = block.selected_cell_km
            risk_rows.append(row)
    write_csv(risk_cards_path, risk_rows)
    final_status = "formal_results_available" if risk_rows else "calibration_invalid_descriptive_complete"
    manifest_path = output / "postprocess_manifest.json"
    write_json(manifest_path, {"schema": EVIDENCE_REBUILD_SCHEMA, "task": "alphaearth", "stage": "validation_only_spatial_recalibration",
        "package_version": __version__, "status": final_status, "test_used_for_selection": False,
        "source_artifacts_modified": False, "model_training_or_inference": False,
        "sources": _source_contract([calibration_bundle_b64, test_bundle_b64, eval_metadata, protocol_path]),
        "calibration_signature": calibration_signature, "selected_cell_km": block.selected_cell_km,
        "spatial_inference_valid": bool(risk_rows), "old_v1_v2_evidence_preserved": True})
    return {"calibration": calibration_path, "risk_cards": risk_cards_path, "manifest": manifest_path}


def _fmow_site_id(sample_id: str) -> str:
    match = re.match(r"^(.*)_([0-9]+)_([0-9]+)_(.+)$", sample_id)
    if not match:
        raise ValueError(f"Cannot recover frozen fMoW site identity: {sample_id}")
    return f"{match.group(1)}|{match.group(2)}"


def run_cluster_uncertainty_v060(
    *, fmow_calibration_npz: str | Path, fmow_test_npz: str | Path,
    fmow_test_table: str | Path, alpha_calibration_bundle_b64: str | Path,
    alpha_test_bundle_b64: str | Path, alpha_eval_metadata: str | Path,
    output_dir: str | Path, alpha: float = 0.10,
) -> dict[str, Path]:
    """Build cluster-aware LAC/CRC evidence without training or inference."""
    output = _new_output(output_dir)
    rows: list[dict[str, Any]] = []

    with np.load(fmow_calibration_npz, allow_pickle=False) as archive:
        fcal = {name: archive[name].copy() for name in archive.files}
    with np.load(fmow_test_npz, allow_pickle=False) as archive:
        ftest = {name: archive[name].copy() for name in archive.files}
    test_rows = _index(read_csv(fmow_test_table))
    fcal_ids = list(map(str, fcal["sample_id"].tolist()))
    ftest_ids = list(map(str, ftest["sample_id"].tolist()))
    if set(ftest_ids) != set(test_rows):
        raise ValueError("fMoW probability bundle and formal audit table do not align.")
    fcal_clusters = [_fmow_site_id(value) for value in fcal_ids]
    ftest_clusters = [str(test_rows[value]["site_id"]) for value in ftest_ids]
    for method in (cluster_max_lac, cluster_hoeffding_crc_lac):
        result = method(
            fcal["probabilities"], fcal["targets"], fcal_clusters,
            ftest["probabilities"], ftest["targets"], ftest_clusters,
            alpha=alpha, cluster_design_valid=True,
        )
        rows.append({"task": "fmow_sentinel", **asdict(result),
            "interpretation": "conditional_on_exchangeable_independent_sites"})

    acal = _load_b64_npz(alpha_calibration_bundle_b64)
    atest = _load_b64_npz(alpha_test_bundle_b64)
    alpha_meta = _index(read_csv(alpha_eval_metadata))
    acal_ids = list(map(str, acal["sample_id"].tolist()))
    atest_ids = list(map(str, atest["sample_id"].tolist()))
    if set(acal_ids) | set(atest_ids) != set(alpha_meta):
        raise ValueError("AlphaEarth probability bundles and metadata do not align.")
    acal_clusters = [str(alpha_meta[value]["spatial_block_id"]) for value in acal_ids]
    atest_clusters = [str(alpha_meta[value]["spatial_block_id"]) for value in atest_ids]
    for method in (cluster_max_lac, cluster_hoeffding_crc_lac):
        result = method(
            acal["probabilities"], acal["targets"], acal_clusters,
            atest["probabilities"], atest["targets"], atest_clusters,
            alpha=alpha, cluster_design_valid=False,
        )
        rows.append({"task": "alphaearth", **asdict(result),
            "interpretation": "descriptive_only_because_validation_spatial_gate_failed"})
    result_path = output / "cluster_aware_conformal_crc.csv"
    write_csv(result_path, rows)
    manifest_path = output / "postprocess_manifest.json"
    write_json(manifest_path, {
        "schema": EVIDENCE_REBUILD_SCHEMA, "stage": "cluster_aware_conformal_crc",
        "package_version": __version__, "alpha": alpha,
        "test_used_for_calibration": False, "model_training_or_inference": False,
        "unsupported_tasks": {
            "reben": "full frozen calibration probability bundles not materialized in this CPU snapshot",
            "sen1floods11": "11-event panel cannot support a nontrivial cluster-level calibration claim",
        },
        "sources": _source_contract([fmow_calibration_npz, fmow_test_npz, fmow_test_table,
            alpha_calibration_bundle_b64, alpha_test_bundle_b64, alpha_eval_metadata]),
    })
    return {"results": result_path, "manifest": manifest_path}


def run_fmow_proper_score_sensitivity(
    *, dofa_tables: Mapping[int, str | Path], resnet_tables: Mapping[int, str | Path],
    output_dir: str | Path,
    axes: Sequence[str] = ("country", "region", "un_region", "continent", "class_label"),
    beta: float = 0.10,
) -> dict[str, Path]:
    """Descriptive paired log-loss sensitivity on the exact frozen support.

    Log loss is kept in nats and is deliberately not forced into the bounded
    certification machinery.  This is a risk-construct sensitivity check, not
    a replacement primary estimand.
    """
    output = _new_output(output_dir)
    results: list[dict[str, Any]] = []
    for seed in sorted(set(dofa_tables) & set(resnet_tables)):
        left, right = _index(read_csv(dofa_tables[seed])), _index(read_csv(resnet_tables[seed]))
        if set(left) != set(right):
            raise ValueError("Proper-score sensitivity requires exact paired sample support.")
        ids = sorted(left)
        for axis in axes:
            by_group_a: dict[str, list[float]] = {}; by_group_b: dict[str, list[float]] = {}
            for sample_id in ids:
                a, b = left[sample_id], right[sample_id]
                if str(a.get(axis, "")) != str(b.get(axis, "")) or not str(a.get(axis, "")):
                    raise ValueError(f"Proper-score group mismatch: sample={sample_id}, axis={axis}")
                group = str(a[axis])
                by_group_a.setdefault(group, []).append(float(a["log_loss"]))
                by_group_b.setdefault(group, []).append(float(b["log_loss"]))
            ga = {name: float(np.mean(values)) for name, values in by_group_a.items()}
            gb = {name: float(np.mean(values)) for name, values in by_group_b.items()}
            ca, cb = compute_geobwer(ga, beta=beta), compute_geobwer(gb, beta=beta)
            results.append({
                "seed": seed, "axis": axis, "beta": beta, "risk": "multiclass_log_loss_nats",
                "dofa_mean": ca.mean_risk, "dofa_tail": ca.tail_risk, "dofa_geobwer": ca.bwer,
                "resnet_mean": cb.mean_risk, "resnet_tail": cb.tail_risk, "resnet_geobwer": cb.bwer,
                "delta_mean": ca.mean_risk - cb.mean_risk,
                "delta_tail": ca.tail_risk - cb.tail_risk,
                "delta_geobwer": ca.bwer - cb.bwer,
                "evidence_status": "descriptive_only",
                "reason": "unbounded proper score; no bounded-risk Certification 1.2 claim",
            })
    result_path = output / "fmow_log_loss_geobwer_sensitivity.csv"
    write_csv(result_path, results)
    manifest_path = output / "postprocess_manifest.json"
    write_json(manifest_path, {
        "schema": EVIDENCE_REBUILD_SCHEMA, "stage": "fmow_proper_score_sensitivity",
        "risk": "multiclass_log_loss_nats", "primary_estimand_changed": False,
        "evidence_status": "descriptive_only", "model_training_or_inference": False,
        "sources": _source_contract([*dofa_tables.values(), *resnet_tables.values()]),
    })
    return {"results": result_path, "manifest": manifest_path}


def run_reben_labelwise_sensitivity(
    *, probability_dir: str | Path, unified_metrics: str | Path,
    output_dir: str | Path, betas: Sequence[float] = DEFAULT_BETAS,
    expected_runs: int = 27, expected_samples: int = 119825, expected_labels: int = 19,
) -> dict[str, Path]:
    """Re-audit frozen reBEN predictions using labelwise FNR/FPR risks.

    Per-label thresholds are consumed from the frozen probability bundles and
    are never selected on test.  The analysis is descriptive because labels
    are outcome dimensions rather than independent geographic deployment
    clusters; it is a risk-construct sensitivity for the primary Hamming audit.
    """
    output = _new_output(output_dir)
    root = Path(probability_dir)
    paths = sorted(root.glob("*.npz"))
    if len(paths) != int(expected_runs):
        raise ValueError(f"Expected exactly {expected_runs} frozen reBEN probability bundles, found {len(paths)}.")
    metric_rows = {row["run_id"]: row for row in read_csv(unified_metrics)}
    label_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    reference_ids: np.ndarray | None = None
    reference_targets: np.ndarray | None = None
    reference_classes: tuple[str, ...] | None = None
    source_hashes: list[dict[str, Any]] = []
    for path in paths:
        stem = path.stem
        match = re.fullmatch(r"(.+)__(s1|s2|s1_plus_s2)__seed_(42|73|101)", stem)
        if not match:
            raise ValueError(f"Unrecognized reBEN bundle identity: {path.name}")
        family, mode, seed_text = match.groups(); seed = int(seed_text)
        run_id = f"{family}__{mode}__seed_{seed}"
        if run_id not in metric_rows:
            raise ValueError(f"Missing frozen unified metric row for {run_id}.")
        with np.load(path, allow_pickle=False) as archive:
            required = {"sample_id", "probabilities", "targets", "class_names", "thresholds"}
            if not required.issubset(archive.files):
                raise ValueError(f"Missing arrays in {path.name}: {sorted(required - set(archive.files))}")
            sample_ids = archive["sample_id"].astype(str)
            probabilities = np.asarray(archive["probabilities"], dtype=float)
            targets = np.asarray(archive["targets"], dtype=int)
            class_names = tuple(map(str, archive["class_names"].tolist()))
            thresholds = np.asarray(archive["thresholds"], dtype=float)
            if "threshold" in archive.files and not np.array_equal(thresholds, np.asarray(archive["threshold"], dtype=float)):
                raise ValueError(f"Duplicate threshold arrays disagree in {path.name}.")
        if probabilities.shape != targets.shape or probabilities.shape != (int(expected_samples), int(expected_labels)):
            raise ValueError(f"Frozen reBEN probability shape mismatch: {path.name}")
        if thresholds.shape != (int(expected_labels),) or not np.all(np.isfinite(thresholds)) or np.any((thresholds <= 0) | (thresholds >= 1)):
            raise ValueError(f"Invalid validation-locked thresholds: {path.name}")
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError(f"Duplicate reBEN sample IDs: {path.name}")
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"Invalid reBEN probabilities: {path.name}")
        if not np.all(np.isin(targets, [0, 1])):
            raise ValueError(f"Invalid reBEN multilabel targets: {path.name}")
        if reference_ids is None:
            reference_ids, reference_targets, reference_classes = sample_ids.copy(), targets.copy(), class_names
        elif not np.array_equal(sample_ids, reference_ids) or not np.array_equal(targets, reference_targets) or class_names != reference_classes:
            raise ValueError(f"Cross-run reBEN sample/target/class alignment failed: {path.name}")
        predicted = probabilities >= thresholds[None, :]
        risks_by_type: dict[str, dict[str, float]] = {"fnr": {}, "fpr": {}, "balanced_error": {}}
        for index, label in enumerate(class_names):
            positive = targets[:, index] == 1; negative = ~positive
            pos_n, neg_n = int(positive.sum()), int(negative.sum())
            if pos_n == 0 or neg_n == 0:
                raise ValueError(f"Label {label} lacks positive or negative test support in {path.name}.")
            fn = int(np.sum(~predicted[positive, index])); fp = int(np.sum(predicted[negative, index]))
            fnr, fpr = fn / pos_n, fp / neg_n
            balanced = 0.5 * (fnr + fpr)
            risks_by_type["fnr"][label] = fnr
            risks_by_type["fpr"][label] = fpr
            risks_by_type["balanced_error"][label] = balanced
            label_rows.append({
                "run_id": run_id, "family": family, "mode": mode, "seed": seed,
                "label": label, "threshold": float(thresholds[index]),
                "positive_support": pos_n, "negative_support": neg_n,
                "false_negative_count": fn, "false_positive_count": fp,
                "fnr": fnr, "fpr": fpr, "balanced_error": balanced,
                "evidence_status": "descriptive_only",
                "threshold_source": "frozen_validation_calibrated_per_label",
            })
        for risk_name, risks in risks_by_type.items():
            primary = compute_geobwer(risks, beta=0.10)
            run_rows.append({
                "run_id": run_id, "family": family, "mode": mode, "seed": seed,
                "risk": risk_name, "mean_risk": primary.mean_risk,
                "tail_risk": primary.tail_risk, "geobwer": primary.bwer,
                "primary_hamming_geobwer": float(metric_rows[run_id]["geobwer"]),
                "evidence_status": "descriptive_only",
            })
            for beta in betas:
                card = compute_geobwer(risks, beta=float(beta))
                profile_rows.append({
                    "run_id": run_id, "family": family, "mode": mode, "seed": seed,
                    "risk": risk_name, "beta": float(beta), "mean_risk": card.mean_risk,
                    "tail_risk": card.tail_risk, "geobwer": card.bwer,
                    "tail_effective_labels": card.allocation.tail_effective_groups,
                    "max_tail_atom_share": card.allocation.max_tail_atom_share,
                    "boundary_risk": card.allocation.boundary_risk,
                    "tail_regime": card.allocation.tail_regime,
                    "evidence_status": "descriptive_only",
                })
        source_hashes.append({"file": path.name, "sha256": sha256_file(path), "read_only": True})
    label_path = output / "labelwise_fnr_fpr.csv"; write_csv(label_path, label_rows)
    run_path = output / "run_level_labelwise_geobwer.csv"; write_csv(run_path, run_rows)
    profile_path = output / "labelwise_beta_profile.csv"; write_csv(profile_path, profile_rows)
    summary_rows: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in run_rows}):
        for mode in sorted({row["mode"] for row in run_rows if row["family"] == family}):
            for risk in ("fnr", "fpr", "balanced_error"):
                selected = [row for row in run_rows if row["family"] == family and row["mode"] == mode and row["risk"] == risk]
                values = [float(row["geobwer"]) for row in selected]
                summary_rows.append({
                    "family": family, "mode": mode, "risk": risk,
                    "seed_count": len(selected), "geobwer_mean": float(np.mean(values)),
                    "geobwer_min": min(values), "geobwer_max": max(values),
                    "geobwer_range": max(values) - min(values),
                })
    summary_path = output / "three_seed_labelwise_summary.csv"; write_csv(summary_path, summary_rows)
    manifest_path = output / "postprocess_manifest.json"
    write_json(manifest_path, {
        "schema": "geobwer.reben.labelwise_risk_sensitivity.v1",
        "package_version": __version__, "status": "descriptive_complete",
        "run_count": len(paths), "test_sample_count": expected_samples, "label_count": expected_labels,
        "threshold_selection_data": "frozen_validation_only", "test_used_for_threshold_selection": False,
        "primary_estimand_changed": False, "model_training_or_inference": False,
        "source_artifacts_modified": False, "source_bundles": source_hashes,
        "unified_metrics": _source_contract([unified_metrics])[0],
        "interpretation": "labelwise FNR/FPR/balanced-error sensitivity; not geographic cluster inference",
    })
    return {"labelwise": label_path, "runs": run_path, "profile": profile_path,
        "summary": summary_path, "manifest": manifest_path}


def build_evidence_status_matrix(
    *, task_records: Sequence[Mapping[str, Any]], output_dir: str | Path,
) -> dict[str, Path]:
    output = _new_output(output_dir)
    matrix_path = output / "four_task_evidence_status_matrix.csv"
    write_csv(matrix_path, task_records)
    manifest_path = output / "postprocess_manifest.json"
    write_json(manifest_path, {"schema": EVIDENCE_REBUILD_SCHEMA, "stage": "four_task_evidence_status_matrix",
        "package_version": __version__, "rows": len(task_records), "allowed_statuses": [
            "formal_confirmed", "formal_partial", "descriptive_only", "not_identified", "invalid", "revoked"],
        "model_training_or_inference": False})
    return {"matrix": matrix_path, "manifest": manifest_path}
