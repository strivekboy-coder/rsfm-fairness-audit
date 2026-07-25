from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_core import compute_geobwer
from rsfm_fairness_audit.bwer_inference import (
    paired_bwer_comparison,
    simultaneous_group_risk_band,
)
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.fmow_dofav2_campaign import (
    _copy_rows_with_protocol_hash,
)
from rsfm_fairness_audit.fmow_geography_contract import (
    validate_fmow_geography_contract,
)
from rsfm_fairness_audit.fmow_superclass_feasibility import (
    FEASIBILITY_SCHEMA,
    TAXONOMY_SCHEMA,
    load_fmow_superclass_taxonomy,
)
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.persistent_cache import persist_output


class FmowSuperclassPostprocessError(RuntimeError):
    """Raised when frozen fMoW superclass evidence cannot be reproduced."""


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FmowSuperclassPostprocessError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FmowSuperclassPostprocessError(
            f"{label} is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise FmowSuperclassPostprocessError(
            f"{label} is not a JSON object: {path}"
        )
    return value


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _cell_key(geography: Any, superclass: Any) -> str:
    return f"{str(geography).strip()}|{str(superclass).strip()}"


def _sharp_fixed_universe_bounds(
    known_risks: Sequence[float],
    *,
    fixed_group_count: int,
    beta: float,
) -> tuple[float, float]:
    """Sharp GeoBWER bounds when unsupported group risks lie in [0, 1]."""

    known = np.asarray(known_risks, dtype=float)
    if (
        fixed_group_count < len(known)
        or fixed_group_count < 2
        or np.any(~np.isfinite(known))
        or np.any((known < 0.0) | (known > 1.0))
    ):
        raise ValueError("Invalid known-risk/fixed-universe contract.")
    unknown = fixed_group_count - len(known)
    candidates = np.unique(np.concatenate(([0.0, 1.0], known)))
    lower = float("inf")
    for value in candidates:
        risks = {
            f"known_{index}": float(risk)
            for index, risk in enumerate(known)
        }
        risks.update(
            {
                f"unknown_{index}": float(value)
                for index in range(unknown)
            }
        )
        lower = min(lower, compute_geobwer(risks, beta).bwer)
    upper = 0.0
    for ones in range(unknown + 1):
        risks = {
            f"known_{index}": float(risk)
            for index, risk in enumerate(known)
        }
        risks.update(
            {
                f"unknown_{index}": float(index < ones)
                for index in range(unknown)
            }
        )
        upper = max(upper, compute_geobwer(risks, beta).bwer)
    return max(0.0, lower), min(1.0 - float(beta), upper)


def _augment_rows(
    rows: Sequence[Mapping[str, Any]],
    class_to_superclass: Mapping[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        sample_id = str(row.get("sample_id") or "").strip()
        class_label = str(
            row.get("class_label") or row.get("label") or ""
        ).strip()
        country = str(row.get("country") or "").strip()
        region = str(
            row.get("region")
            or row.get("un_region")
            or row.get("continent")
            or ""
        ).strip()
        site_id = str(row.get("site_id") or "").strip()
        if (
            not sample_id
            or sample_id in seen
            or class_label not in class_to_superclass
            or not country
            or not region
            or not site_id
        ):
            raise FmowSuperclassPostprocessError(
                "Formal rows require unique sample_id, frozen class mapping, "
                "resolved geography, and site_id."
            )
        seen.add(sample_id)
        superclass = class_to_superclass[class_label]
        row.update(
            {
                "class_label": class_label,
                "superclass": superclass,
                "resolved_region": region,
                "region_superclass": _cell_key(region, superclass),
                "country_superclass": _cell_key(country, superclass),
            }
        )
        output.append(row)
    return output


def _align_rows(
    dofa_rows: Sequence[Mapping[str, Any]],
    resnet_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dofa = {str(row["sample_id"]): dict(row) for row in dofa_rows}
    resnet = {str(row["sample_id"]): dict(row) for row in resnet_rows}
    if set(dofa) != set(resnet):
        raise FmowSuperclassPostprocessError(
            "DOFAv2 and ResNet formal sample universes differ; the frozen "
            "same-metadata comparison requires exact sample equality."
        )
    ordered = sorted(dofa)
    left = [dofa[sample_id] for sample_id in ordered]
    right = [resnet[sample_id] for sample_id in ordered]
    for row_a, row_b in zip(left, right):
        for field in (
            "site_id",
            "class_label",
            "country",
            "resolved_region",
            "region_superclass",
            "country_superclass",
        ):
            if str(row_a.get(field)) != str(row_b.get(field)):
                raise FmowSuperclassPostprocessError(
                    f"Paired row metadata mismatch for {field}: "
                    f"{row_a['sample_id']}"
                )
    return left, right


def _supported_cells(
    feasibility_cells: Sequence[Mapping[str, Any]],
    axis: str,
) -> set[str]:
    return {
        _cell_key(row["geography"], row["superclass"])
        for row in feasibility_cells
        if str(row.get("axis")) == axis
        and str(row.get("support_status"))
        in {"descriptive_supported", "confirmatory_supported"}
    }


def _mean_risks(
    rows: Sequence[Mapping[str, Any]],
    axis: str,
    cells: set[str],
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        group = str(row[axis])
        if group in cells:
            values[group].append(float(row["risk"]))
    if set(values) != cells:
        raise FmowSuperclassPostprocessError(
            f"Formal table does not realize every frozen supported {axis} cell."
        )
    return {
        group: float(np.mean(risks))
        for group, risks in sorted(values.items())
    }


def _eligible_groups(
    rows: Sequence[Mapping[str, Any]],
    axis: str,
    protocol: BWERProtocol,
    *,
    allowed: set[str] | None = None,
) -> set[str]:
    samples: dict[str, int] = defaultdict(int)
    clusters: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = str(row[axis])
        if allowed is not None and group not in allowed:
            continue
        samples[group] += 1
        clusters[group].add(str(row["site_id"]))
    return {
        group
        for group in samples
        if samples[group] >= protocol.min_units_per_slice
        and len(clusters[group]) >= protocol.min_clusters_per_slice
    }


def _verify_recorded_artifacts(
    run_dir: Path,
    contract: Mapping[str, Any],
) -> None:
    recorded = contract.get("artifacts")
    if not isinstance(recorded, Mapping):
        raise FmowSuperclassPostprocessError(
            f"Completion contract has no artifact inventory: {run_dir}"
        )
    required = {
        "checkpoint",
        "calibration_probabilities",
        "formal_audit_table",
        "formal_probabilities",
        "formal_output_manifest",
        "run_manifest",
    }
    missing = sorted(required - set(recorded))
    if missing:
        raise FmowSuperclassPostprocessError(
            "Completion contract is missing required artifacts: "
            + ", ".join(missing)
        )
    for name, item in recorded.items():
        if not isinstance(item, Mapping):
            raise FmowSuperclassPostprocessError(
                f"Malformed completion artifact record: {name}"
            )
        path = run_dir / str(item.get("path") or "")
        if (
            not path.is_file()
            or path.stat().st_size != int(item.get("size_bytes", -1))
            or file_sha256(path) != item.get("sha256")
        ):
            raise FmowSuperclassPostprocessError(
                f"Completion artifact failed signature validation: {name}"
            )


def run_fmow_superclass_postprocess(
    *,
    axis_role_freeze: str | Path,
    feasibility_dir: str | Path,
    taxonomy_path: str | Path,
    geography_contract: str | Path,
    dofa_source_root: str | Path,
    dofa_provenance_overlay: str | Path,
    resnet_source_root: str | Path,
    output_dir: str | Path,
    persistent_output_dir: str | Path | None = None,
    geobwer_protocol: str | Path = Path("configs/geobwer/fmow_sentinel.yaml"),
    seeds: Sequence[int] = (42, 73, 101),
    n_bootstrap: int = 2000,
) -> dict[str, Path]:
    """Execute only analyses allowed by the frozen superclass axis roles."""

    freeze_path = Path(axis_role_freeze)
    feasibility_root = Path(feasibility_dir)
    taxonomy_source = Path(taxonomy_path)
    geography_path = Path(geography_contract)
    dofa_root = Path(dofa_source_root)
    overlay_root = Path(dofa_provenance_overlay)
    resnet_root = Path(resnet_source_root)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FmowSuperclassPostprocessError(
            f"Refusing to overwrite non-empty output: {output}"
        )
    freeze = _json_object(freeze_path, "axis-role freeze")
    if (
        freeze.get("schema")
        != "geobwer.fmow.superclass_axis_role_freeze.v1"
        or freeze.get("status")
        != "frozen_before_resnet50_superclass_results"
    ):
        raise FmowSuperclassPostprocessError(
            "Unsupported or non-frozen superclass axis-role contract."
        )
    feasibility_manifest_path = (
        feasibility_root / "superclass_feasibility_manifest.json"
    )
    feasibility_manifest = _json_object(
        feasibility_manifest_path, "feasibility manifest"
    )
    source_contracts = freeze["source_contracts"]
    if (
        feasibility_manifest.get("schema") != FEASIBILITY_SCHEMA
        or feasibility_manifest.get("contract_hash")
        != source_contracts["feasibility_contract_hash"]
        or feasibility_manifest.get("metadata", {}).get("sha256")
        != source_contracts["metadata_sha256"]
        or feasibility_manifest.get("metadata", {}).get(
            "selected_assignment_hash"
        )
        != source_contracts["selected_assignment_hash"]
    ):
        raise FmowSuperclassPostprocessError(
            "Feasibility manifest does not match the axis-role freeze."
        )
    taxonomy, class_to_superclass = load_fmow_superclass_taxonomy(
        taxonomy_source
    )
    if (
        taxonomy.get("schema") != TAXONOMY_SCHEMA
        or file_sha256(taxonomy_source) != source_contracts["taxonomy_sha256"]
        or feasibility_manifest.get("taxonomy", {}).get("sha256")
        != source_contracts["taxonomy_sha256"]
    ):
        raise FmowSuperclassPostprocessError(
            "Taxonomy does not match the frozen feasibility contract."
        )
    geography = validate_fmow_geography_contract(
        geography_path,
        require_formal=True,
    )
    if (
        geography.get("contract_hash")
        != source_contracts["geography_contract_hash"]
        or file_sha256(geography_path)
        != source_contracts["geography_contract_file_sha256"]
    ):
        raise FmowSuperclassPostprocessError(
            "Geography contract does not match the axis-role freeze."
        )
    feasibility_cells_path = (
        feasibility_root / "superclass_feasibility_cells.csv"
    )
    feasibility_summary_path = (
        feasibility_root / "superclass_feasibility_summary.csv"
    )
    for key, path in (
        ("cells", feasibility_cells_path),
        ("summary", feasibility_summary_path),
    ):
        if (
            file_sha256(path)
            != feasibility_manifest.get("artifacts", {})
            .get(key, {})
            .get("sha256")
        ):
            raise FmowSuperclassPostprocessError(
                f"Feasibility {key} artifact hash mismatch."
            )
    feasibility_cells = read_csv_rows(feasibility_cells_path)
    feasibility_summaries = read_csv_rows(feasibility_summary_path)
    region_supported = _supported_cells(
        feasibility_cells, "region_superclass"
    )
    country_supported = _supported_cells(
        feasibility_cells, "country_superclass"
    )
    if (
        len(region_supported)
        != int(
            freeze["axis_roles"]["region_superclass"][
                "descriptive_supported_cell_count"
            ]
        )
        or len(country_supported)
        != int(
            freeze["axis_roles"]["country_superclass"][
                "descriptive_supported_cell_count"
            ]
        )
    ):
        raise FmowSuperclassPostprocessError(
            "Supported cell sets do not match the frozen axis-role counts."
        )
    frozen_panel = {
        _cell_key(item["geography"], item["superclass"])
        for item in freeze["high_support_region_cell_panel"]["cells"]
    }
    observed_panel = {
        _cell_key(row["geography"], row["superclass"])
        for row in feasibility_cells
        if row.get("axis") == "region_superclass"
        and row.get("support_status") == "confirmatory_supported"
    }
    if frozen_panel != observed_panel or len(frozen_panel) != 6:
        raise FmowSuperclassPostprocessError(
            "The exact six-cell validation panel differs from feasibility evidence."
        )

    overlay_manifest_path = overlay_root / "postprocess_manifest.json"
    overlay_manifest = _json_object(
        overlay_manifest_path, "DOFA provenance overlay"
    )
    if (
        overlay_manifest.get("schema")
        != "geobwer.fmow.dofav2_provenance_overlay.v1"
        or overlay_manifest.get("source_artifacts_immutable") is not True
        or overlay_manifest.get("geography_contract_hash")
        != geography["contract_hash"]
    ):
        raise FmowSuperclassPostprocessError(
            "DOFA provenance overlay is not the frozen immutable lineage overlay."
        )
    protocol = BWERProtocol.from_mapping(load_yaml(geobwer_protocol))
    if overlay_manifest.get("protocol_hash") != protocol.signature:
        raise FmowSuperclassPostprocessError(
            "DOFA overlay protocol differs from the requested GeoBWER protocol."
        )
    region_protocol = replace(
        protocol,
        partition_rule="explicit_intersection",
        group_variable="region_superclass",
        metadata=tuple(
            sorted(
                {
                    **dict(protocol.metadata),
                    "axis_role_freeze_sha256": file_sha256(freeze_path),
                    "superclass_taxonomy_sha256": file_sha256(taxonomy_source),
                }.items()
            )
        ),
    )
    country_protocol = replace(
        region_protocol,
        group_variable="country_superclass",
    )

    output.mkdir(parents=True, exist_ok=True)
    paired_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    evidence_sources: dict[str, str] = {
        "axis_role_freeze": file_sha256(freeze_path),
        "feasibility_manifest": file_sha256(feasibility_manifest_path),
        "feasibility_cells": file_sha256(feasibility_cells_path),
        "taxonomy": file_sha256(taxonomy_source),
        "geography_contract": file_sha256(geography_path),
        "dofa_provenance_overlay": file_sha256(overlay_manifest_path),
    }

    for seed in seeds:
        dofa_seed = dofa_root / "probe_seeds" / f"seed_{int(seed)}"
        resnet_seed = resnet_root / f"seed_{int(seed)}"
        dofa_manifest_path = (
            dofa_seed / "formal_outputs" / "formal_output_manifest.json"
        )
        resnet_manifest_path = (
            resnet_seed / "formal_outputs" / "formal_output_manifest.json"
        )
        dofa_manifest = _json_object(
            dofa_manifest_path, f"DOFA seed {seed} manifest"
        )
        resnet_manifest = _json_object(
            resnet_manifest_path, f"ResNet seed {seed} manifest"
        )
        completion_path = resnet_seed / "completion_contract.json"
        completion = _json_object(
            completion_path, f"ResNet seed {seed} completion contract"
        )
        if (
            completion.get("schema")
            != "geobwer.fmow.resnet50_seed_completion.v1"
            or completion.get("complete") is not True
            or int(completion.get("seed", -1)) != int(seed)
            or completion.get("protocol_hash") != protocol.signature
            or completion.get("metadata_sha256")
            != source_contracts["metadata_sha256"]
            or completion.get("geography_contract_hash")
            != geography["contract_hash"]
        ):
            raise FmowSuperclassPostprocessError(
                f"ResNet seed {seed} is not a matching completed formal run."
            )
        _verify_recorded_artifacts(resnet_seed, completion)
        for name, manifest in (
            ("DOFA", dofa_manifest),
            ("ResNet", resnet_manifest),
        ):
            if (
                manifest.get("protocol_hash") != protocol.signature
                or manifest.get("protocol", {}).get("metric_version")
                != protocol.metric_version
                or manifest.get("dataset_lineage", {}).get("metadata_sha256")
                != source_contracts["metadata_sha256"]
            ):
                raise FmowSuperclassPostprocessError(
                    f"{name} seed {seed} formal lineage mismatch."
                )
        if (
            resnet_manifest.get("dataset_lineage", {}).get(
                "geography_contract_hash"
            )
            != geography["contract_hash"]
        ):
            raise FmowSuperclassPostprocessError(
                f"ResNet seed {seed} geography lineage mismatch."
            )
        recorded_dofa_manifest = (
            overlay_manifest.get("source_artifacts", {})
            .get("seed_formal_manifest_sha256", {})
            .get(str(seed))
        )
        if recorded_dofa_manifest != file_sha256(dofa_manifest_path):
            raise FmowSuperclassPostprocessError(
                f"DOFA seed {seed} manifest is not bound by the provenance overlay."
            )
        dofa_table = dofa_seed / "formal_outputs" / "formal_audit_table.csv"
        resnet_table = resnet_seed / "formal_outputs" / "formal_audit_table.csv"
        dofa_rows = _augment_rows(
            read_csv_rows(dofa_table), class_to_superclass
        )
        resnet_rows = _augment_rows(
            read_csv_rows(resnet_table), class_to_superclass
        )
        dofa_rows, resnet_rows = _align_rows(dofa_rows, resnet_rows)
        if len(dofa_rows) != int(source_contracts["selected_row_count"]):
            raise FmowSuperclassPostprocessError(
                f"Seed {seed} row count differs from the frozen test universe."
            )
        evidence_sources[f"dofa_seed_{seed}_formal_table"] = file_sha256(
            dofa_table
        )
        evidence_sources[f"resnet_seed_{seed}_completion"] = file_sha256(
            completion_path
        )

        for model_name, model_rows in (
            ("dofav2", dofa_rows),
            ("resnet50", resnet_rows),
        ):
            for axis, supported, derived_protocol, role in (
                (
                    "region_superclass",
                    region_supported,
                    region_protocol,
                    "secondary_exploratory_supported_universe",
                ),
                (
                    "country_superclass",
                    country_supported,
                    country_protocol,
                    "appendix_supported_cell_exploratory",
                ),
            ):
                selected = [
                    row for row in model_rows if str(row[axis]) in supported
                ]
                retagged = _copy_rows_with_protocol_hash(
                    selected, derived_protocol
                )
                audit_rows(
                    retagged,
                    group_columns=(axis,),
                    protocol=derived_protocol,
                    loss_column="risk",
                    unit_column="independent_unit_id",
                    cluster_column="site_id",
                    formal=True,
                    require_probabilities=True,
                    n_bootstrap=n_bootstrap,
                    seed=int(seed),
                ).to_report(
                    output
                    / "supported_universe"
                    / axis
                    / model_name
                    / f"seed_{int(seed)}"
                )
                fixed_count = int(
                    freeze["axis_roles"][axis][
                        "fixed_universe_cell_count"
                    ]
                )
                known = _mean_risks(model_rows, axis, supported)
                lower, upper = _sharp_fixed_universe_bounds(
                    list(known.values()),
                    fixed_group_count=fixed_count,
                    beta=protocol.beta,
                )
                partial_rows.append(
                    {
                        "seed": int(seed),
                        "model": model_name,
                        "axis": axis,
                        "evidence_role": role,
                        "fixed_universe_cell_count": fixed_count,
                        "identified_supported_cell_count": len(known),
                        "unidentified_cell_count": fixed_count - len(known),
                        "partial_geobwer_lower": lower,
                        "partial_geobwer_upper": upper,
                        "risk_bounds_for_unidentified_cells": "[0,1]",
                        "bounds_status": "sharp_given_supported_cell_point_risks",
                    }
                )

        comparison_axes = (
            ("country", None, "primary_marginal"),
            ("resolved_region", None, "primary_marginal"),
            ("class_label", None, "primary_marginal"),
            (
                "region_superclass",
                region_supported,
                "secondary_exploratory_supported_universe",
            ),
            (
                "country_superclass",
                country_supported,
                "appendix_supported_cell_exploratory",
            ),
        )
        for axis, allowed, role in comparison_axes:
            eligible = _eligible_groups(
                dofa_rows, axis, protocol, allowed=allowed
            )
            if len(eligible) < protocol.min_slices:
                continue
            mask = [
                str(row[axis]) in eligible
                for row in dofa_rows
            ]
            selected_a = [
                row for row, keep in zip(dofa_rows, mask) if keep
            ]
            selected_b = [
                row for row, keep in zip(resnet_rows, mask) if keep
            ]
            comparison = paired_bwer_comparison(
                [float(row["risk"]) for row in selected_a],
                [float(row["risk"]) for row in selected_b],
                [str(row[axis]) for row in selected_a],
                [str(row["site_id"]) for row in selected_a],
                model_a="dofav2",
                model_b="resnet50",
                beta=protocol.beta,
                confidence_level=protocol.confidence_level,
                n_bootstrap=n_bootstrap,
                seed=int(seed),
            )
            paired_rows.append(
                {
                    "seed": int(seed),
                    "axis": axis,
                    "evidence_role": role,
                    "delta_definition": "GeoBWER(DOFAv2)-GeoBWER(ResNet50)",
                    "delta_geobwer": comparison.delta_bwer,
                    "ci_low": comparison.ci_low,
                    "ci_high": comparison.ci_high,
                    "direct_multiplier_ci_low": (
                        comparison.direct_multiplier_ci_low
                    ),
                    "direct_multiplier_ci_high": (
                        comparison.direct_multiplier_ci_high
                    ),
                    "validity": comparison.validity.value,
                    "common_group_count": len(comparison.common_groups),
                    "common_sample_count": comparison.common_units,
                    "common_cluster_count": comparison.cluster_count,
                }
            )

        panel_mask = [
            str(row["region_superclass"]) in frozen_panel
            for row in dofa_rows
        ]
        panel_a = [row for row, keep in zip(dofa_rows, panel_mask) if keep]
        panel_b = [row for row, keep in zip(resnet_rows, panel_mask) if keep]
        groups = [str(row["region_superclass"]) for row in panel_a]
        clusters = [str(row["site_id"]) for row in panel_a]
        band_a = simultaneous_group_risk_band(
            [float(row["risk"]) for row in panel_a],
            groups,
            clusters,
            confidence_level=protocol.confidence_level,
            n_bootstrap=n_bootstrap,
            seed=int(seed),
            risk_bounds=(0.0, 1.0),
            min_clusters_per_group=30,
        )
        band_b = simultaneous_group_risk_band(
            [float(row["risk"]) for row in panel_b],
            groups,
            clusters,
            confidence_level=protocol.confidence_level,
            n_bootstrap=n_bootstrap,
            seed=int(seed),
            risk_bounds=(0.0, 1.0),
            min_clusters_per_group=30,
        )
        band_delta = simultaneous_group_risk_band(
            [
                float(row_a["risk"]) - float(row_b["risk"])
                for row_a, row_b in zip(panel_a, panel_b)
            ],
            groups,
            clusters,
            confidence_level=protocol.confidence_level,
            n_bootstrap=n_bootstrap,
            seed=int(seed),
            risk_bounds=(-1.0, 1.0),
            min_clusters_per_group=30,
        )
        if not (
            band_a.validity.value
            == band_b.validity.value
            == band_delta.validity.value
            == "valid"
        ):
            raise FmowSuperclassPostprocessError(
                f"Six-cell simultaneous band is invalid for seed {seed}."
            )
        estimates_a, lower_a, upper_a = (
            dict(band_a.estimates),
            dict(band_a.lower),
            dict(band_a.upper),
        )
        estimates_b, lower_b, upper_b = (
            dict(band_b.estimates),
            dict(band_b.lower),
            dict(band_b.upper),
        )
        estimates_delta, lower_delta, upper_delta = (
            dict(band_delta.estimates),
            dict(band_delta.lower),
            dict(band_delta.upper),
        )
        cluster_support = dict(band_delta.clusters_per_group)
        sample_support: dict[str, int] = defaultdict(int)
        for group in groups:
            sample_support[group] += 1
        for group in sorted(frozen_panel):
            panel_rows.append(
                {
                    "seed": int(seed),
                    "region_superclass": group,
                    "sample_count": sample_support[group],
                    "independent_site_count": cluster_support[group],
                    "dofav2_risk": estimates_a[group],
                    "dofav2_simultaneous_ci_low": lower_a[group],
                    "dofav2_simultaneous_ci_high": upper_a[group],
                    "resnet50_risk": estimates_b[group],
                    "resnet50_simultaneous_ci_low": lower_b[group],
                    "resnet50_simultaneous_ci_high": upper_b[group],
                    "paired_risk_difference_definition": (
                        "risk(DOFAv2)-risk(ResNet50)"
                    ),
                    "paired_risk_difference": estimates_delta[group],
                    "paired_simultaneous_ci_low": lower_delta[group],
                    "paired_simultaneous_ci_high": upper_delta[group],
                }
            )

    paired_path = output / "same_seed_common_support_paired_geobwer.csv"
    partial_path = output / "fixed_universe_partial_identification.csv"
    panel_path = output / "six_cell_simultaneous_paired_risk.csv"
    coverage_path = output / "axis_support_coverage.csv"
    write_csv(paired_path, paired_rows)
    write_csv(partial_path, partial_rows)
    write_csv(panel_path, panel_rows)
    coverage_rows: list[dict[str, Any]] = []
    for row in feasibility_summaries:
        axis = str(row["axis"])
        fixed = int(row["fixed_universe_cell_count"])
        supported = int(row["descriptive_supported_cell_count"])
        sample_fraction = float(row["descriptive_supported_sample_fraction"])
        coverage_rows.append(
            {
                "axis": axis,
                "fixed_universe_cell_count": fixed,
                "supported_universe_cell_count": supported,
                "supported_cell_fraction": supported / fixed,
                "excluded_equal_cell_deployment_mass": 1.0
                - supported / fixed,
                "supported_sample_fraction": sample_fraction,
                "excluded_sample_fraction": 1.0 - sample_fraction,
                "axis_role": freeze["axis_roles"][axis][
                    "supported_universe_role"
                ],
            }
        )
    write_csv(coverage_path, coverage_rows)
    source_after = {
        key: value for key, value in evidence_sources.items()
    }
    manifest_path = output / "postprocess_manifest.json"
    manifest = {
        "schema": "geobwer.fmow.superclass_frozen_postprocess.v1",
        "formal_evidence": True,
        "source_artifacts_modified": False,
        "axis_role_freeze_sha256": file_sha256(freeze_path),
        "feasibility_contract_hash": feasibility_manifest["contract_hash"],
        "taxonomy_id": taxonomy["taxonomy_id"],
        "taxonomy_sha256": file_sha256(taxonomy_source),
        "geography_contract_hash": geography["contract_hash"],
        "base_protocol_hash": protocol.signature,
        "metric_version": protocol.metric_version,
        "region_supported_protocol_hash": region_protocol.signature,
        "country_supported_protocol_hash": country_protocol.signature,
        "seeds": [int(seed) for seed in seeds],
        "bootstrap_replicates": int(n_bootstrap),
        "comparison_direction": "DOFAv2_minus_ResNet50",
        "source_sha256": source_after,
        "artifacts": {
            "paired_geobwer": {
                "filename": paired_path.name,
                "sha256": file_sha256(paired_path),
            },
            "partial_identification": {
                "filename": partial_path.name,
                "sha256": file_sha256(partial_path),
            },
            "six_cell_panel": {
                "filename": panel_path.name,
                "sha256": file_sha256(panel_path),
            },
            "axis_support_coverage": {
                "filename": coverage_path.name,
                "sha256": file_sha256(coverage_path),
            },
        },
        "claim_limits": {
            "region_superclass": freeze["axis_roles"]["region_superclass"][
                "required_wording"
            ],
            "country_superclass": freeze["axis_roles"]["country_superclass"][
                "required_wording"
            ],
            "six_cell_panel_is_complete_axis": False,
        },
    }
    manifest["contract_hash"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    persist_output(
        output,
        persistent_output_dir,
        label="fmow-superclass-frozen-postprocess-complete",
    )
    return {
        "paired_geobwer": paired_path,
        "partial_identification": partial_path,
        "six_cell_panel": panel_path,
        "axis_support_coverage": coverage_path,
        "manifest": manifest_path,
    }


__all__ = [
    "FmowSuperclassPostprocessError",
    "_sharp_fixed_universe_bounds",
    "run_fmow_superclass_postprocess",
]
