from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.fmow_formal_split import (
    FmowFormalSplitError,
    fmow_site_id,
)
from rsfm_fairness_audit.fmow_geography_contract import (
    FmowGeographyContractError,
    validate_fmow_geography_contract,
)
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import read_csv_rows, write_csv


TAXONOMY_SCHEMA = "geobwer.fmow.superclass_taxonomy.v1"
FEASIBILITY_SCHEMA = "geobwer.fmow.superclass_feasibility.v1"
DEFAULT_TAXONOMY = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "fmow"
    / "superclass_taxonomy_v1.json"
)
DEFAULT_MIN_SAMPLES = 20
DEFAULT_MIN_SITES = 2
DEFAULT_CONFIRMATORY_MIN_SITES = 30


class FmowSuperclassFeasibilityError(RuntimeError):
    """Raised when an fMoW superclass design scan is not reproducible."""


def _text(row: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = str(row.get(field) or "").strip()
        if value and value.casefold() not in {"nan", "none", "null"}:
            return value
    return ""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_fmow_superclass_taxonomy(
    path: str | Path = DEFAULT_TAXONOMY,
) -> tuple[dict[str, Any], dict[str, str]]:
    taxonomy_path = Path(path)
    if not taxonomy_path.is_file():
        raise FmowSuperclassFeasibilityError(
            f"Superclass taxonomy is missing: {taxonomy_path}"
        )
    value = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != TAXONOMY_SCHEMA:
        raise FmowSuperclassFeasibilityError(
            f"Unsupported superclass taxonomy schema: {value.get('schema')!r}"
        )
    superclasses = value.get("superclasses")
    if not isinstance(superclasses, dict) or not 8 <= len(superclasses) <= 12:
        raise FmowSuperclassFeasibilityError(
            "The frozen fMoW taxonomy must define 8-12 superclasses."
        )
    class_to_superclass: dict[str, str] = {}
    duplicates: list[str] = []
    for superclass, specification in superclasses.items():
        if not str(superclass).strip() or not isinstance(specification, dict):
            raise FmowSuperclassFeasibilityError(
                "Every superclass requires a non-empty name and object specification."
            )
        classes = specification.get("classes")
        if not isinstance(classes, list) or not classes:
            raise FmowSuperclassFeasibilityError(
                f"Superclass {superclass!r} has no classes."
            )
        for raw_class in classes:
            class_name = str(raw_class).strip()
            if not class_name:
                raise FmowSuperclassFeasibilityError(
                    f"Superclass {superclass!r} contains an empty class."
                )
            if class_name in class_to_superclass:
                duplicates.append(class_name)
            else:
                class_to_superclass[class_name] = str(superclass)
    if duplicates:
        raise FmowSuperclassFeasibilityError(
            f"Classes occur in multiple superclasses: {sorted(set(duplicates))}"
        )
    expected = int(value.get("expected_class_count") or 0)
    if expected <= 0 or len(class_to_superclass) != expected:
        raise FmowSuperclassFeasibilityError(
            "Superclass taxonomy class count does not match "
            f"expected_class_count={expected}: observed={len(class_to_superclass)}."
        )
    return value, class_to_superclass


def _resolved_site_id(row: Mapping[str, Any]) -> str:
    site = _text(row, "site_id")
    if site:
        return site
    try:
        return fmow_site_id(row)
    except FmowFormalSplitError as exc:
        raise FmowSuperclassFeasibilityError(str(exc)) from exc


def _support_status(
    *,
    sample_count: int,
    site_count: int,
    min_samples: int,
    min_sites: int,
    confirmatory_min_sites: int,
) -> str:
    if sample_count >= min_samples and site_count >= confirmatory_min_sites:
        return "confirmatory_supported"
    if sample_count >= min_samples and site_count >= min_sites:
        return "descriptive_supported"
    return "insufficient_support"


def _axis_summary(
    *,
    axis: str,
    cell_rows: Sequence[Mapping[str, Any]],
    selected_sample_count: int,
) -> dict[str, Any]:
    fixed_count = len(cell_rows)
    observed = [row for row in cell_rows if int(row["sample_count"]) > 0]
    descriptive = [
        row
        for row in cell_rows
        if row["support_status"]
        in {"descriptive_supported", "confirmatory_supported"}
    ]
    confirmatory = [
        row
        for row in cell_rows
        if row["support_status"] == "confirmatory_supported"
    ]
    descriptive_samples = sum(int(row["sample_count"]) for row in descriptive)
    confirmatory_samples = sum(int(row["sample_count"]) for row in confirmatory)
    observed_samples = [int(row["sample_count"]) for row in observed]
    return {
        "axis": axis,
        "fixed_universe_cell_count": fixed_count,
        "observed_cell_count": len(observed),
        "zero_support_cell_count": fixed_count - len(observed),
        "descriptive_supported_cell_count": len(descriptive),
        "descriptive_supported_cell_fraction": (
            len(descriptive) / fixed_count if fixed_count else 0.0
        ),
        "confirmatory_supported_cell_count": len(confirmatory),
        "confirmatory_supported_cell_fraction": (
            len(confirmatory) / fixed_count if fixed_count else 0.0
        ),
        "descriptive_supported_sample_count": descriptive_samples,
        "descriptive_supported_sample_fraction": (
            descriptive_samples / selected_sample_count
            if selected_sample_count
            else 0.0
        ),
        "confirmatory_supported_sample_count": confirmatory_samples,
        "confirmatory_supported_sample_fraction": (
            confirmatory_samples / selected_sample_count
            if selected_sample_count
            else 0.0
        ),
        "observed_cell_sample_min": min(observed_samples) if observed_samples else 0,
        "observed_cell_sample_median": (
            float(median(observed_samples)) if observed_samples else 0.0
        ),
        "observed_cell_sample_max": max(observed_samples) if observed_samples else 0,
        "axis_decision_status": "pending_preregistered_axis_freeze",
    }


def scan_fmow_superclass_feasibility(
    metadata_csv: str | Path,
    geography_contract: str | Path,
    output_dir: str | Path,
    *,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY,
    split: str = "test",
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_sites: int = DEFAULT_MIN_SITES,
    confirmatory_min_sites: int = DEFAULT_CONFIRMATORY_MIN_SITES,
) -> dict[str, Path]:
    """Scan metadata-only support for pre-registered fMoW superclass axes."""

    metadata_path = Path(metadata_csv)
    geography_contract_path = Path(geography_contract)
    taxonomy_source = Path(taxonomy_path)
    output = Path(output_dir)
    if not metadata_path.is_file():
        raise FmowSuperclassFeasibilityError(
            f"Clean fMoW metadata is missing: {metadata_path}"
        )
    if output.exists() and any(output.iterdir()):
        raise FmowSuperclassFeasibilityError(
            f"Refusing to overwrite a non-empty output directory: {output}"
        )
    if min_samples < 1 or min_sites < 1:
        raise FmowSuperclassFeasibilityError(
            "min_samples and min_sites must be positive."
        )
    if confirmatory_min_sites < min_sites:
        raise FmowSuperclassFeasibilityError(
            "confirmatory_min_sites must be at least min_sites."
        )
    taxonomy, class_to_superclass = load_fmow_superclass_taxonomy(
        taxonomy_source
    )
    try:
        geography_contract_value = validate_fmow_geography_contract(
            geography_contract_path,
            metadata_csv=metadata_path,
            require_formal=True,
        )
    except FmowGeographyContractError as exc:
        raise FmowSuperclassFeasibilityError(str(exc)) from exc
    rows = read_csv_rows(metadata_path)
    if not rows:
        raise FmowSuperclassFeasibilityError("Clean fMoW metadata is empty.")

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    observed_classes: set[str] = set()
    prepared: list[dict[str, str]] = []
    for row in rows:
        sample_id = _text(row, "sample_id")
        category = _text(row, "category", "class_label", "label")
        row_split = _text(row, "split")
        if not sample_id or not category or not row_split:
            raise FmowSuperclassFeasibilityError(
                "Every metadata row requires sample_id, split, and category."
            )
        if sample_id in seen_ids:
            duplicate_ids.add(sample_id)
        seen_ids.add(sample_id)
        observed_classes.add(category)
        if row_split != split:
            continue
        country = _text(row, "country")
        region = _text(row, "region", "un_region", "continent")
        if not country or not region:
            raise FmowSuperclassFeasibilityError(
                f"Selected split row {sample_id!r} has unresolved geography."
            )
        superclass = class_to_superclass.get(category)
        if superclass is None:
            raise FmowSuperclassFeasibilityError(
                f"Selected split class is absent from taxonomy: {category!r}"
            )
        prepared.append(
            {
                "sample_id": sample_id,
                "site_id": _resolved_site_id(row),
                "category": category,
                "superclass": superclass,
                "country": country,
                "region": region,
            }
        )
    if duplicate_ids:
        raise FmowSuperclassFeasibilityError(
            f"Duplicate sample_id values in metadata: {sorted(duplicate_ids)[:10]}"
        )
    taxonomy_classes = set(class_to_superclass)
    unknown = observed_classes - taxonomy_classes
    missing = taxonomy_classes - observed_classes
    if unknown or missing:
        raise FmowSuperclassFeasibilityError(
            "Metadata/taxonomy class universe mismatch: "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    if not prepared:
        raise FmowSuperclassFeasibilityError(
            f"No rows found for split={split!r}."
        )
    split_classes = {row["category"] for row in prepared}
    if split_classes != taxonomy_classes:
        raise FmowSuperclassFeasibilityError(
            f"split={split!r} does not contain the frozen 62-class universe; "
            f"missing={sorted(taxonomy_classes - split_classes)}."
        )

    superclasses = sorted(set(class_to_superclass.values()))
    axis_fields = {
        "country_superclass": "country",
        "region_superclass": "region",
    }
    all_cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for axis, geography_field in axis_fields.items():
        geography_levels = sorted(
            {row[geography_field] for row in prepared}
        )
        samples_by_cell: dict[tuple[str, str], int] = defaultdict(int)
        sites_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in prepared:
            key = (row[geography_field], row["superclass"])
            samples_by_cell[key] += 1
            sites_by_cell[key].add(row["site_id"])
        axis_cells: list[dict[str, Any]] = []
        for geography_value in geography_levels:
            for superclass in superclasses:
                key = (geography_value, superclass)
                sample_count = samples_by_cell.get(key, 0)
                site_count = len(sites_by_cell.get(key, set()))
                axis_cells.append(
                    {
                        "axis": axis,
                        "geography": geography_value,
                        "superclass": superclass,
                        "sample_count": sample_count,
                        "independent_site_count": site_count,
                        "support_status": _support_status(
                            sample_count=sample_count,
                            site_count=site_count,
                            min_samples=min_samples,
                            min_sites=min_sites,
                            confirmatory_min_sites=confirmatory_min_sites,
                        ),
                    }
                )
        all_cells.extend(axis_cells)
        summaries.append(
            _axis_summary(
                axis=axis,
                cell_rows=axis_cells,
                selected_sample_count=len(prepared),
            )
        )

    output.mkdir(parents=True, exist_ok=True)
    taxonomy_snapshot = output / "superclass_taxonomy.json"
    shutil.copy2(taxonomy_source, taxonomy_snapshot)
    mapping_path = output / "class_to_superclass.csv"
    write_csv(
        mapping_path,
        [
            {
                "class_label": class_name,
                "superclass": class_to_superclass[class_name],
            }
            for class_name in sorted(class_to_superclass)
        ],
    )
    cells_path = output / "superclass_feasibility_cells.csv"
    summary_path = output / "superclass_feasibility_summary.csv"
    write_csv(cells_path, all_cells)
    write_csv(summary_path, summaries)

    assignment_payload = sorted(
        prepared,
        key=lambda row: row["sample_id"],
    )
    manifest: dict[str, Any] = {
        "schema": FEASIBILITY_SCHEMA,
        "formal_evidence": False,
        "evidence_role": "metadata_only_design_feasibility",
        "model_outputs_used": False,
        "performance_outcomes_used": False,
        "metadata": {
            "filename": metadata_path.name,
            "sha256": file_sha256(metadata_path),
            "row_count": len(rows),
            "selected_split": split,
            "selected_row_count": len(prepared),
            "selected_assignment_hash": _canonical_hash(assignment_payload),
        },
        "taxonomy": {
            "taxonomy_id": taxonomy["taxonomy_id"],
            "schema": taxonomy["schema"],
            "sha256": file_sha256(taxonomy_source),
            "class_count": len(class_to_superclass),
            "superclass_count": len(superclasses),
        },
        "geography_contract": {
            "filename": geography_contract_path.name,
            "file_sha256": file_sha256(geography_contract_path),
            "contract_hash": geography_contract_value["contract_hash"],
            "mapping_artifact_type": geography_contract_value[
                "mapping_artifact_type"
            ],
            "metadata_sha256": geography_contract_value["metadata"]["sha256"],
            "assignment_hashes": geography_contract_value[
                "assignment_hashes"
            ],
            "unresolved_country_row_count": geography_contract_value[
                "unresolved_country_row_count"
            ],
            "rows_dropped": geography_contract_value["rows_dropped"],
        },
        "support_contract": {
            "min_samples_per_cell": min_samples,
            "min_sites_per_cell": min_sites,
            "confirmatory_min_sites_per_cell": confirmatory_min_sites,
            "site_definition": (
                "metadata.site_id, otherwise category-scoped "
                "site_id=category|location_id"
            ),
            "fixed_universe": (
                "observed geography levels in selected split x all "
                "pre-registered superclasses"
            ),
        },
        "used_metadata_fields": [
            "sample_id",
            "split",
            "category",
            "site_id_or_location_id",
            "country",
            "first_nonempty(region,un_region,continent)",
        ],
        "summaries": summaries,
        "artifacts": {
            "taxonomy_snapshot": {
                "filename": taxonomy_snapshot.name,
                "sha256": file_sha256(taxonomy_snapshot),
            },
            "class_to_superclass": {
                "filename": mapping_path.name,
                "sha256": file_sha256(mapping_path),
            },
            "cells": {
                "filename": cells_path.name,
                "sha256": file_sha256(cells_path),
            },
            "summary": {
                "filename": summary_path.name,
                "sha256": file_sha256(summary_path),
            },
        },
    }
    report_path = output / "superclass_feasibility_report.md"
    lines = [
        "# fMoW Superclass Feasibility Scan",
        "",
        "This is metadata-only design evidence. It does not use model "
        "predictions, losses, GeoBWER values, or uncertainty outputs.",
        "",
        f"- Metadata SHA-256: `{manifest['metadata']['sha256']}`",
        f"- Taxonomy SHA-256: `{manifest['taxonomy']['sha256']}`",
        f"- Selected split: `{split}` ({len(prepared)} samples)",
        f"- Minimum support: {min_samples} samples and {min_sites} sites",
        f"- Confirmatory support: at least {confirmatory_min_sites} sites",
        "",
        "## Axis summary",
        "",
        "| Axis | Fixed cells | Zero cells | Descriptive cells | "
        "Confirmatory cells | Descriptive sample coverage | "
        "Confirmatory sample coverage | Decision status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {axis} | {fixed_universe_cell_count} | "
            "{zero_support_cell_count} | "
            "{descriptive_supported_cell_count} | "
            "{confirmatory_supported_cell_count} | "
            "{descriptive_supported_sample_fraction:.1%} | "
            "{confirmatory_supported_sample_fraction:.1%} | "
            "{axis_decision_status} |".format(**summary)
        )
    lines.extend(
        [
            "",
            "The scan does not automatically promote an axis. Final "
            "confirmatory, exploratory, or omitted roles must be frozen from "
            "these support facts before inspecting ResNet-50 comparison results.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest["artifacts"]["report"] = {
        "filename": report_path.name,
        "sha256": file_sha256(report_path),
    }
    manifest["contract_hash"] = _canonical_hash(manifest)
    manifest_path = output / "superclass_feasibility_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "taxonomy_snapshot": taxonomy_snapshot,
        "class_to_superclass": mapping_path,
        "cells": cells_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "report": report_path,
    }


__all__ = [
    "DEFAULT_CONFIRMATORY_MIN_SITES",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_MIN_SITES",
    "DEFAULT_TAXONOMY",
    "FEASIBILITY_SCHEMA",
    "TAXONOMY_SCHEMA",
    "FmowSuperclassFeasibilityError",
    "load_fmow_superclass_taxonomy",
    "scan_fmow_superclass_feasibility",
]
