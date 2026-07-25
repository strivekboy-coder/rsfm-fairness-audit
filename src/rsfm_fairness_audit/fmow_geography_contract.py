from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.fmow_geography import read_country_region_map
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import read_csv_rows


GEOGRAPHY_CONTRACT_SCHEMA = "geobwer.fmow.geography_contract.v1"
FMOW_REMAP_AUDIT_EXPECTED_ROWS = 720
DEFAULT_CODE_POLICY = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "fmow"
    / "geography_code_policy_v1.json"
)


class FmowGeographyContractError(RuntimeError):
    """Raised when an fMoW geography contract is invalid or incompatible."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _resolved_region(row: Mapping[str, Any]) -> str:
    return str(
        row.get("region")
        or row.get("un_region")
        or row.get("continent")
        or ""
    ).strip()


def geography_assignment_hash(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the actual sample-to-country/region assignment used by GeoBWER."""

    assignments: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise FmowGeographyContractError(
                "Every geography assignment requires a non-empty sample_id."
            )
        if sample_id in seen:
            raise FmowGeographyContractError(
                f"Duplicate sample_id in geography assignment: {sample_id}"
            )
        seen.add(sample_id)
        assignments.append(
            {
                "sample_id": sample_id,
                "country": str(row.get("country") or "").strip(),
                "region": _resolved_region(row),
            }
        )
    assignments.sort(key=lambda row: row["sample_id"])
    return hashlib.sha256(
        json.dumps(
            assignments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FmowGeographyContractError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FmowGeographyContractError(f"{label} must be a JSON object: {path}")
    return value


def _country_inventory(
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    recognized = {
        str(key): dict(value)
        for key, value in dict(policy.get("recognized_project_codes", {})).items()
    }
    unresolved = {
        str(key): str(value)
        for key, value in dict(policy.get("requires_explicit_resolution", {})).items()
    }
    placeholders = {
        str(value).strip().casefold()
        for value in policy.get("invalid_placeholders", [])
    }
    counts: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        raw = str(row.get("country") or "").strip()
        entry = counts.setdefault(
            raw or "<missing>",
            {
                "value": raw or "<missing>",
                "count": 0,
                "by_split": {},
                "classification": "",
                "policy": {},
            },
        )
        entry["count"] += 1
        split = str(row.get("split") or "<missing>").strip() or "<missing>"
        entry["by_split"][split] = entry["by_split"].get(split, 0) + 1
        if raw in recognized:
            entry["classification"] = "accepted_project_code"
            entry["policy"] = recognized[raw]
        elif raw in unresolved:
            entry["classification"] = "requires_explicit_resolution"
            entry["policy"] = {"reason": unresolved[raw]}
        elif raw.casefold() in placeholders:
            entry["classification"] = "invalid_placeholder"
        elif re.fullmatch(r"[A-Z]{3}", raw):
            entry["classification"] = "uppercase_alpha3_syntax"
        else:
            entry["classification"] = "invalid_or_unrecognized_syntax"
    for entry in counts.values():
        if entry["classification"] in {
            "requires_explicit_resolution",
            "invalid_placeholder",
            "invalid_or_unrecognized_syntax",
        }:
            errors.append(
                f"country={entry['value']!r} requires explicit resolution "
                f"(count={entry['count']})"
            )
    return sorted(counts.values(), key=lambda item: item["value"]), errors


def _distribution(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "").strip() or "<missing>"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _truthy(value: Any) -> bool:
    return str(value or "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "y",
        "mapped",
        "remapped",
    }


def _country_region_map_contract(
    mapping_path: Path,
    metadata_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    country_map, mapping_warnings = read_country_region_map(mapping_path)
    errors: list[str] = []
    if not country_map:
        errors.append(
            "mapping artifact has no readable country-to-geography records."
        )
    mapped_metadata_rows = 0
    conflicts: list[dict[str, str]] = []
    for row in metadata_rows:
        country = str(row.get("country") or "").strip()
        mapped = country_map.get(country.casefold())
        if mapped is None:
            continue
        mapped_metadata_rows += 1
        for field in ("continent", "un_region", "region"):
            expected = str(mapped.get(field) or "").strip()
            observed = str(row.get(field) or "").strip()
            if expected and observed and expected != observed:
                conflicts.append(
                    {
                        "type": "field_conflict",
                        "sample_id": str(row.get("sample_id") or ""),
                        "country": country,
                        "field": field,
                        "metadata_value": observed,
                        "mapping_value": expected,
                    }
                )
    if conflicts:
        errors.append(
            f"{len(conflicts)} metadata geography values conflict "
            "with the mapping artifact."
        )
    return (
        {
            "mapping_artifact_type": "country_region_map",
            "country_record_count": len(country_map),
            "mapped_metadata_rows": mapped_metadata_rows,
            "unmapped_metadata_rows": len(metadata_rows)
            - mapped_metadata_rows,
            "warnings": mapping_warnings,
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
        },
        errors,
    )


def _sample_assignment_audit_contract(
    audit_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    conflicts: list[dict[str, Any]] = []
    required_fields = (
        "sample_id",
        "original_country",
        "mapped_country",
        "mapped_continent",
        "mapped_un_region",
        "mapped_region",
        "mapping_source",
    )
    if len(audit_rows) != FMOW_REMAP_AUDIT_EXPECTED_ROWS:
        errors.append(
            "sample_assignment_audit must contain exactly "
            f"{FMOW_REMAP_AUDIT_EXPECTED_ROWS} rows; observed={len(audit_rows)}."
        )

    metadata_by_id: dict[str, Mapping[str, Any]] = {}
    metadata_duplicate_ids: set[str] = set()
    for row in metadata_rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id in metadata_by_id:
            metadata_duplicate_ids.add(sample_id)
        elif sample_id:
            metadata_by_id[sample_id] = row
    for sample_id in sorted(metadata_duplicate_ids):
        conflicts.append(
            {
                "type": "metadata_duplicate_sample_id",
                "sample_id": sample_id,
            }
        )

    audit_by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for index, row in enumerate(audit_rows, start=2):
        for field in required_fields:
            if not str(row.get(field) or "").strip():
                conflicts.append(
                    {
                        "type": "missing_required_field",
                        "row": index,
                        "sample_id": str(row.get("sample_id") or ""),
                        "field": field,
                    }
                )
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if sample_id in audit_by_id:
            duplicate_ids.add(sample_id)
        else:
            audit_by_id[sample_id] = row
    for sample_id in sorted(duplicate_ids):
        conflicts.append(
            {
                "type": "duplicate_audit_sample_id",
                "sample_id": sample_id,
            }
        )

    matched_ids: set[str] = set()
    field_pairs = (
        ("mapped_country", "country"),
        ("mapped_continent", "continent"),
        ("mapped_un_region", "un_region"),
        ("mapped_region", "region"),
    )
    audit_columns = set(audit_rows[0]) if audit_rows else set()
    for sample_id, audit_row in audit_by_id.items():
        metadata_row = metadata_by_id.get(sample_id)
        if metadata_row is None:
            conflicts.append(
                {
                    "type": "audit_sample_absent_from_metadata",
                    "sample_id": sample_id,
                }
            )
            continue
        matched_ids.add(sample_id)
        for audit_field, metadata_field in field_pairs:
            audit_value = str(audit_row.get(audit_field) or "").strip()
            metadata_value = str(metadata_row.get(metadata_field) or "").strip()
            if audit_value != metadata_value:
                conflicts.append(
                    {
                        "type": "field_conflict",
                        "sample_id": sample_id,
                        "audit_field": audit_field,
                        "metadata_field": metadata_field,
                        "audit_value": audit_value,
                        "metadata_value": metadata_value,
                    }
                )
        for field in ("split", "site_id"):
            if field not in audit_columns:
                continue
            audit_value = str(audit_row.get(field) or "").strip()
            metadata_value = str(metadata_row.get(field) or "").strip()
            if not audit_value:
                conflicts.append(
                    {
                        "type": "missing_optional_contract_field_value",
                        "sample_id": sample_id,
                        "field": field,
                    }
                )
            elif audit_value != metadata_value:
                conflicts.append(
                    {
                        "type": "field_conflict",
                        "sample_id": sample_id,
                        "audit_field": field,
                        "metadata_field": field,
                        "audit_value": audit_value,
                        "metadata_value": metadata_value,
                    }
                )

    marker_field = next(
        (
            field
            for field in ("geography_remapped", "mapping_applied", "remapped")
            if any(field in row for row in metadata_rows)
        ),
        None,
    )
    if marker_field is None:
        target_ids = set(audit_by_id)
        target_scope_rule = (
            "sample_assignment_audit sample_ids define the remapped target "
            "population; metadata rows outside this set are not required."
        )
    else:
        target_ids = {
            str(row.get("sample_id") or "").strip()
            for row in metadata_rows
            if _truthy(row.get(marker_field))
        }
        target_scope_rule = (
            f"metadata field {marker_field!r} defines the remapped target "
            "population; all marked rows must occur in the audit."
        )
        for sample_id in sorted(target_ids - set(audit_by_id)):
            conflicts.append(
                {
                    "type": "metadata_target_missing_from_audit",
                    "sample_id": sample_id,
                    "marker_field": marker_field,
                }
            )
        for sample_id in sorted(set(audit_by_id) - target_ids):
            conflicts.append(
                {
                    "type": "audit_sample_not_marked_as_target",
                    "sample_id": sample_id,
                    "marker_field": marker_field,
                }
            )

    nearest_distances: list[float] = []
    for index, row in enumerate(audit_rows, start=2):
        raw = str(row.get("nearest_boundary_distance_km") or "").strip()
        if not raw:
            continue
        try:
            nearest_distances.append(float(raw))
        except ValueError:
            conflicts.append(
                {
                    "type": "invalid_nearest_boundary_distance",
                    "row": index,
                    "sample_id": str(row.get("sample_id") or ""),
                    "value": raw,
                }
            )

    if duplicate_ids:
        errors.append(
            f"sample_assignment_audit contains {len(duplicate_ids)} duplicate "
            "sample_id values."
        )
    if metadata_duplicate_ids:
        errors.append(
            f"final metadata contains {len(metadata_duplicate_ids)} duplicate "
            "sample_id values."
        )
    if conflicts:
        errors.append(
            f"sample_assignment_audit has {len(conflicts)} validation conflicts."
        )
    audit_ids = set(audit_by_id)
    return (
        {
            "mapping_artifact_type": "sample_assignment_audit",
            "audit_row_count": len(audit_rows),
            "expected_audit_row_count": FMOW_REMAP_AUDIT_EXPECTED_ROWS,
            "unique_sample_count": len(audit_ids),
            "matched_sample_count": len(matched_ids),
            "conflict_count": len(conflicts),
            "conflicts": conflicts,
            "original_country_distribution": _distribution(
                audit_rows, "original_country"
            ),
            "mapped_country_distribution": _distribution(
                audit_rows, "mapped_country"
            ),
            "mapping_source_distribution": _distribution(
                audit_rows, "mapping_source"
            ),
            "nearest_boundary_rows": len(nearest_distances),
            "nearest_boundary_max_distance_km": (
                max(nearest_distances) if nearest_distances else None
            ),
            "target_scope_rule": target_scope_rule,
            "metadata_target_marker_field": marker_field,
            "metadata_target_sample_count": len(target_ids),
            "metadata_target_missing_from_audit_count": len(
                target_ids - audit_ids
            ),
            "metadata_target_missing_from_audit": sorted(
                target_ids - audit_ids
            ),
            "metadata_rows_outside_audit_count": len(metadata_rows)
            - len(matched_ids),
            "metadata_rows_outside_audit_are_required": False,
            "rows_dropped": 0,
        },
        errors,
    )


def build_fmow_geography_contract(
    metadata_csv: str | Path,
    mapping_artifact: str | Path,
    *,
    mapping_source_name: str,
    mapping_source_version: str,
    mapping_source_url: str,
    code_policy: str | Path = DEFAULT_CODE_POLICY,
) -> dict[str, Any]:
    """Build a deterministic contract without modifying the metadata."""

    metadata_path = Path(metadata_csv)
    mapping_path = Path(mapping_artifact)
    policy_path = Path(code_policy)
    if not metadata_path.is_file():
        raise FmowGeographyContractError(
            f"Final fMoW metadata is missing: {metadata_path}"
        )
    if not mapping_path.is_file():
        raise FmowGeographyContractError(
            f"Geography mapping artifact is missing: {mapping_path}"
        )
    source = {
        "name": str(mapping_source_name).strip(),
        "version": str(mapping_source_version).strip(),
        "url": str(mapping_source_url).strip(),
    }
    if not all(source.values()):
        raise FmowGeographyContractError(
            "mapping_source_name, mapping_source_version, and "
            "mapping_source_url are required."
        )
    policy = _load_json_object(policy_path, label="Country-code policy")
    if policy.get("schema") != "geobwer.fmow.geography_code_policy.v1":
        raise FmowGeographyContractError(
            f"Unsupported country-code policy schema: {policy.get('schema')!r}"
        )
    rows = read_csv_rows(metadata_path)
    if not rows:
        raise FmowGeographyContractError("Final fMoW metadata is empty.")

    errors: list[str] = []
    required = ("sample_id", "split", "country")
    missing_required = {
        field: sum(not str(row.get(field) or "").strip() for row in rows)
        for field in required
    }
    if any(missing_required.values()):
        errors.append(f"required fields missing: {missing_required}")
    missing_region = sum(not _resolved_region(row) for row in rows)
    if missing_region:
        errors.append(
            f"{missing_region} rows have no region/un_region/continent fallback."
        )
    inventory, country_errors = _country_inventory(rows, policy)
    errors.extend(country_errors)
    mapping_rows = read_csv_rows(mapping_path)
    mapping_columns = set(mapping_rows[0]) if mapping_rows else set()
    if {"sample_id", "mapped_country"}.issubset(mapping_columns):
        mapping_details, mapping_errors = _sample_assignment_audit_contract(
            mapping_rows,
            rows,
        )
    else:
        mapping_details, mapping_errors = _country_region_map_contract(
            mapping_path,
            rows,
        )
    errors.extend(mapping_errors)

    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        split = str(row.get("split") or "").strip()
        by_split.setdefault(split or "<missing>", []).append(dict(row))
    assignment_hashes: dict[str, str] = {}
    for split, split_rows in sorted(by_split.items()):
        try:
            assignment_hashes[split] = geography_assignment_hash(split_rows)
        except FmowGeographyContractError as exc:
            errors.append(f"split={split}: {exc}")

    provenance_counts: dict[str, dict[str, int]] = {}
    for field in ("continent", "un_region", "region"):
        key = f"{field}_provenance"
        values: dict[str, int] = {}
        for row in rows:
            provenance = str(row.get(key) or "").strip() or "<not_recorded>"
            values[provenance] = values.get(provenance, 0) + 1
        provenance_counts[key] = dict(sorted(values.items()))

    payload: dict[str, Any] = {
        "schema": GEOGRAPHY_CONTRACT_SCHEMA,
        "formal_compatible": not errors,
        "errors": errors,
        "metadata": {
            "filename": metadata_path.name,
            "sha256": file_sha256(metadata_path),
            "row_count": len(rows),
            "split_counts": {
                split: len(split_rows)
                for split, split_rows in sorted(by_split.items())
            },
        },
        "mapping_artifact": {
            "filename": mapping_path.name,
            "sha256": file_sha256(mapping_path),
            "source": source,
            **mapping_details,
        },
        "mapping_artifact_type": mapping_details["mapping_artifact_type"],
        "country_code_policy": {
            "filename": policy_path.name,
            "sha256": file_sha256(policy_path),
            "policy_id": policy.get("policy_id"),
            "preserve_raw_country_values": bool(
                policy.get("preserve_raw_country_values")
            ),
            "authoritative_references": policy.get(
                "authoritative_references", []
            ),
        },
        "resolved_region_rule": "first_nonempty(region,un_region,continent)",
        "assignment_hashes": assignment_hashes,
        "country_inventory": inventory,
        "geography_provenance_counts": provenance_counts,
        "unresolved_country_row_count": sum(
            int(item["count"])
            for item in inventory
            if item["classification"]
            in {
                "requires_explicit_resolution",
                "invalid_placeholder",
                "invalid_or_unrecognized_syntax",
            }
        ),
        "rows_dropped": 0,
        "metadata_modified": False,
    }
    payload["contract_hash"] = _payload_hash(payload)
    return payload


def write_fmow_geography_contract(
    path: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(contract), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def validate_fmow_geography_contract(
    path: str | Path,
    *,
    metadata_csv: str | Path | None = None,
    mapping_artifact: str | Path | None = None,
    require_formal: bool = True,
) -> dict[str, Any]:
    contract_path = Path(path)
    contract = _load_json_object(contract_path, label="Geography contract")
    if contract.get("schema") != GEOGRAPHY_CONTRACT_SCHEMA:
        raise FmowGeographyContractError(
            f"Unsupported geography contract schema: {contract.get('schema')!r}"
        )
    expected_hash = str(contract.get("contract_hash") or "")
    payload = dict(contract)
    payload.pop("contract_hash", None)
    observed_hash = _payload_hash(payload)
    if not expected_hash or observed_hash != expected_hash:
        raise FmowGeographyContractError(
            "Geography contract hash mismatch; the contract was modified."
        )
    if require_formal and not bool(contract.get("formal_compatible")):
        raise FmowGeographyContractError(
            "Geography contract is not formal-compatible: "
            + "; ".join(str(value) for value in contract.get("errors", []))
        )
    if metadata_csv is not None:
        metadata_path = Path(metadata_csv)
        expected = str(contract.get("metadata", {}).get("sha256") or "")
        if not metadata_path.is_file() or file_sha256(metadata_path) != expected:
            raise FmowGeographyContractError(
                "Metadata SHA-256 does not match the geography contract."
            )
    if mapping_artifact is not None:
        mapping_path = Path(mapping_artifact)
        expected = str(
            contract.get("mapping_artifact", {}).get("sha256") or ""
        )
        if not mapping_path.is_file() or file_sha256(mapping_path) != expected:
            raise FmowGeographyContractError(
                "Mapping artifact SHA-256 does not match the geography contract."
            )
    return contract


def geography_contract_lineage(
    path: str | Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    contract_path = Path(path)
    return {
        "geography_contract_schema": contract["schema"],
        "geography_contract_hash": contract["contract_hash"],
        "geography_contract_file": contract_path.name,
        "geography_contract_file_sha256": file_sha256(contract_path),
        "geography_mapping_artifact_sha256": contract["mapping_artifact"][
            "sha256"
        ],
        "geography_mapping_source": contract["mapping_artifact"]["source"],
        "country_code_policy_id": contract["country_code_policy"]["policy_id"],
        "country_code_policy_sha256": contract["country_code_policy"]["sha256"],
        "resolved_region_rule": contract["resolved_region_rule"],
    }


__all__ = [
    "DEFAULT_CODE_POLICY",
    "GEOGRAPHY_CONTRACT_SCHEMA",
    "FmowGeographyContractError",
    "build_fmow_geography_contract",
    "geography_assignment_hash",
    "geography_contract_lineage",
    "validate_fmow_geography_contract",
    "write_fmow_geography_contract",
]
