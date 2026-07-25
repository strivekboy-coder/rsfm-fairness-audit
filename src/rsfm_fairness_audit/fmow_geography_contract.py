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
    country_map, mapping_warnings = read_country_region_map(mapping_path)
    if not country_map:
        errors.append(
            "mapping artifact has no readable country-to-geography records."
        )
    mapped_metadata_rows = 0
    mapping_conflicts: list[dict[str, str]] = []
    for row in rows:
        country = str(row.get("country") or "").strip()
        mapped = country_map.get(country.casefold())
        if mapped is None:
            continue
        mapped_metadata_rows += 1
        for field in ("continent", "un_region", "region"):
            expected = str(mapped.get(field) or "").strip()
            observed = str(row.get(field) or "").strip()
            if expected and observed and expected != observed:
                mapping_conflicts.append(
                    {
                        "sample_id": str(row.get("sample_id") or ""),
                        "country": country,
                        "field": field,
                        "metadata_value": observed,
                        "mapping_value": expected,
                    }
                )
    if mapping_conflicts:
        errors.append(
            f"{len(mapping_conflicts)} metadata geography values conflict "
            "with the mapping artifact."
        )

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
            "country_record_count": len(country_map),
            "mapped_metadata_rows": mapped_metadata_rows,
            "unmapped_metadata_rows": len(rows) - mapped_metadata_rows,
            "warnings": mapping_warnings,
            "conflicts": mapping_conflicts,
        },
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
