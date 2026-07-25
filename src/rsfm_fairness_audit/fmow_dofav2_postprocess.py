from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.fmow_geography_contract import (
    FmowGeographyContractError,
    geography_assignment_hash,
    validate_fmow_geography_contract,
)
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import read_csv_rows, write_csv


class FmowDOFAv2PostprocessError(RuntimeError):
    """Raised when existing DOFA evidence cannot be safely overlaid."""


def read_axis_geobwer_summary(
    path: str | Path,
    *,
    axis: str,
) -> dict[str, str]:
    rows = read_csv_rows(path)
    match = next(
        (row for row in rows if str(row.get("axis") or "") == axis),
        None,
    )
    if match is None:
        raise FmowDOFAv2PostprocessError(
            f"GeoBWER summary has no axis={axis!r}: {path}"
        )
    return {
        "geobwer": str(match.get("bwer") or match.get("geobwer") or ""),
        "validity": str(match.get("validity") or ""),
        "ci_low": str(match.get("ci_low") or match.get("geobwer_ci_low") or ""),
        "ci_high": str(
            match.get("ci_high") or match.get("geobwer_ci_high") or ""
        ),
        "lower_confidence_bound": str(
            match.get("lower_confidence_bound")
            or match.get("geobwer_lcb")
            or ""
        ),
        "protocol_hash": str(match.get("protocol_hash") or ""),
        "metric_version": str(match.get("metric_version") or ""),
    }


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FmowDOFAv2PostprocessError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FmowDOFAv2PostprocessError(f"{label} is not a JSON object: {path}")
    return value


def _source_digest(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in paths}


def postprocess_fmow_dofav2_provenance(
    source_root: str | Path,
    geography_contract: str | Path,
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (42, 73, 101),
) -> dict[str, Path]:
    """Create a provenance overlay without changing formal DOFA evidence."""

    source = Path(source_root)
    output = Path(output_dir)
    contract_path = Path(geography_contract)
    if source.resolve() == output.resolve():
        raise FmowDOFAv2PostprocessError(
            "The provenance overlay must use a new output directory."
        )
    if output.exists() and any(output.iterdir()):
        raise FmowDOFAv2PostprocessError(
            f"Refusing to overwrite a non-empty overlay directory: {output}"
        )
    try:
        contract = validate_fmow_geography_contract(
            contract_path,
            require_formal=True,
        )
    except FmowGeographyContractError as exc:
        raise FmowDOFAv2PostprocessError(str(exc)) from exc

    run_manifest_path = source / "run_manifest.json"
    formal_manifest_path = source / "formal_outputs" / "formal_output_manifest.json"
    formal_table_path = source / "formal_outputs" / "formal_audit_table.csv"
    seed_summary_path = source / "probe_seed_robustness.csv"
    run_manifest = _json_object(run_manifest_path, label="DOFA run manifest")
    formal_manifest = _json_object(
        formal_manifest_path,
        label="DOFA formal manifest",
    )
    metadata_sha = str(
        formal_manifest.get("dataset_lineage", {}).get("metadata_sha256") or ""
    )
    if metadata_sha != str(contract["metadata"]["sha256"]):
        raise FmowDOFAv2PostprocessError(
            "DOFA metadata SHA-256 does not match the geography contract."
        )
    formal_rows = read_csv_rows(formal_table_path)
    test_split = str(
        formal_manifest.get("dataset_lineage", {}).get("test_split")
        or formal_manifest.get("extra", {}).get("split")
        or "test"
    )
    expected_assignment = str(
        contract.get("assignment_hashes", {}).get(test_split) or ""
    )
    observed_assignment = geography_assignment_hash(formal_rows)
    if not expected_assignment or observed_assignment != expected_assignment:
        raise FmowDOFAv2PostprocessError(
            "DOFA test geography assignments do not match the contract."
        )

    source_paths = [
        run_manifest_path,
        formal_manifest_path,
        formal_table_path,
        seed_summary_path,
    ]
    original_seed_rows = read_csv_rows(seed_summary_path)
    rows_by_seed = {
        int(str(row["seed"])): dict(row)
        for row in original_seed_rows
        if str(row.get("seed") or "").strip()
    }
    seed_manifest_hashes: dict[str, str] = {}
    seed_summary_hashes: dict[str, str] = {}
    protocol_hash = str(formal_manifest.get("protocol_hash") or "")
    metric_version = str(
        formal_manifest.get("protocol", {}).get("metric_version") or ""
    )
    for seed in seeds:
        if int(seed) not in rows_by_seed:
            raise FmowDOFAv2PostprocessError(
                f"probe_seed_robustness.csv has no seed={seed}."
            )
        seed_root = source / "probe_seeds" / f"seed_{int(seed)}"
        seed_manifest_path = seed_root / "formal_outputs" / "formal_output_manifest.json"
        geobwer_path = seed_root / "geobwer_raw" / "geobwer_summary.csv"
        seed_manifest = _json_object(
            seed_manifest_path,
            label=f"seed={seed} formal manifest",
        )
        summary = read_axis_geobwer_summary(geobwer_path, axis="country")
        if str(
            seed_manifest.get("dataset_lineage", {}).get("metadata_sha256")
            or ""
        ) != metadata_sha:
            raise FmowDOFAv2PostprocessError(
                f"seed={seed} metadata lineage differs from the ensemble manifest."
            )
        if str(seed_manifest.get("protocol_hash") or "") != protocol_hash:
            raise FmowDOFAv2PostprocessError(
                f"seed={seed} protocol hash differs from the ensemble manifest."
            )
        if summary["protocol_hash"] != protocol_hash:
            raise FmowDOFAv2PostprocessError(
                f"seed={seed} GeoBWER protocol hash mismatch."
            )
        if summary["metric_version"] != metric_version:
            raise FmowDOFAv2PostprocessError(
                f"seed={seed} GeoBWER metric version mismatch."
            )
        row = rows_by_seed[int(seed)]
        row.pop("country_geobwer_lcb", None)
        row.pop("country_geobwer_ucb", None)
        row.update(
            {
                "country_geobwer": summary["geobwer"],
                "country_geobwer_validity": summary["validity"],
                "country_geobwer_ci_low": summary["ci_low"],
                "country_geobwer_ci_high": summary["ci_high"],
                "country_geobwer_lower_confidence_bound": summary[
                    "lower_confidence_bound"
                ],
            }
        )
        source_paths.extend((seed_manifest_path, geobwer_path))
        seed_manifest_hashes[str(seed)] = file_sha256(seed_manifest_path)
        seed_summary_hashes[str(seed)] = file_sha256(geobwer_path)

    before = _source_digest(source_paths)
    output.mkdir(parents=True, exist_ok=True)
    contract_copy = output / "geography_contract.json"
    shutil.copy2(contract_path, contract_copy)
    corrected_summary = output / "probe_seed_robustness_geobwer.csv"
    write_csv(
        corrected_summary,
        [rows_by_seed[int(seed)] for seed in seeds],
    )
    lineage_overlay = output / "dataset_lineage_overlay.json"
    lineage_overlay.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.dataset_lineage_overlay.v1",
                "source_dataset_lineage": formal_manifest.get(
                    "dataset_lineage", {}
                ),
                "geography_contract_hash": contract["contract_hash"],
                "geography_contract_file_sha256": file_sha256(contract_copy),
                "geography_mapping_artifact_sha256": contract[
                    "mapping_artifact"
                ]["sha256"],
                "country_code_policy_id": contract["country_code_policy"][
                    "policy_id"
                ],
                "resolved_region_rule": contract["resolved_region_rule"],
                "test_geography_assignment_hash": observed_assignment,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    after = _source_digest(source_paths)
    if before != after:
        raise FmowDOFAv2PostprocessError(
            "A source DOFA artifact changed during provenance postprocessing."
        )
    overlay_manifest = output / "postprocess_manifest.json"
    overlay_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.dofav2_provenance_overlay.v1",
                "formal_evidence_role": "lineage_overlay_only",
                "source_root": str(source),
                "source_run_schema": run_manifest.get("schema"),
                "source_artifacts_immutable": True,
                "predictions_modified": False,
                "risks_modified": False,
                "bootstrap_recomputed": False,
                "formal_manifests_rewritten": False,
                "protocol_hash": protocol_hash,
                "metric_version": metric_version,
                "geography_contract_hash": contract["contract_hash"],
                "source_artifacts": {
                    "run_manifest_sha256": file_sha256(run_manifest_path),
                    "formal_output_manifest_sha256": file_sha256(
                        formal_manifest_path
                    ),
                    "formal_audit_table_sha256": file_sha256(formal_table_path),
                    "probe_seed_robustness_sha256": file_sha256(
                        seed_summary_path
                    ),
                    "seed_formal_manifest_sha256": seed_manifest_hashes,
                    "seed_geobwer_summary_sha256": seed_summary_hashes,
                },
                "derived_artifacts": {
                    "geography_contract": contract_copy.name,
                    "dataset_lineage_overlay": lineage_overlay.name,
                    "probe_seed_robustness_geobwer": corrected_summary.name,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "geography_contract": contract_copy,
        "dataset_lineage_overlay": lineage_overlay,
        "probe_seed_robustness": corrected_summary,
        "postprocess_manifest": overlay_manifest,
    }


__all__ = [
    "FmowDOFAv2PostprocessError",
    "postprocess_fmow_dofav2_provenance",
    "read_axis_geobwer_summary",
]
