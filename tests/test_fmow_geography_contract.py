from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsfm_fairness_audit.fmow_dofav2_postprocess import (
    FmowDOFAv2PostprocessError,
    postprocess_fmow_dofav2_provenance,
    read_axis_geobwer_summary,
)
from rsfm_fairness_audit.fmow_geography_contract import (
    FmowGeographyContractError,
    build_fmow_geography_contract,
    validate_fmow_geography_contract,
    write_fmow_geography_contract,
)
from rsfm_fairness_audit.fmow_resnet50_campaign import (
    FmowResNet50CampaignConfig,
    FmowResNet50CampaignError,
    _prepare_geography_contract,
)
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import read_csv_rows, write_csv


def _metadata_rows() -> list[dict[str, str]]:
    return [
        {
            "sample_id": "train-1",
            "split": "train",
            "country": "DEU",
            "region": "Western Europe",
            "category": "airport",
        },
        {
            "sample_id": "calibration-1",
            "split": "calibration",
            "country": "DEU",
            "region": "Western Europe",
            "category": "airport",
        },
        {
            "sample_id": "test-1",
            "split": "test",
            "country": "KOS",
            "region": "Southern Europe",
            "category": "airport",
        },
    ]


def _contract(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict]:
    metadata = tmp_path / "metadata.csv"
    mapping = tmp_path / "mapping.csv"
    write_csv(metadata, _metadata_rows())
    write_csv(
        mapping,
        [
            {
                "country": "KOS",
                "region": "Southern Europe",
                "source": "project-curated UN-M49-compatible mapping",
            }
        ],
    )
    value = build_fmow_geography_contract(
        metadata,
        mapping,
        mapping_source_name="project-curated UN M49 mapping",
        mapping_source_version="2026-07-25",
        mapping_source_url="https://unstats.un.org/unsd/methodology/m49/",
    )
    path = write_fmow_geography_contract(
        tmp_path / "geography_contract.json",
        value,
    )
    return path, metadata, mapping, value


def test_contract_is_deterministic_and_records_kos_without_rewriting(
    tmp_path: Path,
) -> None:
    path, metadata, mapping, first = _contract(tmp_path)
    second = build_fmow_geography_contract(
        metadata,
        mapping,
        mapping_source_name="project-curated UN M49 mapping",
        mapping_source_version="2026-07-25",
        mapping_source_url="https://unstats.un.org/unsd/methodology/m49/",
    )
    assert first["formal_compatible"] is True
    assert first["contract_hash"] == second["contract_hash"]
    kos = next(
        item for item in first["country_inventory"] if item["value"] == "KOS"
    )
    assert kos["classification"] == "accepted_project_code"
    assert kos["policy"]["rewrite"] is False
    assert kos["policy"]["external_reference_code"] == "XKX"
    validated = validate_fmow_geography_contract(
        path,
        metadata_csv=metadata,
        mapping_artifact=mapping,
    )
    assert validated["contract_hash"] == first["contract_hash"]


def test_contract_rejects_unresolved_country_and_metadata_drift(
    tmp_path: Path,
) -> None:
    path, metadata, _mapping, _value = _contract(tmp_path)
    rows = _metadata_rows()
    rows[-1]["country"] = "CA-"
    invalid_metadata = tmp_path / "invalid.csv"
    write_csv(invalid_metadata, rows)
    invalid = build_fmow_geography_contract(
        invalid_metadata,
        tmp_path / "mapping.csv",
        mapping_source_name="project-curated UN M49 mapping",
        mapping_source_version="2026-07-25",
        mapping_source_url="https://unstats.un.org/unsd/methodology/m49/",
    )
    assert invalid["formal_compatible"] is False
    assert any("CA-" in error for error in invalid["errors"])
    with pytest.raises(FmowGeographyContractError, match="Metadata SHA-256"):
        validate_fmow_geography_contract(path, metadata_csv=invalid_metadata)
    with pytest.raises(FmowGeographyContractError, match="not formal-compatible"):
        invalid_path = write_fmow_geography_contract(
            tmp_path / "invalid_contract.json",
            invalid,
        )
        validate_fmow_geography_contract(invalid_path, require_formal=True)
    assert file_sha256(metadata) != file_sha256(invalid_metadata)


def test_contract_rejects_mapping_metadata_conflict(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    mapping = tmp_path / "mapping.csv"
    write_csv(metadata, _metadata_rows())
    write_csv(
        mapping,
        [{"country": "KOS", "region": "Western Europe"}],
    )
    contract = build_fmow_geography_contract(
        metadata,
        mapping,
        mapping_source_name="project-curated UN M49 mapping",
        mapping_source_version="2026-07-25",
        mapping_source_url="https://unstats.un.org/unsd/methodology/m49/",
    )
    assert contract["formal_compatible"] is False
    assert len(contract["mapping_artifact"]["conflicts"]) == 1
    assert any("conflict" in error for error in contract["errors"])


def _sample_assignment_rows(
    *,
    include_markers: bool = False,
    extra_marked_target: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metadata_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    for index in range(720):
        sample_id = f"remapped-{index:04d}"
        country = "USA" if index % 2 == 0 else "DEU"
        continent = "North America" if country == "USA" else "Europe"
        un_region = (
            "Northern America" if country == "USA" else "Western Europe"
        )
        region = un_region
        split = ("train", "calibration", "test")[index % 3]
        site_id = f"site-{index:04d}"
        metadata_row = {
            "sample_id": sample_id,
            "site_id": site_id,
            "split": split,
            "country": country,
            "continent": continent,
            "un_region": un_region,
            "region": region,
            "category": "airport",
        }
        if include_markers:
            metadata_row["geography_remapped"] = "true"
        metadata_rows.append(metadata_row)
        nearest = index % 100 == 0
        audit_rows.append(
            {
                "sample_id": sample_id,
                "site_id": site_id,
                "split": split,
                "original_country": (
                    "ambiguous_country" if index % 2 == 0 else "CA-"
                ),
                "mapped_country": country,
                "mapped_continent": continent,
                "mapped_un_region": un_region,
                "mapped_region": region,
                "mapping_source": (
                    "nearest_boundary" if nearest else "natural_earth_polygon"
                ),
                "polygon_point_latitude": "40.0",
                "polygon_point_longitude": "-75.0",
                "nearest_boundary_distance_km": "12.5" if nearest else "",
                "nearest_boundary_latitude": "40.1" if nearest else "",
                "nearest_boundary_longitude": "-75.1" if nearest else "",
                "natural_earth_admin": (
                    "United States of America"
                    if country == "USA"
                    else "Germany"
                ),
                "natural_earth_geounit": (
                    "United States of America"
                    if country == "USA"
                    else "Germany"
                ),
                "spatial_match_count": "0" if nearest else "1",
            }
        )
    for index in range(3):
        metadata_rows.append(
            {
                "sample_id": f"unchanged-{index}",
                "site_id": f"unchanged-site-{index}",
                "split": "test",
                "country": "FRA",
                "continent": "Europe",
                "un_region": "Western Europe",
                "region": "Western Europe",
                "category": "airport",
                **(
                    {"geography_remapped": "false"}
                    if include_markers
                    else {}
                ),
            }
        )
    if extra_marked_target:
        metadata_rows.append(
            {
                "sample_id": "marked-target-missing-from-audit",
                "site_id": "marked-site",
                "split": "test",
                "country": "FRA",
                "continent": "Europe",
                "un_region": "Western Europe",
                "region": "Western Europe",
                "category": "airport",
                "geography_remapped": "true",
            }
        )
    return metadata_rows, audit_rows


def _sample_assignment_contract(
    tmp_path: Path,
    *,
    metadata_rows: list[dict[str, str]] | None = None,
    audit_rows: list[dict[str, str]] | None = None,
) -> tuple[Path, Path, dict]:
    if metadata_rows is None or audit_rows is None:
        metadata_rows, audit_rows = _sample_assignment_rows()
    metadata = tmp_path / "metadata_sample_assignment.csv"
    mapping = tmp_path / "sample_assignment_audit.csv"
    write_csv(metadata, metadata_rows)
    write_csv(mapping, audit_rows)
    contract = build_fmow_geography_contract(
        metadata,
        mapping,
        mapping_source_name="Natural Earth sample remap audit",
        mapping_source_version="2026-07-25",
        mapping_source_url="https://www.naturalearthdata.com/",
    )
    return metadata, mapping, contract


def test_sample_assignment_audit_real_720_schema_is_strictly_validated(
    tmp_path: Path,
) -> None:
    metadata, mapping, contract = _sample_assignment_contract(tmp_path)
    details = contract["mapping_artifact"]
    assert contract["formal_compatible"] is True
    assert contract["mapping_artifact_type"] == "sample_assignment_audit"
    assert details["mapping_artifact_type"] == "sample_assignment_audit"
    assert details["audit_row_count"] == 720
    assert details["unique_sample_count"] == 720
    assert details["matched_sample_count"] == 720
    assert details["conflict_count"] == 0
    assert details["metadata_rows_outside_audit_count"] == 3
    assert details["metadata_rows_outside_audit_are_required"] is False
    assert details["metadata_target_missing_from_audit_count"] == 0
    assert details["original_country_distribution"] == {
        "CA-": 360,
        "ambiguous_country": 360,
    }
    assert details["mapped_country_distribution"] == {
        "DEU": 360,
        "USA": 360,
    }
    assert details["mapping_source_distribution"] == {
        "natural_earth_polygon": 712,
        "nearest_boundary": 8,
    }
    assert details["nearest_boundary_rows"] == 8
    assert details["nearest_boundary_max_distance_km"] == 12.5
    assert details["sha256"] == file_sha256(mapping)
    assert contract["unresolved_country_row_count"] == 0
    assert contract["rows_dropped"] == 0
    contract_path = write_fmow_geography_contract(
        tmp_path / "sample_assignment_contract.json",
        contract,
    )
    validated = validate_fmow_geography_contract(
        contract_path,
        metadata_csv=metadata,
        mapping_artifact=mapping,
    )
    assert validated["contract_hash"] == contract["contract_hash"]


@pytest.mark.parametrize(
    ("audit_field", "bad_value"),
    [
        ("mapped_country", "FRA"),
        ("mapped_region", "Northern Europe"),
    ],
)
def test_sample_assignment_audit_rejects_mapped_field_conflicts(
    tmp_path: Path,
    audit_field: str,
    bad_value: str,
) -> None:
    metadata_rows, audit_rows = _sample_assignment_rows()
    audit_rows[0][audit_field] = bad_value
    _metadata, _mapping, contract = _sample_assignment_contract(
        tmp_path,
        metadata_rows=metadata_rows,
        audit_rows=audit_rows,
    )
    assert contract["formal_compatible"] is False
    conflicts = contract["mapping_artifact"]["conflicts"]
    assert any(
        item.get("type") == "field_conflict"
        and item.get("audit_field") == audit_field
        for item in conflicts
    )


def test_sample_assignment_audit_rejects_duplicate_sample_id(
    tmp_path: Path,
) -> None:
    metadata_rows, audit_rows = _sample_assignment_rows()
    audit_rows[-1]["sample_id"] = audit_rows[0]["sample_id"]
    _metadata, _mapping, contract = _sample_assignment_contract(
        tmp_path,
        metadata_rows=metadata_rows,
        audit_rows=audit_rows,
    )
    assert contract["formal_compatible"] is False
    assert any(
        item.get("type") == "duplicate_audit_sample_id"
        for item in contract["mapping_artifact"]["conflicts"]
    )


def test_sample_assignment_audit_rejects_sample_absent_from_metadata(
    tmp_path: Path,
) -> None:
    metadata_rows, audit_rows = _sample_assignment_rows()
    audit_rows[-1]["sample_id"] = "not-in-final-metadata"
    _metadata, _mapping, contract = _sample_assignment_contract(
        tmp_path,
        metadata_rows=metadata_rows,
        audit_rows=audit_rows,
    )
    assert contract["formal_compatible"] is False
    assert any(
        item.get("type") == "audit_sample_absent_from_metadata"
        for item in contract["mapping_artifact"]["conflicts"]
    )


def test_sample_assignment_audit_requires_only_explicitly_marked_targets(
    tmp_path: Path,
) -> None:
    metadata_rows, audit_rows = _sample_assignment_rows(
        include_markers=True,
        extra_marked_target=True,
    )
    _metadata, _mapping, contract = _sample_assignment_contract(
        tmp_path,
        metadata_rows=metadata_rows,
        audit_rows=audit_rows,
    )
    details = contract["mapping_artifact"]
    assert contract["formal_compatible"] is False
    assert details["metadata_target_marker_field"] == "geography_remapped"
    assert details["metadata_target_missing_from_audit_count"] == 1
    assert details["metadata_target_missing_from_audit"] == [
        "marked-target-missing-from-audit"
    ]
    assert details["metadata_rows_outside_audit_count"] == 4
    assert any(
        item.get("type") == "metadata_target_missing_from_audit"
        for item in details["conflicts"]
    )


def _write_fake_dofa_source(
    root: Path,
    *,
    metadata_sha: str,
    protocol_hash: str,
    metric_version: str,
) -> None:
    (root / "formal_outputs").mkdir(parents=True)
    (root / "run_manifest.json").write_text(
        json.dumps({"schema": "geobwer.fmow_dofav2_campaign.v2"}),
        encoding="utf-8",
    )
    write_csv(
        root / "formal_outputs" / "formal_audit_table.csv",
        [
            {
                "sample_id": "test-1",
                "country": "KOS",
                "region": "Southern Europe",
            }
        ],
    )
    main_manifest = {
        "protocol_hash": protocol_hash,
        "protocol": {"metric_version": metric_version},
        "dataset_lineage": {
            "metadata_sha256": metadata_sha,
            "test_split": "test",
        },
    }
    (root / "formal_outputs" / "formal_output_manifest.json").write_text(
        json.dumps(main_manifest),
        encoding="utf-8",
    )
    write_csv(
        root / "probe_seed_robustness.csv",
        [
            {
                "seed": seed,
                "test_accuracy": 0.2,
                "country_geobwer_lcb": "",
                "country_geobwer_ucb": "",
            }
            for seed in (42, 73, 101)
        ],
    )
    for offset, seed in enumerate((42, 73, 101)):
        seed_root = root / "probe_seeds" / f"seed_{seed}"
        (seed_root / "formal_outputs").mkdir(parents=True)
        (seed_root / "geobwer_raw").mkdir(parents=True)
        (seed_root / "formal_outputs" / "formal_output_manifest.json").write_text(
            json.dumps(
                {
                    "protocol_hash": protocol_hash,
                    "dataset_lineage": {"metadata_sha256": metadata_sha},
                }
            ),
            encoding="utf-8",
        )
        write_csv(
            seed_root / "geobwer_raw" / "geobwer_summary.csv",
            [
                {
                    "axis": "country",
                    "validity": "descriptive_only",
                    "protocol_hash": protocol_hash,
                    "metric_version": metric_version,
                    "bwer": 0.20 + offset / 100,
                    "ci_low": 0.0,
                    "ci_high": 0.7,
                    "lower_confidence_bound": 0.0,
                }
            ],
        )


def test_dofa_cpu_overlay_repairs_seed_summary_without_mutating_source(
    tmp_path: Path,
) -> None:
    contract_path, metadata, _mapping, contract = _contract(tmp_path)
    source = tmp_path / "dofa"
    _write_fake_dofa_source(
        source,
        metadata_sha=file_sha256(metadata),
        protocol_hash="protocol-1",
        metric_version="geobwer_fractional_1.1",
    )
    source_table = source / "formal_outputs" / "formal_audit_table.csv"
    source_hash_before = file_sha256(source_table)
    artifacts = postprocess_fmow_dofav2_provenance(
        source,
        contract_path,
        tmp_path / "overlay",
    )
    assert file_sha256(source_table) == source_hash_before
    rows = read_csv_rows(artifacts["probe_seed_robustness"])
    assert [float(row["country_geobwer"]) for row in rows] == pytest.approx(
        [0.20, 0.21, 0.22]
    )
    assert all(
        row["country_geobwer_validity"] == "descriptive_only" for row in rows
    )
    assert all(
        row["country_geobwer_lower_confidence_bound"] == "0.0"
        for row in rows
    )
    assert all("country_geobwer_lcb" not in row for row in rows)
    assert all("country_geobwer_ucb" not in row for row in rows)
    overlay = json.loads(
        artifacts["postprocess_manifest"].read_text(encoding="utf-8")
    )
    assert overlay["source_artifacts_immutable"] is True
    assert overlay["predictions_modified"] is False
    assert overlay["geography_contract_hash"] == contract["contract_hash"]


def test_axis_reader_accepts_current_bwer_column(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    write_csv(
        path,
        [
            {
                "axis": "country",
                "bwer": 0.3,
                "validity": "descriptive_only",
                "ci_low": 0.0,
                "ci_high": 0.8,
                "lower_confidence_bound": 0.0,
                "protocol_hash": "hash",
                "metric_version": "geobwer_fractional_1.1",
            }
        ],
    )
    result = read_axis_geobwer_summary(path, axis="country")
    assert result["geobwer"] == "0.3"
    assert result["lower_confidence_bound"] == "0.0"


def test_formal_resnet_requires_and_copies_matching_contract(
    tmp_path: Path,
) -> None:
    contract_path, metadata, _mapping, contract = _contract(tmp_path)
    missing = FmowResNet50CampaignConfig(
        metadata_csv=metadata,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "missing-output",
    )
    with pytest.raises(FmowResNet50CampaignError, match="geography-contract"):
        _prepare_geography_contract(missing, missing.output_dir)

    output = tmp_path / "formal-output"
    output.mkdir()
    configured = FmowResNet50CampaignConfig(
        metadata_csv=metadata,
        data_root=tmp_path / "data",
        output_dir=output,
        geography_contract=contract_path,
    )
    copied, lineage, copied_path = _prepare_geography_contract(
        configured,
        output,
    )
    assert copied is not None
    assert copied_path == output / "geography_contract.json"
    assert lineage["geography_contract_hash"] == contract["contract_hash"]


def test_overlay_rejects_metadata_contract_mismatch(tmp_path: Path) -> None:
    contract_path, _metadata, _mapping, _contract_value = _contract(tmp_path)
    source = tmp_path / "dofa"
    _write_fake_dofa_source(
        source,
        metadata_sha="wrong",
        protocol_hash="protocol-1",
        metric_version="geobwer_fractional_1.1",
    )
    with pytest.raises(
        FmowDOFAv2PostprocessError,
        match="metadata SHA-256",
    ):
        postprocess_fmow_dofav2_provenance(
            source,
            contract_path,
            tmp_path / "overlay",
        )
