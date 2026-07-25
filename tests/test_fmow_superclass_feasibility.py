from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsfm_fairness_audit.fmow_superclass_feasibility import (
    DEFAULT_TAXONOMY,
    FmowSuperclassFeasibilityError,
    load_fmow_superclass_taxonomy,
    scan_fmow_superclass_feasibility,
)
from rsfm_fairness_audit.fmow_geography_contract import (
    build_fmow_geography_contract,
    write_fmow_geography_contract,
)
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import read_csv_rows, write_csv


AXIS_ROLE_FREEZE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "fmow"
    / "fmow_superclass_axis_role_freeze_v1.json"
)


def _metadata_rows() -> list[dict[str, str]]:
    _taxonomy, mapping = load_fmow_superclass_taxonomy()
    rows: list[dict[str, str]] = []
    for index, category in enumerate(sorted(mapping)):
        country = (
            "DEU"
            if mapping[category] == "food_production" or index % 2 == 0
            else "USA"
        )
        region = "Western Europe" if country == "DEU" else "Northern America"
        rows.append(
            {
                "sample_id": f"test-{index:03d}",
                "site_id": f"{category}|site-{index:03d}",
                "location_id": f"site-{index:03d}",
                "split": "test",
                "category": category,
                "country": country,
                "continent": "Europe" if country == "DEU" else "North America",
                "un_region": region,
                "region": region,
            }
        )
        rows.append(
            {
                "sample_id": f"train-{index:03d}",
                "site_id": f"{category}|train-site-{index:03d}",
                "location_id": f"train-site-{index:03d}",
                "split": "train",
                "category": category,
                "country": country,
                "continent": "Europe" if country == "DEU" else "North America",
                "un_region": region,
                "region": region,
            }
        )
    return rows


def _write_contract_for_metadata(
    tmp_path: Path,
    metadata: Path,
    *,
    suffix: str = "",
) -> Path:
    mapping = tmp_path / f"country_region_map{suffix}.csv"
    write_csv(
        mapping,
        [
            {
                "country": "DEU",
                "continent": "Europe",
                "un_region": "Western Europe",
                "region": "Western Europe",
            },
            {
                "country": "USA",
                "continent": "North America",
                "un_region": "Northern America",
                "region": "Northern America",
            },
        ],
    )
    contract = build_fmow_geography_contract(
        metadata,
        mapping,
        mapping_source_name="test country-region map",
        mapping_source_version="v1",
        mapping_source_url="https://example.test/geography",
    )
    return write_fmow_geography_contract(
        tmp_path / f"geography_contract{suffix}.json",
        contract,
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    metadata = tmp_path / "clean_metadata.csv"
    write_csv(metadata, _metadata_rows())
    contract_path = _write_contract_for_metadata(tmp_path, metadata)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["formal_compatible"] is True
    return metadata, contract_path


def test_default_taxonomy_has_exactly_62_unique_classes() -> None:
    taxonomy, mapping = load_fmow_superclass_taxonomy()
    assert taxonomy["expected_class_count"] == 62
    assert len(mapping) == 62
    assert len(set(mapping)) == 62
    assert len(set(mapping.values())) == 8
    assert mapping["airport"] == "mobility_logistics"
    assert mapping["surface_mine"] == "construction_extraction_damage"


def test_axis_role_freeze_binds_scan_and_limits_claims() -> None:
    freeze = json.loads(AXIS_ROLE_FREEZE.read_text(encoding="utf-8"))
    assert freeze["schema"] == "geobwer.fmow.superclass_axis_role_freeze.v1"
    assert (
        freeze["source_contracts"]["feasibility_contract_hash"]
        == "e8414bad51118b7506d0109c66c08e36360ac7af4679aacbef251f946fdc2442"
    )
    assert (
        freeze["source_contracts"]["metadata_sha256"]
        == "6fcf4ab5bb6648ecab0b00dea4b439d84f594e752f4d5a1d79f6589ccabcb249"
    )
    region = freeze["axis_roles"]["region_superclass"]
    country = freeze["axis_roles"]["country_superclass"]
    assert region["fixed_universe_confirmatory_claim_eligible"] is False
    assert region["supported_universe_role"] == (
        "preregistered_secondary_exploratory"
    )
    assert country["supported_universe_main_text_eligible"] is False
    assert country["broad_intersectional_fairness_claim_eligible"] is False
    cells = freeze["high_support_region_cell_panel"]["cells"]
    assert len(cells) == 6
    assert all(int(cell["sample_count"]) >= 20 for cell in cells)
    assert all(int(cell["independent_site_count"]) >= 30 for cell in cells)
    assert (
        freeze["country_high_support_reference"]["fairness_axis_eligible"]
        is False
    )


def test_scan_is_metadata_only_and_includes_zero_support_cells(
    tmp_path: Path,
) -> None:
    metadata, geography_contract = _write_inputs(tmp_path)
    artifacts = scan_fmow_superclass_feasibility(
        metadata,
        geography_contract,
        tmp_path / "scan",
        min_samples=1,
        min_sites=1,
        confirmatory_min_sites=2,
    )
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["formal_evidence"] is False
    assert manifest["model_outputs_used"] is False
    assert manifest["performance_outcomes_used"] is False
    assert manifest["metadata"]["row_count"] == 124
    assert manifest["metadata"]["selected_row_count"] == 62
    assert manifest["metadata"]["sha256"] == file_sha256(metadata)
    assert manifest["taxonomy"]["sha256"] == file_sha256(DEFAULT_TAXONOMY)
    assert (
        manifest["geography_contract"]["metadata_sha256"]
        == file_sha256(metadata)
    )
    assert manifest["geography_contract"]["unresolved_country_row_count"] == 0
    assert manifest["geography_contract"]["rows_dropped"] == 0
    assert manifest["artifacts"]["report"]["sha256"] == file_sha256(
        artifacts["report"]
    )
    assert len(manifest["contract_hash"]) == 64
    cells = read_csv_rows(artifacts["cells"])
    assert len(cells) == 32
    assert any(int(row["sample_count"]) == 0 for row in cells)
    assert {
        row["support_status"] for row in cells
    } <= {
        "insufficient_support",
        "descriptive_supported",
        "confirmatory_supported",
    }
    summaries = {
        row["axis"]: row for row in read_csv_rows(artifacts["summary"])
    }
    assert set(summaries) == {"country_superclass", "region_superclass"}
    assert summaries["country_superclass"]["fixed_universe_cell_count"] == "16"
    assert artifacts["report"].is_file()


def test_scan_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    rows = _metadata_rows()
    rows[-1]["sample_id"] = rows[0]["sample_id"]
    metadata = tmp_path / "duplicate.csv"
    write_csv(metadata, rows)
    geography_contract = _write_contract_for_metadata(
        tmp_path,
        metadata,
        suffix="_duplicate",
    )
    with pytest.raises(
        FmowSuperclassFeasibilityError,
        match="Duplicate sample_id",
    ):
        scan_fmow_superclass_feasibility(
            metadata,
            geography_contract,
            tmp_path / "scan",
        )


def test_scan_rejects_taxonomy_metadata_class_mismatch(
    tmp_path: Path,
) -> None:
    rows = _metadata_rows()
    rows[0]["category"] = "not_an_fmow_class"
    metadata = tmp_path / "unknown.csv"
    write_csv(metadata, rows)
    geography_contract = _write_contract_for_metadata(
        tmp_path,
        metadata,
        suffix="_unknown",
    )
    with pytest.raises(
        FmowSuperclassFeasibilityError,
        match="absent from taxonomy",
    ):
        scan_fmow_superclass_feasibility(
            metadata,
            geography_contract,
            tmp_path / "scan",
        )


def test_scan_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    metadata, geography_contract = _write_inputs(tmp_path)
    output = tmp_path / "scan"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(
        FmowSuperclassFeasibilityError,
        match="Refusing to overwrite",
    ):
        scan_fmow_superclass_feasibility(
            metadata,
            geography_contract,
            output,
        )


def test_scan_rejects_metadata_not_bound_by_geography_contract(
    tmp_path: Path,
) -> None:
    metadata, geography_contract = _write_inputs(tmp_path)
    rows = read_csv_rows(metadata)
    rows[0]["region"] = "Changed after contract freeze"
    changed = tmp_path / "changed_metadata.csv"
    write_csv(changed, rows)
    with pytest.raises(
        FmowSuperclassFeasibilityError,
        match="Metadata SHA-256",
    ):
        scan_fmow_superclass_feasibility(
            changed,
            geography_contract,
            tmp_path / "scan",
        )
