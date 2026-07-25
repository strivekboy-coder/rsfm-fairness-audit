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
