from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from rsfm_fairness_audit.cli import main
from rsfm_fairness_audit.fmow_sentinel_enrichment import (
    FmowMetadataEnrichmentConfig,
    run_fmow_sentinel_metadata_enrichment,
)
from rsfm_fairness_audit.fmow_sentinel_preflight import FmowPreflightConfig, run_fmow_sentinel_preflight
from rsfm_fairness_audit.io import read_csv_rows, write_csv


def _root(name: str) -> Path:
    path = Path("outputs") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_satmae_csv(path: Path) -> None:
    write_csv(
        path,
        [
            {
                "category": "airport",
                "location_id": "loc_a",
                "timestamp": "2020-01-15T10:00:00",
                "image_id": "001",
            },
            {
                "category": "port",
                "location_id": "loc_b",
                "timestamp": "2020-07-02",
                "image_id": "002",
            },
        ],
    )


def test_fmow_enrichment_without_external_metadata_does_not_fake_geography() -> None:
    root = _root("test_fmow_enrich_missing_geo")
    satmae = root / "train.csv"
    _write_satmae_csv(satmae)
    out = root / "enriched"
    run_fmow_sentinel_metadata_enrichment(
        FmowMetadataEnrichmentConfig(satmae_csvs=(satmae,), output_dir=out)
    )
    rows = read_csv_rows(out / "fmow_enriched_metadata.csv")
    assert rows[0]["split"] == "train"
    assert rows[0]["country"] == ""
    assert rows[0]["latitude"] == ""
    assert rows[0]["latitude_band"] == ""
    assert rows[0]["geography_warning"] == "location_id_available_but_not_country"
    warnings = json.loads((out / "warnings.json").read_text(encoding="utf-8"))["warnings"]
    assert any("do not interpret location_id as country" in warning for warning in warnings)
    shutil.rmtree(root, ignore_errors=True)


def test_fmow_enrichment_joins_external_geography_and_feeds_preflight() -> None:
    root = _root("test_fmow_enrich_join")
    satmae = root / "val.csv"
    external = root / "gps.csv"
    _write_satmae_csv(satmae)
    write_csv(
        external,
        [
            {
                "category": "airport",
                "location_id": "loc_a",
                "image_id": "001",
                "latitude": "42.1",
                "longitude": "-71.2",
                "country": "United States",
                "region": "North America",
                "continent": "North America",
                "un_region": "Americas",
            },
            {
                "category": "port",
                "location_id": "loc_b",
                "image_id": "002",
                "latitude": "-12.5",
                "longitude": "34.1",
                "country": "Mozambique",
                "region": "Eastern Africa",
                "continent": "Africa",
                "un_region": "Africa",
            },
        ],
    )
    enrich_out = root / "enriched"
    run_fmow_sentinel_metadata_enrichment(
        FmowMetadataEnrichmentConfig(
            satmae_csvs=(satmae,),
            external_metadata_csvs=(external,),
            output_dir=enrich_out,
        )
    )
    rows = read_csv_rows(enrich_out / "fmow_enriched_metadata.csv")
    assert rows[0]["join_status"] == "matched"
    assert rows[0]["country"] == "United States"
    assert rows[0]["latitude_band"] == "north_mid_latitude"
    coverage = read_csv_rows(enrich_out / "fmow_geography_coverage_summary.csv")
    country = next(row for row in coverage if row["field"] == "country")
    assert float(country["coverage"]) == 1.0

    preflight_out = root / "preflight"
    run_fmow_sentinel_preflight(
        FmowPreflightConfig(
            metadata_csvs=(enrich_out / "fmow_enriched_metadata.csv",),
            output_dir=preflight_out,
            metadata_only=True,
            min_support=1,
            subset_max_per_split=2,
        )
    )
    inventory = read_csv_rows(preflight_out / "fmow_metadata_inventory.csv")
    assert next(row for row in inventory if row["canonical_field"] == "country")["status"] == "present"
    shutil.rmtree(root, ignore_errors=True)


def test_fmow_enrichment_can_fill_continent_and_un_region_from_country_map() -> None:
    root = _root("test_fmow_enrich_country_map")
    satmae = root / "train.csv"
    external = root / "geo.csv"
    country_map = root / "country_region_map.csv"
    _write_satmae_csv(satmae)
    write_csv(
        external,
        [
            {"category": "airport", "location_id": "loc_a", "image_id": "001", "country": "United States"},
            {"category": "port", "location_id": "loc_b", "image_id": "002", "country": "Mozambique"},
        ],
    )
    write_csv(
        country_map,
        [
            {"country": "United States", "continent": "North America", "un_region": "Americas", "region": "Northern America"},
            {"country": "Mozambique", "continent": "Africa", "un_region": "Africa", "region": "Eastern Africa"},
        ],
    )
    out = root / "enriched"
    run_fmow_sentinel_metadata_enrichment(
        FmowMetadataEnrichmentConfig(
            satmae_csvs=(satmae,),
            external_metadata_csvs=(external,),
            output_dir=out,
            country_region_map=country_map,
        )
    )
    rows = read_csv_rows(out / "fmow_enriched_metadata.csv")
    assert rows[0]["country"] == "United States"
    assert rows[0]["continent"] == "North America"
    assert rows[0]["un_region"] == "Americas"
    assert rows[0]["continent_provenance"] == "country_region_map"
    coverage = read_csv_rows(out / "fmow_geography_coverage_summary.csv")
    assert next(row for row in coverage if row["field"] == "continent")["coverage"] == "1.0"
    shutil.rmtree(root, ignore_errors=True)


def test_fmow_enrichment_cli_reports_join_failures(monkeypatch) -> None:
    root = _root("test_fmow_enrich_cli")
    satmae = root / "train.csv"
    external = root / "geo.csv"
    _write_satmae_csv(satmae)
    write_csv(external, [{"category": "airport", "location_id": "different", "image_id": "999", "country": "X"}])
    out = root / "enriched"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "enrich-fmow-sentinel-metadata",
            "--satmae-csv",
            str(satmae),
            "--external-metadata-csv",
            str(external),
            "--output-dir",
            str(out),
        ],
    )
    main()
    failures = read_csv_rows(out / "fmow_join_failures.csv")
    assert len(failures) == 2
    assert failures[0]["reason"] == "unmatched"
    report = (out / "fmow_metadata_join_report.md").read_text(encoding="utf-8")
    assert "no SatMAE rows matched" in report
    shutil.rmtree(root, ignore_errors=True)
