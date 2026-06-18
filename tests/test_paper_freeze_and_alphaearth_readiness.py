from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

from rsfm_fairness_audit.io import read_csv_rows
from scripts.analysis.build_paper_freeze_v3_and_alphaearth_readiness import build_alphaearth_readiness, build_paper_freeze


def test_paper_freeze_outputs_and_zip_smoke() -> None:
    out = Path("outputs") / f"test_paper_freeze_v3_{uuid4().hex}"
    artifacts = build_paper_freeze(out)
    assert artifacts["MANIFEST"].exists()
    assert artifacts["asset_index"].exists()
    assert artifacts["formal_vs_diagnostic_vs_sanity_table"].exists()
    assert artifacts["source_output_references"].exists()
    assert artifacts["zip"].exists()
    rows = read_csv_rows(artifacts["source_output_references"])
    assert any(Path(row["source_output"]).as_posix() == "outputs/unified_paper_package_v3" for row in rows)
    with ZipFile(artifacts["zip"]) as archive:
        names = set(archive.namelist())
    assert "MANIFEST.md" in names
    assert any(name.startswith("source_outputs/unified_paper_package_v3/") for name in names)


def test_alphaearth_readiness_schema_smoke() -> None:
    out = Path("outputs") / f"test_alphaearth_readiness_{uuid4().hex}"
    artifacts = build_alphaearth_readiness(out)
    for path in artifacts.values():
        assert path.exists()
    manifest_rows = read_csv_rows(artifacts["gee_export_manifest_template"])
    schema_rows = read_csv_rows(artifacts["alphaearth_audit_table_schema"])
    risk_rows = read_csv_rows(artifacts["alphaearth_risk_register"])
    assert {"sample_id", "country_iso3", "biome_or_ecoregion", "confidence"}.issubset(manifest_rows[0])
    assert any(row["field"] == "confidence" and row["required"] == "optional" for row in schema_rows)
    assert any("Spatial leakage" in row["risk"] for row in risk_rows)
