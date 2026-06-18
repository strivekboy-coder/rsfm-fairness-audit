from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from scripts.analysis.check_alphaearth_export_schema import EMBEDDING_BANDS, check_alphaearth_export_schema
from scripts.analysis.run_alphaearth_landcover_pilot_audit import run_alphaearth_landcover_pilot_audit


def _synthetic_alphaearth_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    countries = ["USA", "BRA", "IND", "ZAF"]
    labels = [10, 20, 30]
    for split in ["train", "test"]:
        for country_i, country in enumerate(countries):
            for label_i, label in enumerate(labels):
                for rep in range(4):
                    row: dict[str, object] = {
                        "sample_id": f"{split}_{country}_{label}_{rep}",
                        "lon": -100 + country_i + rep * 0.01,
                        "lat": 30 + label_i + rep * 0.01,
                        "year": 2021,
                        "country_iso3": country,
                        "region": f"region_{country_i % 2}",
                        "income_group": "High income" if country in {"USA"} else "Middle income",
                        "biome_or_ecoregion": f"biome_{label_i % 2}",
                        "urban_rural_or_built_proxy": "built_proxy" if label == 30 else "non_built_proxy",
                        "spatial_block_id": f"block_{country_i}_{rep}",
                        "split": split,
                        "worldcover_label": label,
                        "worldcover_class_name": f"class_{label}",
                        "dynamic_world_label": label_i,
                        "dynamic_world_confidence": 0.8,
                    }
                    for band_i, band in enumerate(EMBEDDING_BANDS):
                        row[band] = (label_i + 1) * 0.5 + (band_i % 5) * 0.01 + country_i * 0.001
                    rows.append(row)
    return rows


def test_alphaearth_schema_checker_with_synthetic_export() -> None:
    out = Path("outputs") / f"test_alphaearth_schema_{uuid4().hex}"
    csv_path = out / "synthetic_alphaearth_export.csv"
    write_csv(csv_path, _synthetic_alphaearth_rows())
    status, report = check_alphaearth_export_schema(csv_path, out)
    assert status["schema_status"] == "ok"
    assert report.exists()
    assert "dynamic_world_confidence" in status["optional_columns_present"]


def test_alphaearth_bwer_smoke_with_tiny_synthetic_table() -> None:
    out = Path("outputs") / f"test_alphaearth_audit_{uuid4().hex}"
    csv_path = out / "synthetic_alphaearth_export.csv"
    write_csv(csv_path, _synthetic_alphaearth_rows())
    artifacts = run_alphaearth_landcover_pilot_audit(csv_path, out, min_samples_per_slice=2, seed=7)
    assert artifacts["alphaearth_landcover_pilot_metrics"].exists()
    assert artifacts["alphaearth_bwer_summary"].exists()
    bwer_rows = read_csv_rows(artifacts["alphaearth_bwer_summary"])
    assert any(row["slice_variable"] == "country_iso3" and row["analysis_type"] == "raw" for row in bwer_rows)


def test_alphaearth_output_existence_smoke() -> None:
    out = Path("outputs") / f"test_alphaearth_outputs_{uuid4().hex}"
    csv_path = out / "synthetic_alphaearth_export.csv"
    write_csv(csv_path, _synthetic_alphaearth_rows())
    artifacts = run_alphaearth_landcover_pilot_audit(csv_path, out, min_samples_per_slice=2)
    for key in [
        "alphaearth_export_schema_report",
        "alphaearth_sample_support_summary",
        "alphaearth_landcover_pilot_metrics",
        "alphaearth_bwer_summary",
        "alphaearth_slice_risk_summary",
        "alphaearth_pilot_caveats",
        "alphaearth_pilot_report",
        "figure_aggregate_vs_bwer_alphaearth_pilot_png",
        "figure_support_by_slice_pdf",
    ]:
        assert artifacts[key].exists(), key
