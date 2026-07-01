from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from scripts.analysis.check_alphaearth_full_export_schema import EMBEDDING_BANDS, check_alphaearth_full_export_schema
from scripts.analysis.run_alphaearth_landcover_full_audit import run_alphaearth_full_audit


def _synthetic_full_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    countries = [("USA", "North America", "High income"), ("BRA", "Latin America", "Upper middle income"), ("IND", "South Asia", "Lower middle income")]
    labels = [(10, "Tree cover"), (40, "Cropland"), (50, "Built-up")]
    for country_i, (country, region, income) in enumerate(countries):
        for label_i, (label, class_name) in enumerate(labels):
            for rep in range(12):
                split = "test" if rep in {0, 1} else "calibration" if rep in {2, 3} else "train"
                row: dict[str, object] = {
                    "sample_id": f"{country}_{label}_{rep}",
                    "lon": -100 + country_i + rep * 0.01,
                    "lat": 20 + label_i + rep * 0.01,
                    "year": 2021,
                    "country_iso3": country,
                    "region": region,
                    "income_group": income,
                    "biome_or_ecoregion": f"biome_{label_i}",
                    "urban_rural_or_built_proxy": "built_proxy" if label == 50 else "non_built_proxy",
                    "spatial_block_id": f"block_{country_i}_{rep // 2}",
                    "split": split,
                    "worldcover_label": label,
                    "worldcover_class_name": class_name,
                    "dynamic_world_label": label_i,
                    "dynamic_world_confidence": 0.8,
                }
                for band_i, band in enumerate(EMBEDDING_BANDS):
                    row[band] = label_i * 0.7 + country_i * 0.02 + (band_i % 7) * 0.005
                rows.append(row)
    return rows


def test_full_export_schema_synthetic() -> None:
    root = Path("outputs") / f"test_alphaearth_full_schema_{uuid4().hex}"
    export = root / "alphaearth_worldcover_full_export.csv"
    write_csv(export, _synthetic_full_rows())
    artifacts = check_alphaearth_full_export_schema(export, root / "missing_manifest.csv", root)
    status = read_csv_rows(artifacts["alphaearth_full_export_schema_status"])[0]
    assert status["schema_status"] == "ok"
    assert int(status["n_classes"]) == 3


def test_full_support_preflight_smoke() -> None:
    root = Path("outputs") / f"test_alphaearth_full_support_{uuid4().hex}"
    export = root / "alphaearth_worldcover_full_export.csv"
    write_csv(export, _synthetic_full_rows())
    artifacts = check_alphaearth_full_export_schema(export, root / "missing_manifest.csv", root)
    support = read_csv_rows(artifacts["alphaearth_full_support_preflight"])
    assert any(row["slice_variable"] == "country_iso3" for row in support)
    assert any(row["slice_variable"] == "country_iso3|worldcover_class_name" for row in support)


def test_full_bwer_conformal_metric_smoke() -> None:
    root = Path("outputs") / f"test_alphaearth_full_audit_{uuid4().hex}"
    export = root / "alphaearth_worldcover_full_export.csv"
    audit = root / "audit"
    unified = root / "unified"
    write_csv(export, _synthetic_full_rows())
    artifacts = run_alphaearth_full_audit(export, root / "missing_manifest.csv", root, audit, unified, seed=11)
    assert artifacts["alphaearth_full_metrics"].exists()
    assert artifacts["alphaearth_full_conformal_coverage_summary"].exists()
    conformal = read_csv_rows(artifacts["alphaearth_full_conformal_coverage_summary"])
    assert {row["coverage_target"] for row in conformal} == {"0.7", "0.8", "0.9"}


def test_full_output_existence_smoke() -> None:
    root = Path("outputs") / f"test_alphaearth_full_outputs_{uuid4().hex}"
    export = root / "alphaearth_worldcover_full_export.csv"
    audit = root / "audit"
    unified = root / "unified"
    write_csv(export, _synthetic_full_rows())
    artifacts = run_alphaearth_full_audit(export, root / "missing_manifest.csv", root, audit, unified, seed=13)
    required = [
        "alphaearth_full_metrics",
        "alphaearth_full_predictions",
        "alphaearth_full_bwer_summary",
        "alphaearth_full_standardised_bwer",
        "alphaearth_full_selective_risk_summary",
        "alphaearth_full_selective_bwer",
        "alphaearth_full_calibrated_threshold_bwer",
        "alphaearth_full_conformal_bwer",
        "alphaearth_full_claim_support",
        "alphaearth_logistic_baseline_metrics",
        "alphaearth_logistic_baseline_bwer_summary",
        "alphaearth_logistic_baseline_standardised_bwer",
        "alphaearth_full_rank_divergence",
        "alphaearth_grassland_confusion_diagnostic",
        "alphaearth_conformal_slice_gap_diagnostic",
        "alphaearth_scale_sensitivity_summary",
        "figure_alphaearth_aggregate_vs_bwer_png",
    ]
    for key in required:
        assert artifacts[key].exists(), key
    assert (unified / "rsfm_bwer_paper_freeze_v4.zip").exists()
