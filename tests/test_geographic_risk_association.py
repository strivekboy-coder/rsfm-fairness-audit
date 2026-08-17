from __future__ import annotations

import csv
from pathlib import Path

from rsfm_fairness_audit.geographic_risk_association import (
    _geographic_summaries,
    _validate_fmow_rebuilt_geography,
    _maximum_spatial_cell_count,
    _spatial_cell,
    _spatial_cluster_qa,
    build_geographic_risk_association,
    derive_alphaearth_covariates,
)
from rsfm_fairness_audit.io import read_csv_rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_alphaearth_preregistered_covariates_have_fixed_semantics(tmp_path: Path) -> None:
    source = tmp_path / "alpha.csv"
    _write(source, [
        {"spatial_block_id": "a", "latitude": 10, "longitude": 20, "split": "test",
         "worldcover_label": 10, "dynamic_world_label": 1, "dynamic_world_confidence": .9},
        {"spatial_block_id": "a", "latitude": 10, "longitude": 20, "split": "test",
         "worldcover_label": 50, "dynamic_world_label": 1, "dynamic_world_confidence": .7},
    ])
    rows, evidence = derive_alphaearth_covariates(source)
    assert len(rows) == 1
    assert 0 < rows[0]["land_cover_heterogeneity"] < 1
    assert rows[0]["reference_confidence"] == .8
    assert rows[0]["reference_disagreement"] == .5
    assert "log_11" in evidence["land_cover_heterogeneity_definition"]


def test_spatial_cells_are_fixed_within_latitude_band() -> None:
    # Exact latitude must not enter the key: nearby points in one fixed cell cluster together.
    assert _spatial_cell(10.01, 20.01) == _spatial_cell(10.49, 20.49)
    cells = [_spatial_cell(lat, lon) for lat in range(-89, 90) for lon in range(-179, 180, 10)]
    qa = _spatial_cluster_qa(cells)
    assert qa["status"] == "pass"
    assert qa["cluster_count"] <= _maximum_spatial_cell_count()
    assert qa["cluster_count"] < qa["unit_count"]


def test_association_builds_confirmatory_and_exploratory_outputs(tmp_path: Path) -> None:
    atlas = tmp_path / "atlas"
    risk_rows = []
    sample_rows = []
    external_rows = []
    for index in range(40):
        unit = f"u{index:02d}"
        lat = -60 + 3 * index
        lon = -170 + 8 * index
        risk_rows.append({"spatial_unit": unit, "latitude": lat, "longitude": lon,
                          "mean_risk": index / 50, "support": 2,
                          "tail_excess_over_unit_q90": 0})
        sample_rows.extend([
            {"spatial_block_id": unit, "latitude": lat, "longitude": lon, "split": "test",
             "worldcover_label": 10, "dynamic_world_label": 1, "dynamic_world_confidence": .9 - index / 100},
            {"spatial_block_id": unit, "latitude": lat, "longitude": lon, "split": "test",
             "worldcover_label": 50 if index % 2 else 10, "dynamic_world_label": 1,
             "dynamic_world_confidence": .8 - index / 100},
        ])
        external_rows.append({"spatial_unit": unit, "ghsl_urbanization": index / 40,
                              "population_density": index * 10, "nightlights": index / 2})
    _write(atlas / "alphaearth_spatial_unit_risk.csv", risk_rows)
    _write(tmp_path / "alpha_samples.csv", sample_rows)
    _write(tmp_path / "external.csv", external_rows)
    # reBEN is reproduced only; no regression inputs are accepted for it.
    _write(atlas / "reben_country_label_burden.csv", [
        {"country": "DEU", "class_label": "urban", "mean_delta_risk": .2},
        {"country": "BRA", "class_label": "urban", "mean_delta_risk": .4},
    ])
    result = build_geographic_risk_association(
        atlas, tmp_path / "output", alphaearth_sample_csv=tmp_path / "alpha_samples.csv",
        alphaearth_external_csv=tmp_path / "external.csv", n_boot=40,
    )
    assert result["status"] == "complete"
    assert result["reben"]["status"] == "reproduced"
    assert "no_socioeconomic_regression" in result["reben"]["analysis_boundary"]
    completed = {row["variable"] for row in result["results"]}
    assert {"land_cover_heterogeneity", "reference_confidence", "reference_disagreement",
            "ghsl_urbanization", "population_density", "nightlights"} <= completed
    assert all(row["status"] == "pass" for row in result["visual_qa"])
    cluster_qa = read_csv_rows(tmp_path / "output" / "spatial_cluster_qa.csv")
    assert cluster_qa
    assert all(row["definition"] == "fixed_latitude_band_centre_adjusted_longitude_grid" for row in cluster_qa)
    assert all(int(row["cluster_count"]) <= int(row["maximum_possible_global_cluster_count"]) for row in cluster_qa)


def test_fmow_original_sequence_association_keeps_raw_and_robustness_layers(tmp_path: Path) -> None:
    atlas = tmp_path / "atlas"
    risks, covariates = [], []
    for index in range(40):
        category = "airport" if index < 20 else "port"
        location_id = str(index)
        site_id = f"test|{category}|{location_id}"
        continent = "America" if index < 20 else "Europe"
        country = "USA" if index < 20 else "DEU"
        latitude = -55 + index * 2.5
        longitude = -170 + index * 8
        risks.append({
            "spatial_unit": site_id, "fmow_geographic_site_id": site_id,
            "split_original": "test", "category": category, "location_id": location_id,
            "archive_parent": f"fmow-sentinel/test/{category}/{category}_{location_id}",
            "polygon_centroid_span_m": 0, "coordinate_source": "original_polygon_wkt_centroid",
            "latitude": latitude, "longitude": longitude,
            "mean_risk": index / 50, "support": 3,
            "tail_excess_over_unit_q90": 0,
        })
        covariates.append({
            "spatial_unit": site_id, "fmow_geographic_site_id": site_id,
            "split_original": "test", "category": category, "location_id": location_id,
            "archive_parent": f"fmow-sentinel/test/{category}/{category}_{location_id}",
            "polygon_centroid_span_m": 0, "coordinate_source": "original_polygon_wkt_centroid",
            "country": country, "country_code": country, "continent": continent,
            "region": continent, "geography_match_count": 1,
            "geography_source": "USDOS/LSIB_SIMPLE/2017",
            "latitude": latitude, "longitude": longitude,
            "ghsl_urbanization": 11 if index < 20 else 30,
            "population_density": index * 100, "nightlights": index / 2,
        })
    _write(atlas / "fmow_DOFAv2_spatial_unit_risk.csv", risks)
    _write(tmp_path / "fmow_covariates.csv", covariates)
    result = build_geographic_risk_association(
        atlas, tmp_path / "output",
        fmow_external_csvs={"DOFAv2": tmp_path / "fmow_covariates.csv"},
        n_boot=40, fmow_expected_site_count=40,
    )
    night = next(row for row in result["results"] if row["variable"] == "nightlights")
    assert night["raw_spearman_rho"] == night["spearman_rho"]
    assert "partial_spatial_cluster_bootstrap_ci_low" in night
    layers = read_csv_rows(tmp_path / "output" / "association_layers.csv")
    assert {row["analysis_layer"] for row in layers} == {
        "descriptive_spatial_covariation", "robustness_independent_association",
    }
    summaries = read_csv_rows(tmp_path / "output" / "fmow_geographic_summary.csv")
    assert {row["geography"] for row in summaries if row["variable"] == "nightlights"} >= {
        "America", "Europe", "USA", "DEU",
    }


def test_fmow_association_rejects_location_id_only_predecessor(tmp_path: Path) -> None:
    import pytest

    atlas = tmp_path / "atlas"
    _write(atlas / "fmow_DOFAv2_spatial_unit_risk.csv", [{
        "spatial_unit": "1", "latitude": 10, "longitude": 20,
        "mean_risk": .5, "support": 10,
    }])
    with pytest.raises(ValueError, match="fMoW geographic identity requires"):
        build_geographic_risk_association(
            atlas, tmp_path / "output", n_boot=20, fmow_expected_site_count=1,
        )


def test_admin_unmatched_sites_are_excluded_only_from_geographic_summaries(tmp_path: Path) -> None:
    rows = []
    for index in range(40):
        reliable = index < 20
        rows.append({
            "spatial_unit": f"test|airport|{index}",
            "mean_risk": index / 50, "nightlights": index,
            "geography_match_count": 1 if reliable else 0,
            "country": "USA" if reliable else "unknown",
            "continent": "North America" if reliable else "unknown",
            "region": "North America" if reliable else "unknown",
        })
    coverage = _validate_fmow_rebuilt_geography(rows, tmp_path / "covariates.csv")
    assert coverage["exact_admin_match_count"] == 20
    assert coverage["unmatched_admin_site_count"] == 20
    summaries, qa = _geographic_summaries("fMoW", "DOFAv2", rows, "nightlights")
    assert {row["geography"] for row in summaries} == {"USA", "North America"}
    assert all(row["admin_match_coverage"] == .5 for row in qa)
