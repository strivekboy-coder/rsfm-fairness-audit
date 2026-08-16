from __future__ import annotations

import csv
from pathlib import Path

from rsfm_fairness_audit.geographic_risk_association import (
    build_geographic_risk_association,
    derive_alphaearth_covariates,
)


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
