from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rsfm_fairness_audit.geographic_risk_association import load_external_covariates
from rsfm_fairness_audit.geographic_risk_covariates import (
    aggregate_covariates,
    build_alphaearth_sampling_rows,
    prepare_geographic_risk_covariates,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_alphaearth_sampling_is_limited_to_frozen_test_risk_units(tmp_path: Path) -> None:
    risk = tmp_path / "risk.csv"
    samples = tmp_path / "samples.csv"
    _write(risk, [{"spatial_unit": "a", "latitude": 10, "longitude": 20}])
    _write(samples, [
        {"sample_id": "keep", "spatial_block_id": "a", "split": "test", "latitude": 10, "longitude": 20, "worldcover_label": 10},
        {"sample_id": "train", "spatial_block_id": "a", "split": "train", "latitude": 10, "longitude": 20, "worldcover_label": 10},
        {"sample_id": "other", "spatial_block_id": "b", "split": "test", "latitude": 11, "longitude": 21, "worldcover_label": 50},
    ])
    rows, evidence = build_alphaearth_sampling_rows(samples, risk)
    assert [row["row_id"] for row in rows] == ["keep"]
    assert evidence["unit_coverage"] == 1


def test_aggregate_covariates_preserves_smod_class_and_reference_semantics() -> None:
    rows = aggregate_covariates("AlphaEarth", [
        {"spatial_unit": "a", "latitude": 10, "longitude": 20, "source_worldcover_label": 10,
         "dynamic_world_label": 1, "reference_confidence": .8, "ghsl_urbanization": 11,
         "population_density": 100, "nightlights": 2},
        {"spatial_unit": "a", "latitude": 10.2, "longitude": 20.2, "source_worldcover_label": 50,
         "dynamic_world_label": 1, "reference_confidence": .6, "ghsl_urbanization": 11,
         "population_density": 300, "nightlights": 4},
        {"spatial_unit": "a", "latitude": 10.1, "longitude": 20.1, "source_worldcover_label": 50,
         "dynamic_world_label": 6, "reference_confidence": .7, "ghsl_urbanization": 30,
         "population_density": 200, "nightlights": 3},
    ])
    assert rows[0]["ghsl_urbanization"] == 11
    assert rows[0]["population_density"] == 200
    assert rows[0]["reference_confidence"] == pytest.approx(.7)
    assert rows[0]["reference_disagreement"] == pytest.approx(1 / 3)
    assert 0 < rows[0]["land_cover_heterogeneity"] < 1


def test_prepare_writes_canonical_manifest_qa_and_reuses_fixed_task_contract(tmp_path: Path, monkeypatch) -> None:
    atlas = tmp_path / "atlas"
    alpha_samples = tmp_path / "alpha.csv"
    _write(atlas / "alphaearth_spatial_unit_risk.csv", [
        {"spatial_unit": "a", "latitude": 10, "longitude": 20, "mean_risk": .2},
    ])
    for model in ("dofa", "resnet"):
        _write(atlas / f"fmow_{model}_spatial_unit_risk.csv", [
            {"spatial_unit": "site", "latitude": 30, "longitude": 40, "mean_risk": .3},
        ])
    _write(alpha_samples, [
        {"sample_id": "x", "spatial_block_id": "a", "split": "test", "latitude": 10,
         "longitude": 20, "worldcover_label": 10},
    ])

    def fake_extract(rows, *, include_dynamic_world, **kwargs):
        return [{**dict(row), "ghsl_urbanization": 21, "population_density": 50,
                 "nightlights": 1.5, "dynamic_world_label": 1 if include_dynamic_world else "",
                 "reference_confidence": .9 if include_dynamic_world else ""} for row in rows]

    monkeypatch.setattr(
        "rsfm_fairness_audit.geographic_risk_covariates.extract_official_covariates_gee",
        fake_extract,
    )
    output = tmp_path / "covariates"
    result = prepare_geographic_risk_covariates(
        atlas_dir=atlas, alphaearth_sample_csv=alpha_samples,
        output_dir=output, cache_dir=tmp_path / "cache",
    )
    assert result["status"] == "complete"
    assert (output / "alphaearth_covariates.csv").is_file()
    assert (output / "fmow_covariates.csv").is_file()
    assert (output / "covariate_qa.csv").is_file()
    manifest = json.loads((output / "geographic_covariate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["spatial_matching"]["buffer_selection"] == "none"
    assert "no outcome-informed" in manifest["scientific_boundary"]["selection_policy"]


def test_association_loader_reads_recovered_reference_fields(tmp_path: Path) -> None:
    path = tmp_path / "external.csv"
    _write(path, [{
        "spatial_unit": "a", "latitude": 10, "longitude": 20,
        "land_cover_heterogeneity": .2, "reference_confidence": .8,
        "reference_disagreement": .3, "ghsl_urbanization": 21,
        "population_density": 100, "nightlights": 3,
    }])
    rows, evidence = load_external_covariates(path)
    assert rows[0]["reference_confidence"] == .8
    assert rows[0]["reference_disagreement"] == .3
    assert evidence["variables"]["land_cover_heterogeneity"] == "land_cover_heterogeneity"
