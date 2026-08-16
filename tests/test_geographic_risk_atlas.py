from __future__ import annotations

import csv
from pathlib import Path

from rsfm_fairness_audit.geographic_risk_atlas import (
    aggregate_coordinate_risk,
    aggregate_reben_country_label_burden,
    build_geographic_risk_atlas,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_coordinate_asset_uses_real_units_and_test_split(tmp_path: Path) -> None:
    source = tmp_path / "alpha.csv"
    _write(source, [
        {"sample_id": "a", "spatial_block_id": "x", "lat": 10, "lon": 20, "risk": 0, "split": "test"},
        {"sample_id": "b", "spatial_block_id": "x", "lat": 12, "lon": 22, "risk": 1, "split": "test"},
        {"sample_id": "c", "spatial_block_id": "y", "lat": 30, "lon": 40, "risk": 1, "split": "validation"},
    ])
    rows, status = aggregate_coordinate_risk(source)
    assert status["usable_spatial_unit_count"] == 1
    assert rows[0]["spatial_unit"] == "x"
    assert rows[0]["mean_risk"] == .5
    assert rows[0]["latitude"] == 11
    manifest = build_geographic_risk_atlas(tmp_path / "coordinate_atlas", alphaearth_csv=source)
    assert manifest["status"] == "complete"
    assert (tmp_path / "coordinate_atlas" / "alphaearth_coordinate_risk.png").is_file()


def test_reben_country_label_delta_is_aggregated_within_seed(tmp_path: Path) -> None:
    for seed, ood in ((42, [.8, .6]), (73, [.7, .5])):
        folder = tmp_path / f"seed_{seed}"
        base = [
            {"country": "DEU", "class_label": "urban", "risk": .2},
            {"country": "DEU", "class_label": "urban", "risk": .4},
        ]
        shifted = [dict(row, risk=value) for row, value in zip(base, ood)]
        _write(folder / "id_label_audit.csv", base)
        _write(folder / "ood_label_audit.csv", shifted)
    rows, status = aggregate_reben_country_label_burden(tmp_path)
    assert status["seed_count"] == 2
    assert len(rows) == 1
    assert abs(rows[0]["mean_delta_risk"] - .35) < 1e-12
    assert rows[0]["positive_seed_count"] == 2
    manifest = build_geographic_risk_atlas(
        tmp_path / "atlas", reben_paired_dir=tmp_path,
    )
    assert manifest["status"] == "complete"
    assert (tmp_path / "atlas" / "reben_country_label_burden.png").is_file()
