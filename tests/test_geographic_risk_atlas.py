from __future__ import annotations

import csv
from pathlib import Path

from rsfm_fairness_audit.geographic_risk_atlas import (
    aggregate_coordinate_risk,
    aggregate_coordinate_risk_across_seeds,
    aggregate_reben_country_label_burden,
    build_geographic_risk_atlas,
    discover_alphaearth_atlas_asset,
    plot_reben_burden,
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
            {"country": "DEU", "class_label": "urban", "risk_binary_error": .2},
            {"country": "DEU", "class_label": "urban", "risk_binary_error": .4},
        ]
        shifted = [dict(row, risk_binary_error=value) for row, value in zip(base, ood)]
        _write(folder / "id_label_audit.csv", base)
        _write(folder / "ood_label_audit.csv", shifted)
    rows, status = aggregate_reben_country_label_burden(tmp_path)
    assert status["seed_count"] == 2
    assert len(rows) == 1
    assert abs(rows[0]["mean_delta_risk"] - .35) < 1e-12
    assert rows[0]["positive_seed_count"] == 2
    assert status["risk_contract"].startswith("risk_binary_error")
    manifest = build_geographic_risk_atlas(
        tmp_path / "atlas", reben_paired_dir=tmp_path,
    )
    assert manifest["status"] == "complete"
    assert (tmp_path / "atlas" / "reben_country_label_burden.png").is_file()


def test_alphaearth_discovery_selects_formal_sample_table_not_geobwer_aggregates(tmp_path: Path) -> None:
    root = tmp_path / "alphaearth_geobwer_spatial_v2"
    _write(root / "geobwer_raw" / "geobwer_by_group.csv", [
        {"axis": "spatial_block_id", "group": "block_a", "risk": .4, "support": 20},
    ])
    _write(root / "geobwer_raw" / "geobwer_profile.csv", [
        {"axis": "spatial_block_id", "mean_risk": .2, "tail_risk": .4, "bwer": .2},
    ])
    _write(root / "geobwer_raw" / "geobwer_summary.csv", [
        {"axis": "spatial_block_id", "mean_risk": .2, "tail_risk": .4, "bwer": .2},
    ])
    formal = root / "formal_outputs" / "formal_audit_table.csv"
    _write(formal, [{
        "sample_id": "a", "spatial_block_id": "block_a", "latitude": 12,
        "longitude": 34, "risk": 1, "split": "test",
    }])
    selected, evidence = discover_alphaearth_atlas_asset(root)
    assert selected == formal
    assert len(evidence["ignored_aggregate_tables"]) == 3
    manifest = build_geographic_risk_atlas(tmp_path / "auto_atlas", alphaearth_root=root)
    alpha = manifest["readiness"][0]
    assert alpha["unit_field"] == "spatial_block_id"
    assert alpha["automatic_discovery"]["selected_path"] == str(formal)


def test_alphaearth_discovery_rejects_aggregate_only_root(tmp_path: Path) -> None:
    import pytest

    root = tmp_path / "aggregates_only"
    _write(root / "geobwer_raw" / "geobwer_by_group.csv", [
        {"axis": "spatial_block_id", "group": "block_a", "risk": .4, "support": 20},
    ])
    with pytest.raises(ValueError, match="aggregate_only_tables"):
        discover_alphaearth_atlas_asset(root)


def test_reben_country_label_accepts_correct_as_binary_error_fallback(tmp_path: Path) -> None:
    folder = tmp_path / "seed_42"
    _write(folder / "id_label_audit.csv", [
        {"country": "DEU", "class_label": "urban", "correct": 1},
    ])
    _write(folder / "ood_label_audit.csv", [
        {"country": "DEU", "class_label": "urban", "correct": 0},
    ])
    rows, _ = aggregate_reben_country_label_burden(tmp_path)
    assert rows[0]["mean_delta_risk"] == 1.0


def test_fmow_three_seed_aggregation_is_strict_and_seed_first(tmp_path: Path) -> None:
    paths = []
    for seed, risks in ((101, (0, 1)), (202, (1, 1)), (303, (0, 0))):
        path = tmp_path / f"seed_{seed}" / "formal_audit_table.csv"
        _write(path, [
            {"sample_id": f"{seed}-a", "location_id": "site-a", "latitude": 10,
             "longitude": 20, "risk": risks[0], "split": "test"},
            {"sample_id": f"{seed}-b", "location_id": "site-b", "latitude": 30,
             "longitude": 40, "risk": risks[1], "split": "test"},
        ])
        paths.append(path)
    rows, status = aggregate_coordinate_risk_across_seeds(paths, expected_seed_count=3)
    assert status["seed_count"] == 3
    assert status["seeds"] == ["101", "202", "303"]
    assert len(rows) == 2
    assert rows[0]["mean_risk"] == 1 / 3
    assert rows[0]["seed_count"] == 3
    manifest = build_geographic_risk_atlas(
        tmp_path / "three_seed_atlas", fmow_csvs={"ResNet50": paths},
        fmow_expected_seed_counts={"ResNet50": 3},
    )
    fmow = manifest["readiness"][0]
    assert fmow["seed_count"] == 3
    assert (tmp_path / "three_seed_atlas" / "fmow_ResNet50_coordinate_risk.png").is_file()
    assert all(row["status"] == "pass" for row in manifest["visual_qa"])


def test_fmow_three_seed_aggregation_rejects_unit_drift(tmp_path: Path) -> None:
    import pytest

    paths = []
    for seed, unit in ((101, "site-a"), (202, "site-a"), (303, "site-b")):
        path = tmp_path / f"seed_{seed}" / "formal_audit_table.csv"
        _write(path, [{"location_id": unit, "latitude": 10, "longitude": 20, "risk": 0}])
        paths.append(path)
    with pytest.raises(ValueError, match="universe differs"):
        aggregate_coordinate_risk_across_seeds(paths, expected_seed_count=3)


def test_reben_full_panel_layout_emits_no_margin_warning(tmp_path: Path) -> None:
    import warnings

    rows = [
        {"country": f"C{country:02d}", "class_label": f"long_label_{label:02d}",
         "mean_delta_risk": (country - label) / 100}
        for country in range(10) for label in range(19)
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = plot_reben_burden(rows, tmp_path)
    assert not [item for item in caught if issubclass(item.category, UserWarning)]
    assert all(path.is_file() for path in paths)
