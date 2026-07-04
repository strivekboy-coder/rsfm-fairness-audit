from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from uuid import uuid4

import pytest

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from scripts.analysis.build_alphaearth_final_evidence_hardening_v2 import build_hardening
from scripts.analysis.check_alphaearth_full_export_schema import EMBEDDING_BANDS


def _prediction_rows() -> list[dict[str, object]]:
    rows = []
    labels = [("30", "Grassland"), ("40", "Cropland"), ("20", "Shrubland")]
    for idx in range(90):
        label, class_name = labels[idx % len(labels)]
        correct = idx % 5 != 0
        pred = label if correct else "40"
        rows.append(
            {
                "sample_id": f"sample_{idx}",
                "split": "test" if idx % 2 == 0 else "calibration",
                "country_iso3": ["USA", "BRA", "IND"][idx % 3],
                "region": ["North America", "Latin America", "South Asia"][idx % 3],
                "worldcover_class_name": class_name,
                "class_label": class_name,
                "label": label,
                "prediction": pred,
                "predicted_class_name": "Cropland" if pred == "40" else class_name,
                "correct": int(correct),
                "risk": 1 - int(correct),
                "confidence": 0.8 if correct else 0.45,
                "dynamic_world_label": label,
                "dynamic_world_confidence": 0.7,
                "spatial_block_id": f"block_{idx // 3}",
            }
        )
    return rows


def _export_rows() -> list[dict[str, object]]:
    rows = []
    for row in _prediction_rows():
        item = {
            "sample_id": row["sample_id"],
            "lon": 0.01 * len(rows),
            "lat": 0.02 * len(rows),
            "year": 2021,
            "country_iso3": row["country_iso3"],
            "region": row["region"],
            "income_group": "High income",
            "biome_or_ecoregion": "",
            "urban_rural_or_built_proxy": "non_built_proxy",
            "spatial_block_id": row["spatial_block_id"],
            "split": row["split"],
            "worldcover_label": row["label"],
            "worldcover_class_name": row["class_label"],
        }
        for band_index, band in enumerate(EMBEDDING_BANDS):
            item[band] = float(int(row["label"]) * 0.01 + (band_index % 5) * 0.001)
        rows.append(item)
    return rows


def _dw_rows() -> list[dict[str, object]]:
    rows = []
    for row in _prediction_rows():
        rows.append(
            {
                "sample_id": row["sample_id"],
                "worldcover_label": -1,
                "dynamic_world_label": {"80": 0, "10": 1, "30": 2, "90": 3, "40": 4, "20": 5}.get(str(row["label"]), 2),
                "dw_confidence": 0.7,
                "dw_top_probability": 0.72,
                "dw_entropy": 0.28,
                "alphaearth_prediction": -1,
            }
        )
    return rows


def test_alphaearth_final_hardening_outputs() -> None:
    root = Path("outputs") / f"test_alphaearth_hardening_{uuid4().hex}"
    audit = root / "audit"
    unified = root / "unified"
    audit.mkdir(parents=True)
    rows = _prediction_rows()
    write_csv(audit / "alphaearth_full_predictions.csv", rows[:30])
    write_csv(audit / "alphaearth_full_all_split_predictions.csv", rows)
    write_csv(audit / "alphaearth_full_dw_aligned.csv", _dw_rows())
    shard = root / "synthetic_shard.csv"
    manifest = root / "manifest.csv"
    write_csv(shard, _export_rows())
    write_csv(manifest, [{"shard_id": "synthetic", "path": str(shard.resolve()), "status": "available", "bytes": shard.stat().st_size}])
    write_csv(
        audit / "alphaearth_full_conformal_slice_coverage.csv",
        [
            {"coverage_target": "0.9", "slice_variable": "region", "slice_value": "North America", "support_count": 30, "slice_coverage": 0.83, "average_set_size": 1.4},
            {"coverage_target": "0.9", "slice_variable": "worldcover_class_name", "slice_value": "Grassland", "support_count": 30, "slice_coverage": 0.8, "average_set_size": 1.7},
        ],
    )
    args = Namespace(
        audit_root=audit,
        reference_audit_root=root / "missing_reference",
        unified_v4_out=unified,
        input=root / "missing_input.csv",
        manifest=manifest,
        seeds="42,73,101",
        max_scales=None,
    )
    paths = build_hardening(args)
    for path in paths.values():
        assert path.exists(), path
    assert read_csv_rows(audit / "alphaearth_scale_sensitivity_repeated.csv")
    dynamic = read_csv_rows(audit / "alphaearth_dynamic_world_agreement.csv")
    assert {"worldcover_dynamicworld_agreement", "alphaearth_worldcover_accuracy", "alphaearth_dynamicworld_agreement"}.issubset({row["metric"] for row in dynamic})
    assert {"test_only", "eval_calibration_test", "all_split_descriptive"}.issubset({row.get("scope") for row in dynamic})
    accuracy = next(row for row in dynamic if row["metric"] == "alphaearth_worldcover_accuracy" and row["group"] == "all")
    assert 0 < float(accuracy["value"]) < 1
    validation = next(row for row in dynamic if row["metric"] == "dw_aligned_table_validation" and row["scope"] == "all_split_descriptive")
    assert int(validation["raw_aligned_placeholder_label_count"]) > 0
    assert int(validation["matched_prediction_rows"]) == len(rows)
    assert int(validation["prediction_table_rows"]) == len(rows)
    eval_validation = next(row for row in dynamic if row["metric"] == "dw_aligned_table_validation" and row["scope"] == "eval_calibration_test")
    assert int(eval_validation["matched_prediction_rows"]) == 30
    assert any(row["group"] == "dw_confidence_bin" and row["group_value"] != "missing" for row in dynamic)
    assert any(row["group"] == "dw_entropy_bin" and row["group_value"] != "missing" for row in dynamic)
    assert read_csv_rows(audit / "alphaearth_conformal_slice_gap_diagnostic.csv")
    assert (unified / "rsfm_bwer_paper_freeze_v4.zip").exists()


def test_alphaearth_final_hardening_requires_manifest_for_formal_scale() -> None:
    root = Path("outputs") / f"test_alphaearth_hardening_no_manifest_{uuid4().hex}"
    audit = root / "audit"
    audit.mkdir(parents=True)
    write_csv(audit / "alphaearth_full_predictions.csv", _prediction_rows())
    args = Namespace(
        audit_root=audit,
        reference_audit_root=root / "missing_reference",
        unified_v4_out=root / "unified",
        input=root / "missing_input.csv",
        manifest=None,
        seeds="42",
        max_scales=None,
    )
    with pytest.raises(FileNotFoundError, match="requires --manifest"):
        build_hardening(args)


def test_alphaearth_dynamic_world_invalid_without_prediction_join() -> None:
    root = Path("outputs") / f"test_alphaearth_hardening_bad_dw_{uuid4().hex}"
    audit = root / "audit"
    unified = root / "unified"
    audit.mkdir(parents=True)
    write_csv(audit / "alphaearth_full_dw_aligned.csv", _dw_rows())
    shard = root / "synthetic_shard.csv"
    manifest = root / "manifest.csv"
    write_csv(shard, _export_rows())
    write_csv(manifest, [{"shard_id": "synthetic", "path": str(shard.resolve()), "status": "available", "bytes": shard.stat().st_size}])
    args = Namespace(
        audit_root=audit,
        reference_audit_root=root / "missing_reference",
        unified_v4_out=unified,
        input=root / "missing_input.csv",
        manifest=manifest,
        seeds="42",
        max_scales=1,
    )
    build_hardening(args)
    dynamic = read_csv_rows(audit / "alphaearth_dynamic_world_agreement.csv")
    assert dynamic[0]["status"] == "invalid"
    assert dynamic[0]["reason"] == "no_valid_prediction_label_join"
    assert "alphaearth_worldcover_accuracy" not in {row.get("metric") for row in dynamic}
