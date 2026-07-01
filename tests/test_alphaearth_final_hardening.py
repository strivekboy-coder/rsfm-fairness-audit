from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from uuid import uuid4

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from scripts.analysis.build_alphaearth_final_evidence_hardening_v2 import build_hardening


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


def test_alphaearth_final_hardening_outputs() -> None:
    root = Path("outputs") / f"test_alphaearth_hardening_{uuid4().hex}"
    audit = root / "audit"
    unified = root / "unified"
    audit.mkdir(parents=True)
    rows = _prediction_rows()
    write_csv(audit / "alphaearth_full_predictions.csv", rows)
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
        manifest=None,
        seeds="42",
        max_scales=None,
    )
    paths = build_hardening(args)
    for path in paths.values():
        assert path.exists(), path
    assert read_csv_rows(audit / "alphaearth_scale_sensitivity_repeated.csv")
    assert read_csv_rows(audit / "alphaearth_dynamic_world_agreement.csv")
    assert read_csv_rows(audit / "alphaearth_conformal_slice_gap_diagnostic.csv")
    assert (unified / "rsfm_bwer_paper_freeze_v4.zip").exists()
