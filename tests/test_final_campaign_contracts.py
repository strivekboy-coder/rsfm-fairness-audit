from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rsfm_fairness_audit.reben_terramind_campaign import build_reben_dataset_lineage
from rsfm_fairness_audit.sen1floods11_formal import (
    finalize_sen1_probability_export,
    write_sen1_probability_export,
)


def _write_metadata(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "sample_id,event_id,latitude,longitude,country",
                "Somalia_001,Somalia,2.0,45.0,Somalia",
                "USA_002,USA,35.0,-95.0,USA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_sen1_dataset_signature_is_model_and_sensor_independent(tmp_path):
    targets = [
        np.asarray([[0, 1], [1, 0]], dtype=np.int16),
        np.asarray([[1, 1], [0, 0]], dtype=np.int16),
    ]
    probabilities_a = [
        np.asarray([[0.1, 0.8], [0.7, 0.2]], dtype=np.float32),
        np.asarray([[0.8, 0.9], [0.2, 0.1]], dtype=np.float32),
    ]
    probabilities_b = [
        np.asarray([[0.2, 0.7], [0.6, 0.3]], dtype=np.float32),
        np.asarray([[0.7, 0.8], [0.3, 0.2]], dtype=np.float32),
    ]
    export_a = write_sen1_probability_export(
        tmp_path / "s1_export",
        probabilities=probabilities_a,
        targets=targets,
        filenames=[
            {"S1GRD": "Somalia_001_S1Hand.tif"},
            {"S1GRD": "USA_002_S1Hand.tif"},
        ],
    )
    export_b = write_sen1_probability_export(
        tmp_path / "s2_export",
        probabilities=probabilities_b,
        targets=targets,
        filenames=[
            {"S2L1C": "Somalia_001_S2Hand.tif"},
            {"S2L1C": "USA_002_S2Hand.tif"},
        ],
    )
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.common_spatial_block_calibration.v2",
                "selection_data": "validation_only",
                "validity": "valid",
                "all_models_passed": True,
                "selected_cell_km": 100.0,
            }
        ),
        encoding="utf-8",
    )
    project = Path(__file__).resolve().parents[1]
    protocol = project / "configs" / "geobwer" / "sen1floods11.yaml"
    bundles = []
    for name, export, mode in (
        ("model_s1", export_a, "S1"),
        ("model_s2", export_b, "S2"),
    ):
        bundle = finalize_sen1_probability_export(
            export,
            tmp_path / name,
            model_name=name,
            protocol_path=protocol,
            block_calibration_path=calibration,
            model_lineage={"model": name, "sensor_mode": mode},
            dataset_lineage={
                "dataset": "Sen1Floods11-v1.1-HandLabeled",
                "split": "test",
                "sensor_mode": mode,
            },
            metadata_csv=metadata,
        )
        bundles.append(bundle)
    manifests = [
        json.loads(bundle.manifest.read_text(encoding="utf-8")) for bundle in bundles
    ]
    assert bundles[0].dataset_signature == bundles[1].dataset_signature
    assert bundles[0].model_signature != bundles[1].model_signature
    assert "sensor_mode" not in manifests[0]["dataset_lineage"]
    assert manifests[0]["dataset_lineage"]["reference_targets_sha256"]


def test_reben_dataset_lineage_is_identical_across_sensor_models(tmp_path):
    metadata = tmp_path / "metadata.parquet"
    metadata.write_bytes(b"frozen-metadata")
    rows = [
        {"sample_id": "A", "source_tile_id": "31TCJ"},
        {"sample_id": "B", "source_tile_id": "32TLP"},
    ]
    targets = np.zeros((2, 19), dtype=np.int8)
    targets[0, 3] = 1
    left = build_reben_dataset_lineage(rows, targets, metadata_parquet=metadata)
    right = build_reben_dataset_lineage(rows, targets.copy(), metadata_parquet=metadata)
    assert left == right
    assert "sensor_mode" not in left
    changed = targets.copy()
    changed[1, 4] = 1
    assert (
        build_reben_dataset_lineage(rows, changed, metadata_parquet=metadata)[
            "reference_targets_sha256"
        ]
        != left["reference_targets_sha256"]
    )
