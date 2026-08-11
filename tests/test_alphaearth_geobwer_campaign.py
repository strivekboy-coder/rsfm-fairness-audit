from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np
import pytest

from rsfm_fairness_audit.alphaearth_geobwer_campaign import (
    AlphaEarthCampaignError,
    _probabilities_targets,
    _read_predictions,
    _split_rows,
    _formal_protocol,
)
from rsfm_fairness_audit.bwer_protocol import BWERProtocol


WORK = Path("work/test_alphaearth_geobwer_campaign")


@pytest.fixture(autouse=True)
def _clean_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    yield
    if WORK.exists():
        shutil.rmtree(WORK)


def test_reads_probability_columns_in_frozen_header_order() -> None:
    path = WORK / "predictions.csv"
    fields = [
        "sample_id",
        "split",
        "spatial_block_id",
        "label",
        "prob_10",
        "prob_100",
        "prob_20",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sample_id": "cal",
                    "split": "calibration",
                    "spatial_block_id": "old-a",
                    "label": "100",
                    "prob_10": 0.1,
                    "prob_100": 0.8,
                    "prob_20": 0.1,
                },
                {
                    "sample_id": "test",
                    "split": "test",
                    "spatial_block_id": "old-b",
                    "label": "20",
                    "prob_10": 0.2,
                    "prob_100": 0.1,
                    "prob_20": 0.7,
                },
            ]
        )
    rows, columns, classes = _read_predictions(path)
    assert columns == ("prob_10", "prob_100", "prob_20")
    assert classes == ("10", "100", "20")
    calibration, test = _split_rows(rows)
    calibration_probs, calibration_targets = _probabilities_targets(calibration, columns, classes)
    test_probs, test_targets = _probabilities_targets(test, columns, classes)
    np.testing.assert_allclose(calibration_probs, [[0.1, 0.8, 0.1]])
    np.testing.assert_array_equal(calibration_targets, [1])
    np.testing.assert_array_equal(test_targets, [2])
    np.testing.assert_allclose(test_probs.sum(axis=1), 1.0)


def test_rejects_legacy_spatial_block_leakage() -> None:
    rows = [
        {"sample_id": "cal", "split": "calibration", "spatial_block_id": "same"},
        {"sample_id": "test", "split": "test", "spatial_block_id": "same"},
    ]
    with pytest.raises(AlphaEarthCampaignError, match="leakage"):
        _split_rows(rows)


def test_formal_protocol_preserves_calibrated_inference_threshold() -> None:
    base = BWERProtocol(min_clusters_per_slice=2, min_clusters_for_inference=75)
    protocol = _formal_protocol(
        base, metadata={"spatial_block_calibrated": "true"},
        calibration_signature="frozen-alpha-calibration",
    )
    assert protocol.min_clusters_per_slice == 2
    assert protocol.min_clusters_for_inference == 75
    assert protocol.cluster_eligibility_calibration_signature == "frozen-alpha-calibration"
