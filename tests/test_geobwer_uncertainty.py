from __future__ import annotations

import numpy as np

from rsfm_fairness_audit.geobwer_uncertainty import (
    apply_selective_threshold,
    conformal_quantile,
    crc_audit_rows,
    fit_false_negative_crc,
    fit_multiclass_conformal,
    fit_selective_threshold,
    multiclass_conformal_audit_rows,
    multiclass_nonconformity_scores,
    multiclass_prediction_sets,
)


def test_split_conformal_quantile_uses_finite_sample_rank():
    assert conformal_quantile([0.1, 0.2, 0.3, 0.4], alpha=0.2) == 0.4


def test_lac_calibration_and_test_audit_keep_coverage_and_efficiency_separate():
    probabilities = np.asarray(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6], [0.5, 0.4, 0.1]], dtype=float
    )
    targets = np.asarray([0, 1, 2, 1])
    model = fit_multiclass_conformal(probabilities, targets, alpha=0.25, method="lac")
    rows = multiclass_conformal_audit_rows(
        model,
        probabilities,
        targets,
        sample_rows=[{"sample_id": str(index)} for index in range(4)],
    )
    assert all("set_size" in row and "risk" in row for row in rows)
    assert all(row["set_size"] >= 1 for row in rows)


def test_deterministic_aps_uses_same_candidate_score_for_fit_and_prediction():
    probabilities = np.asarray([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])
    targets = np.asarray([1, 0])
    scores = multiclass_nonconformity_scores(probabilities, targets, method="aps")
    np.testing.assert_allclose(scores, [0.6, 0.0])
    sets = multiclass_prediction_sets(probabilities, 0.6, method="aps")
    assert sets[0, 1]
    assert sets[1, 0]


def test_mondrian_groups_below_support_fall_back_to_global_threshold():
    probabilities = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7]])
    model = fit_multiclass_conformal(
        probabilities,
        [0, 0, 1, 1],
        calibration_groups=["A", "A", "B", "C"],
        minimum_group_calibration_support=2,
    )
    assert set(dict(model.group_thresholds)) == {"A"}
    thresholds = model.threshold_for(["A", "C", "unseen"], 3)
    assert thresholds[1] == thresholds[2] == model.global_threshold


def test_selective_threshold_is_fit_on_calibration_then_applied_to_test():
    model = fit_selective_threshold([0.9, 0.8, 0.7, 0.1], target_coverage=0.5)
    rows = apply_selective_threshold(
        model,
        [0.0, 1.0, 0.0],
        [0.85, 0.75, 0.2],
        sample_rows=[{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}],
    )
    assert [row["accepted"] for row in rows] == [True, False, False]


def test_crc_controls_false_negative_risk_and_reports_set_cost():
    probabilities = np.asarray(
        [[0.9, 0.8, 0.1], [0.8, 0.2, 0.1], [0.7, 0.6, 0.2], [0.95, 0.4, 0.3]], dtype=float
    )
    targets = np.asarray([[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=int)
    model = fit_false_negative_crc(probabilities, targets, alpha=0.5)
    assert model.calibration_corrected_risk <= 0.5
    rows = crc_audit_rows(
        model,
        probabilities,
        targets,
        sample_rows=[{"sample_id": str(index)} for index in range(4)],
    )
    assert all(0.0 <= row["risk"] <= 1.0 for row in rows)
    assert all(0.0 <= row["prediction_set_fraction"] <= 1.0 for row in rows)
