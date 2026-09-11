from __future__ import annotations

import math

import numpy as np

from rsfm_fairness_audit.paper_supplementary import (
    ClusterSliceSufficient,
    build_sen1_consensus_event_risk,
    build_sen1_decision_scenario,
    cluster_bootstrap_mtd,
    cluster_bootstrap_standardized_mtd,
    compute_fractional_mtd,
    infer_verified_datetime,
    robust_water_land_contrast,
    summarize_sen1_validation_locked_mtd,
    summarize_paired_multilabel_arrays,
)
from scripts.analysis.build_cluster_mtd_intervals import aggregate, verify_canonical_point


def test_fractional_mtd_uses_exact_tail_mass():
    result = compute_fractional_mtd({"a": 0.0, "b": 0.2, "c": 0.8}, beta=0.5)
    assert math.isclose(result["M"], 1.0 / 3.0)
    assert math.isclose(result["T"], (0.8 / 3 + 0.2 / 6) / 0.5)
    assert math.isclose(result["D"], result["T"] - result["M"])


def test_sen1_decision_scenario_keeps_observed_event_pairing():
    rows = []
    for seed, s2, fusion in ((42, [0.2, 0.8], [0.1, 0.9]), (73, [0.2, 0.8], [0.1, 0.9]), (101, [0.2, 0.8], [0.1, 0.9])):
        for event, value in zip(("A", "B"), s2):
            rows.append({"dataset": "Sen1Floods11", "slice_axis": "event", "model_family": "supervised_resnet34_unet", "mode": "S2", "slice_value": event, "risk": value, "support": 5, "eligible_for_primary_metric": True})
        for event, value in zip(("A", "B"), fusion):
            rows.append({"dataset": "Sen1Floods11", "slice_axis": "event", "model_family": "supervised_resnet34_unet", "mode": "S1+S2", "slice_value": event, "risk": value, "support": 5, "eligible_for_primary_metric": True})
    detail, summary = build_sen1_decision_scenario(rows)
    assert len(detail) == 2
    assert {row["mode"] for row in summary} == {"S2", "S1+S2"}
    assert {row["event_winner"] for row in detail} == {"S2", "S1+S2"}


def _canonical_sen1_rows():
    rows = []
    for event_index, event in enumerate((f"E{i}" for i in range(11))):
        for family in ("supervised_resnet34_unet", "terramind_v1_base"):
            for mode in ("S1", "S2", "S1+S2"):
                for seed in (42, 73, 101):
                    rows.append({
                        "family": family, "mode": mode, "seed": seed,
                        "split": "combined_held_out", "comparison_role": "same_grid_primary_panel",
                        "event_id": event, "auditable_sample_count": 5,
                        "mean_chip_iou_risk": 0.1 * event_index + (0.01 if mode == "S1+S2" else 0.02),
                    })
    return rows


def test_canonical_sen1_event_metrics_replace_local_derivatives():
    rows = _canonical_sen1_rows()
    detail, summary = build_sen1_decision_scenario(rows)
    assert len(detail) == 11
    assert {row["mode"] for row in summary} == {"S2", "S1+S2"}
    consensus = build_sen1_consensus_event_risk(rows)
    assert len(consensus) == 11
    assert {row["replicate_count"] for row in consensus} == {18}


def test_canonical_sen1_consensus_rejects_incomplete_seed_panel():
    rows = _canonical_sen1_rows()
    rows.pop()
    with __import__("pytest").raises(RuntimeError, match="Incomplete canonical Sen1 panel"):
        build_sen1_consensus_event_risk(rows)


def test_validation_locked_sen1_mtd_replaces_thesis_derived_omnibus():
    rows = []
    for mode in ("s1_plus_s2", "s2"):
        for index, seed in enumerate((42, 73, 101)):
            rows.append({
                "family": "supervised_resnet34_unet", "mode": mode, "seed": seed,
                "split": "combined_held_out", "is_validation_selected_operating_point": True,
                "event_mean_risk": 0.4 + index * 0.01,
                "event_tail_risk": 0.6 + index * 0.01,
                "event_geobwer": 0.2,
            })
    result = summarize_sen1_validation_locked_mtd(rows)
    assert math.isclose(result["s2"]["M"], 0.41)
    assert math.isclose(result["s1_plus_s2"]["T_seed_sd"], 0.01)


def test_multilabel_summary_is_transition_not_exclusive_confusion():
    truth = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int8)
    id_pred = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int8)
    shifted = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int8)
    rows, before, after = summarize_paired_multilabel_arrays(truth, id_pred, shifted, ["wetland", "urban"])
    assert rows[0]["delta_fnr"] == 0.5
    assert rows[1]["delta_fnr"] == 0.5
    assert before.sum() == 0
    assert after[0, 1] == 1


def test_cluster_bootstrap_computes_and_blocks_one_cluster_slices():
    records = [
        ClusterSliceSufficient("c1", "A", 1.0, 2),
        ClusterSliceSufficient("c2", "A", 0.0, 2),
        ClusterSliceSufficient("c1", "B", 2.0, 2),
        ClusterSliceSufficient("c2", "B", 1.0, 2),
    ]
    result = cluster_bootstrap_mtd(records, beta=.5, n_boot=200, seed=1)
    assert result["status"] == "computed"
    assert result["M_ci_low"] <= result["M"] <= result["M_ci_high"]
    blocked = cluster_bootstrap_mtd([ClusterSliceSufficient("event-a", "A", .5, 1), ClusterSliceSufficient("event-b", "B", .4, 1)])
    assert blocked["status"] == "insufficient_independent_clusters"


def test_dates_are_not_invented_and_contrast_is_scale_resistant():
    assert infer_verified_datetime({}, "Pakistan_123_S1Hand.tif") == ("", "unavailable_not_imputed")
    assert infer_verified_datetime({"ACQUISITION_DATE": "2021-08-09"})[0] == "2021-08-09"
    image = np.array([[[1.0, 1.0], [5.0, 5.0]]])
    flood = np.array([[False, False], [True, True]])
    valid = np.ones((2, 2), dtype=bool)
    assert robust_water_land_contrast(image, flood, valid) > 0


def test_reben_cluster_aggregator_derives_hamming_risk_across_chunks():
    fixture = __import__("pathlib").Path(__file__).parent / "fixtures/paper_supplementary/reben_label_audit.csv"
    totals, lineage = aggregate(fixture, axis="country", chunksize=4)
    assert lineage["risk_column"] == "derived_label_mismatch_hamming_primitive"
    assert lineage["cluster_column"] == "derived_source_tile_from_sample_id"
    assert totals[("T31UFS", "BEL", "")] == [2.0, 4]
    assert totals[("T29SNC", "PRT", "")] == [0.0, 2]


def test_standardized_cluster_bootstrap_preserves_equal_balance_weights():
    records = []
    for cluster in ("c1", "c2"):
        records.extend([
            ClusterSliceSufficient(cluster, "A", 0.0, 10, "common"),
            ClusterSliceSufficient(cluster, "A", 1.0, 1, "rare"),
            ClusterSliceSufficient(cluster, "B", 5.0, 10, "common"),
            ClusterSliceSufficient(cluster, "B", 0.0, 1, "rare"),
        ])
    result = cluster_bootstrap_standardized_mtd(records, beta=.5, n_boot=200, seed=2)
    assert result["status"] == "computed"
    assert math.isclose(result["M"], 0.375)
    assert math.isclose(result["T"], 0.5)


def test_canonical_point_verification_is_exact_and_explicit():
    fixture = __import__("pathlib").Path(__file__).parent / "fixtures/paper_supplementary/geobwer_summary.csv"
    verified = verify_canonical_point(
        {"M": 0.375, "T": 0.5, "D": 0.125},
        {"canonical_summary_path": str(fixture), "canonical_axis": "country", "balance_col": "class_label"},
    )
    assert verified["point_reconstruction_verified"] is True
