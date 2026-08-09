from __future__ import annotations

import csv
import json
from pathlib import Path
import uuid

from rsfm_fairness_audit.bwer_inference import paired_simultaneous_risk_boxes
from rsfm_fairness_audit.evidence_rebuild_v060 import (
    build_evidence_status_matrix,
    run_fmow_proper_score_sensitivity,
    run_fmow_same_seed_paired_v12,
    run_reben_labelwise_sensitivity,
    run_reben_fixed_universe_v12,
    run_sen1_event_geobwer,
)


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_joint_paired_box_covers_both_aligned_models() -> None:
    boxes = paired_simultaneous_risk_boxes(
        [0.0, 0.2, 0.8, 1.0], [0.1, 0.3, 0.7, 0.9],
        ["a", "a", "b", "b"], ["c1", "c2", "c3", "c4"],
        n_bootstrap=100, min_clusters_per_group=2,
    )
    assert boxes.validity.value == "valid"
    assert boxes.groups == ("a", "b")
    assert set(dict(boxes.lower_a)) == {"a", "b"}
    assert set(dict(boxes.lower_b)) == {"a", "b"}


def _workspace_case() -> Path:
    path = Path("work") / f"evidence_rebuild_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_fmow_partial_pair_keeps_sparse_group_as_risk_box() -> None:
    tmp_path = _workspace_case()
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    rows_a = [
        {"sample_id": f"s{i}", "risk": float(i % 2), "country": "A" if i < 4 else "B", "site_id": f"z{i}"}
        for i in range(6)
    ]
    rows_b = [{**row, "risk": 1.0 - float(row["risk"])} for row in rows_a]
    _write(a, rows_a); _write(b, rows_b)
    paths = run_fmow_same_seed_paired_v12(
        dofa_tables={42: a}, resnet_tables={42: b}, output_dir=tmp_path / "out",
        axes=("country",), min_clusters=3, n_bootstrap=100,
    )
    result = list(csv.DictReader(paths["results"].open()))[0]
    assert result["evidence_status"] == "formal_partial"
    assert result["eligible_group_count"] == "1"


def test_reben_v12_refuses_old_two_cluster_support_as_formal() -> None:
    tmp_path = _workspace_case()
    metrics = tmp_path / "metrics.csv"
    _write(metrics, [{"run_id":"m","family":"f","mode":"s1","seed":"42",
        "deployment_mean_risk":"0.2","tail_risk":"0.4","geobwer":"0.2"}])
    support = tmp_path / "support.json"
    support.write_text(json.dumps({"fixed_countries":["A","B"],"support_rows":[
        {"country":"A","cluster_count":7},{"country":"B","cluster_count":2}]}))
    paths = run_reben_fixed_universe_v12(
        unified_metrics=metrics, support_universe=support, output_dir=tmp_path / "out", min_clusters=75)
    row = list(csv.DictReader(paths["results"].open()))[0]
    assert row["eligible_country_count"] == "0"
    assert row["evidence_status"] == "formal_partial"


def test_sen1_event_profile_is_descriptive_and_separates_splits() -> None:
    tmp_path = _workspace_case()
    source = tmp_path / "events.csv"
    _write(source, [
        {"model":"m","split":"standard_test","event_id":"A","mean_chip_iou_risk":"0.2"},
        {"model":"m","split":"standard_test","event_id":"B","mean_chip_iou_risk":"0.8"},
        {"model":"m","split":"combined_heldout","event_id":"A","mean_chip_iou_risk":"0.2"},
        {"model":"m","split":"combined_heldout","event_id":"Bolivia","mean_chip_iou_risk":"0.9"},
    ])
    paths = run_sen1_event_geobwer(event_metrics=source, output_dir=tmp_path / "out", betas=(0.1,))
    rows = list(csv.DictReader(paths["profile"].open()))
    assert {row["split"] for row in rows} == {"standard_test", "combined_heldout"}
    assert all(row["evidence_status"] == "descriptive_only" for row in rows)


def test_fmow_proper_score_is_descriptive_and_paired() -> None:
    tmp_path = _workspace_case()
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    rows = [
        {"sample_id": f"s{i}", "country": "A" if i < 2 else "B", "log_loss": str(0.2 + i)}
        for i in range(4)
    ]
    _write(a, rows)
    _write(b, [{**row, "log_loss": str(float(row["log_loss"]) + 0.1)} for row in rows])
    paths = run_fmow_proper_score_sensitivity(
        dofa_tables={42: a}, resnet_tables={42: b}, output_dir=tmp_path / "out", axes=("country",))
    result = list(csv.DictReader(paths["results"].open()))[0]
    assert result["risk"] == "multiclass_log_loss_nats"
    assert result["evidence_status"] == "descriptive_only"


def test_evidence_matrix_uses_explicit_status_records() -> None:
    tmp_path = _workspace_case()
    paths = build_evidence_status_matrix(task_records=[{
        "task": "x", "mechanism": "m", "status": "formal_partial"
    }], output_dir=tmp_path / "out")
    rows = list(csv.DictReader(paths["matrix"].open()))
    assert rows == [{"mechanism": "m", "status": "formal_partial", "task": "x"}]


def test_reben_labelwise_uses_frozen_thresholds_without_test_selection() -> None:
    import numpy as np
    tmp_path = _workspace_case(); bundles = tmp_path / "bundles"; bundles.mkdir()
    sample_id = np.asarray(["a", "b", "c", "d"])
    targets = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int8)
    probabilities = np.asarray([[.8, .2], [.3, .7], [.4, .1], [.2, .6]], dtype=np.float32)
    np.savez_compressed(bundles / "croma__s1__seed_42.npz", sample_id=sample_id,
        probabilities=probabilities, targets=targets, class_names=np.asarray(["x", "y"]),
        thresholds=np.asarray([.5, .5], dtype=np.float32), threshold=np.asarray([.5, .5], dtype=np.float32))
    metrics = tmp_path / "metrics.csv"
    _write(metrics, [{"run_id": "croma__s1__seed_42", "geobwer": "0.1"}])
    paths = run_reben_labelwise_sensitivity(probability_dir=bundles, unified_metrics=metrics,
        output_dir=tmp_path / "out", expected_runs=1, expected_samples=4, expected_labels=2, betas=(.1,))
    rows = list(csv.DictReader(paths["labelwise"].open()))
    assert len(rows) == 2
    assert all(row["threshold_source"] == "frozen_validation_calibrated_per_label" for row in rows)
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["test_used_for_threshold_selection"] is False
