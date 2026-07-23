from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.bwer_protocol import BWERProtocol, METRIC_VERSION, Validity
from rsfm_fairness_audit.bwer_risk_adapters import (
    conformal_audit_rows,
    multiclass_audit_rows,
    multilabel_audit_rows,
    segmentation_audit_rows,
    selective_subset,
)
from rsfm_fairness_audit.bwer_schema import class_mapping_hash, validate_formal_audit_rows
from rsfm_fairness_audit.geobwer import audit


def test_task_risk_adapters_produce_bounded_primary_losses() -> None:
    multiclass = multiclass_audit_rows([[0.8, 0.2], [0.1, 0.9]], [0, 0], class_names=["a", "b"])
    assert [row["risk"] for row in multiclass] == [0.0, 1.0]
    assert len(multiclass[0]["class_mapping_hash"]) == 64
    multilabel = multilabel_audit_rows([[0.8, 0.2], [0.4, 0.9]], [[1, 0], [1, 1]])
    assert all(0.0 <= row["risk"] <= 1.0 for row in multilabel)
    segmentation = segmentation_audit_rows([{"TP": 8, "FP": 1, "FN": 1, "TN": 10}])
    assert segmentation[0]["risk"] == pytest.approx(1.0 - 8.0 / 10.0)
    conformal = conformal_audit_rows([[0, 1], [1]], [0, 0], number_of_classes=2)
    assert [row["risk"] for row in conformal] == [0.0, 1.0]


def test_selective_subset_is_fixed_coverage_not_conformal() -> None:
    rows = [{"sample_id": index, "confidence": index / 10, "risk": float(index % 2)} for index in range(10)]
    result = selective_subset(rows, coverage=0.5)
    assert len(result.rows) == 5
    assert result.retained_coverage == pytest.approx(0.5)
    assert result.threshold == pytest.approx(0.5)


def test_formal_schema_requires_lineage_and_probability_mapping() -> None:
    row = {
        "dataset": "fmow",
        "model": "dofa2",
        "task": "classification",
        "split": "test",
        "sample_id": "1",
        "independent_unit_id": "loc1",
        "risk": 0.0,
        "split_role": "test",
        "model_signature": "m",
        "dataset_signature": "d",
        "protocol_hash": "p",
        "metric_version": METRIC_VERSION,
        "probability_vector": (0.9, 0.1),
        "class_mapping_hash": class_mapping_hash(["a", "b"]),
    }
    result = validate_formal_audit_rows([row], task_adapter="multiclass", require_probabilities=True)
    assert result.ok
    broken = dict(row)
    broken.pop("independent_unit_id")
    assert validate_formal_audit_rows([broken], task_adapter="multiclass").validity == Validity.MISSING_INDEPENDENT_UNIT
    duplicate = validate_formal_audit_rows([row, dict(row)], task_adapter="multiclass")
    assert not duplicate.ok
    assert any("duplicate" in error for error in duplicate.errors)


def test_public_api_fails_fast_without_formal_clusters() -> None:
    protocol = BWERProtocol(min_units_per_slice=1, min_clusters_for_default=2)
    with pytest.raises(ValueError, match="requires cluster_id"):
        audit(loss=[0.1, 0.2, 0.8, 0.9], groups=["a", "a", "b", "b"], unit_id=[1, 2, 3, 4], protocol=protocol)
    result = audit(
        loss=[0.1, 0.2, 0.8, 0.9],
        groups=["a", "a", "b", "b"],
        unit_id=[1, 2, 3, 4],
        protocol=BWERProtocol(inference_method="none"),
        formal=False,
    )
    assert result.axes[0].point is not None


def test_report_card_is_versioned_and_json_serializable() -> None:
    groups = np.repeat(["a", "b"], 40)
    losses = np.concatenate([np.linspace(0.1, 0.2, 40), np.linspace(0.4, 0.5, 40)])
    clusters = [f"{group}_{index}" for group in ["a", "b"] for index in range(40)]
    result = audit(
        loss=losses,
        groups=groups,
        unit_id=range(80),
        cluster_id=clusters,
        protocol=BWERProtocol(min_clusters_for_default=30),
        n_bootstrap=300,
    )
    output = Path("work/geobwer_report_test")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = result.to_report(output)
    payload = json.loads(artifacts["report_card"].read_text(encoding="utf-8"))
    assert payload["protocol_hash"] == result.protocol.signature
    assert artifacts["summary"].exists()


def test_formal_api_refuses_unimplemented_slice_superpopulation_target() -> None:
    protocol = BWERProtocol(
        beta=0.5,
        beta_profile=(0.5,),
        inference_target="slice_superpopulation",
        estimand_scope="slice_superpopulation",
        inference_method="cluster_maxt",
        min_clusters_for_default=2,
    )
    with pytest.raises(ValueError, match="slice_superpopulation"):
        audit(
            loss=[0.1, 0.2, 0.3, 0.4],
            groups=["a", "a", "b", "b"],
            unit_id=["u1", "u2", "u3", "u4"],
            cluster_id=["c1", "c2", "c3", "c4"],
            protocol=protocol,
            formal=True,
            n_bootstrap=100,
        )


def test_partial_standardization_reports_bwer_identification_bounds():
    result = audit(
        loss=[0.1, 0.9, 0.2],
        groups=["A", "A", "B"],
        balance=["x", "y", "x"],
        unit_id=["u1", "u2", "u3"],
        protocol=BWERProtocol(
            beta=0.5,
            beta_profile=(0.5,),
            missingness_rule="partial_bounds",
            inference_method="none",
        ),
        formal=False,
    )
    axis = result.axes[0]
    assert axis.validity == Validity.NOT_IDENTIFIED
    assert axis.point is None
    assert axis.partial_bounds is not None
    assert 0.0 <= axis.partial_bounds.lower <= axis.partial_bounds.upper <= 1.0


def test_selective_audit_marks_zero_coverage_group_not_identified() -> None:
    result = audit(
        loss=[0.1, 0.2],
        groups={"country": ["A", "A"]},
        unit_id=["u1", "u2"],
        required_group_universe={"country": ("A", "B")},
        protocol=BWERProtocol(inference_method="none", group_variable="country"),
        formal=False,
    )
    axis = result.axes[0]
    assert axis.validity == Validity.NOT_IDENTIFIED_SELECTIVE_COVERAGE
    assert axis.point is None
    assert dict(axis.group_support)["B"] == 0
