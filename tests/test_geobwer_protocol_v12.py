from __future__ import annotations

from dataclasses import replace

import pytest

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.cluster_eligibility import (
    ClusterEligibilityRule,
    ClusterEvidenceLevel,
    assess_cluster_eligibility,
)
from rsfm_fairness_audit.estimand_identity import build_estimand_identity
from rsfm_fairness_audit.evidence_registry import load_canonical_evidence_registry
from rsfm_fairness_audit.risk_spec import RiskSpec


def test_risk_spec_is_hashed_into_protocol_and_rejects_wrong_direction_or_bounds() -> None:
    protocol = BWERProtocol(loss_name="one_minus_iou", task_adapter="segmentation")
    assert protocol.risk_spec.name == "one_minus_iou"
    changed = replace(
        protocol,
        risk_spec=replace(protocol.risk_spec, reference="different_reference"),
    )
    assert changed.signature != protocol.signature
    with pytest.raises(ValueError, match="higher_is_worse"):
        RiskSpec(direction="higher_is_better")
    with pytest.raises(ValueError, match="outside"):
        protocol.risk_spec.validate_values([0.0, 1.1])


def test_estimand_identity_changes_with_partition_measure_or_universe() -> None:
    protocol = BWERProtocol()
    base = build_estimand_identity(protocol, axis="country", group_universe=("A", "B"))
    weighted = build_estimand_identity(
        protocol,
        axis="country",
        group_universe=("A", "B"),
        deployment_weights={"A": 0.8, "B": 0.2},
    )
    expanded = build_estimand_identity(protocol, axis="country", group_universe=("A", "B", "C"))
    assert not base.comparable_with(weighted)
    assert not base.comparable_with(expanded)


def test_cluster_eligibility_separates_computable_from_inferential() -> None:
    rule = ClusterEligibilityRule(
        min_units_per_group=5,
        min_clusters_per_group=3,
        min_total_clusters=30,
    )
    result = assess_cluster_eligibility(
        {"dense": 100, "sparse": 8, "missing": 0},
        {"dense": 20, "sparse": 2, "missing": 0},
        total_clusters=25,
        rule=rule,
    )
    levels = {record.group: record.level for record in result.groups}
    assert levels == {
        "dense": ClusterEvidenceLevel.DESCRIPTIVE_ONLY,
        "missing": ClusterEvidenceLevel.NOT_OBSERVED,
        "sparse": ClusterEvidenceLevel.DESCRIPTIVE_ONLY,
    }
    calibrated = assess_cluster_eligibility(
        {"dense": 100},
        {"dense": 20},
        total_clusters=25,
        rule=replace(rule, calibration_signature="frozen-simulation-contract"),
    )
    assert calibrated.groups[0].level == ClusterEvidenceLevel.INFERENTIAL_ELIGIBLE


def test_canonical_registry_has_unambiguous_sources_and_revocations() -> None:
    registry = load_canonical_evidence_registry(
        "configs/analysis/canonical_evidence_registry_v1.yaml"
    )
    assert registry.resolve(task="fmow_sentinel", role="dofav2_frozen_predictions").immutable
    assert registry.resolve(task="fmow_sentinel", role="resnet50_frozen_predictions").immutable
    assert {asset.asset_id for asset in registry.revoked()} >= {
        "alphaearth_spatial_v1_revoked",
        "sen1_prithvi_v0429_revoked",
    }
