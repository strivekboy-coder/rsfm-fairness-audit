from __future__ import annotations

import math

import numpy as np
import pytest

from rsfm_fairness_audit.bwer_core import (
    compute_geobwer,
    compute_geobwer_profile,
    fractional_tail_allocation,
    legacy_whole_slice_bwer,
)
from rsfm_fairness_audit.bwer_protocol import BWERProtocol, Validity
from rsfm_fairness_audit.bwer_standardization import common_group_support, partial_bwer_bounds, standardize_group_risks


def test_fractional_bwer_embeds_legacy_at_integer_equal_mass() -> None:
    risks = {f"g{index}": value for index, value in enumerate([0.9, 0.7, 0.4, 0.2, 0.1])}
    for beta in (0.2, 0.4, 0.6, 1.0):
        assert compute_geobwer(risks, beta).bwer == pytest.approx(legacy_whole_slice_bwer(risks, beta))


def test_fractional_tail_has_exact_mass_for_small_g() -> None:
    risks = {f"event_{index}": index / 10 for index in range(11)}
    point = compute_geobwer(risks, beta=0.10)
    selected = dict(point.allocation.selected_mass)
    assert sum(selected.values()) == pytest.approx(0.10)
    assert selected["event_10"] == pytest.approx(1.0 / 11.0)
    assert selected["event_9"] == pytest.approx(0.10 - 1.0 / 11.0)
    assert 1.0 < point.allocation.tail_effective_groups < 2.0


def test_bwer_profile_endpoints_and_monotonicity() -> None:
    risks = {"a": 0.8, "b": 0.5, "c": 0.1}
    profile = compute_geobwer_profile(risks, (0.01, 0.2, 0.5, 1.0))
    assert profile[0].bwer == pytest.approx(max(risks.values()) - np.mean(list(risks.values())))
    assert profile[-1].bwer == pytest.approx(0.0)
    assert all(left.bwer >= right.bwer - 1e-12 for left, right in zip(profile, profile[1:]))


def test_mass_preserving_equal_risk_clone_is_invariant() -> None:
    original = compute_geobwer({"a": 0.8, "b": 0.2}, 0.3, {"a": 0.6, "b": 0.4})
    cloned = compute_geobwer({"a1": 0.8, "a2": 0.8, "b": 0.2}, 0.3, {"a1": 0.2, "a2": 0.4, "b": 0.4})
    assert cloned.mean_risk == pytest.approx(original.mean_risk)
    assert cloned.tail_risk == pytest.approx(original.tail_risk)
    assert cloned.bwer == pytest.approx(original.bwer)


def test_tie_value_is_unique_even_if_allocation_label_is_not() -> None:
    tail, allocation = fractional_tail_allocation({"z": 0.9, "a": 0.9, "b": 0.1}, 0.2)
    assert tail == pytest.approx(0.9)
    assert allocation.boundary_tie_groups == ("a", "z")


def test_protocol_signature_is_deterministic_and_sensitive() -> None:
    first = BWERProtocol(metadata=(("dataset", "fmow"),))
    second = BWERProtocol(metadata=(("dataset", "fmow"),))
    third = BWERProtocol(beta=0.2, metadata=(("dataset", "fmow"),))
    assert first.signature == second.signature
    assert first.signature != third.signature


def test_strict_standardization_rejects_group_specific_renormalization() -> None:
    rows = [
        {"g": "A", "z": "x", "risk": 0.1},
        {"g": "A", "z": "y", "risk": 0.9},
        {"g": "B", "z": "x", "risk": 0.1},
    ]
    strict = standardize_group_risks(rows, group_column="g", balance_column="z")
    assert strict.validity == Validity.NOT_IDENTIFIED
    assert ("B", "y") in strict.missing_cells
    overlap = standardize_group_risks(rows, group_column="g", balance_column="z", missingness_rule="overlap")
    assert overlap.validity == Validity.VALID
    assert dict(overlap.group_risks) == pytest.approx({"A": 0.1, "B": 0.1})


def test_custom_standardization_target_can_preregister_a_positive_mass_subset() -> None:
    rows = [
        {"g": "A", "z": "x", "risk": 0.2},
        {"g": "A", "z": "structurally_absent", "risk": 0.9},
        {"g": "B", "z": "x", "risk": 0.4},
    ]
    result = standardize_group_risks(
        rows,
        group_column="g",
        balance_column="z",
        target_weights={"x": 1.0},
    )
    assert result.validity == Validity.VALID
    assert dict(result.group_risks) == pytest.approx({"A": 0.2, "B": 0.4})
    assert result.used_balance_levels == ("x",)


def test_partial_identification_bounds_are_ordered() -> None:
    bounds = partial_bwer_bounds(
        {"A": 0.2, "B": 0.1},
        {"A": 0.6, "B": 0.5},
        beta=0.5,
    )
    assert bounds.validity == Validity.NOT_IDENTIFIED
    assert 0.0 <= bounds.lower <= bounds.upper <= 1.0


def test_common_support_is_model_independent_and_hashed() -> None:
    result = common_group_support({"a": {"x": 0.1, "y": 0.2}, "b": {"y": 0.3, "z": 0.4}})
    assert result.validity == Validity.NO_COMMON_SUPPORT
    result = common_group_support(
        {"a": {"x": 0.1, "y": 0.2, "q": 0.5}, "b": {"x": 0.3, "y": 0.4, "z": 0.6}}
    )
    assert result.validity == Validity.VALID
    assert result.groups == ("x", "y")
    assert len(result.support_hash) == 64


def test_invalid_beta_and_weights_fail_loudly() -> None:
    with pytest.raises(ValueError):
        compute_geobwer({"a": 0.1, "b": 0.2}, 0.0)
    with pytest.raises(ValueError):
        compute_geobwer({"a": 0.1, "b": 0.2}, 0.5, {"a": 1.0})
    with pytest.raises(ValueError):
        BWERProtocol(confidence_level=1.0)
    with pytest.raises(ValueError, match="requires standardization_weights"):
        BWERProtocol(standardization_target="custom")
