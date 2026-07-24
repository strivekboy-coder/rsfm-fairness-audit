from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.probe_selection import (
    MulticlassProbeSearchConfig,
    ProbeSelectionError,
    fit_selected_multiclass_probe,
    group_disjoint_inner_split,
    group_stratified_inner_split,
)


def test_group_stratified_inner_split_is_group_disjoint_and_deterministic() -> None:
    labels = ["a"] * 6 + ["b"] * 6
    groups = [f"a-{index // 2}" for index in range(6)] + [
        f"b-{index // 2}" for index in range(6)
    ]
    fit_a, val_a = group_stratified_inner_split(
        labels, groups, validation_fraction=0.25, seed=42
    )
    fit_b, val_b = group_stratified_inner_split(
        labels, groups, validation_fraction=0.25, seed=42
    )
    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(val_a, val_b)
    assert {groups[index] for index in fit_a}.isdisjoint(
        {groups[index] for index in val_a}
    )
    assert {labels[index] for index in fit_a} == {"a", "b"}


def test_group_stratified_split_rejects_group_crossing_labels() -> None:
    with pytest.raises(ProbeSelectionError, match="spans labels"):
        group_stratified_inner_split(
            ["a", "b", "a", "b"],
            ["shared", "shared", "a-2", "b-2"],
            validation_fraction=0.25,
            seed=1,
        )


def test_group_disjoint_inner_split_is_deterministic_and_nonoverlapping() -> None:
    groups = ["A", "A", "B", "B", "C", "C", "D", "D"]
    fit, validation = group_disjoint_inner_split(
        groups, validation_fraction=0.25, seed=42
    )
    fit_again, validation_again = group_disjoint_inner_split(
        groups, validation_fraction=0.25, seed=42
    )
    assert np.array_equal(fit, fit_again)
    assert np.array_equal(validation, validation_again)
    assert {groups[index] for index in fit}.isdisjoint(
        {groups[index] for index in validation}
    )
    assert set(fit) | set(validation) == set(range(len(groups)))


def test_selected_probe_uses_only_training_labels_and_writes_full_probabilities(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    centers = {
        "a": np.asarray([-2.0, 0.0], dtype=np.float32),
        "b": np.asarray([2.0, 0.0], dtype=np.float32),
    }
    labels: list[str] = []
    groups: list[str] = []
    rows: list[np.ndarray] = []
    for label, center in centers.items():
        for group_index in range(4):
            for _ in range(4):
                labels.append(label)
                groups.append(f"{label}-{group_index}")
                rows.append(center + rng.normal(0.0, 0.2, size=2))
    train = np.asarray(rows, dtype=np.float32)
    evaluation = {
        "calibration": np.asarray([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        "test": np.asarray([[-1.8, 0.1], [1.8, -0.1]], dtype=np.float32),
    }
    result = fit_selected_multiclass_probe(
        train,
        labels,
        groups,
        evaluation,
        tmp_path,
        config=MulticlassProbeSearchConfig(
            learning_rates=(1e-2,),
            max_epochs=20,
            patience=5,
            inner_validation_fraction=0.25,
            batch_size=16,
        ),
        seed=11,
        device="cpu",
    )
    probabilities = result["predictions"]["test"]["probabilities"]
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(probabilities, axis=1), np.asarray([0, 1]))
    assert result["selection"]["outer_calibration_or_test_labels_used"] is False
    assert (tmp_path / "linear_probe.pt").exists()
    assert (tmp_path / "probe_selection_manifest.json").exists()
