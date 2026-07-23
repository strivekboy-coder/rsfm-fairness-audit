from __future__ import annotations

from rsfm_fairness_audit.geobwer_validation import _property_checks, _simulated_panel

import numpy as np


def test_production_population_property_gate_passes() -> None:
    checks = _property_checks(9)
    assert checks
    assert all(check.passes for check in checks)


def test_validation_panel_is_bounded_and_clustered() -> None:
    losses, groups, clusters = _simulated_panel(
        np.random.default_rng(3),
        [0.2, 0.3, 0.4],
        clusters=5,
        units_per_group=[1, 2, 3],
    )
    assert losses.min() >= 0.0
    assert losses.max() <= 1.0
    assert len(losses) == 5 * (1 + 2 + 3)
    assert len(set(groups)) == 3
    assert len(set(clusters)) == 5
