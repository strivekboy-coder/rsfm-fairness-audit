from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "build_granular_scientific_discovery.py"
SPEC = importlib.util.spec_from_file_location("granular", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_fractional_tail_distinguishes_broad_and_isolated_geometry() -> None:
    isolated = np.array([1.0] + [0.0] * 19)
    broad = np.array([0.5] * 4 + [0.0] * 16)
    mi, ti, di, ei = MOD.fractional_tail(isolated, 0.1)
    mb, tb, db, eb = MOD.fractional_tail(broad, 0.1)
    # Equal top-decile risk can hide different population means and therefore
    # different excess tail burdens; preserve the joint M/T/D interpretation.
    assert ti == pytest.approx(tb)
    assert mi < mb
    assert ei == eb == 2.0
    assert di > db


def test_seed_parser_is_strict() -> None:
    assert MOD.seed_from_run("model_seed_42") == 42
    assert MOD.seed_from_run("model_seed_101") == 101
    assert np.isnan(MOD.seed_from_run("single_frozen_run"))


def test_tail_candidate_mask_retains_all_boundary_ties() -> None:
    risks = pd.Series([1.0] * 9 + [0.4] * 10)
    mask = MOD.tail_candidate_mask(risks, beta=0.1)
    assert mask.sum() == 9
    assert mask.iloc[:9].all()
    assert not mask.iloc[9:].any()


def test_selective_frontier_uses_complete_panel_and_derived_quantities() -> None:
    rows = []
    for run_id in ("dofa_scaled10000", "resnet50_13band"):
        for coverage in (0.7, 0.8, 0.9):
            for axis in ("country", "class", "country_x_class"):
                rows.append({
                    "run_id": run_id,
                    "coverage_target": coverage,
                    "slice_axis": axis,
                    "tail_minus_nontail_retained_coverage": -0.1 * (1 - coverage),
                    "mean_baseline_risk": 0.8,
                    "mean_retained_risk": 0.6,
                    "mean_rejected_risk": 0.95,
                })
    result = MOD.selective_risk_service_frontier(pd.DataFrame(rows))
    assert len(result) == 18
    assert result.tail_service_deficit.all()
    assert result.rejected_is_harder.all()
    assert np.allclose(result.remaining_risk_reduction, 0.2)
    assert set(result.model) == {"DOFAv2", "ResNet50"}


def test_shift_adaptation_turnover_is_tie_aware_and_fixed_universe() -> None:
    rows = []
    for seed in (42, 73, 101):
        rows.extend([
            {"seed": seed, "slice_axis": "label", "slice_value": "old burden", "risk_A_id": .1, "risk_A_shifted": .9, "risk_B": .8, "risk_C": .2,
             "support_A_id": 100, "support_A_shifted": 100, "support_B": 100, "support_C": 100, "positive_support_A_id": 40, "positive_support_A_shifted": 40, "positive_support_B": 40, "positive_support_C": 40,
             "tail_A_id": False, "tail_A_shifted": True, "tail_B": True, "tail_C": False, "tail_candidate_count_A_shifted": 2, "tail_candidate_count_C": 2},
            {"seed": seed, "slice_axis": "label", "slice_value": "new burden", "risk_A_id": .2, "risk_A_shifted": .3, "risk_B": .3, "risk_C": .8,
             "support_A_id": 100, "support_A_shifted": 100, "support_B": 100, "support_C": 100, "positive_support_A_id": 40, "positive_support_A_shifted": 40, "positive_support_B": 40, "positive_support_C": 40,
             "tail_A_id": False, "tail_A_shifted": False, "tail_B": False, "tail_C": True, "tail_candidate_count_A_shifted": 2, "tail_candidate_count_C": 2},
        ])
    seed_level, summary = MOD.shift_adaptation_tail_turnover(pd.DataFrame(rows))
    old = summary[summary.slice_id.eq("old burden")].iloc[0]
    new = summary[summary.slice_id.eq("new burden")].iloc[0]
    assert old.stable_shift_tail_exit_after_C
    assert new.stable_new_C_tail
    assert set(seed_level.tail_turnover_shift_to_C) == {"tail_exit", "new_C_tail"}
