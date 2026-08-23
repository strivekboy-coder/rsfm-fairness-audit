from __future__ import annotations

from rsfm_fairness_audit.final_paper_cleanup import (
    build_experiment8_tables,
    build_experiment9_tables,
    build_reben_example_tables,
)


def test_experiment9_is_taskwise_and_direction_preserving() -> None:
    rows = []
    for task in ("fmow", "reben"):
        for seed in (42, 73, 101):
            rows.append({"task": task, "model": "dofav2", "seed": str(seed), "primary_risk": ".8", "M": ".7", "T": ".9", "D": ".2"})
            rows.append({"task": task, "model": "terramind", "seed": str(seed), "primary_risk": ".6", "M": ".5", "T": ".8", "D": ".3"})
    per_seed, summary = build_experiment9_tables(rows)
    assert len(per_seed) == 24
    assert len(summary) == 8
    assert all(row["direction_consistency"] == "3/3" for row in summary)
    assert {row["task"] for row in summary} == {"fmow", "reben"}
    assert not any(row.get("task") in {"all", "cross_task"} for row in summary)


def test_experiment8_excludes_stage_d_and_keeps_validation_gate_separate() -> None:
    stage_rows = []
    for seed in (42, 73, 101):
        for stage, split in (("A", "test_id"), ("A", "test_shifted"), ("B", "test"), ("C", "test")):
            stage_rows.append({"seed": str(seed), "stage": stage, "split_role": split, "macro_auroc": ".8", "mean_risk": ".2", "tail_risk_beta_0_10": ".3", "geobwer_beta_0_10": ".1"})
    recovery_rows = []
    for seed in (42, 73, 101):
        for stage in ("B", "C"):
            for metric in ("macro_auroc", "mean_risk", "tail_risk_beta_0_10", "geobwer_beta_0_10"):
                recovery_rows.append({"seed": str(seed), "split_role": "test", "stage": stage, "metric": metric, "recovery": ".5"})
    stages, recovery = build_experiment8_tables(stage_rows, recovery_rows)
    assert len(stages) == 16
    assert len(recovery) == 8
    assert "D" not in {row["stage"] for row in stages}
    assert all("validation_only" in row["selection_role"] for row in recovery)


def test_reben_selection_retains_candidate_universe_and_transparent_rules() -> None:
    labels = []
    burden = []
    specs = [("A", .8, .1), ("B", .7, -.2), ("C", -.3, .4), ("D", -.5, .2)]
    for idx, (label, terra_ppr, _) in enumerate(specs):
        labels.append({"class_label": label, "mean_delta_auroc": "-.2", "mean_delta_ap": "-.3", "mean_delta_predicted_positive_rate": str(terra_ppr)})
        for country, delta, positive in (("X", .9 - idx * .1, 3), ("Y", -.1 - idx * .1, 0)):
            burden.append({"country": country, "class_label": label, "seed_count": "3", "mean_delta_risk": str(delta), "delta_risk_sd": ".01", "positive_seed_count": str(positive), "minimum_cell_support": "2000"})
    croma = [{**row, "mean_delta_predicted_positive_rate": str(specs[i][2])} for i, row in enumerate(labels)]
    universe, selected = build_reben_example_tables(burden, labels, croma)
    assert len(universe) == 8
    assert 2 <= len(selected) <= 4
    assert all(row["eligible"] for row in selected)
    assert all(row["selection_rule"] for row in selected)
