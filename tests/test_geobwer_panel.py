from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.formal_outputs import write_multiclass_bundle
from rsfm_fairness_audit.geobwer_panel import GeoBWERPanelError, run_geobwer_model_panel
from rsfm_fairness_audit.io import read_csv_rows, write_csv


WORK = Path("work/test_geobwer_panel")


@pytest.fixture(autouse=True)
def _clean_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    yield
    if WORK.exists():
        shutil.rmtree(WORK)


def _bundle(model: str, probabilities: np.ndarray):
    protocol = BWERProtocol(
        beta=0.5,
        beta_profile=(0.5,),
        inference_method="none",
        group_variable="country",
        cluster_column="location_id",
        task_adapter="multiclass",
        loss_name="risk",
    )
    rows = [
        {
            "sample_id": f"s{index}",
            "independent_unit_id": f"s{index}",
            "country": "A" if index < 3 else "B",
            "location_id": f"l{index}",
        }
        for index in range(6)
    ]
    return protocol, write_multiclass_bundle(
        WORK / model,
        sample_rows=rows,
        probabilities=probabilities,
        targets=np.asarray([0, 0, 0, 1, 1, 1]),
        class_names=("a", "b"),
        dataset="demo",
        model=model,
        split="test",
        protocol=protocol,
        model_lineage={"checkpoint": model},
        dataset_lineage={"dataset": "demo"},
    )


def test_common_unit_panel_reports_rank_and_paired_interval():
    protocol, first = _bundle(
        "m1",
        np.asarray([[0.9, 0.1]] * 3 + [[0.8, 0.2]] * 3),
    )
    _, second = _bundle(
        "m2",
        np.asarray([[0.6, 0.4]] * 3 + [[0.2, 0.8]] * 3),
    )
    _, third = _bundle(
        "m3",
        np.asarray([[0.7, 0.3]] * 3 + [[0.3, 0.7]] * 3),
    )
    artifacts = run_geobwer_model_panel(
        {"m1": first.audit_table, "m2": second.audit_table, "m3": third.audit_table},
        WORK / "panel",
        protocol=protocol,
        n_bootstrap=100,
    )
    assert artifacts.model_summary.exists()
    assert artifacts.paired_comparisons.exists()
    assert artifacts.common_support.exists()
    comparisons = read_csv_rows(artifacts.paired_comparisons)
    assert len(comparisons) == 3
    assert {row["multiplicity_method"] for row in comparisons} == {"bonferroni_familywise"}
    assert float(comparisons[0]["pairwise_adjusted_confidence_level"]) == pytest.approx(1.0 - 0.05 / 3.0)


def test_panel_rejects_metadata_drift_on_the_same_physical_unit():
    protocol, first = _bundle("m1", np.asarray([[0.9, 0.1]] * 6))
    _, second = _bundle("m2", np.asarray([[0.8, 0.2]] * 6))
    rows = read_csv_rows(second.audit_table)
    rows[0]["country"] = "DRIFT"
    write_csv(second.audit_table, rows)
    with pytest.raises(GeoBWERPanelError, match="metadata drift"):
        run_geobwer_model_panel(
            {"m1": first.audit_table, "m2": second.audit_table},
            WORK / "panel",
            protocol=protocol,
            n_bootstrap=100,
        )


def test_panel_honors_only_pre_registered_comparison_pairs():
    protocol, first = _bundle("m1", np.asarray([[0.9, 0.1]] * 6))
    _, second = _bundle("m2", np.asarray([[0.8, 0.2]] * 6))
    _, third = _bundle("m3", np.asarray([[0.7, 0.3]] * 6))
    artifacts = run_geobwer_model_panel(
        {"m1": first.audit_table, "m2": second.audit_table, "m3": third.audit_table},
        WORK / "panel",
        protocol=protocol,
        comparison_pairs=(("m1", "m3"),),
        n_bootstrap=100,
    )
    rows = read_csv_rows(artifacts.paired_comparisons)
    assert [(row["model_a"], row["model_b"]) for row in rows] == [("m1", "m3")]
    assert float(rows[0]["pairwise_adjusted_confidence_level"]) == pytest.approx(0.95)
