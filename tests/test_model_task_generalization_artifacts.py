from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from rsfm_fairness_audit.io import write_csv
from rsfm_fairness_audit.model_task_generalization import summarize_cell


WORK = Path("work/test_model_task_generalization_artifacts")


@pytest.fixture(autouse=True)
def _clean_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    yield
    if WORK.exists():
        shutil.rmtree(WORK)


def _write_geobwer_summary(path: Path, *, mean_risk: float) -> None:
    write_csv(path, [{
        "axis": "country",
        "mean_risk": mean_risk,
        "tail_risk": mean_risk + 0.1,
        "bwer": 0.1,
        "evidence_status": "formal_confirmed",
        "risk_spec_signature": "frozen-risk-spec",
    }])


def test_canonical_geobwer_wins_over_uncertainty_extension_summaries() -> None:
    seed_dir = WORK / "probe_seeds" / "seed_42"
    formal_dir = seed_dir / "formal_outputs"
    write_csv(formal_dir / "formal_audit_table.csv", [
        {"sample_id": "sample_0", "risk": 0.2},
        {"sample_id": "sample_1", "risk": 0.4},
    ])
    formal_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        formal_dir / "probabilities.npz",
        sample_id=np.asarray(["sample_0", "sample_1"]),
        probabilities=np.asarray([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32),
        targets=np.asarray([0, 1], dtype=np.int64),
        class_names=np.asarray(["a", "b"]),
    )
    derived_formal_dir = seed_dir / "uncertainty_extensions" / "selective_low" / "formal_outputs"
    write_csv(derived_formal_dir / "formal_audit_table.csv", [
        {"sample_id": "sample_0", "risk": 0.99},
        {"sample_id": "sample_1", "risk": 0.99},
    ])
    np.savez_compressed(
        derived_formal_dir / "probabilities.npz",
        sample_id=np.asarray(["sample_0", "sample_1"]),
        probabilities=np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        targets=np.asarray([0, 1], dtype=np.int64),
        class_names=np.asarray(["a", "b"]),
    )

    canonical = seed_dir / "geobwer" / "geobwer_summary.csv"
    _write_geobwer_summary(canonical, mean_risk=0.3)
    _write_geobwer_summary(
        seed_dir / "uncertainty_extensions" / "selective_low" / "geobwer" / "geobwer_summary.csv",
        mean_risk=0.8,
    )
    _write_geobwer_summary(
        seed_dir / "uncertainty_extensions" / "conformal_risk_control" / "geobwer" / "geobwer_summary.csv",
        mean_risk=0.9,
    )

    rows = summarize_cell(WORK, model="dofav2", task="reben")

    assert len(rows) == 1
    assert np.isclose(rows[0]["primary_risk"], 0.3)
    assert rows[0]["M"] == 0.3
    assert Path(rows[0]["geobwer_summary"]) == canonical.resolve()
    assert Path(rows[0]["audit_table"]) == (formal_dir / "formal_audit_table.csv").resolve()
