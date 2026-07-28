from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from rsfm_fairness_audit.sen1floods11_formal import write_sen1_probability_export
from scripts.colab.run_sen1floods11_gpu_smoke_colab import (
    validate_probability_export,
)
from scripts.colab.run_terramind_sen1floods11_final_colab import (
    _terratorch_predict_command,
    _validate_diagnostic_export,
)


WORK = Path("work/test_sen1_gpu_smoke_runner")


@pytest.fixture(autouse=True)
def _clean_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    yield
    if WORK.exists():
        shutil.rmtree(WORK)


def test_probability_export_gate_checks_real_segmentation_contract():
    export = write_sen1_probability_export(
        WORK / "export",
        probabilities=[np.asarray([[0.2, 0.9], [0.4, 0.7]], dtype=np.float32)],
        targets=[np.asarray([[0, 1], [0, -1]], dtype=np.int16)],
        filenames=[{"S2L1C": "Bolivia_001_S2Hand.tif"}],
        batch_size=1,
    )
    report = validate_probability_export(export)
    assert report["row_count"] == 1
    assert len(report["sample_ids"]) == 1
    assert report["target_shapes"] == [[2, 2]]
    terramind_report = _validate_diagnostic_export(export, maximum_rows=1)
    assert terramind_report["row_count"] == 1
    assert terramind_report["samples"][0]["probability_shape"] == [2, 2]


def test_probability_export_gate_rejects_nonprobabilities():
    export = write_sen1_probability_export(
        WORK / "export",
        probabilities=[np.asarray([[0.2, 0.9], [0.4, 0.7]], dtype=np.float32)],
        targets=[np.asarray([[0, 1], [0, 1]], dtype=np.int16)],
        filenames=[{"S2L1C": "Bolivia_001_S2Hand.tif"}],
        batch_size=1,
    )
    index = next((export / "index_parts").glob("*.jsonl"))
    row = json.loads(index.read_text(encoding="utf-8").splitlines()[0])
    artifact = export / row["probability_path"]
    with np.load(artifact) as payload:
        target = np.asarray(payload["target"])
    np.savez_compressed(
        artifact,
        probabilities=np.full((2, 2, 2), 0.8, dtype=np.float32),
        target=target,
    )
    try:
        validate_probability_export(export)
    except RuntimeError as exc:
        assert "sum to one" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Malformed probability maps were accepted.")


def test_absolute_gpu_smoke_runner_help_works_outside_repo():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts/colab/run_sen1floods11_gpu_smoke_colab.py"
    alternate_cwd = (WORK / "alternate_cwd").resolve()
    alternate_cwd.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=alternate_cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "TerraMind" in result.stdout


def test_terramind_prediction_uses_repository_callback_safe_wrapper(monkeypatch):
    monkeypatch.setattr(
        "scripts.colab.run_terramind_sen1floods11_final_colab.validate_terratorch_runtime",
        lambda: None,
    )
    command = _terratorch_predict_command()
    assert command[:2] == [sys.executable, "-m"]
    assert command[2] == "rsfm_fairness_audit.terratorch_predict_cli"
