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


def test_diagnostic_export_allows_one_all_ignore_row_with_valid_export_support():
    original_targets = [
        np.full((2, 2), -1, dtype=np.int16),
        np.asarray([[-1, 0], [1, -1]], dtype=np.int16),
    ]
    export = write_sen1_probability_export(
        WORK / "mixed_ignore_export",
        probabilities=[
            np.asarray([[0.2, 0.9], [0.4, 0.7]], dtype=np.float32),
            np.asarray([[0.3, 0.8], [0.6, 0.1]], dtype=np.float32),
        ],
        targets=original_targets,
        filenames=["Ghana_5079.tif", "valid-chip.tif"],
        batch_size=2,
    )

    outer = validate_probability_export(export)
    inner = _validate_diagnostic_export(export, maximum_rows=2)

    assert outer["valid_pixel_counts"] == [0, 2]
    assert outer["all_ignore_row_count"] == 1
    assert outer["valid_row_count"] == 1
    assert outer["aggregate_valid_pixel_count"] == 2
    assert outer["observed_target_values"] == [-1, 0, 1]
    assert [sample["valid_pixel_count"] for sample in inner["samples"]] == [0, 2]
    assert inner["all_ignore_row_count"] == 1
    assert inner["valid_row_count"] == 1
    assert inner["aggregate_valid_pixel_count"] == 2
    assert inner["observed_target_values"] == [-1, 0, 1]

    first_index = next((export / "index_parts").glob("*.jsonl"))
    first_row = json.loads(first_index.read_text(encoding="utf-8").splitlines()[0])
    with np.load(export / first_row["probability_path"]) as artifact:
        np.testing.assert_array_equal(artifact["target"], original_targets[0])


def test_diagnostic_export_rejects_export_when_every_row_is_all_ignore():
    export = write_sen1_probability_export(
        WORK / "all_ignore_export",
        probabilities=[
            np.asarray([[0.2, 0.9], [0.4, 0.7]], dtype=np.float32),
            np.asarray([[0.3, 0.8], [0.6, 0.1]], dtype=np.float32),
        ],
        targets=[
            np.full((2, 2), -1, dtype=np.int16),
            np.full((2, 2), -1, dtype=np.int16),
        ],
        filenames=["ignore-a.tif", "ignore-b.tif"],
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="across any row"):
        validate_probability_export(export)
    with pytest.raises(RuntimeError, match="across any row"):
        _validate_diagnostic_export(export, maximum_rows=2)


def test_diagnostic_export_rejects_target_values_outside_formal_contract():
    export = write_sen1_probability_export(
        WORK / "invalid_target_export",
        probabilities=[
            np.asarray([[0.2, 0.9], [0.4, 0.7]], dtype=np.float32),
        ],
        targets=[
            np.asarray([[0, 1], [-1, 2]], dtype=np.int16),
        ],
        filenames=["invalid-label.tif"],
        batch_size=1,
    )

    with pytest.raises(RuntimeError, match=r"\{-1,0,1\}"):
        validate_probability_export(export)
    with pytest.raises(RuntimeError, match=r"\{-1,0,1\}"):
        _validate_diagnostic_export(export, maximum_rows=1)


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
