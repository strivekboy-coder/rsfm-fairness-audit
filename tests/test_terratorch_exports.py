from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.terratorch_exports import TerraTorchExportError, write_probability_batch


WORK = Path("work/test_terratorch_exports")


@pytest.fixture(autouse=True)
def _clean_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    yield
    if WORK.exists():
        shutil.rmtree(WORK)


def test_writes_labeled_probability_map_per_independent_unit():
    probabilities = np.asarray(
        [
            [[[0.8, 0.2], [0.4, 0.1]], [[0.2, 0.8], [0.6, 0.9]]],
            [[[0.1, 0.3], [0.7, 0.6]], [[0.9, 0.7], [0.3, 0.4]]],
        ],
        dtype=np.float32,
    )
    target = np.asarray([[[0, 1], [1, 1]], [[1, 1], [0, 0]]], dtype=np.int64)
    rows = write_probability_batch(
        WORK,
        outputs={"probabilities": probabilities, "filename": ["event-a.tif", "event-b.tif"]},
        batch={"mask": target, "event_id": ["a", "b"]},
        batch_idx=2,
    )
    assert len(rows) == 2
    assert rows[0]["event_id"] == "a"
    assert rows[0]["probability_path"].startswith("samples/")
    assert not Path(rows[0]["probability_path"]).is_absolute()
    saved = np.load(WORK / rows[0]["probability_path"])
    np.testing.assert_allclose(saved["probabilities"], probabilities[0])
    np.testing.assert_array_equal(saved["target"], target[0])


def test_rejects_logits_and_unlabeled_prediction_batches():
    with pytest.raises(TerraTorchExportError, match="outside"):
        write_probability_batch(
            WORK,
            outputs={"probabilities": np.asarray([[2.0, -1.0]], dtype=np.float32)},
            batch={"label": np.asarray([0])},
            batch_idx=0,
        )
    with pytest.raises(TerraTorchExportError, match="requires labels"):
        write_probability_batch(
            WORK,
            outputs={"probabilities": np.asarray([[0.2, 0.8]], dtype=np.float32)},
            batch={},
            batch_idx=0,
        )
