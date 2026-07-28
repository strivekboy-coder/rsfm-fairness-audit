from __future__ import annotations

import ast
import sys
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rsfm_fairness_audit.terratorch_exports import (
    TerraTorchExportError,
    segmentation_probabilities_from_logits,
    write_probability_batch,
)


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


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)
        self.ndim = self.value.ndim
        self.shape = self.value.shape


def _fake_softmax(value: _FakeTensor, dim: int):
    shifted = value.value - np.max(value.value, axis=dim, keepdims=True)
    exponent = np.exp(shifted)
    return _FakeTensor(exponent / exponent.sum(axis=dim, keepdims=True))


def test_segmentation_logits_export_retains_full_two_class_softmax(monkeypatch):
    fake_torch = SimpleNamespace(
        is_tensor=lambda value: isinstance(value, _FakeTensor),
        softmax=_fake_softmax,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    logits = _FakeTensor(
        [
            [
                [[2.0, 0.0], [1.0, -1.0]],
                [[0.0, 2.0], [-1.0, 1.0]],
            ]
        ]
    )
    probabilities = segmentation_probabilities_from_logits(logits)
    assert probabilities.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(
        probabilities.value.sum(axis=1),
        np.ones((1, 2, 2)),
        atol=1e-12,
    )
    assert np.all((probabilities.value >= 0.0) & (probabilities.value <= 1.0))


def test_segmentation_tuple_selector_output_is_rejected(monkeypatch):
    fake_torch = SimpleNamespace(
        is_tensor=lambda value: isinstance(value, _FakeTensor),
        softmax=_fake_softmax,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(TerraTorchExportError, match="raw logits tensor"):
        segmentation_probabilities_from_logits(
            (_FakeTensor(np.zeros((1, 2, 2), dtype=np.float32)), "pred")
        )


def test_segmentation_predict_ast_never_calls_select_classes_and_keeps_tiled_raw_logits():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rsfm_fairness_audit"
        / "terratorch_exports.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    task = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "GeoBWERSemanticSegmentationTask"
    )
    predict = next(
        node
        for node in task.body
        if isinstance(node, ast.FunctionDef) and node.name == "predict_step"
    )
    attributes = {
        node.attr for node in ast.walk(predict) if isinstance(node, ast.Attribute)
    }
    names = {node.id for node in ast.walk(predict) if isinstance(node, ast.Name)}
    assert "select_classes" not in attributes
    assert "segmentation_probabilities_from_logits" in names
    assert "tiled_inference" in names
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "logits" for target in node.targets)
        for node in ast.walk(predict)
    )


def test_classification_and_multilabel_probability_paths_remain_unchanged():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rsfm_fairness_audit"
        / "terratorch_exports.py"
    ).read_text(encoding="utf-8")
    classification = source.split("class GeoBWERClassificationTask", 1)[1].split(
        "class GeoBWERMultiLabelClassificationTask", 1
    )[0]
    multilabel = source.split("class GeoBWERMultiLabelClassificationTask", 1)[1].split(
        "class GeoBWERSemanticSegmentationTask", 1
    )[0]
    assert "torch.softmax(logits, dim=1)" in classification
    assert "torch.sigmoid(logits)" in multilabel
