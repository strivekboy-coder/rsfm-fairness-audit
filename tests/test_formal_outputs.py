from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.formal_outputs import write_multiclass_bundle, write_multilabel_bundle, write_segmentation_bundle
from rsfm_fairness_audit.io import read_csv_rows


TEST_ROOT = Path("work/geobwer_formal_output_tests")


def _protocol(task: str) -> BWERProtocol:
    return BWERProtocol(
        task_adapter=task,
        inference_method="cluster_maxt",
        cluster_column="cluster_id",
        min_clusters_for_default=2,
    )


def _rows() -> list[dict[str, str]]:
    return [
        {"sample_id": "s1", "location_id": "l1", "cluster_id": "c1", "country": "A"},
        {"sample_id": "s2", "location_id": "l2", "cluster_id": "c2", "country": "B"},
    ]


def test_multiclass_bundle_preserves_full_probability_matrix() -> None:
    bundle = write_multiclass_bundle(
        TEST_ROOT / "multiclass",
        sample_rows=_rows(),
        probabilities=np.asarray([[0.8, 0.2], [0.1, 0.9]]),
        targets=["a", "b"],
        class_names=["a", "b"],
        dataset="demo",
        model="model",
        split="test",
        protocol=_protocol("multiclass"),
        model_lineage={"checkpoint": "x"},
        dataset_lineage={"manifest": "y"},
        independent_unit_column="location_id",
    )
    with np.load(bundle.probability_artifact) as data:
        assert data["probabilities"].shape == (2, 2)
        assert data["class_names"].tolist() == ["a", "b"]
    rows = read_csv_rows(bundle.audit_table)
    assert rows[0]["probabilities_path"] == "probabilities.npz"
    assert rows[0]["independent_unit_id"] == "l1"
    assert json.loads(bundle.manifest.read_text(encoding="utf-8"))["output_schema"] == "geobwer.multiclass.v1"


def test_multiclass_bundle_rejects_nonprobabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        write_multiclass_bundle(
            TEST_ROOT / "bad",
            sample_rows=_rows(),
            probabilities=[[0.8, 0.8], [0.2, 0.2]],
            targets=[0, 1],
            class_names=["a", "b"],
            dataset="demo",
            model="model",
            split="test",
            protocol=_protocol("multiclass"),
            model_lineage={"checkpoint": "x"},
            dataset_lineage={"manifest": "y"},
            independent_unit_column="location_id",
        )


def test_multilabel_bundle_writes_hamming_and_fnr() -> None:
    bundle = write_multilabel_bundle(
        TEST_ROOT / "multilabel",
        sample_rows=_rows(),
        probabilities=[[0.8, 0.2], [0.4, 0.9]],
        targets=[[1, 0], [1, 1]],
        class_names=["a", "b"],
        dataset="demo",
        model="model",
        split="test",
        protocol=_protocol("multilabel"),
        model_lineage={"checkpoint": "x"},
        dataset_lineage={"manifest": "y"},
        independent_unit_column="location_id",
    )
    rows = read_csv_rows(bundle.audit_table)
    assert float(rows[0]["risk"]) == 0.0
    assert float(rows[1]["false_negative_rate"]) == 0.5


def test_segmentation_bundle_writes_probability_maps_and_counts() -> None:
    bundle = write_segmentation_bundle(
        TEST_ROOT / "segmentation",
        sample_rows=_rows(),
        positive_probability_maps=[np.asarray([[0.9, 0.1], [0.8, 0.2]]), np.asarray([[0.1, 0.8], [0.2, 0.7]])],
        target_masks=[np.asarray([[1, 0], [1, 0]]), np.asarray([[0, 1], [0, 1]])],
        dataset="demo",
        model="model",
        split="test",
        protocol=_protocol("segmentation"),
        model_lineage={"checkpoint": "x"},
        dataset_lineage={"manifest": "y"},
        independent_unit_column="location_id",
    )
    rows = read_csv_rows(bundle.audit_table)
    assert float(rows[0]["risk"]) == 0.0
    assert Path(bundle.output_dir / rows[0]["probability_map_path"]).exists()
    assert len(json.loads(bundle.probability_artifact.read_text(encoding="utf-8"))) == 2
