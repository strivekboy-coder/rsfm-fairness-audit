from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.evidence_registry import load_canonical_evidence_registry
from rsfm_fairness_audit.formal_outputs import write_multiclass_bundle
from rsfm_fairness_audit.frozen_evidence_reaudit import (
    certification_protocol,
    reaudit_frozen_table,
    sha256_file,
)


def test_frozen_table_reaudit_preserves_source_and_writes_new_contract() -> None:
    tmp_path = Path("work/test_frozen_evidence_reaudit_v1")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    source_protocol = BWERProtocol(
        task_adapter="multiclass",
        loss_name="zero_one_loss",
        min_clusters_for_default=2,
    )
    rows = []
    probabilities = []
    targets = []
    for group_index, country in enumerate(("A", "B")):
        for index in range(80):
            rows.append(
                {
                    "sample_id": f"{country}_{index}",
                    "independent_unit_id": f"{country}_{index}",
                    "cluster_id": f"{country}_{index}",
                    "country": country,
                }
            )
            error = group_index == 1 and index < 40
            probabilities.append([0.1, 0.9] if error else [0.9, 0.1])
            targets.append("a")
    bundle = write_multiclass_bundle(
        tmp_path / "source",
        sample_rows=rows,
        probabilities=np.asarray(probabilities),
        targets=targets,
        class_names=("a", "b"),
        dataset="fixture",
        model="fixture_model",
        split="test",
        protocol=source_protocol,
        model_lineage={"checkpoint": "fixture"},
        dataset_lineage={"manifest": "fixture"},
        independent_unit_column="independent_unit_id",
    )
    before = sha256_file(bundle.audit_table)
    registry = load_canonical_evidence_registry(
        "configs/analysis/canonical_evidence_registry_v1.yaml"
    )
    protocol = certification_protocol(
        source_protocol,
        task="fmow_sentinel",
        calibration_signature="fixture-calibration",
        min_clusters_for_inference=75,
    )
    artifacts = reaudit_frozen_table(
        source_table=bundle.audit_table,
        output_dir=tmp_path / "reaudit",
        protocol=protocol,
        group_columns=("country",),
        cluster_column="cluster_id",
        registry=registry,
        registry_asset_id="fmow_predictions_v3",
        n_bootstrap=100,
    )
    assert sha256_file(bundle.audit_table) == before
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["source_immutable"] is True
    assert manifest["certification_version"] == "geobwer_certification_1.2"
    assert manifest["evidence_status_by_axis"]["country"] == "formal_confirmed"
    assert artifacts["completion"].exists()
