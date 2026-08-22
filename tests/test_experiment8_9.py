from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.model_task_generalization import analyze_model_task_matrix
from rsfm_fairness_audit.reben_adaptation_ablation import (
    _decision_partition,
    _validate_frozen_seed,
    directional_recovery,
)


def test_directional_recovery_handles_higher_and_lower_is_better() -> None:
    assert np.isclose(directional_recovery(0.9, 0.5, 0.8, higher_is_better=True)["recovery"], 0.75)
    assert np.isclose(directional_recovery(0.1, 0.5, 0.2, higher_is_better=False)["recovery"], 0.75)
    assert directional_recovery(0.5, 0.5, 0.6, higher_is_better=True)["identifiable"] is False


def test_validation_partition_is_unit_disjoint_and_deterministic() -> None:
    rows = [{"independent_unit_id": f"tile_{index // 2}"} for index in range(20)]
    left_a, right_a = _decision_partition(rows, 42)
    left_b, right_b = _decision_partition(rows, 42)
    assert np.array_equal(left_a, left_b) and np.array_equal(right_a, right_b)
    assert set(left_a).isdisjoint(set(right_a))
    assert set(left_a) | set(right_a) == set(range(len(rows)))
    left_units = {rows[index]["independent_unit_id"] for index in left_a}
    right_units = {rows[index]["independent_unit_id"] for index in right_a}
    assert left_units.isdisjoint(right_units)


def test_frozen_seed_accepts_legacy_contract_without_model_family() -> None:
    source = Path("work/test_experiment8_legacy_contract")
    checkpoint = source / "s2_trained_probe" / "linear_probe.pt"
    thresholds = source / "s2_validation_locked_thresholds.csv"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"legacy-frozen-checkpoint")
    thresholds.write_text("class_index,threshold\n0,0.5\n", encoding="utf-8")
    (source / "paired_shift_contract.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "same_head": True,
                "test_used_for_selection": False,
                "checkpoint_sha256": file_sha256(checkpoint),
                "thresholds_sha256": file_sha256(thresholds),
            }
        ),
        encoding="utf-8",
    )

    _validate_frozen_seed(source, checkpoint, thresholds)


def _cell(root: Path, model: str, task: str, model_offset: float) -> None:
    from rsfm_fairness_audit.risk_spec import RiskSpec

    for seed, seed_offset in ((42, 0.00), (73, 0.01), (101, -0.01)):
        run = root / "probe_seeds" / f"seed_{seed}"
        if task == "fmow":
            targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
            errors = np.asarray([0, 1, 1, 0] if model == "dofav2" else [0, 0, 0, 1], dtype=np.int64)
            predictions = np.where(errors == 1, 1 - targets, targets)
            probabilities = np.full((4, 2), 0.1, dtype=np.float32)
            probabilities[np.arange(4), predictions] = 0.9
            risks = errors.astype(float)
            formal_task, loss_name, task_adapter = "multiclass_classification", "risk", "multiclass"
            balance_variable = "class_label"
        else:
            targets = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int8)
            predictions = np.asarray(
                [[1, 0], [1, 0], [1, 0], [0, 0]]
                if model == "dofav2"
                else [[1, 0], [0, 1], [1, 1], [1, 0]],
                dtype=np.int8,
            )
            probabilities = np.where(predictions == 1, 0.9, 0.1).astype(np.float32)
            risks = np.mean(predictions != targets, axis=1)
            formal_task, loss_name, task_adapter = "multilabel_classification", "hamming_loss", "multilabel"
            balance_variable = ""
        protocol = {
            "beta": 0.1,
            "deployment_weighting": "equal",
            "audit_measure": "balanced",
            "partition_rule": "one_axis_at_a_time",
            "missingness_rule": "strict",
            "standardization_target": "uniform",
            "standardization_weights": {},
            "support_rule": f"{task}_country_preflight",
            "inference_target": "fixed_slice_universe",
            "estimand_scope": "fixed_slice_universe",
            "group_variable": "country",
            "balance_variable": balance_variable,
            "independent_unit_column": "sample_id",
            "metric_version": "geobwer_fractional_1.1",
            "loss_name": loss_name,
            "task_adapter": task_adapter,
            "risk_spec": RiskSpec(name=loss_name, task_adapter=task_adapter).to_dict(),
        }
        write_csv(run / "formal_outputs" / "formal_audit_table.csv", [
            {
                "sample_id": f"sample_{index}",
                "country": "AA" if index < 2 else "BB",
                "class_label": "a" if index % 2 == 0 else "b",
                "task": formal_task,
                "risk": risks[index],
            }
            for index in range(4)
        ])
        formal = run / "formal_outputs" / "probabilities.npz"
        formal.parent.mkdir(parents=True, exist_ok=True)
        probability_payload = dict(
            sample_id=np.asarray([f"sample_{index}" for index in range(4)]),
            probabilities=probabilities,
            targets=targets,
            class_names=np.asarray(["a", "b"]),
        )
        if task == "reben":
            probability_payload["thresholds"] = np.asarray([0.5, 0.5], dtype=np.float32)
        np.savez_compressed(formal, **probability_payload)
        (run / "formal_outputs" / "formal_output_manifest.json").write_text(
            json.dumps({"protocol": protocol}), encoding="utf-8"
        )
        geobwer = run / "geobwer_raw"
        geobwer.mkdir(parents=True, exist_ok=True)
        (geobwer / "geobwer_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        write_csv(run / "geobwer_raw" / "geobwer_summary.csv", [{
            "axis": "country", "mean_risk": model_offset + seed_offset,
            "tail_risk": model_offset + seed_offset + 0.1,
            "bwer": 0.1, "evidence_status": "descriptive",
            "risk_spec_signature": RiskSpec(name=loss_name, task_adapter=task_adapter).signature,
            **{key: value for key, value in protocol.items() if key != "risk_spec"},
        }])


def test_model_task_analysis_never_averages_raw_cross_task_geobwer() -> None:
    tmp_path = Path("work/test_experiment8_9_matrix")
    roots = {}
    for model, offset in (("dofav2", 0.3), ("terramind", 0.2)):
        for task in ("fmow", "reben"):
            root = tmp_path / f"{model}_{task}"
            _cell(root, model, task, offset + (0.1 if task == "reben" else 0.0))
            roots[(model, task)] = root
    artifacts = analyze_model_task_matrix(roots, tmp_path / "analysis")
    manifest = artifacts["manifest"].read_text(encoding="utf-8")
    assert '"raw_geobwer_averaged_across_tasks": false' in manifest
    effects = read_csv_rows(artifacts["effects"])
    assert len(effects) == 8
    assert {row["task"] for row in effects} == {"fmow", "reben"}
