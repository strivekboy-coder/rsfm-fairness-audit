from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from rsfm_fairness_audit.io import write_csv
from rsfm_fairness_audit.model_task_generalization import (
    ModelTaskGeneralizationError,
    _risk_spec_comparison_contract,
    summarize_cell,
)
from rsfm_fairness_audit.risk_spec import RiskSpec


WORK = Path("work/test_model_task_generalization_artifacts")


@pytest.fixture(autouse=True)
def _clean_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    yield
    if WORK.exists():
        shutil.rmtree(WORK)


def _protocol(*, beta: float = 0.1, deployment_weighting: str = "equal", loss_name: str = "risk") -> dict:
    return {
        "beta": beta,
        "deployment_weighting": deployment_weighting,
        "audit_measure": "balanced",
        "partition_rule": "one_axis_at_a_time",
        "missingness_rule": "strict",
        "standardization_target": "uniform",
        "standardization_weights": {},
        "support_rule": "country_location_preflight",
        "inference_target": "fixed_slice_universe",
        "estimand_scope": "fixed_slice_universe",
        "group_variable": "country",
        "balance_variable": "class_label",
        "independent_unit_column": "sample_id",
        "metric_version": "geobwer_fractional_1.1",
        "loss_name": loss_name,
        "task_adapter": "multiclass",
        "risk_spec": RiskSpec(name=loss_name, task_adapter="multiclass").to_dict(),
    }


def _write_cell(
    root: Path,
    *,
    modern_signature: bool,
    beta: float = 0.1,
    deployment_weighting: str = "equal",
    loss_name: str = "risk",
    omit_field: str | None = None,
) -> Path:
    seed_dir = root / "probe_seeds" / "seed_42"
    formal_dir = seed_dir / "formal_outputs"
    probabilities = np.asarray([[0.9, 0.1], [0.9, 0.1]], dtype=np.float32)
    targets = np.asarray([0, 1], dtype=np.int64)
    risks = (np.argmax(probabilities, axis=1) != targets).astype(float)
    write_csv(formal_dir / "formal_audit_table.csv", [
        {
            "sample_id": f"sample_{index}",
            "independent_unit_id": f"sample_{index}",
            "country": country,
            "class_label": label,
            "task": "multiclass_classification",
            "risk": risks[index],
        }
        for index, (country, label) in enumerate((("AA", "a"), ("BB", "b")))
    ])
    np.savez_compressed(
        formal_dir / "probabilities.npz",
        sample_id=np.asarray(["sample_0", "sample_1"]),
        probabilities=probabilities,
        targets=targets,
        class_names=np.asarray(["a", "b"]),
    )

    protocol = _protocol(beta=beta, deployment_weighting=deployment_weighting, loss_name=loss_name)
    if not modern_signature:
        protocol.pop("risk_spec")
    if omit_field:
        protocol.pop(omit_field, None)
    geobwer_dir = seed_dir / "geobwer"
    geobwer_dir.mkdir(parents=True, exist_ok=True)
    (geobwer_dir / "geobwer_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    manifest_protocol = dict(protocol)
    if omit_field:
        manifest_protocol.pop(omit_field, None)
    (formal_dir / "formal_output_manifest.json").write_text(
        json.dumps({"output_schema": "geobwer.multiclass.v1", "protocol": manifest_protocol}),
        encoding="utf-8",
    )
    summary = {
        "axis": "country",
        "mean_risk": 0.5,
        "tail_risk": 1.0,
        "bwer": 0.5,
        "evidence_status": "formal_confirmed",
        "risk_spec_signature": RiskSpec(name="risk", task_adapter="multiclass").signature
        if modern_signature and loss_name == "risk"
        else "",
        **{key: value for key, value in protocol.items() if key != "risk_spec"},
    }
    if omit_field:
        summary.pop(omit_field, None)
    write_csv(geobwer_dir / "geobwer_summary.csv", [summary])
    return seed_dir


def _pair_contract(modern_root: Path, legacy_root: Path) -> dict:
    rows = [
        *summarize_cell(modern_root, model="modern", task="fmow"),
        *summarize_cell(legacy_root, model="legacy", task="fmow"),
    ]
    return _risk_spec_comparison_contract("fmow", rows, ("modern", "legacy"))


def _write_reben_cell(root: Path, *, modern_signature: bool) -> None:
    seed_dir = root / "probe_seeds" / "seed_42"
    formal_dir = seed_dir / "formal_outputs"
    probabilities = np.asarray([[0.9, 0.1], [0.9, 0.1]], dtype=np.float32)
    targets = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
    thresholds = np.asarray([0.5, 0.5], dtype=np.float32)
    risks = np.mean((probabilities >= thresholds[None, :]) != targets, axis=1)
    write_csv(formal_dir / "formal_audit_table.csv", [
        {
            "sample_id": f"sample_{index}",
            "independent_unit_id": f"sample_{index}",
            "country": country,
            "task": "multilabel_classification",
            "risk": risks[index],
        }
        for index, country in enumerate(("AA", "BB"))
    ])
    np.savez_compressed(
        formal_dir / "probabilities.npz",
        sample_id=np.asarray(["sample_0", "sample_1"]),
        probabilities=probabilities,
        targets=targets,
        class_names=np.asarray(["a", "b"]),
        thresholds=thresholds,
    )
    risk_spec = RiskSpec(name="hamming_loss", task_adapter="multilabel")
    protocol = {
        **_protocol(),
        "support_rule": "country_patch_preflight",
        "balance_variable": "",
        "independent_unit_column": "independent_unit_id",
        "loss_name": "hamming_loss",
        "task_adapter": "multilabel",
        "risk_spec": risk_spec.to_dict(),
    }
    if not modern_signature:
        protocol.pop("risk_spec")
    geobwer_dir = seed_dir / "geobwer"
    geobwer_dir.mkdir(parents=True, exist_ok=True)
    (geobwer_dir / "geobwer_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (formal_dir / "formal_output_manifest.json").write_text(
        json.dumps({"output_schema": "geobwer.multilabel.v1", "protocol": protocol}),
        encoding="utf-8",
    )
    write_csv(geobwer_dir / "geobwer_summary.csv", [{
        "axis": "country",
        "mean_risk": 0.5,
        "tail_risk": 1.0,
        "bwer": 0.5,
        "risk_spec_signature": risk_spec.signature if modern_signature else "",
        **{key: value for key, value in protocol.items() if key != "risk_spec"},
    }])


def test_modern_signature_and_matching_legacy_semantics_pass() -> None:
    _write_cell(WORK / "modern", modern_signature=True)
    _write_cell(WORK / "legacy", modern_signature=False)

    contract = _pair_contract(WORK / "modern", WORK / "legacy")

    assert contract["same_risk_spec"] is True
    assert contract["risk_spec_contract_equality"] is True
    assert contract["risk_spec_verification_sources"] == ["explicit_signature", "legacy_reconstructed"]
    assert len(contract["explicit_risk_spec_signatures"]) == 1


def test_reben_modern_signature_and_matching_legacy_semantics_pass() -> None:
    _write_reben_cell(WORK / "modern_reben", modern_signature=True)
    _write_reben_cell(WORK / "legacy_reben", modern_signature=False)
    rows = [
        *summarize_cell(WORK / "modern_reben", model="modern", task="reben"),
        *summarize_cell(WORK / "legacy_reben", model="legacy", task="reben"),
    ]
    contract = _risk_spec_comparison_contract("reben", rows, ("modern", "legacy"))
    assert contract["same_risk_spec"] is True
    assert contract["risk_spec_verification_sources"] == ["explicit_signature", "legacy_reconstructed"]


def test_legacy_beta_mismatch_fails_semantic_contract() -> None:
    _write_cell(WORK / "modern", modern_signature=True)
    _write_cell(WORK / "legacy", modern_signature=False, beta=0.2)
    assert _pair_contract(WORK / "modern", WORK / "legacy")["same_risk_spec"] is False


def test_legacy_loss_semantics_mismatch_fails() -> None:
    _write_cell(WORK / "legacy", modern_signature=False, loss_name="log_loss")
    with pytest.raises(ModelTaskGeneralizationError, match="loss/task adapter disagrees"):
        summarize_cell(WORK / "legacy", model="legacy", task="fmow")


def test_legacy_deployment_weighting_mismatch_fails_semantic_contract() -> None:
    _write_cell(WORK / "modern", modern_signature=True)
    _write_cell(WORK / "legacy", modern_signature=False, deployment_weighting="empirical")
    assert _pair_contract(WORK / "modern", WORK / "legacy")["same_risk_spec"] is False


def test_unverifiable_legacy_contract_hard_fails() -> None:
    _write_cell(WORK / "legacy", modern_signature=False, omit_field="deployment_weighting")
    with pytest.raises(ModelTaskGeneralizationError, match="deployment_weighting"):
        summarize_cell(WORK / "legacy", model="legacy", task="fmow")


def test_canonical_geobwer_wins_over_uncertainty_extension_summaries() -> None:
    seed_dir = _write_cell(WORK / "canonical", modern_signature=True)
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
    write_csv(
        seed_dir / "uncertainty_extensions" / "selective_low" / "geobwer" / "geobwer_summary.csv",
        [{"axis": "country", "mean_risk": 0.8, "tail_risk": 0.9, "bwer": 0.1}],
    )
    write_csv(
        seed_dir / "uncertainty_extensions" / "conformal_risk_control" / "geobwer" / "geobwer_summary.csv",
        [{"axis": "country", "mean_risk": 0.9, "tail_risk": 1.0, "bwer": 0.1}],
    )

    rows = summarize_cell(WORK / "canonical", model="dofav2", task="fmow")

    assert len(rows) == 1
    assert np.isclose(rows[0]["primary_risk"], 0.5)
    assert rows[0]["M"] == 0.5
    assert Path(rows[0]["geobwer_summary"]) == (seed_dir / "geobwer" / "geobwer_summary.csv").resolve()
    assert Path(rows[0]["audit_table"]) == (seed_dir / "formal_outputs" / "formal_audit_table.csv").resolve()
