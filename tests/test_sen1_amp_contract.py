from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import rsfm_fairness_audit.sen1_supervised_campaign as campaign
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.sen1_amp_carry_forward import (
    SOURCE_COMMIT,
    build_carry_forward_manifest,
    load_carry_forward_manifest,
    reuse_carry_forward_seed,
)
from rsfm_fairness_audit.sen1_supervised_campaign import (
    AMP_MAX_CONSECUTIVE_OVERFLOWS,
    AMP_MAX_TOTAL_OVERFLOWS,
    AMP_OVERFLOW_POLICY_SCHEMA,
    Sen1SupervisedCampaignError,
    _finish_scaled_optimizer_step,
    _amp_manifest_fields,
    _new_amp_overflow_tracker,
    _record_optimizer_outcome,
)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return bool(self.value)


class _Parameter:
    def __init__(self, value: float, grad: float):
        self.value = np.asarray([value], dtype=np.float64)
        self.grad = np.asarray([grad], dtype=np.float64)


class _Model:
    def __init__(self, *, value: float = 1.0, grad: float = 1.0):
        self.parameter = _Parameter(value, grad)

    def named_parameters(self):
        return [("head.3.weight", self.parameter)]


class _Torch:
    @staticmethod
    def isfinite(value):
        if isinstance(value, _Parameter):
            value = value.value
        return _Scalar(np.isfinite(np.asarray(value)).all())


class _Optimizer:
    def __init__(self, model: _Model, lr: float = 0.1):
        self.model = model
        self.lr = lr
        self.step_count = 0

    def step(self):
        self.model.parameter.value -= self.lr * self.model.parameter.grad
        self.step_count += 1

    def zero_grad(self, set_to_none=True):
        self.model.parameter.grad = None if set_to_none else np.zeros(1)


class _Scaler:
    def __init__(self, *, scale: float = 8.0, enabled: bool = True):
        self.scale_value = scale
        self.enabled = enabled
        self.found_inf = False

    def is_enabled(self):
        return self.enabled

    def get_scale(self):
        return self.scale_value

    def unscale_(self, optimizer):
        grad = optimizer.model.parameter.grad
        self.found_inf = grad is not None and not np.isfinite(grad).all()

    def step(self, optimizer):
        if not self.found_inf:
            optimizer.step()

    def update(self):
        if self.found_inf:
            self.scale_value *= 0.5


def _finish(model, optimizer, scaler, *, amp=True, batch_index=0):
    return _finish_scaled_optimizer_step(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        amp=amp,
        scale_before=float(scaler.get_scale()),
        mode="S1",
        seed=101,
        epoch=5,
        batch_index=batch_index,
        training_stage="unit_test",
        sample_ids=["Ghana_134751"],
    )


def test_amp_overflow_recovers_skips_update_and_next_batch_updates(monkeypatch):
    monkeypatch.setattr(campaign, "_require_torch", lambda: _Torch)
    model = _Model(grad=float("inf"))
    optimizer = _Optimizer(model)
    scaler = _Scaler()
    before = model.parameter.value.copy()

    result = _finish(model, optimizer, scaler)

    np.testing.assert_array_equal(model.parameter.value, before)
    assert optimizer.step_count == 0
    assert scaler.get_scale() == 4.0
    assert result["amp_overflow"] is True
    assert result["optimizer_step_skipped"] is True
    assert result["amp_overflow_record"]["overflow_parameter_names"] == [
        "head.3.weight"
    ]

    model.parameter.grad = np.asarray([2.0])
    finite = _finish(model, optimizer, scaler, batch_index=1)
    assert finite["amp_overflow"] is False
    assert optimizer.step_count == 1
    assert model.parameter.value[0] == pytest.approx(0.8)


def test_non_amp_nonfinite_gradient_still_hard_fails(monkeypatch):
    monkeypatch.setattr(campaign, "_require_torch", lambda: _Torch)
    model = _Model(grad=float("nan"))
    optimizer = _Optimizer(model)
    scaler = _Scaler(enabled=False)

    with pytest.raises(
        Sen1SupervisedCampaignError,
        match="AMP disabled",
    ):
        _finish(model, optimizer, scaler, amp=False)
    assert optimizer.step_count == 0


def test_consecutive_overflow_limit_and_journal(tmp_path):
    tracker = _new_amp_overflow_tracker()
    journal = tmp_path / "amp_overflow_journal.json"
    for index in range(AMP_MAX_CONSECUTIVE_OVERFLOWS):
        _record_optimizer_outcome(
            tracker,
            {
                "skipped_all_ignore": False,
                "amp_overflow": True,
                "amp_overflow_record": {
                    "schema": AMP_OVERFLOW_POLICY_SCHEMA,
                    "mode": "S1",
                    "seed": 101,
                    "training_stage": "unit_test",
                    "epoch": 5,
                    "batch_index": index,
                    "sample_ids": ["Ghana_134751"],
                    "scale_before": 8.0 / (2**index),
                    "scale_after": 4.0 / (2**index),
                    "overflow_parameter_names": ["head.3.weight"],
                    "optimizer_step_skipped": True,
                },
            },
            journal_path=journal,
        )
    with pytest.raises(
        Sen1SupervisedCampaignError,
        match="consecutive=4",
    ):
        _record_optimizer_outcome(
            tracker,
            {
                "skipped_all_ignore": False,
                "amp_overflow": True,
                "amp_overflow_record": {
                    "schema": AMP_OVERFLOW_POLICY_SCHEMA,
                    "mode": "S1",
                    "seed": 101,
                    "training_stage": "unit_test",
                    "epoch": 5,
                    "batch_index": 3,
                    "sample_ids": ["Ghana_134751"],
                    "scale_before": 1.0,
                    "scale_after": 0.5,
                    "overflow_parameter_names": ["head.3.weight"],
                    "optimizer_step_skipped": True,
                },
            },
            journal_path=journal,
        )
    payload = json.loads(journal.read_text(encoding="utf-8"))
    assert payload["amp_overflow_count"] == 4
    assert payload["skipped_optimizer_step_count"] == 4
    assert len(payload["amp_overflow_records"]) == 4
    fields = _amp_manifest_fields(tracker)
    assert fields["amp_overflow_count"] == 4
    assert fields["skipped_optimizer_step_count"] == 4
    assert len(fields["amp_overflow_records"]) == 4


def test_total_overflow_limit_cannot_be_avoided_by_finite_batches(tmp_path):
    tracker = _new_amp_overflow_tracker()
    journal = tmp_path / "amp_overflow_journal.json"
    for index in range(AMP_MAX_TOTAL_OVERFLOWS):
        _record_optimizer_outcome(
            tracker,
            {
                "skipped_all_ignore": False,
                "amp_overflow": True,
                "amp_overflow_record": {
                    "schema": AMP_OVERFLOW_POLICY_SCHEMA,
                    "mode": "S1",
                    "seed": 101,
                    "training_stage": "unit_test",
                    "epoch": index + 1,
                    "batch_index": 0,
                    "sample_ids": ["sample"],
                    "scale_before": 8.0,
                    "scale_after": 4.0,
                    "overflow_parameter_names": ["head.3.weight"],
                    "optimizer_step_skipped": True,
                },
            },
            journal_path=journal,
        )
        _record_optimizer_outcome(
            tracker,
            {
                "skipped_all_ignore": False,
                "amp_overflow": False,
            },
            journal_path=journal,
        )
    with pytest.raises(Sen1SupervisedCampaignError, match="total=21"):
        _record_optimizer_outcome(
            tracker,
            {
                "skipped_all_ignore": False,
                "amp_overflow": True,
                "amp_overflow_record": {
                    "schema": AMP_OVERFLOW_POLICY_SCHEMA,
                    "mode": "S1",
                    "seed": 101,
                    "training_stage": "unit_test",
                    "epoch": 99,
                    "batch_index": 0,
                    "sample_ids": ["sample"],
                    "scale_before": 8.0,
                    "scale_after": 4.0,
                    "overflow_parameter_names": ["head.3.weight"],
                    "optimizer_step_skipped": True,
                },
            },
            journal_path=journal,
        )


def test_amp_finite_path_matches_v0427_reference_update(monkeypatch):
    monkeypatch.setattr(campaign, "_require_torch", lambda: _Torch)
    legacy_model = _Model(grad=2.0)
    current_model = _Model(grad=2.0)
    legacy_optimizer = _Optimizer(legacy_model)
    current_optimizer = _Optimizer(current_model)
    legacy_scaler = _Scaler()
    current_scaler = _Scaler()

    legacy_scaler.unscale_(legacy_optimizer)
    legacy_scaler.step(legacy_optimizer)
    legacy_scaler.update()
    current = _finish(current_model, current_optimizer, current_scaler)

    assert current["amp_overflow"] is False
    np.testing.assert_array_equal(
        current_model.parameter.value,
        legacy_model.parameter.value,
    )
    assert current_scaler.get_scale() == legacy_scaler.get_scale()


def _write_legacy_seed(root: Path, seed: int) -> tuple[str, str]:
    run = root / "s1" / f"seed_{seed}"
    checkpoint = run / "best_resnet34_unet.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-{seed}".encode())
    split_support = {}
    for split, count in {"validation": 89, "test": 90, "bolivia_holdout": 15}.items():
        export = run / "probabilities" / split
        (export / "index_parts").mkdir(parents=True, exist_ok=True)
        (export / "samples").mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(count):
            relative = f"samples/sample-{index:03d}.npz"
            (export / relative).write_bytes(f"{split}-{index}".encode())
            rows.append({"probability_path": relative})
        (export / "index_parts" / "part-000000.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        support = export / "support_contract.json"
        support.write_text(
            json.dumps(
                {
                    "schema": "geobwer.sen1floods11.probability_support.v1",
                    "row_count": count,
                    "aggregate_valid_pixel_count": count,
                }
            ),
            encoding="utf-8",
        )
        quality = export / "input_quality_binding.json"
        quality.write_text("{}", encoding="utf-8")
        split_support[split] = {
            "support_contract_sha256": file_sha256(support),
            "input_quality_binding_sha256": file_sha256(quality),
        }
    normalization_sha = f"normalization-{seed}"
    quality_sha = f"quality-{seed}"
    manifest = {
        "schema": "geobwer.sen1floods11.supervised_resnet34_unet.v5",
        "formal_evidence": True,
        "sensor_mode": "S1",
        "seed": seed,
        "checkpoint_sha256": file_sha256(checkpoint),
        "normalization_sha256": normalization_sha,
        "input_quality_contract": {"sha256": quality_sha},
        "best_validation_iou": 0.5,
        "split_metrics": {"test": 0.4, "bolivia_holdout": 0.3},
        "split_support": split_support,
    }
    (run / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return normalization_sha, quality_sha


def test_carry_forward_requires_provenance_and_binds_all_artifacts(tmp_path):
    source = tmp_path / "legacy"
    normalization_sha, quality_sha = _write_legacy_seed(source, 42)
    log = tmp_path / "v0427.log"
    log.write_text(
        f"GIT_HEAD = {SOURCE_COMMIT}\nVERSION = 0.4.27\n",
        encoding="utf-8",
    )
    output = tmp_path / "migration" / "carry_forward.json"
    build_carry_forward_manifest(
        project_root=Path.cwd(),
        source_root=source,
        source_run_log=log,
        output_path=output,
        seeds=(42,),
    )
    payload = load_carry_forward_manifest(output)
    reused = reuse_carry_forward_seed(
        payload,
        mode="S1",
        seed=42,
        expected_normalization_sha256=normalization_sha,
        expected_input_quality_contract_sha256=quality_sha,
    )
    assert reused is not None
    assert reused["carry_forward"] is True

    sample = source / "s1" / "seed_42" / "probabilities" / "test" / "samples" / "sample-000.npz"
    sample.write_bytes(b"tampered")
    with pytest.raises(Exception, match="source artifacts changed"):
        reuse_carry_forward_seed(
            payload,
            mode="S1",
            seed=42,
            expected_normalization_sha256=normalization_sha,
            expected_input_quality_contract_sha256=quality_sha,
        )
