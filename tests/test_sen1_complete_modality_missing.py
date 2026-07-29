from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import scripts.colab.run_terramind_sen1floods11_final_colab as terramind_runner
from rsfm_fairness_audit.sen1_input_quality import (
    FORMAL_COMPLETE_MODALITY_MISSING_EXPECTATION,
    Sen1InputQualityError,
    input_quality_summary,
    normalize_mode_input,
    normalize_named_modalities,
)
from rsfm_fairness_audit.terratorch_exports import (
    mean_impute_and_normalize_tensor,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_evaluation_only_fully_missing_s1_is_exact_normalized_zero():
    raw = np.full((2, 3, 4), np.nan, dtype=np.float32)
    normalized, quality = normalize_mode_input(
        raw,
        mean=np.asarray([-10.0, -20.0], dtype=np.float32),
        std=np.asarray([4.0, 5.0], dtype=np.float32),
        prefix="Paraguay_34417",
        mode="S1",
        split_role="standard_test",
    )
    np.testing.assert_array_equal(normalized, np.zeros_like(normalized))
    assert quality["availability_status"] == "fully_missing_modality"
    assert quality["fully_missing_modalities"] == ["S1"]
    assert quality["modalities"][0]["nan_count"] == 24


@pytest.mark.parametrize("split_role", ["train", "validation"])
def test_training_or_validation_fully_missing_required_modality_hard_fails(
    split_role,
):
    with pytest.raises(Sen1InputQualityError, match="evaluation splits"):
        normalize_mode_input(
            np.full((2, 2, 2), np.nan, dtype=np.float32),
            mean=np.zeros(2, dtype=np.float32),
            std=np.ones(2, dtype=np.float32),
            prefix="invalid",
            mode="S1",
            split_role=split_role,
        )


def test_fusion_imputes_only_s1_and_preserves_s2_normalization():
    s1 = np.full((2, 2, 3), np.nan, dtype=np.float32)
    s2 = np.arange(13 * 6, dtype=np.float32).reshape(13, 2, 3)
    raw = np.concatenate([s1, s2], axis=0)
    mean = np.arange(15, dtype=np.float32) + 10.0
    std = np.arange(15, dtype=np.float32) + 2.0
    normalized, quality = normalize_mode_input(
        raw,
        mean=mean,
        std=std,
        prefix="Paraguay_34417",
        mode="S1+S2",
        split_role="standard_test",
    )
    np.testing.assert_array_equal(normalized[:2], np.zeros_like(normalized[:2]))
    np.testing.assert_array_equal(
        normalized[2:],
        (s2 - mean[2:, None, None]) / std[2:, None, None],
    )
    assert quality["fully_missing_modalities"] == ["S1"]
    assert quality["modalities"][1]["availability_status"] == "available"


def test_modality_aware_fusion_does_not_require_cross_modality_joint_pixels():
    s1 = np.asarray(
        [[[np.nan, 1.0]], [[2.0, np.nan]]],
        dtype=np.float32,
    )
    s2 = np.ones((13, 1, 2), dtype=np.float32)
    raw = np.concatenate([s1, s2], axis=0)
    normalized, quality = normalize_mode_input(
        raw,
        mean=np.zeros(15, dtype=np.float32),
        std=np.ones(15, dtype=np.float32),
        prefix="partial",
        mode="S1+S2",
        split_role="train",
    )
    assert np.isfinite(normalized).all()
    assert quality["jointly_finite_pixel_count"] == 0
    assert quality["fully_missing_modalities"] == []
    assert quality["partial_nonfinite_modalities"] == ["S1"]


def test_summary_discloses_exact_missing_sample_and_modality():
    records = []
    for sample_id, fully_missing in (
        ("ordinary", False),
        ("Paraguay_34417", True),
    ):
        raw = (
            np.full((2, 1, 2), np.nan, dtype=np.float32)
            if fully_missing
            else np.ones((2, 1, 2), dtype=np.float32)
        )
        _normalized, quality = normalize_mode_input(
            raw,
            mean=np.zeros(2, dtype=np.float32),
            std=np.ones(2, dtype=np.float32),
            prefix=sample_id,
            mode="S1",
            split_role="standard_test",
        )
        records.append(quality)
    summary = input_quality_summary(records, mode="S1")
    assert summary["fully_missing_modality_count"] == 1
    assert summary["fully_missing_sample_ids"] == ["Paraguay_34417"]
    assert summary["fully_missing_modalities_by_sample"] == {
        "Paraguay_34417": ["S1"]
    }
    assert (
        FORMAL_COMPLETE_MODALITY_MISSING_EXPECTATION["S1"]["test"]
        == summary["fully_missing_modalities_by_sample"]
    )


def test_named_terramind_modalities_use_mean_not_raw_zero():
    s1 = np.full((2, 2, 2), np.nan, dtype=np.float32)
    s2 = np.arange(13 * 4, dtype=np.float32).reshape(13, 2, 2)
    normalized, quality = normalize_named_modalities(
        {"S2L1C": s2, "S1GRD": s1},
        means={
            "S2L1C": np.arange(13, dtype=np.float32),
            "S1GRD": [-12.599, -20.293],
        },
        stds={"S2L1C": np.ones(13), "S1GRD": [5.195, 5.890]},
        prefix="Paraguay_34417",
        split_role="standard_test",
    )
    np.testing.assert_array_equal(
        normalized["S1GRD"], np.zeros_like(normalized["S1GRD"])
    )
    np.testing.assert_array_equal(
        normalized["S2L1C"],
        s2 - np.arange(13, dtype=np.float32)[:, None, None],
    )
    assert quality["fully_missing_modalities"] == ["S1"]


def test_named_terramind_training_rejects_fully_missing_modality():
    with pytest.raises(Sen1InputQualityError, match="evaluation splits"):
        normalize_named_modalities(
            {"S1GRD": np.full((2, 2, 2), np.nan, dtype=np.float32)},
            means={"S1GRD": [-12.599, -20.293]},
            stds={"S1GRD": [5.195, 5.890]},
            prefix="train_missing",
            split_role="train",
        )


@pytest.mark.skipif(not HAS_TORCH, reason="Tensor normalization requires PyTorch.")
def test_terramind_tensor_path_is_finite_and_zero_at_nonfinite_values():
    import torch

    tensor = torch.tensor(
        [[[[float("nan"), -12.599]], [[float("inf"), -20.293]]]],
        dtype=torch.float32,
    )
    output = mean_impute_and_normalize_tensor(
        tensor,
        mean=[-12.599, -20.293],
        std=[5.195, 5.890],
    )
    assert bool(torch.isfinite(output).all())
    assert float(output[0, 0, 0, 0]) == 0.0
    assert float(output[0, 1, 0, 0]) == 0.0
    assert float(output[0, 0, 0, 1]) == 0.0
    assert float(output[0, 1, 0, 1]) == 0.0


def test_partial_nonfinite_behavior_remains_mean_imputation():
    raw = np.asarray(
        [[[np.nan, 12.0]], [[22.0, np.inf]]],
        dtype=np.float32,
    )
    normalized, quality = normalize_mode_input(
        raw,
        mean=np.asarray([10.0, 20.0]),
        std=np.asarray([2.0, 2.0]),
        prefix="partial",
        mode="S1",
        split_role="train",
    )
    np.testing.assert_array_equal(
        normalized,
        np.asarray([[[0.0, 1.0]], [[1.0, 0.0]]], dtype=np.float32),
    )
    assert quality["partial_nonfinite_imputed"] is True


def test_terramind_contract_and_export_binding_disclose_paraguay(
    tmp_path,
    monkeypatch,
):
    split_paths = {}
    members = {
        "train": ["Train_001"],
        "validation": ["Validation_001"],
        "test": ["Paraguay_34417"],
        "bolivia_holdout": ["Bolivia_001"],
    }
    for split, prefixes in members.items():
        path = tmp_path / f"{split}.txt"
        path.write_text("".join(f"{value}\n" for value in prefixes))
        split_paths[split] = path

    def fake_read(path):
        if path.name == "Paraguay_34417_S1Hand.tif":
            return np.full((2, 2, 2), np.nan, dtype=np.float32)
        if path.name.endswith("_S1Hand.tif"):
            return np.ones((2, 2, 2), dtype=np.float32)
        return np.ones((13, 2, 2), dtype=np.float32)

    monkeypatch.setattr(terramind_runner, "_read_raster", fake_read)
    contracts = terramind_runner._build_terramind_input_quality_contracts(
        s1_root=tmp_path / "S1",
        s2_root=tmp_path / "S2",
        split_paths=split_paths,
        modes=("S1", "S2", "S1+S2"),
        output_dir=tmp_path / "quality",
    )
    assert contracts["S1"]["contract"]["splits"]["test"]["summary"][
        "fully_missing_sample_ids"
    ] == ["Paraguay_34417"]
    assert contracts["S1+S2"]["contract"]["splits"]["test"]["summary"][
        "fully_missing_modalities_by_sample"
    ] == {"Paraguay_34417": ["S1"]}
    assert contracts["S2"]["contract"]["splits"]["test"]["summary"][
        "fully_missing_modality_count"
    ] == 0

    export = tmp_path / "export"
    export.mkdir()
    binding = terramind_runner._bind_input_quality(
        export,
        split="test",
        mode="S1+S2",
        contract=contracts["S1+S2"]["contract"],
        contract_path=contracts["S1+S2"]["path"],
        contract_sha256=contracts["S1+S2"]["sha256"],
    )
    payload = __import__("json").loads(binding.read_text())
    assert payload["split_role"] == "standard_test"
    assert payload["fully_missing_modality_records"][0]["sample_id"] == (
        "Paraguay_34417"
    )


def test_exact_sample_runtime_gate_absolute_help_outside_repository(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    script = (
        project_root
        / "scripts"
        / "colab"
        / "validate_sen1_complete_modality_missing_colab.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Paraguay_34417" in result.stdout
