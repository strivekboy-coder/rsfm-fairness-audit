from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import rsfm_fairness_audit.sen1_supervised_campaign as campaign
from rsfm_fairness_audit.sen1_supervised_campaign import (
    IMPUTATION_POLICY,
    MODE_CHANNELS,
    Sen1SupervisedCampaignError,
    Sen1SupervisedConfig,
    _Dataset,
    _build_input_quality_contract,
    _evaluate,
    _mask,
    _normalize_input,
    _prefix_sha256,
    _training_batch_step,
    compute_train_normalization,
    run_sen1_supervised_campaign,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


def _config(tmp_path: Path, **overrides) -> Sen1SupervisedConfig:
    values = {
        "s1_root": tmp_path / "S1",
        "s2_root": tmp_path / "S2",
        "label_root": tmp_path / "Label",
        "train_split": tmp_path / "train.txt",
        "validation_split": tmp_path / "validation.txt",
        "test_split": tmp_path / "test.txt",
        "bolivia_split": tmp_path / "bolivia.txt",
        "output_dir": tmp_path / "output",
        "sensor_modes": ("S1",),
        "seeds": (42,),
        "max_epochs": 1,
        "batch_size": 2,
        "num_workers": 0,
        "patience": 1,
        "device": "cpu",
        "amp": False,
        "diagnostic_max_samples": 2,
    }
    values.update(overrides)
    return Sen1SupervisedConfig(**values)


def test_mask_preserves_all_ignore_and_rejects_invalid_values(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        campaign,
        "_read_raster",
        lambda _path: np.full((1, 3, 4), -1, dtype=np.int16),
    )
    result = _mask(config, "Ghana_5079")
    assert result.dtype == np.int64
    assert result.shape == (3, 4)
    assert np.array_equal(result, np.full((3, 4), -1, dtype=np.int64))

    monkeypatch.setattr(
        campaign,
        "_read_raster",
        lambda _path: np.asarray([[[0, 1], [-1, 2]]], dtype=np.int16),
    )
    with pytest.raises(Sen1SupervisedCampaignError, match=r"\{-1,0,1\}"):
        _mask(config, "invalid")


@pytest.mark.parametrize("mode", ["S1", "S2", "S1+S2"])
def test_nonfinite_values_are_imputed_to_normalized_zero_for_every_mode(mode):
    channels = MODE_CHANNELS[mode]
    raw = np.arange(channels * 6, dtype=np.float32).reshape(channels, 2, 3)
    raw[0, 0, 0] = np.nan
    if channels > 1:
        raw[1, 0, 1] = np.inf
    if channels > 2:
        raw[2, 0, 2] = -np.inf
    mean = np.linspace(10.0, 20.0, channels, dtype=np.float32)
    std = np.linspace(2.0, 4.0, channels, dtype=np.float32)

    normalized, quality = _normalize_input(
        raw,
        mean=mean,
        std=std,
        prefix=f"{mode}-nonfinite",
        mode=mode,
    )

    assert np.all(np.isfinite(normalized))
    assert normalized[0, 0, 0] == 0.0
    assert quality["channel_counts"][0]["nan_count"] == 1
    if channels > 1:
        assert normalized[1, 0, 1] == 0.0
        assert quality["channel_counts"][1]["posinf_count"] == 1
    if channels > 2:
        assert normalized[2, 0, 2] == 0.0
        assert quality["channel_counts"][2]["neginf_count"] == 1
    assert quality["imputed_value_count"] == min(channels, 3)


@pytest.mark.parametrize("mode", ["S1", "S2", "S1+S2"])
def test_finite_input_is_identical_to_previous_normalization(mode):
    channels = MODE_CHANNELS[mode]
    raw = np.linspace(
        -20.0,
        30.0,
        channels * 12,
        dtype=np.float32,
    ).reshape(channels, 3, 4)
    mean = np.linspace(-5.0, 5.0, channels, dtype=np.float32)
    std = np.linspace(1.5, 3.5, channels, dtype=np.float32)
    expected = (
        raw - mean[:, None, None]
    ) / std[:, None, None]

    normalized, quality = _normalize_input(
        raw,
        mean=mean,
        std=std,
        prefix=f"{mode}-finite",
        mode=mode,
    )

    np.testing.assert_array_equal(normalized, expected)
    assert quality["imputed_value_count"] == 0


@pytest.mark.parametrize("mode", ["S1", "S2", "S1+S2"])
def test_sample_without_any_jointly_finite_pixel_hard_fails(mode):
    channels = MODE_CHANNELS[mode]
    raw = np.full((channels, 2, 3), np.nan, dtype=np.float32)
    with pytest.raises(
        Sen1SupervisedCampaignError,
        match="evaluation splits",
    ):
        _normalize_input(
            raw,
            mean=np.zeros(channels, dtype=np.float32),
            std=np.ones(channels, dtype=np.float32),
            prefix=f"{mode}-all-nonfinite",
            mode=mode,
            split_role="train",
        )


@pytest.mark.parametrize("mode", ["S1", "S2", "S1+S2"])
def test_evaluation_sample_without_finite_required_modality_is_retained(mode):
    channels = MODE_CHANNELS[mode]
    raw = np.full((channels, 2, 3), np.nan, dtype=np.float32)
    normalized, quality = _normalize_input(
        raw,
        mean=np.zeros(channels, dtype=np.float32),
        std=np.ones(channels, dtype=np.float32),
        prefix="Paraguay_34417",
        mode=mode,
        split_role="standard_test",
    )
    np.testing.assert_array_equal(normalized, np.zeros_like(normalized))
    assert quality["fully_missing_modality"] is True


def test_imputation_uses_frozen_train_mean_not_evaluation_values():
    mean = np.asarray([10.0, 20.0], dtype=np.float32)
    std = np.asarray([2.0, 5.0], dtype=np.float32)
    validation = np.asarray(
        [[[np.nan, 1000.0]], [[40.0, 50.0]]],
        dtype=np.float32,
    )
    test = np.asarray(
        [[[np.nan, -1000.0]], [[400.0, 500.0]]],
        dtype=np.float32,
    )
    normalized_validation, _ = _normalize_input(
        validation,
        mean=mean,
        std=std,
        prefix="validation",
        mode="S1",
    )
    normalized_test, _ = _normalize_input(
        test,
        mean=mean,
        std=std,
        prefix="test",
        mode="S1",
    )
    assert normalized_validation[0, 0, 0] == 0.0
    assert normalized_test[0, 0, 0] == 0.0


def test_input_quality_contract_records_per_sample_and_channel_counts(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    arrays = {
        "train-a": np.asarray(
            [[[np.nan, 1.0]], [[2.0, 3.0]]],
            dtype=np.float32,
        ),
        "validation-a": np.asarray(
            [[[4.0, 5.0]], [[np.inf, 6.0]]],
            dtype=np.float32,
        ),
        "test-a": np.asarray(
            [[[7.0, 8.0]], [[9.0, -np.inf]]],
            dtype=np.float32,
        ),
    }
    monkeypatch.setattr(
        campaign,
        "_mode_array",
        lambda _config, prefix, _mode: arrays[prefix].copy(),
    )
    normalization = {
        "normalization_sample_count": 252,
        "sample_prefix_sha256": "official-train-hash",
        "mean": [10.0, 20.0],
        "std": [2.0, 5.0],
    }
    contract = _build_input_quality_contract(
        config,
        {
            "train": ["train-a"],
            "validation": ["validation-a"],
            "test": ["test-a"],
        },
        "S1",
        normalization,
        normalization_sha256="normalization-sha",
    )

    assert contract["imputation_policy"] == IMPUTATION_POLICY
    assert contract["normalization_test_rows_used"] is False
    assert contract["summary"]["sample_count"] == 3
    assert contract["summary"]["samples_with_imputation"] == 3
    assert contract["summary"]["aggregate_imputed_value_count"] == 3
    assert contract["summary"]["maximum_imputed_fraction"] == 0.25
    assert (
        contract["splits"]["train"]["records"][0]["channel_counts"][0][
            "nan_count"
        ]
        == 1
    )
    assert (
        contract["splits"]["validation"]["records"][0]["channel_counts"][1][
            "posinf_count"
        ]
        == 1
    )
    assert (
        contract["splits"]["test"]["records"][0]["channel_counts"][1][
            "neginf_count"
        ]
        == 1
    )


def test_train_normalization_uses_only_jointly_finite_official_train_pixels(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    arrays = {
        "train-a": np.asarray(
            [[[1.0, np.nan, 3.0]], [[10.0, 20.0, 30.0]]],
            dtype=np.float32,
        ),
        "train-b": np.asarray(
            [[[5.0, 7.0, 9.0]], [[50.0, np.inf, 90.0]]],
            dtype=np.float32,
        ),
    }
    monkeypatch.setattr(
        campaign,
        "_mode_array",
        lambda _config, prefix, _mode: arrays[prefix].copy(),
    )

    normalization = compute_train_normalization(
        config,
        ["train-a", "train-b"],
        "S1",
    )

    assert normalization["selection_split"] == "official_train"
    assert normalization["test_rows_used"] is False
    assert normalization["pixel_count"] == 4
    np.testing.assert_allclose(normalization["mean"], [4.5, 45.0])
    assert (
        normalization["input_quality_summary"]["aggregate_imputed_value_count"]
        == 2
    )
    assert normalization["input_quality_summary"]["samples_with_imputation"] == 2


@pytest.mark.skipif(not HAS_TORCH, reason="Dataset label contract requires PyTorch.")
def test_dataset_imputation_does_not_change_label(tmp_path, monkeypatch):
    config = _config(tmp_path)
    raw_image = np.asarray(
        [[[np.nan, 1.0]], [[2.0, 3.0]]],
        dtype=np.float32,
    )
    original_label = np.asarray([[-1, 0]], dtype=np.int64)
    monkeypatch.setattr(
        campaign,
        "_mode_array",
        lambda *_args, **_kwargs: raw_image.copy(),
    )
    monkeypatch.setattr(
        campaign,
        "_mask",
        lambda *_args, **_kwargs: original_label.copy(),
    )
    dataset = _Dataset(
        config,
        ["sample"],
        "S1",
        {"mean": [10.0, 20.0], "std": [2.0, 5.0]},
        augment=False,
        seed=42,
    )
    image, label, _prefix = dataset[0]
    assert float(image[0, 0, 0]) == 0.0
    np.testing.assert_array_equal(label.numpy(), original_label)


@pytest.mark.skipif(not HAS_TORCH, reason="Training-step contract requires PyTorch.")
def test_all_ignore_training_batch_is_skipped_without_parameter_update():
    import torch

    model = torch.nn.Conv2d(2, 1, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = _training_batch_step(
        model=lambda value: model(value)[:, 0],
        optimizer=optimizer,
        scaler=scaler,
        images=torch.ones((1, 2, 3, 3), dtype=torch.float32),
        masks=torch.full((1, 3, 3), -1, dtype=torch.int64),
        prefixes=["Ghana_5079"],
        device=torch.device("cpu"),
        mode="S1",
        amp=False,
    )

    assert result["skipped_all_ignore"] is True
    assert result["aggregate_valid_pixel_count"] == 0
    for previous, current in zip(before, model.parameters()):
        torch.testing.assert_close(previous, current)


@pytest.mark.skipif(not HAS_TORCH, reason="Finite-loss contract requires PyTorch.")
def test_effective_training_batch_with_nan_logits_hard_fails():
    import torch

    class NaNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, images):
            return self.scale * torch.full(
                (images.shape[0], images.shape[2], images.shape[3]),
                float("nan"),
                device=images.device,
            )

    model = NaNModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with pytest.raises(
        Sen1SupervisedCampaignError,
        match="Training logits contain NaN/Inf",
    ):
        _training_batch_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            images=torch.ones((1, 2, 3, 3), dtype=torch.float32),
            masks=torch.zeros((1, 3, 3), dtype=torch.int64),
            prefixes=["valid-prefix"],
            device=torch.device("cpu"),
            mode="S1",
            amp=False,
        )


@pytest.mark.skipif(not HAS_TORCH, reason="Evaluation contract requires PyTorch.")
def test_mixed_valid_and_all_ignore_evaluation_preserves_every_row():
    import torch

    class ZeroModel(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(
                (images.shape[0], images.shape[2], images.shape[3]),
                dtype=images.dtype,
                device=images.device,
            )

    masks = torch.tensor(
        [
            [[-1, -1], [-1, -1]],
            [[-1, 0], [1, -1]],
        ],
        dtype=torch.int64,
    )
    loader = [
        (
            torch.ones((2, 2, 2, 2), dtype=torch.float32),
            masks,
            ["Ghana_5079", "valid-prefix"],
        )
    ]
    _iou, probabilities, targets, prefixes, support = _evaluate(
        ZeroModel(),
        loader,
        torch.device("cpu"),
        mode="S1",
    )
    assert prefixes == ["Ghana_5079", "valid-prefix"]
    assert len(probabilities) == len(targets) == 2
    assert support["valid_pixel_counts"] == [0, 2]
    assert support["all_ignore_row_count"] == 1
    assert support["valid_row_count"] == 1
    assert support["aggregate_valid_pixel_count"] == 2
    assert support["observed_target_values"] == [-1, 0, 1]
    np.testing.assert_array_equal(targets[0], np.full((2, 2), -1))


@pytest.mark.skipif(not HAS_TORCH, reason="Evaluation contract requires PyTorch.")
def test_all_ignore_evaluation_split_hard_fails():
    import torch

    class ZeroModel(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(
                (images.shape[0], images.shape[2], images.shape[3]),
                dtype=images.dtype,
                device=images.device,
            )

    loader = [
        (
            torch.ones((1, 2, 2, 2), dtype=torch.float32),
            torch.full((1, 2, 2), -1, dtype=torch.int64),
            ["Ghana_5079"],
        )
    ]
    with pytest.raises(Sen1SupervisedCampaignError, match="across any row"):
        _evaluate(ZeroModel(), loader, torch.device("cpu"), mode="S1")


def test_diagnostic_normalization_uses_all_252_official_train_prefixes_for_all_modes(
    tmp_path,
    monkeypatch,
):
    train = [f"TrainEvent{i % 4}_{i:03d}" for i in range(252)]
    validation = [f"ValidationEvent{i % 3}_{i:03d}" for i in range(89)]
    test = [f"TestEvent{i % 3}_{i:03d}" for i in range(90)]
    bolivia = [f"Bolivia_{i:03d}" for i in range(15)]
    config = _config(
        tmp_path,
        sensor_modes=("S1", "S2", "S1+S2"),
        diagnostic_max_samples=12,
    )
    split_map = {
        config.train_split: train,
        config.validation_split: validation,
        config.test_split: test,
        config.bolivia_split: bolivia,
    }
    monkeypatch.setattr(campaign, "hydrate_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(campaign, "persist_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        campaign,
        "prepare_terramind_sen1_splits",
        lambda *_args, **_kwargs: {
            "split_counts": {
                "train": 252,
                "validation": 89,
                "test": 90,
                "bolivia_holdout": 15,
            }
        },
    )
    monkeypatch.setattr(
        campaign,
        "read_sen1floods11_split_prefixes",
        lambda path: list(split_map[Path(path)]),
    )
    normalization_calls: list[tuple[str, list[str]]] = []

    def fake_normalization(_config, prefixes, mode):
        prefixes = list(prefixes)
        normalization_calls.append((mode, prefixes))
        channels = MODE_CHANNELS[mode]
        return {
            "schema": "geobwer.sen1floods11.train_normalization.v4",
            "sensor_mode": mode,
            "selection_split": "official_train",
            "test_rows_used": False,
            "imputation_policy": IMPUTATION_POLICY,
            "normalization_sample_count": len(prefixes),
            "sample_count": len(prefixes),
            "sample_prefix_sha256": _prefix_sha256(prefixes),
            "sample_prefixes": prefixes,
            "pixel_count": len(prefixes),
            "mean": [0.0] * channels,
            "std": [1.0] * channels,
            "min": [0.0] * channels,
            "max": [1.0] * channels,
            "input_quality_summary": {
                "sample_count": len(prefixes),
                "samples_with_imputation": 0,
                "aggregate_imputed_value_count": 0,
                "maximum_imputed_fraction": 0.0,
                "channel_totals": [],
            },
            "input_quality_records": [],
        }

    reuse_calls: list[tuple[str, int, int, int]] = []

    def fake_quality(
        _config,
        split_prefixes,
        mode,
        _normalization,
        *,
        normalization_sha256,
    ):
        return {
            "schema": "geobwer.sen1floods11.input_quality.v2",
            "sensor_mode": mode,
            "imputation_policy": IMPUTATION_POLICY,
            "normalization_sha256": normalization_sha256,
            "summary": {
                "sample_count": sum(map(len, split_prefixes.values())),
                "samples_with_imputation": 0,
                "aggregate_imputed_value_count": 0,
                "maximum_imputed_fraction": 0.0,
                "channel_totals": [],
            },
            "splits": {
                split: {
                    "prefix_sha256": _prefix_sha256(prefixes),
                    "summary": {},
                    "records": [],
                }
                for split, prefixes in split_prefixes.items()
            },
        }

    def fake_reuse(
        _run_dir,
        *,
        mode,
        seed,
        expected_validation,
        expected_test,
        expected_bolivia_holdout,
        expected_normalization_sha256,
        expected_input_quality_contract_sha256,
    ):
        assert expected_normalization_sha256
        assert expected_input_quality_contract_sha256
        reuse_calls.append(
            (
                mode,
                expected_validation,
                expected_test,
                expected_bolivia_holdout,
            )
        )
        return {
            "checkpoint": Path("checkpoint.pt"),
            "manifest": Path("run_manifest.json"),
            "validation_export": Path("validation"),
            "test_export": Path("test"),
            "bolivia_holdout_export": Path("bolivia_holdout"),
            "validation_iou": 0.0,
            "test_iou": 0.0,
            "bolivia_holdout_iou": 0.0,
        }

    monkeypatch.setattr(campaign, "compute_train_normalization", fake_normalization)
    monkeypatch.setattr(campaign, "_build_input_quality_contract", fake_quality)
    monkeypatch.setattr(campaign, "_reuse_completed_seed", fake_reuse)

    artifacts = run_sen1_supervised_campaign(config)

    assert [mode for mode, _prefixes in normalization_calls] == [
        "S1",
        "S2",
        "S1+S2",
    ]
    assert all(prefixes == train for _mode, prefixes in normalization_calls)
    assert all(len(prefixes) == 252 for _mode, prefixes in normalization_calls)
    assert reuse_calls == [
        ("S1", 12, 12, 12),
        ("S2", 12, 12, 12),
        ("S1+S2", 12, 12, 12),
    ]

    manifest = json.loads(
        artifacts["campaign_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["formal_evidence"] is False
    assert manifest["official_split_lineage"]["train"]["sample_count"] == 252
    assert manifest["official_split_lineage"]["train"]["prefixes"] == train
    assert (
        manifest["official_split_lineage"]["train"]["prefix_sha256"]
        == _prefix_sha256(train)
    )
    assert manifest["execution_split_lineage"]["train"]["sample_count"] == 12
    assert manifest["execution_split_lineage"]["validation"]["sample_count"] == 12
    assert manifest["execution_split_lineage"]["test"]["sample_count"] == 12
    assert (
        manifest["execution_split_lineage"]["bolivia_holdout"]["sample_count"]
        == 12
    )
    assert (
        manifest["official_split_lineage"]["bolivia_holdout"]["sample_count"]
        == 15
    )
    assert manifest["no_training_or_calibration_leakage"] is True
    assert all(
        contract["normalization_sample_count"] == 252
        for contract in manifest["normalization_contracts"].values()
    )
    assert manifest["schema"] == "geobwer.sen1floods11.supervised_panel.v5"
    assert all(
        contract["imputation_policy"] == IMPUTATION_POLICY
        for contract in manifest["input_quality_contracts"].values()
    )
