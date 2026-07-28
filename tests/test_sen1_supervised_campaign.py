from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import rsfm_fairness_audit.sen1_supervised_campaign as campaign
from rsfm_fairness_audit.sen1_supervised_campaign import (
    MODE_CHANNELS,
    Sen1SupervisedCampaignError,
    Sen1SupervisedConfig,
    _evaluate,
    _mask,
    _prefix_sha256,
    _training_batch_step,
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
    config = _config(
        tmp_path,
        sensor_modes=("S1", "S2", "S1+S2"),
        diagnostic_max_samples=12,
    )
    split_map = {
        config.train_split: train,
        config.validation_split: validation,
        config.test_split: test,
    }
    monkeypatch.setattr(campaign, "hydrate_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(campaign, "persist_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        campaign,
        "prepare_terramind_sen1_splits",
        lambda *_args, **_kwargs: {"split_counts": {"train": 252, "validation": 89, "test": 90}},
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
            "schema": "geobwer.sen1floods11.train_normalization.v2",
            "sensor_mode": mode,
            "selection_split": "official_train",
            "test_rows_used": False,
            "normalization_sample_count": len(prefixes),
            "sample_count": len(prefixes),
            "sample_prefix_sha256": _prefix_sha256(prefixes),
            "sample_prefixes": prefixes,
            "pixel_count": len(prefixes),
            "mean": [0.0] * channels,
            "std": [1.0] * channels,
            "min": [0.0] * channels,
            "max": [1.0] * channels,
        }

    reuse_calls: list[tuple[str, int, int]] = []

    def fake_reuse(_run_dir, *, mode, seed, expected_validation, expected_test):
        reuse_calls.append((mode, expected_validation, expected_test))
        return {
            "checkpoint": Path("checkpoint.pt"),
            "manifest": Path("run_manifest.json"),
            "validation_export": Path("validation"),
            "test_export": Path("test"),
            "validation_iou": 0.0,
            "test_iou": 0.0,
        }

    monkeypatch.setattr(campaign, "compute_train_normalization", fake_normalization)
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
        ("S1", 12, 12),
        ("S2", 12, 12),
        ("S1+S2", 12, 12),
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
    assert all(
        contract["normalization_sample_count"] == 252
        for contract in manifest["normalization_contracts"].values()
    )
