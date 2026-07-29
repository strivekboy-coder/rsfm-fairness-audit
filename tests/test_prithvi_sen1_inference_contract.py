from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rsfm_fairness_audit.adapters.prithvi import (
    PRITHVI_SEN1_OFFICIAL_INFERENCE_SOURCE,
    PRITHVI_SEN1_PADDING_POLICY,
    PrithviConfigurationError,
    PrithviSen1Floods11TLAdapter,
    pad_prithvi_sen1_official_windows,
)

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover - the core audit package keeps torch optional
    _TORCH = None


@pytest.mark.parametrize("batch_size", [1, 2])
def test_official_numpy_padding_preserves_temporal_axis(batch_size: int) -> None:
    values = np.arange(
        batch_size * 6 * 1 * 224 * 224,
        dtype=np.float32,
    ).reshape(batch_size, 6, 1, 224, 224)
    padded, contract = pad_prithvi_sen1_official_windows(values)
    assert padded.shape == (batch_size, 6, 1, 512, 512)
    assert contract["pad_bottom"] == 288
    assert contract["pad_right"] == 288
    assert contract["temporal_dimension_preserved"] is True
    assert contract["policy"] == PRITHVI_SEN1_PADDING_POLICY
    assert contract["official_source"] == PRITHVI_SEN1_OFFICIAL_INFERENCE_SOURCE
    np.testing.assert_array_equal(padded[..., :224, :224], values)


def test_official_numpy_padding_is_identity_at_window_size() -> None:
    values = np.zeros((1, 6, 1, 512, 512), dtype=np.float32)
    padded, contract = pad_prithvi_sen1_official_windows(values)
    assert padded.shape == values.shape
    assert contract["pad_bottom"] == 0
    assert contract["pad_right"] == 0
    np.testing.assert_array_equal(padded, values)


def test_official_numpy_padding_rejects_wrong_layout() -> None:
    with pytest.raises(PrithviConfigurationError, match=r"\[B,C,T,H,W\]"):
        pad_prithvi_sen1_official_windows(np.zeros((1, 6, 224, 224), dtype=np.float32))


def _torch():
    return pytest.importorskip("torch")


class _RecordingSegmentationModel:
    """Small torch-compatible wrapper created lazily to keep torch optional."""

    def __new__(cls):
        torch = _torch()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))
                self.register_buffer("offset", torch.zeros(()))
                self.seen_shapes: list[tuple[int, ...]] = []
                self.seen_devices: list[str] = []

            def forward(self, values, **_kwargs):
                self.seen_shapes.append(tuple(values.shape))
                self.seen_devices.append(str(values.device))
                logits = torch.stack(
                    [
                        torch.zeros_like(values[:, 0, 0]) + self.offset,
                        torch.ones_like(values[:, 0, 0]) * self.scale,
                    ],
                    dim=1,
                )
                return SimpleNamespace(output=logits)

        return Model()


@pytest.mark.parametrize("batch_size,image_size", [(1, 224), (2, 224), (1, 512)])
def test_official_forward_contract_returns_cropped_two_class_probabilities(
    batch_size: int,
    image_size: int,
) -> None:
    model = _RecordingSegmentationModel()
    adapter = PrithviSen1Floods11TLAdapter(model=model, device="cpu")
    adapter.load_model()
    images = np.ones((batch_size, 1, 6, image_size, image_size), dtype=np.float32)
    masks = np.zeros((batch_size, image_size, image_size), dtype=np.int16)
    result = adapter.predict_segmentation({"images": images, "masks": masks})
    probabilities = result["probabilities"]
    assert probabilities.shape == (batch_size, 2, image_size, image_size)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert all(shape == (1, 6, 1, 512, 512) for shape in model.seen_shapes)
    assert all(device == "cpu" for device in model.seen_devices)
    assert adapter.debug_records[-1]["padding_contract"]["pad_bottom"] == (
        0 if image_size == 512 else 288
    )


def test_explicit_cpu_moves_wrapped_lightning_model_and_validates_tensors() -> None:
    model = _RecordingSegmentationModel()
    wrapped = SimpleNamespace(model=model, datamodule=None)
    adapter = PrithviSen1Floods11TLAdapter(model_loader=lambda: wrapped, device="cpu")
    adapter.load_model()
    torch = _torch()
    assert next(adapter.model.parameters()).device == torch.device("cpu")
    assert next(adapter.model.buffers()).device == torch.device("cpu")
    assert adapter.device_contract["resolved_device"] == "cpu"
    assert adapter.device_contract["status"] == "pass"


def test_explicit_cuda_never_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    adapter = PrithviSen1Floods11TLAdapter(model=_RecordingSegmentationModel(), device="cuda")
    with pytest.raises(PrithviConfigurationError, match="CUDA is not available"):
        adapter.load_model()


@pytest.mark.skipif(
    _TORCH is None or not bool(_TORCH.cuda.is_available()),
    reason="CUDA is required to verify the real wrapped-model migration contract.",
)
def test_wrapped_cpu_model_is_moved_to_requested_cuda() -> None:
    torch = _torch()
    model = _RecordingSegmentationModel().cpu()
    wrapped = SimpleNamespace(model=model, datamodule=None)
    adapter = PrithviSen1Floods11TLAdapter(model_loader=lambda: wrapped, device="cuda")
    adapter.load_model()
    expected = torch.device("cuda", torch.cuda.current_device())
    assert next(adapter.model.parameters()).device == expected
    assert next(adapter.model.buffers()).device == expected
    assert adapter._resolve_device() == expected
    images = np.ones((1, 1, 6, 224, 224), dtype=np.float32)
    masks = np.zeros((1, 224, 224), dtype=np.int16)
    result = adapter.predict_segmentation({"images": images, "masks": masks})
    assert result["probabilities"].shape == (1, 2, 224, 224)
    assert adapter.device_contract["model_input_device"] == str(expected)
