from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import ModelAdapter
from rsfm_fairness_audit.config import load_yaml


class PrithviConfigurationError(RuntimeError):
    """Raised when Prithvi-EO-2.0 cannot be loaded or shaped safely."""


class PrithviAdapter(ModelAdapter):
    """Prithvi-EO-2.0 300M non-TL adapter for Sen1Floods11 smoke runs."""

    official_hf_model_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    official_terratorch_model_name = "terratorch_prithvi_eo_v2_300"
    accepted_model_names = (official_hf_model_id, official_terratorch_model_name)
    official_band_names = ["B02", "B03", "B04", "B05", "B06", "B07"]
    official_mean = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    official_std = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

    def __init__(
        self,
        hf_model_id: str = official_hf_model_id,
        terratorch_model_name: str | None = None,
        allow_hf_download: bool = False,
        device: str = "cpu",
        batch_size: int = 2,
        expected_frames: int = 4,
        expected_bands: int = 6,
        image_size: int = 224,
        band_names: Sequence[str] | None = None,
        normalization_mean: Sequence[float] | None = None,
        normalization_std: Sequence[float] | None = None,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.hf_model_id = hf_model_id
        self.terratorch_model_name = terratorch_model_name or hf_model_id
        self.allow_hf_download = allow_hf_download
        self.device = device
        self.batch_size = batch_size
        self.expected_frames = int(expected_frames)
        self.expected_bands = int(expected_bands)
        self.image_size = int(image_size)
        self.band_names = list(band_names or self.official_band_names)
        self.normalization_mean = list(normalization_mean or self.official_mean)
        self.normalization_std = list(normalization_std or self.official_std)
        self.model = model
        self.model_loader = model_loader
        self._torch_device: Any | None = None

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> "PrithviAdapter":
        return cls.from_config(load_yaml(path), model=model, model_loader=model_loader)

    @classmethod
    def from_config(
        cls,
        data: Mapping[str, Any],
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> "PrithviAdapter":
        return cls(
            hf_model_id=str(data.get("hf_model_id", cls.official_hf_model_id)),
            terratorch_model_name=data.get("terratorch_model_name"),
            allow_hf_download=bool(data.get("allow_hf_download", False)),
            device=str(data.get("device", "cpu")),
            batch_size=int(data.get("batch_size", 2)),
            expected_frames=int(data.get("expected_frames", 4)),
            expected_bands=int(data.get("expected_bands", 6)),
            image_size=int(data.get("image_size", 224)),
            band_names=data.get("band_names"),
            normalization_mean=data.get("normalization_mean"),
            normalization_std=data.get("normalization_std"),
            model=model,
            model_loader=model_loader,
        )

    def load_model(self) -> None:
        self._validate_config()
        if self.model is not None:
            self._maybe_eval()
            return
        if self.model_loader is not None:
            self.model = self.model_loader()
            self._maybe_eval()
            return
        if not self.allow_hf_download:
            raise PrithviConfigurationError(
                "Prithvi checkpoint/model download is disabled. Set allow_hf_download: true for the official "
                "TerraTorch/Hugging Face loading path, or inject a mock model in tests."
            )
        try:
            from terratorch.registry import BACKBONE_REGISTRY
        except ImportError as exc:
            raise PrithviConfigurationError("TerraTorch is required for official Prithvi-EO-2.0 loading.") from exc
        self.model = BACKBONE_REGISTRY.build(self.terratorch_model_name, pretrained=True)
        self.model = self._move_to_device(self.model)
        self._maybe_eval()

    def _validate_config(self) -> None:
        if self.hf_model_id != self.official_hf_model_id:
            raise PrithviConfigurationError(
                f"Phase 3 uses only official HF model {self.official_hf_model_id!r}; got {self.hf_model_id!r}."
            )
        if self.terratorch_model_name not in self.accepted_model_names:
            raise PrithviConfigurationError(
                "Prithvi TerraTorch model name must be one of "
                f"{list(self.accepted_model_names)}; got {self.terratorch_model_name!r}."
            )
        if self.expected_frames != 4 or self.expected_bands != 6 or self.image_size != 224:
            raise PrithviConfigurationError("Prithvi-EO-2.0-300M expects 4 frames, 6 bands, and 224x224 inputs.")
        if self.band_names != self.official_band_names:
            raise PrithviConfigurationError(f"Prithvi band_names must match official config: {self.official_band_names}.")
        if len(self.normalization_mean) != 6 or len(self.normalization_std) != 6:
            raise PrithviConfigurationError("Prithvi normalization_mean/std must each contain 6 values.")

    def _resolve_device(self) -> Any:
        if self._torch_device is not None:
            return self._torch_device
        try:
            import torch
        except ImportError as exc:
            raise PrithviConfigurationError("PyTorch is required for Prithvi inference.") from exc
        name = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if name == "auto":
            name = "cpu"
        if name == "cuda" and not torch.cuda.is_available():
            raise PrithviConfigurationError("Prithvi device='cuda' was requested, but CUDA is not available.")
        self._torch_device = torch.device(name)
        return self._torch_device

    def _move_to_device(self, model: Any) -> Any:
        if hasattr(model, "to"):
            model = model.to(self._resolve_device())
        return model

    def _maybe_eval(self) -> None:
        if hasattr(self.model, "eval"):
            self.model.eval()

    def preprocess(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        self._validate_config()
        samples = list(batch["samples"])
        arrays = []
        masks = []
        for sample in samples:
            image = np.asarray(sample["image"], dtype=np.float32)
            if image.ndim == 3:
                # Sen1Floods11 is single timestamp; repeat it to Prithvi's four-frame input.
                image = np.repeat(image[None, :, :, :], self.expected_frames, axis=0)
            if image.shape != (self.expected_frames, self.expected_bands, self.image_size, self.image_size):
                raise PrithviConfigurationError(
                    "Expected Prithvi image shape "
                    f"({self.expected_frames}, {self.expected_bands}, {self.image_size}, {self.image_size}), got {image.shape}."
                )
            arrays.append(image)
            if "mask" in sample:
                masks.append(np.asarray(sample["mask"]))
        images = np.stack(arrays).astype(np.float32)
        mean = np.asarray(self.normalization_mean, dtype=np.float32)[None, None, :, None, None]
        std = np.asarray(self.normalization_std, dtype=np.float32)[None, None, :, None, None]
        normalized = (images - mean) / np.maximum(std, 1e-8)
        result: dict[str, Any] = {
            "images": normalized,
            "raw_images": images,
            "metadata": batch["metadata"],
        }
        if masks:
            result["masks"] = np.stack(masks)
        return result

    def extract_embeddings(self, batch: Mapping[str, Any]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("load_model() must be called before extract_embeddings().")
        images = batch["images"]
        if hasattr(self.model, "extract_embeddings"):
            return np.asarray(self.model.extract_embeddings(images), dtype=np.float32)
        output = self._forward_model(images)
        return self._pool_output(output)

    def _forward_model(self, images: np.ndarray) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise PrithviConfigurationError("PyTorch is required for real Prithvi inference.") from exc
        tensor = torch.as_tensor(images, dtype=torch.float32).to(self._resolve_device())
        with torch.no_grad():
            try:
                return self.model(tensor)
            except Exception as first_exc:
                try:
                    return self.model(tensor.permute(0, 2, 1, 3, 4))
                except Exception as second_exc:
                    raise PrithviConfigurationError(
                        "Prithvi forward failed for both [B,T,C,H,W] and [B,C,T,H,W] layouts. "
                        f"First error: {first_exc}; fallback error: {second_exc}"
                    ) from second_exc

    def _pool_output(self, output: Any) -> np.ndarray:
        if isinstance(output, Mapping):
            for key in ["features", "embedding", "last_hidden_state", "x"]:
                if key in output:
                    output = output[key]
                    break
            else:
                output = next(iter(output.values()))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 2:
            return array
        return array.reshape(array.shape[0], -1).mean(axis=1, keepdims=True) if array.ndim == 1 else array.reshape(array.shape[0], -1)

    def segmentation_features(self, batch: Mapping[str, Any]) -> np.ndarray:
        if self.model is not None and hasattr(self.model, "extract_patch_features"):
            return np.asarray(self.model.extract_patch_features(batch["images"]), dtype=np.float32)
        # Transparent smoke fallback: use normalized 6-band pixels averaged over the repeated time axis.
        return np.asarray(batch["images"], dtype=np.float32).mean(axis=1)

    def get_supported_modalities(self) -> Sequence[str]:
        return ("HLS", "S2_6B_COMPAT")
