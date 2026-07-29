from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import ModelAdapter
from rsfm_fairness_audit.config import load_yaml


class PrithviConfigurationError(RuntimeError):
    """Raised when Prithvi-EO-2.0 cannot be loaded or shaped safely."""


PRITHVI_SEN1_OFFICIAL_INFERENCE_SOURCE = (
    "https://huggingface.co/ibm-nasa-geospatial/"
    "Prithvi-EO-2.0-300M-TL-Sen1Floods11/blob/main/inference.py"
)
PRITHVI_SEN1_OFFICIAL_WINDOW_SIZE = 512
PRITHVI_SEN1_PADDING_POLICY = "numpy_reflect_right_bottom_spatial_only"


def pad_prithvi_sen1_official_windows(
    values: np.ndarray,
    *,
    window_size: int = PRITHVI_SEN1_OFFICIAL_WINDOW_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the official Sen1Floods11 inference padding to [B,C,T,H,W].

    The released inference script uses NumPy ``reflect`` padding on only the
    final two spatial axes before extracting 512x512 windows. Keeping this as a
    NumPy helper avoids PyTorch's unsupported four-value reflect padding on a
    five-dimensional tensor and makes the frozen policy independently testable.
    """

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 5:
        raise PrithviConfigurationError(
            f"Official Prithvi Sen1 padding expects [B,C,T,H,W], got {array.shape}."
        )
    if window_size <= 0:
        raise PrithviConfigurationError("Official Prithvi inference window must be positive.")
    original_shape = tuple(int(value) for value in array.shape)
    height, width = original_shape[-2:]
    pad_h = (int(window_size) - (height % int(window_size))) % int(window_size)
    pad_w = (int(window_size) - (width % int(window_size))) % int(window_size)
    if (pad_h and height < 2) or (pad_w and width < 2):
        raise PrithviConfigurationError(
            "NumPy reflect padding requires spatial axes of length at least two."
        )
    if pad_h or pad_w:
        array = np.pad(
            array,
            ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="reflect",
        )
    padded_shape = tuple(int(value) for value in array.shape)
    if padded_shape[:3] != original_shape[:3]:
        raise PrithviConfigurationError("Official padding changed batch, channel, or temporal dimensions.")
    if padded_shape[-2] % window_size or padded_shape[-1] % window_size:
        raise PrithviConfigurationError("Official padding did not produce complete inference windows.")
    contract = {
        "policy": PRITHVI_SEN1_PADDING_POLICY,
        "official_source": PRITHVI_SEN1_OFFICIAL_INFERENCE_SOURCE,
        "window_size": int(window_size),
        "original_shape_B_C_T_H_W": list(original_shape),
        "padded_shape_B_C_T_H_W": list(padded_shape),
        "pad_bottom": int(pad_h),
        "pad_right": int(pad_w),
        "temporal_dimension_preserved": padded_shape[2] == original_shape[2],
    }
    return np.asarray(array, dtype=np.float32), contract


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
        if str(name).startswith("cuda") and not torch.cuda.is_available():
            raise PrithviConfigurationError("Prithvi device='cuda' was requested, but CUDA is not available.")
        resolved = torch.device(name)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self._torch_device = resolved
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
        # The fairness pipeline needs one pooled embedding per sample. Some
        # Prithvi/TerraTorch outputs are token or feature maps; flattening them
        # creates very wide embeddings and can OOM Colab after chunks are
        # written. Pool instead and keep only sample-level representations.
        if array.ndim == 2:
            return array
        if array.ndim == 1:
            return array.reshape(-1, 1)
        if array.ndim == 3:
            return array.mean(axis=1)
        if array.ndim == 4:
            return array.mean(axis=(2, 3))
        if array.ndim == 5:
            return array.mean(axis=(2, 3, 4))
        return array.reshape(array.shape[0], -1)

    def segmentation_features(self, batch: Mapping[str, Any]) -> np.ndarray:
        if self.model is not None and hasattr(self.model, "extract_patch_features"):
            return np.asarray(self.model.extract_patch_features(batch["images"]), dtype=np.float32)
        # Transparent smoke fallback: use normalized 6-band pixels averaged over the repeated time axis.
        return np.asarray(batch["images"], dtype=np.float32).mean(axis=1)

    def get_supported_modalities(self) -> Sequence[str]:
        return ("HLS", "S2_6B_COMPAT")


class PrithviSen1Floods11TLAdapter(PrithviAdapter):
    """Official Prithvi-EO-2.0 Sen1Floods11 segmentation fine-tune adapter."""

    official_hf_model_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
    official_checkpoint_name = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
    official_band_names = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
    accepted_model_names = (official_hf_model_id,)
    protocol_model_name = "prithvi_tl_sen1floods11"
    model_family = "Prithvi"
    adaptation_protocol = "task_adapted_decoder"
    training_budget = "official_sen1floods11_finetune"
    checkpoint_source = "official_huggingface"
    expected_band_profile = "prithvi_tl_sen1floods11"

    def __init__(
        self,
        hf_model_id: str = official_hf_model_id,
        terratorch_model_name: str | None = None,
        allow_hf_download: bool = False,
        device: str = "cpu",
        batch_size: int = 1,
        expected_frames: int = 1,
        expected_bands: int = 6,
        image_size: int = 224,
        band_names: Sequence[str] | None = None,
        normalization_mean: Sequence[float] | None = None,
        normalization_std: Sequence[float] | None = None,
        checkpoint_path: str | Path | None = None,
        terratorch_config_path: str | Path | None = None,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            hf_model_id=hf_model_id,
            terratorch_model_name=terratorch_model_name or hf_model_id,
            allow_hf_download=allow_hf_download,
            device=device,
            batch_size=batch_size,
            expected_frames=expected_frames,
            expected_bands=expected_bands,
            image_size=image_size,
            band_names=band_names or self.official_band_names,
            normalization_mean=normalization_mean or [0.0] * expected_bands,
            normalization_std=normalization_std or [1.0] * expected_bands,
            model=model,
            model_loader=model_loader,
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.terratorch_config_path = Path(terratorch_config_path) if terratorch_config_path else None
        self.datamodule: Any | None = None
        self.load_diagnostics: dict[str, Any] = {}
        self.debug_records: list[dict[str, Any]] = []
        self.device_contract: dict[str, Any] = {}
        self._device_input_logged = False
        self.official_inference_window_size = PRITHVI_SEN1_OFFICIAL_WINDOW_SIZE

    @classmethod
    def from_config(
        cls,
        data: Mapping[str, Any],
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> "PrithviSen1Floods11TLAdapter":
        return cls(
            hf_model_id=str(data.get("hf_model_id", cls.official_hf_model_id)),
            terratorch_model_name=data.get("terratorch_model_name"),
            allow_hf_download=bool(data.get("allow_hf_download", False)),
            device=str(data.get("device", "cpu")),
            batch_size=int(data.get("batch_size", 1)),
            expected_frames=int(data.get("expected_frames", 1)),
            expected_bands=int(data.get("expected_bands", 6)),
            image_size=int(data.get("image_size", 224)),
            band_names=data.get("band_names"),
            checkpoint_path=data.get("checkpoint_path"),
            terratorch_config_path=data.get("terratorch_config_path") or data.get("config_path"),
            model=model,
            model_loader=model_loader,
        )

    def load_model(self) -> None:
        self._validate_config()
        if self.model is not None:
            self.model = self._move_and_validate_requested_device(self.model)
            self._maybe_eval()
            self._set_load_diagnostics("injected_model")
            return
        if self.model_loader is not None:
            self.model = self.model_loader()
            self.model = self._unwrap_lightning_inference(self.model)
            self._maybe_eval()
            self._set_load_diagnostics("model_loader")
            return
        if self.terratorch_config_path and self.checkpoint_path:
            self.model = self._load_lightning_inference(self.terratorch_config_path, self.checkpoint_path)
            self._maybe_eval()
            self._set_load_diagnostics("local_config_checkpoint")
            return
        if not self.allow_hf_download:
            raise PrithviConfigurationError(
                "Official Prithvi Sen1Floods11 TL loading is disabled. Set allow_hf_download: true, "
                "or provide terratorch_config_path plus checkpoint_path, or inject a mock model in tests."
            )
        config_path, checkpoint_path = self._download_official_inference_files()
        self.terratorch_config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model = self._load_lightning_inference(config_path, checkpoint_path)
        self._maybe_eval()
        self._set_load_diagnostics("hf_config_checkpoint")

    def _download_official_inference_files(self) -> tuple[Path, Path]:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise PrithviConfigurationError(
                "huggingface_hub is required to download the official Prithvi Sen1Floods11 TL config/checkpoint."
            ) from exc
        config_path = Path(hf_hub_download(repo_id=self.hf_model_id, filename="config.yaml"))
        checkpoint_path = Path(hf_hub_download(repo_id=self.hf_model_id, filename=self.official_checkpoint_name))
        return config_path, checkpoint_path

    def _load_lightning_inference(self, config_path: Path, checkpoint_path: Path) -> Any:
        try:
            from terratorch.cli_tools import LightningInferenceModel
        except ImportError as exc:
            raise PrithviConfigurationError("TerraTorch is required for official Prithvi Sen1Floods11 TL loading.") from exc
        inference_model = LightningInferenceModel.from_config(str(config_path), str(checkpoint_path))
        return self._unwrap_lightning_inference(inference_model)

    def _validate_config(self) -> None:
        if self.hf_model_id != self.official_hf_model_id:
            raise PrithviConfigurationError(
                f"Official Sen1Floods11 TL runs use {self.official_hf_model_id!r}; got {self.hf_model_id!r}."
            )
        if self.terratorch_model_name not in self.accepted_model_names:
            raise PrithviConfigurationError(
                f"Prithvi Sen1Floods11 TL model name must be {self.official_hf_model_id!r}; got {self.terratorch_model_name!r}."
            )
        if self.expected_frames < 1 or self.expected_bands != 6:
            raise PrithviConfigurationError("Prithvi Sen1Floods11 TL expects at least one timestamp and six optical bands.")
        if self.band_names != self.official_band_names:
            raise PrithviConfigurationError(f"Prithvi band_names must match official config: {self.official_band_names}.")

    def _unwrap_lightning_inference(self, model: Any) -> Any:
        if hasattr(model, "model"):
            self.datamodule = getattr(model, "datamodule", None)
            wrapped = model.model
            return self._move_and_validate_requested_device(wrapped)
        return self._move_and_validate_requested_device(model)

    @staticmethod
    def _tensor_devices(model: Any) -> list[str]:
        devices: list[str] = []
        for accessor in ("parameters", "buffers"):
            if not hasattr(model, accessor):
                continue
            try:
                tensors = getattr(model, accessor)()
            except TypeError:
                continue
            for tensor in tensors:
                if hasattr(tensor, "device"):
                    devices.append(str(tensor.device))
        return devices

    def _move_and_validate_requested_device(self, model: Any) -> Any:
        has_tensor_interface = any(
            hasattr(model, accessor) for accessor in ("to", "parameters", "buffers")
        )
        if not has_tensor_interface and str(self.device) in {"cpu", "auto"}:
            # Lightweight NumPy mocks keep the core package testable without
            # installing Torch. Official Lightning models always expose the
            # tensor interface and therefore always take the strict path below.
            self.device_contract = {
                "requested_device": str(self.device),
                "resolved_device": "not_applicable_numpy_mock",
                "model_tensor_count": 0,
                "model_tensor_devices": [],
                "model_parameter_device": "no_tensors_reported",
                "strict_no_cpu_fallback": False,
                "status": "non_torch_test_double",
            }
            return model
        requested = self._resolve_device()
        if hasattr(model, "to"):
            moved = model.to(requested)
            if moved is not None:
                model = moved
        devices = self._tensor_devices(model)
        if requested.type == "cuda" and not devices:
            raise PrithviConfigurationError(
                "Prithvi device='cuda' requires a model exposing parameters or buffers; "
                "the official model device could not be certified."
            )
        mismatches = sorted({value for value in devices if value != str(requested)})
        if mismatches:
            raise PrithviConfigurationError(
                "Prithvi model parameters/buffers did not move to the requested device "
                f"{requested}; observed={sorted(set(devices))}."
            )
        self._torch_device = requested
        self.device_contract = {
            "requested_device": str(self.device),
            "resolved_device": str(requested),
            "model_tensor_count": len(devices),
            "model_tensor_devices": sorted(set(devices)),
            "model_parameter_device": devices[0] if devices else "no_tensors_reported",
            "strict_no_cpu_fallback": str(self.device).startswith("cuda"),
            "status": "pass",
        }
        return model

    def _validate_runtime_device_contract(self, model_input: Any | None = None) -> None:
        requested = self._resolve_device()
        devices = self._tensor_devices(self.model)
        mismatches = sorted({value for value in devices if value != str(requested)})
        if mismatches:
            raise PrithviConfigurationError(
                "Prithvi device contract drifted before forward: "
                f"requested={requested}, model_devices={sorted(set(devices))}."
            )
        if model_input is not None:
            input_device = str(getattr(model_input, "device", "not_reported"))
            if input_device != str(requested):
                raise PrithviConfigurationError(
                    f"Prithvi input device {input_device} does not match model device {requested}."
                )
            self.device_contract["model_input_device"] = input_device
            if not self._device_input_logged:
                print(
                    "[prithvi:sen1:device] "
                    f"resolved={requested} "
                    f"model_parameter_device={self.device_contract.get('model_parameter_device')} "
                    f"model_input_device={input_device}",
                    flush=True,
                )
                self._device_input_logged = True

    def _set_load_diagnostics(self, load_source: str) -> None:
        missing_keys = getattr(self.model, "missing_keys", None)
        unexpected_keys = getattr(self.model, "unexpected_keys", None)
        self.load_diagnostics = {
            "load_source": load_source,
            "hf_model_id": self.hf_model_id,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "",
            "terratorch_config_path": str(self.terratorch_config_path) if self.terratorch_config_path else "",
            "model_class": f"{self.model.__class__.__module__}.{self.model.__class__.__name__}" if self.model is not None else "",
            "datamodule_class": f"{self.datamodule.__class__.__module__}.{self.datamodule.__class__.__name__}" if self.datamodule is not None else "",
            "head_class": f"{getattr(self.model, 'head').__class__.__module__}.{getattr(self.model, 'head').__class__.__name__}"
            if self.model is not None and hasattr(self.model, "head")
            else "",
            "decoder_class": f"{getattr(self.model, 'decoder').__class__.__module__}.{getattr(self.model, 'decoder').__class__.__name__}"
            if self.model is not None and hasattr(self.model, "decoder")
            else "",
            "missing_keys": list(missing_keys) if missing_keys is not None else "not_reported_by_loader",
            "unexpected_keys": list(unexpected_keys) if unexpected_keys is not None else "not_reported_by_loader",
            "class_semantics": {"0": "no_water/background", "1": "water/flood", "-1": "ignore_label_in_ground_truth"},
            "expected_input_layout": "[B,6,1,H,W]",
            "expected_band_order": self.official_band_names,
            "official_preprocessing": "scale_to_0_1_then_datamodule.test_transform_and_aug_then_restore_[B,C,1,H,W]",
            "official_inference_window_size": self.official_inference_window_size,
            "official_inference_source": PRITHVI_SEN1_OFFICIAL_INFERENCE_SOURCE,
            "padding_policy": PRITHVI_SEN1_PADDING_POLICY,
            "device_contract": dict(self.device_contract),
            "expected_input_size_note": "Official inference uses 512x512 sliding windows; config trains with 224x224 random crops. This adapter applies the released NumPy spatial reflect-padding/window/crop policy.",
        }

    def preprocess(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        self._validate_config()
        samples = list(batch["samples"])
        for meta in batch.get("metadata", []):
            profile = meta.get("band_profile") if isinstance(meta, Mapping) else None
            if profile and str(profile) != self.expected_band_profile:
                raise PrithviConfigurationError(
                    "Official Prithvi Sen1Floods11 TL requires prepared band_profile="
                    f"{self.expected_band_profile!r} with BLUE/GREEN/RED/NIR_NARROW/SWIR_1/SWIR_2. "
                    f"Got {profile!r}. Re-run scripts/prepare_sen1floods11_subset.py with "
                    "--band-profile prithvi_tl_sen1floods11; cached B02-B07 prepared zips are invalid for this model."
                )
        arrays = []
        masks = []
        for sample in samples:
            image = np.asarray(sample["image"], dtype=np.float32)
            if image.ndim == 3:
                image = image[None, :, :, :]
            if image.ndim != 4 or image.shape[1] != self.expected_bands:
                raise PrithviConfigurationError(f"Expected Prithvi TL image [T,6,H,W], got {image.shape}.")
            if image.shape[0] > self.expected_frames:
                image = image[: self.expected_frames]
            elif image.shape[0] < self.expected_frames:
                image = np.repeat(image[:1], self.expected_frames, axis=0)
            arrays.append(image)
            if "mask" in sample:
                masks.append(np.asarray(sample["mask"]))
        images = np.stack(arrays).astype(np.float32)
        result: dict[str, Any] = {
            "images": images,
            "raw_images": images.copy(),
            "metadata": batch["metadata"],
        }
        if masks:
            result["masks"] = np.stack(masks)
        return result

    def predict_segmentation(self, batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("load_model() must be called before predict_segmentation().")
        if hasattr(self.model, "predict_segmentation"):
            output = self.model.predict_segmentation(batch)
            return {
                "predictions": np.asarray(output["predictions"], dtype=np.int16),
                "score_maps": np.asarray(output["score_maps"], dtype=np.float32),
                "confidence": np.asarray(output.get("confidence", output["score_maps"]), dtype=np.float32),
                **({"probabilities": np.asarray(output["probabilities"], dtype=np.float32)} if "probabilities" in output else {}),
            }
        logits = self._forward_segmentation(batch["images"])
        target_shape = tuple(np.asarray(batch["masks"]).shape[-2:]) if "masks" in batch else tuple(logits.shape[-2:])
        logits = self._resize_logits(logits, target_shape)
        probabilities = self._softmax(logits, axis=1)
        predictions = np.argmax(probabilities, axis=1).astype(np.int16)
        self._record_prediction_debug(batch["images"], logits, probabilities, predictions)
        return {
            "predictions": predictions,
            "score_maps": probabilities[:, 1, :, :].astype(np.float32),
            "confidence": np.max(probabilities, axis=1).astype(np.float32),
            "probabilities": probabilities.astype(np.float32),
        }

    def _forward_segmentation(self, images: np.ndarray) -> np.ndarray:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise PrithviConfigurationError("PyTorch is required for official Prithvi Sen1Floods11 TL inference.") from exc
        array = np.asarray(images, dtype=np.float32)
        if array.ndim != 5:
            raise PrithviConfigurationError(f"Expected prepared Prithvi input [B,T,C,H,W], got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise PrithviConfigurationError("Prepared Prithvi input contains NaN/Inf.")
        if np.nanmax(array) > 1.5:
            array = array / 10000.0
        official_layout = np.transpose(array, (0, 2, 1, 3, 4))
        original_h, original_w = official_layout.shape[-2:]
        window_size = int(self.official_inference_window_size)
        padded, padding_contract = pad_prithvi_sen1_official_windows(
            official_layout,
            window_size=window_size,
        )
        tensor = torch.as_tensor(padded, dtype=torch.float32)
        self._validate_runtime_device_contract()
        batch_logits = []
        model_inputs_seen: list[list[int]] = []
        with torch.no_grad():
            for sample in tensor:
                sample_logits = []
                for top in range(0, sample.shape[-2], window_size):
                    row_logits = []
                    for left in range(0, sample.shape[-1], window_size):
                        window = sample[:, :, top : top + window_size, left : left + window_size]
                        model_input = self._prepare_official_window(window).to(self._resolve_device())
                        if tuple(model_input.shape[2:]) != (self.expected_frames, window_size, window_size):
                            raise PrithviConfigurationError(
                                "Official Prithvi transformed input must preserve T and the 512x512 window; "
                                f"got {tuple(model_input.shape)}."
                            )
                        self._validate_runtime_device_contract(model_input)
                        model_inputs_seen.append(list(model_input.shape))
                        try:
                            output = self.model(model_input, temporal_coords=None, location_coords=None)
                        except TypeError:
                            output = self.model(model_input)
                        logits_window = self._extract_logits(output)
                        if hasattr(logits_window, "detach"):
                            logits_window = logits_window.detach().cpu()
                        logits_window = torch.as_tensor(logits_window, dtype=torch.float32)
                        if logits_window.ndim == 5:
                            logits_window = logits_window.mean(dim=2)
                        if logits_window.ndim == 3:
                            logits_window = logits_window.unsqueeze(0)
                        if logits_window.ndim != 4 or logits_window.shape[1] < 2:
                            raise PrithviConfigurationError(
                                "Official Prithvi raw model output must be [B,K,H,W] with K>=2; "
                                f"got {tuple(logits_window.shape)}."
                            )
                        if not bool(torch.isfinite(logits_window).all()):
                            raise PrithviConfigurationError("Official Prithvi raw logits contain NaN/Inf.")
                        if logits_window.shape[-2:] != (window_size, window_size):
                            logits_window = F.interpolate(logits_window, size=(window_size, window_size), mode="bilinear", align_corners=False)
                        row_logits.append(logits_window)
                    sample_logits.append(torch.cat(row_logits, dim=-1))
                sample_full = torch.cat(sample_logits, dim=-2)[..., :original_h, :original_w]
                batch_logits.append(sample_full)
        logits = torch.cat(batch_logits, dim=0).cpu().numpy().astype(np.float32)
        self.debug_records.append(
            {
                "stage": "raw_model_output",
                "input_array_shape_B_T_C_H_W": list(np.asarray(images).shape),
                "scaled_input_min": float(np.nanmin(array)),
                "scaled_input_max": float(np.nanmax(array)),
                "scaled_input_mean": float(np.nanmean(array)),
                "padded_tensor_shape_B_C_T_H_W": list(tensor.shape),
                "padding_contract": padding_contract,
                "model_window_input_shapes": model_inputs_seen[:5],
                "official_inference_window_size": window_size,
                "used_datamodule_transform": bool(self.datamodule is not None and hasattr(self.datamodule, "test_transform") and hasattr(self.datamodule, "aug")),
                "input_min": float(np.nanmin(array)),
                "input_max": float(np.nanmax(array)),
                "input_mean": float(np.nanmean(array)),
                "output_summary": self._summarize_array(logits),
            }
        )
        if logits.ndim != 4:
            raise PrithviConfigurationError(f"Expected segmentation logits [B,C,H,W], got {logits.shape}.")
        return logits

    def _prepare_official_window(self, window: Any) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise PrithviConfigurationError("PyTorch is required for official Prithvi Sen1Floods11 TL inference.") from exc
        if self.datamodule is None or not (hasattr(self.datamodule, "test_transform") and hasattr(self.datamodule, "aug")):
            return window.unsqueeze(0)
        # Official inference does: x.squeeze().numpy().transpose(1,2,0),
        # datamodule.test_transform(...), datamodule.aug(...). That drops the
        # singleton time axis; restore it before calling the TerraTorch model.
        image = window.squeeze().cpu().numpy().transpose(1, 2, 0)
        transformed = self.datamodule.test_transform(image=image)
        augmented = self.datamodule.aug(transformed)
        model_input = augmented["image"] if isinstance(augmented, Mapping) else augmented
        if not hasattr(model_input, "detach"):
            model_input = torch.as_tensor(model_input, dtype=torch.float32)
        model_input = model_input.float()
        if model_input.ndim == 3:
            model_input = model_input.unsqueeze(0).unsqueeze(2)
        elif model_input.ndim == 4:
            model_input = model_input.unsqueeze(2)
        if model_input.ndim != 5:
            raise PrithviConfigurationError(f"Expected transformed TL input [B,C,T,H,W], got {tuple(model_input.shape)}.")
        return model_input

    def _extract_logits(self, output: Any) -> Any:
        if hasattr(output, "output"):
            return output.output
        if isinstance(output, Mapping):
            for key in ["output", "logits", "prediction", "pred"]:
                if key in output:
                    return output[key]
            return next(iter(output.values()))
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _summarize_output(self, output: Any) -> Any:
        if hasattr(output, "detach"):
            return self._summarize_array(output)
        if hasattr(output, "output"):
            return {"type": f"{output.__class__.__module__}.{output.__class__.__name__}", "output": self._summarize_array(output.output)}
        if isinstance(output, Mapping):
            return {
                "type": "dict",
                "keys": list(output.keys()),
                "values": {str(key): self._summarize_output(value) for key, value in output.items()},
            }
        if isinstance(output, (tuple, list)):
            return {
                "type": type(output).__name__,
                "length": len(output),
                "values": [self._summarize_output(value) for value in output],
            }
        return {"type": f"{output.__class__.__module__}.{output.__class__.__name__}"}

    def _summarize_array(self, value: Any) -> dict[str, Any]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        summary: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if array.size:
            summary.update(
                {
                    "min": float(np.nanmin(array)),
                    "max": float(np.nanmax(array)),
                    "mean": float(np.nanmean(array)),
                }
            )
            channel_axis = 1 if array.ndim >= 4 else 0 if array.ndim == 3 and array.shape[0] <= 16 else None
            if channel_axis is not None:
                summary["per_channel"] = [
                    {
                        "channel": index,
                        "min": float(np.nanmin(np.take(array, index, axis=channel_axis))),
                        "max": float(np.nanmax(np.take(array, index, axis=channel_axis))),
                        "mean": float(np.nanmean(np.take(array, index, axis=channel_axis))),
                    }
                    for index in range(min(array.shape[channel_axis], 16))
                ]
                if array.shape[channel_axis] > 16:
                    summary["per_channel_truncated"] = int(array.shape[channel_axis])
        return summary

    def _record_prediction_debug(
        self,
        images: np.ndarray,
        logits: np.ndarray,
        probabilities: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        values, counts = np.unique(predictions, return_counts=True)
        self.debug_records.append(
            {
                "stage": "decoded_prediction",
                "input_array_shape_B_T_C_H_W": list(np.asarray(images).shape),
                "logits": self._summarize_array(logits),
                "probabilities": self._summarize_array(probabilities),
                "background_prob": self._summarize_array(probabilities[:, 0, :, :]),
                "water_prob": self._summarize_array(probabilities[:, 1, :, :]) if probabilities.shape[1] > 1 else {},
                "argmax_class_distribution": {str(int(value)): int(count) for value, count in zip(values, counts)},
                "predicted_positive_ratio": float(np.mean(predictions == 1)),
                "positive_class_definition": "prediction == 1",
                "softmax_axis": 1,
            }
        )

    def _resize_logits(self, logits: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if tuple(logits.shape[-2:]) == target_shape:
            return logits
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise PrithviConfigurationError("PyTorch is required to resize Prithvi TL logits.") from exc
        tensor = torch.as_tensor(logits, dtype=torch.float32)
        resized = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
        return resized.cpu().numpy().astype(np.float32)

    def _softmax(self, logits: np.ndarray, axis: int) -> np.ndarray:
        shifted = logits - np.nanmax(logits, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-8)

    def extract_embeddings(self, batch: Mapping[str, Any]) -> np.ndarray:
        prediction = self.predict_segmentation(batch)
        return prediction["score_maps"].mean(axis=(1, 2), keepdims=False).reshape(-1, 1)

    def segmentation_features(self, batch: Mapping[str, Any]) -> np.ndarray:
        prediction = self.predict_segmentation(batch)
        return prediction["score_maps"][:, None, :, :]

    def get_supported_modalities(self) -> Sequence[str]:
        return ("S2", "S2_6B_SEN1FLOODS11")
