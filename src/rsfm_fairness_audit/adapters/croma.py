from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import ModelAdapter
from rsfm_fairness_audit.config import load_yaml


class CROMAConfigurationError(RuntimeError):
    """Raised when CROMA cannot be loaded without verified reproduction details."""


class CROMAAdapter(ModelAdapter):
    """CROMA optical-only adapter for Phase 2A BigEarthNet S2 smoke runs."""

    official_hf_repo = "antofuller/CROMA"
    official_checkpoint_filenames = {"CROMA_base.pt", "CROMA_large.pt"}
    supported_modalities = ("optical", "SAR", "both")

    def __init__(
        self,
        model_size: str = "base",
        checkpoint_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        source_file_path: str | Path | None = None,
        hf_repo_id: str = official_hf_repo,
        hf_checkpoint_filename: str = "CROMA_base.pt",
        allow_hf_download: bool = False,
        device: str = "auto",
        batch_size: int = 4,
        input_modality: str = "optical",
        expected_bands: int = 12,
        expected_s1_bands: int = 2,
        expected_s2_bands: int = 12,
        image_size: int = 120,
        embedding_key: str = "optical_GAP",
        preprocessing: str = "official_croma_channel_scale",
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.model_size = model_size
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.repo_path = Path(repo_path) if repo_path else None
        self.source_file_path = Path(source_file_path) if source_file_path else None
        self.hf_repo_id = hf_repo_id
        self.hf_checkpoint_filename = hf_checkpoint_filename
        self.allow_hf_download = allow_hf_download
        self.device = device
        self.batch_size = batch_size
        self.input_modality = input_modality
        self.expected_bands = int(expected_bands)
        self.expected_s1_bands = int(expected_s1_bands)
        self.expected_s2_bands = int(expected_s2_bands)
        self.image_size = int(image_size)
        self.embedding_key = embedding_key
        self.preprocessing = preprocessing
        self.model = model
        self.model_loader = model_loader
        self._torch_device: Any | None = None
        self._logged_profile = False
        self._logged_device_state = False

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> "CROMAAdapter":
        return cls.from_config(load_yaml(path), model=model, model_loader=model_loader)

    @classmethod
    def from_config(
        cls,
        data: Mapping[str, Any],
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> "CROMAAdapter":
        return cls(
            model_size=str(data.get("model_size", "base")),
            checkpoint_path=data.get("checkpoint_path"),
            repo_path=data.get("repo_path"),
            source_file_path=data.get("source_file_path"),
            hf_repo_id=str(data.get("hf_repo_id", cls.official_hf_repo)),
            hf_checkpoint_filename=str(data.get("hf_checkpoint_filename", "CROMA_base.pt")),
            allow_hf_download=bool(data.get("allow_hf_download", False)),
            device=str(data.get("device", "auto")),
            batch_size=int(data.get("batch_size", 4)),
            input_modality=str(data.get("input_modality", "optical")),
            expected_bands=int(data.get("expected_bands", 12)),
            expected_s1_bands=int(data.get("expected_s1_bands", 2)),
            expected_s2_bands=int(data.get("expected_s2_bands", data.get("expected_bands", 12))),
            image_size=int(data.get("image_size", 120)),
            embedding_key=str(data.get("embedding_key", "optical_GAP")),
            preprocessing=str(data.get("preprocessing", "official_croma_channel_scale")),
            model=model,
            model_loader=model_loader,
        )

    def load_model(self) -> None:
        self._validate_config()
        if self.model is not None:
            self.model = self._move_to_device(self.model)
            self._maybe_eval()
            return
        if self.model_loader is not None:
            self.model = self.model_loader()
            self.model = self._move_to_device(self.model)
            self._maybe_eval()
            return
        checkpoint_path = self._resolve_checkpoint_path()
        source_file = self._resolve_source_file()
        pretrained_cls = self._load_pretrained_croma_class(source_file)
        self.model = pretrained_cls(
            pretrained_path=str(checkpoint_path),
            size=self.model_size,
            modality=self.input_modality,
            image_resolution=self.image_size,
        )
        self.model = self._move_to_device(self.model)
        self._maybe_eval()

    def _validate_config(self) -> None:
        modality = self.input_modality
        if modality not in self.supported_modalities:
            raise CROMAConfigurationError(
                f"CROMA input_modality must be one of {self.supported_modalities}, got {modality!r}."
            )
        if self.model_size not in {"base", "large"}:
            raise CROMAConfigurationError("CROMA model_size must be 'base' or 'large'.")
        if modality == "optical" and self.expected_s2_bands != 12:
            raise CROMAConfigurationError("CROMA optical mode expects 12 Sentinel-2 channels.")
        if modality == "SAR" and self.expected_s1_bands != 2:
            raise CROMAConfigurationError("CROMA SAR mode expects 2 Sentinel-1 channels.")
        if modality == "both" and (self.expected_s1_bands != 2 or self.expected_s2_bands != 12):
            raise CROMAConfigurationError("CROMA both mode expects 2 Sentinel-1 channels and 12 Sentinel-2 channels.")
        if self.image_size <= 0 or self.image_size % 8 != 0:
            raise CROMAConfigurationError("CROMA image_size must be positive and divisible by official patch size 8.")
        if self.preprocessing != "official_croma_channel_scale":
            raise CROMAConfigurationError("Only preprocessing='official_croma_channel_scale' is verified for CROMA Phase 2A.")
        self._validate_hf_settings()

    def _validate_hf_settings(self) -> None:
        if self.hf_repo_id != self.official_hf_repo:
            raise CROMAConfigurationError(
                f"CROMA HF downloads are restricted to the official repo {self.official_hf_repo!r}; "
                f"got {self.hf_repo_id!r}."
            )
        if self.hf_checkpoint_filename not in self.official_checkpoint_filenames:
            raise CROMAConfigurationError(
                "CROMA HF checkpoint filename must be one of "
                f"{sorted(self.official_checkpoint_filenames)}, got {self.hf_checkpoint_filename!r}."
            )

    def _resolve_checkpoint_path(self) -> Path:
        if self.checkpoint_path is not None:
            if not self.checkpoint_path.exists():
                raise CROMAConfigurationError(f"Configured CROMA checkpoint_path does not exist: {self.checkpoint_path}")
            return self.checkpoint_path
        if not self.allow_hf_download:
            raise CROMAConfigurationError(
                "CROMA checkpoint_path is not configured. Set checkpoint_path to an official local checkpoint, "
                "or set allow_hf_download: true to download only an official antofuller/CROMA checkpoint."
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise CROMAConfigurationError("huggingface_hub is required for CROMA allow_hf_download mode.") from exc
        path = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=self.hf_checkpoint_filename,
            repo_type="model",
        )
        return Path(path)

    def _resolve_source_file(self) -> Path:
        if self.source_file_path is not None:
            source_file = self.source_file_path
        elif self.repo_path is not None:
            source_file = self.repo_path / "use_croma.py"
        else:
            raise CROMAConfigurationError(
                "CROMA official implementation is not configured. Set source_file_path to the official use_croma.py "
                "or repo_path to a local clone of https://github.com/antofuller/CROMA. "
                "This adapter will not guess or silently vendor model code."
            )
        if not source_file.exists():
            raise CROMAConfigurationError(f"Configured CROMA source file does not exist: {source_file}")
        return source_file

    def _load_pretrained_croma_class(self, source_file: Path) -> Any:
        spec = importlib.util.spec_from_file_location("official_croma_use_croma", source_file)
        if spec is None or spec.loader is None:
            raise CROMAConfigurationError(f"Could not import CROMA source file: {source_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "PretrainedCROMA"):
            raise CROMAConfigurationError(
                f"{source_file} does not expose PretrainedCROMA. Use the official CROMA use_croma.py file."
            )
        return module.PretrainedCROMA

    def _resolve_device(self) -> Any:
        if self._torch_device is not None:
            return self._torch_device
        try:
            import torch
        except ImportError as exc:
            raise CROMAConfigurationError("PyTorch is required for CROMA inference.") from exc
        if self.device == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            name = self.device
        if name == "cuda" and not torch.cuda.is_available():
            raise CROMAConfigurationError("CROMA device='cuda' was requested, but CUDA is not available.")
        self._torch_device = torch.device(name)
        return self._torch_device

    def _move_to_device(self, model: Any) -> Any:
        if hasattr(model, "to"):
            model = model.to(self._resolve_device())
        return model

    def _maybe_eval(self) -> None:
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _model_parameter_device(self) -> str:
        if self.model is None or not hasattr(self.model, "parameters"):
            return "unavailable"
        try:
            first_parameter = next(self.model.parameters())
        except StopIteration:
            return "no_parameters"
        except TypeError:
            return "unavailable"
        return str(getattr(first_parameter, "device", "unknown"))

    def _gpu_name(self) -> str:
        try:
            import torch
        except ImportError:
            return "torch_unavailable"
        if not torch.cuda.is_available():
            return "cuda_unavailable"
        try:
            return str(torch.cuda.get_device_name(self._resolve_device()))
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            return f"cuda_name_unavailable:{type(exc).__name__}"

    def _log_device_state(self, kwargs: Mapping[str, Any]) -> None:
        if self._logged_device_state:
            return
        input_devices = {
            key: str(getattr(value, "device", "unknown"))
            for key, value in kwargs.items()
            if key.endswith("_images")
        }
        print(
            "[info] CROMA device: "
            f"requested={self.device} resolved={self._resolve_device()} "
            f"gpu={self._gpu_name()} model_parameter_device={self._model_parameter_device()} "
            f"input_tensor_devices={input_devices}"
        )
        self._logged_device_state = True

    def preprocess(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        self._validate_config()
        samples = list(batch["samples"])
        images = [sample["image"] for sample in samples]
        if self.input_modality == "optical":
            optical = [image["S2"] if isinstance(image, dict) else image for image in images]
            array = self._prepare_array(optical, self.expected_s2_bands, "optical/S2")
            return {"optical_images": array, "metadata": batch["metadata"]}
        if self.input_modality == "SAR":
            sar = [image["S1"] if isinstance(image, dict) else image for image in images]
            array = self._prepare_array(sar, self.expected_s1_bands, "SAR/S1")
            return {"SAR_images": array, "metadata": batch["metadata"]}
        if not all(isinstance(image, dict) for image in images):
            raise CROMAConfigurationError("CROMA both mode requires paired dict samples with S1 and S2 arrays.")
        sar = [image["S1"] for image in images]
        optical = [image["S2"] for image in images]
        return {
            "SAR_images": self._prepare_array(sar, self.expected_s1_bands, "SAR/S1"),
            "optical_images": self._prepare_array(optical, self.expected_s2_bands, "optical/S2"),
            "metadata": batch["metadata"],
        }

    def _prepare_array(self, images: Sequence[Any], expected_channels: int, name: str) -> np.ndarray:
        array = np.stack(images).astype(np.float32)
        if array.ndim != 4:
            raise CROMAConfigurationError(f"Expected CROMA {name} input shape [batch, bands, height, width], got {array.shape}.")
        if array.shape[1] != expected_channels:
            raise CROMAConfigurationError(
                f"Configured {name} expected channels={expected_channels} but input has {array.shape[1]} channels."
            )
        self._log_profile(array.shape[1])
        array = self._resize_if_needed(array)
        return self._official_channel_scale(array)

    def _log_profile(self, channels: int) -> None:
        if self._logged_profile:
            return
        print(
            "[info] CROMA profile: "
            f"modality={self.input_modality} expected_s1_bands={self.expected_s1_bands} expected_s2_bands={self.expected_s2_bands} "
            f"input_channels={channels} image_size={self.image_size}"
        )
        self._logged_profile = True

    def _resize_if_needed(self, array: np.ndarray) -> np.ndarray:
        if array.shape[-2:] == (self.image_size, self.image_size):
            return array
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise CROMAConfigurationError("PyTorch is required to resize CROMA inputs.") from exc
        tensor = torch.as_tensor(array, dtype=torch.float32)
        resized = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return resized.cpu().numpy()

    def _official_channel_scale(self, array: np.ndarray) -> np.ndarray:
        mean = array.mean(axis=(2, 3), keepdims=True)
        std = array.std(axis=(2, 3), keepdims=True)
        lower = mean - 2.0 * std
        upper = mean + 2.0 * std
        clipped = np.clip(array, lower, upper)
        denom = np.maximum(upper - lower, 1e-6)
        return ((clipped - lower) / denom).astype(np.float32)

    def extract_embeddings(self, batch: Mapping[str, Any]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("load_model() must be called before extract_embeddings().")
        if hasattr(self.model, "extract_embeddings"):
            embeddings = self._extract_mock_embeddings(batch)
            return np.asarray(embeddings, dtype=np.float32)
        try:
            import torch
        except ImportError as exc:
            raise CROMAConfigurationError("PyTorch is required to run a real CROMA model.") from exc
        kwargs: dict[str, Any] = {}
        if "SAR_images" in batch:
            kwargs["SAR_images"] = torch.as_tensor(batch["SAR_images"], dtype=torch.float32).to(self._resolve_device())
        if "optical_images" in batch:
            kwargs["optical_images"] = torch.as_tensor(batch["optical_images"], dtype=torch.float32).to(self._resolve_device())
        self._log_device_state(kwargs)
        with torch.inference_mode():
            outputs = self.model(**kwargs)
        if not isinstance(outputs, Mapping):
            raise CROMAConfigurationError("Official CROMA model output is expected to be a mapping of embedding tensors.")
        if self.embedding_key not in outputs:
            raise CROMAConfigurationError(
                f"CROMA output does not contain embedding_key={self.embedding_key!r}. "
                f"Available keys: {sorted(outputs.keys())}"
            )
        embeddings = outputs[self.embedding_key]
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings

    def _extract_mock_embeddings(self, batch: Mapping[str, Any]) -> Any:
        if self.input_modality == "SAR":
            return self.model.extract_embeddings(batch["SAR_images"])
        if self.input_modality == "optical":
            return self.model.extract_embeddings(batch["optical_images"])
        try:
            return self.model.extract_embeddings(batch["SAR_images"], batch["optical_images"])
        except TypeError:
            return self.model.extract_embeddings({"SAR_images": batch["SAR_images"], "optical_images": batch["optical_images"]})

    def get_supported_modalities(self) -> Sequence[str]:
        return self.supported_modalities
