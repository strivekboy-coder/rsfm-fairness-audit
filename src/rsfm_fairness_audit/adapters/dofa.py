from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import ModelAdapter
from rsfm_fairness_audit.config import load_yaml


class DOFAConfigurationError(RuntimeError):
    """Raised when DOFA cannot be loaded without guessing reproduction details."""


class DOFAAdapter(ModelAdapter):
    """First DOFA adapter with explicit, no-surprise loading behavior."""

    verified_wavelengths = {
        "S1": [5.405, 5.405],
        "S2_OFFICIAL_DEMO_9CH": [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19],
        "NAIP_RGB": [0.665, 0.56, 0.49],
    }
    official_means = {
        "S1": [166.36275909, 88.45542715],
        "S2": [114.1099739, 114.81779093, 126.63977424, 84.33539309, 97.84789168, 103.94461911, 101.435633, 72.32804172, 56.66528851],
        "NAIP_RGB": [123.675, 116.28, 103.53],
    }
    official_stds = {
        "S1": [64.83126309, 43.07350145],
        "S2": [77.84352553, 69.96844919, 67.42465279, 64.57022983, 61.72545487, 61.34187099, 60.29744676, 47.88519516, 42.55886798],
        "NAIP_RGB": [58.395, 57.12, 57.375],
    }

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        torch_hub_repo: str = "zhu-xlab/DOFA",
        model_variant: str = "vit_base_dofa",
        device: str = "cpu",
        batch_size: int = 32,
        expected_bands: int | None = None,
        image_size: int | None = None,
        embedding_layer: str = "forward_features",
        normalization_mean: Sequence[float] | None = None,
        normalization_std: Sequence[float] | None = None,
        sensor_mode: str = "S2",
        wavelengths: Sequence[float] | None = None,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
        allow_torch_hub_download: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.repo_path = Path(repo_path) if repo_path else None
        self.torch_hub_repo = torch_hub_repo
        self.model_variant = model_variant
        self.device = device
        self.batch_size = batch_size
        self.expected_bands = expected_bands
        self.image_size = image_size
        self.embedding_layer = embedding_layer
        self.normalization_mean = list(normalization_mean) if normalization_mean is not None else None
        self.normalization_std = list(normalization_std) if normalization_std is not None else None
        self.sensor_mode = sensor_mode.upper()
        self.wavelengths = list(wavelengths) if wavelengths is not None else None
        self.model = model
        self.model_loader = model_loader
        self.allow_torch_hub_download = allow_torch_hub_download
        self._torch_device: Any | None = None

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> "DOFAAdapter":
        data = load_yaml(path)
        return cls.from_config(data, model=model, model_loader=model_loader)

    @classmethod
    def from_config(
        cls,
        data: Mapping[str, Any],
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> "DOFAAdapter":
        return cls(
            checkpoint_path=data.get("checkpoint_path"),
            repo_path=data.get("repo_path"),
            torch_hub_repo=str(data.get("torch_hub_repo", "zhu-xlab/DOFA")),
            model_variant=str(data.get("model_variant", "vit_base_dofa")),
            device=str(data.get("device", "cpu")),
            batch_size=int(data.get("batch_size", 32)),
            expected_bands=data.get("expected_bands"),
            image_size=data.get("image_size"),
            embedding_layer=str(data.get("embedding_layer", "forward_features")),
            normalization_mean=data.get("normalization_mean"),
            normalization_std=data.get("normalization_std"),
            sensor_mode=str(data.get("input_modality", data.get("sensor_mode", "S2"))),
            wavelengths=data.get("wavelength_list"),
            model=model,
            model_loader=model_loader,
            allow_torch_hub_download=bool(data.get("allow_torch_hub_download", False)),
        )

    def load_model(self) -> None:
        if self.model is not None:
            self._maybe_eval()
            return
        if self.model_loader is not None:
            self.model = self.model_loader()
            self._maybe_eval()
            return
        self._validate_real_loading_config()
        if self.repo_path is not None:
            self.model = self._load_from_local_repo()
        elif self.allow_torch_hub_download:
            self.model = self._load_from_torch_hub()
        else:
            raise DOFAConfigurationError(
                "DOFA is not configured. To keep this smoke path reproducible and checkpoint-free, "
                "provide an injected model/model_loader, set repo_path plus checkpoint_path in configs/models/dofa.yaml, "
                "or explicitly enable the official torch.hub loader."
            )
        self._maybe_eval()

    def _validate_real_loading_config(self) -> None:
        if self.repo_path is not None and not self.repo_path.exists():
            raise DOFAConfigurationError(f"Configured DOFA repo_path does not exist: {self.repo_path}")
        if self.checkpoint_path is not None and self.repo_path is None and not self.allow_torch_hub_download:
            raise DOFAConfigurationError(
                "Configured DOFA checkpoint_path requires repo_path so the official model implementation can be imported. "
                "Set repo_path to a local clone of https://github.com/zhu-xlab/DOFA, or use allow_torch_hub_download=True."
            )
        if self.repo_path is not None and self.checkpoint_path is None:
            raise DOFAConfigurationError(
                "Configured DOFA repo_path requires checkpoint_path. The official base checkpoint is "
                "DOFA_ViT_base_e100.pth from https://huggingface.co/earthflow/DOFA, but this adapter will not download it automatically."
            )
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise DOFAConfigurationError(f"Configured DOFA checkpoint_path does not exist: {self.checkpoint_path}")
        if self.embedding_layer not in {"forward_features", "forward"}:
            raise DOFAConfigurationError("embedding_layer must be 'forward_features' or 'forward'.")

    def _load_from_torch_hub(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise DOFAConfigurationError("PyTorch is required for the official DOFA torch.hub loading path.") from exc
        model = torch.hub.load(self.torch_hub_repo, self.model_variant, pretrained=True)
        return self._move_to_device(model)

    def _load_from_local_repo(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise DOFAConfigurationError("PyTorch is required to load DOFA from a local official repository.") from exc
        if self.model_variant != "vit_base_dofa":
            raise DOFAConfigurationError(
                f"Local repo loading currently supports model_variant='vit_base_dofa', got {self.model_variant!r}. "
                "Add and verify the official constructor before using another variant."
            )
        assert self.repo_path is not None
        assert self.checkpoint_path is not None
        sys.path.insert(0, str(self.repo_path))
        try:
            from dofa_v1 import vit_base_patch16
        except ImportError as exc:
            raise DOFAConfigurationError(
                f"Could not import dofa_v1.vit_base_patch16 from repo_path={self.repo_path}. "
                "Clone the official https://github.com/zhu-xlab/DOFA repository or use torch_hub loading."
            ) from exc
        finally:
            try:
                sys.path.remove(str(self.repo_path))
            except ValueError:
                pass
        model = vit_base_patch16()
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        return self._move_to_device(model)

    def _resolve_device(self) -> Any:
        if self._torch_device is not None:
            return self._torch_device
        try:
            import torch
        except ImportError as exc:
            raise DOFAConfigurationError("PyTorch is required for real DOFA inference.") from exc
        if self.device == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            name = self.device
        if name == "cuda" and not torch.cuda.is_available():
            raise DOFAConfigurationError("DOFA device='cuda' was requested, but CUDA is not available.")
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
        samples = list(batch["samples"])
        images = [sample["image"] for sample in samples]
        if any(isinstance(image, dict) for image in images):
            raise DOFAConfigurationError(
                "DOFAAdapter received S1+S2 dictionary samples. Concatenated multimodal DOFA input needs a verified "
                "band order and wavelength list before implementation. Use sensor_mode S1 or S2 for this smoke run."
            )
        array = np.stack(images).astype(np.float32)
        if array.ndim != 4:
            raise DOFAConfigurationError(f"Expected image tensor shape [batch, bands, height, width], got {array.shape}.")
        if self.expected_bands is not None and array.shape[1] != int(self.expected_bands):
            raise DOFAConfigurationError(
                f"Configured expected_bands={self.expected_bands} but input has {array.shape[1]} channels."
            )
        wavelengths = self._resolve_wavelengths(array.shape[1])
        array = self._normalize(array)
        array = self._resize_if_needed(array)
        return {"images": array, "metadata": batch["metadata"], "wavelengths": wavelengths}

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalization_mean is None and self.normalization_std is None:
            mean = self.official_means.get(self.sensor_mode)
            std = self.official_stds.get(self.sensor_mode)
        else:
            mean = self.normalization_mean
            std = self.normalization_std
        if mean is None or std is None:
            return array
        if len(mean) != array.shape[1] or len(std) != array.shape[1]:
            raise DOFAConfigurationError(
                f"Normalization mean/std lengths must match input channels ({array.shape[1]})."
            )
        mean_arr = np.asarray(mean, dtype=np.float32)[None, :, None, None]
        std_arr = np.asarray(std, dtype=np.float32)[None, :, None, None]
        return (array - mean_arr) / np.maximum(std_arr, 1e-8)

    def _resize_if_needed(self, array: np.ndarray) -> np.ndarray:
        if self.image_size is None:
            return array
        if array.shape[-2:] == (int(self.image_size), int(self.image_size)):
            return array
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise DOFAConfigurationError("PyTorch is required to resize DOFA inputs to image_size.") from exc
        tensor = torch.as_tensor(array, dtype=torch.float32)
        resized = F.interpolate(tensor, size=(int(self.image_size), int(self.image_size)), mode="bilinear", align_corners=False)
        return resized.cpu().numpy()

    def _resolve_wavelengths(self, channels: int) -> list[float]:
        if self.wavelengths is not None:
            if len(self.wavelengths) != channels:
                raise DOFAConfigurationError(
                    f"Configured DOFA wavelength count ({len(self.wavelengths)}) does not match input channels ({channels})."
                )
            return self.wavelengths
        if self.sensor_mode == "S1" and channels == 2:
            return self.verified_wavelengths["S1"]
        if self.sensor_mode == "S2" and channels == 9:
            return self.verified_wavelengths["S2_OFFICIAL_DEMO_9CH"]
        raise DOFAConfigurationError(
            f"DOFA preprocessing for sensor_mode={self.sensor_mode} with {channels} channels is still to_verify. "
            "Provide --dofa-wavelengths with one official wavelength per channel, or preconvert the subset to the "
            "official 9-channel Sentinel-2 demo order documented in docs/reproduction/dofa.md."
        )

    def extract_embeddings(self, batch: Mapping[str, Any]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("load_model() must be called before extract_embeddings().")
        images = batch["images"]
        wavelengths = batch["wavelengths"]

        if hasattr(self.model, "extract_embeddings"):
            embeddings = self.model.extract_embeddings(images, wavelengths)
            return np.asarray(embeddings, dtype=np.float32)

        try:
            import torch
        except ImportError as exc:
            raise DOFAConfigurationError("PyTorch is required to run a real DOFA model.") from exc

        tensor = torch.as_tensor(images, dtype=torch.float32)
        tensor = tensor.to(self._resolve_device())
        with torch.no_grad():
            if self.embedding_layer == "forward_features" and hasattr(self.model, "forward_features"):
                output = self.model.forward_features(tensor, wave_list=wavelengths)
            else:
                output = self.model(tensor, wave_list=wavelengths)
        if isinstance(output, dict):
            output = next(iter(output.values()))
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        embeddings = np.asarray(output, dtype=np.float32)
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings

    def get_supported_modalities(self) -> Sequence[str]:
        return ("S1", "S2")
