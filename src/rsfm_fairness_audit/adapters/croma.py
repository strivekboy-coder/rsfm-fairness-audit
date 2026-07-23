from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import ModelAdapter
from rsfm_fairness_audit.config import load_yaml


class CROMAConfigurationError(RuntimeError):
    """Raised when CROMA cannot be loaded without verified reproduction details."""


_CHECKPOINT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _checkpoint_sha256(path: str | Path) -> str:
    checkpoint = Path(path).resolve()
    stat = checkpoint.stat()
    cache_key = (str(checkpoint), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _CHECKPOINT_HASH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    _CHECKPOINT_HASH_CACHE[cache_key] = observed
    return observed


class CROMAAdapter(ModelAdapter):
    """CROMA adapter for pinned S1, S2, and joint frozen-embedding audits."""

    official_hf_repo = "antofuller/CROMA"
    official_checkpoint_filenames = {"CROMA_base.pt", "CROMA_large.pt"}
    supported_modalities = ("optical", "SAR", "both")
    official_repo_revision = "59505a6bcadbf36ba20767270154bf9f3067c5e7"
    official_hf_revision = "0dd28e3d633bd6715856ae9890e8c49360040598"
    official_base_sha256 = "0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63"

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
        preprocessing: str = "per_sample_channel_2sigma_0_1",
        normalization_stats: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        normalization_stats_source: str | Path | None = None,
        repo_revision: str = official_repo_revision,
        checkpoint_expected_sha256: str = official_base_sha256,
        strict_reproducibility: bool = False,
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
        self.normalization_stats = {
            str(sensor): {
                "mean": tuple(float(value) for value in values["mean"]),
                "std": tuple(float(value) for value in values["std"]),
            }
            for sensor, values in (normalization_stats or {}).items()
        }
        self.normalization_stats_source = (
            Path(normalization_stats_source) if normalization_stats_source is not None else None
        )
        self.repo_revision = str(repo_revision)
        self.checkpoint_expected_sha256 = str(checkpoint_expected_sha256).lower()
        self.strict_reproducibility = bool(strict_reproducibility)
        self.actual_repo_revision: str | None = None
        self.actual_checkpoint_sha256: str | None = None
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
            preprocessing=str(data.get("preprocessing", "per_sample_channel_2sigma_0_1")),
            normalization_stats=data.get("normalization_stats"),
            normalization_stats_source=data.get("normalization_stats_source"),
            repo_revision=str(data.get("repo_revision", cls.official_repo_revision)),
            checkpoint_expected_sha256=str(
                data.get("checkpoint_expected_sha256", cls.official_base_sha256)
            ),
            strict_reproducibility=bool(data.get("strict_reproducibility", False)),
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
        if self.preprocessing not in {
            "per_sample_channel_2sigma_0_1",
            "train_split_fixed_channel_2sigma_0_1",
            "train_split_fixed_channel_2sigma_uint8_0_1",
        }:
            raise CROMAConfigurationError("Unsupported CROMA preprocessing policy.")
        required_sensors = {"S1" if self.input_modality in {"SAR", "both"} else ""}
        if self.input_modality in {"optical", "both"}:
            required_sensors.add("S2")
        required_sensors.discard("")
        if self.preprocessing.startswith("train_split_fixed_channel_2sigma"):
            expected = {"S1": self.expected_s1_bands, "S2": self.expected_s2_bands}
            for sensor in required_sensors:
                stats = self.normalization_stats.get(sensor, {})
                if len(stats.get("mean", ())) != expected[sensor] or len(stats.get("std", ())) != expected[sensor]:
                    raise CROMAConfigurationError(
                        f"Frozen CROMA preprocessing requires train-only {sensor} mean/std for {expected[sensor]} channels."
                    )
                if any(value <= 0.0 for value in stats["std"]):
                    raise CROMAConfigurationError(f"Frozen CROMA {sensor} standard deviations must be positive.")
        if self.strict_reproducibility and (
            self.model_size != "base"
            or self.hf_checkpoint_filename != "CROMA_base.pt"
            or self.repo_revision != self.official_repo_revision
            or self.checkpoint_expected_sha256 != self.official_base_sha256
            or self.preprocessing != "train_split_fixed_channel_2sigma_uint8_0_1"
        ):
            raise CROMAConfigurationError(
                "The frozen formal reBEN panel uses the official CROMA base checkpoint and pinned repository revision."
            )
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
            checkpoint = self.checkpoint_path
        else:
            if not self.allow_hf_download:
                raise CROMAConfigurationError(
                    "CROMA checkpoint_path is not configured. Set checkpoint_path to an official local checkpoint, "
                    "or set allow_hf_download: true to download only an official antofuller/CROMA checkpoint."
                )
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise CROMAConfigurationError("huggingface_hub is required for CROMA allow_hf_download mode.") from exc
            checkpoint = Path(
                hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=self.hf_checkpoint_filename,
                    repo_type="model",
                    revision=self.official_hf_revision,
                )
            )
        if self.strict_reproducibility:
            observed = _checkpoint_sha256(checkpoint)
            if observed != self.checkpoint_expected_sha256:
                raise CROMAConfigurationError(
                    "CROMA checkpoint SHA-256 mismatch: "
                    f"expected={self.checkpoint_expected_sha256}, observed={observed}."
                )
            self.actual_checkpoint_sha256 = observed
        return checkpoint

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
        if self.strict_reproducibility:
            if self.repo_path is None or len(self.repo_revision) != 40:
                raise CROMAConfigurationError(
                    "Formal CROMA runs require repo_path and a full 40-character official repo_revision."
                )
            expected_source = (self.repo_path / "use_croma.py").resolve()
            if source_file.resolve() != expected_source:
                raise CROMAConfigurationError(
                    "Formal CROMA runs require the pinned repository's tracked use_croma.py; "
                    f"observed source_file={source_file.resolve()}."
                )
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                check=False,
                capture_output=True,
                text=True,
            )
            observed = result.stdout.strip().lower()
            if result.returncode != 0 or observed != self.repo_revision.lower():
                raise CROMAConfigurationError(
                    f"CROMA repository revision mismatch: expected={self.repo_revision}, observed={observed or 'unavailable'}."
                )
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no", "--", "use_croma.py"],
                cwd=self.repo_path,
                check=False,
                capture_output=True,
                text=True,
            )
            if dirty.returncode != 0 or dirty.stdout.strip():
                raise CROMAConfigurationError(
                    "Formal CROMA runs require an unmodified tracked use_croma.py at the pinned revision."
                )
            self.actual_repo_revision = observed
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
            array = self._prepare_array(optical, self.expected_s2_bands, "optical/S2", "S2")
            return {"optical_images": array, "metadata": batch["metadata"]}
        if self.input_modality == "SAR":
            sar = [image["S1"] if isinstance(image, dict) else image for image in images]
            array = self._prepare_array(sar, self.expected_s1_bands, "SAR/S1", "S1")
            return {"SAR_images": array, "metadata": batch["metadata"]}
        if not all(isinstance(image, dict) for image in images):
            raise CROMAConfigurationError("CROMA both mode requires paired dict samples with S1 and S2 arrays.")
        sar = [image["S1"] for image in images]
        optical = [image["S2"] for image in images]
        return {
            "SAR_images": self._prepare_array(sar, self.expected_s1_bands, "SAR/S1", "S1"),
            "optical_images": self._prepare_array(optical, self.expected_s2_bands, "optical/S2", "S2"),
            "metadata": batch["metadata"],
        }

    def _prepare_array(
        self,
        images: Sequence[Any],
        expected_channels: int,
        name: str,
        sensor: str,
    ) -> np.ndarray:
        array = np.stack(images).astype(np.float32)
        if array.ndim != 4:
            raise CROMAConfigurationError(f"Expected CROMA {name} input shape [batch, bands, height, width], got {array.shape}.")
        if array.shape[1] != expected_channels:
            raise CROMAConfigurationError(
                f"Configured {name} expected channels={expected_channels} but input has {array.shape[1]} channels."
            )
        self._log_profile(array.shape[1])
        array = self._resize_if_needed(array)
        return self._channel_scale(array, sensor)

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

    def _channel_scale(self, array: np.ndarray, sensor: str) -> np.ndarray:
        if self.preprocessing.startswith("train_split_fixed_channel_2sigma"):
            stats = self.normalization_stats[sensor]
            mean = np.asarray(stats["mean"], dtype=np.float32)[None, :, None, None]
            std = np.asarray(stats["std"], dtype=np.float32)[None, :, None, None]
        else:
            # Backward-compatible diagnostic mode. Formal reBEN runs use
            # train-split fixed scaling to avoid test leakage and
            # batch-composition dependence in the public CROMA example code.
            mean = array.mean(axis=(2, 3), keepdims=True)
            std = array.std(axis=(2, 3), keepdims=True)
        lower = mean - 2.0 * std
        upper = mean + 2.0 * std
        clipped = np.clip(array, lower, upper)
        denom = np.maximum(upper - lower, 1e-6)
        scaled = np.clip((clipped - lower) / denom, 0.0, 1.0)
        if self.preprocessing == "train_split_fixed_channel_2sigma_uint8_0_1":
            # Match the released CROMA example's use_8_bit=True path exactly:
            # scale to [0,255], cast (truncate) to uint8, then feed float/255.
            scaled = (scaled * 255.0).astype(np.uint8).astype(np.float32) / np.float32(255.0)
        return scaled.astype(np.float32)

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

    def provenance(self) -> dict[str, Any]:
        return {
            "model": f"CROMA_{self.model_size}",
            "official_repo": "https://github.com/antofuller/CROMA",
            "official_repo_revision": self.repo_revision,
            "actual_repo_revision": self.actual_repo_revision or "injected_or_not_loaded",
            "official_hf_repo": self.hf_repo_id,
            "official_hf_revision": self.official_hf_revision,
            "checkpoint_filename": self.hf_checkpoint_filename,
            "checkpoint_path": str(self.checkpoint_path or ""),
            "checkpoint_expected_sha256": self.checkpoint_expected_sha256,
            "checkpoint_actual_sha256": self.actual_checkpoint_sha256 or "injected_or_not_loaded",
            "model_size": self.model_size,
            "input_modality": self.input_modality,
            "embedding_key": self.embedding_key,
            "image_size": self.image_size,
            "preprocessing": self.preprocessing,
            "normalization_stats": self.normalization_stats,
            "normalization_stats_source": str(self.normalization_stats_source or ""),
            "strict_reproducibility": self.strict_reproducibility,
        }
