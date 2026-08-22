from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import ModelAdapter
from rsfm_fairness_audit.config import load_yaml


class TerraMindConfigurationError(RuntimeError):
    """Raised when TerraMind inputs or runtime provenance are ambiguous."""


TERRAMIND_OFFICIAL_HF_REPO = "ibm-esa-geospatial/TerraMind-1.0-base"
TERRAMIND_OFFICIAL_HF_FILENAME = "TerraMind_v1_base.pt"
TERRAMIND_OFFICIAL_REVISION = "fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a"
TERRAMIND_OFFICIAL_SHA256 = "83c3a0938067c83867a46e564443c2fa38383bf4f966d931b11cb025b847d7ec"
S2_DN_Q999_UPPER_GUARD = 32767.0


_CHECKPOINT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint without loading the multi-GB tensor payload into RAM."""

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


def validate_terramind_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str = TERRAMIND_OFFICIAL_SHA256,
) -> tuple[Path, str]:
    """Require the frozen official TerraMind checkpoint used by formal campaigns."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise TerraMindConfigurationError(f"TerraMind checkpoint does not exist: {checkpoint}")
    expected = str(expected_sha256).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise TerraMindConfigurationError("expected_sha256 must be a 64-character lowercase hexadecimal digest.")
    observed = checkpoint_sha256(checkpoint)
    if observed != expected:
        raise TerraMindConfigurationError(
            "TerraMind checkpoint SHA-256 mismatch. Formal runs are pinned to "
            f"{TERRAMIND_OFFICIAL_HF_REPO}@{TERRAMIND_OFFICIAL_REVISION}/"
            f"{TERRAMIND_OFFICIAL_HF_FILENAME}: expected={expected}, observed={observed}."
        )
    return checkpoint, observed


def validate_terratorch_runtime() -> str:
    """Validate the frozen TerraMind/TerraTorch compatibility window."""

    try:
        version = importlib.metadata.version("terratorch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TerraMindConfigurationError(
            "TerraMind requires the official TerraTorch runtime (terratorch>=1.2.5,<1.3)."
        ) from exc
    numeric = version.split("+", 1)[0].split("-", 1)[0]
    try:
        parts = tuple(int(value) for value in numeric.split(".")[:3])
    except ValueError as exc:
        raise TerraMindConfigurationError(f"Unrecognized TerraTorch version: {version!r}.") from exc
    parts = parts + (0,) * (3 - len(parts))
    if not ((1, 2, 5) <= parts < (1, 3, 0)):
        raise TerraMindConfigurationError(
            f"Formal TerraMind runs require terratorch>=1.2.5,<1.3; observed {version}."
        )
    return version


@dataclass(frozen=True)
class TerraMindInputProfile:
    s2_modality: str
    s2_channels: int
    s2_mean: tuple[float, ...]
    s2_std: tuple[float, ...]


S1_MEAN = (-12.599, -20.293)
S1_STD = (5.195, 5.890)

INPUT_PROFILES: dict[str, TerraMindInputProfile] = {
    "sen1floods11_l1c": TerraMindInputProfile(
        s2_modality="S2L1C",
        s2_channels=13,
        s2_mean=(
            2357.089,
            2137.385,
            2018.788,
            2082.986,
            2295.651,
            2854.537,
            3122.849,
            3040.560,
            3306.481,
            1473.847,
            506.070,
            2472.825,
            1838.929,
        ),
        s2_std=(
            1624.683,
            1675.806,
            1557.708,
            1833.702,
            1823.738,
            1733.977,
            1732.131,
            1679.732,
            1727.260,
            1024.687,
            442.165,
            1331.411,
            1160.419,
        ),
    ),
    "reben_l2a": TerraMindInputProfile(
        s2_modality="S2L2A",
        s2_channels=12,
        s2_mean=(
            1390.458,
            1503.317,
            1718.197,
            1853.910,
            2199.100,
            2779.975,
            2987.011,
            3083.234,
            3132.220,
            3162.988,
            2424.884,
            1857.648,
        ),
        s2_std=(
            2106.761,
            2141.107,
            2038.973,
            2134.138,
            2085.321,
            1889.926,
            1820.257,
            1871.918,
            1753.829,
            1797.379,
            1434.261,
            1334.311,
        ),
    ),
}


class TerraMindAdapter(ModelAdapter):
    """Frozen TerraMind encoder with explicit EO modality and unit contracts.

    The adapter deliberately uses TerraTorch's registered official backbone.  It
    does not silently reinterpret SAR units: formal runs must declare whether S1
    is already in dB or is linear power/amplitude.  This is essential because
    the official S1GRD normalization statistics are in dB.
    """

    def __init__(
        self,
        *,
        sensor_mode: str,
        input_profile: str,
        model_name: str = "terramind_v1_base",
        model_release: str = "terramind_v1",
        device: str = "auto",
        image_size: int = 224,
        merge_method: str = "mean",
        embedding_pooling: str = "mean_tokens",
        layer_index: int = -1,
        s1_unit_policy: str = "already_db",
        strict_range_check: bool = True,
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        checkpoint_expected_sha256: str = TERRAMIND_OFFICIAL_SHA256,
        model_revision: str = TERRAMIND_OFFICIAL_REVISION,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        mode = str(sensor_mode).upper().replace(" ", "")
        if mode in {"FUSION", "S2+S1"}:
            mode = "S1+S2"
        if mode not in {"S1", "S2", "S1+S2"}:
            raise TerraMindConfigurationError("sensor_mode must be S1, S2, or S1+S2.")
        if input_profile not in INPUT_PROFILES:
            raise TerraMindConfigurationError(
                f"Unknown TerraMind input_profile={input_profile!r}; choose one of {sorted(INPUT_PROFILES)}."
            )
        if merge_method != "mean":
            raise TerraMindConfigurationError(
                "The frozen primary protocol fixes merge_method='mean' so S1, S2, and S1+S2 have the same "
                "embedding dimension and a controlled probe comparison."
            )
        if embedding_pooling != "mean_tokens":
            raise TerraMindConfigurationError("The frozen primary protocol uses mean_tokens pooling.")
        if s1_unit_policy not in {"already_db", "linear_power_to_db", "linear_amplitude_to_db"}:
            raise TerraMindConfigurationError(
                "s1_unit_policy must be already_db, linear_power_to_db, or linear_amplitude_to_db."
            )
        if int(image_size) <= 0:
            raise TerraMindConfigurationError("image_size must be positive.")

        self.sensor_mode = mode
        self.input_profile_name = input_profile
        self.input_profile = INPUT_PROFILES[input_profile]
        self.model_name = str(model_name)
        self.model_release = str(model_release)
        self.device = str(device)
        self.image_size = int(image_size)
        self.merge_method = merge_method
        self.embedding_pooling = embedding_pooling
        self.layer_index = int(layer_index)
        self.s1_unit_policy = s1_unit_policy
        self.strict_range_check = bool(strict_range_check)
        self.pretrained = bool(pretrained)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_expected_sha256 = str(checkpoint_expected_sha256).lower()
        self.model_revision = str(model_revision)
        self.actual_checkpoint_sha256: str | None = None
        self.model = model
        self.model_loader = model_loader
        self._torch_device: Any | None = None
        self._runtime_logged = False
        self.preprocessing_report: dict[str, Any] = {}

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        *,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> "TerraMindAdapter":
        return cls.from_config(load_yaml(path), model=model, model_loader=model_loader)

    @classmethod
    def from_config(
        cls,
        data: Mapping[str, Any],
        *,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> "TerraMindAdapter":
        return cls(
            sensor_mode=str(data["sensor_mode"]),
            input_profile=str(data["input_profile"]),
            model_name=str(data.get("model_name", "terramind_v1_base")),
            model_release=str(data.get("model_release", "terramind_v1")),
            device=str(data.get("device", "auto")),
            image_size=int(data.get("image_size", 224)),
            merge_method=str(data.get("merge_method", "mean")),
            embedding_pooling=str(data.get("embedding_pooling", "mean_tokens")),
            layer_index=int(data.get("layer_index", -1)),
            s1_unit_policy=str(data.get("s1_unit_policy", "already_db")),
            strict_range_check=bool(data.get("strict_range_check", True)),
            pretrained=bool(data.get("pretrained", True)),
            checkpoint_path=data.get("checkpoint_path"),
            checkpoint_expected_sha256=str(
                data.get("checkpoint_expected_sha256", TERRAMIND_OFFICIAL_SHA256)
            ),
            model_revision=str(data.get("model_revision", TERRAMIND_OFFICIAL_REVISION)),
            model=model,
            model_loader=model_loader,
        )

    @property
    def modality_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if "S1" in self.sensor_mode:
            names.append("S1GRD")
        if "S2" in self.sensor_mode:
            names.append(self.input_profile.s2_modality)
        return tuple(names)

    def load_model(self) -> None:
        if self.model is None and self.model_loader is not None:
            self.model = self.model_loader()
        if self.model is None:
            validate_terratorch_runtime()
            if not self.pretrained:
                raise TerraMindConfigurationError(
                    "The formal TerraMind adapter requires pretrained=True and the pinned official checkpoint."
                )
            if self.checkpoint_path is None:
                raise TerraMindConfigurationError(
                    "Formal TerraMind inference requires an explicit checkpoint_path. Download the official file "
                    f"at revision {self.model_revision}; mutable online pretrained=True resolution is not accepted."
                )
            if self.model_revision != TERRAMIND_OFFICIAL_REVISION:
                raise TerraMindConfigurationError(
                    f"Formal TerraMind runs are pinned to revision {TERRAMIND_OFFICIAL_REVISION}; "
                    f"observed {self.model_revision}."
                )
            checkpoint, observed_sha256 = validate_terramind_checkpoint(
                self.checkpoint_path,
                expected_sha256=self.checkpoint_expected_sha256,
            )
            self.actual_checkpoint_sha256 = observed_sha256
            try:
                import terratorch.models.backbones  # noqa: F401
                from terratorch.registry import BACKBONE_REGISTRY
            except ImportError as exc:  # pragma: no cover - exercised in Colab
                raise TerraMindConfigurationError(
                    "TerraMind requires the official TerraTorch runtime (frozen protocol: terratorch>=1.2.5,<1.3)."
                ) from exc
            self.model = BACKBONE_REGISTRY.build(
                self.model_name,
                pretrained=False,
                ckpt_path=str(checkpoint),
                modalities=list(self.modality_names),
                merge_method=self.merge_method,
                img_size=self.image_size,
            )
        if hasattr(self.model, "to"):
            self.model = self.model.to(self._resolve_device())
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _resolve_device(self) -> Any:
        if self._torch_device is not None:
            return self._torch_device
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised in Colab
            raise TerraMindConfigurationError("PyTorch is required for TerraMind inference.") from exc
        if self.device == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            name = self.device
        if name.startswith("cuda") and not torch.cuda.is_available():
            raise TerraMindConfigurationError("A CUDA TerraMind run was requested but CUDA is unavailable.")
        self._torch_device = torch.device(name)
        return self._torch_device

    @staticmethod
    def _find_image(sample: Mapping[str, Any], sensor: str) -> Any:
        image = sample.get("image")
        if not isinstance(image, Mapping):
            return image
        aliases = {
            "S1": ("S1", "s1", "S1GRD", "s1grd"),
            "S2": ("S2", "s2", "S2L1C", "s2l1c", "S2L2A", "s2l2a"),
        }
        for key in aliases[sensor]:
            if key in image:
                return image[key]
        raise TerraMindConfigurationError(f"Sample image dictionary has no recognized {sensor} key: {sorted(image)}")

    def _stack_sensor(self, samples: Sequence[Mapping[str, Any]], sensor: str) -> np.ndarray:
        arrays = [np.asarray(self._find_image(sample, sensor), dtype=np.float32) for sample in samples]
        if any(array.ndim != 3 for array in arrays):
            raise TerraMindConfigurationError(f"{sensor} inputs must use [channels,height,width] arrays.")
        try:
            stacked = np.stack(arrays)
        except ValueError as exc:
            raise TerraMindConfigurationError(f"{sensor} samples do not share one tensor shape.") from exc
        expected = 2 if sensor == "S1" else self.input_profile.s2_channels
        if stacked.shape[1] != expected:
            raise TerraMindConfigurationError(
                f"{sensor} profile expects {expected} channels but received {stacked.shape[1]}."
            )
        if not np.isfinite(stacked).all():
            raise TerraMindConfigurationError(f"{sensor} input contains NaN or infinity; repair source data before inference.")
        return stacked

    @staticmethod
    def _quantiles(array: np.ndarray) -> dict[str, float]:
        q = np.quantile(array, [0.001, 0.01, 0.5, 0.99, 0.999])
        return {name: float(value) for name, value in zip(("q001", "q01", "q50", "q99", "q999"), q)}

    def _convert_and_validate_s1(self, array: np.ndarray) -> np.ndarray:
        raw_summary = self._quantiles(array)
        if self.s1_unit_policy == "already_db":
            converted = array
        else:
            if np.any(array < 0):
                raise TerraMindConfigurationError(
                    f"s1_unit_policy={self.s1_unit_policy} requires nonnegative linear values, but negative values exist."
                )
            multiplier = 10.0 if self.s1_unit_policy == "linear_power_to_db" else 20.0
            converted = multiplier * np.log10(np.maximum(array, np.finfo(np.float32).tiny))
        converted_summary = self._quantiles(converted)
        if self.strict_range_check and (
            converted_summary["q001"] < -80.0 or converted_summary["q999"] > 30.0
        ):
            raise TerraMindConfigurationError(
                "S1 values are inconsistent with the official S1GRD dB normalization after applying "
                f"{self.s1_unit_policy}: {converted_summary}. Verify source units; do not bypass this in a formal run."
            )
        self.preprocessing_report["S1"] = {
            "unit_policy": self.s1_unit_policy,
            "raw_quantiles": raw_summary,
            "converted_db_quantiles": converted_summary,
        }
        return converted

    def _validate_s2(self, array: np.ndarray) -> np.ndarray:
        summary = self._quantiles(array)
        if self.strict_range_check and (
            summary["q001"] < -1000.0 or summary["q999"] > S2_DN_Q999_UPPER_GUARD
        ):
            raise TerraMindConfigurationError(
                f"S2 values are inconsistent with unscaled Sentinel reflectance/DN values: {summary}. "
                "TerraMind official means/std are not defined on [0,1] inputs."
            )
        self.preprocessing_report["S2"] = {"raw_quantiles": summary, "unit_policy": "unscaled_reflectance_dn"}
        return array

    def _resize(self, array: np.ndarray) -> np.ndarray:
        if array.shape[-2:] == (self.image_size, self.image_size):
            return array
        try:
            import torch
            import torch.nn.functional as torch_functional
        except ImportError as exc:  # pragma: no cover - exercised in Colab
            raise TerraMindConfigurationError("PyTorch is required to resize TerraMind inputs.") from exc
        tensor = torch.as_tensor(array, dtype=torch.float32)
        resized = torch_functional.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.numpy()

    @staticmethod
    def _normalize(array: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
        mean_array = np.asarray(mean, dtype=np.float32)[None, :, None, None]
        std_array = np.asarray(std, dtype=np.float32)[None, :, None, None]
        return (array - mean_array) / np.maximum(std_array, np.float32(1e-8))

    def preprocess(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        samples = list(batch["samples"])
        if not samples:
            raise TerraMindConfigurationError("TerraMind received an empty batch.")
        tensors: dict[str, np.ndarray] = {}
        if "S1" in self.sensor_mode:
            s1 = self._convert_and_validate_s1(self._stack_sensor(samples, "S1"))
            tensors["S1GRD"] = self._normalize(self._resize(s1), S1_MEAN, S1_STD)
        if "S2" in self.sensor_mode:
            s2 = self._validate_s2(self._stack_sensor(samples, "S2"))
            tensors[self.input_profile.s2_modality] = self._normalize(
                self._resize(s2), self.input_profile.s2_mean, self.input_profile.s2_std
            )
        self.preprocessing_report.update(
            {
                "input_profile": self.input_profile_name,
                "modalities": list(tensors),
                "image_size": self.image_size,
                "merge_method": self.merge_method,
            }
        )
        return {"images": tensors, "metadata": batch.get("metadata", [])}

    def _log_runtime(self, tensors: Mapping[str, Any]) -> None:
        if self._runtime_logged:
            return
        model_device = "unknown"
        if hasattr(self.model, "parameters"):
            try:
                model_device = str(next(self.model.parameters()).device)
            except (StopIteration, TypeError):
                pass
        input_devices = {key: str(value.device) for key, value in tensors.items() if hasattr(value, "device")}
        gpu_name = "none"
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(self._resolve_device())
        except (ImportError, ValueError, RuntimeError):
            pass
        print(
            "[terramind:runtime] "
            f"resolved_device={self._resolve_device()} gpu={gpu_name} model_device={model_device} "
            f"input_devices={input_devices} modalities={list(tensors)}"
        )
        self._runtime_logged = True

    def extract_embeddings(self, batch: Mapping[str, Any]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("load_model() must be called before extract_embeddings().")
        images = batch["images"]
        if hasattr(self.model, "extract_embeddings"):
            output = self.model.extract_embeddings(images)
            embeddings = np.asarray(output, dtype=np.float32)
        else:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - exercised in Colab
                raise TerraMindConfigurationError("PyTorch is required for TerraMind inference.") from exc
            tensors = {
                key: torch.as_tensor(value, dtype=torch.float32, device=self._resolve_device())
                for key, value in images.items()
            }
            self._log_runtime(tensors)
            with torch.inference_mode():
                output = self.model(tensors)
            if isinstance(output, (list, tuple)):
                if not output:
                    raise TerraMindConfigurationError("TerraMind returned no encoder layers.")
                output = output[self.layer_index]
            if isinstance(output, Mapping):
                raise TerraMindConfigurationError(
                    "TerraMind returned per-modality dictionaries. The frozen protocol requires merge_method='mean'."
                )
            embeddings = output.detach().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
            embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(axis=1)
        elif embeddings.ndim > 3:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        if embeddings.ndim != 2:
            raise TerraMindConfigurationError(f"Expected TerraMind embeddings [N,D], got {embeddings.shape}.")
        return embeddings.astype(np.float32, copy=False)

    def get_supported_modalities(self) -> Sequence[str]:
        return ("S1", "S2", "S1+S2")

    def provenance(self) -> dict[str, Any]:
        try:
            terratorch_version = importlib.metadata.version("terratorch")
        except importlib.metadata.PackageNotFoundError:
            terratorch_version = "injected_or_unavailable"
        return {
            "model_name": self.model_name,
            "model_release": self.model_release,
            "official_hf_repo": TERRAMIND_OFFICIAL_HF_REPO,
            "official_hf_filename": TERRAMIND_OFFICIAL_HF_FILENAME,
            "official_hf_revision": self.model_revision,
            "checkpoint_path": str(self.checkpoint_path or ""),
            "checkpoint_expected_sha256": self.checkpoint_expected_sha256,
            "checkpoint_actual_sha256": self.actual_checkpoint_sha256 or "not_loaded_in_injected_model_test",
            "pretrained": self.pretrained,
            "sensor_mode": self.sensor_mode,
            "modalities": list(self.modality_names),
            "input_profile": self.input_profile_name,
            "image_size": self.image_size,
            "merge_method": self.merge_method,
            "embedding_pooling": self.embedding_pooling,
            "layer_index": self.layer_index,
            "s1_unit_policy": self.s1_unit_policy,
            "strict_range_check": self.strict_range_check,
            "s2_dn_q999_upper_guard": S2_DN_Q999_UPPER_GUARD,
            "terratorch_version": terratorch_version,
            "preprocessing_report": dict(self.preprocessing_report),
        }


__all__ = [
    "INPUT_PROFILES",
    "S1_MEAN",
    "S1_STD",
    "S2_DN_Q999_UPPER_GUARD",
    "TERRAMIND_OFFICIAL_HF_FILENAME",
    "TERRAMIND_OFFICIAL_HF_REPO",
    "TERRAMIND_OFFICIAL_REVISION",
    "TERRAMIND_OFFICIAL_SHA256",
    "TerraMindAdapter",
    "TerraMindConfigurationError",
    "TerraMindInputProfile",
    "checkpoint_sha256",
    "validate_terramind_checkpoint",
    "validate_terratorch_runtime",
]
