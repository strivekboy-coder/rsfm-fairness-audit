from __future__ import annotations

from functools import partial
import hashlib
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.base import ModelAdapter
from rsfm_fairness_audit.band_profiles import get_band_profile
from rsfm_fairness_audit.config import load_yaml


class DOFAConfigurationError(RuntimeError):
    """Raised when DOFA cannot be loaded without guessing reproduction details."""


_CHECKPOINT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


class DOFAAdapter(ModelAdapter):
    """First DOFA adapter with explicit, no-surprise loading behavior."""

    official_dofav2_repo_revision = "0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9"
    official_dofav2_architecture_repo = "https://github.com/xiong-zhitong/terratorch"
    official_dofav2_architecture_revision = "208fbf53654b263091db3a648d210ad532ad1aad"
    official_dofav2_timm_version = "1.0.15"
    official_dofav2_patch_size = 14
    official_dofav2_embedding_semantics = "mean_final_normalized_patch_tokens_excluding_cls"
    official_dofav2_checkpoint_sha256 = (
        "e1be9d50fb3e4e3640e337d098b92d67797eaf2a579de3b7a1e363095885314d"
    )

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
        model_release: str = "dofa_v1_base_e100",
        device: str = "cpu",
        batch_size: int = 32,
        band_profile: str | None = None,
        expected_bands: int | None = None,
        image_size: int | None = None,
        embedding_layer: str = "forward_features",
        embedding_pooling: str = "flatten",
        normalization_mean: Sequence[float] | None = None,
        normalization_std: Sequence[float] | None = None,
        input_scale: float | None = None,
        sensor_mode: str = "S2",
        wavelengths: Sequence[float] | None = None,
        model: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
        allow_torch_hub_download: bool = False,
        checkpoint_sha256: str | None = None,
        minimum_checkpoint_key_coverage: float = 0.90,
        require_exact_checkpoint_match: bool = False,
        repo_revision: str | None = None,
        architecture_source_repo: str | None = None,
        architecture_source_revision: str | None = None,
        required_timm_version: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.repo_path = Path(repo_path) if repo_path else None
        self.torch_hub_repo = torch_hub_repo
        self.model_variant = model_variant
        self.model_release = model_release
        self.device = device
        self.batch_size = batch_size
        self.band_profile = band_profile
        self.expected_bands = expected_bands
        self.image_size = image_size
        self.embedding_layer = embedding_layer
        self.embedding_pooling = embedding_pooling
        self.normalization_mean = list(normalization_mean) if normalization_mean is not None else None
        self.normalization_std = list(normalization_std) if normalization_std is not None else None
        self.input_scale = float(input_scale) if input_scale not in (None, "", 0) else None
        self.sensor_mode = sensor_mode.upper()
        self.wavelengths = list(wavelengths) if wavelengths is not None else None
        self.model = model
        self.model_loader = model_loader
        self.allow_torch_hub_download = allow_torch_hub_download
        self.checkpoint_sha256 = str(checkpoint_sha256).lower() if checkpoint_sha256 else None
        self.minimum_checkpoint_key_coverage = float(minimum_checkpoint_key_coverage)
        self.require_exact_checkpoint_match = bool(require_exact_checkpoint_match)
        self.repo_revision = str(repo_revision).lower() if repo_revision else None
        is_dofav2 = self.model_variant == "dofav2_vit_base" or self.model_release.startswith("dofav2")
        self.architecture_source_repo = (
            str(architecture_source_repo)
            if architecture_source_repo is not None
            else self.official_dofav2_architecture_repo if is_dofav2 else None
        )
        self.architecture_source_revision = (
            str(architecture_source_revision).lower()
            if architecture_source_revision is not None
            else self.official_dofav2_architecture_revision if is_dofav2 else None
        )
        self.required_timm_version = (
            str(required_timm_version)
            if required_timm_version is not None
            else self.official_dofav2_timm_version if is_dofav2 else None
        )
        self.actual_repo_revision: str | None = None
        self.actual_checkpoint_sha256: str | None = None
        self.actual_timm_version: str | None = None
        self.checkpoint_load_report: dict[str, Any] = {}
        self._torch_device: Any | None = None
        self._logged_profile = False
        self._logged_device_state = False

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
        profile_name = data.get("band_profile")
        profile = get_band_profile(str(profile_name)) if profile_name else {}
        merged = dict(profile)
        merged.update({key: value for key, value in data.items() if value is not None})
        if "wavelengths" in merged and "wavelength_list" not in merged:
            raise DOFAConfigurationError("Use 'wavelength_list' in DOFA config; legacy 'wavelengths' is ignored.")
        return cls(
            checkpoint_path=merged.get("checkpoint_path"),
            repo_path=merged.get("repo_path"),
            torch_hub_repo=str(merged.get("torch_hub_repo", "zhu-xlab/DOFA")),
            model_variant=str(merged.get("model_variant", "vit_base_dofa")),
            model_release=str(merged.get("model_release", "dofa_v1_base_e100")),
            device=str(merged.get("device", "cpu")),
            batch_size=int(merged.get("batch_size", 32)),
            band_profile=str(profile_name) if profile_name else None,
            expected_bands=merged.get("expected_bands"),
            image_size=merged.get("image_size"),
            embedding_layer=str(merged.get("embedding_layer", "forward_features")),
            embedding_pooling=str(merged.get("embedding_pooling", "flatten")),
            normalization_mean=merged.get("normalization_mean"),
            normalization_std=merged.get("normalization_std"),
            input_scale=merged.get("input_scale"),
            sensor_mode=str(merged.get("input_modality", merged.get("sensor_mode", "S2"))),
            wavelengths=merged.get("wavelength_list"),
            model=model,
            model_loader=model_loader,
            allow_torch_hub_download=bool(data.get("allow_torch_hub_download", False)),
            checkpoint_sha256=merged.get("checkpoint_sha256"),
            minimum_checkpoint_key_coverage=float(merged.get("minimum_checkpoint_key_coverage", 0.90)),
            require_exact_checkpoint_match=bool(merged.get("require_exact_checkpoint_match", False)),
            repo_revision=merged.get("repo_revision"),
            architecture_source_repo=merged.get("architecture_source_repo"),
            architecture_source_revision=merged.get("architecture_source_revision"),
            required_timm_version=merged.get("required_timm_version"),
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
        if self.embedding_pooling not in {"flatten", "mean_tokens"}:
            raise DOFAConfigurationError("embedding_pooling must be 'flatten' or 'mean_tokens'.")
        if not 0.0 < self.minimum_checkpoint_key_coverage <= 1.0:
            raise DOFAConfigurationError("minimum_checkpoint_key_coverage must be in (0, 1].")
        if self.model_release.startswith("dofav2") and self.allow_torch_hub_download and self.repo_path is None:
            raise DOFAConfigurationError(
                "The official zhu-xlab/DOFA torch.hub entrypoint still loads the DOFA v1 base checkpoint. "
                "DOFAv2 requires a pinned local official repo plus the verified DOFAv2 checkpoint."
            )
        if self.model_release.startswith("dofav2") and self.repo_path is not None and not self.repo_revision:
            raise DOFAConfigurationError(
                "DOFAv2 formal loading requires repo_revision so the matching wave_dynamic_layer implementation is pinned."
            )
        if self.model_release.startswith("dofav2"):
            if self.image_size != 224:
                raise DOFAConfigurationError("The frozen DOFAv2 Base checkpoint requires image_size=224.")
            if self.embedding_layer != "forward_features" or self.embedding_pooling != "mean_tokens":
                raise DOFAConfigurationError(
                    "The frozen DOFAv2 embedding protocol requires forward_features plus mean_tokens."
                )
            if self.architecture_source_repo != self.official_dofav2_architecture_repo:
                raise DOFAConfigurationError("DOFAv2 architecture_source_repo differs from the verified author source.")
            if self.architecture_source_revision != self.official_dofav2_architecture_revision:
                raise DOFAConfigurationError(
                    "DOFAv2 architecture_source_revision differs from the verified author implementation."
                )
            if self.required_timm_version != self.official_dofav2_timm_version:
                raise DOFAConfigurationError(
                    f"DOFAv2 requires frozen timm=={self.official_dofav2_timm_version}; "
                    f"configured {self.required_timm_version!r}."
                )
        if self.repo_revision and not re.fullmatch(r"[0-9a-f]{40}", self.repo_revision):
            raise DOFAConfigurationError("repo_revision must be a full 40-character Git commit SHA.")
        if self.architecture_source_revision and not re.fullmatch(r"[0-9a-f]{40}", self.architecture_source_revision):
            raise DOFAConfigurationError("architecture_source_revision must be a full 40-character Git commit SHA.")

    def _load_from_torch_hub(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise DOFAConfigurationError("PyTorch is required for the official DOFA torch.hub loading path.") from exc
        model = torch.hub.load(self.torch_hub_repo, self.model_variant, pretrained=True)
        self.checkpoint_load_report = {"loader": "torch_hub", "model_release": self.model_release}
        return self._move_to_device(model)

    def _verify_dofav2_runtime(self) -> None:
        try:
            import timm
        except ImportError as exc:
            raise DOFAConfigurationError(
                f"DOFAv2 requires timm=={self.required_timm_version}. "
                "Install the frozen optional environment with requirements-dofa.txt."
            ) from exc
        self.actual_timm_version = str(getattr(timm, "__version__", "unknown"))
        if self.actual_timm_version != self.required_timm_version:
            raise DOFAConfigurationError(
                f"DOFAv2 runtime requires timm=={self.required_timm_version}, "
                f"but found timm=={self.actual_timm_version}. "
                "Install requirements-dofa.txt in the DOFAv2 runtime instead of weakening checkpoint checks."
            )

    @staticmethod
    def _build_dofav2_base_patch14(ofa_vit: Any, torch: Any) -> Any:
        class LayerScale(torch.nn.Module):
            def __init__(self, dim: int, init_values: float = 1e-5) -> None:
                super().__init__()
                self.gamma = torch.nn.Parameter(init_values * torch.ones(dim))

            def forward(self, value: Any) -> Any:
                return value * self.gamma

        model = ofa_vit(
            img_size=224,
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=0,
            global_pool=False,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        )
        for block in model.blocks:
            block.ls1 = LayerScale(768)
            block.ls2 = LayerScale(768)
        return model

    def _load_from_local_repo(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise DOFAConfigurationError("PyTorch is required to load DOFA from a local official repository.") from exc
        if self.model_variant not in {"vit_base_dofa", "dofav2_vit_base"}:
            raise DOFAConfigurationError(
                f"Local repo loading currently supports vit_base_dofa/dofav2_vit_base, got {self.model_variant!r}. "
                "Add and verify the official constructor before using another variant."
            )
        assert self.repo_path is not None
        assert self.checkpoint_path is not None
        if self.repo_revision:
            try:
                result = subprocess.run(
                    ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise DOFAConfigurationError(
                    f"Could not verify the pinned DOFA repository revision under {self.repo_path}."
                ) from exc
            self.actual_repo_revision = result.stdout.strip().lower()
            if self.actual_repo_revision != self.repo_revision:
                raise DOFAConfigurationError(
                    f"DOFA repository revision mismatch: expected {self.repo_revision}, got {self.actual_repo_revision}."
                )
            dirty = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_path),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    "dofa_v1.py",
                    "wave_dynamic_layer.py",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if dirty.returncode != 0 or dirty.stdout.strip():
                raise DOFAConfigurationError(
                    "Formal DOFA loading requires unmodified tracked dofa_v1.py and wave_dynamic_layer.py."
                )
        if self.model_variant == "dofav2_vit_base":
            self._verify_dofav2_runtime()
        sys.path.insert(0, str(self.repo_path))
        try:
            if self.model_variant == "dofav2_vit_base":
                from dofa_v1 import OFAViT

                model = self._build_dofav2_base_patch14(OFAViT, torch)
            else:
                from dofa_v1 import vit_base_patch16

                model = vit_base_patch16()
        except ImportError as exc:
            raise DOFAConfigurationError(
                f"Could not import the verified DOFA model implementation from repo_path={self.repo_path}. "
                "Clone the official https://github.com/zhu-xlab/DOFA repository or use torch_hub loading."
            ) from exc
        finally:
            try:
                sys.path.remove(str(self.repo_path))
            except ValueError:
                pass
        self.actual_checkpoint_sha256 = self._file_sha256(self.checkpoint_path)
        if self.checkpoint_sha256 and self.actual_checkpoint_sha256 != self.checkpoint_sha256:
            raise DOFAConfigurationError(
                f"DOFA checkpoint SHA256 mismatch: expected {self.checkpoint_sha256}, got {self.actual_checkpoint_sha256}."
            )
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = self._extract_state_dict(checkpoint)
        model_state = model.state_dict()
        model_keys_missing_from_checkpoint = sorted(set(model_state) - set(state_dict))
        checkpoint_keys_missing_from_model = sorted(set(state_dict) - set(model_state))
        same_name_shape_mismatches = [
            {
                "key": key,
                "model_shape": list(model_state[key].shape),
                "checkpoint_shape": list(getattr(state_dict[key], "shape", ())),
                "model_numel": int(model_state[key].numel()),
                "checkpoint_numel": int(state_dict[key].numel()) if hasattr(state_dict[key], "numel") else 0,
            }
            for key in sorted(set(model_state) & set(state_dict))
            if not hasattr(state_dict[key], "shape")
            or tuple(state_dict[key].shape) != tuple(model_state[key].shape)
        ]
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and hasattr(value, "shape") and tuple(value.shape) == tuple(model_state[key].shape)
        }
        matched_numel = sum(int(model_state[key].numel()) for key in compatible)
        total_numel = sum(int(value.numel()) for value in model_state.values())
        coverage = matched_numel / max(total_numel, 1)
        self.checkpoint_load_report = {
            "loader": "official_local_repo",
            "model_release": self.model_release,
            "checkpoint_sha256": self.actual_checkpoint_sha256,
            "repo_revision": self.actual_repo_revision,
            "architecture_source_repo": self.architecture_source_repo,
            "architecture_source_revision": self.architecture_source_revision,
            "required_timm_version": self.required_timm_version if self.model_variant == "dofav2_vit_base" else None,
            "actual_timm_version": self.actual_timm_version,
            "patch_size": self.official_dofav2_patch_size if self.model_variant == "dofav2_vit_base" else 16,
            "embedding_semantics": (
                self.official_dofav2_embedding_semantics
                if self.model_variant == "dofav2_vit_base"
                else self.embedding_pooling
            ),
            "parameter_coverage": coverage,
            "matched_parameter_numel": matched_numel,
            "model_parameter_numel": total_numel,
            "matched_keys": len(compatible),
            "model_keys": len(model_state),
            "checkpoint_keys": len(state_dict),
            "model_keys_missing_from_checkpoint": model_keys_missing_from_checkpoint,
            "checkpoint_keys_missing_from_model": checkpoint_keys_missing_from_model,
            "same_name_shape_mismatches": same_name_shape_mismatches,
        }
        if self.require_exact_checkpoint_match and (
            model_keys_missing_from_checkpoint
            or checkpoint_keys_missing_from_model
            or same_name_shape_mismatches
        ):
            raise DOFAConfigurationError(
                "DOFA checkpoint is not an exact structural match: "
                f"model_missing={len(model_keys_missing_from_checkpoint)}, "
                f"checkpoint_extra={len(checkpoint_keys_missing_from_model)}, "
                f"shape_mismatches={len(same_name_shape_mismatches)}. "
                "Run scripts/diagnose_dofa_checkpoint_compatibility.py for the full report."
            )
        if coverage < self.minimum_checkpoint_key_coverage:
            raise DOFAConfigurationError(
                f"Only {coverage:.3%} of DOFA model parameters match the checkpoint; "
                f"minimum is {self.minimum_checkpoint_key_coverage:.3%}. "
                f"shape_mismatches={len(same_name_shape_mismatches)}. Refusing a partial silent load."
            )
        incompatible = model.load_state_dict(
            state_dict if self.require_exact_checkpoint_match else compatible,
            strict=self.require_exact_checkpoint_match,
        )
        self.checkpoint_load_report["load_missing_keys"] = list(incompatible.missing_keys)
        self.checkpoint_load_report["load_unexpected_keys"] = list(incompatible.unexpected_keys)
        return self._move_to_device(model)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        resolved = path.resolve()
        stat = resolved.stat()
        cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
        cached = _CHECKPOINT_HASH_CACHE.get(cache_key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        observed = digest.hexdigest()
        _CHECKPOINT_HASH_CACHE[cache_key] = observed
        return observed

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        state = checkpoint
        if isinstance(state, Mapping):
            for key in ("state_dict", "model_state_dict", "model", "backbone"):
                candidate = state.get(key)
                if isinstance(candidate, Mapping):
                    state = candidate
                    break
        if not isinstance(state, Mapping):
            raise DOFAConfigurationError("DOFA checkpoint does not contain a recognizable state dictionary.")
        output: dict[str, Any] = {}
        prefixes = ("module.", "_orig_mod.", "model.", "backbone.")
        for raw_key, value in state.items():
            key = str(raw_key)
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                        changed = True
            output[key] = value
        return output

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

    def _forward_dofav2_patch_tokens(self, images: Any, wavelengths: Sequence[float]) -> Any:
        if self.model is None:
            raise RuntimeError("load_model() must be called before DOFAv2 feature extraction.")
        try:
            import torch
        except ImportError as exc:
            raise DOFAConfigurationError("PyTorch is required for DOFAv2 feature extraction.") from exc
        wave_tensor = torch.tensor(wavelengths, device=images.device, dtype=torch.float32)
        tokens, _ = self.model.patch_embed(images, wave_tensor)
        expected_tokens = int(self.model.pos_embed.shape[1]) - 1
        if int(tokens.shape[1]) != expected_tokens:
            raise DOFAConfigurationError(
                "DOFAv2 patch-token count does not match the frozen positional embedding: "
                f"tokens={int(tokens.shape[1])}, expected={expected_tokens}."
            )
        tokens = tokens + self.model.pos_embed[:, 1:, :]
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        return tokens[:, 1:, :]

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
        self._log_profile(array.shape[1])
        wavelengths = self._resolve_wavelengths(array.shape[1])
        self._validate_normalization_lengths(array.shape[1])
        array = self._scale_input(array)
        array = self._normalize(array)
        array = self._resize_if_needed(array)
        return {"images": array, "metadata": batch["metadata"], "wavelengths": wavelengths}

    def _scale_input(self, array: np.ndarray) -> np.ndarray:
        if self.input_scale is None:
            return array
        if self.input_scale <= 0:
            raise DOFAConfigurationError("input_scale must be positive when provided.")
        return array / np.float32(self.input_scale)

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

    def _validate_normalization_lengths(self, channels: int) -> None:
        mean = self.normalization_mean if self.normalization_mean is not None else self.official_means.get(self.sensor_mode)
        std = self.normalization_std if self.normalization_std is not None else self.official_stds.get(self.sensor_mode)
        if mean is not None and len(mean) != channels:
            raise DOFAConfigurationError(
                f"normalization_mean length ({len(mean)}) does not match input channels ({channels})."
            )
        if std is not None and len(std) != channels:
            raise DOFAConfigurationError(
                f"normalization_std length ({len(std)}) does not match input channels ({channels})."
            )

    def _log_profile(self, channels: int) -> None:
        if self._logged_profile:
            return
        print(
            "[info] DOFA band profile: "
            f"{self.band_profile or 'custom'} expected_bands={self.expected_bands} input_channels={channels}"
        )
        self._logged_profile = True

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
        if not self._logged_device_state:
            first_parameter = next(iter(self.model.parameters()), None) if hasattr(self.model, "parameters") else None
            parameter_device = str(first_parameter.device) if first_parameter is not None else "unavailable"
            gpu_name = (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "none"
            )
            print(
                "[dofa:runtime] "
                f"resolved_device={self._resolve_device()} cuda_available={torch.cuda.is_available()} "
                f"gpu={gpu_name} model_parameter_device={parameter_device} input_tensor_device={tensor.device}",
                flush=True,
            )
            self._logged_device_state = True
        with torch.inference_mode():
            if self.model_variant == "dofav2_vit_base" and self.embedding_layer == "forward_features":
                output = self._forward_dofav2_patch_tokens(tensor, wavelengths)
            elif self.embedding_layer == "forward_features" and hasattr(self.model, "forward_features"):
                output = self.model.forward_features(tensor, wave_list=wavelengths)
            else:
                output = self.model(tensor, wave_list=wavelengths)
        if isinstance(output, dict):
            output = next(iter(output.values()))
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        embeddings = np.asarray(output, dtype=np.float32)
        if embeddings.ndim > 2:
            if self.embedding_pooling == "mean_tokens":
                if embeddings.ndim == 3:
                    embeddings = embeddings.mean(axis=1)
                elif embeddings.ndim == 4:
                    embeddings = embeddings.mean(axis=(2, 3))
                else:
                    embeddings = embeddings.reshape(embeddings.shape[0], -1)
            else:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings

    def get_supported_modalities(self) -> Sequence[str]:
        return ("S1", "S2")

    @staticmethod
    def _json_serializable(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): DOFAAdapter._json_serializable(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [DOFAAdapter._json_serializable(item) for item in value]
        return str(value)

    def provenance(self) -> dict[str, Any]:
        """Return the complete JSON-safe lineage contract for DOFA embeddings."""

        actual_timm_version = self.actual_timm_version
        if actual_timm_version is None:
            try:
                actual_timm_version = importlib.metadata.version("timm")
            except importlib.metadata.PackageNotFoundError:
                actual_timm_version = None

        if self.wavelengths is not None:
            wavelength_list = [float(value) for value in self.wavelengths]
            wavelength_profile = self.band_profile or "explicit_wavelength_list"
        elif self.sensor_mode == "S1" and self.expected_bands == 2:
            wavelength_list = list(self.verified_wavelengths["S1"])
            wavelength_profile = "S1"
        elif self.sensor_mode == "S2" and self.expected_bands == 9:
            wavelength_list = list(self.verified_wavelengths["S2_OFFICIAL_DEMO_9CH"])
            wavelength_profile = "S2_OFFICIAL_DEMO_9CH"
        else:
            wavelength_list = None
            wavelength_profile = "unresolved"

        if self.normalization_mean is None and self.normalization_std is None:
            normalization_mean = self.official_means.get(self.sensor_mode)
            normalization_std = self.official_stds.get(self.sensor_mode)
            normalization_source = "official_sensor_profile" if normalization_mean is not None else "none"
        else:
            normalization_mean = self.normalization_mean
            normalization_std = self.normalization_std
            normalization_source = "configured_band_profile_or_override"

        embedding_semantics = (
            self.official_dofav2_embedding_semantics
            if self.model_variant == "dofav2_vit_base"
            else self.embedding_pooling
        )
        lineage = {
            "model_variant": self.model_variant,
            "model_release": self.model_release,
            "sensor_mode": self.sensor_mode,
            "band_profile": self.band_profile,
            "expected_bands": self.expected_bands,
            "image_size": self.image_size,
            "input_scale": self.input_scale,
            "embedding_layer": self.embedding_layer,
            "embedding_pooling": self.embedding_pooling,
            "embedding_semantics": embedding_semantics,
            "wavelength_list": wavelength_list,
            "resolved_official_wavelength_profile": wavelength_profile,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_expected_sha256": self.checkpoint_sha256,
            "checkpoint_actual_sha256": self.actual_checkpoint_sha256,
            "repo_path": self.repo_path,
            "repo_revision_expected": self.repo_revision,
            "repo_revision_actual": self.actual_repo_revision,
            "architecture_source_repo": self.architecture_source_repo,
            "architecture_source_revision": self.architecture_source_revision,
            "required_timm_version": self.required_timm_version,
            "actual_timm_version": actual_timm_version,
            "checkpoint_load_report": self.checkpoint_load_report,
            "normalization": {
                "source": normalization_source,
                "mean": normalization_mean,
                "std": normalization_std,
            },
            "preprocessing": {
                "input_dtype": "float32",
                "input_layout": "batch_channels_height_width",
                "input_scale_divisor": self.input_scale,
                "normalization_order": "scale_then_normalize",
                "resize": {
                    "target_size": self.image_size,
                    "mode": "bilinear",
                    "align_corners": False,
                },
                "wavelength_profile": wavelength_profile,
            },
        }
        return self._json_serializable(lineage)
