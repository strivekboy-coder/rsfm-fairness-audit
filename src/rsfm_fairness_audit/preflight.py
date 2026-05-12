from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rsfm_fairness_audit.adapters.bigearthnet import BigEarthNetDatasetAdapter, BigEarthNetDatasetError
from rsfm_fairness_audit.config import ConfigError, load_yaml


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    message: str


def _check_import(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _path_from_config(config: dict[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value in (None, "", "null"):
        return None
    return Path(str(value))


def run_real_preflight(
    model: str,
    dataset: str,
    model_config: str | Path,
    data_root: str | Path,
    metadata_path: str | Path | None = None,
    subset_manifest_path: str | Path | None = None,
    sensor_mode: str = "S2",
    split: str = "all",
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    if model != "dofa":
        return [PreflightCheck("model", "fail", "Milestone 3C preflight only supports model='dofa'.")]
    if dataset != "bigearthnet":
        return [PreflightCheck("dataset", "fail", "Milestone 3C preflight only supports dataset='bigearthnet'.")]

    config: dict[str, Any] = {}
    try:
        config = load_yaml(model_config)
        checks.append(PreflightCheck("model_config", "pass", f"Loaded model config: {model_config}"))
    except ConfigError as exc:
        checks.append(PreflightCheck("model_config", "fail", str(exc)))
        return checks

    repo_path = _path_from_config(config, "repo_path")
    checkpoint_path = _path_from_config(config, "checkpoint_path")
    allow_hub = bool(config.get("allow_torch_hub_download", False))
    if repo_path and checkpoint_path:
        repo_ok = repo_path.exists()
        ckpt_ok = checkpoint_path.exists()
        checks.append(
            PreflightCheck(
                "dofa_repo",
                "pass" if repo_ok else "fail",
                f"DOFA repo_path {'exists' if repo_ok else 'does not exist'}: {repo_path}",
            )
        )
        checks.append(
            PreflightCheck(
                "dofa_checkpoint",
                "pass" if ckpt_ok else "fail",
                f"DOFA checkpoint_path {'exists' if ckpt_ok else 'does not exist'}: {checkpoint_path}",
            )
        )
    elif allow_hub:
        checks.append(
            PreflightCheck(
                "dofa_loading",
                "warn",
                "torch.hub mode is enabled. The first real run may download the official DOFA checkpoint.",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "dofa_loading",
                "fail",
                "DOFA is not configured for real inference. Set repo_path + checkpoint_path, or explicitly set allow_torch_hub_download: true.",
            )
        )

    expected_bands = config.get("expected_bands")
    wavelengths = config.get("wavelength_list")
    if isinstance(wavelengths, list) and expected_bands == len(wavelengths):
        checks.append(PreflightCheck("dofa_bands", "pass", f"expected_bands matches wavelength_list length: {expected_bands}"))
    else:
        checks.append(
            PreflightCheck(
                "dofa_bands",
                "fail",
                "expected_bands must match wavelength_list length before real DOFA inference.",
            )
        )

    root = Path(data_root)
    checks.append(
        PreflightCheck(
            "data_root",
            "pass" if root.exists() else "fail",
            f"BigEarthNet data_root {'exists' if root.exists() else 'does not exist'}: {root}",
        )
    )

    if root.exists():
        try:
            adapter = BigEarthNetDatasetAdapter(
                data_root=root,
                metadata_path=metadata_path,
                subset_manifest_path=subset_manifest_path,
                subset_size=1,
                split=split,
                sensor_mode=sensor_mode,
            )
            metadata = adapter.load_metadata()
            sample = adapter.load_sample(0)
            image = sample["image"]
            shape = getattr(image, "shape", None)
            checks.append(PreflightCheck("bigearthnet_metadata", "pass", f"Loaded one metadata row: {metadata[0].get('sample_id')}"))
            if shape is not None and expected_bands is not None and int(shape[0]) != int(expected_bands):
                checks.append(
                    PreflightCheck(
                        "bigearthnet_bands",
                        "fail",
                        f"First chip has {shape[0]} bands but DOFA config expects {expected_bands}.",
                    )
                )
            else:
                checks.append(PreflightCheck("bigearthnet_sample", "pass", f"Loaded first chip with shape {shape}."))
        except (BigEarthNetDatasetError, ValueError, OSError) as exc:
            checks.append(PreflightCheck("bigearthnet_metadata", "fail", str(exc)))

    for module in ["numpy", "matplotlib", "yaml"]:
        checks.append(
            PreflightCheck(
                f"dependency_{module}",
                "pass" if _check_import(module) else "fail",
                f"Python module '{module}' {'is importable' if _check_import(module) else 'is not importable'}.",
            )
        )
    for module in ["torch", "timm"]:
        checks.append(
            PreflightCheck(
                f"dependency_{module}",
                "pass" if _check_import(module) else "warn",
                f"Optional DOFA module '{module}' {'is importable' if _check_import(module) else 'is not importable yet'}.",
            )
        )

    if _check_import("torch"):
        import torch

        cuda = torch.cuda.is_available()
        checks.append(
            PreflightCheck(
                "device",
                "pass" if cuda else "warn",
                "CUDA is available for Colab real smoke runs." if cuda else "CUDA is not available; tiny CPU smoke may work but Colab GPU is recommended.",
            )
        )
    else:
        checks.append(PreflightCheck("device", "warn", "PyTorch is not installed, so CUDA availability could not be checked."))

    return checks


def checks_to_json(checks: list[PreflightCheck]) -> str:
    return json.dumps([asdict(check) for check in checks], indent=2)
