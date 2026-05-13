from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rsfm_fairness_audit.adapters.ben_ge import BenGEDatasetAdapter, BenGEDatasetError
from rsfm_fairness_audit.adapters.bigearthnet import BigEarthNetDatasetAdapter, BigEarthNetDatasetError
from rsfm_fairness_audit.adapters.sen1floods11 import Sen1Floods11DatasetAdapter, Sen1Floods11DatasetError
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
    if model not in {"dofa", "croma", "prithvi"}:
        return [PreflightCheck("model", "fail", "Preflight only supports model='dofa', model='croma', or model='prithvi'.")]
    if dataset not in {"bigearthnet", "ben_ge", "sen1floods11"}:
        return [PreflightCheck("dataset", "fail", "Preflight only supports dataset='bigearthnet', dataset='ben_ge', or dataset='sen1floods11'.")]

    config: dict[str, Any] = {}
    try:
        config = load_yaml(model_config)
        checks.append(PreflightCheck("model_config", "pass", f"Loaded model config: {model_config}"))
    except ConfigError as exc:
        checks.append(PreflightCheck("model_config", "fail", str(exc)))
        return checks

    expected_bands = config.get("expected_bands")
    if model == "dofa":
        _append_dofa_preflight_checks(checks, config)
    elif model == "croma":
        _append_croma_preflight_checks(checks, config)
    else:
        _append_prithvi_preflight_checks(checks, config)

    wavelengths = config.get("wavelength_list")
    if model == "dofa":
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
    elif model == "croma":
        _append_croma_band_checks(checks, config)
    else:
        _append_prithvi_shape_checks(checks, config)

    root = Path(data_root)
    checks.append(
        PreflightCheck(
            "data_root",
            "pass" if root.exists() else "fail",
            f"{dataset} data_root {'exists' if root.exists() else 'does not exist'}: {root}",
        )
    )

    if root.exists():
        try:
            if dataset == "bigearthnet":
                adapter = BigEarthNetDatasetAdapter(
                    data_root=root,
                    metadata_path=metadata_path,
                    subset_manifest_path=subset_manifest_path,
                    subset_size=1,
                    split=split,
                    sensor_mode=sensor_mode,
                )
            elif dataset == "ben_ge":
                adapter = BenGEDatasetAdapter(
                    data_root=root,
                    metadata_path=metadata_path,
                    subset_size=1,
                    split=split,
                    sensor_mode=sensor_mode,
                )
            else:
                adapter = Sen1Floods11DatasetAdapter(
                    data_root=root,
                    metadata_path=metadata_path,
                    subset_size=1,
                    split=split,
                )
            metadata = adapter.load_metadata()
            sample = adapter.load_sample(0)
            image = sample["image"]
            shape = getattr(image, "shape", None)
            checks.append(PreflightCheck(f"{dataset}_metadata", "pass", f"Loaded one metadata row: {metadata[0].get('sample_id')}"))
            if isinstance(image, dict):
                s1_shape = getattr(image.get("S1"), "shape", None)
                s2_shape = getattr(image.get("S2"), "shape", None)
                checks.append(PreflightCheck(f"{dataset}_sample", "pass", f"Loaded paired sample with S1 shape {s1_shape} and S2 shape {s2_shape}."))
            elif shape is not None and expected_bands is not None and int(shape[0]) != int(expected_bands):
                if model == "prithvi" and len(shape) == 4:
                    expected_frames = config.get("expected_frames")
                    expected_size = config.get("image_size")
                    ok = (
                        int(shape[0]) == int(expected_frames)
                        and int(shape[1]) == int(expected_bands)
                        and int(shape[2]) == int(expected_size)
                        and int(shape[3]) == int(expected_size)
                    )
                    checks.append(
                        PreflightCheck(
                            f"{dataset}_sample",
                            "pass" if ok else "fail",
                            "Loaded Prithvi-ready chip with shape "
                            f"{shape}; expected ({expected_frames}, {expected_bands}, {expected_size}, {expected_size}).",
                        )
                    )
                else:
                    checks.append(
                        PreflightCheck(
                            f"{dataset}_bands",
                            "fail",
                            f"First chip has {shape[0]} bands but {model.upper()} config expects {expected_bands}.",
                        )
                    )
            else:
                checks.append(PreflightCheck(f"{dataset}_sample", "pass", f"Loaded first chip with shape {shape}."))
        except (BigEarthNetDatasetError, BenGEDatasetError, Sen1Floods11DatasetError, ValueError, OSError) as exc:
            checks.append(PreflightCheck(f"{dataset}_metadata", "fail", str(exc)))

    for module in ["numpy", "matplotlib", "yaml"]:
        checks.append(
            PreflightCheck(
                f"dependency_{module}",
                "pass" if _check_import(module) else "fail",
                f"Python module '{module}' {'is importable' if _check_import(module) else 'is not importable'}.",
            )
        )
    if model == "dofa":
        optional_modules = ["torch", "timm"]
    elif model == "croma":
        optional_modules = ["torch", "einops", "huggingface_hub"]
    else:
        optional_modules = ["torch", "terratorch", "huggingface_hub", "rasterio"]
    for module in optional_modules:
        checks.append(
            PreflightCheck(
                f"dependency_{module}",
                "pass" if _check_import(module) else "warn",
                f"Optional {model.upper()} module '{module}' {'is importable' if _check_import(module) else 'is not importable yet'}.",
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


def _append_dofa_preflight_checks(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    repo_path = _path_from_config(config, "repo_path")
    checkpoint_path = _path_from_config(config, "checkpoint_path")
    allow_hub = bool(config.get("allow_torch_hub_download", False))
    if repo_path and checkpoint_path:
        checks.append(
            PreflightCheck(
                "dofa_repo",
                "pass" if repo_path.exists() else "fail",
                f"DOFA repo_path {'exists' if repo_path.exists() else 'does not exist'}: {repo_path}",
            )
        )
        checks.append(
            PreflightCheck(
                "dofa_checkpoint",
                "pass" if checkpoint_path.exists() else "fail",
                f"DOFA checkpoint_path {'exists' if checkpoint_path.exists() else 'does not exist'}: {checkpoint_path}",
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


def _append_croma_preflight_checks(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    checkpoint_path = _path_from_config(config, "checkpoint_path")
    repo_path = _path_from_config(config, "repo_path")
    source_file_path = _path_from_config(config, "source_file_path")
    allow_hf = bool(config.get("allow_hf_download", False))
    hf_repo_id = str(config.get("hf_repo_id", ""))
    hf_filename = str(config.get("hf_checkpoint_filename", ""))
    modality = str(config.get("input_modality", ""))
    if modality in {"optical", "SAR", "both"}:
        checks.append(PreflightCheck("croma_modality", "pass", f"CROMA is configured for modality={modality}."))
    else:
        checks.append(PreflightCheck("croma_modality", "fail", "CROMA input_modality must be optical, SAR, or both."))
    if hf_repo_id == "antofuller/CROMA" and hf_filename in {"CROMA_base.pt", "CROMA_large.pt"}:
        checks.append(PreflightCheck("croma_hf_checkpoint", "pass", f"Official CROMA checkpoint target: {hf_repo_id}/{hf_filename}"))
    else:
        checks.append(
            PreflightCheck(
                "croma_hf_checkpoint",
                "fail",
                "CROMA HF download is restricted to antofuller/CROMA with CROMA_base.pt or CROMA_large.pt.",
            )
        )
    if checkpoint_path:
        checks.append(
            PreflightCheck(
                "croma_checkpoint",
                "pass" if checkpoint_path.exists() else "fail",
                f"CROMA checkpoint_path {'exists' if checkpoint_path.exists() else 'does not exist'}: {checkpoint_path}",
            )
        )
    elif allow_hf:
        checks.append(
            PreflightCheck(
                "croma_loading",
                "warn",
                "allow_hf_download is enabled. The first real run may download only the configured official CROMA checkpoint.",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "croma_loading",
                "fail",
                "CROMA checkpoint_path is missing and allow_hf_download is false.",
            )
        )
    source_file = source_file_path or (repo_path / "use_croma.py" if repo_path else None)
    if source_file:
        checks.append(
            PreflightCheck(
                "croma_source",
                "pass" if source_file.exists() else "fail",
                f"CROMA source file {'exists' if source_file.exists() else 'does not exist'}: {source_file}",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "croma_source",
                "fail",
                "Set source_file_path to official use_croma.py or repo_path to a local clone of https://github.com/antofuller/CROMA.",
            )
        )


def _append_croma_band_checks(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    modality = str(config.get("input_modality", "optical"))
    expected_bands = config.get("expected_bands")
    expected_s1 = config.get("expected_s1_bands")
    expected_s2 = config.get("expected_s2_bands", expected_bands)
    if modality == "SAR" and expected_s1 == 2:
        checks.append(PreflightCheck("croma_bands", "pass", "CROMA SAR mode expects 2 Sentinel-1 channels."))
    elif modality == "optical" and expected_s2 == 12:
        checks.append(PreflightCheck("croma_bands", "pass", "CROMA optical mode expects 12 Sentinel-2 channels."))
    elif modality == "both" and expected_s1 == 2 and expected_s2 == 12:
        checks.append(PreflightCheck("croma_bands", "pass", "CROMA both mode expects 2 S1 channels and 12 S2 channels."))
    else:
        checks.append(PreflightCheck("croma_bands", "fail", "CROMA band counts must match modality: SAR=2 S1, optical=12 S2, both=2+12."))


def _append_prithvi_preflight_checks(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    hf_model_id = str(config.get("hf_model_id", ""))
    terratorch_model_name = str(config.get("terratorch_model_name", hf_model_id))
    allow_hf = bool(config.get("allow_hf_download", False))
    if hf_model_id == "ibm-nasa-geospatial/Prithvi-EO-2.0-300M":
        checks.append(PreflightCheck("prithvi_model_id", "pass", f"Using official Prithvi model: {hf_model_id}"))
    else:
        checks.append(PreflightCheck("prithvi_model_id", "fail", "Phase 3 uses only ibm-nasa-geospatial/Prithvi-EO-2.0-300M."))
    if terratorch_model_name in {"terratorch_prithvi_eo_v2_300", "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"}:
        checks.append(PreflightCheck("prithvi_terratorch_name", "pass", f"TerraTorch registry target: {terratorch_model_name}"))
    else:
        checks.append(
            PreflightCheck(
                "prithvi_terratorch_name",
                "fail",
                "Use terratorch_prithvi_eo_v2_300 for current TerraTorch or the official HF model id as a compatibility fallback.",
            )
        )
    checks.append(
        PreflightCheck(
            "prithvi_loading",
            "warn" if allow_hf else "fail",
            "allow_hf_download is enabled; the first run may download official Prithvi weights." if allow_hf else "Set allow_hf_download: true for official Prithvi loading.",
        )
    )


def _append_prithvi_shape_checks(checks: list[PreflightCheck], config: dict[str, Any]) -> None:
    ok = config.get("expected_frames") == 4 and config.get("expected_bands") == 6 and config.get("image_size") == 224
    checks.append(
        PreflightCheck(
            "prithvi_shape",
            "pass" if ok else "fail",
            "Prithvi-EO-2.0-300M expects expected_frames=4, expected_bands=6, image_size=224.",
        )
    )


def checks_to_json(checks: list[PreflightCheck]) -> str:
    return json.dumps([asdict(check) for check in checks], indent=2)
