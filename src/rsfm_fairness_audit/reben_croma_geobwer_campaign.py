from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from rsfm_fairness_audit.adapters.croma import CROMAAdapter
from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.persistent_cache import persist_output
from rsfm_fairness_audit.reben_terramind_campaign import (
    _dataset,
    _sample_id_hash,
    run_reben_frozen_adapter_campaign,
)


class RebenCROMAError(RuntimeError):
    """Raised when the frozen CROMA/reBEN formal contract is incomplete."""


def validate_croma_assets(repo_path: str | Path, checkpoint_path: str | Path) -> dict[str, str]:
    repo = Path(repo_path)
    checkpoint = Path(checkpoint_path)
    if not (repo / "use_croma.py").is_file():
        raise RebenCROMAError(f"Official CROMA use_croma.py is missing under {repo}.")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=False, capture_output=True, text=True
    )
    revision = result.stdout.strip().lower()
    if result.returncode != 0 or revision != CROMAAdapter.official_repo_revision:
        raise RebenCROMAError(
            f"CROMA repo revision mismatch: expected={CROMAAdapter.official_repo_revision}, "
            f"observed={revision or 'unavailable'}."
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "use_croma.py"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise RebenCROMAError("Pinned CROMA use_croma.py has tracked local modifications.")
    if not checkpoint.is_file():
        raise RebenCROMAError(f"CROMA checkpoint is missing: {checkpoint}")
    digest = file_sha256(checkpoint)
    if digest != CROMAAdapter.official_base_sha256:
        raise RebenCROMAError(
            f"CROMA checkpoint SHA-256 mismatch: expected={CROMAAdapter.official_base_sha256}, observed={digest}."
        )
    return {"repo_revision": revision, "checkpoint_sha256": digest}


@dataclass(frozen=True)
class RebenCROMAConfig:
    lmdb_root: Path
    metadata_parquet: Path
    output_dir: Path
    sensor_mode: str
    croma_checkpoint_path: Path
    croma_repo_path: Path
    normalization_stats_path: Path
    persistent_output_dir: Path | None = None
    metadata_snow_cloud_parquet: Path | None = None
    geobwer_protocol: Path = Path("configs/geobwer/reben.yaml")
    device: str = "auto"
    batch_size: int = 64
    embedding_chunk_size: int = 4096
    probe_epochs: int = 100
    probe_learning_rate: float = 1e-2
    probe_weight_decay: float = 1e-4
    probe_batch_size: int = 512
    seed: int = 42
    max_samples: int | None = None
    n_bootstrap: int = 2000
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        if self.sensor_mode not in {"S1", "S2", "S1+S2"}:
            raise ValueError("sensor_mode must be S1, S2, or S1+S2.")
        if min(self.batch_size, self.embedding_chunk_size, self.probe_batch_size, self.probe_epochs) <= 0:
            raise ValueError("Batch/chunk sizes and probe_epochs must be positive.")
        if self.max_samples is not None and not self.diagnostic_only:
            raise ValueError("max_samples is allowed only in diagnostic_only mode.")
        if self.diagnostic_only and self.max_samples is None:
            raise ValueError("diagnostic_only requires max_samples.")


def _stats_contract(config: RebenCROMAConfig, metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metadata_parquet_sha256": file_sha256(config.metadata_parquet),
        "train_sample_id_hash": _sample_id_hash(metadata_rows),
        "train_sample_count": len(metadata_rows),
        "max_samples": config.max_samples,
        "diagnostic_only": config.diagnostic_only,
    }


def calibrate_reben_croma_train_normalization(
    config: RebenCROMAConfig,
    output_path: str | Path | None = None,
) -> Path:
    """Estimate CROMA channel scaling on train pixels only, once.

    The public CROMA example estimates channel moments from its input tensor.
    A fixed train-only estimate preserves that convention without allowing test
    leakage or making an embedding depend on inference batch composition.
    """

    output = Path(output_path or config.normalization_stats_path)
    paired_config = replace(config, sensor_mode="S1+S2")
    dataset = _dataset(paired_config, "train")
    metadata_rows = [dict(row) for row in dataset.load_metadata()]
    contract = _stats_contract(config, metadata_rows)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("contract") != contract:
            raise RebenCROMAError(
                f"CROMA normalization contract changed under {output}; use a new formal output directory."
            )
        if existing.get("selection_split") != "train" or existing.get("test_rows_used") is not False:
            raise RebenCROMAError("Existing CROMA normalization statistics do not prove train-only selection.")
        return output

    sums = {"S1": np.zeros(2, dtype=np.float64), "S2": np.zeros(12, dtype=np.float64)}
    sums_of_squares = {"S1": np.zeros(2, dtype=np.float64), "S2": np.zeros(12, dtype=np.float64)}
    pixel_counts = {"S1": 0, "S2": 0}
    for index in range(len(metadata_rows)):
        sample = dataset.load_sample(index)
        images = sample.get("image")
        if not isinstance(images, dict) or "S1" not in images or "S2" not in images:
            raise RebenCROMAError("CROMA train normalization requires paired raw S1/S2 arrays.")
        for sensor, channels in (("S1", 2), ("S2", 12)):
            array = np.asarray(images[sensor], dtype=np.float64)
            if array.ndim != 3 or array.shape[0] != channels or not np.all(np.isfinite(array)):
                raise RebenCROMAError(
                    f"Invalid {sensor} train sample shape/values at index={index}: {array.shape}."
                )
            flattened = array.reshape(channels, -1)
            sums[sensor] += flattened.sum(axis=1)
            sums_of_squares[sensor] += np.square(flattened).sum(axis=1)
            pixel_counts[sensor] += flattened.shape[1]
        if (index + 1) % 2500 == 0 or index + 1 == len(metadata_rows):
            print(f"[reben:croma:normalization] train samples={index + 1}/{len(metadata_rows)}", flush=True)

    stats: dict[str, dict[str, Any]] = {}
    for sensor in ("S1", "S2"):
        count = pixel_counts[sensor]
        mean = sums[sensor] / count
        variance = np.maximum(sums_of_squares[sensor] / count - np.square(mean), 1e-12)
        std = np.sqrt(variance)
        stats[sensor] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "pixel_count_per_channel": count,
        }
    payload = {
        "schema": "geobwer.reben.croma_train_normalization.v1",
        "selection_split": "train",
        "test_rows_used": False,
        "policy": "train_split_fixed_channel_mean_plus_minus_2std_then_uint8_and_float_0_1",
        "contract": contract,
        "stats": stats,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _load_stats(
    path: Path,
    config: RebenCROMAConfig,
) -> dict[str, dict[str, tuple[float, ...]]]:
    if not path.is_file():
        raise RebenCROMAError(
            f"Missing train-only CROMA normalization statistics: {path}. Run calibration before inference."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "geobwer.reben.croma_train_normalization.v1":
        raise RebenCROMAError("Unrecognized CROMA normalization schema.")
    if payload.get("selection_split") != "train" or payload.get("test_rows_used") is not False:
        raise RebenCROMAError("CROMA normalization must be frozen on train data only.")
    metadata_rows = [dict(row) for row in _dataset(replace(config, sensor_mode="S1+S2"), "train").load_metadata()]
    if payload.get("contract") != _stats_contract(config, metadata_rows):
        raise RebenCROMAError("CROMA normalization statistics do not match this train split/data contract.")
    return {
        sensor: {
            "mean": tuple(float(value) for value in payload["stats"][sensor]["mean"]),
            "std": tuple(float(value) for value in payload["stats"][sensor]["std"]),
        }
        for sensor in ("S1", "S2")
    }


def run_reben_croma_campaign(config: RebenCROMAConfig) -> dict[str, Path]:
    stats = _load_stats(config.normalization_stats_path, config)
    mode = config.sensor_mode
    modality = {"S1": "SAR", "S2": "optical", "S1+S2": "both"}[mode]
    embedding_key = {"S1": "SAR_GAP", "S2": "optical_GAP", "S1+S2": "joint_GAP"}[mode]
    adapter = CROMAAdapter(
        model_size="base",
        checkpoint_path=config.croma_checkpoint_path,
        repo_path=config.croma_repo_path,
        device=config.device,
        batch_size=config.batch_size,
        input_modality=modality,
        expected_s1_bands=2,
        expected_s2_bands=12,
        image_size=120,
        embedding_key=embedding_key,
        preprocessing="train_split_fixed_channel_2sigma_uint8_0_1",
        normalization_stats=stats,
        normalization_stats_source=config.normalization_stats_path,
        strict_reproducibility=True,
    )
    slug = mode.lower().replace("+", "_plus_")
    artifacts = run_reben_frozen_adapter_campaign(
        config,
        adapter=adapter,
        model_name=f"croma_base_{slug}",
        campaign_schema="geobwer.reben.croma_campaign.v1",
    )
    persist_output(config.output_dir, config.persistent_output_dir, label=f"croma-{mode}-complete")
    return artifacts


__all__ = [
    "RebenCROMAConfig",
    "RebenCROMAError",
    "calibrate_reben_croma_train_normalization",
    "run_reben_croma_campaign",
    "validate_croma_assets",
]
