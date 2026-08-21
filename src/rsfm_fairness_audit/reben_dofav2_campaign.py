from __future__ import annotations

"""DOFAv2 x reBEN missing cell for the Experiment 9 pipeline matrix."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsfm_fairness_audit.adapters.dofa import DOFAAdapter
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.fmow_dofav2_campaign import _validate_frozen_model_config
from rsfm_fairness_audit.reben_terramind_campaign import run_reben_frozen_adapter_campaign


@dataclass(frozen=True)
class RebenDOFAv2Config:
    lmdb_root: Path
    metadata_parquet: Path
    output_dir: Path
    model_config: Path
    dofa_repo_path: Path
    dofa_checkpoint_path: Path
    persistent_output_dir: Path | None = None
    embedding_cache_root: Path | None = None
    persistent_embedding_cache_root: Path | None = None
    metadata_snow_cloud_parquet: Path | None = None
    geobwer_protocol: Path = Path("configs/geobwer/reben.yaml")
    device: str = "auto"
    batch_size: int = 32
    embedding_chunk_size: int = 4096
    probe_epochs: int = 100
    probe_learning_rate: float = 1e-2
    probe_weight_decay: float = 1e-4
    probe_batch_size: int = 512
    seed: int = 42
    max_samples: int | None = None
    n_bootstrap: int = 2000
    diagnostic_only: bool = False
    sensor_mode: str = "S2"
    channel_profile: str = "dofav2_s2_9"

    def __post_init__(self) -> None:
        if self.sensor_mode != "S2" or self.channel_profile != "dofav2_s2_9":
            raise ValueError("Experiment 9 DOFAv2 x reBEN is frozen to S2 and the official 9-band DOFA order.")
        if self.max_samples is not None and not self.diagnostic_only:
            raise ValueError("Subsampling is diagnostic-only.")
        if self.diagnostic_only and self.max_samples is None:
            raise ValueError("diagnostic_only requires max_samples.")


def run_reben_dofav2_campaign(config: RebenDOFAv2Config) -> dict[str, Path]:
    values: dict[str, Any] = dict(load_yaml(config.model_config))
    values.update({
        "device": config.device,
        "batch_size": config.batch_size,
        "repo_path": str(config.dofa_repo_path),
        "checkpoint_path": str(config.dofa_checkpoint_path),
        "input_modality": "S2",
        "band_profile": "sentinel2_9_legacy",
        "image_size": 224,
    })
    _validate_frozen_model_config(values)
    adapter = DOFAAdapter.from_config(values)
    if not str(adapter.model_release).startswith("dofav2"):
        raise ValueError("Experiment 9 requires the frozen DOFAv2 release, not legacy DOFA.")
    return run_reben_frozen_adapter_campaign(
        config,
        adapter=adapter,
        model_name=f"dofav2_vit_base_s2_seed_{config.seed}",
        campaign_schema="geobwer.reben.dofav2_campaign.v1",
        adaptation_protocol="frozen_dofav2_encoder_train_only_linear_multilabel_probe",
    )


__all__ = ["RebenDOFAv2Config", "run_reben_dofav2_campaign"]
