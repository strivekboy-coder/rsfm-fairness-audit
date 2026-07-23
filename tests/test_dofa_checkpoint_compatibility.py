from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rsfm_fairness_audit.adapters.dofa import DOFAAdapter, DOFAConfigurationError
from scripts.diagnose_dofa_checkpoint_compatibility import (
    compare_state_dicts,
    extract_state_dict,
)


class FakeTensor:
    def __init__(self, *shape: int) -> None:
        self.shape = shape

    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result


def test_checkpoint_diagnostic_reports_each_mismatch_category() -> None:
    model = {
        "matched": FakeTensor(2, 3),
        "missing": FakeTensor(5),
        "wrong_shape": FakeTensor(4, 4),
    }
    checkpoint = {
        "matched": FakeTensor(2, 3),
        "wrong_shape": FakeTensor(8, 2),
        "extra": FakeTensor(7),
    }
    report = compare_state_dicts(model, checkpoint)

    assert report["counts"] == {
        "model_keys": 3,
        "checkpoint_keys": 3,
        "matched_keys": 1,
        "model_keys_missing_from_checkpoint": 1,
        "checkpoint_keys_missing_from_model": 1,
        "same_name_shape_mismatches": 1,
    }
    assert report["parameters"]["model_total_numel"] == 27
    assert report["parameters"]["matched_model_numel"] == 6
    assert report["model_keys_missing_from_checkpoint"] == ["missing"]
    assert report["checkpoint_keys_missing_from_model"] == ["extra"]
    assert report["same_name_shape_mismatches"][0]["key"] == "wrong_shape"


def test_checkpoint_diagnostic_records_selected_container_and_prefix_normalisation() -> None:
    state, container = extract_state_dict({"model": {"module.backbone.weight": FakeTensor(2)}})
    assert container == "model"
    assert list(state) == ["weight"]


def test_dofav2_config_freezes_exact_architecture_contract() -> None:
    adapter = DOFAAdapter.from_config_file("configs/models/dofav2_fmow_sentinel.yaml")
    assert adapter.model_variant == "dofav2_vit_base"
    assert adapter.architecture_source_repo == DOFAAdapter.official_dofav2_architecture_repo
    assert adapter.architecture_source_revision == DOFAAdapter.official_dofav2_architecture_revision
    assert adapter.required_timm_version == "1.0.15"
    assert adapter.require_exact_checkpoint_match is True
    assert adapter.image_size == 224
    assert adapter.embedding_pooling == "mean_tokens"


def test_dofav2_runtime_rejects_unfrozen_timm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "timm", SimpleNamespace(__version__="0.9.2"))
    adapter = DOFAAdapter(
        model_variant="dofav2_vit_base",
        model_release="dofav2_vit_base_e150",
        required_timm_version="1.0.15",
    )
    with pytest.raises(DOFAConfigurationError, match="requires timm==1.0.15"):
        adapter._verify_dofav2_runtime()


def test_official_dofav2_checkpoint_loads_exactly_through_adapter_when_assets_are_available() -> None:
    repo = os.environ.get("DOFAV2_OFFICIAL_REPO")
    checkpoint = os.environ.get("DOFAV2_OFFICIAL_CHECKPOINT")
    if not repo or not checkpoint:
        pytest.skip("Set DOFAV2_OFFICIAL_REPO and DOFAV2_OFFICIAL_CHECKPOINT for the real compatibility test.")
    pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    assert timm.__version__ == "1.0.15"

    adapter = DOFAAdapter(
        repo_path=Path(repo),
        checkpoint_path=Path(checkpoint),
        model_variant="dofav2_vit_base",
        model_release="dofav2_vit_base_e150",
        repo_revision=DOFAAdapter.official_dofav2_repo_revision,
        checkpoint_sha256=DOFAAdapter.official_dofav2_checkpoint_sha256,
        architecture_source_repo=DOFAAdapter.official_dofav2_architecture_repo,
        architecture_source_revision=DOFAAdapter.official_dofav2_architecture_revision,
        required_timm_version=DOFAAdapter.official_dofav2_timm_version,
        minimum_checkpoint_key_coverage=0.90,
        require_exact_checkpoint_match=True,
        image_size=224,
        embedding_layer="forward_features",
        embedding_pooling="mean_tokens",
        device="cpu",
    )
    adapter.load_model()
    report = adapter.checkpoint_load_report
    assert report["model_keys"] == 194
    assert report["checkpoint_keys"] == 194
    assert report["matched_keys"] == 194
    assert report["model_keys_missing_from_checkpoint"] == []
    assert report["checkpoint_keys_missing_from_model"] == []
    assert report["same_name_shape_mismatches"] == []
    assert report["model_parameter_numel"] == 105_432_320
    assert report["matched_parameter_numel"] == 105_432_320
    assert report["parameter_coverage"] == 1.0
    assert report["load_missing_keys"] == []
    assert report["load_unexpected_keys"] == []
    assert adapter.model.patch_embed.kernel_size == 14
    assert tuple(adapter.model.pos_embed.shape) == (1, 257, 768)
    embeddings = adapter.extract_embeddings(
        {
            "images": np.zeros((1, 9, 224, 224), dtype=np.float32),
            "wavelengths": DOFAAdapter.verified_wavelengths["S2_OFFICIAL_DEMO_9CH"],
        }
    )
    assert embeddings.shape == (1, 768)
    assert np.all(np.isfinite(embeddings))
