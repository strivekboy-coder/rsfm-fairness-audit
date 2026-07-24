from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

import rsfm_fairness_audit.reben_resnet50_campaign as campaign
from rsfm_fairness_audit.reben_resnet50_campaign import (
    RebenDiagnosticSupportError,
    RebenResNet50Config,
    run_reben_resnet50_campaign,
)
from rsfm_fairness_audit.reben_sensor_audit import default_reben_class_names


def _rows(count: int = 96, *, groups: int = 8) -> list[dict[str, Any]]:
    classes = default_reben_class_names()
    return [
        {
            "sample_id": f"sample-{index:04d}",
            "patch_id": f"S2A_TEST_T{index % groups:02d}ABC_{index:02d}_00",
            "source_tile_id": f"T{index % groups:02d}ABC",
            "country": f"C{index % 4}",
            "labels": [
                classes[index % 7],
                classes[(index * 3 + 1) % 11],
            ],
        }
        for index in range(count)
    ]


class _FakeRawAdapter:
    seen_max_samples: list[int | None] = []

    def __init__(
        self,
        _lmdb_root: Path,
        _metadata_parquet: Path,
        _snow: Path | None,
        *,
        split: str,
        sensor_mode: str,
        max_samples: int | None,
        channel_profile: str,
    ) -> None:
        self.split = split
        self.sensor_mode = sensor_mode
        self.channel_profile = channel_profile
        self.rows = _rows()
        self.seen_max_samples.append(max_samples)

    def load_metadata(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows]

    def load_sample(self, index: int) -> Mapping[str, Any]:
        row = self.rows[index]
        channels = campaign.MODE_CHANNELS[self.sensor_mode]
        image = np.zeros((channels, 4, 4), dtype=np.float32)
        label_vector = campaign._row_label_vector(row)
        return {
            "image": image,
            "metadata": {**row, "label_vector": label_vector.tolist()},
        }

    def loader_info(self) -> dict[str, Any]:
        return {"sensor_mode": self.sensor_mode}


@pytest.mark.parametrize("mode", campaign.SENSOR_MODES)
def test_diagnostic_32_sampling_preserves_groups_labels_and_mode_identity(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _FakeRawAdapter.seen_max_samples.clear()
    monkeypatch.setattr(
        campaign,
        "resolve_reben_root_dir",
        lambda root: (Path(root), Path(root) / "BigEarthNetEncoded.lmdb", {}),
    )
    monkeypatch.setattr(campaign, "detect_lmdb_payload_format", lambda _path: "safetensors")
    monkeypatch.setattr(campaign, "LmdbSafetensorsRebenDatasetAdapter", _FakeRawAdapter)
    config = RebenResNet50Config(
        lmdb_root=Path("demo"),
        metadata_parquet=Path("metadata.parquet"),
        output_dir=Path("unused"),
        sensor_modes=(mode,),
        seeds=(42,),
        diagnostic_max_samples=32,
    )
    adapter = campaign._dataset(config, "train", mode)
    selected = adapter.load_metadata()
    assert len(selected) == 32
    assert len({row["source_tile_id"] for row in selected}) >= 2
    covered = np.any(
        np.stack([campaign._row_label_vector(row) for row in selected]), axis=0
    )
    assert int(np.sum(covered)) >= 2
    assert adapter.diagnostic_sampling["formal_evidence"] is False
    assert adapter.diagnostic_sampling["status"] == "ready"
    assert _FakeRawAdapter.seen_max_samples == [None]


def _selected_adapter(split: str) -> Any:
    raw = _FakeRawAdapter(
        Path("demo"),
        Path("metadata.parquet"),
        None,
        split=split,
        sensor_mode="S1",
        max_samples=None,
        channel_profile="croma",
    )
    indices, diagnostics = campaign.select_reben_diagnostic_indices(
        raw.load_metadata(),
        max_samples=32,
        seed=42,
        split=split,
    )
    return campaign._IndexedRebenAdapter(raw, indices, diagnostics)


def test_three_route_campaign_writes_nonformal_diagnostic_manifests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapters = {
        split: _selected_adapter(split) for split in ("train", "val", "test")
    }

    def fake_dataset(
        _config: RebenResNet50Config, split: str, mode: str
    ) -> Any:
        adapter = adapters[split]
        adapter.base.sensor_mode = mode
        return adapter

    def fake_contract(dataset: Any, *, mode: str, pixel_stride: int) -> dict[str, Any]:
        return {
            "schema": "test",
            "selection_split": "official_train",
            "test_rows_used": False,
            "sensor_mode": mode,
            "sample_count": len(dataset.load_metadata()),
            "mean": [0.0] * campaign.MODE_CHANNELS[mode],
            "std": [1.0] * campaign.MODE_CHANNELS[mode],
            "pos_weight": [1.0] * 19,
            "diagnostic_sampling": dataset.diagnostic_sampling,
        }

    def fake_train(
        _config: RebenResNet50Config,
        *,
        mode: str,
        seed: int,
        adapters: Mapping[str, Any],
        contract: Mapping[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        del contract
        output_dir.mkdir(parents=True, exist_ok=True)
        targets = np.zeros((32, 19), dtype=np.int8)
        targets[np.arange(32), np.arange(32) % 5] = 1
        probabilities = np.where(targets == 1, 0.8, 0.2).astype(np.float32)
        train_groups = sorted(
            {row["source_tile_id"] for row in adapters["train"].load_metadata()}
        )
        assert len(train_groups) >= 2
        return {
            "checkpoint": output_dir / "unused.pt",
            "best_epoch": 1,
            "best_validation_weighted_bce": 0.25,
            "history": [],
            "refit_history": [],
            "datasets": {},
            "outputs": {
                "validation": {
                    "probabilities": probabilities,
                    "targets": targets,
                },
                "test": {
                    "probabilities": probabilities,
                    "targets": targets,
                },
            },
            "inner_fit_groups": train_groups[:-1],
            "inner_selection_groups": train_groups[-1:],
            "diagnostic_sampling": adapters["train"].diagnostic_sampling,
            "diagnostic_sampling_by_split": {
                split: adapter.diagnostic_sampling
                for split, adapter in adapters.items()
            },
        }

    monkeypatch.setattr(campaign, "_dataset", fake_dataset)
    monkeypatch.setattr(campaign, "compute_reben_train_contract", fake_contract)
    monkeypatch.setattr(campaign, "_train_seed", fake_train)
    monkeypatch.setattr(campaign, "persist_output", lambda *args, **kwargs: None)
    result = run_reben_resnet50_campaign(
        RebenResNet50Config(
            lmdb_root=tmp_path / "lmdb",
            metadata_parquet=tmp_path / "metadata.parquet",
            output_dir=tmp_path / "out",
            sensor_modes=campaign.SENSOR_MODES,
            seeds=(42,),
            diagnostic_max_samples=32,
            max_epochs=1,
            patience=1,
            batch_size=4,
        )
    )
    assert len(result["runs"]) == 3
    for mode in campaign.SENSOR_MODES:
        slug = mode.lower().replace("+", "_plus_")
        manifest_path = result["runs"][f"resnet50_{slug}_seed_42"][
            "diagnostic_manifest"
        ]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["formal_evidence"] is False
        assert manifest["status"] == "completed"
        assert manifest["group_disjoint_model_selection_preserved"] is True
        assert manifest["diagnostic_sampling"]["selected_samples"] == 32
        assert manifest["diagnostic_sampling"]["selected_group_count"] >= 2
    panel = json.loads(result["campaign_manifest"].read_text(encoding="utf-8"))
    assert panel["formal_evidence"] is False


def test_insufficient_diagnostic_support_writes_manifest_instead_of_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    diagnostics = {
        "schema": "geobwer.reben.diagnostic_sampling.v1",
        "formal_evidence": False,
        "status": "insufficient_selection_groups",
        "selected_group_count": 1,
    }

    def fail_train_dataset(
        _config: RebenResNet50Config, split: str, mode: str
    ) -> Any:
        del mode
        if split == "train":
            raise RebenDiagnosticSupportError("one group only", diagnostics)
        raise AssertionError("validation/test must not be opened after failed train preflight")

    monkeypatch.setattr(campaign, "_dataset", fail_train_dataset)
    monkeypatch.setattr(campaign, "persist_output", lambda *args, **kwargs: None)
    result = run_reben_resnet50_campaign(
        RebenResNet50Config(
            lmdb_root=tmp_path / "lmdb",
            metadata_parquet=tmp_path / "metadata.parquet",
            output_dir=tmp_path / "out",
            sensor_modes=("S1",),
            seeds=(42,),
            diagnostic_max_samples=1,
            max_epochs=1,
            patience=1,
            batch_size=1,
        )
    )
    manifest_path = result["runs"]["resnet50_s1_seed_42"][
        "diagnostic_manifest"
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["formal_evidence"] is False
    assert manifest["status"] == "not_run_insufficient_diagnostic_support"
    assert manifest["formal_protocol_changed"] is False
