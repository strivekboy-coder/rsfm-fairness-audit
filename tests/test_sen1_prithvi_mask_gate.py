from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import scripts.prepare_sen1floods11_subset as prep
import rsfm_fairness_audit.prithvi_sen1_campaign as campaign
from rsfm_fairness_audit.prithvi_sen1_campaign import PrithviSen1CampaignConfig
from rsfm_fairness_audit.sen1_prithvi_mask_gate import (
    Sen1PrithviMaskGateError,
    gate_prithvi_prepared_masks,
    read_mask_npz,
    validate_prepared_mask,
    write_verified_mask_npz,
)


def test_illegal_high_mask_value_is_rejected() -> None:
    mask = np.zeros((224, 224), dtype=np.int16)
    mask[0, 0] = 255
    with pytest.raises(Sen1PrithviMaskGateError, match="subset"):
        validate_prepared_mask(mask, stage="illegal_high")


def test_label_nearest_resize_preserves_discrete_values(monkeypatch) -> None:
    source = np.asarray(
        [[-1, -1, 0], [-1, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    monkeypatch.setattr(prep, "_read_tif", lambda _path: source[None])
    result = prep._mask_from_label(Path("Bolivia_LabelHand.tif"), target_size=224)
    assert result.shape == (224, 224)
    assert result.dtype == np.int16
    assert set(np.unique(result).tolist()) == {-1, 0, 1}


def test_verified_mask_npz_round_trip_and_explicit_key(tmp_path: Path) -> None:
    mask = np.zeros((224, 224), dtype=np.int16)
    mask[:5, :5] = -1
    mask[10:20, 10:20] = 1
    path = tmp_path / "mask.npz"
    write_verified_mask_npz(path, mask)
    assert np.array_equal(read_mask_npz(path), mask)
    with np.load(path, allow_pickle=False) as bundle:
        assert "mask" in bundle.files
    with pytest.raises(Sen1PrithviMaskGateError, match="overwrite"):
        write_verified_mask_npz(path, mask)


def test_missing_mask_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong_key.npz"
    np.savez_compressed(path, image=np.zeros((224, 224), dtype=np.int16))
    with pytest.raises(Sen1PrithviMaskGateError, match="explicit 'mask' key"):
        read_mask_npz(path)


def _prepared_root(root: Path, prefixes: list[str]) -> Path:
    masks = root / "masks"
    masks.mkdir(parents=True)
    rows = []
    template = np.zeros((224, 224), dtype=np.int16)
    template[0, 0] = -1
    template[1, 1] = 1
    for index, prefix in enumerate(prefixes):
        path = masks / f"{prefix}.npz"
        np.savez_compressed(path, mask=template)
        rows.append(
            {
                "sample_id": prefix,
                "mask_path": path.relative_to(root).as_posix(),
            }
        )
    metadata = root / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "mask_path"])
        writer.writeheader()
        writer.writerows(rows)
    return metadata


def test_real_431_plus_15_mask_gate_contract(tmp_path: Path) -> None:
    core_prefixes = [
        f"Event{index % 10:02d}_{index:04d}" for index in range(431)
    ]
    bolivia_prefixes = [f"Bolivia_{index:04d}" for index in range(15)]
    core = tmp_path / "core"
    bolivia = tmp_path / "bolivia"
    core_metadata = _prepared_root(core, core_prefixes)
    bolivia_metadata = _prepared_root(bolivia, bolivia_prefixes)
    report = gate_prithvi_prepared_masks(
        core_root=core,
        core_metadata=core_metadata,
        bolivia_root=bolivia,
        bolivia_metadata=bolivia_metadata,
    )
    assert report["status"] == "pass"
    assert report["core"]["sample_count"] == 431
    assert report["bolivia"]["sample_count"] == 15
    assert report["combined_sample_count"] == 446
    assert report["model_loaded_by_gate"] is False


def test_prepare_refuses_existing_output_before_conversion(tmp_path: Path) -> None:
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        prep.prepare_sen1floods11_subset(
            output_dir=output,
            source_root=tmp_path / "unused",
            max_samples=1,
        )


def test_mask_gate_failure_occurs_before_prithvi_model_load(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeDataset:
        def __init__(self, *_args, **_kwargs):
            pass

        def load_metadata(self):
            return [{"sample_id": "placeholder"}]

    model_load_reached = False

    def forbidden_model_factory(_values):
        nonlocal model_load_reached
        model_load_reached = True
        raise AssertionError("model construction must not be reached")

    monkeypatch.setattr(campaign, "hydrate_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(campaign, "Sen1Floods11DatasetAdapter", FakeDataset)
    monkeypatch.setattr(
        campaign,
        "gate_prithvi_prepared_masks",
        lambda **_kwargs: (_ for _ in ()).throw(
            Sen1PrithviMaskGateError("corrupt prepared mask")
        ),
    )
    monkeypatch.setattr(
        campaign.PrithviSen1Floods11TLAdapter,
        "from_config",
        forbidden_model_factory,
    )
    config = PrithviSen1CampaignConfig(
        prepared_data_root=tmp_path / "core",
        bolivia_prepared_data_root=tmp_path / "bolivia",
        model_config=tmp_path / "model.yaml",
        train_split=tmp_path / "train.txt",
        validation_split=tmp_path / "validation.txt",
        test_split=tmp_path / "test.txt",
        bolivia_split=tmp_path / "bolivia.txt",
        output_dir=tmp_path / "output",
    )
    with pytest.raises(Sen1PrithviMaskGateError, match="corrupt"):
        campaign.run_prithvi_sen1_probability_campaign(config)
    assert model_load_reached is False
