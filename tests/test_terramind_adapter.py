from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from rsfm_fairness_audit.adapters.terramind import (
    S1_MEAN,
    INPUT_PROFILES,
    TerraMindAdapter,
    TerraMindConfigurationError,
    checkpoint_sha256,
    validate_terramind_checkpoint,
)


class _NumpyTerraMind:
    def eval(self):
        return self

    def extract_embeddings(self, images):
        pooled = [value.mean(axis=(1, 2, 3), keepdims=False)[:, None] for value in images.values()]
        return np.mean(np.stack(pooled), axis=0) if len(pooled) > 1 else pooled[0]


def _sample(*, s1_value: float = -12.0, s2_value: float = 2000.0):
    return {
        "image": {
            "S1": np.full((2, 8, 8), s1_value, dtype=np.float32),
            "S2": np.full((12, 8, 8), s2_value, dtype=np.float32),
        },
        "metadata": {"sample_id": "a"},
    }


def test_reben_fusion_uses_official_modalities_and_fixed_dimension():
    adapter = TerraMindAdapter(
        sensor_mode="S1+S2",
        input_profile="reben_l2a",
        image_size=8,
        strict_range_check=True,
        model=_NumpyTerraMind(),
    )
    prepared = adapter.preprocess({"samples": [_sample(), _sample()], "metadata": [{}, {}]})
    assert set(prepared["images"]) == {"S1GRD", "S2L2A"}
    assert prepared["images"]["S1GRD"].shape == (2, 2, 8, 8)
    assert prepared["images"]["S2L2A"].shape == (2, 12, 8, 8)
    adapter.load_model()
    embeddings = adapter.extract_embeddings(prepared)
    assert embeddings.shape == (2, 1)
    assert adapter.provenance()["s1_unit_policy"] == "already_db"


def test_s1_linear_power_conversion_is_explicit():
    linear_power = float(10 ** (-12.0 / 10.0))
    sample = {"image": np.full((2, 8, 8), linear_power, dtype=np.float32), "metadata": {}}
    adapter = TerraMindAdapter(
        sensor_mode="S1",
        input_profile="reben_l2a",
        image_size=8,
        s1_unit_policy="linear_power_to_db",
        model=_NumpyTerraMind(),
    )
    prepared = adapter.preprocess({"samples": [sample], "metadata": [{}]})
    expected = (-12.0 - np.asarray(S1_MEAN)) / np.asarray((5.195, 5.890))
    np.testing.assert_allclose(prepared["images"]["S1GRD"][0, :, 0, 0], expected, atol=1e-5)


def test_s1_unit_mismatch_hard_fails():
    sample = {"image": np.full((2, 8, 8), 10_000.0, dtype=np.float32), "metadata": {}}
    adapter = TerraMindAdapter(
        sensor_mode="S1",
        input_profile="reben_l2a",
        image_size=8,
        s1_unit_policy="already_db",
        model=_NumpyTerraMind(),
    )
    with pytest.raises(TerraMindConfigurationError, match="source units"):
        adapter.preprocess({"samples": [sample], "metadata": [{}]})


def test_s2_dn_q999_guard_accepts_28000_and_rejects_above_int16_max():
    adapter = TerraMindAdapter(
        sensor_mode="S2",
        input_profile="reben_l2a",
        image_size=8,
        strict_range_check=True,
        model=_NumpyTerraMind(),
    )
    valid = _sample(s2_value=28_000.0)
    prepared = adapter.preprocess({"samples": [valid], "metadata": [{}]})
    assert prepared["images"]["S2L2A"].shape == (1, 12, 8, 8)

    invalid = _sample(s2_value=32_768.0)
    with pytest.raises(TerraMindConfigurationError, match="unscaled Sentinel reflectance/DN"):
        adapter.preprocess({"samples": [invalid], "metadata": [{}]})


def test_sen1floods_profile_is_official_13_band_l1c():
    profile = INPUT_PROFILES["sen1floods11_l1c"]
    assert profile.s2_modality == "S2L1C"
    assert profile.s2_channels == 13
    assert len(profile.s2_mean) == len(profile.s2_std) == 13


def test_non_mean_fusion_is_not_a_primary_protocol_option():
    with pytest.raises(TerraMindConfigurationError, match="merge_method='mean'"):
        TerraMindAdapter(sensor_mode="S2", input_profile="reben_l2a", merge_method="concat")


def test_checkpoint_identity_is_verified_before_formal_loading():
    checkpoint = Path("work/test_terramind_checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint.write_bytes(b"small-unit-test-checkpoint")
        expected = checkpoint_sha256(checkpoint)
        resolved, observed = validate_terramind_checkpoint(checkpoint, expected_sha256=expected)
        assert resolved == checkpoint
        assert observed == expected
        with pytest.raises(TerraMindConfigurationError, match="SHA-256 mismatch"):
            validate_terramind_checkpoint(checkpoint, expected_sha256="0" * 64)
    finally:
        checkpoint.unlink(missing_ok=True)
