from __future__ import annotations

import pytest

from rsfm_fairness_audit.fmow_dofav2_campaign import (
    FmowDOFAv2CampaignError,
    _FROZEN_DOFA_CONFIG,
    _formal_rows,
    _validate_frozen_model_config,
    _validate_split_contract,
)


def _row(sample_id: str, location_id: str, *, category: str = "airport") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "location_id": location_id,
        "category": category,
        "country": "DEU",
        "region": "Europe",
    }


def test_fmow_split_contract_requires_location_disjointness() -> None:
    with pytest.raises(FmowDOFAv2CampaignError, match="site leakage"):
        _validate_split_contract(
            [_row("train-1", "shared-location")],
            [_row("cal-1", "cal-location")],
            [_row("test-1", "shared-location")],
        )


def test_fmow_split_contract_requires_sample_disjointness() -> None:
    with pytest.raises(FmowDOFAv2CampaignError, match="sample IDs"):
        _validate_split_contract(
            [_row("shared-sample", "train-location")],
            [_row("cal-1", "cal-location")],
            [_row("shared-sample", "test-location")],
        )


def test_fmow_formal_unit_is_image_and_location_is_cluster() -> None:
    [formal] = _formal_rows([_row("image-1", "location-1")])
    assert formal["independent_unit_id"] == "image-1"
    assert formal["location_id"] == "location-1"
    assert formal["site_id"] == "airport|location-1"
    assert formal["class_label"] == "airport"
    assert formal["country_class"] == "DEU|airport"


def test_same_numeric_location_in_different_categories_is_not_leakage() -> None:
    _validate_split_contract(
        [_row("train-1", "7", category="airport")],
        [_row("cal-1", "7", category="port")],
        [_row("test-1", "8", category="airport")],
    )


def test_frozen_dofav2_config_accepts_only_pinned_protocol() -> None:
    values = {**_FROZEN_DOFA_CONFIG, "minimum_checkpoint_key_coverage": 0.90}
    _validate_frozen_model_config(values)

    values["embedding_pooling"] = "flatten"
    with pytest.raises(FmowDOFAv2CampaignError, match="embedding_pooling"):
        _validate_frozen_model_config(values)


def test_frozen_dofav2_config_rejects_weaker_checkpoint_loading() -> None:
    values = {**_FROZEN_DOFA_CONFIG, "minimum_checkpoint_key_coverage": 0.80}
    with pytest.raises(FmowDOFAv2CampaignError, match="minimum_checkpoint_key_coverage"):
        _validate_frozen_model_config(values)
