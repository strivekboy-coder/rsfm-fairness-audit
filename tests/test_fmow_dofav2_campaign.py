from __future__ import annotations

from dataclasses import replace
import ast
import inspect

import pytest

from rsfm_fairness_audit import alphaearth_geobwer_campaign, cli
from rsfm_fairness_audit import fmow_dofav2_campaign, fmow_resnet50_campaign
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.bwer_schema import (
    class_mapping_hash,
    validate_formal_audit_rows,
)
from rsfm_fairness_audit.fmow_dofav2_campaign import (
    FmowDOFAv2CampaignError,
    _FROZEN_DOFA_CONFIG,
    _copy_rows_with_protocol_hash,
    _fmow_formal_metadata_preflight,
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


def test_base_audit_rows_calls_do_not_receive_spatial_extension_config() -> None:
    targets = (
        (fmow_dofav2_campaign, "audit_rows"),
        (fmow_resnet50_campaign, "audit_rows"),
        (alphaearth_geobwer_campaign, "audit_rows"),
        (cli, "geobwer_audit_rows"),
    )
    for module, function_name in targets:
        tree = ast.parse(inspect.getsource(module))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function_name
        ]
        assert calls
        assert all(
            "spatial_conformal_config"
            not in {keyword.arg for keyword in call.keywords}
            for call in calls
        )


def test_partial_protocol_rows_are_copied_and_retagged_without_mutation() -> None:
    strict_protocol = BWERProtocol(inference_method="none")
    partial_protocol = replace(strict_protocol, missingness_rule="partial_bounds")
    strict_rows = [
        {
            "dataset": "fmow",
            "model": "dofa2",
            "task": "classification",
            "split": "test",
            "sample_id": "image-1",
            "independent_unit_id": "image-1",
            "risk": 0.0,
            "split_role": "test",
            "model_signature": "model",
            "dataset_signature": "dataset",
            "protocol_hash": strict_protocol.signature,
            "metric_version": strict_protocol.metric_version,
            "probability_vector": (0.9, 0.1),
            "class_mapping_hash": class_mapping_hash(("airport", "port")),
        }
    ]
    copied = _copy_rows_with_protocol_hash(strict_rows, partial_protocol)
    assert copied is not strict_rows
    assert copied[0] is not strict_rows[0]
    assert strict_rows[0]["protocol_hash"] == strict_protocol.signature
    assert copied[0]["protocol_hash"] == partial_protocol.signature

    wrong = validate_formal_audit_rows(
        strict_rows,
        task_adapter="multiclass",
        expected_protocol_hash=partial_protocol.signature,
        expected_metric_version=partial_protocol.metric_version,
    )
    assert not wrong.ok
    assert any("Expected protocol_hash" in error for error in wrong.errors)
    corrected = validate_formal_audit_rows(
        copied,
        task_adapter="multiclass",
        expected_protocol_hash=partial_protocol.signature,
        expected_metric_version=partial_protocol.metric_version,
    )
    assert corrected.ok


def test_fmow_formal_metadata_preflight_reports_split_and_country_issues() -> None:
    rows = [
        {
            **_row("train-1", "location-1"),
            "site_id": "airport|location-1",
            "split": "train",
            "continent": "Europe",
        },
        {
            **_row("cal-1", "location-2"),
            "site_id": "airport|location-2",
            "split": "calibration",
            "country": "CA-",
            "region": "",
            "un_region": "",
            "continent": "",
        },
        {
            **_row("test-1", "location-3"),
            "site_id": "airport|location-3",
            "split": "test",
            "country": "ambiguous_country",
            "continent": "ambiguous_country",
        },
    ]
    report = _fmow_formal_metadata_preflight(
        rows,
        expected_splits=("train", "calibration", "test"),
    )
    assert report["ok"] is False
    assert report["region_fallback_all_missing"] == 1
    assert report["region_fallback_all_missing_by_split"]["calibration"] == 1
    assert {item["value"] for item in report["country_issues"]} == {
        "CA-",
        "ambiguous_country",
    }
    assert report["automatic_mapping_applied"] is False
    assert report["rows_dropped"] == 0
