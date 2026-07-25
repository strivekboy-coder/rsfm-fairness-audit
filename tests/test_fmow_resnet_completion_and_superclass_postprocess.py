from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.bwer_core import compute_geobwer
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.fmow_resnet50_campaign import (
    FmowResNet50CampaignConfig,
    FmowResNet50CampaignError,
    _required_seed_artifacts,
    _validate_seed_completion_contract,
    _write_seed_completion_contract,
)
from rsfm_fairness_audit.fmow_superclass_postprocess import (
    _sharp_fixed_universe_bounds,
)
from rsfm_fairness_audit.formal_outputs import file_sha256


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _completion_fixture(
    tmp_path: Path,
) -> tuple[FmowResNet50CampaignConfig, BWERProtocol, Path, str, str]:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "sample_id,split,category,site_id\n"
        "a,train,airport,airport|1\n"
        "b,calibration,airport,airport|2\n"
        "c,test,airport,airport|3\n",
        encoding="utf-8",
    )
    normalization = tmp_path / "norm.json"
    normalization.write_text('{"mean":[0],"std":[1]}', encoding="utf-8")
    run_dir = tmp_path / "seed_42"
    artifacts = _required_seed_artifacts(run_dir)
    for path in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    artifacts["checkpoint"].write_bytes(b"checkpoint")
    probabilities = np.asarray([[0.8, 0.2]], dtype=np.float32)
    targets = np.asarray([0], dtype=np.int64)
    np.savez_compressed(
        artifacts["calibration_probabilities"],
        probabilities=probabilities,
        targets=targets,
    )
    np.savez_compressed(
        artifacts["formal_probabilities"],
        probabilities=probabilities,
        targets=targets,
    )
    artifacts["class_mapping"].write_text(
        '{"classes":["airport","port"]}', encoding="utf-8"
    )
    _write_rows(
        artifacts["formal_audit_table"],
        [{"sample_id": "c", "risk": 0.0}],
    )
    protocol = BWERProtocol(inference_method="none")
    geography_hash = "geo-contract-hash"
    artifacts["formal_output_manifest"].write_text(
        json.dumps(
            {
                "row_count": 1,
                "protocol_hash": protocol.signature,
                "protocol": {"metric_version": protocol.metric_version},
                "dataset_lineage": {
                    "metadata_sha256": file_sha256(metadata),
                    "geography_contract_hash": geography_hash,
                },
                "model_lineage": {
                    "seed": 42,
                    "band_profile": "sentinel2_9_legacy",
                },
                "artifacts": {
                    "probability_sha256": file_sha256(
                        artifacts["formal_probabilities"]
                    ),
                    "class_mapping_sha256": file_sha256(
                        artifacts["class_mapping"]
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    artifacts["calibration_manifest"].write_text(
        json.dumps(
            {
                "split_role": "calibration",
                "test_rows_used": False,
                "probabilities_sha256": file_sha256(
                    artifacts["calibration_probabilities"]
                ),
            }
        ),
        encoding="utf-8",
    )
    artifacts["run_manifest"].write_text(
        json.dumps(
            {
                "training": {
                    "selected_epoch": 2,
                    "inner_validation_cross_entropy": 0.5,
                    "calibration_metrics": {"accuracy": 0.5},
                    "test_metrics": {"accuracy": 0.5},
                }
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "raw_summary",
        "strict_standardized_summary",
        "partial_standardized_summary",
    ):
        _write_rows(
            artifacts[name],
            [
                {
                    "axis": "country",
                    "bwer": 0.1,
                    "validity": "descriptive_only",
                    "ci_low": 0.0,
                    "ci_high": 0.9,
                    "lower_confidence_bound": 0.0,
                }
            ],
        )
    _write_rows(
        artifacts["uncertainty_summary"],
        [{"extension": "conformal", "value": 0.1}],
    )
    config = FmowResNet50CampaignConfig(
        metadata_csv=metadata,
        data_root=tmp_path,
        output_dir=tmp_path / "out",
        geography_contract=tmp_path / "geo.json",
    )
    return (
        config,
        protocol,
        run_dir,
        geography_hash,
        file_sha256(normalization),
    )


def test_fmow_completion_contract_skips_only_exact_complete_seed(
    tmp_path: Path,
) -> None:
    config, protocol, run_dir, geography_hash, normalization_sha = (
        _completion_fixture(tmp_path)
    )
    completion = _write_seed_completion_contract(
        config,
        seed=42,
        protocol=protocol,
        geography_contract_hash=geography_hash,
        normalization_sha256=normalization_sha,
        run_dir=run_dir,
    )
    assert completion.is_file()
    artifacts = _validate_seed_completion_contract(
        config,
        seed=42,
        protocol=protocol,
        geography_contract_hash=geography_hash,
        normalization_sha256=normalization_sha,
        run_dir=run_dir,
    )
    assert artifacts is not None
    assert artifacts["completion_contract"] == completion

    with pytest.raises(FmowResNet50CampaignError, match="does not match"):
        _validate_seed_completion_contract(
            replace(config, max_epochs=99),
            seed=42,
            protocol=protocol,
            geography_contract_hash=geography_hash,
            normalization_sha256=normalization_sha,
            run_dir=run_dir,
        )


def test_fmow_completion_contract_rejects_partial_and_changed_artifact(
    tmp_path: Path,
) -> None:
    config, protocol, run_dir, geography_hash, normalization_sha = (
        _completion_fixture(tmp_path)
    )
    with pytest.raises(FmowResNet50CampaignError, match="partial artifacts"):
        _validate_seed_completion_contract(
            config,
            seed=42,
            protocol=protocol,
            geography_contract_hash=geography_hash,
            normalization_sha256=normalization_sha,
            run_dir=run_dir,
        )
    _write_seed_completion_contract(
        config,
        seed=42,
        protocol=protocol,
        geography_contract_hash=geography_hash,
        normalization_sha256=normalization_sha,
        run_dir=run_dir,
    )
    _required_seed_artifacts(run_dir)["raw_summary"].write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(FmowResNet50CampaignError, match="signature mismatch"):
        _validate_seed_completion_contract(
            config,
            seed=42,
            protocol=protocol,
            geography_contract_hash=geography_hash,
            normalization_sha256=normalization_sha,
            run_dir=run_dir,
        )


def test_fmow_completion_contract_absent_seed_is_pending(
    tmp_path: Path,
) -> None:
    config, protocol, _run_dir, geography_hash, normalization_sha = (
        _completion_fixture(tmp_path)
    )
    assert (
        _validate_seed_completion_contract(
            config,
            seed=73,
            protocol=protocol,
            geography_contract_hash=geography_hash,
            normalization_sha256=normalization_sha,
            run_dir=tmp_path / "seed_73",
        )
        is None
    )


def test_sharp_fixed_universe_bounds_match_small_grid() -> None:
    known = [0.2, 0.8]
    lower, upper = _sharp_fixed_universe_bounds(
        known,
        fixed_group_count=4,
        beta=0.5,
    )
    grid = np.linspace(0.0, 1.0, 101)
    values = [
        compute_geobwer(
            {
                "known_0": known[0],
                "known_1": known[1],
                "unknown_0": float(first),
                "unknown_1": float(second),
            },
            0.5,
        ).bwer
        for first in grid
        for second in grid
    ]
    assert lower == pytest.approx(min(values), abs=1e-12)
    assert upper == pytest.approx(max(values), abs=1e-12)
    assert 0.0 <= lower <= upper <= 0.5
