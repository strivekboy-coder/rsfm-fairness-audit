from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest

import rsfm_fairness_audit.alphaearth_existing_upgrade as upgrade
from rsfm_fairness_audit.io import write_csv


CLASSES = tuple(str(10 * (index + 1)) for index in range(11))
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, split in enumerate(("calibration", "calibration", "test", "test")):
        target = index % len(CLASSES)
        probability = np.full(len(CLASSES), 0.001, dtype=float)
        probability[target] = 1.0 - 0.001 * (len(CLASSES) - 1)
        row: dict[str, object] = {
            "sample_id": f"sample_{index}",
            "split": split,
            "spatial_block_id": f"{split}_block_{index}",
            "label": CLASSES[target],
            "prediction": CLASSES[target],
            "country_iso3": "USA",
            "region": "Northern America",
            "worldcover_class_name": f"class_{target}",
            "income_group": "High income",
            "biome_or_ecoregion": "temperate",
            "urban_rural_or_built_proxy": "non_built",
            "lat": 10.0 + index,
            "lon": 20.0 + index,
        }
        for name, value in zip(CLASSES, probability):
            row[f"prob_{name}"] = value
        rows.append(row)
    return rows


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_rejects_calibration_test_sample_leakage() -> None:
    rows = _rows()
    rows[2]["sample_id"] = rows[0]["sample_id"]
    with pytest.raises(
        upgrade.AlphaEarthExistingUpgradeError, match="sample leakage"
    ):
        upgrade._validate_splits(rows)


def _test_root() -> Path:
    root = Path("outputs") / f"test_alphaearth_existing_upgrade_{uuid4().hex}"
    root.mkdir(parents=True)
    return root


def test_rejects_probability_column_order_conflict() -> None:
    rows = _rows()
    rows[0]["prediction"] = CLASSES[1]
    path = _test_root() / "eval.csv"
    _write_predictions(path, rows)
    with pytest.raises(
        upgrade.AlphaEarthExistingUpgradeError, match="column order conflicts"
    ):
        upgrade._read_eval_predictions(path)


def test_rejects_spatial_block_leakage() -> None:
    rows = _rows()
    rows[2]["spatial_block_id"] = rows[0]["spatial_block_id"]
    with pytest.raises(
        upgrade.AlphaEarthExistingUpgradeError, match="Spatial-block leakage"
    ):
        upgrade._validate_splits(rows)


def test_rejects_dynamic_world_scope_promotion() -> None:
    with pytest.raises(
        upgrade.AlphaEarthExistingUpgradeError,
        match="cannot be promoted",
    ):
        upgrade._validate_dynamic_world_scopes(
            [
                {
                    "scope": "all_split_descriptive",
                    "claim_role": "formal_model_accuracy",
                }
            ]
        )


def test_completion_contract_resume_and_drift() -> None:
    root = _test_root() / "complete"
    root.mkdir()
    (root / "result.csv").write_text("a\n1\n", encoding="utf-8")
    payload = {"protocol_hash": "abc", "seed": 42}
    upgrade._write_completion(root, payload)
    signature = upgrade._completion_signature(payload)
    assert upgrade._validate_completion(root, signature)
    with pytest.raises(
        upgrade.AlphaEarthExistingUpgradeError, match="signature drift"
    ):
        upgrade._validate_completion(
            root, upgrade._completion_signature({"protocol_hash": "other"})
        )
    (root / "result.csv").write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(
        upgrade.AlphaEarthExistingUpgradeError, match="artifact mismatch"
    ):
        upgrade._validate_completion(root, signature)


def test_rejects_frozen_source_output_overlap() -> None:
    source = _test_root() / "source"
    source.mkdir()
    with pytest.raises(
        upgrade.AlphaEarthExistingUpgradeError, match="must be disjoint"
    ):
        upgrade._assert_distinct_roots(source, source / "new_output", None)


def test_valid_probability_and_split_contract() -> None:
    path = _test_root() / "eval.csv"
    _write_predictions(path, _rows())
    rows, classes, probabilities, targets = upgrade._read_eval_predictions(path)
    calibration, test, evidence = upgrade._validate_splits(rows)
    assert classes == CLASSES
    assert probabilities.shape == (4, 11)
    assert targets.tolist() == [0, 1, 2, 3]
    assert calibration.tolist() == [True, True, False, False]
    assert test.tolist() == [False, False, True, True]
    assert evidence["calibration_test_spatial_block_overlap"] == 0


def test_joint_risk_card_does_not_call_unidentified_rows_tail_saturated() -> None:
    card = upgrade._joint_risk_card(
        [],
        [
            {
                "axis": "region",
                "beta": 0.1,
                "bwer": "",
                "validity": "not_identified_missing_standardization_cells",
            }
        ],
        [
            {
                "axis": "region",
                "excluded_deployment_mass": 0.0,
                "fixed_universe_groups": 7,
                "supported_universe_groups": 7,
            }
        ],
    )
    assert len(card) == 2
    standardized = next(
        row for row in card if row["risk_family"] == "class_standardized"
    )
    assert standardized["tail_saturation"] == "not_applicable"
    assert standardized["beta_effective_tail_slices"] == ""
    assert standardized["partial_identification_lower"] == ""
    assert standardized["partial_identification_upper"] == ""
    assert standardized["partial_identification_scope"] == "not_identified"


def test_small_existing_output_upgrade_and_resume() -> None:
    root = _test_root()
    source = root / "frozen_source"
    output = root / "new_output"
    rows: list[dict[str, object]] = []
    for split_index, split in enumerate(("calibration", "test")):
        for index in range(66):
            target = index % len(CLASSES)
            country_index = index % 2
            probability = np.full(len(CLASSES), 0.01, dtype=float)
            probability[target] = 0.90
            row: dict[str, object] = {
                "sample_id": f"{split}_{index}",
                "split": split,
                "spatial_block_id": f"{split}_block_{index}",
                "label": CLASSES[target],
                "prediction": CLASSES[target],
                "country_iso3": ("USA", "BRA")[country_index],
                "region": ("Northern America", "South America")[country_index],
                "worldcover_class_name": f"class_{target}",
                "income_group": ("High income", "Upper middle income")[country_index],
                "biome_or_ecoregion": ("temperate", "tropical")[country_index],
                "urban_rural_or_built_proxy": ("built", "non_built")[country_index],
                "lat": -30.0 + index * 0.5,
                "lon": -100.0 + index,
                "risk": 0,
            }
            for name, value in zip(CLASSES, probability):
                row[f"prob_{name}"] = value
            rows.append(row)
    _write_predictions(source / "alphaearth_full_eval_predictions.csv", rows)
    write_csv(source / "alphaearth_full_metrics.csv", [{"accuracy": 1.0}])
    write_csv(source / "alphaearth_full_bwer_summary.csv", [{"bwer": 0.0}])
    artifacts = upgrade.run_alphaearth_existing_upgrade(
        upgrade.AlphaEarthExistingUpgradeConfig(
            source_root=source,
            output_dir=output,
            protocol_path=Path("configs/geobwer/alphaearth.yaml"),
            audit_bootstrap=100,
        )
    )
    for path in artifacts.values():
        assert path.exists()
    with np.load(output / "formal_outputs" / "probabilities.npz") as data:
        formal_targets = data["targets"]
        assert set(formal_targets.tolist()) == set(range(11))
        assert int(np.count_nonzero(formal_targets == 10)) == 6
    formal_manifest = json.loads(
        (output / "formal_outputs" / "formal_output_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert formal_manifest["extra"]["target_encoding"] == "integer_indices"
    postprocess_manifest = json.loads(
        (output / "postprocess_manifest.json").read_text(encoding="utf-8")
    )
    roundtrip = postprocess_manifest["formal_probability_roundtrip"]
    assert roundtrip["status"] == "exact_roundtrip"
    assert roundtrip["target_histogram"]["110"] == 6
    resumed = upgrade.run_alphaearth_existing_upgrade(
        upgrade.AlphaEarthExistingUpgradeConfig(
            source_root=source,
            output_dir=output,
            protocol_path=Path("configs/geobwer/alphaearth.yaml"),
            audit_bootstrap=100,
        )
    )
    assert resumed["completion_contract"].exists()


def test_colab_runner_absolute_path_from_external_cwd_imports_scripts_analysis() -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "colab"
        / "upgrade_alphaearth_existing_outputs_geobwer_colab.py"
    ).resolve()
    probe = (
        "import runpy\n"
        f"runpy.run_path({str(script)!r}, run_name='alphaearth_runner_probe')\n"
        "import scripts.analysis.build_alphaearth_final_evidence_hardening_v2\n"
        "print('ALPHAEARTH_ABSOLUTE_IMPORT=PASS')\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=PROJECT_ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ALPHAEARTH_ABSOLUTE_IMPORT=PASS" in result.stdout
