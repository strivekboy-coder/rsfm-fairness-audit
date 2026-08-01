from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts.colab.run_terramind_sen1floods11_final_colab import (
    CALIBRATION_PANEL_SCOPE,
    MODES,
    _audited_supervised_validation_exports,
    _expected_calibration_model_names,
    _fit_if_needed,
    _predict_if_needed,
    _validated_prithvi_validation_export,
)
from rsfm_fairness_audit.adapters.terramind import TERRAMIND_OFFICIAL_REVISION
from rsfm_fairness_audit.formal_outputs import file_sha256


def _probability_export(root: Path, count: int = 89) -> Path:
    (root / "index_parts").mkdir(parents=True)
    (root / "samples").mkdir()
    (root / "writer_manifest_rank_0.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    rows = []
    for index in range(count):
        artifact = root / "samples" / f"sample_{index:03d}.npz"
        artifact.touch()
        rows.append(
            json.dumps(
                {
                    "sample_id": f"sample_{index:03d}",
                    "probability_path": f"samples/{artifact.name}",
                }
            )
        )
    (root / "index_parts" / "rank_0.jsonl").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return root


def _prithvi_manifest(path: Path, validation_export: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": (
                    "geobwer.sen1floods11."
                    "prithvi_tl_probability_migration.v3"
                ),
                "formal_evidence": True,
                "split_protocol": (
                    "official_252_89_90_plus_15_bolivia_holdout"
                ),
                "train_count": 252,
                "validation_count": 89,
                "test_count": 90,
                "bolivia_holdout_count": 15,
                "combined_evaluation_count": 105,
                "bolivia_holdout_used_for_training_or_calibration": False,
                "no_training_or_calibration_leakage": True,
                "probability_exports": {
                    "validation": str(validation_export)
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prithvi_formal_validation_export_is_read_only_and_bound(
    tmp_path: Path,
) -> None:
    export = _probability_export(
        tmp_path / "persistent" / "probabilities" / "validation"
    )
    manifest = _prithvi_manifest(
        tmp_path / "campaign_manifest.json",
        Path("/content/local/prithvi/probabilities/validation"),
    )

    lineage = _validated_prithvi_validation_export(manifest, export)

    assert lineage["validation_row_count"] == 89
    assert lineage["read_only"] is True
    assert lineage["validation_export"] == str(export)
    assert len(lineage["campaign_manifest_sha256"]) == 64


def test_prithvi_manifest_rejects_calibration_leakage(
    tmp_path: Path,
) -> None:
    export = _probability_export(
        tmp_path / "probabilities" / "validation"
    )
    manifest = _prithvi_manifest(
        tmp_path / "campaign_manifest.json", export
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bolivia_holdout_used_for_training_or_calibration"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not a formal official"):
        _validated_prithvi_validation_export(manifest, export)


def test_audited_unet_exports_resolve_persistent_copy_without_rewrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "supervised"
    root.mkdir()
    runs = {}
    for mode in MODES:
        slug = mode.lower().replace("+", "_plus_")
        for seed in (42, 73, 101):
            name = f"resnet34_unet_{slug}_seed_{seed}"
            _probability_export(
                root
                / slug
                / f"seed_{seed}"
                / "probabilities"
                / "validation"
            )
            runs[name] = {
                "validation_export": (
                    f"/content/expired/{slug}/seed_{seed}/"
                    "probabilities/validation"
                )
            }
    campaign = root / "campaign_manifest.json"
    campaign.write_text(
        json.dumps(
            {
                "schema": (
                    "geobwer.sen1floods11.supervised_panel.v6"
                ),
                "formal_evidence": True,
                "package_version": "0.4.28",
                "code_commit": (
                    "60cff004057c99799ae3c9523a0eab5de4070f59"
                ),
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": (
                    "geobwer.sen1floods11.unet_artifact_audit.v1"
                ),
                "status": "pass",
                "model_count": 9,
                "cross_model_sample_and_target_identity": "exact",
                "target": {
                    "campaign_manifest_sha256": file_sha256(campaign),
                    "code_commit": (
                        "60cff004057c99799ae3c9523a0eab5de4070f59"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    exports, lineage = _audited_supervised_validation_exports(
        campaign, audit
    )

    assert len(exports) == 9
    assert lineage["read_only"] is True
    assert all(str(path).startswith(str(root)) for path in exports.values())


def test_frozen_calibration_scope_is_exactly_all_nineteen_models() -> None:
    names = _expected_calibration_model_names()
    assert CALIBRATION_PANEL_SCOPE == "all_19_models_unet9_terramind9_prithvi1"
    assert len(names) == 19
    assert len([name for name in names if name.startswith("resnet34_unet_")]) == 9
    assert len([name for name in names if name.startswith("terramind_v1_base_")]) == 9
    assert "prithvi_eo_v2_300_tl_s2" in names


def test_resume_contract_reuses_fit_and_validation_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = Path("work/test_terramind_resume_contract")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "best-epoch.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config = run_dir / "fit.yaml"
    config.write_text("frozen: true\n", encoding="utf-8")
    backbone_sha = "a" * 64
    fit_protocol = run_dir / "fit_protocol.json"
    fit_protocol.write_text(
        json.dumps(
            {
                "schema": "geobwer.terramind.fit_protocol.v2",
                "config_sha256": file_sha256(config),
                "backbone_checkpoint_sha256": backbone_sha,
                "backbone_revision": TERRAMIND_OFFICIAL_REVISION,
                "training_length_policy": "fixed_100_epochs_no_early_stopping",
                "early_stopping_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "fit_complete.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": file_sha256(checkpoint),
                "fit_protocol_sha256": file_sha256(fit_protocol),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.colab.run_terramind_sen1floods11_final_colab._run",
        lambda *_args, **_kwargs: pytest.fail("resume unexpectedly executed fit/predict"),
    )
    assert _fit_if_needed(
        config,
        run_dir,
        mode="S1",
        seed=42,
        backbone_checkpoint_sha256=backbone_sha,
        dry_run=False,
        reuse_only=True,
    ) == checkpoint

    export = _probability_export(tmp_path / "validation", count=89)
    prediction_config = tmp_path / "predict.yaml"
    prediction_config.write_text("frozen: true\n", encoding="utf-8")
    input_quality_sha = "b" * 64
    (export / "prediction_completion_contract.json").write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.terramind_prediction_protocol.v1",
                "config_sha256": file_sha256(prediction_config),
                "checkpoint_sha256": file_sha256(checkpoint),
                "input_quality_contract_sha256": input_quality_sha,
                "imputation_policy": "official_train_band_mean_normalized_zero",
                "expected_row_count": 89,
            }
        ),
        encoding="utf-8",
    )
    _predict_if_needed(
        prediction_config,
        checkpoint,
        export,
        mode="S1",
        seed=42,
        split="validation",
        expected=89,
        dry_run=False,
        input_quality_contract_sha256=input_quality_sha,
        reuse_only=True,
    )
    shutil.rmtree(tmp_path)
