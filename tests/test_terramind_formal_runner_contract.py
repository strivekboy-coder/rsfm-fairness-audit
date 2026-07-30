from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.colab.run_terramind_sen1floods11_final_colab import (
    MODES,
    _audited_supervised_validation_exports,
    _validated_prithvi_validation_export,
)
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
