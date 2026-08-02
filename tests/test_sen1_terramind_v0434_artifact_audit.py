from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.sen1_terramind_v0434_artifact_audit import (
    MODES,
    PANEL_SCOPE,
    SEEDS,
    Sen1TerraMindV0434ArtifactAuditError,
    audit_sen1_terramind_v0434_artifacts,
)


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _export(
    root: Path,
    *,
    count: int,
    checkpoint_sha: str,
) -> None:
    (root / "samples").mkdir(parents=True)
    (root / "index_parts").mkdir()
    rows = []
    for index in range(count):
        artifact = root / "samples" / f"{index:03d}.npz"
        artifact.write_bytes(f"probability-{index}".encode())
        rows.append(
            {
                "sample_id": f"sample-{index:03d}",
                "probability_path": f"samples/{artifact.name}",
            }
        )
    (root / "index_parts/part-000000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "writer_manifest_rank_0.json").write_text(
        json.dumps({"status": "complete", "row_count": count}),
        encoding="utf-8",
    )
    (root / "prediction_completion_contract.json").write_text(
        json.dumps(
            {
                "schema": (
                    "geobwer.sen1floods11."
                    "terramind_prediction_protocol.v1"
                ),
                "expected_row_count": count,
                "checkpoint_sha256": checkpoint_sha,
            }
        ),
        encoding="utf-8",
    )


def _candidate(cell: float, *, passes: bool) -> dict[str, object]:
    return {
        "cell_km": cell,
        "passes": passes,
        "null_coverage": 0.9,
        "null_coverage_ci_low": 0.8,
        "null_coverage_ci_high": 0.94,
        "false_positive_rate": 0.1,
        "false_positive_ci_low": 0.06,
        "false_positive_ci_high": 0.15,
        "moderate_tail_power": 0.7,
        "moderate_tail_power_ci_low": 0.6,
        "moderate_tail_power_ci_high": 0.8,
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "v0434"
    old = tmp_path / "v0430"
    runs = {}
    for mode in MODES:
        slug = mode.lower().replace("+", "_plus_")
        for seed in SEEDS:
            name = f"terramind_v1_base_{slug}_seed_{seed}"
            run = root / slug / f"seed_{seed}"
            old_run = old / slug / f"seed_{seed}"
            checkpoint = run / "checkpoints/best-epoch.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"checkpoint-{mode}-{seed}".encode())
            checkpoint_sha = file_sha256(checkpoint)
            fit_protocol = run / "fit_protocol.json"
            fit_protocol.write_text(json.dumps({"schema": "fit.v2"}), encoding="utf-8")
            fit_complete = run / "fit_complete.json"
            fit_complete.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": checkpoint_sha,
                        "fit_protocol_sha256": file_sha256(fit_protocol),
                    }
                ),
                encoding="utf-8",
            )
            for split, count in (
                ("validation", 89),
                ("test", 90),
                ("bolivia_holdout", 15),
            ):
                _export(
                    run / "probabilities" / split,
                    count=count,
                    checkpoint_sha=checkpoint_sha,
                )
            descriptive_root = run / "descriptive_only_outputs"
            descriptive_root.mkdir(parents=True)
            individual = {}
            for split in (
                "validation",
                "standard_test",
                "bolivia_holdout",
                "combined_held_out",
            ):
                path = descriptive_root / f"{split}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema": "descriptive.v1",
                            "status": "descriptive_only",
                            "formal_evidence": False,
                            "inferential_geobwer_run": False,
                            "bootstrap_run": False,
                        }
                    ),
                    encoding="utf-8",
                )
                individual[split] = path
            split_report = descriptive_root / "descriptive_split_report.json"
            split_report.write_text(
                json.dumps(
                    {
                        "schema": "descriptive.panel.v1",
                        "status": "descriptive_only",
                        "formal_evidence": False,
                        "inferential_geobwer_run": False,
                        "bootstrap_run": False,
                        "model_panel_inference_run": False,
                    }
                ),
                encoding="utf-8",
            )
            records = {
                "checkpoint": _artifact(checkpoint, root),
                "fit_protocol": _artifact(fit_protocol, root),
                "fit_completion": _artifact(fit_complete, root),
                "validation_prediction_contract": _artifact(
                    run / "probabilities/validation/prediction_completion_contract.json",
                    root,
                ),
                "test_prediction_contract": _artifact(
                    run / "probabilities/test/prediction_completion_contract.json",
                    root,
                ),
                "bolivia_prediction_contract": _artifact(
                    run
                    / "probabilities/bolivia_holdout/prediction_completion_contract.json",
                    root,
                ),
                "descriptive_split_report": _artifact(split_report, root),
                **{
                    f"descriptive_{split}": _artifact(path, root)
                    for split, path in individual.items()
                },
            }
            runs[name] = records
            for relative in (
                Path("fit_complete.json"),
                Path("fit_protocol.json"),
                Path("checkpoints/best-epoch.ckpt"),
                Path("probabilities/validation/prediction_completion_contract.json"),
            ):
                destination = old_run / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run / relative, destination)

    model_names = {
        f"{prefix}_{mode.lower().replace('+', '_plus_')}_seed_{seed}"
        for prefix in ("resnet34_unet", "terramind_v1_base")
        for mode in MODES
        for seed in SEEDS
    }
    model_names.add("prithvi_eo_v2_300_tl_s2")
    cells = [25.0, 50.0]
    calibration = {
        "schema": (
            "geobwer.sen1floods11."
            "common_spatial_block_calibration_failure.v1"
        ),
        "status": "calibration_invalid",
        "formal_evidence": True,
        "validation_only": True,
        "calibration_panel_scope": PANEL_SCOPE,
        "model_count": 19,
        "model_names": sorted(model_names),
        "models": {
            name: {"candidates": [_candidate(cell, passes=False) for cell in cells]}
            for name in model_names
        },
        "candidate_cell_km": cells,
        "common_passing_cells": [],
        "failures_by_cell": {
            str(cell): {
                "cell_km": cell,
                "failed_models": sorted(model_names),
                "failure_reasons_by_model": {
                    name: ["coverage_gate_failed"] for name in model_names
                },
            }
            for cell in cells
        },
        "calibration_signature": "a" * 64,
    }
    calibration_path = root / "calibration_failure_report.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    invalid_path = root / "calibration_invalid_contract.json"
    invalid_path.write_text(
        json.dumps(
            {
                "status": "calibration_invalid",
                "formal_evidence": True,
                "validation_only": True,
                "calibration_panel_scope": PANEL_SCOPE,
                "model_count": 19,
            }
        ),
        encoding="utf-8",
    )
    completion = {
        "schema": (
            "geobwer.sen1floods11."
            "terramind_descriptive_only_panel.v1"
        ),
        "status": "descriptive_only_complete",
        "formal_evidence": False,
        "package_version": "0.4.34",
        "code_commit": "8122085ad69e660957a8515d62f78cc1f337a787",
        "calibration_panel_scope": PANEL_SCOPE,
        "validation_only_calibration": True,
        "test_or_bolivia_used_for_calibration": False,
        "run_count": 9,
        "runs": runs,
        "inference_disabled": {
            "scale_dependent_geobwer": True,
            "bootstrap_significance": True,
            "model_panel": True,
        },
    }
    (root / "descriptive_only_completion_contract.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    return {"root": root, "old": old, "output": tmp_path / "evidence/audit.json"}


def test_v0434_descriptive_artifact_audit_passes_read_only(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    before = {
        root: {
            path.relative_to(root): (path.stat().st_size, file_sha256(path))
            for path in root.rglob("*")
            if path.is_file()
        }
        for root in (paths["root"], paths["old"])
    }
    report = audit_sen1_terramind_v0434_artifacts(
        paths["root"], old_resume_root=paths["old"], output_json=paths["output"]
    )
    after = {
        root: {
            path.relative_to(root): (path.stat().st_size, file_sha256(path))
            for path in root.rglob("*")
            if path.is_file()
        }
        for root in (paths["root"], paths["old"])
    }
    assert before == after
    assert report["status"] == "pass"
    assert report["counts"]["run_count"] == 9
    assert report["counts"]["validation_probability_units"] == 801
    assert report["counts"]["test_probability_units"] == 810
    assert report["counts"]["bolivia_probability_units"] == 135
    assert report["calibration_failure"]["model_count"] == 19
    assert report["calibration_failure"]["model_candidate_matrix_row_count"] == 38
    assert report["limitations"]["spatial_inference_valid"] is False
    assert report["limitations"]["descriptive_outputs_complete"] is True
    assert paths["output"].is_file()


def test_v0434_audit_rejects_formal_completion(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["root"] / "campaign_completion_contract.json").write_text(
        json.dumps({"status": "complete", "formal_evidence": True}),
        encoding="utf-8",
    )
    with pytest.raises(
        Sen1TerraMindV0434ArtifactAuditError,
        match="Formal completion",
    ):
        audit_sen1_terramind_v0434_artifacts(
            paths["root"],
            old_resume_root=paths["old"],
            output_json=paths["output"],
        )
    assert not paths["output"].exists()


def test_v0434_audit_rejects_missing_probability_file(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    missing = next((paths["root"] / "s1/seed_42/probabilities/test/samples").iterdir())
    missing.unlink()
    with pytest.raises(
        Sen1TerraMindV0434ArtifactAuditError,
        match="Missing referenced probability file",
    ):
        audit_sen1_terramind_v0434_artifacts(
            paths["root"],
            old_resume_root=paths["old"],
            output_json=paths["output"],
        )


def test_v0434_audit_rejects_resume_source_sha_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    old_checkpoint = paths["old"] / "s1/seed_42/checkpoints/best-epoch.ckpt"
    old_checkpoint.write_bytes(b"changed")
    with pytest.raises(
        Sen1TerraMindV0434ArtifactAuditError,
        match="artifact SHA mismatch",
    ):
        audit_sen1_terramind_v0434_artifacts(
            paths["root"],
            old_resume_root=paths["old"],
            output_json=paths["output"],
        )


def test_v0434_audit_refuses_output_inside_frozen_root(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(
        Sen1TerraMindV0434ArtifactAuditError,
        match="outside both frozen source roots",
    ):
        audit_sen1_terramind_v0434_artifacts(
            paths["root"],
            old_resume_root=paths["old"],
            output_json=paths["root"] / "audit.json",
        )
