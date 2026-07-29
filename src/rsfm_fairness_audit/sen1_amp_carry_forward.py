from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.formal_outputs import file_sha256


SOURCE_VERSION = "0.4.27"
SOURCE_COMMIT = "2ef26b2e1d1951910666f19b910b597834bb3d16"
TARGET_VERSION = "0.4.28"
SOURCE_RUN_SCHEMA = "geobwer.sen1floods11.supervised_resnet34_unet.v5"
CARRY_FORWARD_SCHEMA = "geobwer.sen1floods11.amp_carry_forward.v1"
EXPECTED_SPLIT_COUNTS = {
    "validation": 89,
    "test": 90,
    "bolivia_holdout": 15,
}


class Sen1AMPCarryForwardError(RuntimeError):
    """Raised when a legacy completed seed cannot be proven reusable."""


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _tree_binding(run_dir: Path) -> dict[str, Any]:
    paths = [
        run_dir / "run_manifest.json",
        run_dir / "best_resnet34_unet.pt",
    ]
    probability_root = run_dir / "probabilities"
    if probability_root.is_dir():
        paths.extend(
            path
            for path in sorted(probability_root.rglob("*"))
            if path.is_file()
        )
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise Sen1AMPCarryForwardError(
            f"Carry-forward source is incomplete under {run_dir}: {missing[:20]}"
        )
    artifacts = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in paths
    ]
    digest = hashlib.sha256()
    for artifact in artifacts:
        digest.update(str(artifact["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(artifact["sha256"]).encode("ascii"))
        digest.update(b"\0")
    return {
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(item["size_bytes"] for item in artifacts),
        "collection_sha256": digest.hexdigest(),
        "artifacts": artifacts,
    }


def _parse_source_log(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise Sen1AMPCarryForwardError(
            f"Source run log is required for code provenance: {path}"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    if SOURCE_COMMIT not in text:
        raise Sen1AMPCarryForwardError(
            "Source run log does not bind the completed seeds to the frozen "
            f"v0.4.27 commit {SOURCE_COMMIT}."
        )
    version_patterns = (
        rf"(?:VERSION|version|package_version)\s*[:=]\s*[\"']?v?{re.escape(SOURCE_VERSION)}",
        rf"\bv{re.escape(SOURCE_VERSION)}\b",
    )
    if not any(re.search(pattern, text) for pattern in version_patterns):
        raise Sen1AMPCarryForwardError(
            "Source run log does not bind the completed seeds to version "
            f"{SOURCE_VERSION}."
        )
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "source_commit": SOURCE_COMMIT,
        "source_version": SOURCE_VERSION,
    }


def _source_overflow_hard_fail_proof(project_root: Path) -> dict[str, Any]:
    command = [
        "git",
        "show",
        f"{SOURCE_COMMIT}:src/rsfm_fairness_audit/sen1_supervised_campaign.py",
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise Sen1AMPCarryForwardError(
            "Cannot inspect the frozen v0.4.27 training source from Git: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    source = completed.stdout
    required_fragments = (
        b"if invalid_gradient_parameters:",
        b"Training gradients contain NaN/Inf before optimizer.step",
        b"raise Sen1SupervisedCampaignError",
        b"scaler.step(optimizer)",
        b"scaler.update()",
    )
    if not all(fragment in source for fragment in required_fragments):
        raise Sen1AMPCarryForwardError(
            "Frozen v0.4.27 source no longer proves hard failure on every "
            "detected non-finite gradient."
        )
    return {
        "source_path": "src/rsfm_fairness_audit/sen1_supervised_campaign.py",
        "source_blob_sha256": hashlib.sha256(source).hexdigest(),
        "proof": (
            "A completed v0.4.27 run could not have observed a detected AMP "
            "gradient overflow because that exact source raised before "
            "checkpoint/probability completion."
        ),
        "finite_path_operations": [
            "scale(loss).backward",
            "unscale_(optimizer)",
            "scaler.step(optimizer)",
            "scaler.update",
            "finite_parameter_check",
        ],
    }


def _current_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip() if completed.returncode == 0 else ""
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Sen1AMPCarryForwardError(
            "Carry-forward must be generated from a Git-frozen v0.4.28 checkout."
        )
    return value


def _validate_source_seed(
    run_dir: Path,
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    checkpoint = run_dir / "best_resnet34_unet.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise Sen1AMPCarryForwardError(
            f"Completed source artifacts are missing for mode={mode}, seed={seed}."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SOURCE_RUN_SCHEMA
        or payload.get("formal_evidence") is not True
        or str(payload.get("sensor_mode")) != mode
        or int(payload.get("seed", -1)) != int(seed)
        or str(payload.get("checkpoint_sha256", "")) != file_sha256(checkpoint)
    ):
        raise Sen1AMPCarryForwardError(
            f"Source run manifest is not a completed formal v0.4.27 run: {manifest_path}"
        )
    if int(payload.get("amp_overflow_count", 0)) != 0:
        raise Sen1AMPCarryForwardError(
            f"Source run reports an AMP overflow and cannot be carried forward: {run_dir}"
        )
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        root = run_dir / "probabilities" / split
        index = root / "index_parts" / "part-000000.jsonl"
        support_path = root / "support_contract.json"
        quality_path = root / "input_quality_binding.json"
        if not all(path.is_file() for path in (index, support_path, quality_path)):
            raise Sen1AMPCarryForwardError(
                f"Source probability export is incomplete: mode={mode}, "
                f"seed={seed}, split={split}."
            )
        rows = [
            json.loads(line)
            for line in index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != expected_count:
            raise Sen1AMPCarryForwardError(
                f"Source split count changed for mode={mode}, seed={seed}, "
                f"split={split}: expected={expected_count}, observed={len(rows)}."
            )
        if not all((root / str(row["probability_path"])).is_file() for row in rows):
            raise Sen1AMPCarryForwardError(
                f"Source probability samples are incomplete: {root}"
            )
        support = json.loads(support_path.read_text(encoding="utf-8"))
        manifest_support = payload.get("split_support", {}).get(split, {})
        if (
            int(support.get("row_count", -1)) != expected_count
            or int(support.get("aggregate_valid_pixel_count", 0)) <= 0
            or str(manifest_support.get("support_contract_sha256", ""))
            != file_sha256(support_path)
            or str(manifest_support.get("input_quality_binding_sha256", ""))
            != file_sha256(quality_path)
        ):
            raise Sen1AMPCarryForwardError(
                f"Source support contract is invalid: {support_path}"
            )
    return {
        "mode": mode,
        "seed": int(seed),
        "run_dir": str(run_dir.resolve()),
        "normalization_sha256": str(payload.get("normalization_sha256", "")),
        "input_quality_contract_sha256": str(
            payload.get("input_quality_contract", {}).get("sha256", "")
        ),
        "best_validation_iou": float(payload["best_validation_iou"]),
        "test_iou": float(payload["split_metrics"]["test"]),
        "bolivia_holdout_iou": float(
            payload["split_metrics"]["bolivia_holdout"]
        ),
        "artifact_binding": _tree_binding(run_dir),
        "overflow_observation": {
            "observed_amp_overflow_count": 0,
            "proof_basis": (
                "formal run completed under v0.4.27 hard-fail gradient contract"
            ),
        },
    }


def build_carry_forward_manifest(
    *,
    project_root: Path,
    source_root: Path,
    source_run_log: Path,
    output_path: Path,
    mode: str = "S1",
    seeds: Sequence[int] = (42, 73),
) -> Path:
    source_root = source_root.resolve()
    output_path = output_path.resolve()
    if _path_is_within(output_path, source_root):
        raise Sen1AMPCarryForwardError(
            "Carry-forward manifest must be written outside the immutable "
            f"v0.4.27 source root: {source_root}"
        )
    provenance = _parse_source_log(source_run_log)
    hard_fail_proof = _source_overflow_hard_fail_proof(project_root.resolve())
    normalized_mode = str(mode).upper().replace(" ", "")
    entries = [
        _validate_source_seed(
            source_root
            / normalized_mode.lower().replace("+", "_plus_")
            / f"seed_{int(seed)}",
            mode=normalized_mode,
            seed=int(seed),
        )
        for seed in seeds
    ]
    payload = {
        "schema": CARRY_FORWARD_SCHEMA,
        "status": "validated",
        "source_version": SOURCE_VERSION,
        "source_commit": SOURCE_COMMIT,
        "target_version": TARGET_VERSION,
        "target_commit": _current_commit(project_root.resolve()),
        "source_root": str(source_root),
        "source_run_log": provenance,
        "source_overflow_hard_fail_proof": hard_fail_proof,
        "no_overflow_numerical_semantics": {
            "status": "preserved",
            "claim": (
                "v0.4.28 changes only the detected non-finite-gradient branch; "
                "finite-gradient batches retain the v0.4.27 backward, unscale, "
                "step, update, and parameter-check sequence."
            ),
            "regression_test": (
                "test_amp_finite_path_matches_v0427_reference_update"
            ),
        },
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_carry_forward_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise Sen1AMPCarryForwardError(
            f"Carry-forward manifest does not exist: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_commit = _current_commit(Path(__file__).resolve().parents[2])
    if (
        payload.get("schema") != CARRY_FORWARD_SCHEMA
        or payload.get("status") != "validated"
        or payload.get("source_commit") != SOURCE_COMMIT
        or payload.get("source_version") != SOURCE_VERSION
        or payload.get("target_version") != TARGET_VERSION
        or len(str(payload.get("target_commit", ""))) != 40
        or payload.get("target_commit") != current_commit
    ):
        raise Sen1AMPCarryForwardError(
            f"Carry-forward manifest is not valid for v0.4.28: {path}"
        )
    payload["_manifest_path"] = str(path.resolve())
    payload["_manifest_sha256"] = file_sha256(path)
    return payload


def reuse_carry_forward_seed(
    carry_forward: Mapping[str, Any] | None,
    *,
    mode: str,
    seed: int,
    expected_normalization_sha256: str,
    expected_input_quality_contract_sha256: str,
    candidate_run_dir: Path | None = None,
) -> dict[str, Any] | None:
    if carry_forward is None:
        return None
    matches = [
        entry
        for entry in carry_forward.get("entries", [])
        if str(entry.get("mode")) == mode and int(entry.get("seed", -1)) == seed
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise Sen1AMPCarryForwardError(
            f"Carry-forward manifest contains duplicate mode/seed: {mode}/{seed}."
        )
    entry = matches[0]
    if (
        str(entry.get("normalization_sha256", ""))
        != str(expected_normalization_sha256)
        or str(entry.get("input_quality_contract_sha256", ""))
        != str(expected_input_quality_contract_sha256)
    ):
        raise Sen1AMPCarryForwardError(
            f"Carry-forward data contract drifted for mode={mode}, seed={seed}."
        )
    run_dir = (
        candidate_run_dir.resolve()
        if candidate_run_dir is not None
        else Path(str(entry["run_dir"]))
    )
    current_binding = _tree_binding(run_dir)
    if current_binding != entry.get("artifact_binding"):
        raise Sen1AMPCarryForwardError(
            f"Carry-forward source artifacts changed for mode={mode}, seed={seed}."
        )
    manifest = run_dir / "run_manifest.json"
    checkpoint = run_dir / "best_resnet34_unet.pt"
    exports = {
        split: run_dir / "probabilities" / split
        for split in EXPECTED_SPLIT_COUNTS
    }
    return {
        "checkpoint": checkpoint,
        "manifest": manifest,
        "validation_export": exports["validation"],
        "test_export": exports["test"],
        "bolivia_holdout_export": exports["bolivia_holdout"],
        "validation_iou": float(entry["best_validation_iou"]),
        "test_iou": float(entry["test_iou"]),
        "bolivia_holdout_iou": float(entry["bolivia_holdout_iou"]),
        "carry_forward": True,
        "carry_forward_manifest": Path(
            str(carry_forward["_manifest_path"])
        ),
        "carry_forward_manifest_sha256": str(
            carry_forward["_manifest_sha256"]
        ),
    }


__all__ = [
    "CARRY_FORWARD_SCHEMA",
    "SOURCE_COMMIT",
    "SOURCE_VERSION",
    "TARGET_VERSION",
    "Sen1AMPCarryForwardError",
    "build_carry_forward_manifest",
    "load_carry_forward_manifest",
    "reuse_carry_forward_seed",
]
