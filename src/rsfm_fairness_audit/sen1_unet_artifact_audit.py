from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.formal_outputs import file_sha256


class Sen1UNetArtifactAuditError(RuntimeError):
    """Raised when frozen Sen1 U-Net evidence violates its formal contract."""


TARGET_VERSION = "0.4.28"
TARGET_COMMIT = "60cff004057c99799ae3c9523a0eab5de4070f59"
SOURCE_VERSION = "0.4.27"
SOURCE_COMMIT = "2ef26b2e1d1951910666f19b910b597834bb3d16"
MODES = ("S1", "S2", "S1+S2")
SEEDS = (42, 73, 101)
MODE_CHANNELS = {"S1": 2, "S2": 13, "S1+S2": 15}
SPLIT_COUNTS = {"validation": 89, "test": 90, "bolivia_holdout": 15}
AMP_SCHEMA = "geobwer.sen1floods11.amp_overflow.v1"
IMPUTATION_POLICY = "official_train_band_mean_normalized_zero"


@dataclass(frozen=True)
class _Expectation:
    version: str
    commit: str
    modes: tuple[str, ...]
    seeds: tuple[int, ...]
    split_counts: Mapping[str, int]


FORMAL_EXPECTATION = _Expectation(
    version=TARGET_VERSION,
    commit=TARGET_COMMIT,
    modes=MODES,
    seeds=SEEDS,
    split_counts=SPLIT_COUNTS,
)


def _fail(message: str) -> None:
    raise Sen1UNetArtifactAuditError(message)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"Required JSON artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Sen1UNetArtifactAuditError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        _fail(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"Required JSONL artifact is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise Sen1UNetArtifactAuditError(
                f"Invalid JSONL row {line_number}: {path}"
            ) from exc
        if not isinstance(row, dict):
            _fail(f"JSONL row {line_number} is not an object: {path}")
        rows.append(row)
    return rows


def _relative_suffix_matches(recorded: Any, expected: Path, root: Path) -> bool:
    recorded_parts = tuple(
        part.lower()
        for part in str(recorded).replace("\\", "/").split("/")
        if part
    )
    expected_parts = tuple(
        part.lower() for part in expected.relative_to(root).parts
    )
    return len(recorded_parts) >= len(expected_parts) and (
        recorded_parts[-len(expected_parts) :] == expected_parts
    )


def _assert_path_reference(
    recorded: Any, expected: Path, root: Path, *, label: str
) -> None:
    if not _relative_suffix_matches(recorded, expected, root):
        _fail(
            f"{label} does not reference the canonical frozen artifact: "
            f"recorded={recorded!r}, expected_suffix={expected.relative_to(root)}"
        )


def _assert_finite_tree(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _assert_finite_tree(item, label=f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        _fail(f"Non-finite numeric value in {label}.")


def _canonical_run(root: Path, mode: str, seed: int) -> Path:
    return root / mode.lower().replace("+", "_plus_") / f"seed_{seed}"


def _run_name(mode: str, seed: int) -> str:
    return f"resnet34_unet_{mode.lower().replace('+', '_plus_')}_seed_{seed}"


def _artifact_tree_binding(run_dir: Path) -> dict[str, Any]:
    paths = [run_dir / "run_manifest.json", run_dir / "best_resnet34_unet.pt"]
    probability_root = run_dir / "probabilities"
    if probability_root.is_dir():
        paths.extend(
            path for path in sorted(probability_root.rglob("*")) if path.is_file()
        )
    if not all(path.is_file() for path in paths):
        _fail(f"Carry-forward artifact tree is incomplete: {run_dir}")
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
        "total_size_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "collection_sha256": digest.hexdigest(),
        "artifacts": artifacts,
    }


def _resolve_carry_forward(
    campaign: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[tuple[str, int], dict[str, Any]]]:
    summary = campaign.get("carry_forward")
    if summary is None:
        return None, {}
    if not isinstance(summary, Mapping):
        _fail("Campaign carry_forward field is not a mapping.")
    path = Path(str(summary.get("manifest", "")))
    if not path.is_file():
        _fail(f"Carry-forward manifest is not accessible: {path}")
    if str(summary.get("manifest_sha256", "")) != file_sha256(path):
        _fail(f"Carry-forward manifest SHA mismatch: {path}")
    payload = _read_json(path)
    if (
        payload.get("schema") != "geobwer.sen1floods11.amp_carry_forward.v1"
        or payload.get("status") != "validated"
        or payload.get("source_version") != SOURCE_VERSION
        or payload.get("source_commit") != SOURCE_COMMIT
        or payload.get("target_version") != TARGET_VERSION
        or payload.get("target_commit") != TARGET_COMMIT
        or summary.get("source_version") != SOURCE_VERSION
        or summary.get("source_commit") != SOURCE_COMMIT
        or summary.get("target_version") != TARGET_VERSION
        or summary.get("target_commit") != TARGET_COMMIT
        or payload.get("no_overflow_numerical_semantics", {}).get("status")
        != "preserved"
    ):
        _fail(f"Carry-forward manifest is not valid for the frozen target: {path}")
    proof = payload.get("source_overflow_hard_fail_proof", {})
    if (
        not proof.get("source_blob_sha256")
        or "could not have observed" not in str(proof.get("proof", ""))
    ):
        _fail(f"Carry-forward overflow proof is incomplete: {path}")
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in payload.get("entries", []):
        key = (str(entry.get("mode", "")), int(entry.get("seed", -1)))
        if key in entries:
            _fail(f"Duplicate carry-forward entry: {key}")
        entries[key] = dict(entry)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "source_commit": SOURCE_COMMIT,
        "target_commit": TARGET_COMMIT,
        "entry_count": len(entries),
    }, entries


def _hash_target_rows(
    rows: Sequence[tuple[str, np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    for sample_id, target in rows:
        contiguous = np.ascontiguousarray(target)
        digest.update(sample_id.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _audit_export(
    *,
    export: Path,
    split: str,
    mode: str,
    expected_count: int,
    manifest: Mapping[str, Any],
    input_quality_sha: str,
) -> dict[str, Any]:
    index = export / "index_parts" / "part-000000.jsonl"
    rows = _read_jsonl(index)
    if len(rows) != expected_count:
        _fail(
            f"{mode}/{split} has {len(rows)} rows; expected {expected_count}."
        )
    sample_ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(rows):
        _fail(f"{mode}/{split} sample_id values are empty or duplicated.")

    binding_path = export / "input_quality_binding.json"
    support_path = export / "support_contract.json"
    binding = _read_json(binding_path)
    support = _read_json(support_path)
    manifest_support = manifest.get("split_support", {}).get(split, {})
    expected_role = "standard_test" if split == "test" else split
    if (
        binding.get("schema")
        != "geobwer.sen1floods11.input_quality_binding.v1"
        or binding.get("split") != split
        or binding.get("split_role") != expected_role
        or binding.get("sensor_mode") != mode
        or binding.get("imputation_policy") != IMPUTATION_POLICY
        or binding.get("input_quality_contract_sha256") != input_quality_sha
    ):
        _fail(f"Invalid input-quality binding: {binding_path}")
    binding_sha = file_sha256(binding_path)
    if (
        support.get("schema")
        != "geobwer.sen1floods11.probability_support.v1"
        or support.get("split") != split
        or support.get("sensor_mode") != mode
        or support.get("input_quality_binding_sha256") != binding_sha
        or manifest_support.get("support_contract_sha256")
        != file_sha256(support_path)
        or manifest_support.get("input_quality_binding_sha256") != binding_sha
    ):
        _fail(f"Invalid probability support binding: {support_path}")

    aggregate_valid = 0
    all_ignore = 0
    observed_values: set[int] = set()
    target_rows: list[tuple[str, np.ndarray]] = []
    probability_paths: set[str] = set()
    max_probability_sum_error = 0.0
    for row in rows:
        relative = str(row.get("probability_path", ""))
        if (
            not relative
            or relative in probability_paths
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            _fail(f"Unsafe or duplicated probability_path in {index}: {relative!r}")
        probability_paths.add(relative)
        artifact = export / relative
        if not artifact.is_file():
            _fail(f"Probability artifact is missing: {artifact}")
        try:
            with np.load(artifact, allow_pickle=False) as bundle:
                if "probabilities" not in bundle or "target" not in bundle:
                    _fail(f"Probability bundle lacks required arrays: {artifact}")
                probabilities = np.asarray(bundle["probabilities"])
                target = np.asarray(bundle["target"])
        except Sen1UNetArtifactAuditError:
            raise
        except Exception as exc:
            raise Sen1UNetArtifactAuditError(
                f"Cannot read probability bundle: {artifact}"
            ) from exc
        if probabilities.ndim != 3 or probabilities.shape[0] != 2:
            _fail(f"Expected [2,H,W] probabilities, got {probabilities.shape}: {artifact}")
        if target.shape != probabilities.shape[1:]:
            _fail(
                f"Target/probability spatial shape mismatch in {artifact}: "
                f"{target.shape} vs {probabilities.shape[1:]}"
            )
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < -1e-6)
            or np.any(probabilities > 1.0 + 1e-6)
        ):
            _fail(f"Invalid probability values in {artifact}")
        sum_error = float(
            np.max(np.abs(np.sum(probabilities, axis=0) - 1.0))
        )
        if sum_error > 1e-5:
            _fail(f"Class probabilities do not sum to one in {artifact}: {sum_error}")
        max_probability_sum_error = max(max_probability_sum_error, sum_error)
        if not np.issubdtype(target.dtype, np.integer):
            if not np.all(np.isfinite(target)) or not np.all(target == np.round(target)):
                _fail(f"Target is not integer-valued in {artifact}")
        values = {int(value) for value in np.unique(target).tolist()}
        if not values.issubset({-1, 0, 1}):
            _fail(f"Unexpected target values in {artifact}: {sorted(values)}")
        valid_count = int(np.count_nonzero(np.isin(target, [0, 1])))
        aggregate_valid += valid_count
        all_ignore += int(valid_count == 0)
        observed_values.update(values)
        target_rows.append((str(row["sample_id"]), target))

    if aggregate_valid <= 0:
        _fail(f"{mode}/{split} has no valid labeled pixels.")
    valid_rows = expected_count - all_ignore
    expected_values = sorted(observed_values)
    for contract in (support, manifest_support):
        if (
            int(contract.get("row_count", -1)) != expected_count
            or int(contract.get("all_ignore_row_count", -1)) != all_ignore
            or int(contract.get("valid_row_count", -1)) != valid_rows
            or int(contract.get("aggregate_valid_pixel_count", -1))
            != aggregate_valid
            or sorted(int(value) for value in contract.get("observed_target_values", []))
            != expected_values
        ):
            _fail(f"Recomputed support disagrees with contract for {mode}/{split}.")
    return {
        "row_count": expected_count,
        "sample_ids": sample_ids,
        "sample_id_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest(),
        "target_sha256": _hash_target_rows(target_rows),
        "all_ignore_row_count": all_ignore,
        "valid_row_count": valid_rows,
        "aggregate_valid_pixel_count": aggregate_valid,
        "observed_target_values": expected_values,
        "max_probability_sum_error": max_probability_sum_error,
        "support_contract_sha256": file_sha256(support_path),
        "input_quality_binding_sha256": binding_sha,
    }


def _audit_normalization(
    path: Path, *, mode: str, expected_sha: str
) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        file_sha256(path) != expected_sha
        or payload.get("schema")
        != "geobwer.sen1floods11.train_normalization.v4"
        or payload.get("sensor_mode") != mode
        or payload.get("selection_split") != "official_train"
        or payload.get("test_rows_used") is not False
        or payload.get("imputation_policy") != IMPUTATION_POLICY
        or int(payload.get("normalization_sample_count", -1)) != 252
        or len(payload.get("sample_prefixes", [])) != 252
    ):
        _fail(f"Invalid official-train normalization contract: {path}")
    _assert_finite_tree(payload, label=f"normalization[{mode}]")
    return payload


def _audit_quality(
    path: Path,
    *,
    mode: str,
    expected_sha: str,
    normalization_sha: str,
    split_counts: Mapping[str, int],
) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        file_sha256(path) != expected_sha
        or payload.get("schema") != "geobwer.sen1floods11.input_quality.v2"
        or payload.get("sensor_mode") != mode
        or payload.get("imputation_policy") != IMPUTATION_POLICY
        or payload.get("normalization_sha256") != normalization_sha
    ):
        _fail(f"Invalid input-quality contract: {path}")
    for split, expected in split_counts.items():
        split_payload = payload.get("splits", {}).get(split, {})
        records = split_payload.get("records", [])
        if len(records) != expected:
            _fail(f"{path}: {split} contains {len(records)} records, expected {expected}.")
    return payload


def _audit_amp(
    manifest: Mapping[str, Any],
    run_dir: Path,
    *,
    carry_forward_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if carry_forward_entry is not None:
        if (
            manifest.get("schema")
            != "geobwer.sen1floods11.supervised_resnet34_unet.v5"
            or manifest.get("package_version") != SOURCE_VERSION
            or manifest.get("code_commit") != SOURCE_COMMIT
            or int(manifest.get("amp_overflow_count", 0)) != 0
            or int(
                carry_forward_entry.get("overflow_observation", {}).get(
                    "observed_amp_overflow_count", -1
                )
            )
            != 0
        ):
            _fail(f"Invalid carry-forward AMP evidence: {run_dir}")
        return {
            "amp_overflow_count": 0,
            "skipped_optimizer_step_count": 0,
            "maximum_consecutive_amp_overflow_count": 0,
            "journal_sha256": None,
            "evidence": "validated_v0.4.27_hard_fail_carry_forward",
        }
    records = manifest.get("amp_overflow_records")
    if (
        manifest.get("amp_overflow_policy_schema") != AMP_SCHEMA
        or not isinstance(records, list)
    ):
        _fail(f"Missing AMP overflow contract: {run_dir}")
    count = int(manifest.get("amp_overflow_count", -1))
    skipped = int(manifest.get("skipped_optimizer_step_count", -1))
    maximum_consecutive = int(
        manifest.get("maximum_consecutive_amp_overflow_count", -1)
    )
    max_total = int(manifest.get("amp_max_total_overflows", -1))
    max_consecutive = int(manifest.get("amp_max_consecutive_overflows", -1))
    if (
        count != len(records)
        or skipped != count
        or max_total != 20
        or max_consecutive != 3
        or count > max_total
        or maximum_consecutive > max_consecutive
    ):
        _fail(f"Invalid AMP overflow counts or thresholds: {run_dir}")
    for record in records:
        if (
            record.get("schema") != AMP_SCHEMA
            or record.get("optimizer_step_skipped") is not True
            or not record.get("sample_ids")
            or not record.get("overflow_parameter_names")
            or not math.isfinite(float(record.get("scale_before", math.nan)))
            or not math.isfinite(float(record.get("scale_after", math.nan)))
            or float(record["scale_after"]) >= float(record["scale_before"])
        ):
            _fail(f"Invalid AMP overflow record: {run_dir}")
    journal = run_dir / "amp_overflow_journal.json"
    if count:
        if (
            not journal.is_file()
            or manifest.get("amp_overflow_journal_sha256") != file_sha256(journal)
        ):
            _fail(f"AMP overflow journal is missing or has a SHA mismatch: {run_dir}")
        journal_payload = _read_json(journal)
        if (
            int(journal_payload.get("amp_overflow_count", -1)) != count
            or journal_payload.get("amp_overflow_records") != records
        ):
            _fail(f"AMP journal content disagrees with run manifest: {run_dir}")
    elif manifest.get("amp_overflow_journal") is not None or manifest.get(
        "amp_overflow_journal_sha256"
    ) is not None:
        _fail(f"Zero-overflow run records a non-null AMP journal: {run_dir}")
    return {
        "amp_overflow_count": count,
        "skipped_optimizer_step_count": skipped,
        "maximum_consecutive_amp_overflow_count": maximum_consecutive,
        "journal_sha256": file_sha256(journal) if count else None,
    }


def _audit_engine(
    source_root: Path,
    *,
    expectation: _Expectation,
    repository_root: Path | None,
) -> dict[str, Any]:
    root = source_root.resolve()
    campaign_path = root / "campaign_manifest.json"
    campaign = _read_json(campaign_path)
    carry_summary, carry_entries = _resolve_carry_forward(campaign)
    expected_run_names = {
        _run_name(mode, seed)
        for mode in expectation.modes
        for seed in expectation.seeds
    }
    if (
        campaign.get("schema") != "geobwer.sen1floods11.supervised_panel.v6"
        or campaign.get("package_version") != expectation.version
        or campaign.get("code_commit") != expectation.commit
        or campaign.get("formal_evidence") is not True
        or campaign.get("design") != "resnet34_unet_x_sensor_mode_x_seed"
        or campaign.get("split_protocol")
        != "official_252_89_90_plus_15_bolivia_holdout"
        or int(campaign.get("evaluation_sample_count", -1))
        != int(expectation.split_counts["test"])
        + int(expectation.split_counts["bolivia_holdout"])
        or int(campaign.get("standard_test_count", -1))
        != int(expectation.split_counts["test"])
        or int(campaign.get("bolivia_holdout_count", -1))
        != int(expectation.split_counts["bolivia_holdout"])
        or campaign.get("no_training_or_calibration_leakage") is not True
        or tuple(campaign.get("sensor_modes", [])) != expectation.modes
        or tuple(int(value) for value in campaign.get("seeds", []))
        != expectation.seeds
        or campaign.get("config", {}).get("diagnostic_max_samples") is not None
        or set(campaign.get("runs", {})) != expected_run_names
    ):
        _fail(f"Campaign manifest does not match frozen v0.4.28 formal panel: {campaign_path}")

    mode_contracts: dict[str, Any] = {}
    for mode in expectation.modes:
        slug = mode.lower().replace("+", "_plus_")
        normalization_path = root / "normalization" / f"{slug}.json"
        quality_path = root / "input_quality" / f"{slug}.json"
        norm_entry = campaign.get("normalization_contracts", {}).get(mode, {})
        quality_entry = campaign.get("input_quality_contracts", {}).get(mode, {})
        _assert_path_reference(
            norm_entry.get("path"), normalization_path, root, label=f"{mode} normalization"
        )
        _assert_path_reference(
            quality_entry.get("path"), quality_path, root, label=f"{mode} input quality"
        )
        normalization = _audit_normalization(
            normalization_path, mode=mode, expected_sha=str(norm_entry.get("sha256", ""))
        )
        quality = _audit_quality(
            quality_path,
            mode=mode,
            expected_sha=str(quality_entry.get("sha256", "")),
            normalization_sha=file_sha256(normalization_path),
            split_counts=expectation.split_counts,
        )
        mode_contracts[mode] = {
            "normalization_sha256": file_sha256(normalization_path),
            "input_quality_sha256": file_sha256(quality_path),
            "normalization_sample_count": normalization["normalization_sample_count"],
            "input_quality_summary": quality.get("summary", {}),
        }

    run_reports: dict[str, Any] = {}
    reference_ids: dict[str, list[str]] = {}
    reference_targets: dict[str, str] = {}
    total_overflows = 0
    for mode in expectation.modes:
        for seed in expectation.seeds:
            name = _run_name(mode, seed)
            campaign_run = campaign["runs"][name]
            is_carried = campaign_run.get("carry_forward") is True
            carry_entry = carry_entries.get((mode, seed)) if is_carried else None
            if is_carried and carry_entry is None:
                _fail(f"Campaign marks {name} as carried without a matching entry.")
            if not is_carried and (mode, seed) in carry_entries:
                _fail(f"Carry-forward entry for {name} is not identified in campaign runs.")
            run_dir = (
                Path(str(campaign_run.get("manifest", ""))).resolve().parent
                if is_carried
                else _canonical_run(root, mode, seed)
            )
            manifest_path = run_dir / "run_manifest.json"
            checkpoint = run_dir / "best_resnet34_unet.pt"
            manifest = _read_json(manifest_path)
            expected_schema = (
                "geobwer.sen1floods11.supervised_resnet34_unet.v5"
                if is_carried
                else "geobwer.sen1floods11.supervised_resnet34_unet.v6"
            )
            expected_version = SOURCE_VERSION if is_carried else expectation.version
            expected_commit = SOURCE_COMMIT if is_carried else expectation.commit
            if (
                manifest.get("schema")
                != expected_schema
                or manifest.get("package_version") != expected_version
                or manifest.get("code_commit") != expected_commit
                or manifest.get("formal_evidence") is not True
                or manifest.get("sensor_mode") != mode
                or int(manifest.get("input_channels", -1)) != MODE_CHANNELS[mode]
                or int(manifest.get("seed", -1)) != seed
                or manifest.get("model_selection")
                != "official_train_inner_event_disjoint"
                or manifest.get("outer_validation_used_for_model_selection") is not False
                or manifest.get("bolivia_holdout_used_for_training_or_selection") is not False
                or manifest.get("imputation_policy") != IMPUTATION_POLICY
            ):
                _fail(f"Invalid formal run manifest: {manifest_path}")
            _assert_finite_tree(manifest, label=name)
            if not checkpoint.is_file() or manifest.get("checkpoint_sha256") != file_sha256(
                checkpoint
            ):
                _fail(f"Checkpoint missing or SHA mismatch: {checkpoint}")
            _assert_path_reference(
                manifest.get("checkpoint"), checkpoint, run_dir, label=f"{name} checkpoint"
            )
            _assert_path_reference(
                campaign_run.get("checkpoint"), checkpoint, run_dir, label=f"{name} campaign checkpoint"
            )
            _assert_path_reference(
                campaign_run.get("manifest"), manifest_path, run_dir, label=f"{name} campaign manifest"
            )
            mode_contract = mode_contracts[mode]
            if (
                manifest.get("normalization_sha256")
                != mode_contract["normalization_sha256"]
                or manifest.get("input_quality_contract", {}).get("sha256")
                != mode_contract["input_quality_sha256"]
            ):
                _fail(f"Run-level normalization/input-quality SHA mismatch: {manifest_path}")
            if carry_entry is not None:
                if (
                    carry_entry.get("normalization_sha256")
                    != mode_contract["normalization_sha256"]
                    or carry_entry.get("input_quality_contract_sha256")
                    != mode_contract["input_quality_sha256"]
                    or _artifact_tree_binding(run_dir)
                    != carry_entry.get("artifact_binding")
                ):
                    _fail(f"Carry-forward artifact binding changed for {name}.")
            amp = _audit_amp(
                manifest, run_dir, carry_forward_entry=carry_entry
            )
            total_overflows += amp["amp_overflow_count"]
            export_reports: dict[str, Any] = {}
            split_ids: dict[str, set[str]] = {}
            for split, expected_count in expectation.split_counts.items():
                export = run_dir / "probabilities" / split
                _assert_path_reference(
                    manifest.get("probability_exports", {}).get(split),
                    export,
                    run_dir,
                    label=f"{name}/{split} run export",
                )
                campaign_export_key = (
                    "validation_export"
                    if split == "validation"
                    else "test_export"
                    if split == "test"
                    else "bolivia_holdout_export"
                )
                _assert_path_reference(
                    campaign_run.get(campaign_export_key),
                    export,
                    run_dir,
                    label=f"{name}/{split} campaign export",
                )
                report = _audit_export(
                    export=export,
                    split=split,
                    mode=mode,
                    expected_count=int(expected_count),
                    manifest=manifest,
                    input_quality_sha=mode_contract["input_quality_sha256"],
                )
                split_ids[split] = set(report["sample_ids"])
                if split not in reference_ids:
                    reference_ids[split] = report["sample_ids"]
                    reference_targets[split] = report["target_sha256"]
                elif (
                    report["sample_ids"] != reference_ids[split]
                    or report["target_sha256"] != reference_targets[split]
                ):
                    _fail(
                        f"Cross-model sample order/target mismatch for {name}/{split}."
                    )
                export_reports[split] = {
                    key: value
                    for key, value in report.items()
                    if key != "sample_ids"
                }
            if any(
                split_ids[left] & split_ids[right]
                for left, right in (
                    ("validation", "test"),
                    ("validation", "bolivia_holdout"),
                    ("test", "bolivia_holdout"),
                )
            ):
                _fail(f"Split sample overlap detected for {name}.")
            for metric_key, run_metric_key in (
                ("validation_iou", "best_validation_iou"),
                ("test_iou", None),
                ("bolivia_holdout_iou", None),
            ):
                expected_metric = (
                    manifest[run_metric_key]
                    if run_metric_key
                    else manifest["split_metrics"][
                        "test" if metric_key == "test_iou" else "bolivia_holdout"
                    ]
                )
                if not math.isclose(
                    float(campaign_run[metric_key]),
                    float(expected_metric),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    _fail(f"Campaign metric disagrees with run manifest for {name}/{metric_key}.")
            run_reports[name] = {
                "run_manifest_sha256": file_sha256(manifest_path),
                "checkpoint_sha256": file_sha256(checkpoint),
                "amp": amp,
                "carry_forward": is_carried,
                "exports": export_reports,
            }

    downstream: dict[str, Any] = {
        "supervised_outputs_ready_for_common_finalization": True,
        "gpu_runtime_reexecuted_by_this_audit": False,
        "claim_scope": (
            "Artifact and static entrypoint readiness only; this read-only audit "
            "does not execute TerraMind, Prithvi, or the 19-model GPU campaign."
        ),
    }
    if repository_root is not None:
        repo = repository_root.resolve()
        required = {
            "extended_panel": "scripts/colab/finalize_sen1_extended_panel_colab.py",
            "terramind": "scripts/colab/run_terramind_sen1floods11_final_colab.py",
            "prithvi": "scripts/colab/run_prithvi_sen1_geobwer_migration_colab.py",
            "protocol": "configs/geobwer/sen1floods11.yaml",
        }
        downstream["entrypoints"] = {
            key: {"path": relative, "exists": (repo / relative).is_file()}
            for key, relative in required.items()
        }
        if not all(item["exists"] for item in downstream["entrypoints"].values()):
            _fail("One or more downstream Sen1 formal entrypoints are missing.")

    return {
        "schema": "geobwer.sen1floods11.unet_artifact_audit.v1",
        "status": "pass",
        "target": {
            "package_version": expectation.version,
            "code_commit": expectation.commit,
            "source_root": str(root),
            "campaign_manifest_sha256": file_sha256(campaign_path),
        },
        "formal_evidence": True,
        "model_count": len(run_reports),
        "modes": list(expectation.modes),
        "seeds": list(expectation.seeds),
        "split_counts": dict(expectation.split_counts),
        "evaluation_sample_count": (
            int(expectation.split_counts["test"])
            + int(expectation.split_counts["bolivia_holdout"])
        ),
        "no_training_or_calibration_leakage": True,
        "bolivia_holdout_used_for_training_or_selection": False,
        "cross_model_sample_and_target_identity": "exact",
        "total_amp_overflow_count": total_overflows,
        "carry_forward": carry_summary,
        "mode_contracts": mode_contracts,
        "runs": run_reports,
        "downstream_readiness": downstream,
        "blocking_errors": [],
        "scientific_interpretation": {
            "artifact_integrity": "formal_pass",
            "checkpoint_scope": (
                "Checkpoint bytes and SHA-256 are verified; pickle checkpoint "
                "contents are deliberately not deserialized by the read-only auditor."
            ),
            "result_scope": (
                "This audit establishes completeness, lineage, numerical probability "
                "validity, target identity, and leakage contracts. It does not by "
                "itself establish model quality or comparative scientific conclusions."
            ),
        },
    }


def audit_sen1_unet_v0428_artifacts(
    source_root: str | Path,
    *,
    output_json: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the immutable v0.4.28 nine-run U-Net panel and write external evidence."""

    source = Path(source_root).resolve()
    destination = Path(output_json).resolve()
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        _fail("Audit evidence must be written outside the frozen source root.")
    if destination.exists():
        _fail(f"Refusing to overwrite existing audit evidence: {destination}")
    report = _audit_engine(
        source,
        expectation=FORMAL_EXPECTATION,
        repository_root=Path(repository_root) if repository_root else None,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


__all__ = [
    "Sen1UNetArtifactAuditError",
    "audit_sen1_unet_v0428_artifacts",
]
