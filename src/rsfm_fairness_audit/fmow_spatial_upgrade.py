from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows


class FmowSpatialUpgradeError(RuntimeError):
    """Raised when a frozen fMoW calibration artifact cannot be identified safely."""


def _row_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("sample_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("image_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            str(row.get("image_path", row.get("extracted_path", ""))).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _split_role(row: Mapping[str, Any]) -> str:
    return str(row.get("split_role", row.get("split", ""))).strip().lower()


def _label(row: Mapping[str, Any]) -> str:
    for key in ("category", "class_label", "class", "label"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _coordinate(row: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            try:
                result = float(value)
            except ValueError:
                continue
            if np.isfinite(result):
                return result
    return float("nan")


def _assignment_hash(
    sample_ids: Sequence[str],
    targets: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> str:
    payload = [
        [
            str(sample_id),
            int(target),
            format(float(lat), ".17g"),
            format(float(lon), ".17g"),
        ]
        for sample_id, target, lat, lon in zip(
            sample_ids, targets.tolist(), latitude.tolist(), longitude.tolist()
        )
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FmowSpatialUpgradeError(f"Required frozen manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FmowSpatialUpgradeError(f"Expected a JSON object: {path}")
    return value


def _calibration_cache(source_root: Path, run_manifest: Mapping[str, Any]) -> Path:
    recorded = (
        run_manifest.get("embedding_caches", {})
        .get("calibration", {})
        .get("path", "")
    )
    if recorded:
        candidate = source_root / "embedding_cache" / Path(str(recorded)).name
        if candidate.is_file():
            return candidate
    candidates = sorted((source_root / "embedding_cache").glob("dofa_calibration_*.npz"))
    if len(candidates) != 1:
        raise FmowSpatialUpgradeError(
            "Could not resolve exactly one frozen calibration embedding cache from "
            f"run lineage; candidates={len(candidates)}."
        )
    return candidates[0]


def _checkpoint_lineage(source_root: Path, seed: int) -> tuple[Path, Path, dict[str, Any]]:
    panel_path = source_root / "probe_panel_manifest.json"
    panel = _load_json(panel_path)
    component = dict(panel.get("components", {}).get(str(int(seed)), {}))
    seed_root = source_root / "probe_seeds" / f"seed_{int(seed)}"
    checkpoint = seed_root / "linear_probe.pt"
    selection = seed_root / "probe_selection_manifest.json"
    if not checkpoint.is_file() or not selection.is_file():
        raise FmowSpatialUpgradeError(
            f"Frozen seed={seed} checkpoint or selection manifest is missing."
        )
    expected_checkpoint = str(component.get("checkpoint_sha256", ""))
    expected_selection = str(component.get("selection_manifest_sha256", ""))
    if expected_checkpoint and file_sha256(checkpoint) != expected_checkpoint:
        raise FmowSpatialUpgradeError(
            f"Frozen seed={seed} checkpoint hash differs from probe panel lineage."
        )
    if expected_selection and file_sha256(selection) != expected_selection:
        raise FmowSpatialUpgradeError(
            f"Frozen seed={seed} selection manifest hash differs from probe panel lineage."
        )
    return checkpoint, selection, panel


def _replay_linear_probe(
    checkpoint_path: Path,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Colab and formal env include torch.
        raise FmowSpatialUpgradeError(
            "CPU replay of the frozen linear probe requires torch."
        ) from exc
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise FmowSpatialUpgradeError("Frozen linear probe checkpoint is not a mapping.")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or "weight" not in state or "bias" not in state:
        raise FmowSpatialUpgradeError(
            "Frozen linear probe checkpoint lacks state_dict weight/bias."
        )
    classes = tuple(str(value) for value in checkpoint.get("classes", ()))
    mean = np.asarray(checkpoint.get("embedding_mean"), dtype=np.float32)
    std = np.asarray(checkpoint.get("embedding_std"), dtype=np.float32)
    weight = np.asarray(state["weight"].detach().cpu(), dtype=np.float32)
    bias = np.asarray(state["bias"].detach().cpu(), dtype=np.float32)
    if (
        mean.shape != (1, embeddings.shape[1])
        or std.shape != mean.shape
        or weight.shape != (len(classes), embeddings.shape[1])
        or bias.shape != (len(classes),)
    ):
        raise FmowSpatialUpgradeError("Frozen checkpoint shapes do not match embeddings.")
    normalized = (np.asarray(embeddings, dtype=np.float32) - mean) / std
    logits = normalized @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return logits.astype(np.float32), probabilities.astype(np.float32), classes


def derive_legacy_dofa_calibration(
    source_root: str | Path,
    calibration_metadata_csv: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    test_formal_dir: str | Path,
    expected_count: int = 4485,
    expected_class_count: int = 62,
) -> tuple[Path, Path]:
    """Create an identity-bearing calibration bundle without changing frozen inputs.

    Row identity is certified twice: the frozen embedding cache carries sample IDs,
    and a CPU replay of the frozen probe must reproduce the legacy logits and
    probabilities in exactly that embedding order.
    """

    source = Path(source_root)
    metadata_path = Path(calibration_metadata_csv)
    output = ensure_dir(output_dir)
    seed_root = source / "probe_seeds" / f"seed_{int(seed)}"
    legacy_path = seed_root / "calibration_predictions.npz"
    if not legacy_path.is_file():
        raise FmowSpatialUpgradeError(
            f"Legacy seed={seed} calibration_predictions.npz is missing."
        )
    run_manifest_path = source / "run_manifest.json"
    run_manifest = _load_json(run_manifest_path)
    metadata_rows = read_csv_rows(metadata_path)
    calibration_rows = [row for row in metadata_rows if _split_role(row) == "calibration"]
    if len(calibration_rows) != int(expected_count):
        raise FmowSpatialUpgradeError(
            f"Expected {expected_count} calibration metadata rows, got {len(calibration_rows)}."
        )
    recorded_row_hash = str(
        run_manifest.get("dataset_lineage", {}).get("calibration_row_hash", "")
    )
    observed_row_hash = _row_hash(calibration_rows)
    if not recorded_row_hash or observed_row_hash != recorded_row_hash:
        raise FmowSpatialUpgradeError(
            "Calibration metadata order/hash does not match frozen run lineage."
        )
    metadata_ids = [str(row.get("sample_id", "")).strip() for row in calibration_rows]
    if (
        any(not sample_id for sample_id in metadata_ids)
        or len(set(metadata_ids)) != len(metadata_ids)
    ):
        raise FmowSpatialUpgradeError(
            "Calibration metadata sample_id values must be non-empty and unique."
        )

    cache_path = _calibration_cache(source, run_manifest)
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"embeddings", "labels", "sample_ids"}
        if not required.issubset(cache.files):
            raise FmowSpatialUpgradeError(
                "Frozen calibration embedding cache lacks embeddings/labels/sample_ids."
            )
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        cache_labels = tuple(str(value) for value in cache["labels"].tolist())
        cache_ids = tuple(str(value) for value in cache["sample_ids"].tolist())
    if cache_ids != tuple(metadata_ids):
        raise FmowSpatialUpgradeError(
            "Frozen calibration embedding sample_id order differs from metadata order."
        )
    metadata_labels = tuple(_label(row) for row in calibration_rows)
    if any(not value for value in metadata_labels) or cache_labels != metadata_labels:
        raise FmowSpatialUpgradeError(
            "Frozen calibration embedding labels differ from metadata labels/order."
        )

    with np.load(legacy_path, allow_pickle=False) as legacy:
        required = {"logits", "probabilities", "class_names"}
        if not required.issubset(legacy.files):
            raise FmowSpatialUpgradeError(
                "Legacy calibration predictions lack logits/probabilities/class_names."
            )
        legacy_logits = np.asarray(legacy["logits"], dtype=np.float32)
        probabilities = np.asarray(legacy["probabilities"], dtype=np.float32)
        legacy_classes = tuple(str(value) for value in legacy["class_names"].tolist())
    if (
        probabilities.shape != (expected_count, expected_class_count)
        or legacy_logits.shape != probabilities.shape
    ):
        raise FmowSpatialUpgradeError(
            f"Legacy calibration arrays have unexpected shape {probabilities.shape}."
        )
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=2e-4)
    ):
        raise FmowSpatialUpgradeError("Legacy probabilities are not valid multiclass probabilities.")

    checkpoint_path, selection_path, panel = _checkpoint_lineage(source, seed)
    replay_logits, replay_probabilities, checkpoint_classes = _replay_linear_probe(
        checkpoint_path, embeddings
    )
    if legacy_classes != checkpoint_classes:
        raise FmowSpatialUpgradeError(
            "Legacy class_names differ from the frozen checkpoint class mapping."
        )
    logits_difference = float(np.max(np.abs(legacy_logits - replay_logits)))
    probabilities_difference = float(
        np.max(np.abs(probabilities - replay_probabilities))
    )
    if not np.allclose(legacy_logits, replay_logits, atol=5e-4, rtol=5e-5):
        raise FmowSpatialUpgradeError(
            "CPU checkpoint replay does not reproduce legacy logits; "
            f"max_abs={logits_difference}."
        )
    if not np.allclose(probabilities, replay_probabilities, atol=2e-4, rtol=5e-5):
        raise FmowSpatialUpgradeError(
            "CPU checkpoint replay does not reproduce legacy probabilities; "
            f"max_abs={probabilities_difference}."
        )
    class_to_index = {name: index for index, name in enumerate(legacy_classes)}
    if set(metadata_labels) - set(class_to_index):
        raise FmowSpatialUpgradeError(
            "Calibration metadata contains labels absent from frozen class_names."
        )
    targets = np.asarray(
        [class_to_index[label] for label in metadata_labels], dtype=np.int64
    )
    latitude = np.asarray(
        [_coordinate(row, ("latitude", "lat")) for row in calibration_rows],
        dtype=np.float64,
    )
    longitude = np.asarray(
        [_coordinate(row, ("longitude", "lon", "lng")) for row in calibration_rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise FmowSpatialUpgradeError(
            "Frozen calibration metadata lacks finite latitude/longitude."
        )
    if np.any(latitude < -90.0) or np.any(latitude > 90.0):
        raise FmowSpatialUpgradeError("Calibration latitude lies outside [-90, 90].")
    if np.any(longitude < -180.0) or np.any(longitude > 180.0):
        raise FmowSpatialUpgradeError("Calibration longitude lies outside [-180, 180].")

    test_rows = read_csv_rows(Path(test_formal_dir) / "formal_audit_table.csv")
    test_ids = {str(row.get("sample_id", "")).strip() for row in test_rows}
    if not test_ids or set(metadata_ids) & test_ids:
        raise FmowSpatialUpgradeError(
            "Calibration/test sample IDs are empty or overlap."
        )

    derived_path = output / "calibration_probabilities_with_identity.npz"
    np.savez_compressed(
        derived_path,
        probabilities=probabilities,
        targets=targets,
        class_names=np.asarray(legacy_classes, dtype=str),
        sample_id=np.asarray(metadata_ids, dtype=str),
        latitude=latitude,
        longitude=longitude,
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    assignment_hash = _assignment_hash(metadata_ids, targets, latitude, longitude)
    derived_sha256 = file_sha256(derived_path)
    manifest_path = output / "legacy_calibration_derivation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.legacy_dofa_calibration_derivation.v1",
                "formal_evidence": True,
                "seed": int(seed),
                "sample_count": len(metadata_ids),
                "class_count": len(legacy_classes),
                "split_role": "calibration",
                "test_rows_used": False,
                "legacy_calibration_predictions_sha256": file_sha256(legacy_path),
                "metadata_sha256": file_sha256(metadata_path),
                "frozen_run_manifest_sha256": file_sha256(run_manifest_path),
                "frozen_calibration_row_hash": recorded_row_hash,
                "calibration_embedding_cache_sha256": file_sha256(cache_path),
                "probe_checkpoint_sha256": file_sha256(checkpoint_path),
                "probe_selection_manifest_sha256": file_sha256(selection_path),
                "probe_panel_manifest_sha256": file_sha256(
                    source / "probe_panel_manifest.json"
                ),
                "probe_panel_schema": panel.get("schema", ""),
                "order_proof": (
                    "metadata order == frozen embedding-cache sample_ids; "
                    "CPU replay of frozen checkpoint == legacy logits/probabilities"
                ),
                "cpu_replay_logits_allclose": True,
                "cpu_replay_probabilities_allclose": True,
                "cpu_replay_logits_max_abs_difference": logits_difference,
                "cpu_replay_probabilities_max_abs_difference": probabilities_difference,
                "exact_sample_id_assignment_hash": assignment_hash,
                "calibration_test_overlap_count": 0,
                "probabilities_sha256": derived_sha256,
                "derived_probabilities_sha256": derived_sha256,
                "source_artifacts_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return derived_path, manifest_path


def completion_signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_completion_contract(
    seed_output: str | Path,
    expected_signature: str,
) -> bool:
    root = Path(seed_output)
    contract_path = root / "completion_contract.json"
    if not root.exists():
        return False
    if not contract_path.is_file():
        if any(root.iterdir()):
            raise FmowSpatialUpgradeError(
                f"Partial spatial-upgrade output lacks completion contract: {root}"
            )
        return False
    contract = _load_json(contract_path)
    if str(contract.get("completion_signature", "")) != expected_signature:
        raise FmowSpatialUpgradeError(
            f"Spatial-upgrade completion signature mismatch: {root}"
        )
    for relative, expected_hash in dict(contract.get("artifacts", {})).items():
        path = root / str(relative)
        if not path.is_file() or file_sha256(path) != str(expected_hash):
            raise FmowSpatialUpgradeError(
                f"Spatial-upgrade completion artifact mismatch: {path}"
            )
    return True


def write_completion_contract(
    seed_output: str | Path,
    *,
    seed: int,
    signature_payload: Mapping[str, Any],
) -> Path:
    root = Path(seed_output)
    artifacts = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "completion_contract.json"
    }
    contract = root / "completion_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.dofav2_spatial_upgrade_completion.v1",
                "formal_evidence": True,
                "seed": int(seed),
                "completion_signature": completion_signature(signature_payload),
                "signature_payload": dict(signature_payload),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return contract


__all__ = [
    "FmowSpatialUpgradeError",
    "completion_signature",
    "derive_legacy_dofa_calibration",
    "validate_completion_contract",
    "write_completion_contract",
]
