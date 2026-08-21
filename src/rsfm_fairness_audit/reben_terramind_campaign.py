from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.reben import (
    LmdbSafetensorsRebenDatasetAdapter,
    detect_lmdb_payload_format,
    resolve_reben_root_dir,
)
from rsfm_fairness_audit.adapters.terramind import TerraMindAdapter
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.formal_outputs import FormalOutputBundle, file_sha256, write_multilabel_bundle
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.geobwer_extensions import run_multilabel_uncertainty_suite
from rsfm_fairness_audit.io import read_csv_rows
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output
from rsfm_fairness_audit.spatial_conformal import SpatialConformalConfig
from rsfm_fairness_audit.reben_sensor_audit import default_reben_class_names, select_thresholds_from_validation


class RebenTerraMindError(RuntimeError):
    """Raised when the formal reBEN/TerraMind train-calibrate-test chain is invalid."""


@dataclass(frozen=True)
class RebenTerraMindConfig:
    lmdb_root: Path
    metadata_parquet: Path
    output_dir: Path
    sensor_mode: str
    terramind_checkpoint_path: Path
    persistent_output_dir: Path | None = None
    embedding_cache_root: Path | None = None
    persistent_embedding_cache_root: Path | None = None
    metadata_snow_cloud_parquet: Path | None = None
    geobwer_protocol: Path = Path("configs/geobwer/reben.yaml")
    device: str = "auto"
    batch_size: int = 64
    embedding_chunk_size: int = 4096
    probe_epochs: int = 100
    probe_learning_rate: float = 1e-2
    probe_weight_decay: float = 1e-4
    probe_batch_size: int = 512
    seed: int = 42
    max_samples: int | None = None
    n_bootstrap: int = 2000
    s1_unit_policy: str = "already_db"
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        if self.sensor_mode not in {"S1", "S2", "S1+S2"}:
            raise ValueError("sensor_mode must be S1, S2, or S1+S2.")
        if self.batch_size <= 0 or self.embedding_chunk_size <= 0 or self.probe_batch_size <= 0:
            raise ValueError("Batch/chunk sizes must be positive.")
        if self.probe_epochs <= 0:
            raise ValueError("probe_epochs must be positive.")
        if self.s1_unit_policy not in {"already_db", "linear_power_to_db", "linear_amplitude_to_db"}:
            raise ValueError("Unsupported S1 unit policy.")
        if self.max_samples is not None and not self.diagnostic_only:
            raise ValueError("max_samples is permitted only with diagnostic_only=True; subsets cannot be formal evidence.")
        if self.diagnostic_only and self.max_samples is None:
            raise ValueError("diagnostic_only requires an explicit max_samples limit.")


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _signature(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(_native(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset(config: RebenTerraMindConfig, split: str):
    _, lmdb_path, _ = resolve_reben_root_dir(config.lmdb_root)
    payload = detect_lmdb_payload_format(lmdb_path)
    common = {
        "split": split,
        "sensor_mode": config.sensor_mode,
        "max_samples": config.max_samples,
        "channel_profile": str(getattr(config, "channel_profile", "croma")),
    }
    if payload == "safetensors":
        return LmdbSafetensorsRebenDatasetAdapter(
            config.lmdb_root,
            config.metadata_parquet,
            config.metadata_snow_cloud_parquet,
            **common,
        )
    raise RebenTerraMindError(
        f"TerraMind formal runs require the raw-band LMDB+safetensors path, observed payload={payload!r}. "
        "The ConfigILM loader may apply model-specific transforms before this adapter and would make TerraMind "
        "standardization ambiguous. Convert/verify the official raw bands instead of double-normalizing them."
    )


def _sample_id_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("sample_id", row.get("patch_id", ""))).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_reben_dataset_lineage(
    rows: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    *,
    metadata_parquet: str | Path,
) -> dict[str, Any]:
    """Build one model- and sensor-independent reBEN evaluation identity."""

    values = np.asarray(targets, dtype=np.int8)
    if values.shape != (len(rows), 19):
        raise RebenTerraMindError(
            f"Expected aligned reBEN targets [N,19], got {values.shape} for {len(rows)} rows."
        )
    digest = hashlib.sha256()
    for row, labels in zip(rows, values):
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(labels.tobytes(order="C"))
    return {
        "dataset": "BigEarthNet-v2.0/reBEN",
        "metadata_parquet_sha256": file_sha256(metadata_parquet),
        "split": "test",
        "test_sample_id_hash": _sample_id_hash(rows),
        "reference_targets_sha256": digest.hexdigest(),
        "source_tile_definition": "MGRS_100km_tile_parsed_from_official_patch_id",
        "sample_count": len(rows),
    }


def extract_reben_embeddings_chunked(
    dataset: Any,
    adapter: Any,
    output_dir: str | Path,
    *,
    batch_size: int,
    chunk_size: int,
    persistent_output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Resume-safe extraction followed by memory-mapped consolidation."""

    output = Path(output_dir)
    hydrate_output(output, persistent_output_dir)
    chunk_dir = output / "embedding_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = dataset.load_metadata()
    if not metadata_rows:
        raise RebenTerraMindError("No reBEN rows survived the requested split/profile.")
    adapter.load_model()
    adapter_lineage = adapter.provenance()
    adapter_lineage.pop("preprocessing_report", None)
    lineage = {
        "adapter": adapter_lineage,
        "dataset": dataset.loader_info() if hasattr(dataset, "loader_info") else {},
        "sample_count": len(metadata_rows),
        "sample_id_hash": _sample_id_hash(metadata_rows),
        "batch_size": batch_size,
        "chunk_size": chunk_size,
    }
    cache_signature = _signature(lineage)
    manifest_path = output / "embedding_cache_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("cache_signature") != cache_signature:
            raise RebenTerraMindError(
                f"Embedding cache signature changed under {output}. Use a new output directory; do not mix protocols."
            )

    chunk_paths: list[Path] = []
    for chunk_start in range(0, len(metadata_rows), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(metadata_rows))
        chunk_path = chunk_dir / f"chunk_{chunk_start:09d}_{chunk_end:09d}.npz"
        if chunk_path.exists():
            try:
                with np.load(chunk_path, allow_pickle=False) as cached:
                    valid = (
                        cached["embeddings"].shape[0] == chunk_end - chunk_start
                        and cached["labels"].shape == (chunk_end - chunk_start, 19)
                        and str(cached["cache_signature"].item()) == cache_signature
                    )
            except Exception:
                valid = False
            if valid:
                print(f"[reben:terramind] reusing embedding chunk {chunk_path.name}")
                chunk_paths.append(chunk_path)
                continue
            raise RebenTerraMindError(f"Invalid/stale embedding chunk found: {chunk_path}; use a new output directory.")
        chunk_embeddings: list[np.ndarray] = []
        chunk_labels: list[np.ndarray] = []
        chunk_metadata: list[dict[str, Any]] = []
        for start in range(chunk_start, chunk_end, batch_size):
            end = min(start + batch_size, chunk_end)
            samples = [dataset.load_sample(index) for index in range(start, end)]
            prepared = adapter.preprocess(
                {"samples": samples, "metadata": [sample["metadata"] for sample in samples]}
            )
            chunk_embeddings.append(adapter.extract_embeddings(prepared))
            chunk_labels.extend(
                np.asarray(sample["metadata"]["label_vector"], dtype=np.int8) for sample in samples
            )
            chunk_metadata.extend(dict(sample["metadata"]) for sample in samples)
            print(f"[reben:terramind] embeddings {end}/{len(metadata_rows)}")
        embeddings = np.vstack(chunk_embeddings).astype(np.float32)
        labels = np.vstack(chunk_labels).astype(np.int8)
        if embeddings.shape[0] != chunk_end - chunk_start or labels.shape != (chunk_end - chunk_start, 19):
            raise RebenTerraMindError("Embedding chunk alignment failed before writing.")
        np.savez_compressed(
            chunk_path,
            embeddings=embeddings,
            labels=labels,
            metadata_json=np.asarray(
                [json.dumps(_native(row), ensure_ascii=False, sort_keys=True) for row in chunk_metadata], dtype=str
            ),
            cache_signature=np.asarray(cache_signature),
        )
        chunk_paths.append(chunk_path)
        print(f"[reben:terramind] wrote {chunk_path.name}")
        persist_output(output, persistent_output_dir, label=f"embedding-chunk-{chunk_start}-{chunk_end}")

    embedding_path = output / "embeddings.npy"
    labels_path = output / "labels.npy"
    metadata_path = output / "metadata.jsonl"
    dimensions: set[int] = set()
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as chunk:
            dimensions.add(int(chunk["embeddings"].shape[1]))
    if len(dimensions) != 1:
        raise RebenTerraMindError(f"Inconsistent TerraMind embedding dimensions: {sorted(dimensions)}")
    dimension = dimensions.pop()
    embeddings_mm = np.lib.format.open_memmap(
        embedding_path, mode="w+", dtype=np.float32, shape=(len(metadata_rows), dimension)
    )
    labels_mm = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.int8, shape=(len(metadata_rows), 19))
    cursor = 0
    with metadata_path.open("w", encoding="utf-8") as metadata_handle:
        for path in chunk_paths:
            with np.load(path, allow_pickle=False) as chunk:
                count = int(chunk["embeddings"].shape[0])
                embeddings_mm[cursor : cursor + count] = chunk["embeddings"]
                labels_mm[cursor : cursor + count] = chunk["labels"]
                for value in chunk["metadata_json"]:
                    metadata_handle.write(str(value) + "\n")
                cursor += count
    embeddings_mm.flush()
    labels_mm.flush()
    del embeddings_mm, labels_mm
    manifest = {
        "schema": "geobwer.reben.embedding_cache.v1",
        "cache_signature": cache_signature,
        "lineage": lineage,
        "observed_preprocessing_report": adapter.provenance().get("preprocessing_report", {}),
        "embedding_shape": [len(metadata_rows), dimension],
        "labels_shape": [len(metadata_rows), 19],
        "chunks": [str(path) for path in chunk_paths],
        "artifacts": {
            "embeddings": str(embedding_path),
            "labels": str(labels_path),
            "metadata": str(metadata_path),
        },
    }
    manifest_path.write_text(json.dumps(_native(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    persist_output(output, persistent_output_dir, label="embedding-split-complete")
    return {
        "embeddings": embedding_path,
        "labels": labels_path,
        "metadata": metadata_path,
        "manifest": manifest_path,
    }


def _streaming_mean_std(values: np.ndarray, *, batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros(values.shape[1], dtype=np.float64)
    square = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, len(values), batch_size):
        chunk = np.asarray(values[start : start + batch_size], dtype=np.float64)
        total += chunk.sum(axis=0)
        square += np.square(chunk).sum(axis=0)
        count += len(chunk)
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def train_streaming_multilabel_probe(
    train_embeddings_path: str | Path,
    train_labels_path: str | Path,
    evaluation_embeddings: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    seed: int,
    cache_signature: str | None = None,
    train_indices: Sequence[int] | np.ndarray | None = None,
) -> tuple[dict[str, Path], Path]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - Colab path
        raise RebenTerraMindError("PyTorch is required for the reBEN linear probe.") from exc
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "linear_probe.pt"
    probability_paths = {str(split): output / f"{split}_probabilities.npy" for split in evaluation_embeddings}
    probe_manifest_path = output / "probe_manifest.json"
    probe_signature = _signature(
        {
            "cache_signature": cache_signature or "",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": seed,
        }
    )
    if probe_manifest_path.exists():
        manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("probe_signature") != probe_signature:
            raise RebenTerraMindError(
                f"Probe protocol changed under {output}; use a new output directory instead of mixing checkpoints."
            )
        if checkpoint_path.exists() and all(path.exists() for path in probability_paths.values()):
            expected = {
                split: len(np.load(path, mmap_mode="r")) for split, path in evaluation_embeddings.items()
            }
            observed = {split: np.load(path, mmap_mode="r").shape for split, path in probability_paths.items()}
            if all(observed[split] == (expected[split], 19) for split in expected):
                print("[reben:terramind:probe] reusing frozen probe and complete probability outputs")
                return probability_paths, checkpoint_path

    x_train = np.load(train_embeddings_path, mmap_mode="r")
    y_train = np.load(train_labels_path, mmap_mode="r")
    if x_train.ndim != 2 or y_train.shape != (len(x_train), 19):
        raise RebenTerraMindError("Train embedding/label cache shapes are invalid.")
    selected_indices = (
        np.arange(len(x_train), dtype=np.int64)
        if train_indices is None
        else np.asarray(train_indices, dtype=np.int64)
    )
    if selected_indices.ndim != 1 or len(selected_indices) == 0:
        raise RebenTerraMindError("train_indices must select at least one training row.")
    if np.any(selected_indices < 0) or np.any(selected_indices >= len(x_train)):
        raise RebenTerraMindError("train_indices contain an out-of-range row.")
    print(
        f"[reben:terramind:probe] computing train normalization "
        f"selected_rows={len(selected_indices)} embedding_dim={x_train.shape[1]}"
    )
    count = 0
    total = np.zeros(x_train.shape[1], dtype=np.float64)
    square = np.zeros(x_train.shape[1], dtype=np.float64)
    for start in range(0, len(selected_indices), 8192):
        chunk = np.asarray(x_train[selected_indices[start : start + 8192]], dtype=np.float64)
        total += chunk.sum(axis=0)
        square += np.square(chunk).sum(axis=0)
        count += len(chunk)
    mean64 = total / count
    variance64 = np.maximum(square / count - np.square(mean64), 1e-12)
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance64).astype(np.float32)
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RebenTerraMindError("CUDA probe training requested but unavailable.")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = nn.Linear(x_train.shape[1], 19).to(torch_device)
    device_probe = torch.as_tensor(
        (np.asarray(x_train[selected_indices[:1]], dtype=np.float32) - mean) / std,
        device=torch_device,
    )
    gpu_name = torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else "none"
    print(
        f"[reben:terramind:probe] resolved_device={torch_device} gpu_name={gpu_name} "
        f"model_parameter_device={next(model.parameters()).device} input_tensor_device={device_probe.device}"
    )
    del device_probe
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float | int]] = []
    indices = selected_indices.copy()
    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        losses: list[float] = []
        model.train()
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_x = (np.asarray(x_train[batch_indices], dtype=np.float32) - mean) / std
            batch_y = np.asarray(y_train[batch_indices], dtype=np.float32)
            xb = torch.as_tensor(batch_x, device=torch_device)
            yb = torch.as_tensor(batch_y, device=torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_bce_loss": epoch_loss})
        print(f"[reben:terramind:probe] epoch={epoch}/{epochs} bce={epoch_loss:.6f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_mean": mean,
            "embedding_std": std,
            "embedding_dim": int(x_train.shape[1]),
            "label_count": 19,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "train_sample_count": int(len(selected_indices)),
            "history": history,
        },
        checkpoint_path,
    )
    model.eval()
    with torch.inference_mode():
        for split, path in evaluation_embeddings.items():
            values = np.load(path, mmap_mode="r")
            target_path = output / f"{split}_probabilities.npy"
            target = np.lib.format.open_memmap(
                target_path, mode="w+", dtype=np.float32, shape=(len(values), 19)
            )
            for start in range(0, len(values), batch_size):
                end = min(start + batch_size, len(values))
                normalized = (np.asarray(values[start:end], dtype=np.float32) - mean) / std
                logits = model(torch.as_tensor(normalized, device=torch_device))
                target[start:end] = torch.sigmoid(logits).cpu().numpy()
            target.flush()
            del target
            probability_paths[str(split)] = target_path
    probe_manifest_path.write_text(
        json.dumps(
            {
                "schema": "geobwer.reben.streaming_probe.v1",
                "probe_signature": probe_signature,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "probabilities": {split: str(path) for split, path in probability_paths.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return probability_paths, checkpoint_path


def _read_metadata_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        for required in ("sample_id", "country", "source_tile_id", "independent_unit_id"):
            if str(row.get(required, "")).strip() == "":
                raise RebenTerraMindError(f"Formal reBEN metadata is missing {required} for sample={row.get('sample_id')}.")
    return rows


def _write_split_contract(
    split_artifacts: Mapping[str, Mapping[str, Path]],
    output_path: str | Path,
) -> Path:
    metadata = {
        split: _read_metadata_jsonl(artifacts["metadata"])
        for split, artifacts in split_artifacts.items()
    }
    sample_sets = {
        split: {str(row["sample_id"]) for row in rows}
        for split, rows in metadata.items()
    }
    overlaps: dict[str, list[str]] = {}
    names = sorted(sample_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(sample_sets[left] & sample_sets[right])
            overlaps[f"{left}__{right}"] = overlap
    if any(overlaps.values()):
        raise RebenTerraMindError(f"Sample leakage exists across official reBEN splits: {overlaps}")
    tile_sets = {
        split: {str(row["source_tile_id"]) for row in rows}
        for split, rows in metadata.items()
    }
    tile_overlaps = {
        f"{left}__{right}": sorted(tile_sets[left] & tile_sets[right])
        for left_index, left in enumerate(names)
        for right in names[left_index + 1 :]
    }
    payload = {
        "schema": "geobwer.reben.split_contract.v1",
        "sample_disjoint": True,
        "split_counts": {split: len(rows) for split, rows in metadata.items()},
        "sample_id_hashes": {split: _sample_id_hash(rows) for split, rows in metadata.items()},
        "source_tile_overlap_counts": {key: len(value) for key, value in tile_overlaps.items()},
        "source_tile_overlap_interpretation": (
            "reported_not_failed: official reBEN sensor-modality evaluation is not claimed as location-disjoint; "
            "source_tile_id remains the dependence cluster"
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def run_reben_frozen_adapter_campaign(
    config: Any,
    *,
    adapter: Any,
    model_name: str,
    campaign_schema: str,
    adaptation_protocol: str = "frozen_encoder_streaming_linear_multilabel_probe",
) -> dict[str, Path]:
    output = config.output_dir
    hydrate_output(output, config.persistent_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(getattr(config, "embedding_cache_root", None) or (output / "cache"))
    persistent_cache_root_raw = getattr(config, "persistent_embedding_cache_root", None)
    persistent_cache_root = (
        Path(persistent_cache_root_raw)
        if persistent_cache_root_raw is not None
        else (config.persistent_output_dir / "cache" if config.persistent_output_dir is not None else None)
    )
    split_artifacts: dict[str, dict[str, Path]] = {}
    for split in ("train", "val", "test"):
        split_artifacts[split] = extract_reben_embeddings_chunked(
            _dataset(config, split),
            adapter,
            cache_root / split,
            batch_size=config.batch_size,
            chunk_size=config.embedding_chunk_size,
            persistent_output_dir=(
                persistent_cache_root / split
                if persistent_cache_root is not None
                else None
            ),
        )
    cache_signatures = {
        split: json.loads(paths["manifest"].read_text(encoding="utf-8"))["cache_signature"]
        for split, paths in split_artifacts.items()
    }
    split_contract = _write_split_contract(split_artifacts, output / "split_contract.json")
    persist_output(output, config.persistent_output_dir, label="split-contract")
    probabilities, probe_checkpoint = train_streaming_multilabel_probe(
        split_artifacts["train"]["embeddings"],
        split_artifacts["train"]["labels"],
        {
            "validation": split_artifacts["val"]["embeddings"],
            "test": split_artifacts["test"]["embeddings"],
        },
        output / "probe",
        epochs=config.probe_epochs,
        learning_rate=config.probe_learning_rate,
        weight_decay=config.probe_weight_decay,
        batch_size=config.probe_batch_size,
        device=config.device,
        seed=config.seed,
        cache_signature=_signature(cache_signatures),
    )
    persist_output(output, config.persistent_output_dir, label="probe-complete")
    if config.diagnostic_only:
        diagnostic_manifest = output / "diagnostic_manifest.json"
        diagnostic_manifest.write_text(
            json.dumps(
                {
                    "schema": "geobwer.reben.frozen_adapter_diagnostic.v1",
                    "formal_evidence": False,
                    "reason": "explicit_bounded_runtime_smoke",
                    "sensor_mode": config.sensor_mode,
                    "max_samples_per_split": config.max_samples,
                    "split_contract": str(split_contract),
                    "probe_checkpoint": str(probe_checkpoint),
                    "probabilities": {key: str(value) for key, value in probabilities.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        persist_output(output, config.persistent_output_dir, label="diagnostic-complete")
        return {
            "diagnostic_manifest": diagnostic_manifest,
            "probe_checkpoint": probe_checkpoint,
            "split_contract": split_contract,
        }
    y_val = np.load(split_artifacts["val"]["labels"], mmap_mode="r")
    p_val = np.load(probabilities["validation"], mmap_mode="r")
    thresholds = select_thresholds_from_validation(y_val, p_val)
    calibration_path = output / "calibration_probabilities.npz"
    np.savez_compressed(
        calibration_path,
        probabilities=np.asarray(p_val),
        targets=np.asarray(y_val),
        thresholds=np.asarray(thresholds, dtype=np.float32),
        sample_id=np.asarray([row["sample_id"] for row in _read_metadata_jsonl(split_artifacts["val"]["metadata"])]),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    calibration_manifest = output / "calibration_manifest.json"
    calibration_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.reben.multilabel_calibration.v1",
                "split_role": "calibration",
                "split": "validation",
                "threshold_policy": "per_label_max_validation_f1",
                "thresholds": np.asarray(thresholds).tolist(),
                "probabilities_sha256": file_sha256(calibration_path),
                "test_rows_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    persist_output(output, config.persistent_output_dir, label="calibration-complete")
    protocol = BWERProtocol.from_mapping(load_yaml(config.geobwer_protocol))
    test_rows = _read_metadata_jsonl(split_artifacts["test"]["metadata"])
    p_test = np.load(probabilities["test"], mmap_mode="r")
    y_test = np.load(split_artifacts["test"]["labels"], mmap_mode="r")
    model_lineage = {
        **adapter.provenance(),
        "adaptation_protocol": adaptation_protocol,
        "probe_checkpoint": str(probe_checkpoint),
        "probe_checkpoint_sha256": file_sha256(probe_checkpoint),
        "threshold_calibration": str(calibration_manifest),
        "threshold_calibration_sha256": file_sha256(calibration_manifest),
    }
    dataset_lineage = build_reben_dataset_lineage(
        test_rows,
        y_test,
        metadata_parquet=config.metadata_parquet,
    )
    bundle: FormalOutputBundle = write_multilabel_bundle(
        output / "formal_outputs",
        sample_rows=test_rows,
        probabilities=p_test,
        targets=y_test,
        class_names=default_reben_class_names(),
        dataset="reben",
        model=model_name,
        split="test",
        protocol=protocol,
        model_lineage=model_lineage,
        dataset_lineage=dataset_lineage,
        threshold=thresholds,
        independent_unit_column="independent_unit_id",
        split_role="evaluation",
    )
    audit = audit_rows(
        read_csv_rows(bundle.audit_table),
        group_columns=("country",),
        protocol=protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="source_tile_id",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.n_bootstrap,
        seed=config.seed,
    )
    audit_artifacts = audit.to_report(output / "geobwer")
    uncertainty_artifacts = run_multilabel_uncertainty_suite(
        calibration_path,
        bundle.output_dir,
        output / "uncertainty_extensions",
        protocol=protocol,
        group_columns=("country",),
        calibration_manifest=calibration_manifest,
        n_bootstrap=config.n_bootstrap,
        seed=config.seed,
        spatial_localization_config=SpatialConformalConfig(),
    )
    run_manifest = output / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            _native(
                {
                    "schema": campaign_schema,
                    "config": asdict(config),
                    "model_name": model_name,
                    "formal_output_manifest": bundle.manifest,
                    "calibration_manifest": calibration_manifest,
                    "split_contract": split_contract,
                    "geobwer_artifacts": audit_artifacts,
                    "uncertainty_artifacts": uncertainty_artifacts,
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    persist_output(output, config.persistent_output_dir, label="formal-campaign-complete")
    return {
        "formal_audit_table": bundle.audit_table,
        "formal_output_manifest": bundle.manifest,
        "calibration_probabilities": calibration_path,
        "calibration_manifest": calibration_manifest,
        "probe_checkpoint": probe_checkpoint,
        "geobwer_summary": audit_artifacts["summary"],
        "uncertainty_summary": uncertainty_artifacts["summary"],
        "run_manifest": run_manifest,
        "split_contract": split_contract,
    }


def run_reben_terramind_campaign(config: RebenTerraMindConfig) -> dict[str, Path]:
    slug = config.sensor_mode.lower().replace("+", "_plus_")
    adapter = TerraMindAdapter(
        sensor_mode=config.sensor_mode,
        input_profile="reben_l2a",
        model_name="terramind_v1_base",
        model_release="terramind_v1_base_iccv2025",
        device=config.device,
        image_size=224,
        merge_method="mean",
        embedding_pooling="mean_tokens",
        s1_unit_policy=config.s1_unit_policy,
        strict_range_check=True,
        pretrained=True,
        checkpoint_path=config.terramind_checkpoint_path,
    )
    return run_reben_frozen_adapter_campaign(
        config,
        adapter=adapter,
        model_name=f"terramind_v1_base_{slug}_seed_{config.seed}",
        campaign_schema="geobwer.reben.terramind_campaign.v3",
    )


__all__ = [
    "RebenTerraMindConfig",
    "RebenTerraMindError",
    "build_reben_dataset_lineage",
    "extract_reben_embeddings_chunked",
    "run_reben_frozen_adapter_campaign",
    "run_reben_terramind_campaign",
    "train_streaming_multilabel_probe",
]
