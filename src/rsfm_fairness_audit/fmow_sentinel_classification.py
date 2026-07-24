from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.dofa import DOFAAdapter
from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_from_file
from rsfm_fairness_audit.audit_table import build_audit_table_from_predictions, validate_audit_table, write_audit_table
from rsfm_fairness_audit.band_profiles import get_band_profile
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.formal_outputs import file_sha256, write_multiclass_bundle
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


GEOGRAPHY_COLUMNS = (
    "timestamp",
    "year",
    "month",
    "season",
    "location_id",
    "latitude",
    "longitude",
    "country",
    "continent",
    "un_region",
    "region",
    "latitude_band",
)


@dataclass(frozen=True)
class FmowClassificationConfig:
    metadata_csv: Path
    output_dir: Path
    data_root: Path | None = None
    model: str = "supervised_stats"
    model_config: Path | None = None
    probe: str = "linear"
    probe_epochs: int = 200
    probe_learning_rate: float = 1e-2
    embedding_cache_dir: Path | None = None
    dofa_input_scale: float | None = None
    dofa_embedding_pooling: str | None = None
    train_split: str = "train"
    eval_split: str = "val"
    max_samples: int | None = None
    max_samples_per_split: int | None = None
    image_size: int = 96
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_metric: str = "macro_f1"
    num_workers: int = 2
    device: str = "auto"
    norm_stats: Path | None = None
    seed: int = 42
    split_protocol: str = "official_split"
    eval_scope: str = "val"
    band_profile: str = "sentinel2_13band_fmow"
    allow_torch_hub_download: bool = False
    amp: bool = True
    run_bwer: bool = False
    bwer_bootstrap: int = 0
    write_formal_outputs: bool = False
    geobwer_protocol: Path = Path("configs/geobwer/fmow_sentinel.yaml")


@dataclass(frozen=True)
class FmowBwerConfig:
    input_dir: Path
    output_dir: Path
    audit_table: Path | None = None
    bootstrap: int = 0
    seed: int = 42


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).lower() in {"nan", "none", "null"}


def _safe_float(value: Any) -> float | str:
    if _is_missing(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _derive_date_fields(row: dict[str, Any]) -> None:
    timestamp = str(row.get("timestamp", "") or "")
    if not timestamp:
        return
    if not row.get("year") and len(timestamp) >= 4:
        row["year"] = timestamp[:4]
    if not row.get("month") and len(timestamp) >= 7:
        try:
            month = int(timestamp[5:7])
        except ValueError:
            return
        row["month"] = str(month)
        if not row.get("season"):
            if month in {12, 1, 2}:
                row["season"] = "DJF"
            elif month in {3, 4, 5}:
                row["season"] = "MAM"
            elif month in {6, 7, 8}:
                row["season"] = "JJA"
            else:
                row["season"] = "SON"


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("sample_id", "image_id", "id"):
        if key in row and not _is_missing(row[key]):
            return str(row[key])
    return str(index)


def _resolve_image_path(row: Mapping[str, Any], data_root: Path | None) -> Path:
    values = [row.get("extracted_path"), row.get("extracted_image_path"), row.get("image_path"), row.get("raster_path"), row.get("path")]
    candidates: list[Path] = []
    for value in values:
        if _is_missing(value):
            continue
        path = Path(str(value))
        candidates.append(path)
        if not path.is_absolute() and data_root is not None:
            candidates.append(data_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    raise FileNotFoundError("row is missing extracted_path/image_path/raster_path/path")


def _read_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        data = np.load(path)
        for key in ("image", "chip", "arr_0", "x"):
            if key in data:
                return np.asarray(data[key])
        return np.asarray(data[data.files[0]])
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as src:
                return src.read()
        except ImportError:
            try:
                import tifffile
            except ImportError as exc:
                raise RuntimeError("Reading TIFF rasters requires rasterio or tifffile.") from exc
            return np.asarray(tifffile.imread(path))
    raise ValueError(f"Unsupported fMoW-Sentinel raster extension: {path.suffix}")


def _to_channels_first(array: np.ndarray, expected_bands: int) -> np.ndarray:
    if array.ndim == 2:
        raise ValueError("Expected 13-band image, got a single 2D array.")
    if array.ndim != 3:
        raise ValueError(f"Expected image shape [bands,H,W] or [H,W,bands], got {array.shape}.")
    if array.shape[0] == expected_bands:
        out = array
    elif array.shape[-1] == expected_bands:
        out = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"Expected {expected_bands} bands, got shape {array.shape}.")
    out = out.astype(np.float32, copy=False)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _resize_nearest(chip: np.ndarray, size: int) -> np.ndarray:
    if chip.shape[-2:] == (size, size):
        return chip
    h, w = chip.shape[-2:]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid spatial shape: {chip.shape}.")
    y = np.linspace(0, h - 1, size).round().astype(int)
    x = np.linspace(0, w - 1, size).round().astype(int)
    return chip[:, y][:, :, x]


def load_fmow_sentinel_image(
    row: Mapping[str, Any],
    data_root: Path | None = None,
    image_size: int = 96,
    band_profile: str = "sentinel2_13band_fmow",
) -> np.ndarray:
    profile = get_band_profile(band_profile)
    expected_bands = int(profile["expected_bands"])
    path = _resolve_image_path(row, data_root)
    if not path.exists():
        raise FileNotFoundError(path)
    source_expected = int(profile.get("source_expected_bands", expected_bands))
    chip = _to_channels_first(_read_array(path), source_expected)
    if "source_band_indices" in profile:
        chip = chip[np.asarray(profile["source_band_indices"], dtype=int)]
    if chip.shape[0] != expected_bands:
        raise ValueError(f"Band selection for {band_profile!r} produced {chip.shape[0]} bands; expected {expected_bands}.")
    return _resize_nearest(chip, image_size)


def _image_features(chip: np.ndarray) -> np.ndarray:
    flat = chip.reshape(chip.shape[0], -1)
    return np.concatenate(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            flat.min(axis=1),
            flat.max(axis=1),
        ]
    ).astype(np.float32)


class _NearestCentroidClassifier:
    def __init__(self) -> None:
        self.classes: list[str] = []
        self.centroids: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> None:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for feature, label in zip(features, labels):
            grouped[str(label)].append(feature)
        self.classes = sorted(grouped)
        self.centroids = np.stack([np.mean(grouped[label], axis=0) for label in self.classes]).astype(np.float32)

    def predict(self, features: np.ndarray) -> list[str]:
        if self.centroids is None:
            raise RuntimeError("classifier has not been fitted")
        diff = features[:, None, :] - self.centroids[None, :, :]
        distances = np.sum(diff * diff, axis=2)
        return [self.classes[int(index)] for index in np.argmin(distances, axis=1)]


def _limit_rows(rows: Sequence[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit and limit > 0 and len(rows) > limit:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(rows), size=limit, replace=False).tolist())
        return [dict(rows[index]) for index in indices]
    return list(rows)


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in read_csv_rows(path)]
    for index, row in enumerate(rows):
        row["sample_id"] = _sample_id(row, index)
        row["category"] = row.get("category") or row.get("label") or row.get("class_label") or ""
        row["split"] = row.get("split") or "all"
        _derive_date_fields(row)
    return rows


def _split_rows(rows: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return list(rows)
    return [row for row in rows if str(row.get("split", "")) == split]


def _rows_to_features(rows: Sequence[dict[str, Any]], config: FmowClassificationConfig) -> tuple[np.ndarray, list[str], list[dict[str, Any]], list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    ok_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in rows:
        try:
            chip = load_fmow_sentinel_image(row, config.data_root, config.image_size, config.band_profile)
        except Exception as exc:
            warnings.append(f"Skipping sample_id={row.get('sample_id')}: {exc}")
            continue
        features.append(_image_features(chip))
        labels.append(str(row.get("category", "")))
        ok_rows.append(row)
    if not features:
        raise ValueError("No readable fMoW-Sentinel images found for feature extraction.")
    return np.stack(features).astype(np.float32), labels, ok_rows, warnings


def _dofa_embeddings(
    rows: Sequence[dict[str, Any]],
    config: FmowClassificationConfig,
    adapter: DOFAAdapter,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]], list[str]]:
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    ok_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    batch_samples: list[dict[str, Any]] = []
    batch_meta: list[dict[str, Any]] = []
    for row in rows:
        try:
            chip = load_fmow_sentinel_image(row, config.data_root, config.image_size, config.band_profile)
        except Exception as exc:
            warnings.append(f"Skipping sample_id={row.get('sample_id')}: {exc}")
            continue
        batch_samples.append({"image": chip})
        batch_meta.append(row)
        if len(batch_samples) >= config.batch_size:
            prepared = adapter.preprocess({"samples": batch_samples, "metadata": batch_meta})
            embeddings.append(adapter.extract_embeddings(prepared))
            labels.extend(str(item.get("category", "")) for item in batch_meta)
            ok_rows.extend(batch_meta)
            batch_samples, batch_meta = [], []
    if batch_samples:
        prepared = adapter.preprocess({"samples": batch_samples, "metadata": batch_meta})
        embeddings.append(adapter.extract_embeddings(prepared))
        labels.extend(str(item.get("category", "")) for item in batch_meta)
        ok_rows.extend(batch_meta)
    if not embeddings:
        raise ValueError("No readable fMoW-Sentinel images found for DOFA embedding extraction.")
    return np.concatenate(embeddings, axis=0).astype(np.float32), labels, ok_rows, warnings


def _adapter_source(adapter: DOFAAdapter) -> str:
    if adapter.checkpoint_path is not None:
        return str(adapter.checkpoint_path)
    if adapter.repo_path is not None:
        return str(adapter.repo_path)
    return f"torch_hub:{adapter.torch_hub_repo}:{adapter.model_variant}:pretrained={adapter.allow_torch_hub_download}"


def _row_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("sample_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("image_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("image_path", row.get("extracted_path", ""))).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _dofa_cache_key(
    rows: Sequence[dict[str, Any]],
    split_name: str,
    config: FmowClassificationConfig,
    adapter: DOFAAdapter,
) -> str:
    _ = split_name
    payload = {
        "model_variant": adapter.model_variant,
        "image_size": config.image_size,
        "adapter_image_size": adapter.image_size,
        "band_profile": config.band_profile,
        "checkpoint_source": _adapter_source(adapter),
        "checkpoint_sha256": adapter.actual_checkpoint_sha256 or adapter.checkpoint_sha256 or "",
        "embedding_layer": adapter.embedding_layer,
        "embedding_pooling": adapter.embedding_pooling,
        "input_scale": adapter.input_scale,
        "row_count": len(rows),
        "row_hash": _row_hash(rows),
    }
    if adapter.model_variant == "dofav2_vit_base":
        payload.update(
            {
                "model_release": adapter.model_release,
                "repo_revision": adapter.actual_repo_revision or adapter.repo_revision or "",
                "architecture_source_revision": adapter.architecture_source_revision,
                "required_timm_version": adapter.required_timm_version,
                "patch_size": adapter.official_dofav2_patch_size,
                "embedding_semantics": adapter.official_dofav2_embedding_semantics,
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _cached_dofa_embeddings(
    rows: Sequence[dict[str, Any]],
    split_name: str,
    config: FmowClassificationConfig,
    adapter: DOFAAdapter,
    output_dir: Path,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]], list[str], Path, dict[str, Any]]:
    cache_dir = ensure_dir(config.embedding_cache_dir or output_dir / "embedding_cache")
    key = _dofa_cache_key(rows, split_name, config, adapter)
    cache_path = cache_dir / f"dofa_{split_name}_{key}.npz"
    metadata_path = cache_dir / f"dofa_{split_name}_{key}.json"
    if not (cache_path.exists() and metadata_path.exists()):
        for candidate in sorted(cache_dir.glob(f"dofa_*_{key}.npz")):
            candidate_metadata = candidate.with_suffix(".json")
            if candidate_metadata.exists():
                cache_path = candidate
                metadata_path = candidate_metadata
                break
    row_by_sample = {str(row.get("sample_id", "")): row for row in rows}
    if cache_path.exists() and metadata_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        sample_ids = [str(value) for value in data["sample_ids"].tolist()]
        ok_rows = [row_by_sample[sample_id] for sample_id in sample_ids if sample_id in row_by_sample]
        if len(ok_rows) == len(sample_ids):
            print(f"[cache] loaded DOFA embeddings split={split_name} path={cache_path}")
            return (
                np.asarray(data["embeddings"], dtype=np.float32),
                [str(value) for value in data["labels"].tolist()],
                ok_rows,
                [],
                cache_path,
                json.loads(metadata_path.read_text(encoding="utf-8")),
            )
    embeddings, labels, ok_rows, warnings = _dofa_embeddings(rows, config, adapter)
    sample_ids = np.asarray([str(row.get("sample_id", "")) for row in ok_rows])
    np.savez_compressed(
        cache_path,
        embeddings=embeddings.astype(np.float32),
        labels=np.asarray(labels),
        sample_ids=sample_ids,
    )
    metadata = {
        "cache_key": key,
        "split": split_name,
        "manifest": str(config.metadata_csv),
        "model_variant": adapter.model_variant,
        "model_release": adapter.model_release,
        "image_size": config.image_size,
        "adapter_image_size": adapter.image_size,
        "band_profile": config.band_profile,
        "checkpoint_source": _adapter_source(adapter),
        "checkpoint_sha256": adapter.actual_checkpoint_sha256 or adapter.checkpoint_sha256 or "",
        "repo_revision": adapter.actual_repo_revision or adapter.repo_revision or "",
        "architecture_source_revision": (
            adapter.architecture_source_revision
            if adapter.model_variant == "dofav2_vit_base"
            else ""
        ),
        "required_timm_version": (
            adapter.required_timm_version
            if adapter.model_variant == "dofav2_vit_base"
            else ""
        ),
        "patch_size": (
            adapter.official_dofav2_patch_size
            if adapter.model_variant == "dofav2_vit_base"
            else 16
        ),
        "embedding_layer": adapter.embedding_layer,
        "embedding_pooling": adapter.embedding_pooling,
        "embedding_semantics": (
            adapter.official_dofav2_embedding_semantics
            if adapter.model_variant == "dofav2_vit_base"
            else adapter.embedding_pooling
        ),
        "input_scale": adapter.input_scale,
        "row_count_requested": len(rows),
        "row_count_cached": len(ok_rows),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else "",
        "row_hash": _row_hash(rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[cache] wrote DOFA embeddings split={split_name} path={cache_path}")
    return embeddings, labels, ok_rows, warnings, cache_path, metadata


def _nearest_centroid_with_confidence(classifier: _NearestCentroidClassifier, features: np.ndarray) -> tuple[list[str], list[float]]:
    if classifier.centroids is None:
        raise RuntimeError("classifier has not been fitted")
    diff = features[:, None, :] - classifier.centroids[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    scores = -distances
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    indices = np.argmax(probs, axis=1)
    return [classifier.classes[int(index)] for index in indices], [float(probs[row, index]) for row, index in enumerate(indices)]


def _train_linear_probe(
    train_x: np.ndarray,
    train_y: Sequence[str],
    eval_x: np.ndarray,
    config: FmowClassificationConfig,
    output_dir: Path,
) -> tuple[list[str], list[float], np.ndarray, list[str], dict[str, Any], dict[str, Any]]:
    torch, nn, _F = _require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    classes = sorted({str(label) for label in train_y})
    class_to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([class_to_index[str(label)] for label in train_y], dtype=np.int64)
    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0).astype(np.float32), 1e-6)
    train_z = (train_x.astype(np.float32) - mean) / std
    eval_z = (eval_x.astype(np.float32) - mean) / std
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _device_from_config(config)
    model = nn.Linear(train_z.shape[1], len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.probe_learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.from_numpy(train_z), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    history: list[dict[str, Any]] = []
    print(
        f"[probe] training linear probe embeddings={train_z.shape} classes={len(classes)} "
        f"epochs={config.probe_epochs} device={device}"
    )
    for epoch in range(1, config.probe_epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch_x.shape[0])
            total_seen += int(batch_x.shape[0])
        if epoch == 1 or epoch == config.probe_epochs or epoch % 25 == 0:
            record = {"epoch": epoch, "train_loss": total_loss / max(total_seen, 1)}
            history.append(record)
            print(f"[probe] epoch={epoch}/{config.probe_epochs} loss={record['train_loss']:.6f}")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(eval_z).to(device))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    pred_idx = np.argmax(probs, axis=1)
    predictions = [classes[int(index)] for index in pred_idx]
    confidence = [float(probs[row, index]) for row, index in enumerate(pred_idx)]
    checkpoint_path = output_dir / "dofa_linear_probe_checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "class_to_index": class_to_index,
            "embedding_mean": mean,
            "embedding_std": std,
            "probe_epochs": config.probe_epochs,
            "probe_learning_rate": config.probe_learning_rate,
        },
        checkpoint_path,
    )
    metadata = {
        "probe": "linear",
        "classifier": "torch.nn.Linear",
        "probe_epochs": config.probe_epochs,
        "probe_learning_rate": config.probe_learning_rate,
        "checkpoint_path": str(checkpoint_path),
        "class_to_index": class_to_index,
    }
    debug = {
        "probe_history": history,
        "embedding_dim": int(train_z.shape[1]),
        "linear_probe_checkpoint": str(checkpoint_path),
    }
    return predictions, confidence, probs.astype(np.float32), classes, metadata, debug


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("--model resnet50 requires torch and torchvision in the runtime environment.") from exc
    return torch, nn, F


def build_resnet50_multiband(
    num_classes: int,
    *,
    in_channels: int,
    pretrained: bool = False,
) -> Any:
    """Build a ResNet-50 with a deterministic multispectral stem.

    ``pretrained=False`` is the protocol-matched from-scratch baseline.  When
    ImageNet initialization is requested, RGB conv1 weights are expanded by
    their channel mean and rescaled by ``3 / in_channels`` so activation scale
    does not grow merely because more bands are present.
    """

    torch, nn, _F = _require_torch()
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as exc:
        raise RuntimeError("--model resnet50 requires torchvision in the runtime environment.") from exc
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    original = model.conv1
    replacement = nn.Conv2d(
        int(in_channels),
        original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            mean_weight = original.weight.mean(dim=1, keepdim=True)
            replacement.weight.copy_(
                mean_weight.repeat(1, int(in_channels), 1, 1) * (3.0 / float(in_channels))
            )
    model.conv1 = replacement
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50_13band(num_classes: int) -> Any:
    """Backward-compatible constructor for the completed BWER 1.x baseline."""

    return build_resnet50_multiband(num_classes, in_channels=13, pretrained=False)


def _device_from_config(config: FmowClassificationConfig) -> Any:
    torch, _nn, _F = _require_torch()
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def _class_mapping(train_rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, int]]:
    classes = sorted({str(row.get("category", "")) for row in train_rows if not _is_missing(row.get("category"))})
    if not classes:
        raise ValueError("No training classes found for ResNet-50 training.")
    return classes, {label: index for index, label in enumerate(classes)}


def _compute_or_load_norm_stats(
    train_rows: Sequence[dict[str, Any]],
    config: FmowClassificationConfig,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], Path]:
    if config.norm_stats is not None and config.norm_stats.exists():
        stats = json.loads(config.norm_stats.read_text(encoding="utf-8"))
        return stats, list(train_rows), [], config.norm_stats

    profile = get_band_profile(config.band_profile)
    expected_bands = int(profile["expected_bands"])
    sums = np.zeros(expected_bands, dtype=np.float64)
    squares = np.zeros(expected_bands, dtype=np.float64)
    pixel_count = 0
    ok_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    print(f"[norm] computing train-only per-band stats over {len(train_rows)} rows")
    for index, row in enumerate(train_rows, start=1):
        try:
            chip = load_fmow_sentinel_image(row, config.data_root, config.image_size, config.band_profile)
        except Exception as exc:
            warnings.append(f"Skipping train sample_id={row.get('sample_id')} for norm stats: {exc}")
            continue
        flat = chip.reshape(chip.shape[0], -1).astype(np.float64)
        sums += flat.sum(axis=1)
        squares += np.square(flat).sum(axis=1)
        pixel_count += flat.shape[1]
        ok_rows.append(row)
        if index % 1000 == 0:
            print(f"[norm] processed={index}/{len(train_rows)} readable={len(ok_rows)}")
    if pixel_count <= 0 or not ok_rows:
        raise ValueError("No readable training rasters found for ResNet-50 normalization.")
    mean = sums / pixel_count
    variance = np.maximum((squares / pixel_count) - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    stats = {
        "band_profile": config.band_profile,
        "image_size": config.image_size,
        "source": "train_split_only",
        "train_split": config.train_split,
        "train_rows_requested": len(train_rows),
        "train_rows_readable": len(ok_rows),
        "pixel_count_per_band": int(pixel_count),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }
    path = output_dir / "norm_stats.json"
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats, ok_rows, warnings, path


class _FmowTorchDataset:
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        config: FmowClassificationConfig,
        class_to_index: Mapping[str, int],
        norm_stats: Mapping[str, Any],
    ) -> None:
        torch, _nn, _F = _require_torch()
        self.torch = torch
        self.rows = list(rows)
        self.config = config
        self.class_to_index = dict(class_to_index)
        self.mean = np.asarray(norm_stats["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(norm_stats["std"], dtype=np.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, int]:
        row = self.rows[index]
        chip = load_fmow_sentinel_image(row, self.config.data_root, self.config.image_size, self.config.band_profile)
        chip = (chip - self.mean) / np.maximum(self.std, 1e-6)
        label = self.class_to_index.get(str(row.get("category", "")), -1)
        return self.torch.from_numpy(chip.astype(np.float32, copy=False)), self.torch.tensor(label, dtype=self.torch.long), index


def _macro_f1_from_counts(tp: Mapping[str, int], fp: Mapping[str, int], fn: Mapping[str, int], labels: Sequence[str]) -> float:
    values: list[float] = []
    for label in labels:
        precision = tp.get(label, 0) / max(tp.get(label, 0) + fp.get(label, 0), 1)
        recall = tp.get(label, 0) / max(tp.get(label, 0) + fn.get(label, 0), 1)
        values.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(np.mean(values)) if values else float("nan")


def _evaluate_resnet50(model: Any, loader: Any, rows: Sequence[dict[str, Any]], classes: Sequence[str], device: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch, _nn, F = _require_torch()
    model.eval()
    predictions: list[dict[str, Any]] = []
    top5_correct = 0
    true_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    with torch.no_grad():
        for images, _targets, indices in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probabilities, dim=1)
            k = min(5, len(classes))
            topk = torch.topk(probabilities, k=k, dim=1).indices.cpu().numpy()
            pred_idx_np = pred_idx.cpu().numpy()
            confidence_np = confidence.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            for batch_pos, row_index in enumerate(indices.cpu().numpy().tolist()):
                row = rows[int(row_index)]
                label = str(row.get("category", ""))
                prediction = classes[int(pred_idx_np[batch_pos])]
                top_labels = [classes[int(idx)] for idx in topk[batch_pos]]
                is_correct = prediction == label
                is_top5 = label in top_labels
                top5_correct += int(is_top5)
                true_counts[label] += 1
                if is_correct:
                    correct_counts[label] += 1
                    tp[label] += 1
                else:
                    fp[prediction] += 1
                    fn[label] += 1
                predictions.append(
                    {
                        "row": row,
                        "prediction": prediction,
                        "confidence": float(confidence_np[batch_pos]),
                        "top5_correct": float(is_top5),
                        "probability_vector": probabilities_np[batch_pos].astype(np.float32),
                        "logit_vector": logits_np[batch_pos].astype(np.float32),
                    }
                )
    total = len(predictions)
    accuracy = sum(correct_counts.values()) / total if total else float("nan")
    recalls = [correct_counts[label] / count for label, count in true_counts.items() if count > 0]
    balanced_accuracy = float(np.mean(recalls)) if recalls else float("nan")
    macro_f1 = _macro_f1_from_counts(tp, fp, fn, sorted(set(true_counts) | set(fp) | set(tp)))
    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "top5_accuracy": float(top5_correct / total) if total else float("nan"),
        "eval_rows": total,
    }
    return predictions, metrics


def _train_resnet50(
    train_rows: Sequence[dict[str, Any]],
    eval_rows: Sequence[dict[str, Any]],
    config: FmowClassificationConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], dict[str, Any]]:
    torch, nn, _F = _require_torch()
    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    classes, class_to_index = _class_mapping(train_rows)
    norm_stats, train_ok, norm_warnings, norm_path = _compute_or_load_norm_stats(train_rows, config, output_dir)
    device = _device_from_config(config)
    train_dataset = _FmowTorchDataset(train_ok, config, class_to_index, norm_stats)
    eval_dataset = _FmowTorchDataset(eval_rows, config, class_to_index, norm_stats)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    expected_bands = int(get_band_profile(config.band_profile)["expected_bands"])
    model = build_resnet50_multiband(
        len(classes), in_channels=expected_bands, pretrained=False
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_score = -float("inf")
    best_epoch = 0
    checkpoint_path = output_dir / "best_resnet50_checkpoint.pt"
    history: list[dict[str, Any]] = []
    print(
        f"[train] model=resnet50_13band_from_scratch train={len(train_ok)} "
        f"eval={len(eval_rows)} classes={len(classes)} device={device}"
    )
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, targets, _indices in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if torch.any(targets < 0):
                raise ValueError("Training rows contain labels absent from the train class mapping.")
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch = int(images.shape[0])
            total_loss += float(loss.detach().cpu()) * batch
            total_seen += batch
        eval_predictions, eval_metrics = _evaluate_resnet50(model, eval_loader, eval_rows, classes, device)
        score = eval_metrics.get(config.checkpoint_metric)
        if not isinstance(score, float) or math.isnan(score):
            score = eval_metrics.get("accuracy", -float("inf"))
        train_loss = total_loss / max(total_seen, 1)
        record = {"epoch": epoch, "train_loss": train_loss, **eval_metrics}
        history.append(record)
        print(
            f"[train] epoch={epoch}/{config.epochs} loss={train_loss:.6f} "
            f"val_acc={eval_metrics['accuracy']:.6f} val_macro_f1={eval_metrics['macro_f1']:.6f}"
        )
        if float(score) > best_score:
            best_score = float(score)
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_index": class_to_index,
                    "classes": classes,
                    "norm_stats": norm_stats,
                    "epoch": epoch,
                    "score": best_score,
                    "config": {
                        "image_size": config.image_size,
                        "learning_rate": config.learning_rate,
                        "weight_decay": config.weight_decay,
                        "batch_size": config.batch_size,
                    },
                },
                checkpoint_path,
            )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    final_predictions, final_metrics = _evaluate_resnet50(model, eval_loader, eval_rows, classes, device)
    final_metrics.update(
        {
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "best_validation_metric": config.checkpoint_metric,
            "checkpoint_path": str(checkpoint_path),
            "norm_stats_path": str(norm_path),
            "train_rows_readable": len(train_ok),
        }
    )
    debug = {
        "model": "resnet50_fmow_sentinel",
        "model_family": "resnet",
        "model_variant": f"resnet50_{expected_bands}band_from_scratch",
        "first_conv_in_channels": expected_bands,
        "weights": "none",
        "normalization": "train_split_per_band_mean_std",
        "classes": classes,
        "history": history,
        "device": str(device),
    }
    return final_predictions, final_metrics, norm_warnings, debug


def _model_metadata(config: FmowClassificationConfig) -> dict[str, str]:
    if config.model == "resnet50":
        expected_bands = int(get_band_profile(config.band_profile)["expected_bands"])
        return {
            "model": "resnet50_fmow_sentinel",
            "model_family": "resnet",
            "model_variant": f"resnet50_{expected_bands}band_from_scratch",
            "adaptation_protocol": "supervised_baseline",
            "training_budget": f"adamw_cross_entropy_epochs_{config.epochs}",
        }
    if config.model in {"dofa", "dofav2"}:
        release = "dofav2" if config.model == "dofav2" else "dofa"
        if config.probe == "linear":
            return {
                "model": f"{release}_fmow_sentinel",
                "model_family": "dofa",
                "model_variant": f"{release}_vit_base",
                "adaptation_protocol": "frozen_encoder_linear_probe",
                "training_budget": f"frozen_{release}_vitb_embeddings_linear_probe_epochs_{config.probe_epochs}",
            }
        return {
            "model": f"{release}_fmow_sentinel",
            "model_family": "dofa",
            "model_variant": f"{release}_vit_base_nearest_centroid_sanity",
            "adaptation_protocol": "frozen_probe",
            "training_budget": "frozen_encoder_nearest_centroid_probe",
        }
    return {
        "model": "supervised_stats_fmow_sentinel",
        "model_family": "supervised_stats",
        "model_variant": "band_stats_nearest_centroid",
        "adaptation_protocol": "supervised_baseline",
        "training_budget": "nearest_centroid_on_13band_image_statistics",
    }


def _prediction_rows(
    eval_rows: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    config: FmowClassificationConfig,
    confidences: Sequence[float | str] | None = None,
    top5_correct: Sequence[float | str] | None = None,
) -> list[dict[str, Any]]:
    meta = _model_metadata(config)
    input_mode = f"s2_{int(get_band_profile(config.band_profile)['expected_bands'])}band_image_only"
    rows: list[dict[str, Any]] = []
    confidence_values = list(confidences) if confidences is not None else [""] * len(predictions)
    top5_values = list(top5_correct) if top5_correct is not None else [""] * len(predictions)
    for row, prediction, confidence, top5 in zip(eval_rows, predictions, confidence_values, top5_values):
        label = str(row.get("category", ""))
        correct = float(prediction == label)
        out: dict[str, Any] = {
            "sample_id": row.get("sample_id", ""),
            "image_id": row.get("image_id", ""),
            "image_path": row.get("image_path", ""),
            "extracted_path": row.get("extracted_path", row.get("extracted_image_path", "")),
            "dataset": "fmow_sentinel",
            "task": "scene_classification",
            "split": row.get("split", config.eval_split),
            "label": label,
            "category": label,
            "class_label": label,
            "prediction": prediction,
            "predicted_category": prediction,
            "correct": correct,
            "score": correct,
            "risk": 1.0 - correct,
            "confidence": confidence,
            "max_probability": confidence,
            "top5_correct": top5,
            "model": meta["model"],
            "model_family": meta["model_family"],
            "model_variant": meta["model_variant"],
            "input_mode": input_mode,
            "adaptation_protocol": meta["adaptation_protocol"],
            "training_budget": meta["training_budget"],
            "split_protocol": config.split_protocol,
            "eval_scope": config.eval_scope or config.eval_split,
            "resolution": str(config.image_size),
            "band_profile": config.band_profile,
            "metadata_provenance": row.get("metadata_provenance", "location_level_geography_enrichment"),
        }
        for key in GEOGRAPHY_COLUMNS:
            out[key] = row.get(key, "")
        rows.append(out)
    return rows


def _write_report(output_dir: Path, predictions: Sequence[Mapping[str, Any]], warnings: Sequence[str], config: FmowClassificationConfig) -> Path:
    accuracy = np.mean([float(row["correct"]) for row in predictions]) if predictions else float("nan")
    counts = Counter(str(row.get("country", "")) for row in predictions if not _is_missing(row.get("country")))
    path = output_dir / "report.md"
    path.write_text(
        "\n".join(
            [
                "# fMoW-Sentinel Image-Only Classification Prototype",
                "",
                f"- model: `{_model_metadata(config)['model']}`",
                f"- model_variant: `{_model_metadata(config)['model_variant']}`",
                f"- adaptation_protocol: `{_model_metadata(config)['adaptation_protocol']}`",
                f"- split_protocol: `{config.split_protocol}`",
                f"- train_split: `{config.train_split}`",
                f"- eval_split: `{config.eval_split}`",
                f"- image_size: `{config.image_size}`",
                f"- band_profile: `{config.band_profile}`",
                f"- prediction rows: {len(predictions)}",
                f"- aggregate accuracy: {accuracy:.6f}" if not math.isnan(accuracy) else "- aggregate accuracy: n/a",
                f"- geography slices with non-missing country: {len(counts)}",
                "",
                "Geography metadata is used only for audit slicing, support diagnostics, and BWER reporting. "
                "Country, region, coordinates, timestamps, and location IDs are not model inputs.",
                "",
                "The current fMoW geography enrichment is location-level, not an image-level exact join; reports must preserve that provenance.",
                "",
                "## Warnings",
                *(f"- {warning}" for warning in warnings[:50]),
            ]
        ),
        encoding="utf-8",
    )
    return path


def _metrics_from_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = [float(row.get("correct", 0.0)) for row in rows]
    labels = [str(row.get("label", row.get("category", ""))) for row in rows]
    predictions = [str(row.get("prediction", "")) for row in rows]
    true_counts: Counter[str] = Counter(labels)
    correct_counts: Counter[str] = Counter(label for label, prediction in zip(labels, predictions) if label == prediction)
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            tp[label] += 1
        else:
            fp[prediction] += 1
            fn[label] += 1
    recalls = [correct_counts[label] / count for label, count in true_counts.items() if count > 0]
    top5_values = [float(row["top5_correct"]) for row in rows if not _is_missing(row.get("top5_correct"))]
    return {
        "accuracy": float(np.mean(correct)) if correct else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else "",
        "macro_f1": _macro_f1_from_counts(tp, fp, fn, sorted(set(true_counts) | set(fp) | set(tp))),
        "top5_accuracy": float(np.mean(top5_values)) if top5_values else "",
    }


def run_fmow_sentinel_classification(config: FmowClassificationConfig) -> dict[str, Path]:
    output = ensure_dir(config.output_dir)
    rows = _load_metadata(config.metadata_csv)
    split_limit = config.max_samples_per_split if config.max_samples_per_split is not None else config.max_samples
    train_rows = _limit_rows(_split_rows(rows, config.train_split), split_limit, config.seed)
    eval_rows = _limit_rows(_split_rows(rows, config.eval_split), split_limit, config.seed + 1)
    if not train_rows:
        raise ValueError(f"No training rows found for split={config.train_split!r}.")
    if not eval_rows:
        raise ValueError(f"No evaluation rows found for split={config.eval_split!r}.")

    resnet_metrics: dict[str, Any] = {}
    resnet_debug: dict[str, Any] = {}
    resnet_warnings: list[str] = []
    dofa_run_metadata: dict[str, Any] = {}
    dofa_debug: dict[str, Any] = {}
    formal_probabilities: np.ndarray | None = None
    formal_classes: list[str] | None = None
    if config.model == "resnet50":
        resnet_eval_predictions, resnet_metrics, resnet_warnings, resnet_debug = _train_resnet50(train_rows, eval_rows, config, output)
        eval_ok = [dict(item["row"]) for item in resnet_eval_predictions]
        predictions = [str(item["prediction"]) for item in resnet_eval_predictions]
        prediction_rows = _prediction_rows(
            eval_ok,
            predictions,
            config,
            confidences=[item["confidence"] for item in resnet_eval_predictions],
            top5_correct=[item["top5_correct"] for item in resnet_eval_predictions],
        )
        train_ok = train_rows
        warnings_train = resnet_warnings
        warnings_eval: list[str] = []
        formal_probabilities = np.stack([item["probability_vector"] for item in resnet_eval_predictions]).astype(np.float32)
        formal_classes = list(resnet_debug["classes"])
    elif config.model in {"dofa", "dofav2"}:
        if config.probe not in {"linear", "nearest_centroid"}:
            raise ValueError("--probe must be 'linear' or 'nearest_centroid' for --model dofa.")
        if config.model_config is None:
            raise ValueError("--model-config is required for --model dofa.")
        adapter = DOFAAdapter.from_config_file(config.model_config)
        if config.model == "dofav2" and not str(getattr(adapter, "model_release", "")).startswith("dofav2"):
            raise ValueError("--model dofav2 requires a model config with model_release starting with 'dofav2'.")
        if config.allow_torch_hub_download:
            adapter.allow_torch_hub_download = True
        if config.dofa_input_scale is not None:
            adapter.input_scale = float(config.dofa_input_scale)
        if config.dofa_embedding_pooling is not None:
            adapter.embedding_pooling = str(config.dofa_embedding_pooling)
        adapter.load_model()
        train_x, train_y, train_ok, warnings_train, train_cache, train_cache_meta = _cached_dofa_embeddings(
            train_rows, config.train_split, config, adapter, output
        )
        eval_x, _eval_y, eval_ok, warnings_eval, eval_cache, eval_cache_meta = _cached_dofa_embeddings(
            eval_rows, config.eval_split, config, adapter, output
        )
        if config.probe == "linear":
            predictions, confidences, formal_probabilities, formal_classes, probe_metadata, dofa_debug = _train_linear_probe(
                train_x, train_y, eval_x, config, output
            )
            prediction_rows = _prediction_rows(eval_ok, predictions, config, confidences=confidences)
        else:
            classifier = _NearestCentroidClassifier()
            classifier.fit(train_x, train_y)
            predictions, confidences = _nearest_centroid_with_confidence(classifier, eval_x)
            prediction_rows = _prediction_rows(eval_ok, predictions, config, confidences=confidences)
            probe_metadata = {"probe": "nearest_centroid", "class_to_index": {label: index for index, label in enumerate(classifier.classes)}}
            dofa_debug = {"classifier": "nearest_centroid", "classes": classifier.classes}
        profile = get_band_profile(config.band_profile)
        dofa_run_metadata = {
            "probe": config.probe,
            "probe_epochs": config.probe_epochs if config.probe == "linear" else "",
            "probe_learning_rate": config.probe_learning_rate if config.probe == "linear" else "",
            "band_order": profile.get("band_names", []),
            "wavelength_list": profile.get("wavelength_list", adapter.wavelengths or []),
            "normalization_mean": adapter.normalization_mean,
            "normalization_std": adapter.normalization_std,
            "input_scale": adapter.input_scale,
            "checkpoint_source": _adapter_source(adapter),
            "dofa_model_variant": adapter.model_variant,
            "dofa_model_release": getattr(adapter, "model_release", ""),
            "dofa_checkpoint_sha256": getattr(adapter, "actual_checkpoint_sha256", "") or getattr(adapter, "checkpoint_sha256", ""),
            "dofa_checkpoint_load_report": getattr(adapter, "checkpoint_load_report", {}),
            "embedding_cache_path": str(output / "embedding_cache"),
            "train_embedding_cache_path": str(train_cache),
            "eval_embedding_cache_path": str(eval_cache),
            "train_embedding_cache_key": train_cache_meta.get("cache_key", ""),
            "eval_embedding_cache_key": eval_cache_meta.get("cache_key", ""),
            "embedding_pooling": adapter.embedding_pooling,
            "embedding_dim": train_cache_meta.get("embedding_dim", ""),
            "class_mapping": probe_metadata.get("class_to_index", {}),
            **{key: value for key, value in probe_metadata.items() if key not in {"class_to_index"}},
        }
    elif config.model == "supervised_stats":
        train_x, train_y, train_ok, warnings_train = _rows_to_features(train_rows, config)
        eval_x, _eval_y, eval_ok, warnings_eval = _rows_to_features(eval_rows, config)
    else:
        raise ValueError(f"Unsupported fMoW-Sentinel classification model: {config.model}")

    if config.model == "supervised_stats":
        classifier = _NearestCentroidClassifier()
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(eval_x)
        prediction_rows = _prediction_rows(eval_ok, predictions, config)
    audit_rows = build_audit_table_from_predictions_from_rows(prediction_rows)

    artifacts = {
        "predictions": output / "predictions.csv",
        "audit_table": output / "audit_table.csv",
        "metrics_summary": output / "metrics_summary.csv",
        "run_metadata": output / "run_metadata.json",
        "model_debug": output / "model_debug.json",
        "report": output / "report.md",
    }
    write_csv(artifacts["predictions"], prediction_rows)
    write_audit_table(artifacts["audit_table"], audit_rows)
    warnings = warnings_train + warnings_eval
    row_metrics = _metrics_from_prediction_rows(prediction_rows)
    accuracy = row_metrics["accuracy"]
    metric_row = {
        "dataset": "fmow_sentinel",
        "task": "scene_classification",
        "model": _model_metadata(config)["model"],
        "model_family": _model_metadata(config)["model_family"],
        "model_variant": _model_metadata(config)["model_variant"],
        "split": config.eval_split,
        "eval_scope": config.eval_scope or config.eval_split,
        "accuracy": accuracy,
        "balanced_accuracy": resnet_metrics.get("balanced_accuracy", row_metrics.get("balanced_accuracy", "")),
        "macro_f1": resnet_metrics.get("macro_f1", row_metrics.get("macro_f1", "")),
        "top5_accuracy": resnet_metrics.get("top5_accuracy", row_metrics.get("top5_accuracy", "")),
        "best_epoch": resnet_metrics.get("best_epoch", ""),
        "best_validation_metric": resnet_metrics.get("best_validation_metric", ""),
        "best_validation_score": resnet_metrics.get("best_validation_score", ""),
        "total_epochs": config.epochs if config.model == "resnet50" else "",
    }
    write_csv(artifacts["metrics_summary"], [metric_row])
    metadata = {
        "dataset": "fmow_sentinel",
        "task": "scene_classification",
        "model": _model_metadata(config)["model"],
        "model_family": _model_metadata(config)["model_family"],
        "model_variant": _model_metadata(config)["model_variant"],
        "adaptation_protocol": _model_metadata(config)["adaptation_protocol"],
        "split_protocol": config.split_protocol,
        "train_split": config.train_split,
        "eval_split": config.eval_split,
        "eval_scope": config.eval_scope or config.eval_split,
        "image_size": config.image_size,
        "band_profile": config.band_profile,
        "input_mode": f"s2_{int(get_band_profile(config.band_profile)['expected_bands'])}band_image_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "train_rows_requested": len(train_rows),
        "train_rows_readable": resnet_metrics.get("train_rows_readable", len(train_ok)),
        "eval_rows_requested": len(eval_rows),
        "eval_rows_readable": len(eval_ok),
        "aggregate_accuracy": accuracy,
        "balanced_accuracy": metric_row["balanced_accuracy"],
        "macro_f1": metric_row["macro_f1"],
        "top5_accuracy": metric_row["top5_accuracy"],
        "epochs": config.epochs if config.model == "resnet50" else config.probe_epochs if config.model in {"dofa", "dofav2"} and config.probe == "linear" else "",
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate if config.model == "resnet50" else config.probe_learning_rate if config.model in {"dofa", "dofav2"} and config.probe == "linear" else "",
        "weight_decay": config.weight_decay if config.model == "resnet50" else "",
        "checkpoint_metric": config.checkpoint_metric if config.model == "resnet50" else "",
        "optimizer": "AdamW" if config.model in {"resnet50", "dofa", "dofav2"} and (config.model == "resnet50" or config.probe == "linear") else "nearest_centroid",
        "loss": "cross_entropy" if config.model in {"resnet50", "dofa", "dofav2"} and (config.model == "resnet50" or config.probe == "linear") else "",
        "random_seed": config.seed,
        "geography_metadata_usage": "audit_slicing_only_not_model_input",
        "geography_join_level": "location_level",
    }
    metadata.update({key: value for key, value in resnet_metrics.items() if key in {"best_epoch", "best_validation_score", "best_validation_metric", "checkpoint_path", "norm_stats_path"}})
    metadata.update(dofa_run_metadata)
    artifacts["run_metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if config.model == "resnet50":
        debug_payload = {**resnet_debug, "warnings": warnings[:100]}
    elif config.model in {"dofa", "dofav2"}:
        debug_payload = {
            "model": metadata["model"],
            "model_family": metadata["model_family"],
            "model_variant": metadata["model_variant"],
            "adaptation_protocol": metadata["adaptation_protocol"],
            "probe": config.probe,
            "frozen_backbone": True,
            "embedding_cache_path": metadata.get("embedding_cache_path", ""),
            "band_order": metadata.get("band_order", []),
            "wavelength_list": metadata.get("wavelength_list", []),
            "input_scale": metadata.get("input_scale", ""),
            "embedding_pooling": metadata.get("embedding_pooling", ""),
            **dofa_debug,
            "warnings": warnings[:100],
        }
    else:
        debug_payload = {
            "model": metadata["model"],
            "classifier": "nearest_centroid",
            "feature_dimension": int(train_x.shape[1]),
            "classes": classifier.classes,
            "warnings": warnings[:100],
        }
    artifacts["model_debug"].write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    if config.write_formal_outputs:
        if formal_probabilities is None or formal_classes is None:
            raise ValueError("Formal fMoW outputs require a probabilistic linear/softmax head; nearest-centroid/statistics modes are diagnostic only.")
        protocol = BWERProtocol.from_mapping(load_yaml(config.geobwer_protocol))
        checkpoint = metadata.get("checkpoint_path") or dofa_debug.get("linear_probe_checkpoint", "")
        checkpoint_hash = file_sha256(checkpoint) if checkpoint and Path(str(checkpoint)).exists() else ""
        formal_bundle = write_multiclass_bundle(
            output / "formal_outputs",
            sample_rows=eval_ok,
            probabilities=formal_probabilities,
            targets=[str(row.get("category", "")) for row in eval_ok],
            class_names=formal_classes,
            dataset="fmow_sentinel",
            model=str(metadata["model"]),
            split=config.eval_split,
            protocol=protocol,
            model_lineage={
                "model": metadata["model"],
                "model_variant": metadata["model_variant"],
                "adaptation_protocol": metadata["adaptation_protocol"],
                "checkpoint_source": metadata.get("checkpoint_source", checkpoint),
                "checkpoint_sha256": metadata.get("dofa_checkpoint_sha256", checkpoint_hash),
                "probe_checkpoint_sha256": checkpoint_hash,
                "band_profile": config.band_profile,
                "image_size": config.image_size,
                "input_scale": metadata.get("input_scale", ""),
                "embedding_pooling": metadata.get("embedding_pooling", ""),
            },
            dataset_lineage={
                "metadata_csv": str(config.metadata_csv),
                "metadata_sha256": file_sha256(config.metadata_csv),
                "split_protocol": config.split_protocol,
                "train_split": config.train_split,
                "eval_split": config.eval_split,
                "eval_row_hash": _row_hash(eval_ok),
            },
            independent_unit_column="sample_id",
        )
        artifacts.update(
            {
                "formal_audit_table": formal_bundle.audit_table,
                "formal_probabilities": formal_bundle.probability_artifact,
                "formal_class_mapping": formal_bundle.class_mapping,
                "formal_output_manifest": formal_bundle.manifest,
            }
        )
    _write_report(output, prediction_rows, warnings, config)
    if config.run_bwer:
        artifacts.update(run_fmow_geography_bwer(FmowBwerConfig(input_dir=output, output_dir=output / "bwer", bootstrap=config.bwer_bootstrap, seed=config.seed)))
    return artifacts


def build_audit_table_from_predictions_from_rows(prediction_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    temp_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        out = dict(row)
        out["unit_id"] = out.get("sample_id", out.get("image_id", ""))
        out["y_true"] = out.get("label", "")
        out["y_pred"] = out.get("prediction", "")
        out["aggregation_level"] = "sample"
        temp_rows.append(out)
    validate_audit_table(temp_rows)
    return temp_rows


def run_fmow_geography_bwer(config: FmowBwerConfig) -> dict[str, Path]:
    audit_table = config.audit_table or config.input_dir / "audit_table.csv"
    if not audit_table.exists():
        predictions = config.input_dir / "predictions.csv"
        if not predictions.exists():
            raise FileNotFoundError(f"Expected audit_table.csv or predictions.csv in {config.input_dir}")
        rows = build_audit_table_from_predictions(predictions, dataset="fmow_sentinel", model="", task="scene_classification")
        audit_table = ensure_dir(config.output_dir) / "audit_table.csv"
        write_audit_table(audit_table, rows)
    artifacts = evaluate_bwer_from_file(
        audit_table,
        dataset="fmow_sentinel",
        model=_infer_model_from_audit_table(audit_table),
        task="scene_classification",
        output_dir=config.output_dir,
        bootstrap=config.bootstrap,
        seed=config.seed,
        score_column="correct",
        risk_column="risk",
        audit_level="pilot",
    )
    return {f"bwer_{key}": value for key, value in artifacts.items()}


def _infer_model_from_audit_table(path: Path) -> str:
    rows = read_csv_rows(path)
    return str(rows[0].get("model", "fmow_sentinel_model")) if rows else "fmow_sentinel_model"


def _read_bwer_summary(run_dir: Path) -> list[dict[str, str]]:
    for path in (run_dir / "bwer" / "bwer_summary.csv", run_dir / "bwer_summary.csv"):
        if path.exists():
            return read_csv_rows(path)
    return []


def _bwer_value(rows: Sequence[Mapping[str, Any]], slice_variable: str, balance_variable: str = "") -> dict[str, Any]:
    for row in rows:
        if str(row.get("slice_variable", "")) == slice_variable and str(row.get("balance_variable", "")) == balance_variable:
            return dict(row)
    return {}


def compare_fmow_runs(runs: Mapping[str, Path], output_dir: Path) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    summary_rows: list[dict[str, Any]] = []
    average_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    for run_name, run_dir in runs.items():
        audit_path = run_dir / "audit_table.csv"
        if not audit_path.exists():
            raise FileNotFoundError(f"Missing audit_table.csv for run {run_name}: {run_dir}")
        audit_rows = read_csv_rows(audit_path)
        bwer_rows = _read_bwer_summary(run_dir)
        correct = [float(row.get("correct", row.get("score", 0.0))) for row in audit_rows if not _is_missing(row.get("correct", row.get("score", "")))]
        accuracy = float(np.mean(correct)) if correct else float("nan")
        first = audit_rows[0] if audit_rows else {}
        raw_country = _bwer_value(bwer_rows, "country")
        std_country = _bwer_value(bwer_rows, "country", "class_label") or _bwer_value(bwer_rows, "country", "category")
        summary = {
            "run_name": run_name,
            "model": first.get("model", ""),
            "model_family": first.get("model_family", ""),
            "model_variant": first.get("model_variant", ""),
            "adaptation_protocol": first.get("adaptation_protocol", ""),
            "split_protocol": first.get("split_protocol", ""),
            "eval_scope": first.get("eval_scope", first.get("split", "")),
            "dataset": first.get("dataset", "fmow_sentinel"),
            "task": first.get("task", "scene_classification"),
            "resolution": first.get("resolution", ""),
            "aggregate_accuracy": accuracy,
            "aggregate_error": 1.0 - accuracy if not math.isnan(accuracy) else "",
            "raw_bwer_country": raw_country.get("bwer", ""),
            "standardised_bwer_country_class": std_country.get("bwer", ""),
            "worst_country_slice": raw_country.get("worst_slice", ""),
            "tail_country_slices": raw_country.get("tail_slices", ""),
            "notes": "Protocol-aware image-only comparison; geography metadata is audit-only and not model input.",
        }
        summary_rows.append(summary)
        average_rows.append(
            {
                "run_name": run_name,
                "average_score": accuracy,
                "raw_bwer": raw_country.get("bwer", ""),
                "standardised_bwer": std_country.get("bwer", ""),
                "model_label": summary["model_variant"] or summary["model"],
                "protocol_label": summary["adaptation_protocol"],
                "split_label": summary["split_protocol"],
            }
        )
        by_slice_path = run_dir / "bwer" / "bwer_by_slice.csv"
        if by_slice_path.exists():
            for row in read_csv_rows(by_slice_path):
                if str(row.get("slice_variable", "")) in {"country", "continent", "un_region", "region", "latitude_band", "season", "category"}:
                    item = dict(row)
                    item["run_name"] = run_name
                    slice_rows.append(item)
    artifacts = {
        "comparison_summary": output / "comparison_summary.csv",
        "average_vs_bwer": output / "average_vs_bwer.csv",
        "geography_slice_comparison": output / "geography_slice_comparison.csv",
        "comparison_report": output / "comparison_report.md",
    }
    write_csv(artifacts["comparison_summary"], summary_rows)
    write_csv(artifacts["average_vs_bwer"], average_rows)
    write_csv(artifacts["geography_slice_comparison"], slice_rows)
    _write_comparison_report(artifacts["comparison_report"], summary_rows)
    return artifacts


def _write_comparison_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# fMoW-Sentinel Geography BWER Run Comparison",
        "",
        "This is a protocol-aware comparison of completed image-only fMoW-Sentinel runs.",
        "Geography metadata is used only for audit slicing and reporting, not as model input.",
        "",
        "| run | accuracy | Raw-BWER(country) | Standardised-BWER(country | class) | protocol |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('run_name','')} | {row.get('aggregate_accuracy','')} | "
            f"{row.get('raw_bwer_country','')} | {row.get('standardised_bwer_country_class','')} | "
            f"{row.get('adaptation_protocol','')} / {row.get('split_protocol','')} |"
        )
    lines.extend(
        [
            "",
            "Report average accuracy together with Raw-BWER and class-standardised BWER. "
            "A higher aggregate score does not by itself imply lower geography tail risk.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def zip_output_dir(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.make_archive(str(destination.with_suffix("")), "zip", source.parent, source.name)
    return destination
