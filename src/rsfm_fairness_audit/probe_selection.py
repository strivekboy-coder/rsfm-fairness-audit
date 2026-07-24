from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class ProbeSelectionError(RuntimeError):
    """Raised when a frozen-encoder probe protocol cannot be executed safely."""


@dataclass(frozen=True)
class MulticlassProbeSearchConfig:
    """Pre-registered train-only model-selection protocol for linear probes.

    The outer calibration and test splits must never be passed as
    ``training_rows``. Hyperparameters and stopping time are selected on a
    deterministic, group-disjoint inner holdout carved only from training data.
    The selected probe is then refit on every outer-training row.
    """

    learning_rates: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-4
    inner_validation_fraction: float = 0.15
    batch_size: int = 512
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if not self.learning_rates or any(float(value) <= 0 for value in self.learning_rates):
            raise ValueError("learning_rates must contain positive values.")
        if self.max_epochs <= 0 or self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("max_epochs, patience, and batch_size must be positive.")
        if not 0.0 < self.inner_validation_fraction < 0.5:
            raise ValueError("inner_validation_fraction must be in (0, 0.5).")


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def group_stratified_inner_split(
    labels: Sequence[str],
    groups: Sequence[str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows by group while retaining every feasible class.

    Groups are assigned within class using a stable seeded hash. A class with
    only one group stays in fitting data and is reported through the returned
    split rather than leaking that group into validation.
    """

    if len(labels) != len(groups) or not labels:
        raise ProbeSelectionError("labels and groups must be non-empty and aligned.")
    by_class: dict[str, list[str]] = {}
    group_label: dict[str, str] = {}
    for label, group in zip(labels, groups):
        label_text = str(label)
        group_text = str(group).strip()
        if not group_text:
            raise ProbeSelectionError("Every inner-split row requires a non-empty group.")
        previous = group_label.get(group_text)
        if previous is not None and previous != label_text:
            raise ProbeSelectionError(
                f"Inner split group {group_text!r} spans labels {previous!r} and {label_text!r}; "
                "use a category-scoped group identifier."
            )
        group_label[group_text] = label_text
        by_class.setdefault(label_text, [])
        if group_text not in by_class[label_text]:
            by_class[label_text].append(group_text)
    validation_groups: set[str] = set()
    for label, class_groups in sorted(by_class.items()):
        ordered = sorted(
            class_groups,
            key=lambda group: (_stable_int(f"{seed}|{label}|{group}"), group),
        )
        if len(ordered) < 2:
            continue
        count = max(1, int(round(len(ordered) * float(validation_fraction))))
        count = min(count, len(ordered) - 1)
        validation_groups.update(ordered[:count])
    fit = np.asarray([index for index, group in enumerate(groups) if str(group) not in validation_groups], dtype=np.int64)
    validation = np.asarray(
        [index for index, group in enumerate(groups) if str(group) in validation_groups], dtype=np.int64
    )
    if not len(fit) or not len(validation):
        raise ProbeSelectionError(
            "The train-only group split produced an empty fitting or validation partition."
        )
    if set(str(groups[index]) for index in fit) & set(str(groups[index]) for index in validation):
        raise ProbeSelectionError("Inner fitting and validation groups overlap.")
    missing_fit = sorted(set(map(str, labels)) - {str(labels[index]) for index in fit})
    if missing_fit:
        raise ProbeSelectionError(f"Inner fitting partition lost classes: {missing_fit}")
    return fit, validation


def group_disjoint_inner_split(
    groups: Sequence[str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic group-disjoint holdout from training data only."""

    if not groups or not 0.0 < float(validation_fraction) < 0.5:
        raise ProbeSelectionError(
            "groups must be non-empty and validation_fraction must be in (0, 0.5)."
        )
    normalized = [str(value).strip() for value in groups]
    if any(not value for value in normalized):
        raise ProbeSelectionError("Every inner-split row requires a non-empty group.")
    unique = sorted(
        set(normalized),
        key=lambda group: (_stable_int(f"{seed}|{group}"), group),
    )
    if len(unique) < 2:
        raise ProbeSelectionError("At least two groups are required for a group-disjoint holdout.")
    count = max(1, int(round(len(unique) * float(validation_fraction))))
    count = min(count, len(unique) - 1)
    validation_groups = set(unique[:count])
    fit = np.asarray(
        [index for index, group in enumerate(normalized) if group not in validation_groups],
        dtype=np.int64,
    )
    validation = np.asarray(
        [index for index, group in enumerate(normalized) if group in validation_groups],
        dtype=np.int64,
    )
    if not len(fit) or not len(validation):
        raise ProbeSelectionError("Group-disjoint split produced an empty partition.")
    return fit, validation


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - Colab/runtime path
        raise ProbeSelectionError("PyTorch is required for probe selection.") from exc
    return torch, nn


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values, dtype=np.float32).mean(axis=0, keepdims=True)
    std = np.maximum(np.asarray(values, dtype=np.float32).std(axis=0, keepdims=True), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def _cross_entropy(
    model: Any,
    values: np.ndarray,
    targets: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: Any,
    batch_size: int,
) -> float:
    torch, _nn = _require_torch()
    losses: list[float] = []
    counts: list[int] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            x = (np.asarray(values[start:end], dtype=np.float32) - mean) / std
            y = np.asarray(targets[start:end], dtype=np.int64)
            logits = model(torch.as_tensor(x, device=device))
            loss = torch.nn.functional.cross_entropy(
                logits, torch.as_tensor(y, device=device), reduction="mean"
            )
            losses.append(float(loss.detach().cpu()))
            counts.append(end - start)
    return float(np.average(losses, weights=counts))


def _fit_candidate(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    class_count: int,
    learning_rate: float,
    config: MulticlassProbeSearchConfig,
    seed: int,
    device: Any,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    torch, nn = _require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    mean, std = _normalization(fit_x)
    model = nn.Linear(fit_x.shape[1], class_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(config.weight_decay)
    )
    rng = np.random.default_rng(seed)
    indices = np.arange(len(fit_x), dtype=np.int64)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.max_epochs + 1):
        rng.shuffle(indices)
        model.train()
        train_losses: list[float] = []
        train_counts: list[int] = []
        for start in range(0, len(indices), config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            x = (np.asarray(fit_x[batch_indices], dtype=np.float32) - mean) / std
            y = np.asarray(fit_y[batch_indices], dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.as_tensor(x, device=device))
            loss = torch.nn.functional.cross_entropy(
                logits, torch.as_tensor(y, device=device)
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_counts.append(len(batch_indices))
        validation_loss = _cross_entropy(
            model,
            validation_x,
            validation_y,
            mean=mean,
            std=std,
            device=device,
            batch_size=config.batch_size,
        )
        record = {
            "epoch": epoch,
            "train_cross_entropy": float(np.average(train_losses, weights=train_counts)),
            "inner_validation_cross_entropy": validation_loss,
            "learning_rate": float(learning_rate),
        }
        history.append(record)
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= config.patience:
            break
    if best_state is None or best_epoch <= 0:
        raise ProbeSelectionError("Probe selection did not produce a finite validation checkpoint.")
    return best_state, best_epoch, history


def _fit_full(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    class_count: int,
    learning_rate: float,
    epochs: int,
    config: MulticlassProbeSearchConfig,
    seed: int,
    device: Any,
) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    torch, nn = _require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    mean, std = _normalization(train_x)
    model = nn.Linear(train_x.shape[1], class_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(config.weight_decay)
    )
    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_x), dtype=np.int64)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        rng.shuffle(indices)
        model.train()
        losses: list[float] = []
        counts: list[int] = []
        for start in range(0, len(indices), config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            x = (np.asarray(train_x[batch_indices], dtype=np.float32) - mean) / std
            y = np.asarray(train_y[batch_indices], dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.as_tensor(x, device=device))
            loss = torch.nn.functional.cross_entropy(
                logits, torch.as_tensor(y, device=device)
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            counts.append(len(batch_indices))
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": float(np.average(losses, weights=counts)),
                "learning_rate": float(learning_rate),
            }
        )
    return model, mean, std, history


def _predict(
    model: Any,
    values: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch, _nn = _require_torch()
    logits_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            normalized = (np.asarray(values[start:end], dtype=np.float32) - mean) / std
            logits = model(torch.as_tensor(normalized, device=device))
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32))
    logits_array = np.concatenate(logits_parts, axis=0)
    shifted = logits_array - logits_array.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return logits_array, probabilities.astype(np.float32)


def fit_selected_multiclass_probe(
    train_embeddings: np.ndarray,
    train_labels: Sequence[str],
    train_groups: Sequence[str],
    evaluation_embeddings: Mapping[str, np.ndarray],
    output_dir: str | Path,
    *,
    config: MulticlassProbeSearchConfig = MulticlassProbeSearchConfig(),
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """Select and refit one seed of a linear multiclass probe.

    Returns complete logits and probabilities for every named evaluation
    matrix. The outer calibration and test labels are intentionally absent
    from this API, preventing accidental test-aware model selection.
    """

    x_train = np.asarray(train_embeddings, dtype=np.float32)
    labels = [str(value) for value in train_labels]
    if x_train.ndim != 2 or len(x_train) != len(labels):
        raise ProbeSelectionError("train embeddings and labels must align as [n,d] and [n].")
    classes = sorted(set(labels))
    class_to_index = {name: index for index, name in enumerate(classes)}
    y_train = np.asarray([class_to_index[name] for name in labels], dtype=np.int64)
    fit_indices, validation_indices = group_stratified_inner_split(
        labels,
        train_groups,
        validation_fraction=config.inner_validation_fraction,
        seed=seed,
    )
    torch, _nn = _require_torch()
    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    candidates: list[dict[str, Any]] = []
    for index, learning_rate in enumerate(config.learning_rates):
        _, best_epoch, history = _fit_candidate(
            x_train[fit_indices],
            y_train[fit_indices],
            x_train[validation_indices],
            y_train[validation_indices],
            class_count=len(classes),
            learning_rate=float(learning_rate),
            config=config,
            seed=int(seed + 1009 * index),
            device=resolved_device,
        )
        best_row = min(history, key=lambda row: float(row["inner_validation_cross_entropy"]))
        candidates.append(
            {
                "learning_rate": float(learning_rate),
                "selected_epoch": int(best_epoch),
                "inner_validation_cross_entropy": float(
                    best_row["inner_validation_cross_entropy"]
                ),
                "history": history,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            float(row["inner_validation_cross_entropy"]),
            float(row["learning_rate"]),
            int(row["selected_epoch"]),
        ),
    )
    model, mean, std, full_history = _fit_full(
        x_train,
        y_train,
        class_count=len(classes),
        learning_rate=float(selected["learning_rate"]),
        epochs=int(selected["selected_epoch"]),
        config=config,
        seed=seed,
        device=resolved_device,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "linear_probe.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "class_to_index": class_to_index,
            "embedding_mean": mean,
            "embedding_std": std,
            "seed": int(seed),
            "selected_learning_rate": float(selected["learning_rate"]),
            "selected_epoch": int(selected["selected_epoch"]),
            "search_config": asdict(config),
            "inner_fit_indices_sha256": hashlib.sha256(fit_indices.tobytes()).hexdigest(),
            "inner_validation_indices_sha256": hashlib.sha256(
                validation_indices.tobytes()
            ).hexdigest(),
        },
        checkpoint,
    )
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for split, values in evaluation_embeddings.items():
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != x_train.shape[1]:
            raise ProbeSelectionError(
                f"Evaluation embeddings for {split!r} have incompatible shape {matrix.shape}."
            )
        logits, probabilities = _predict(
            model,
            matrix,
            mean=mean,
            std=std,
            device=resolved_device,
            batch_size=config.batch_size,
        )
        predictions[str(split)] = {"logits": logits, "probabilities": probabilities}
        np.savez_compressed(
            output / f"{split}_predictions.npz",
            logits=logits,
            probabilities=probabilities,
            class_names=np.asarray(classes, dtype=str),
        )
    manifest = {
        "schema": "geobwer.multiclass_probe_selection.v1",
        "seed": int(seed),
        "search_data": "outer_train_only_group_disjoint_inner_holdout",
        "outer_calibration_or_test_labels_used": False,
        "inner_fit_count": int(len(fit_indices)),
        "inner_validation_count": int(len(validation_indices)),
        "inner_fit_group_count": len({str(train_groups[index]) for index in fit_indices}),
        "inner_validation_group_count": len(
            {str(train_groups[index]) for index in validation_indices}
        ),
        "search_config": asdict(config),
        "candidates": candidates,
        "selected_learning_rate": float(selected["learning_rate"]),
        "selected_epoch": int(selected["selected_epoch"]),
        "full_train_history": full_history,
        "checkpoint": str(checkpoint),
    }
    manifest_path = output / "probe_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "classes": classes,
        "class_to_index": class_to_index,
        "checkpoint": checkpoint,
        "manifest": manifest_path,
        "predictions": predictions,
        "selection": manifest,
    }


__all__ = [
    "MulticlassProbeSearchConfig",
    "ProbeSelectionError",
    "fit_selected_multiclass_probe",
    "group_disjoint_inner_split",
    "group_stratified_inner_split",
]
