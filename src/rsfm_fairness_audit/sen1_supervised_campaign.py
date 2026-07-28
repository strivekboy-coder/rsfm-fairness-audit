from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output
from rsfm_fairness_audit.probe_selection import group_disjoint_inner_split
from rsfm_fairness_audit.sen1floods11_formal import write_sen1_probability_export
from rsfm_fairness_audit.terramind_sen1_config import (
    OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS,
    SEN1FLOODS11_SUFFIXES,
    prepare_terramind_sen1_splits,
    read_sen1floods11_split_prefixes,
)
from rsfm_fairness_audit.unet_baseline import S2ResNet34UNet, masked_bce_dice_loss


class Sen1SupervisedCampaignError(RuntimeError):
    """Raised when the protocol-matched supervised Sen1 campaign is invalid."""


SENSOR_MODES = ("S1", "S2", "S1+S2")
MODE_CHANNELS = {"S1": 2, "S2": 13, "S1+S2": 15}
FORMAL_MASK_VALUES = {-1, 0, 1}


@dataclass(frozen=True)
class Sen1SupervisedConfig:
    s1_root: Path
    s2_root: Path
    label_root: Path
    train_split: Path
    validation_split: Path
    test_split: Path
    output_dir: Path
    persistent_output_dir: Path | None = None
    sensor_modes: tuple[str, ...] = SENSOR_MODES
    seeds: tuple[int, ...] = (42, 73, 101)
    max_epochs: int = 100
    batch_size: int = 4
    num_workers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    min_delta: float = 1e-4
    pretrained_encoder: bool = False
    device: str = "auto"
    amp: bool = True
    diagnostic_max_samples: int | None = None

    def __post_init__(self) -> None:
        normalized = tuple(str(value).upper().replace(" ", "") for value in self.sensor_modes)
        if not normalized or any(value not in SENSOR_MODES for value in normalized):
            raise ValueError(f"sensor_modes must be selected from {SENSOR_MODES}.")
        if not self.seeds or len(set(map(int, self.seeds))) != len(self.seeds):
            raise ValueError("seeds must be a non-empty set of unique integers.")
        if self.diagnostic_max_samples is None and len(self.seeds) < 3:
            raise ValueError("Formal Sen1 supervised training requires at least three seeds.")
        if min(self.max_epochs, self.batch_size, self.patience) <= 0:
            raise ValueError("max_epochs, batch_size, and patience must be positive.")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Colab/runtime path
        raise Sen1SupervisedCampaignError("PyTorch is required for the Sen1 supervised campaign.") from exc
    return torch


def _read_raster(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - Colab/runtime path
        raise Sen1SupervisedCampaignError("rasterio is required for Sen1Floods11 GeoTIFFs.") from exc
    with rasterio.open(path) as dataset:
        return np.asarray(dataset.read())


def _paths(config: Sen1SupervisedConfig, prefix: str) -> dict[str, Path]:
    return {
        "S1": config.s1_root / f"{prefix}{SEN1FLOODS11_SUFFIXES['s1']}",
        "S2": config.s2_root / f"{prefix}{SEN1FLOODS11_SUFFIXES['s2']}",
        "label": config.label_root / f"{prefix}{SEN1FLOODS11_SUFFIXES['label']}",
    }


def _mode_array(config: Sen1SupervisedConfig, prefix: str, mode: str) -> np.ndarray:
    paths = _paths(config, prefix)
    if mode == "S1":
        array = _read_raster(paths["S1"])
    elif mode == "S2":
        array = _read_raster(paths["S2"])
    else:
        array = np.concatenate([_read_raster(paths["S1"]), _read_raster(paths["S2"])], axis=0)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != MODE_CHANNELS[mode]:
        raise Sen1SupervisedCampaignError(
            f"Expected {MODE_CHANNELS[mode]} channels for {mode}, got {array.shape} at prefix={prefix}."
        )
    return array


def _mask(config: Sen1SupervisedConfig, prefix: str) -> np.ndarray:
    value = np.asarray(_read_raster(_paths(config, prefix)["label"])).squeeze()
    if value.ndim != 2:
        raise Sen1SupervisedCampaignError(f"Expected a 2D label mask for {prefix}, got {value.shape}.")
    raw_values = np.unique(value).tolist()
    try:
        numeric_values = [float(item) for item in raw_values]
    except (TypeError, ValueError) as exc:
        raise Sen1SupervisedCampaignError(
            f"Label mask contains non-numeric values for {prefix}: {raw_values}."
        ) from exc
    invalid_values = sorted(
        item
        for item in numeric_values
        if not np.isfinite(item) or item not in {-1.0, 0.0, 1.0}
    )
    if invalid_values:
        raise Sen1SupervisedCampaignError(
            f"Label mask values must belong to {{-1,0,1}} for {prefix}; "
            f"invalid={invalid_values}."
        )
    # Preserve the official ignore semantics. A chip containing only -1 is a
    # valid split member even though it contributes no supervised pixels.
    return np.asarray(value, dtype=np.int64)


def _prefix_sha256(prefixes: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(map(str, prefixes)) + "\n").encode("utf-8")
    ).hexdigest()


def _prefix_lineage(prefixes: Sequence[str]) -> dict[str, Any]:
    values = [str(prefix) for prefix in prefixes]
    return {
        "sample_count": len(values),
        "prefix_sha256": _prefix_sha256(values),
        "prefixes": values,
    }


def _diagnostic_prefix_subset(
    prefixes: Sequence[str],
    maximum: int,
    *,
    require_multiple_groups: bool = False,
) -> list[str]:
    """Bound a smoke subset without collapsing an event-ordered split."""

    if int(maximum) <= 0:
        raise Sen1SupervisedCampaignError("diagnostic_max_samples must be positive.")
    by_group: dict[str, list[str]] = {}
    for prefix in prefixes:
        by_group.setdefault(str(prefix).split("_", 1)[0], []).append(str(prefix))
    if require_multiple_groups and len(by_group) < 2:
        raise Sen1SupervisedCampaignError(
            "The diagnostic training subset requires at least two event groups."
        )
    selected: list[str] = []
    depth = 0
    target = min(int(maximum), len(prefixes))
    groups = sorted(by_group)
    while len(selected) < target:
        added = False
        for group in groups:
            members = by_group[group]
            if depth < len(members):
                selected.append(members[depth])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        depth += 1
    if require_multiple_groups and len({item.split("_", 1)[0] for item in selected}) < 2:
        raise Sen1SupervisedCampaignError(
            "diagnostic_max_samples is too small to retain two event groups."
        )
    return selected


def compute_train_normalization(
    config: Sen1SupervisedConfig,
    prefixes: Sequence[str],
    mode: str,
) -> dict[str, Any]:
    """Compute per-band moments from the official train split only."""

    if not prefixes:
        raise Sen1SupervisedCampaignError("Train normalization requires non-empty prefixes.")
    count = 0
    total = np.zeros(MODE_CHANNELS[mode], dtype=np.float64)
    square = np.zeros(MODE_CHANNELS[mode], dtype=np.float64)
    minima = np.full(MODE_CHANNELS[mode], np.inf, dtype=np.float64)
    maxima = np.full(MODE_CHANNELS[mode], -np.inf, dtype=np.float64)
    for index, prefix in enumerate(prefixes, start=1):
        image = _mode_array(config, prefix, mode).astype(np.float64)
        flat = image.reshape(image.shape[0], -1)
        finite = np.all(np.isfinite(flat), axis=0)
        if not np.any(finite):
            raise Sen1SupervisedCampaignError(f"No jointly finite pixels for {prefix}.")
        selected = flat[:, finite]
        total += selected.sum(axis=1)
        square += np.square(selected).sum(axis=1)
        minima = np.minimum(minima, selected.min(axis=1))
        maxima = np.maximum(maxima, selected.max(axis=1))
        count += selected.shape[1]
        if index % 50 == 0:
            print(
                f"[sen1:baseline:norm] mode={mode} files={index}/{len(prefixes)} pixels={count}",
                flush=True,
            )
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-12)
    return {
        "schema": "geobwer.sen1floods11.train_normalization.v2",
        "sensor_mode": mode,
        "selection_split": "official_train",
        "test_rows_used": False,
        "normalization_sample_count": len(prefixes),
        "sample_count": len(prefixes),
        "sample_prefix_sha256": _prefix_sha256(prefixes),
        "sample_prefixes": list(map(str, prefixes)),
        "pixel_count": int(count),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "min": minima.tolist(),
        "max": maxima.tolist(),
    }


class _Dataset:
    def __init__(
        self,
        config: Sen1SupervisedConfig,
        prefixes: Sequence[str],
        mode: str,
        normalization: Mapping[str, Any],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.config = config
        self.prefixes = list(prefixes)
        self.mode = mode
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)[:, None, None]
        self.std = np.maximum(
            np.asarray(normalization["std"], dtype=np.float32)[:, None, None], 1e-6
        )
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.prefixes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> tuple[Any, Any, str]:
        torch = _require_torch()
        prefix = self.prefixes[index]
        image = (_mode_array(self.config, prefix, self.mode) - self.mean) / self.std
        if not np.all(np.isfinite(image)):
            raw = _mode_array(self.config, prefix, self.mode)
            raise Sen1SupervisedCampaignError(
                "Normalized input contains NaN/Inf: "
                f"prefix={prefix}, mode={self.mode}, "
                f"raw_range=[{float(np.nanmin(raw))},{float(np.nanmax(raw))}], "
                f"normalization_mean={self.mean[:, 0, 0].tolist()}, "
                f"normalization_std={self.std[:, 0, 0].tolist()}."
            )
        mask = _mask(self.config, prefix)
        if self.augment:
            rng = random.Random(
                int(hashlib.sha256(f"{self.seed}|{self.epoch}|{prefix}".encode()).hexdigest()[:16], 16)
            )
            rotation = rng.randrange(4)
            image = np.rot90(image, rotation, axes=(-2, -1)).copy()
            mask = np.rot90(mask, rotation, axes=(-2, -1)).copy()
            if rng.random() < 0.5:
                image = image[..., ::-1].copy()
                mask = mask[..., ::-1].copy()
            if rng.random() < 0.5:
                image = image[..., ::-1, :].copy()
                mask = mask[..., ::-1, :].copy()
        return (
            torch.from_numpy(np.asarray(image, dtype=np.float32)),
            torch.from_numpy(np.asarray(mask, dtype=np.int64)),
            prefix,
        )


def _loader(dataset: _Dataset, config: Sen1SupervisedConfig, *, shuffle: bool) -> Any:
    torch = _require_torch()
    generator = torch.Generator()
    generator.manual_seed(dataset.seed + dataset.epoch)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=bool(config.num_workers > 0),
    )


def _device(value: str) -> Any:
    torch = _require_torch()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Sen1SupervisedCampaignError("CUDA was requested but is unavailable.")
    return device


def _confusion(probability: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    valid = np.isin(target, (0, 1))
    truth = target == 1
    prediction = probability >= 0.5
    return (
        int(np.sum(valid & prediction & truth)),
        int(np.sum(valid & prediction & ~truth)),
        int(np.sum(valid & ~prediction & truth)),
    )


def _tensor_range(value: Any) -> dict[str, Any]:
    torch = _require_torch()
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    if finite_count == 0:
        return {
            "shape": list(detached.shape),
            "finite_count": 0,
            "minimum": None,
            "maximum": None,
        }
    selected = detached[finite]
    return {
        "shape": list(detached.shape),
        "finite_count": finite_count,
        "minimum": float(selected.min().cpu()),
        "maximum": float(selected.max().cpu()),
    }


def _batch_support(masks: Any, prefixes: Sequence[str]) -> dict[str, Any]:
    torch = _require_torch()
    unique_values = sorted(int(value) for value in torch.unique(masks).detach().cpu().tolist())
    invalid_values = sorted(set(unique_values) - FORMAL_MASK_VALUES)
    if invalid_values:
        raise Sen1SupervisedCampaignError(
            "Training/evaluation mask values must belong to {-1,0,1}: "
            f"prefixes={list(map(str, prefixes))}, invalid={invalid_values}."
        )
    valid_by_sample = (
        torch.isin(masks, torch.tensor([0, 1], device=masks.device))
        .reshape(masks.shape[0], -1)
        .sum(dim=1)
        .detach()
        .cpu()
        .tolist()
    )
    return {
        "prefixes": list(map(str, prefixes)),
        "valid_pixel_counts": [int(value) for value in valid_by_sample],
        "aggregate_valid_pixel_count": int(sum(valid_by_sample)),
        "observed_target_values": unique_values,
    }


def _evaluate(
    model: Any,
    loader: Any,
    device: Any,
    *,
    mode: str,
) -> tuple[
    float,
    list[np.ndarray],
    list[np.ndarray],
    list[str],
    dict[str, Any],
]:
    torch = _require_torch()
    model.eval()
    tp = fp = fn = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    prefixes: list[str] = []
    aggregate_valid_pixel_count = 0
    all_ignore_row_count = 0
    observed_target_values: set[int] = set()
    with torch.inference_mode():
        for images, masks, batch_prefixes in loader:
            images = images.to(device, non_blocking=True)
            if not bool(torch.isfinite(images).all().item()):
                raise Sen1SupervisedCampaignError(
                    "Evaluation input contains NaN/Inf: "
                    f"mode={mode}, prefixes={list(map(str, batch_prefixes))}, "
                    f"input_range={_tensor_range(images)}."
                )
            logits = model(images)
            if not bool(torch.isfinite(logits).all().item()):
                raise Sen1SupervisedCampaignError(
                    "Evaluation logits contain NaN/Inf: "
                    f"mode={mode}, prefixes={list(map(str, batch_prefixes))}, "
                    f"input_range={_tensor_range(images)}, "
                    f"logits_range={_tensor_range(logits)}."
                )
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            mask_values = masks.numpy().astype(np.int16)
            for probability, target, prefix in zip(probs, mask_values, batch_prefixes):
                target_values = {int(value) for value in np.unique(target).tolist()}
                invalid_values = sorted(target_values - FORMAL_MASK_VALUES)
                if invalid_values:
                    raise Sen1SupervisedCampaignError(
                        f"Evaluation target values are invalid for {prefix}: {invalid_values}."
                    )
                valid_pixel_count = int(np.isin(target, [0, 1]).sum())
                aggregate_valid_pixel_count += valid_pixel_count
                all_ignore_row_count += int(valid_pixel_count == 0)
                observed_target_values.update(target_values)
                ctp, cfp, cfn = _confusion(probability, target)
                tp += ctp
                fp += cfp
                fn += cfn
                probabilities.append(probability)
                targets.append(target)
                prefixes.append(str(prefix))
    if aggregate_valid_pixel_count <= 0:
        raise Sen1SupervisedCampaignError(
            f"Evaluation split has no valid 0/1 pixels across any row for mode={mode}."
        )
    union = tp + fp + fn
    support = {
        "row_count": len(prefixes),
        "all_ignore_row_count": all_ignore_row_count,
        "valid_row_count": len(prefixes) - all_ignore_row_count,
        "aggregate_valid_pixel_count": aggregate_valid_pixel_count,
        "observed_target_values": sorted(observed_target_values),
        "valid_pixel_counts": [
            int(np.isin(target, [0, 1]).sum()) for target in targets
        ],
    }
    return (
        float(tp / union) if union else 1.0,
        probabilities,
        targets,
        prefixes,
        support,
    )


def _training_batch_step(
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    images: Any,
    masks: Any,
    prefixes: Sequence[str],
    device: Any,
    mode: str,
    amp: bool,
) -> dict[str, Any]:
    """Execute one effective optimization step or explicitly skip all-ignore."""

    torch = _require_torch()
    support = _batch_support(masks, prefixes)
    if not bool(torch.isfinite(images).all().item()):
        raise Sen1SupervisedCampaignError(
            "Normalized training input contains NaN/Inf: "
            f"mode={mode}, prefixes={support['prefixes']}, "
            f"input_range={_tensor_range(images)}, "
            f"valid_pixel_counts={support['valid_pixel_counts']}."
        )
    if support["aggregate_valid_pixel_count"] == 0:
        return {
            **support,
            "skipped_all_ignore": True,
            "loss": None,
        }
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        "cuda", enabled=bool(amp and device.type == "cuda")
    ):
        logits = model(images)
        if not bool(torch.isfinite(logits).all().item()):
            raise Sen1SupervisedCampaignError(
                "Training logits contain NaN/Inf: "
                f"mode={mode}, prefixes={support['prefixes']}, "
                f"input_range={_tensor_range(images)}, "
                f"logits_range={_tensor_range(logits)}, "
                f"valid_pixel_counts={support['valid_pixel_counts']}."
            )
        loss = masked_bce_dice_loss(logits, masks)
    if not bool(torch.isfinite(loss).all().item()):
        raise Sen1SupervisedCampaignError(
            "Training loss is NaN/Inf for an effective batch: "
            f"mode={mode}, prefixes={support['prefixes']}, "
            f"input_range={_tensor_range(images)}, "
            f"logits_range={_tensor_range(logits)}, "
            f"valid_pixel_counts={support['valid_pixel_counts']}."
        )
    scale_before = float(scaler.get_scale())
    if not np.isfinite(scale_before):
        raise Sen1SupervisedCampaignError(
            "AMP gradient scale is NaN/Inf before backward: "
            f"mode={mode}, prefixes={support['prefixes']}, scale={scale_before}."
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    invalid_gradient_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
    ]
    if invalid_gradient_parameters:
        optimizer.zero_grad(set_to_none=True)
        raise Sen1SupervisedCampaignError(
            "Training gradients contain NaN/Inf before optimizer.step: "
            f"mode={mode}, prefixes={support['prefixes']}, "
            f"parameters={invalid_gradient_parameters[:20]}, "
            f"input_range={_tensor_range(images)}, "
            f"logits_range={_tensor_range(logits)}, "
            f"valid_pixel_counts={support['valid_pixel_counts']}."
        )
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if not np.isfinite(scale_after):
        raise Sen1SupervisedCampaignError(
            "AMP gradient scale is NaN/Inf after optimizer update: "
            f"mode={mode}, prefixes={support['prefixes']}, scale={scale_after}."
        )
    invalid_parameter_values = [
        name
        for name, parameter in model.named_parameters()
        if not bool(torch.isfinite(parameter).all().item())
    ]
    if invalid_parameter_values:
        raise Sen1SupervisedCampaignError(
            "Model parameters contain NaN/Inf after optimizer.step: "
            f"mode={mode}, prefixes={support['prefixes']}, "
            f"parameters={invalid_parameter_values[:20]}."
        )
    return {
        **support,
        "skipped_all_ignore": False,
        "loss": float(loss.detach().cpu()),
    }


def _train_seed(
    config: Sen1SupervisedConfig,
    *,
    mode: str,
    seed: int,
    split_prefixes: Mapping[str, Sequence[str]],
    normalization: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    torch = _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(config.device)
    model = S2ResNet34UNet(
        in_channels=MODE_CHANNELS[mode],
        pretrained_encoder=config.pretrained_encoder,
    ).to(device)
    parameter_device = str(next(model.parameters()).device)
    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "not_applicable"
    )
    print(
        f"[sen1:baseline:device] resolved={device} gpu={gpu_name} "
        f"model_parameter_device={parameter_device}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, config.patience // 3)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.amp and device.type == "cuda"))
    train_prefixes = list(split_prefixes["train"])
    fit_indices, selection_indices = group_disjoint_inner_split(
        [prefix.split("_", 1)[0] for prefix in train_prefixes],
        validation_fraction=0.18,
        seed=seed,
    )
    inner_fit_prefixes = [train_prefixes[index] for index in fit_indices]
    inner_selection_prefixes = [train_prefixes[index] for index in selection_indices]
    train_dataset = _Dataset(
        config, inner_fit_prefixes, mode, normalization, augment=True, seed=seed
    )
    selection_dataset = _Dataset(
        config,
        inner_selection_prefixes,
        mode,
        normalization,
        augment=False,
        seed=seed,
    )
    selection_loader = _loader(selection_dataset, config, shuffle=False)
    best_iou = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    skipped_all_ignore_batch_count = 0
    for epoch in range(1, config.max_epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loader = _loader(train_dataset, config, shuffle=True)
        model.train()
        losses: list[float] = []
        epoch_skipped_all_ignore_batch_count = 0
        for batch_index, (images, masks, _prefixes) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if epoch == 1 and batch_index == 0:
                print(
                    f"[sen1:baseline:device] mode={mode} seed={seed} "
                    f"input_tensor_device={images.device} mask_tensor_device={masks.device}",
                    flush=True,
                )
            batch_result = _training_batch_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                images=images,
                masks=masks,
                prefixes=_prefixes,
                device=device,
                mode=mode,
                amp=config.amp,
            )
            if batch_result["skipped_all_ignore"]:
                skipped_all_ignore_batch_count += 1
                epoch_skipped_all_ignore_batch_count += 1
                print(
                    "[sen1:baseline] skipped all-ignore training batch "
                    f"mode={mode} seed={seed} epoch={epoch} "
                    f"prefixes={batch_result['prefixes']}",
                    flush=True,
                )
                continue
            losses.append(float(batch_result["loss"]))
        if not losses:
            raise Sen1SupervisedCampaignError(
                f"Training epoch has no effective 0/1 pixels: mode={mode}, "
                f"seed={seed}, epoch={epoch}."
            )
        selection_iou, _, _, _, selection_support = _evaluate(
            model, selection_loader, device, mode=mode
        )
        scheduler.step(selection_iou)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "inner_selection_micro_iou": selection_iou,
                "inner_selection_support": selection_support,
                "effective_batch_count": len(losses),
                "skipped_all_ignore_batch_count": epoch_skipped_all_ignore_batch_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"[sen1:baseline] mode={mode} seed={seed} epoch={epoch}/{config.max_epochs} "
            f"loss={history[-1]['train_loss']:.6f} inner_iou={selection_iou:.6f}",
            flush=True,
        )
        if selection_iou > best_iou + config.min_delta:
            best_iou = selection_iou
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= config.patience:
            break
    if best_state is None:
        raise Sen1SupervisedCampaignError("No finite supervised checkpoint was selected.")
    # Refit a fresh model on every official-training sample for the selected
    # epoch count. The official validation set remains untouched for spatial
    # calibration and CRC.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = S2ResNet34UNet(
        in_channels=MODE_CHANNELS[mode],
        pretrained_encoder=config.pretrained_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.amp and device.type == "cuda"))
    full_train_dataset = _Dataset(
        config, train_prefixes, mode, normalization, augment=True, seed=seed + 100_003
    )
    refit_history: list[dict[str, Any]] = []
    refit_skipped_all_ignore_batch_count = 0
    for epoch in range(1, best_epoch + 1):
        full_train_dataset.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        epoch_skipped_all_ignore_batch_count = 0
        for images, masks, _prefixes in _loader(
            full_train_dataset, config, shuffle=True
        ):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_result = _training_batch_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                images=images,
                masks=masks,
                prefixes=_prefixes,
                device=device,
                mode=mode,
                amp=config.amp,
            )
            if batch_result["skipped_all_ignore"]:
                refit_skipped_all_ignore_batch_count += 1
                epoch_skipped_all_ignore_batch_count += 1
                continue
            losses.append(float(batch_result["loss"]))
        if not losses:
            raise Sen1SupervisedCampaignError(
                f"Refit epoch has no effective 0/1 pixels: mode={mode}, "
                f"seed={seed}, epoch={epoch}."
            )
        refit_history.append(
            {
                "epoch": epoch,
                "full_train_loss": float(np.mean(losses)),
                "effective_batch_count": len(losses),
                "skipped_all_ignore_batch_count": epoch_skipped_all_ignore_batch_count,
            }
        )
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_resnet34_unet.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "sensor_mode": mode,
            "input_channels": MODE_CHANNELS[mode],
            "seed": seed,
            "best_epoch": best_epoch,
            "best_inner_selection_iou": best_iou,
            "selection_split": "official_train_inner_event_disjoint",
            "outer_validation_used_for_model_selection": False,
            "inner_fit_prefixes": inner_fit_prefixes,
            "inner_selection_prefixes": inner_selection_prefixes,
            "normalization": dict(normalization),
            "config": asdict(config),
        },
        checkpoint,
    )
    exports: dict[str, Path] = {}
    split_metrics: dict[str, float] = {}
    split_support: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        dataset = _Dataset(
            config, split_prefixes[split], mode, normalization, augment=False, seed=seed
        )
        iou, probabilities, targets, prefixes, support = _evaluate(
            model, _loader(dataset, config, shuffle=False), device, mode=mode
        )
        filenames = [
            {
                "S1GRD": str(_paths(config, prefix)["S1"]),
                "S2L1C": str(_paths(config, prefix)["S2"]),
            }
            for prefix in prefixes
        ]
        exports[split] = write_sen1_probability_export(
            output_dir / "probabilities" / split,
            probabilities=probabilities,
            targets=targets,
            filenames=filenames,
            batch_size=config.batch_size,
        )
        support_path = exports[split] / "support_contract.json"
        support_path.write_text(
            json.dumps(
                {
                    "schema": "geobwer.sen1floods11.probability_support.v1",
                    "split": split,
                    "sensor_mode": mode,
                    **support,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        split_support[split] = {
            **support,
            "support_contract": str(support_path),
            "support_contract_sha256": file_sha256(support_path),
        }
        split_metrics[split] = iou
    manifest = output_dir / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.supervised_resnet34_unet.v2",
                "formal_evidence": config.diagnostic_max_samples is None,
                "architecture": "resnet34_unet",
                "adaptation_protocol": "supervised_from_scratch_decoder_imagenet_encoder_initialization"
                if config.pretrained_encoder
                else "supervised_from_scratch",
                "sensor_mode": mode,
                "input_channels": MODE_CHANNELS[mode],
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_iou": best_iou,
                "best_inner_selection_iou": best_iou,
                "model_selection": "official_train_inner_event_disjoint",
                "outer_validation_used_for_model_selection": False,
                "split_metrics": split_metrics,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "normalization": dict(normalization),
                "probability_exports": {key: str(value) for key, value in exports.items()},
                "split_support": split_support,
                "skipped_all_ignore_batch_count": skipped_all_ignore_batch_count,
                "refit_skipped_all_ignore_batch_count": refit_skipped_all_ignore_batch_count,
                "history": history,
                "refit_history": refit_history,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint,
        "manifest": manifest,
        "validation_export": exports["validation"],
        "test_export": exports["test"],
        "validation_iou": best_iou,
        "test_iou": split_metrics["test"],
    }


def _reuse_completed_seed(
    run_dir: Path,
    *,
    mode: str,
    seed: int,
    expected_validation: int,
    expected_test: int,
) -> dict[str, Any] | None:
    manifest_path = run_dir / "run_manifest.json"
    checkpoint = run_dir / "best_resnet34_unet.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "geobwer.sen1floods11.supervised_resnet34_unet.v2"
        or str(payload.get("sensor_mode")) != mode
        or int(payload.get("seed", -1)) != int(seed)
        or str(payload.get("checkpoint_sha256", "")) != file_sha256(checkpoint)
    ):
        raise Sen1SupervisedCampaignError(
            f"Completed seed artifacts conflict with the frozen protocol under {run_dir}."
        )
    exports = {
        "validation": run_dir / "probabilities" / "validation",
        "test": run_dir / "probabilities" / "test",
    }
    for split, expected in (("validation", expected_validation), ("test", expected_test)):
        index = exports[split] / "index_parts" / "part-000000.jsonl"
        support_path = exports[split] / "support_contract.json"
        if not index.is_file() or not support_path.is_file():
            return None
        rows = [
            json.loads(line)
            for line in index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != expected:
            return None
        if not all(
            (exports[split] / str(row["probability_path"])).is_file()
            for row in rows
        ):
            return None
        support = json.loads(support_path.read_text(encoding="utf-8"))
        manifest_support = payload.get("split_support", {}).get(split, {})
        if (
            support.get("schema")
            != "geobwer.sen1floods11.probability_support.v1"
            or int(support.get("row_count", -1)) != expected
            or int(support.get("aggregate_valid_pixel_count", 0)) <= 0
            or str(manifest_support.get("support_contract_sha256", ""))
            != file_sha256(support_path)
        ):
            raise Sen1SupervisedCampaignError(
                f"Completed probability support contract is invalid: {support_path}."
            )
    print(f"[sen1:baseline] reusing completed mode={mode} seed={seed}", flush=True)
    return {
        "checkpoint": checkpoint,
        "manifest": manifest_path,
        "validation_export": exports["validation"],
        "test_export": exports["test"],
        "validation_iou": float(payload["best_validation_iou"]),
        "test_iou": float(payload["split_metrics"]["test"]),
    }


def run_sen1_supervised_campaign(config: Sen1SupervisedConfig) -> dict[str, Any]:
    """Train all protocol-matched modes/seeds and export complete maps.

    Formal GeoBWER is intentionally a later stage: common spatial-block
    calibration must first see validation exports from TerraMind, these
    baselines, and any external Prithvi reference.
    """

    hydrate_output(config.output_dir, config.persistent_output_dir)
    split_report = prepare_terramind_sen1_splits(
        {
            "s1_root": config.s1_root,
            "s2_root": config.s2_root,
            "label_root": config.label_root,
            "train_split": config.train_split,
            "val_split": config.validation_split,
            "test_split": config.test_split,
        },
        config.output_dir / "official_split_adapter",
    )
    official_prefixes = {
        "train": read_sen1floods11_split_prefixes(config.train_split),
        "validation": read_sen1floods11_split_prefixes(config.validation_split),
        "test": read_sen1floods11_split_prefixes(config.test_split),
    }
    official_counts = {
        key: len(value) for key, value in official_prefixes.items()
    }
    if official_counts != OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS:
        raise Sen1SupervisedCampaignError(
            "Official Sen1 split counts changed: "
            f"expected={OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS}, "
            f"observed={official_counts}."
        )
    if config.diagnostic_max_samples:
        prefixes = {
            key: _diagnostic_prefix_subset(
                values,
                int(config.diagnostic_max_samples),
                require_multiple_groups=key == "train",
            )
            for key, values in official_prefixes.items()
        }
    else:
        prefixes = {
            key: list(values) for key, values in official_prefixes.items()
        }
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    normalization_contracts: dict[str, dict[str, Any]] = {}
    for mode in config.sensor_modes:
        mode = str(mode).upper().replace(" ", "")
        normalization_path = output / "normalization" / f"{mode.lower().replace('+', '_plus_')}.json"
        normalization_path.parent.mkdir(parents=True, exist_ok=True)
        expected_normalization_hash = _prefix_sha256(official_prefixes["train"])
        if normalization_path.exists():
            normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
            if (
                normalization.get("schema")
                != "geobwer.sen1floods11.train_normalization.v2"
                or normalization.get("sensor_mode") != mode
                or normalization.get("selection_split") != "official_train"
                or normalization.get("test_rows_used") is not False
                or int(normalization.get("normalization_sample_count", -1))
                != OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS["train"]
                or normalization.get("sample_prefix_sha256")
                != expected_normalization_hash
                or list(normalization.get("sample_prefixes", []))
                != list(official_prefixes["train"])
            ):
                raise Sen1SupervisedCampaignError(
                    f"Invalid cached normalization contract: {normalization_path}"
                )
        else:
            normalization = compute_train_normalization(
                config, official_prefixes["train"], mode
            )
            normalization_path.write_text(
                json.dumps(normalization, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        normalization_contracts[mode] = {
            "path": str(normalization_path),
            "sha256": file_sha256(normalization_path),
            "normalization_sample_count": int(
                normalization["normalization_sample_count"]
            ),
            "sample_prefix_sha256": str(
                normalization["sample_prefix_sha256"]
            ),
        }
        for seed in config.seeds:
            name = f"resnet34_unet_{mode.lower().replace('+', '_plus_')}_seed_{int(seed)}"
            run_dir = (
                output
                / mode.lower().replace("+", "_plus_")
                / f"seed_{int(seed)}"
            )
            results[name] = _reuse_completed_seed(
                run_dir,
                mode=mode,
                seed=int(seed),
                expected_validation=len(prefixes["validation"]),
                expected_test=len(prefixes["test"]),
            ) or _train_seed(
                config,
                mode=mode,
                seed=int(seed),
                split_prefixes=prefixes,
                normalization=normalization,
                output_dir=run_dir,
            )
            persist_output(
                run_dir,
                (
                    config.persistent_output_dir
                    / mode.lower().replace("+", "_plus_")
                    / f"seed_{int(seed)}"
                    if config.persistent_output_dir
                    else None
                ),
                label=f"{name}-complete",
            )
    campaign_manifest = output / "campaign_manifest.json"
    campaign_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.supervised_panel.v3",
                "formal_evidence": config.diagnostic_max_samples is None,
                "design": "resnet34_unet_x_sensor_mode_x_seed",
                "sensor_modes": list(config.sensor_modes),
                "seeds": list(config.seeds),
                "config": asdict(config),
                "official_split_report": split_report,
                "official_split_lineage": {
                    key: _prefix_lineage(value)
                    for key, value in official_prefixes.items()
                },
                "execution_split_lineage": {
                    key: _prefix_lineage(value)
                    for key, value in prefixes.items()
                },
                "normalization_contracts": normalization_contracts,
                "runs": {
                    name: {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in artifacts.items()
                    }
                    for name, artifacts in results.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    persist_output(output, config.persistent_output_dir, label="sen1-supervised-panel-complete")
    return {"runs": results, "campaign_manifest": campaign_manifest}


__all__ = [
    "MODE_CHANNELS",
    "SENSOR_MODES",
    "Sen1SupervisedCampaignError",
    "Sen1SupervisedConfig",
    "compute_train_normalization",
    "_diagnostic_prefix_subset",
    "run_sen1_supervised_campaign",
]
