from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_table
from rsfm_fairness_audit.audit_table import write_audit_table
from rsfm_fairness_audit.bwer_v2 import run_bwer_v2_posthoc
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.segmentation import (
    aggregate_segmentation_metrics,
    build_audit_table_from_segmentation_metrics_from_rows,
    plot_segmentation_iou_by_group,
    segmentation_confusion_counts,
    segmentation_metrics_from_counts,
)
from rsfm_fairness_audit.slice_support import evaluate_slice_support


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("The U-Net baseline requires PyTorch. Install torch in the Colab/runtime environment.") from exc
    return torch


@dataclass(frozen=True)
class UnetConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    base_channels: int = 16
    architecture: str = "vanilla_unet"
    pretrained_encoder: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    split_protocol: str = "random_chip_split"
    val_fraction: float = 0.15
    test_fraction: float = 0.20
    held_out_events: tuple[str, ...] = ()
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    max_samples: int | None = None
    eval_split: str = "test"
    run_bwer_v2: bool = False
    debug_samples: int = 0


def _model_variant(config: UnetConfig) -> str:
    if config.architecture == "s2_resnet34_unet":
        return "s2_resnet34_unet"
    return "unet_sen1floods11_s2_512"


def _display_name(config: UnetConfig) -> str:
    if config.architecture == "s2_resnet34_unet":
        return "S2 ResNet34-U-Net / AlbuNet-style baseline"
    return "Vanilla U-Net Sen1Floods11 S2 512 baseline"


def _read_metadata(data_root: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = read_csv_rows(data_root / "metadata.csv")
    output = []
    for index, row in enumerate(rows[: max_samples or None]):
        item = dict(row)
        event_id = str(item.get("event_id") or item.get("event") or item.get("region") or "to_verify")
        item["event_id"] = event_id
        item["event"] = str(item.get("event") or event_id)
        item["region"] = str(item.get("region") or event_id)
        item["sample_id"] = str(item.get("sample_id") or f"sen1floods11-{index:06d}")
        output.append(item)
    if not output:
        raise ValueError(f"No Sen1Floods11 metadata rows found under {data_root}")
    return output


def _load_npz_array(data_root: Path, rel_path: str, key: str) -> np.ndarray:
    path = Path(rel_path)
    if not path.is_absolute():
        path = data_root / path
    data = np.load(path)
    if key in data:
        return data[key]
    return data[data.files[0]]


def _prepare_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected image [T,C,H,W] or [C,H,W], got {arr.shape}")
    if arr.shape[0] != 6:
        raise ValueError(f"Expected 6 Sentinel-2 bands for U-Net baseline, got image shape {arr.shape}")
    if float(np.nanmax(arr)) > 2.0:
        arr = arr / 10000.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _inspect_prepared_data(data_root: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    image = _prepare_image(_load_npz_array(data_root, str(first["chip_path"]), "image"))
    mask = _load_npz_array(data_root, str(first["mask_path"]), "mask")
    warnings: list[str] = []
    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError(f"First U-Net chip/mask shape mismatch: image={image.shape}, mask={mask.shape}")
    if image.shape[-1] != image.shape[-2]:
        warnings.append(f"U-Net baseline expected square chips; first chip shape is {image.shape[-2:]}.")
    band_profile = str(first.get("band_profile", ""))
    if band_profile and band_profile != "prithvi_tl_sen1floods11":
        warnings.append(
            f"Prepared data band_profile is {band_profile!r}, not 'prithvi_tl_sen1floods11'. "
            "The run is allowed because the chip has 6 bands, but protocol comparisons should check band order."
        )
    target_size = str(first.get("target_size", "") or "")
    if target_size and target_size != str(image.shape[-1]):
        warnings.append(f"metadata target_size={target_size} but first chip resolution={image.shape[-1]}.")
    if image.shape[-1] != 512:
        warnings.append(f"U-Net run resolution is {image.shape[-1]}, not the paper-grade 512 setting.")
    return {
        "input_channels": int(image.shape[0]),
        "height": int(image.shape[-2]),
        "width": int(image.shape[-1]),
        "resolution": int(image.shape[-1]) if image.shape[-1] == image.shape[-2] else f"{image.shape[-2]}x{image.shape[-1]}",
        "data_validation_warnings": warnings,
    }


class Sen1Floods11TorchDataset:
    def __init__(self, data_root: str | Path, rows: Sequence[dict[str, Any]]) -> None:
        self.data_root = Path(data_root)
        self.rows = [dict(row) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        row = self.rows[index]
        image = _prepare_image(_load_npz_array(self.data_root, str(row["chip_path"]), "image"))
        mask = _load_npz_array(self.data_root, str(row["mask_path"]), "mask").astype(np.int64)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "metadata": row,
        }


def _collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    torch = _require_torch()
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }


def split_metadata(rows: Sequence[dict[str, Any]], config: UnetConfig) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(config.seed)
    rows = [dict(row) for row in rows]
    if config.split_protocol == "event_held_out":
        events = sorted({str(row["event_id"]) for row in rows})
        held_out = set(config.held_out_events or tuple(events[-2:]))
        train_val = [row for row in rows if str(row["event_id"]) not in held_out]
        test = [row for row in rows if str(row["event_id"]) in held_out]
        if not test:
            raise ValueError(f"event_held_out selected no test chips. Available events: {events}; requested: {sorted(held_out)}")
        if not train_val:
            raise ValueError("event_held_out selected all events for test; at least one training event is required.")
        rng.shuffle(train_val)
        val_n = max(1, int(round(len(train_val) * config.val_fraction))) if len(train_val) > 1 else 0
        return {"train": train_val[val_n:], "val": train_val[:val_n], "test": test}
    if config.split_protocol != "random_chip_split":
        raise ValueError("split_protocol must be random_chip_split or event_held_out")
    shuffled = list(rows)
    rng.shuffle(shuffled)
    test_n = max(1, int(round(len(shuffled) * config.test_fraction))) if len(shuffled) > 2 else 1
    val_n = max(1, int(round(len(shuffled) * config.val_fraction))) if len(shuffled) - test_n > 2 else 1
    test = shuffled[:test_n]
    val = shuffled[test_n : test_n + val_n]
    train = shuffled[test_n + val_n :]
    if not train:
        train, val = val, train
    return {"train": train, "val": val, "test": test}


def _event_distribution(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        event = str(row.get("event_id", "to_verify"))
        counts[event] = counts.get(event, 0) + 1
    return dict(sorted(counts.items()))


def _conv_block(in_channels: int, out_channels: int) -> Any:
    torch = _require_torch()
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetSmall:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = _require_torch()
        nn = torch.nn

        class _UNetSmall(nn.Module):
            def __init__(self, in_channels: int = 6, base_channels: int = 16) -> None:
                super().__init__()
                self.enc1 = _conv_block(in_channels, base_channels)
                self.enc2 = _conv_block(base_channels, base_channels * 2)
                self.enc3 = _conv_block(base_channels * 2, base_channels * 4)
                self.pool = nn.MaxPool2d(2)
                self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
                self.dec2 = _conv_block(base_channels * 4, base_channels * 2)
                self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
                self.dec1 = _conv_block(base_channels * 2, base_channels)
                self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

            def forward(self, x: Any) -> Any:
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                d2 = self.up2(e3)
                if d2.shape[-2:] != e2.shape[-2:]:
                    d2 = torch.nn.functional.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
                d2 = self.dec2(torch.cat([d2, e2], dim=1))
                d1 = self.up1(d2)
                if d1.shape[-2:] != e1.shape[-2:]:
                    d1 = torch.nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
                d1 = self.dec1(torch.cat([d1, e1], dim=1))
                return self.head(d1)[:, 0]

        return _UNetSmall(*args, **kwargs)


def _adapt_resnet_conv1(conv1: Any, in_channels: int, pretrained: bool) -> Any:
    torch = _require_torch()
    nn = torch.nn
    new_conv = nn.Conv2d(
        in_channels,
        conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            old_weight = conv1.weight.data
            new_conv.weight[:, :3] = old_weight
            mean_weight = old_weight.mean(dim=1, keepdim=True)
            for channel in range(3, in_channels):
                new_conv.weight[:, channel : channel + 1] = mean_weight
            if conv1.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv1.bias.data)
    return new_conv


class S2ResNet34UNet:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = _require_torch()
        nn = torch.nn
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("S2 ResNet34-U-Net requires torchvision in the training runtime.") from exc

        class _UpBlock(nn.Module):
            def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
                super().__init__()
                self.conv = _conv_block(in_channels + skip_channels, out_channels)

            def forward(self, x: Any, skip: Any) -> Any:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                return self.conv(torch.cat([x, skip], dim=1))

        class _S2ResNet34UNet(nn.Module):
            def __init__(self, in_channels: int = 6, pretrained_encoder: bool = False) -> None:
                super().__init__()
                weights = ResNet34_Weights.DEFAULT if pretrained_encoder else None
                encoder = resnet34(weights=weights)
                encoder.conv1 = _adapt_resnet_conv1(encoder.conv1, in_channels, pretrained_encoder)
                self.encoder_pretrained = pretrained_encoder
                self.input_channels = in_channels
                self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
                self.pool = encoder.maxpool
                self.layer1 = encoder.layer1
                self.layer2 = encoder.layer2
                self.layer3 = encoder.layer3
                self.layer4 = encoder.layer4
                self.up3 = _UpBlock(512, 256, 256)
                self.up2 = _UpBlock(256, 128, 128)
                self.up1 = _UpBlock(128, 64, 64)
                self.up0 = _UpBlock(64, 64, 32)
                self.head = nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, kernel_size=1),
                )

            def forward(self, x: Any) -> Any:
                input_size = x.shape[-2:]
                s0 = self.stem(x)
                s1 = self.layer1(self.pool(s0))
                s2 = self.layer2(s1)
                s3 = self.layer3(s2)
                x4 = self.layer4(s3)
                x = self.up3(x4, s3)
                x = self.up2(x, s2)
                x = self.up1(x, s1)
                x = self.up0(x, s0)
                x = torch.nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
                return self.head(x)[:, 0]

        return _S2ResNet34UNet(*args, **kwargs)


def build_unet_model(config: UnetConfig) -> Any:
    if config.architecture == "s2_resnet34_unet":
        return S2ResNet34UNet(in_channels=6, pretrained_encoder=config.pretrained_encoder)
    if config.architecture == "vanilla_unet":
        return UNetSmall(in_channels=6, base_channels=config.base_channels)
    raise ValueError("architecture must be vanilla_unet or s2_resnet34_unet")


def masked_bce_dice_loss(logits: Any, masks: Any) -> Any:
    torch = _require_torch()
    valid = masks >= 0
    if int(valid.sum().item()) == 0:
        return logits.sum() * 0.0
    targets = (masks == 1).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], targets[valid])
    probs = torch.sigmoid(logits)
    intersection = (probs[valid] * targets[valid]).sum()
    denom = probs[valid].sum() + targets[valid].sum()
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
    return bce + dice_loss


def _device(name: str) -> Any:
    torch = _require_torch()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loader(rows: Sequence[dict[str, Any]], data_root: Path, batch_size: int, shuffle: bool) -> Any:
    torch = _require_torch()
    return torch.utils.data.DataLoader(
        Sen1Floods11TorchDataset(data_root, rows),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate,
    )


def _evaluate_loader(model: Any, loader: Any, device: Any, run_metadata: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = _require_torch()
    model.eval()
    total = {"valid_pixel_count": 0, "positive_pixel_count": 0, "predicted_positive_pixel_count": 0, "TP": 0, "FP": 0, "FN": 0, "TN": 0}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(torch.int16).cpu().numpy()
            mask_np = masks.cpu().numpy().astype(np.int16)
            prob_np = probs.cpu().numpy().astype(np.float32)
            for index, meta in enumerate(batch["metadata"]):
                run_meta = run_metadata or {}
                model_variant = str(run_meta.get("model_variant", "unet_sen1floods11_s2_512"))
                counts = segmentation_confusion_counts(mask_np[index], preds[index])
                for key in total:
                    total[key] += int(counts[key])
                metrics = segmentation_metrics_from_counts(counts)
                row = dict(meta)
                row.update(
                    {
                        "dataset": "sen1floods11",
                        "model": model_variant,
                        "model_family": "unet",
                        "model_variant": model_variant,
                        "display_name": run_meta.get("display_name", "Vanilla U-Net Sen1Floods11 S2 512 baseline"),
                        "task": "segmentation",
                        "input_mode": run_meta.get("input_mode", "s2_6band_image_only"),
                        "adaptation_protocol": "supervised_baseline",
                        "checkpoint_source": "trained_in_run",
                        "aggregation_level": "chip",
                        "unit_id": row.get("sample_id"),
                        "class_label": "water",
                        "TP_plus_FN_support": counts["TP"] + counts["FN"],
                        "mean_confidence": float(np.nanmean(np.maximum(prob_np[index], 1.0 - prob_np[index])[mask_np[index] >= 0])) if np.any(mask_np[index] >= 0) else float("nan"),
                        "confidence_source": "sigmoid_probability",
                        "input_band_order": str(row.get("band_names", "")),
                        "label_mapping": "0=background;1=water_flood;-1=ignore",
                        **counts,
                        **metrics,
                        "score": metrics["micro_iou"],
                    }
                )
                rows.append(row)
    return segmentation_metrics_from_counts(total), rows


def _write_unet_report(path: Path, metadata: dict[str, Any], event_rows: Sequence[dict[str, Any]], bwer_summary_path: Path | None) -> None:
    lines = [
        "# U-Net Sen1Floods11 Native Segmentation Baseline",
        "",
        "This run is Protocol C: a fully supervised classical U-Net baseline. It is not a foundation model and should not be compared to Prithvi without noting adaptation-protocol differences.",
        "",
        f"- model_variant: {metadata['model_variant']}",
        f"- adaptation_protocol: {metadata['adaptation_protocol']}",
        f"- split_protocol: {metadata['split_protocol']}",
        f"- training_budget: {metadata['training_budget']}",
        f"- band_profile: {metadata['band_profile']}",
        f"- resolution: {metadata['resolution']}",
        "",
        "Event-level BWER is interpreted as deployment slice risk, not causal country fairness. If this run uses `random_chip_split`, it is not event-held-out generalization.",
        "",
        "## Event Metrics",
        "",
        "| event_id | chips | valid pixels | positive pixels | TP | FP | FN | micro IoU | micro Dice | risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in event_rows:
        lines.append(
            f"| {row['event_id']} | {row['sample_count']} | {row['valid_pixel_count']} | {row['positive_pixel_count']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['micro_iou']:.4f} | {row['micro_dice']:.4f} | {row['risk']:.4f} |"
        )
    if bwer_summary_path and bwer_summary_path.exists():
        lines.extend(["", "BWER outputs are written in `bwer_summary.csv` and optional post-hoc BWER v2 outputs under `bwer_v2/`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_unet_protocol_note(path: Path, metadata: dict[str, Any]) -> None:
    if not path.exists():
        return
    note = [
        "",
        "## U-Net Protocol Note",
        "",
        "This run is Protocol C: a fully supervised classical U-Net baseline, not a foundation model.",
        f"- model_family: {metadata['model_family']}",
        f"- model_variant: {metadata['model_variant']}",
        f"- adaptation_protocol: {metadata['adaptation_protocol']}",
        f"- split_protocol: {metadata['split_protocol']}",
        f"- training_budget: {metadata['training_budget']}",
        "- label_mapping: 0=background;1=water_flood;-1=ignore",
        "",
        "If `split_protocol=random_chip_split`, this output is a supervised baseline audit and should not be interpreted as event-held-out generalization. Event-level BWER is deployment slice risk, not causal country fairness.",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(note) + "\n")


def _read_warnings(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [str(value) for value in data.get("warnings", [])]


def _write_combined_warnings(path: Path, extra_warnings: Sequence[str]) -> None:
    current = _read_warnings(path)
    merged: list[str] = []
    for warning in list(extra_warnings) + current:
        if warning not in merged:
            merged.append(warning)
    path.write_text(json.dumps({"warnings": merged}, indent=2), encoding="utf-8")


def _metadata_for_run(rows: Sequence[dict[str, Any]], config: UnetConfig, splits: dict[str, list[dict[str, Any]]], device: Any, best_epoch: int, best_val_iou: float, data_info: Mapping[str, Any]) -> dict[str, Any]:
    first = rows[0]
    model_variant = _model_variant(config)
    return {
        "model_family": "unet",
        "model_variant": model_variant,
        "display_name": _display_name(config),
        "adaptation_protocol": "supervised_baseline",
        "training_budget": f"epochs={config.epochs};batch_size={config.batch_size};learning_rate={config.learning_rate};optimizer=adamw;loss=bce_plus_dice;image_size={data_info['resolution']};architecture={config.architecture};pretrained_encoder={config.pretrained_encoder};early_stopping_patience={config.early_stopping_patience};seed={config.seed}",
        "split_protocol": config.split_protocol,
        "eval_split": config.eval_split,
        "resolution": data_info["resolution"],
        "input_channels": data_info["input_channels"],
        "input_mode": "s2_6band_image_only",
        "architecture": config.architecture,
        "pretrained_encoder": config.pretrained_encoder,
        "first_conv_adaptation": "3_channel_resnet34_weights_repeated_mean_to_channels_4_to_6" if config.pretrained_encoder and config.architecture == "s2_resnet34_unet" else "not_applicable",
        "band_profile": first.get("band_profile", ""),
        "band_indices": first.get("band_indices", ""),
        "band_names": first.get("band_names", ""),
        "label_mapping": "0=background;1=water_flood;-1=ignore",
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss_plus_soft_dice_ignore_-1",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "base_channels": config.base_channels,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "seed": config.seed,
        "device": str(device),
        "train_chip_count": len(splits["train"]),
        "val_chip_count": len(splits["val"]),
        "test_chip_count": len(splits["test"]),
        "train_event_distribution": _event_distribution(splits["train"]),
        "val_event_distribution": _event_distribution(splits["val"]),
        "test_event_distribution": _event_distribution(splits["test"]),
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "data_validation_warnings": list(data_info.get("data_validation_warnings", [])),
    }


def run_unet_sen1floods11(config: UnetConfig) -> dict[str, Path]:
    torch = _require_torch()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    output = ensure_dir(config.output_dir)
    figures = ensure_dir(output / "figures")
    tables = ensure_dir(output / "tables")
    print(f"[stage] Loading Sen1Floods11 U-Net metadata from {config.data_root}", flush=True)
    rows = _read_metadata(config.data_root, config.max_samples)
    data_info = _inspect_prepared_data(config.data_root, rows)
    print(
        f"[stage] U-Net prepared data: chips={len(rows)} resolution={data_info['resolution']} channels={data_info['input_channels']}",
        flush=True,
    )
    for warning in data_info.get("data_validation_warnings", []):
        print(f"[warn] {warning}", flush=True)
    splits = split_metadata(rows, config)
    print(
        f"[stage] Split protocol={config.split_protocol}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
        flush=True,
    )
    device = _device(config.device)
    print(
        f"[stage] Training U-Net on device={device} epochs={config.epochs} batch_size={config.batch_size} architecture={config.architecture}",
        flush=True,
    )
    model = build_unet_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.amp and device.type == "cuda"))
    train_loader = _loader(splits["train"], config.data_root, config.batch_size, shuffle=True)
    val_loader = _loader(splits["val"] or splits["train"], config.data_root, config.batch_size, shuffle=False)
    history: list[dict[str, Any]] = []
    best_state = None
    best_val_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, config.early_stopping_patience // 3) if config.early_stopping_patience else 5,
    )
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(config.amp and device.type == "cuda")):
                logits = model(images)
                loss = masked_bce_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _ = _evaluate_loader(model, val_loader, device)
        val_iou = float(val_metrics["micro_iou"])
        scheduler.step(val_iou)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"), "val_micro_iou": val_iou, "val_micro_dice": val_metrics["micro_dice"], "learning_rate": current_lr})
        print(
            f"[epoch {epoch}/{config.epochs}] train_loss={history[-1]['train_loss']:.4f} val_micro_iou={val_iou:.4f} val_micro_dice={val_metrics['micro_dice']:.4f} lr={current_lr:.6g}",
            flush=True,
        )
        improved = not math.isnan(val_iou) and val_iou > (best_val_iou + config.early_stopping_min_delta)
        if improved:
            best_val_iou = val_iou
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if config.early_stopping_patience and epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"[stage] Early stopping at epoch {epoch}; best_epoch={best_epoch} best_val_iou={best_val_iou:.4f}",
                flush=True,
            )
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    eval_rows_source = rows if config.eval_split == "all" else splits[config.eval_split]
    if not eval_rows_source:
        raise ValueError(f"Evaluation split {config.eval_split!r} is empty; adjust split settings.")
    print(f"[stage] Evaluating U-Net on split={config.eval_split} chips={len(eval_rows_source)}", flush=True)
    eval_loader = _loader(eval_rows_source, config.data_root, config.batch_size, shuffle=False)
    run_metadata = _metadata_for_run(rows, config, splits, device, best_epoch, best_val_iou, data_info)
    _, metric_rows = _evaluate_loader(model, eval_loader, device, run_metadata=run_metadata)
    for row in metric_rows:
        row["split"] = config.eval_split
        row["split_protocol"] = config.split_protocol
        row["training_budget"] = run_metadata["training_budget"]
    event_rows = aggregate_segmentation_metrics(metric_rows, "event_id", aggregation_level="event")
    audit_rows = build_audit_table_from_segmentation_metrics_from_rows(event_rows)
    region_rows = [
        {
            "slice_name": "event_id",
            "group": row["event_id"],
            "n": row["sample_count"],
            "valid_pixel_support": row["valid_pixel_count"],
            "positive_pixel_support": row["positive_pixel_count"],
            "mean_water_iou": row["micro_iou"],
            "mean_pixel_accuracy": row["pixel_accuracy"],
        }
        for row in event_rows
    ]
    artifacts = {
        "segmentation_metrics": output / "segmentation_metrics.csv",
        "segmentation_predictions": output / "segmentation_predictions.csv",
        "event_segmentation_metrics": output / "event_segmentation_metrics.csv",
        "segmentation_audit_table": output / "segmentation_audit_table.csv",
        "audit_table": output / "audit_table.csv",
        "training_history": output / "training_history.csv",
        "run_metadata": output / "run_metadata.json",
        "model_debug": output / "model_debug.json",
        "checkpoint": output / "checkpoints" / f"best_{run_metadata['model_variant']}.pt",
        "segmentation_fairness_matrix_event": output / "segmentation_fairness_matrix_event.csv",
        "tables_segmentation_metrics": tables / "segmentation_metrics.csv",
        "tables_event": tables / "segmentation_fairness_matrix_event.csv",
        "iou_by_group": figures / "segmentation_iou_by_group.png",
        "segmentation_report": output / "segmentation_report.md",
        "report": output / "report.md",
    }
    artifacts["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": run_metadata}, artifacts["checkpoint"])
    write_csv(artifacts["segmentation_metrics"], metric_rows)
    write_csv(artifacts["segmentation_predictions"], metric_rows)
    write_csv(artifacts["event_segmentation_metrics"], event_rows)
    write_audit_table(artifacts["segmentation_audit_table"], audit_rows)
    write_audit_table(artifacts["audit_table"], audit_rows)
    write_csv(artifacts["training_history"], history)
    write_csv(artifacts["segmentation_fairness_matrix_event"], region_rows)
    write_csv(artifacts["tables_segmentation_metrics"], metric_rows)
    write_csv(artifacts["tables_event"], region_rows)
    plot_segmentation_iou_by_group(region_rows, artifacts["iou_by_group"])
    artifacts["run_metadata"].write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["model_debug"].write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    preflight = evaluate_slice_support(
        audit_rows,
        dataset="sen1floods11",
        model=run_metadata["model_variant"],
        task="segmentation",
        output_dir=output,
        candidates=["event_id", "event_id|event", "country|country"],
        score_column="micro_iou",
        risk_column="risk",
    )
    artifacts.update({f"preflight_{key}": value for key, value in preflight.items()})
    preflight_warnings = list(data_info.get("data_validation_warnings", [])) + _read_warnings(output / "warnings.json")
    try:
        bwer = evaluate_bwer_table(
            audit_rows,
            dataset="sen1floods11",
            model=run_metadata["model_variant"],
            task="segmentation",
            output_dir=output,
            slice_variable="event_id",
            balance_variable="raw",
            score_column="micro_iou",
            risk_column="risk",
            audit_level="pilot",
        )
        artifacts.update(bwer)
        _write_combined_warnings(output / "warnings.json", preflight_warnings)
    except ValueError as exc:
        (output / "bwer_not_runnable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        artifacts["bwer_not_runnable"] = output / "bwer_not_runnable.txt"
    _write_unet_report(artifacts["segmentation_report"], run_metadata, event_rows, artifacts.get("bwer_summary"))
    if "bwer_not_runnable" in artifacts:
        _write_unet_report(artifacts["report"], run_metadata, event_rows, None)
    else:
        _append_unet_protocol_note(artifacts["report"], run_metadata)
    if config.run_bwer_v2:
        artifacts.update({f"bwer_v2_{key}": value for key, value in run_bwer_v2_posthoc(output, output / "bwer_v2").items()})
    return artifacts
