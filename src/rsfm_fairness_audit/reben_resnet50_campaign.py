from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.reben import (
    LmdbSafetensorsRebenDatasetAdapter,
    detect_lmdb_payload_format,
    reben_labels_to_multihot,
    resolve_reben_root_dir,
)
from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.fmow_sentinel_classification import build_resnet50_multiband
from rsfm_fairness_audit.formal_outputs import (
    FormalOutputBundle,
    file_sha256,
    write_multilabel_bundle,
)
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.geobwer_extensions import run_multilabel_uncertainty_suite
from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output
from rsfm_fairness_audit.probe_selection import group_disjoint_inner_split
from rsfm_fairness_audit.reben_sensor_audit import (
    compute_multilabel_metrics,
    default_reben_class_names,
    select_thresholds_from_validation,
)
from rsfm_fairness_audit.reben_terramind_campaign import build_reben_dataset_lineage
from rsfm_fairness_audit.spatial_conformal import SpatialConformalConfig


class RebenResNet50CampaignError(RuntimeError):
    """Raised when the supervised reBEN reference violates the frozen contract."""


class RebenDiagnosticSupportError(RebenResNet50CampaignError):
    """Raised when a bounded diagnostic cannot preserve its selection geometry."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


SENSOR_MODES = ("S1", "S2", "S1+S2")
MODE_CHANNELS = {"S1": 2, "S2": 12, "S1+S2": 14}


@dataclass(frozen=True)
class RebenResNet50Config:
    lmdb_root: Path
    metadata_parquet: Path
    output_dir: Path
    persistent_output_dir: Path | None = None
    metadata_snow_cloud_parquet: Path | None = None
    geobwer_protocol: Path = Path("configs/geobwer/reben.yaml")
    sensor_modes: tuple[str, ...] = SENSOR_MODES
    seeds: tuple[int, ...] = (42, 73, 101)
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-4
    batch_size: int = 128
    num_workers: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    pretrained_encoder: bool = False
    normalization_pixel_stride: int = 8
    device: str = "auto"
    amp: bool = True
    audit_bootstrap: int = 2000
    crc_alpha: float = 0.10
    diagnostic_max_samples: int | None = None

    def __post_init__(self) -> None:
        if not self.sensor_modes or any(mode not in SENSOR_MODES for mode in self.sensor_modes):
            raise ValueError(f"sensor_modes must be selected from {SENSOR_MODES}.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique.")
        if self.diagnostic_max_samples is None and len(self.seeds) < 3:
            raise ValueError("Formal reBEN supervised training requires at least three seeds.")
        if min(
            self.max_epochs,
            self.patience,
            self.batch_size,
            self.normalization_pixel_stride,
        ) <= 0:
            raise ValueError("Epoch, batch, patience, and normalization stride must be positive.")


def _stable_rank(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _row_label_vector(row: Mapping[str, Any]) -> np.ndarray:
    if "label_vector" in row:
        values = np.asarray(row["label_vector"], dtype=np.int8)
    else:
        values = reben_labels_to_multihot(row.get("labels", []))
    if values.shape != (19,):
        raise RebenResNet50CampaignError(
            f"Expected a 19-label diagnostic metadata vector, got {values.shape}."
        )
    return values.astype(np.int8, copy=False)


def select_reben_diagnostic_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_samples: int,
    seed: int,
    split: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a deterministic bounded panel with group and label coverage.

    Selection is metadata-only and therefore identical for S1, S2 and S1+S2.
    The first two rows are forced onto distinct source tiles when possible;
    subsequent rows greedily add unseen labels and then unseen tiles.
    """

    limit = min(int(max_samples), len(rows))
    if limit <= 0:
        raise ValueError("diagnostic max_samples must be positive.")
    groups = [str(row.get("source_tile_id", "")).strip() for row in rows]
    if any(not group for group in groups):
        raise RebenResNet50CampaignError(
            "Every reBEN diagnostic row requires a non-empty source_tile_id."
        )
    labels = np.stack([_row_label_vector(row) for row in rows], axis=0)
    all_groups = sorted(set(groups))
    selected: list[int] = []
    selected_groups: set[str] = set()
    covered_labels = np.zeros(19, dtype=bool)
    remaining = set(range(len(rows)))
    while remaining and len(selected) < limit:
        pool = remaining
        if len(selected_groups) == 1:
            new_group_pool = {
                index for index in remaining if groups[index] not in selected_groups
            }
            if new_group_pool:
                pool = new_group_pool

        def ordering(index: int) -> tuple[int, int, int, int]:
            vector = labels[index].astype(bool)
            new_label_count = int(np.sum(vector & ~covered_labels))
            new_group = int(groups[index] not in selected_groups)
            label_count = int(np.sum(vector))
            stable = _stable_rank(
                f"{seed}|{split}|{groups[index]}|{rows[index].get('sample_id', index)}"
            )
            return (-new_label_count, -new_group, -label_count, stable)

        chosen = min(pool, key=ordering)
        selected.append(chosen)
        selected_groups.add(groups[chosen])
        covered_labels |= labels[chosen].astype(bool)
        remaining.remove(chosen)

    selected_array = np.asarray(selected, dtype=np.int64)
    selected_ids = [str(rows[index].get("sample_id", "")).strip() for index in selected]
    digest = hashlib.sha256()
    for sample_id in selected_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    support_status = (
        "ready"
        if len(selected_groups) >= 2 and len(selected_array) >= 2
        else "insufficient_selection_groups"
    )
    diagnostics = {
        "schema": "geobwer.reben.diagnostic_sampling.v1",
        "formal_evidence": False,
        "strategy": "deterministic_group_then_multilabel_coverage",
        "split": str(split),
        "seed": int(seed),
        "requested_max_samples": int(max_samples),
        "source_samples": len(rows),
        "selected_samples": len(selected_array),
        "source_group_count": len(all_groups),
        "selected_group_count": len(selected_groups),
        "selected_groups": sorted(selected_groups),
        "covered_label_count": int(np.sum(covered_labels)),
        "covered_label_indices": np.flatnonzero(covered_labels).astype(int).tolist(),
        "selected_sample_id_sha256": digest.hexdigest(),
        "status": support_status,
    }
    return selected_array, diagnostics


class _IndexedRebenAdapter:
    def __init__(
        self,
        base: Any,
        indices: Sequence[int],
        diagnostics: Mapping[str, Any],
    ) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        source_rows = base.load_metadata()
        self._rows = [source_rows[int(index)] for index in self.indices]
        self.diagnostic_sampling = dict(diagnostics)

    def load_metadata(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]

    def load_sample(self, index: int) -> Mapping[str, Any]:
        return self.base.load_sample(int(self.indices[index]))

    def loader_info(self) -> dict[str, Any]:
        info = dict(self.base.loader_info())
        info["diagnostic_sampling"] = dict(self.diagnostic_sampling)
        return info


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Colab/runtime path
        raise RebenResNet50CampaignError("PyTorch and torchvision are required.") from exc
    return torch


def _dataset(config: RebenResNet50Config, split: str, mode: str) -> Any:
    _, lmdb_path, _ = resolve_reben_root_dir(config.lmdb_root)
    payload = detect_lmdb_payload_format(lmdb_path)
    if payload != "safetensors":
        raise RebenResNet50CampaignError(
            f"Formal supervised reBEN requires the verified raw-band safetensors LMDB; got {payload!r}."
        )
    adapter = LmdbSafetensorsRebenDatasetAdapter(
        config.lmdb_root,
        config.metadata_parquet,
        config.metadata_snow_cloud_parquet,
        split=split,
        sensor_mode=mode,
        max_samples=None,
        channel_profile="croma",
    )
    if config.diagnostic_max_samples is None:
        return adapter
    rows = adapter.load_metadata()
    indices, diagnostics = select_reben_diagnostic_indices(
        rows,
        max_samples=config.diagnostic_max_samples,
        seed=int(config.seeds[0]),
        split=split,
    )
    if split == "train" and diagnostics["status"] != "ready":
        raise RebenDiagnosticSupportError(
            "Bounded reBEN diagnostic cannot preserve at least two source_tile_id "
            "groups for group-disjoint model selection.",
            diagnostics,
        )
    return _IndexedRebenAdapter(adapter, indices, diagnostics)


def _image(sample: Mapping[str, Any], mode: str) -> np.ndarray:
    value = sample["image"]
    if isinstance(value, Mapping):
        array = np.concatenate(
            [
                np.asarray(value["S1"], dtype=np.float32),
                np.asarray(value["S2"], dtype=np.float32),
            ],
            axis=0,
        )
    else:
        array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != MODE_CHANNELS[mode]:
        raise RebenResNet50CampaignError(
            f"Expected {MODE_CHANNELS[mode]} channels for {mode}, got {array.shape}."
        )
    return array


def compute_reben_train_contract(
    dataset: Any,
    *,
    mode: str,
    pixel_stride: int,
) -> dict[str, Any]:
    """Compute deterministic train-only moments and label prevalence."""

    channels = MODE_CHANNELS[mode]
    total = np.zeros(channels, dtype=np.float64)
    square = np.zeros(channels, dtype=np.float64)
    pixel_count = 0
    positives = np.zeros(19, dtype=np.int64)
    count = len(dataset.load_metadata())
    if count <= 0:
        raise RebenResNet50CampaignError("reBEN train split is empty.")
    for index in range(count):
        sample = dataset.load_sample(index)
        image = _image(sample, mode)[:, ::pixel_stride, ::pixel_stride].astype(np.float64)
        flat = image.reshape(channels, -1)
        finite = np.all(np.isfinite(flat), axis=0)
        if not np.any(finite):
            raise RebenResNet50CampaignError(f"No finite pixels for train row {index}.")
        selected = flat[:, finite]
        total += selected.sum(axis=1)
        square += np.square(selected).sum(axis=1)
        pixel_count += selected.shape[1]
        labels = np.asarray(sample["metadata"]["label_vector"], dtype=np.int64)
        positives += labels
        if (index + 1) % 5000 == 0:
            print(
                f"[reben:resnet50:preflight] mode={mode} samples={index + 1}/{count}",
                flush=True,
            )
    mean = total / pixel_count
    variance = np.maximum(square / pixel_count - np.square(mean), 1e-12)
    negatives = count - positives
    pos_weight = negatives / np.maximum(positives, 1)
    contract = {
        "schema": "geobwer.reben.supervised_train_contract.v1",
        "selection_split": "official_train",
        "test_rows_used": False,
        "sensor_mode": mode,
        "channel_profile": "croma_raw_bands",
        "sample_count": count,
        "normalization_pixel_stride": pixel_stride,
        "pixel_count": int(pixel_count),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "positive_count": positives.tolist(),
        "negative_count": negatives.tolist(),
        "pos_weight": pos_weight.tolist(),
    }
    diagnostic_sampling = getattr(dataset, "diagnostic_sampling", None)
    if diagnostic_sampling is not None:
        contract["diagnostic_sampling"] = dict(diagnostic_sampling)
    return contract


class _TorchDataset:
    def __init__(
        self,
        adapter: Any,
        mode: str,
        contract: Mapping[str, Any],
    ) -> None:
        self.adapter = adapter
        self.mode = mode
        self.rows = adapter.load_metadata()
        self.mean = np.asarray(contract["mean"], dtype=np.float32)[:, None, None]
        self.std = np.maximum(
            np.asarray(contract["std"], dtype=np.float32)[:, None, None], 1e-6
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, int]:
        torch = _require_torch()
        sample = self.adapter.load_sample(index)
        image = (_image(sample, self.mode) - self.mean) / self.std
        labels = np.asarray(sample["metadata"]["label_vector"], dtype=np.float32)
        return (
            torch.from_numpy(image.astype(np.float32, copy=False)),
            torch.from_numpy(labels),
            index,
        )


def _loader(
    dataset: _TorchDataset,
    config: RebenResNet50Config,
    *,
    shuffle: bool,
    seed: int,
) -> Any:
    torch = _require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=bool(config.num_workers > 0),
        generator=generator,
    )


def _device(value: str) -> Any:
    torch = _require_torch()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RebenResNet50CampaignError("CUDA was requested but is unavailable.")
    return device


def _evaluate(
    model: Any,
    loader: Any,
    device: Any,
    *,
    pos_weight: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch = _require_torch()
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    counts: list[int] = []
    with torch.inference_mode():
        for images, labels, _indices in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels_device, pos_weight=pos_weight
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
            targets.append(labels.numpy().astype(np.int8))
            losses.append(float(loss.detach().cpu()))
            counts.append(int(images.shape[0]))
    return (
        np.concatenate(probabilities),
        np.concatenate(targets),
        float(np.average(losses, weights=counts)),
    )


def _train_seed(
    config: RebenResNet50Config,
    *,
    mode: str,
    seed: int,
    adapters: Mapping[str, Any],
    contract: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    torch = _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(config.device)
    datasets = {
        split: _TorchDataset(adapter, mode, contract)
        for split, adapter in adapters.items()
    }
    model = build_resnet50_multiband(
        19,
        in_channels=MODE_CHANNELS[mode],
        pretrained=config.pretrained_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pos_weight = torch.as_tensor(
        np.asarray(contract["pos_weight"], dtype=np.float32), device=device
    )
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fit_indices, selection_indices = group_disjoint_inner_split(
        [str(row["source_tile_id"]) for row in datasets["train"].rows],
        validation_fraction=0.15,
        seed=seed,
    )
    fit_groups = sorted(
        {str(datasets["train"].rows[index]["source_tile_id"]) for index in fit_indices}
    )
    selection_groups = sorted(
        {
            str(datasets["train"].rows[index]["source_tile_id"])
            for index in selection_indices
        }
    )
    diagnostic_sampling_by_split = {
        split: getattr(dataset.adapter, "diagnostic_sampling", None)
        for split, dataset in datasets.items()
    }
    inner_fit_dataset = torch.utils.data.Subset(
        datasets["train"], fit_indices.tolist()
    )
    inner_selection_dataset = torch.utils.data.Subset(
        datasets["train"], selection_indices.tolist()
    )
    train_loader = _loader(inner_fit_dataset, config, shuffle=True, seed=seed)
    selection_loader = _loader(
        inner_selection_dataset, config, shuffle=False, seed=seed
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses: list[float] = []
        counts: list[int] = []
        for images, labels, _indices in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(images), labels, pos_weight=pos_weight
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            counts.append(int(images.shape[0]))
        _, _, selection_loss = _evaluate(
            model, selection_loader, device, pos_weight=pos_weight
        )
        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": float(np.average(losses, weights=counts)),
                "inner_selection_weighted_bce": selection_loss,
            }
        )
        print(
            f"[reben:resnet50] mode={mode} seed={seed} epoch={epoch}/{config.max_epochs} "
            f"train_bce={history[-1]['train_weighted_bce']:.6f} "
            f"inner_bce={selection_loss:.6f}",
            flush=True,
        )
        if selection_loss < best_loss - config.min_delta:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= config.patience:
            break
    if best_state is None:
        raise RebenResNet50CampaignError("No finite reBEN validation checkpoint was selected.")
    # Refit on the complete official train split for the selected epoch count.
    # Official validation is reserved exclusively for threshold/conformal
    # calibration, preserving calibration-test separation.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_resnet50_multiband(
        19,
        in_channels=MODE_CHANNELS[mode],
        pretrained=config.pretrained_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    full_train_loader = _loader(
        datasets["train"], config, shuffle=True, seed=seed + 100_003
    )
    refit_history: list[dict[str, Any]] = []
    for epoch in range(1, best_epoch + 1):
        model.train()
        losses: list[float] = []
        counts: list[int] = []
        for images, labels, _indices in full_train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(images), labels, pos_weight=pos_weight
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            counts.append(int(images.shape[0]))
        refit_history.append(
            {
                "epoch": epoch,
                "full_train_weighted_bce": float(
                    np.average(losses, weights=counts)
                ),
            }
        )
    best_state = copy.deepcopy(model.state_dict())
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "resnet50.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "sensor_mode": mode,
            "input_channels": MODE_CHANNELS[mode],
            "seed": seed,
            "best_epoch": best_epoch,
            "best_inner_selection_weighted_bce": best_loss,
            "model_selection": "official_train_inner_source_tile_disjoint",
            "outer_validation_used_for_model_selection": False,
            "inner_fit_count": len(fit_indices),
            "inner_selection_count": len(selection_indices),
            "inner_fit_groups": fit_groups,
            "inner_selection_groups": selection_groups,
            "diagnostic_sampling": getattr(
                datasets["train"].adapter, "diagnostic_sampling", None
            ),
            "diagnostic_sampling_by_split": diagnostic_sampling_by_split,
            "train_contract": dict(contract),
            "config": asdict(config),
        },
        checkpoint,
    )
    outputs: dict[str, Any] = {}
    for split in ("validation", "test"):
        probabilities, targets, weighted_bce = _evaluate(
            model,
            _loader(datasets[split], config, shuffle=False, seed=seed),
            device,
            pos_weight=pos_weight,
        )
        outputs[split] = {
            "probabilities": probabilities,
            "targets": targets,
            "weighted_bce": weighted_bce,
        }
        np.savez_compressed(
            output_dir / f"{split}_probabilities.npz",
            probabilities=probabilities,
            targets=targets,
            sample_id=np.asarray(
                [row["sample_id"] for row in datasets[split].rows], dtype=str
            ),
        )
    return {
        "checkpoint": checkpoint,
        "best_epoch": best_epoch,
        "best_validation_weighted_bce": best_loss,
        "history": history,
        "refit_history": refit_history,
        "datasets": datasets,
        "outputs": outputs,
        "inner_fit_groups": fit_groups,
        "inner_selection_groups": selection_groups,
        "diagnostic_sampling": getattr(
            datasets["train"].adapter, "diagnostic_sampling", None
        ),
        "diagnostic_sampling_by_split": diagnostic_sampling_by_split,
    }


def _formalize_seed(
    config: RebenResNet50Config,
    *,
    mode: str,
    seed: int,
    fit: Mapping[str, Any],
    protocol: BWERProtocol,
    output_dir: Path,
) -> dict[str, Path]:
    validation = fit["outputs"]["validation"]
    test = fit["outputs"]["test"]
    thresholds = select_thresholds_from_validation(
        validation["targets"], validation["probabilities"]
    )
    validation_rows = fit["datasets"]["validation"].rows
    test_rows = fit["datasets"]["test"].rows
    calibration_path = output_dir / "calibration_probabilities.npz"
    np.savez_compressed(
        calibration_path,
        probabilities=validation["probabilities"],
        targets=validation["targets"],
        thresholds=thresholds.astype(np.float32),
        sample_id=np.asarray([row["sample_id"] for row in validation_rows], dtype=str),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    calibration_manifest = output_dir / "calibration_manifest.json"
    calibration_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.reben.supervised_calibration.v1",
                "split_role": "calibration",
                "split": "validation",
                "test_rows_used": False,
                "threshold_policy": "per_label_max_validation_f1",
                "probabilities_sha256": file_sha256(calibration_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model_name = f"resnet50_supervised_{mode.lower().replace('+', '_plus_')}_seed_{seed}"
    bundle: FormalOutputBundle = write_multilabel_bundle(
        output_dir / "formal_outputs",
        sample_rows=test_rows,
        probabilities=test["probabilities"],
        targets=test["targets"],
        class_names=default_reben_class_names(),
        dataset="reben",
        model=model_name,
        split="test",
        protocol=protocol,
        model_lineage={
            "model": model_name,
            "architecture": "torchvision_resnet50",
            "adaptation_protocol": "supervised_baseline",
            "sensor_mode": mode,
            "input_channels": MODE_CHANNELS[mode],
            "channel_profile": "croma_raw_bands",
            "checkpoint": str(fit["checkpoint"]),
            "checkpoint_sha256": file_sha256(fit["checkpoint"]),
            "seed": seed,
            "best_epoch": fit["best_epoch"],
            "model_selection": "official_train_inner_source_tile_disjoint",
            "outer_validation_used_for_model_selection": False,
        },
        dataset_lineage=build_reben_dataset_lineage(
            test_rows,
            test["targets"],
            metadata_parquet=config.metadata_parquet,
        ),
        threshold=thresholds,
        independent_unit_column="independent_unit_id",
        split_role="evaluation",
    )
    rows = read_csv_rows(bundle.audit_table)
    geobwer = audit_rows(
        rows,
        group_columns=("country",),
        protocol=protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="source_tile_id",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=seed,
    ).to_report(output_dir / "geobwer")
    uncertainty = run_multilabel_uncertainty_suite(
        calibration_path,
        bundle.output_dir,
        output_dir / "uncertainty_extensions",
        protocol=protocol,
        group_columns=("country",),
        calibration_manifest=calibration_manifest,
        crc_alpha=config.crc_alpha,
        n_bootstrap=config.audit_bootstrap,
        seed=seed,
        spatial_localization_config=SpatialConformalConfig(),
    )
    summary, per_class = compute_multilabel_metrics(
        test["targets"],
        test["probabilities"],
        thresholds,
        default_reben_class_names(),
    )
    write_csv(output_dir / "metrics_summary.csv", [{**summary, "sensor_mode": mode, "seed": seed}])
    write_csv(output_dir / "per_class_metrics.csv", per_class)
    manifest = output_dir / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.reben.resnet50_supervised_seed.v1",
                "sensor_mode": mode,
                "seed": seed,
                "checkpoint": str(fit["checkpoint"]),
                "formal_output_manifest": str(bundle.manifest),
                "calibration_manifest": str(calibration_manifest),
                "geobwer": {key: str(value) for key, value in geobwer.items()},
                "uncertainty": {key: str(value) for key, value in uncertainty.items()},
                "metrics": summary,
                "training_history": fit["history"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "formal_audit_table": bundle.audit_table,
        "formal_manifest": bundle.manifest,
        "geobwer_summary": geobwer["summary"],
        "uncertainty_summary": uncertainty["summary"],
        "run_manifest": manifest,
    }


def run_reben_resnet50_campaign(config: RebenResNet50Config) -> dict[str, Any]:
    hydrate_output(config.output_dir, config.persistent_output_dir)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol = BWERProtocol.from_mapping(load_yaml(config.geobwer_protocol))
    runs: dict[str, Any] = {}
    robustness_rows: list[dict[str, Any]] = []
    for mode in config.sensor_modes:
        try:
            adapters = {
                "train": _dataset(config, "train", mode),
                "validation": _dataset(config, "val", mode),
                "test": _dataset(config, "test", mode),
            }
        except RebenDiagnosticSupportError as exc:
            if config.diagnostic_max_samples is None:
                raise
            for seed in config.seeds:
                name = f"resnet50_{mode.lower().replace('+', '_plus_')}_seed_{seed}"
                run_dir = output / mode.lower().replace("+", "_plus_") / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                diagnostic = run_dir / "diagnostic_manifest.json"
                diagnostic.write_text(
                    json.dumps(
                        {
                            "schema": "geobwer.reben.resnet50_diagnostic.v2",
                            "formal_evidence": False,
                            "status": "not_run_insufficient_diagnostic_support",
                            "reason": str(exc),
                            "sensor_mode": mode,
                            "seed": int(seed),
                            "diagnostic_sampling": exc.diagnostics,
                            "formal_protocol_changed": False,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                runs[name] = {"diagnostic_manifest": diagnostic}
                robustness_rows.append(
                    {
                        "sensor_mode": mode,
                        "seed": seed,
                        "status": "not_run_insufficient_diagnostic_support",
                        "formal_evidence": False,
                    }
                )
            continue
        contract_path = output / "train_contracts" / f"{mode.lower().replace('+', '_plus_')}.json"
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        if contract_path.exists():
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            invalid_contract = (
                contract.get("selection_split") != "official_train"
                or contract.get("test_rows_used") is not False
            )
            expected_sampling = getattr(adapters["train"], "diagnostic_sampling", None)
            sampling_matches = (
                expected_sampling is None
                or (
                    contract.get("diagnostic_sampling", {}).get(
                        "selected_sample_id_sha256"
                    )
                    == expected_sampling.get("selected_sample_id_sha256")
                    and int(contract.get("sample_count", -1))
                    == len(adapters["train"].load_metadata())
                )
            )
            if invalid_contract:
                raise RebenResNet50CampaignError(
                    f"Invalid cached train contract: {contract_path}"
                )
            if not sampling_matches:
                contract = compute_reben_train_contract(
                    adapters["train"],
                    mode=mode,
                    pixel_stride=config.normalization_pixel_stride,
                )
                contract_path.write_text(
                    json.dumps(contract, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        else:
            contract = compute_reben_train_contract(
                adapters["train"],
                mode=mode,
                pixel_stride=config.normalization_pixel_stride,
            )
            contract_path.write_text(
                json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        for seed in config.seeds:
            name = f"resnet50_{mode.lower().replace('+', '_plus_')}_seed_{seed}"
            run_dir = output / mode.lower().replace("+", "_plus_") / f"seed_{seed}"
            fit = _train_seed(
                config,
                mode=mode,
                seed=int(seed),
                adapters=adapters,
                contract=contract,
                output_dir=run_dir,
            )
            if config.diagnostic_max_samples is not None:
                diagnostic = run_dir / "diagnostic_manifest.json"
                diagnostic.write_text(
                    json.dumps(
                        {
                            "schema": "geobwer.reben.resnet50_diagnostic.v2",
                            "formal_evidence": False,
                            "status": "completed",
                            "reason": "explicit_bounded_real_gpu_smoke",
                            "sensor_mode": mode,
                            "seed": seed,
                            "diagnostic_sampling": fit.get(
                                "diagnostic_sampling"
                            ),
                            "diagnostic_sampling_by_split": fit.get(
                                "diagnostic_sampling_by_split", {}
                            ),
                            "inner_fit_groups": fit.get("inner_fit_groups", []),
                            "inner_selection_groups": fit.get(
                                "inner_selection_groups", []
                            ),
                            "group_disjoint_model_selection_preserved": True,
                            "formal_protocol_changed": False,
                            "best_validation_weighted_bce": fit[
                                "best_validation_weighted_bce"
                            ],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                runs[name] = {"diagnostic_manifest": diagnostic}
            else:
                runs[name] = _formalize_seed(
                    config,
                    mode=mode,
                    seed=int(seed),
                    fit=fit,
                    protocol=protocol,
                    output_dir=run_dir,
                )
            thresholds = select_thresholds_from_validation(
                fit["outputs"]["validation"]["targets"],
                fit["outputs"]["validation"]["probabilities"],
            )
            metrics, _ = compute_multilabel_metrics(
                fit["outputs"]["test"]["targets"],
                fit["outputs"]["test"]["probabilities"],
                thresholds,
                default_reben_class_names(),
            )
            robustness_rows.append(
                {
                    "sensor_mode": mode,
                    "seed": seed,
                    "best_epoch": fit["best_epoch"],
                    "best_validation_weighted_bce": fit[
                        "best_validation_weighted_bce"
                    ],
                    "best_inner_selection_weighted_bce": fit[
                        "best_validation_weighted_bce"
                    ],
                    "outer_validation_used_for_model_selection": False,
                    **metrics,
                }
            )
            persist_output(
                run_dir,
                (
                    config.persistent_output_dir
                    / mode.lower().replace("+", "_plus_")
                    / f"seed_{seed}"
                    if config.persistent_output_dir
                    else None
                ),
                label=f"{name}-complete",
            )
    robustness = output / "seed_robustness.csv"
    write_csv(robustness, robustness_rows)
    manifest = output / "campaign_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.reben.resnet50_supervised_panel.v1",
                "formal_evidence": config.diagnostic_max_samples is None,
                "design": "supervised_resnet50_x_s1_s2_fusion_x_seed",
                "config": asdict(config),
                "runs": {
                    name: {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in artifacts.items()
                    }
                    for name, artifacts in runs.items()
                },
                "seed_robustness": str(robustness),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    persist_output(output, config.persistent_output_dir, label="reben-resnet50-panel-complete")
    return {"runs": runs, "seed_robustness": robustness, "campaign_manifest": manifest}


__all__ = [
    "MODE_CHANNELS",
    "RebenDiagnosticSupportError",
    "RebenResNet50CampaignError",
    "RebenResNet50Config",
    "compute_reben_train_contract",
    "run_reben_resnet50_campaign",
    "select_reben_diagnostic_indices",
]
