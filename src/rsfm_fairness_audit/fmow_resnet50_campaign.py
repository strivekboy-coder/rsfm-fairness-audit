from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.fmow_dofav2_campaign import (
    _formal_rows,
    _validate_split_contract,
)
from rsfm_fairness_audit.fmow_sentinel_classification import (
    FmowClassificationConfig,
    _FmowTorchDataset,
    _class_mapping,
    _compute_or_load_norm_stats,
    _evaluate_resnet50,
    _limit_rows,
    _load_metadata,
    _row_hash,
    _split_rows,
    build_resnet50_multiband,
)
from rsfm_fairness_audit.formal_outputs import (
    FormalOutputBundle,
    file_sha256,
    write_multiclass_bundle,
)
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.geobwer_extensions import run_multiclass_uncertainty_suite
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output
from rsfm_fairness_audit.probe_selection import group_stratified_inner_split


class FmowResNet50CampaignError(RuntimeError):
    """Raised when the protocol-matched fMoW baseline cannot be certified."""


@dataclass(frozen=True)
class FmowResNet50CampaignConfig:
    metadata_csv: Path
    data_root: Path
    output_dir: Path
    persistent_output_dir: Path | None = None
    geobwer_protocol: Path = Path("configs/geobwer/fmow_sentinel.yaml")
    train_split: str = "train"
    calibration_split: str = "calibration"
    test_split: str = "test"
    split_protocol: str = "location_disjoint"
    band_profile: str = "sentinel2_9_legacy"
    image_size: int = 224
    seeds: tuple[int, ...] = (42, 73, 101)
    max_epochs: int = 100
    patience: int = 12
    min_delta: float = 1e-4
    inner_validation_fraction: float = 0.15
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    pretrained_encoder: bool = False
    device: str = "auto"
    amp: bool = True
    audit_bootstrap: int = 2000
    conformal_alpha: float = 0.10
    diagnostic_max_samples_per_split: int | None = None

    def __post_init__(self) -> None:
        if self.band_profile != "sentinel2_9_legacy":
            raise ValueError(
                "The protocol-matched fMoW baseline is frozen to common sentinel2_9_legacy bands."
            )
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique.")
        if self.diagnostic_max_samples_per_split is None and len(self.seeds) < 3:
            raise ValueError("Formal fMoW ResNet-50 training requires at least three seeds.")
        if min(self.max_epochs, self.patience, self.batch_size) <= 0:
            raise ValueError("max_epochs, patience, and batch_size must be positive.")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Colab/runtime path
        raise FmowResNet50CampaignError("PyTorch and torchvision are required.") from exc
    return torch


def _device(name: str) -> Any:
    torch = _require_torch()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FmowResNet50CampaignError("CUDA was requested but is unavailable.")
    return device


def _loader(
    rows: Sequence[dict[str, Any]],
    config: FmowClassificationConfig,
    class_to_index: Mapping[str, int],
    normalization: Mapping[str, Any],
    *,
    shuffle: bool,
    seed: int,
) -> Any:
    torch = _require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        _FmowTorchDataset(rows, config, class_to_index, normalization),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=bool(config.num_workers > 0),
        generator=generator,
    )


def _fit_seed(
    train_rows: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    normalization: Mapping[str, Any],
    runner_config: FmowClassificationConfig,
    campaign: FmowResNet50CampaignConfig,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    torch = _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    classes, class_to_index = _class_mapping(train_rows)
    labels = [str(row["category"]) for row in train_rows]
    groups = [str(row["site_id"]) for row in _formal_rows(train_rows)]
    fit_indices, validation_indices = group_stratified_inner_split(
        labels,
        groups,
        validation_fraction=campaign.inner_validation_fraction,
        seed=seed,
    )
    inner_fit = [dict(train_rows[index]) for index in fit_indices]
    inner_validation = [dict(train_rows[index]) for index in validation_indices]
    device = _device(campaign.device)
    model = build_resnet50_multiband(
        len(classes),
        in_channels=9,
        pretrained=campaign.pretrained_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = bool(campaign.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fit_loader = _loader(
        inner_fit,
        runner_config,
        class_to_index,
        normalization,
        shuffle=True,
        seed=seed,
    )
    inner_validation_loader = _loader(
        inner_validation,
        runner_config,
        class_to_index,
        normalization,
        shuffle=False,
        seed=seed,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, campaign.max_epochs + 1):
        model.train()
        losses: list[float] = []
        counts: list[int] = []
        for images, targets, _indices in fit_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            counts.append(int(images.shape[0]))
        inner_predictions, inner_metrics = _evaluate_resnet50(
            model,
            inner_validation_loader,
            inner_validation,
            classes,
            device,
        )
        true_index = np.asarray(
            [class_to_index[str(item["row"]["category"])] for item in inner_predictions],
            dtype=np.int64,
        )
        probabilities = np.stack([item["probability_vector"] for item in inner_predictions])
        validation_loss = float(
            -np.mean(np.log(np.clip(probabilities[np.arange(len(true_index)), true_index], 1e-12, 1.0)))
        )
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": float(np.average(losses, weights=counts)),
                "inner_validation_cross_entropy": validation_loss,
                **inner_metrics,
            }
        )
        print(
            f"[fmow:resnet50] seed={seed} epoch={epoch}/{campaign.max_epochs} "
            f"train_ce={history[-1]['train_cross_entropy']:.6f} "
            f"inner_val_ce={validation_loss:.6f} inner_val_acc={inner_metrics['accuracy']:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - campaign.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= campaign.patience:
            break
    if best_state is None:
        raise FmowResNet50CampaignError("No finite inner-validation checkpoint was selected.")

    # Refit from a fresh initialization on every outer-training row for the
    # selected number of epochs. Outer calibration/test labels remain unseen.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_resnet50_multiband(
        len(classes),
        in_channels=9,
        pretrained=campaign.pretrained_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    full_loader = _loader(
        train_rows,
        runner_config,
        class_to_index,
        normalization,
        shuffle=True,
        seed=seed + 100_003,
    )
    full_history: list[dict[str, Any]] = []
    for epoch in range(1, best_epoch + 1):
        model.train()
        losses: list[float] = []
        counts: list[int] = []
        for images, targets, _indices in full_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            counts.append(int(images.shape[0]))
        full_history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": float(np.average(losses, weights=counts)),
            }
        )
    outputs: dict[str, Any] = {}
    for split, split_rows in (
        ("calibration", calibration_rows),
        ("test", test_rows),
    ):
        predictions, metrics = _evaluate_resnet50(
            model,
            _loader(
                split_rows,
                runner_config,
                class_to_index,
                normalization,
                shuffle=False,
                seed=seed,
            ),
            split_rows,
            classes,
            device,
        )
        outputs[split] = {
            "probabilities": np.stack(
                [item["probability_vector"] for item in predictions]
            ).astype(np.float32),
            "logits": np.stack([item["logit_vector"] for item in predictions]).astype(
                np.float32
            ),
            "metrics": metrics,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "resnet50_common9.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "class_to_index": class_to_index,
            "normalization": dict(normalization),
            "seed": seed,
            "selected_epoch": best_epoch,
            "inner_validation_cross_entropy": best_loss,
            "band_profile": campaign.band_profile,
            "pretrained_encoder": campaign.pretrained_encoder,
        },
        checkpoint,
    )
    return {
        "classes": classes,
        "class_to_index": class_to_index,
        "checkpoint": checkpoint,
        "selected_epoch": best_epoch,
        "inner_validation_cross_entropy": best_loss,
        "search_history": history,
        "full_history": full_history,
        "outputs": outputs,
    }


def _audit_seed(
    config: FmowResNet50CampaignConfig,
    *,
    seed: int,
    train_rows: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    fit: Mapping[str, Any],
    protocol: BWERProtocol,
    output_dir: Path,
) -> dict[str, Path]:
    class_names = list(fit["classes"])
    class_to_index = dict(fit["class_to_index"])
    calibration_targets = np.asarray(
        [class_to_index[str(row["category"])] for row in calibration_rows],
        dtype=np.int64,
    )
    test_targets = np.asarray(
        [class_to_index[str(row["category"])] for row in test_rows], dtype=np.int64
    )
    checkpoint = Path(fit["checkpoint"])
    model_name = f"resnet50_common9_seed_{seed}"
    model_lineage = {
        "model": model_name,
        "architecture": "torchvision_resnet50",
        "input_channels": 9,
        "band_profile": config.band_profile,
        "pretrained_encoder": config.pretrained_encoder,
        "adaptation_protocol": "supervised_common9_protocol_matched",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "seed": seed,
        "selected_epoch": fit["selected_epoch"],
        "model_selection": "outer_train_only_category_site_disjoint_inner_holdout",
        "outer_calibration_or_test_labels_used_for_model_selection": False,
    }
    dataset_lineage = {
        "dataset": "fMoW-Sentinel",
        "metadata_sha256": file_sha256(config.metadata_csv),
        "split_protocol": config.split_protocol,
        "train_row_hash": _row_hash(train_rows),
        "calibration_row_hash": _row_hash(calibration_rows),
        "test_row_hash": _row_hash(test_rows),
    }
    bundle: FormalOutputBundle = write_multiclass_bundle(
        output_dir / "formal_outputs",
        sample_rows=_formal_rows(test_rows),
        probabilities=np.asarray(fit["outputs"]["test"]["probabilities"]),
        targets=test_targets,
        class_names=class_names,
        dataset="fmow_sentinel",
        model=model_name,
        split=config.test_split,
        protocol=protocol,
        model_lineage=model_lineage,
        dataset_lineage=dataset_lineage,
        independent_unit_column="sample_id",
        split_role="evaluation",
    )
    calibration_path = output_dir / "calibration_probabilities.npz"
    np.savez_compressed(
        calibration_path,
        probabilities=np.asarray(fit["outputs"]["calibration"]["probabilities"]),
        logits=np.asarray(fit["outputs"]["calibration"]["logits"]),
        targets=calibration_targets,
        class_names=np.asarray(class_names, dtype=str),
        sample_id=np.asarray([row["sample_id"] for row in calibration_rows], dtype=str),
        split_role=np.asarray("calibration"),
        test_rows_used=np.asarray(False),
    )
    calibration_manifest = output_dir / "calibration_manifest.json"
    calibration_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.multiclass_calibration.v2",
                "split_role": "calibration",
                "test_rows_used": False,
                "probabilities_sha256": file_sha256(calibration_path),
                "sample_count": len(calibration_rows),
                "class_mapping": class_names,
                "model_selection_data": "outer_train_only",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = read_csv_rows(bundle.audit_table)
    axes = tuple(
        column
        for column in ("country", "region", "class_label", "country_class", "region_class")
        if all(str(row.get(column, "")).strip() for row in rows)
    )
    raw = audit_rows(
        rows,
        group_columns=axes,
        protocol=protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="site_id",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=seed,
    ).to_report(output_dir / "geobwer_raw")
    strict = audit_rows(
        rows,
        group_columns=("country",),
        protocol=protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="site_id",
        balance_column="class_label",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=seed,
    ).to_report(output_dir / "geobwer_standardized_strict")
    partial_protocol = replace(
        protocol,
        missingness_rule="partial_bounds",
        metadata=tuple(
            sorted(
                {
                    **dict(protocol.metadata),
                    "standardization_sensitivity": "partial_identification_bounds",
                }.items()
            )
        ),
    )
    partial = audit_rows(
        rows,
        group_columns=("country",),
        protocol=partial_protocol,
        loss_column="risk",
        unit_column="independent_unit_id",
        cluster_column="site_id",
        balance_column="class_label",
        formal=True,
        require_probabilities=True,
        n_bootstrap=config.audit_bootstrap,
        seed=seed,
    ).to_report(output_dir / "geobwer_standardized_partial_bounds")
    uncertainty = run_multiclass_uncertainty_suite(
        calibration_path,
        bundle.output_dir,
        output_dir / "uncertainty_extensions",
        protocol=protocol,
        group_columns=axes,
        calibration_manifest=calibration_manifest,
        alpha=config.conformal_alpha,
        n_bootstrap=config.audit_bootstrap,
        seed=seed,
    )
    manifest = output_dir / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.resnet50_common9_seed.v1",
                "model_lineage": model_lineage,
                "dataset_lineage": dataset_lineage,
                "formal_output_manifest": str(bundle.manifest),
                "calibration_manifest": str(calibration_manifest),
                "raw_geobwer": {key: str(value) for key, value in raw.items()},
                "standardized_strict": {key: str(value) for key, value in strict.items()},
                "standardized_partial_bounds": {
                    key: str(value) for key, value in partial.items()
                },
                "uncertainty": {key: str(value) for key, value in uncertainty.items()},
                "training": {
                    "selected_epoch": fit["selected_epoch"],
                    "inner_validation_cross_entropy": fit[
                        "inner_validation_cross_entropy"
                    ],
                    "search_history": fit["search_history"],
                    "full_history": fit["full_history"],
                    "calibration_metrics": fit["outputs"]["calibration"]["metrics"],
                    "test_metrics": fit["outputs"]["test"]["metrics"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "formal_audit_table": bundle.audit_table,
        "formal_manifest": bundle.manifest,
        "calibration": calibration_path,
        "raw_summary": raw["summary"],
        "strict_standardized_summary": strict["summary"],
        "partial_standardized_summary": partial["summary"],
        "uncertainty_summary": uncertainty["summary"],
        "run_manifest": manifest,
    }


def run_fmow_resnet50_campaign(
    config: FmowResNet50CampaignConfig,
) -> dict[str, Any]:
    hydrate_output(config.output_dir, config.persistent_output_dir)
    output = ensure_dir(config.output_dir)
    rows = _load_metadata(config.metadata_csv)
    limit = config.diagnostic_max_samples_per_split
    train_rows = _limit_rows(_split_rows(rows, config.train_split), limit, 42)
    calibration_rows = _limit_rows(
        _split_rows(rows, config.calibration_split), limit, 43
    )
    test_rows = _limit_rows(_split_rows(rows, config.test_split), limit, 44)
    _validate_split_contract(train_rows, calibration_rows, test_rows)
    for split_rows in (train_rows, calibration_rows, test_rows):
        for row in split_rows:
            row.update(_formal_rows([row])[0])
    runner_config = FmowClassificationConfig(
        metadata_csv=config.metadata_csv,
        data_root=config.data_root,
        output_dir=output,
        model="resnet50",
        train_split=config.train_split,
        eval_split=config.calibration_split,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        seed=42,
        split_protocol=config.split_protocol,
        band_profile=config.band_profile,
        amp=config.amp,
    )
    normalization_path = output / "norm_stats_common9.json"
    if normalization_path.exists():
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        train_ok = list(train_rows)
        warnings: list[str] = []
    else:
        normalization, train_ok, warnings, generated = _compute_or_load_norm_stats(
            train_rows, runner_config, output
        )
        generated.replace(normalization_path)
    if warnings or len(train_ok) != len(train_rows):
        raise FmowResNet50CampaignError(
            "Formal common-9 baseline refuses unreadable-row dropping; repair the raster cache."
        )
    protocol = BWERProtocol.from_mapping(load_yaml(config.geobwer_protocol))
    runs: dict[str, Any] = {}
    seed_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        run_dir = output / f"seed_{int(seed)}"
        fit = _fit_seed(
            train_rows,
            calibration_rows,
            test_rows,
            normalization=normalization,
            runner_config=replace(runner_config, seed=int(seed)),
            campaign=config,
            seed=int(seed),
            output_dir=run_dir,
        )
        if config.diagnostic_max_samples_per_split is not None:
            diagnostic = run_dir / "diagnostic_manifest.json"
            diagnostic.write_text(
                json.dumps(
                    {
                        "schema": "geobwer.fmow.resnet50_common9_diagnostic.v1",
                        "formal_evidence": False,
                        "seed": seed,
                        "calibration_metrics": fit["outputs"]["calibration"]["metrics"],
                        "test_metrics": fit["outputs"]["test"]["metrics"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            runs[str(seed)] = {"diagnostic_manifest": diagnostic}
        else:
            runs[str(seed)] = _audit_seed(
                config,
                seed=int(seed),
                train_rows=train_rows,
                calibration_rows=calibration_rows,
                test_rows=test_rows,
                fit=fit,
                protocol=protocol,
                output_dir=run_dir,
            )
        seed_rows.append(
            {
                "seed": seed,
                "selected_epoch": fit["selected_epoch"],
                "inner_validation_cross_entropy": fit[
                    "inner_validation_cross_entropy"
                ],
                **{
                    f"calibration_{key}": value
                    for key, value in fit["outputs"]["calibration"]["metrics"].items()
                },
                **{
                    f"test_{key}": value
                    for key, value in fit["outputs"]["test"]["metrics"].items()
                },
            }
        )
        persist_output(
            run_dir,
            (
                config.persistent_output_dir / f"seed_{int(seed)}"
                if config.persistent_output_dir
                else None
            ),
            label=f"fmow-resnet50-seed-{seed}-complete",
        )
    seed_summary = output / "seed_robustness.csv"
    write_csv(seed_summary, seed_rows)
    campaign_manifest = output / "campaign_manifest.json"
    campaign_manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.resnet50_common9_panel.v1",
                "formal_evidence": config.diagnostic_max_samples_per_split is None,
                "role": "protocol_matched_supervised_baseline",
                "config": asdict(config),
                "normalization": str(normalization_path),
                "normalization_sha256": file_sha256(normalization_path),
                "runs": {
                    seed: {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in artifacts.items()
                    }
                    for seed, artifacts in runs.items()
                },
                "seed_robustness": str(seed_summary),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    persist_output(output, config.persistent_output_dir, label="fmow-resnet50-panel-complete")
    return {
        "runs": runs,
        "seed_robustness": seed_summary,
        "campaign_manifest": campaign_manifest,
    }


__all__ = [
    "FmowResNet50CampaignConfig",
    "FmowResNet50CampaignError",
    "run_fmow_resnet50_campaign",
]
