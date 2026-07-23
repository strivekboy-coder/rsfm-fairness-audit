from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from rsfm_fairness_audit.adapters.terramind import INPUT_PROFILES, S1_MEAN, S1_STD


class TerraMindSen1ConfigError(ValueError):
    """Raised when a TerraMind Sen1Floods11 campaign config is incomplete."""


def _path_text(value: str | Path) -> str:
    """Serialize Colab/Linux paths portably even when configs are built on Windows."""

    return Path(value).as_posix()


def _mode(value: str) -> str:
    mode = str(value).upper().replace(" ", "")
    if mode in {"FUSION", "S2+S1"}:
        mode = "S1+S2"
    if mode not in {"S1", "S2", "S1+S2"}:
        raise TerraMindSen1ConfigError("sensor_mode must be S1, S2, or S1+S2.")
    return mode


def build_terramind_sen1floods11_config(
    *,
    sensor_mode: str,
    s1_root: str | Path,
    s2_root: str | Path,
    label_root: str | Path,
    train_split: str | Path,
    val_split: str | Path,
    test_split: str | Path,
    run_dir: str | Path,
    backbone_checkpoint_path: str | Path,
    prediction_split: str | None = None,
    probability_output_dir: str | Path | None = None,
    persistent_checkpoint_dir: str | Path | None = None,
    checkpoint_mirror_every_n_epochs: int = 5,
    seed: int = 42,
    batch_size: int = 8,
    num_workers: int = 4,
    max_epochs: int = 100,
    fast_dev_run: bool = False,
) -> dict[str, Any]:
    mode = _mode(sensor_mode)
    if prediction_split not in {None, "validation", "test"}:
        raise TerraMindSen1ConfigError("prediction_split must be validation, test, or omitted for fit.")
    if prediction_split and probability_output_dir is None:
        raise TerraMindSen1ConfigError("Prediction configs require probability_output_dir.")
    if int(checkpoint_mirror_every_n_epochs) <= 0:
        raise TerraMindSen1ConfigError("checkpoint_mirror_every_n_epochs must be positive.")
    # Keep the released TerraMind *pre-training* standardisation used by IBM's
    # public Sen1Floods11 fine-tuning recipe.  TerraTorch's integration fixture
    # contains dataset statistics for a packaged downstream checkpoint; those
    # are not interchangeable with a run starting from the pretrained encoder.
    profile = INPUT_PROFILES["sen1floods11_l1c"]
    modalities: list[str] = []
    roots: dict[str, str] = {}
    means: dict[str, list[float]] = {}
    stds: dict[str, list[float]] = {}
    image_grep: dict[str, str] = {}
    if "S2" in mode:
        modalities.append("S2L1C")
        roots["S2L1C"] = _path_text(s2_root)
        means["S2L1C"] = list(profile.s2_mean)
        stds["S2L1C"] = list(profile.s2_std)
        image_grep["S2L1C"] = "*_S2Hand.tif"
    if "S1" in mode:
        modalities.append("S1GRD")
        roots["S1GRD"] = _path_text(s1_root)
        means["S1GRD"] = list(S1_MEAN)
        stds["S1GRD"] = list(S1_STD)
        image_grep["S1GRD"] = "*_S1Hand.tif"
    rgb_modality = "S2L1C" if "S2L1C" in modalities else "S1GRD"
    rgb_indices = {rgb_modality: [3, 2, 1] if rgb_modality == "S2L1C" else [0]}
    if prediction_split == "validation":
        datamodule = "rsfm_fairness_audit.terratorch_exports.LabeledValidationAsPredictDataModule"
    elif prediction_split == "test":
        datamodule = "rsfm_fairness_audit.terratorch_exports.LabeledTestAsPredictDataModule"
    else:
        datamodule = "terratorch.datamodules.GenericMultiModalDataModule"
    dataset_bands = {
        modality: list(range(13 if modality == "S2L1C" else 2)) for modality in modalities
    }
    callbacks: list[dict[str, Any]] = [
        {"class_path": "lightning.pytorch.callbacks.RichProgressBar"},
        {
            "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
            "init_args": {"logging_interval": "epoch"},
        },
        {
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_args": {
                "dirpath": (Path(run_dir) / "checkpoints").as_posix(),
                "filename": "best-{epoch:03d}",
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": True,
                "auto_insert_metric_name": False,
            },
        },
    ]
    if prediction_split:
        callbacks.append(
            {
                "class_path": "rsfm_fairness_audit.terratorch_exports.GeoBWERProbabilityWriter",
                "init_args": {"output_dir": _path_text(probability_output_dir), "write_interval": "batch"},
            }
        )
    if prediction_split is None and persistent_checkpoint_dir is not None:
        callbacks.append(
            {
                "class_path": "rsfm_fairness_audit.terratorch_exports.PersistentCheckpointMirror",
                "init_args": {
                    "source_dir": (Path(run_dir) / "checkpoints").as_posix(),
                    "persistent_dir": _path_text(persistent_checkpoint_dir),
                    "every_n_epochs": int(checkpoint_mirror_every_n_epochs),
                },
            }
        )
    return {
        "seed_everything": int(seed),
        "trainer": {
            "accelerator": "auto",
            "strategy": "auto",
            "devices": "auto",
            "num_nodes": 1,
            "precision": "16-mixed",
            "deterministic": True,
            "logger": True,
            "callbacks": callbacks,
            "max_epochs": int(max_epochs),
            "fast_dev_run": bool(fast_dev_run),
            "log_every_n_steps": 5,
            "default_root_dir": _path_text(run_dir),
        },
        "data": {
            "class_path": datamodule,
            "init_args": {
                "task": "segmentation",
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "modalities": modalities,
                "allow_substring_file_names": True,
                "channel_position": -3,
                "check_stackability": True,
                "concat_bands": False,
                "data_with_sample_dim": False,
                "dataset_bands": dataset_bands,
                "expand_temporal_dimension": False,
                "output_bands": dataset_bands,
                "reduce_zero_label": False,
                "rgb_indices": rgb_indices,
                "rgb_modality": rgb_modality,
                "sample_replace": False,
                "shared_transforms": True,
                "train_data_root": roots,
                "val_data_root": roots,
                "test_data_root": roots,
                "train_label_data_root": _path_text(label_root),
                "val_label_data_root": _path_text(label_root),
                "test_label_data_root": _path_text(label_root),
                "train_split": _path_text(train_split),
                "val_split": _path_text(val_split),
                "test_split": _path_text(test_split),
                "image_grep": image_grep,
                "label_grep": "*_LabelHand.tif",
                "no_label_replace": -1,
                "no_data_replace": 0,
                "num_classes": 2,
                "means": means,
                "stds": stds,
                "drop_last": True,
                "pin_memory": True,
                "train_transform": [
                    {"class_path": "albumentations.D4", "init_args": {"p": 1.0}},
                    {
                        "class_path": "albumentations.pytorch.transforms.ToTensorV2",
                        "init_args": {"transpose_mask": False},
                    },
                ],
            },
        },
        "model": {
            "class_path": "rsfm_fairness_audit.terratorch_exports.GeoBWERSemanticSegmentationTask",
            "init_args": {
                "model_factory": "EncoderDecoderFactory",
                "model_args": {
                    "backbone": "terramind_v1_base",
                    # TerraTorch's online pretrained resolver follows a mutable
                    # model-repository ref. The campaign runner verifies the
                    # frozen official file before this config is executed.
                    "backbone_pretrained": False,
                    "backbone_ckpt_path": _path_text(backbone_checkpoint_path),
                    "backbone_modalities": modalities,
                    "backbone_bands": dataset_bands,
                    "backbone_merge_method": "mean",
                    "necks": [
                        {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                        {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                        {"name": "LearnedInterpolateToPyramidal"},
                    ],
                    "decoder": "UNetDecoder",
                    "decoder_channels": [512, 256, 128, 64],
                    "head_dropout": 0.1,
                    "num_classes": 2,
                },
                "loss": "dice",
                "ignore_index": -1,
                "freeze_backbone": False,
                "freeze_decoder": False,
                "freeze_head": False,
                "class_names": ["Others", "Flood"],
            },
        },
        "optimizer": {"class_path": "torch.optim.AdamW", "init_args": {"lr": 2e-5}},
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "init_args": {"monitor": "val/loss", "factor": 0.5, "patience": 5},
        },
    }


def write_terramind_sen1floods11_config(path: str | Path, **kwargs: Any) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise TerraMindSen1ConfigError("PyYAML is required to write TerraTorch configs.") from exc
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(build_terramind_sen1floods11_config(**kwargs), sort_keys=False), encoding="utf-8"
    )
    return output


def validate_terramind_sen1_source_layout(values: Mapping[str, str | Path]) -> dict[str, Any]:
    required = ("s1_root", "s2_root", "label_root", "train_split", "val_split", "test_split")
    missing = [name for name in required if name not in values or not Path(values[name]).exists()]
    if missing:
        raise TerraMindSen1ConfigError(f"Missing TerraMind Sen1Floods11 source paths: {missing}")
    counts = {
        "s1": len(list(Path(values["s1_root"]).glob("*_S1Hand.tif"))),
        "s2": len(list(Path(values["s2_root"]).glob("*_S2Hand.tif"))),
        "labels": len(list(Path(values["label_root"]).glob("*_LabelHand.tif"))),
    }
    if min(counts.values()) == 0 or len(set(counts.values())) != 1:
        raise TerraMindSen1ConfigError(f"S1/S2/label file counts are empty or unpaired: {counts}")
    return {"status": "ready", "counts": counts, "paths": {key: str(values[key]) for key in required}}


__all__ = [
    "TerraMindSen1ConfigError",
    "build_terramind_sen1floods11_config",
    "validate_terramind_sen1_source_layout",
    "write_terramind_sen1floods11_config",
]
