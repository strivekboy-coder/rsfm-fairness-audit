from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.adapters.terramind import INPUT_PROFILES, S1_MEAN, S1_STD


class TerraMindSen1ConfigError(ValueError):
    """Raised when a TerraMind Sen1Floods11 campaign config is incomplete."""


OFFICIAL_SEN1FLOODS11_CORE_SPLIT_COUNTS = {
    "train": 252,
    "validation": 89,
    "test": 90,
}
OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS = {
    **OFFICIAL_SEN1FLOODS11_CORE_SPLIT_COUNTS,
    "bolivia_holdout": 15,
}
OFFICIAL_SEN1FLOODS11_SAMPLE_COUNT = 446
OFFICIAL_SEN1FLOODS11_EVENT_COUNT = 11
SEN1FLOODS11_SUFFIXES = {
    "s1": "_S1Hand.tif",
    "s2": "_S2Hand.tif",
    "label": "_LabelHand.tif",
}


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
    diagnostic_batch_limit: int | None = None,
) -> dict[str, Any]:
    mode = _mode(sensor_mode)
    if prediction_split not in {None, "validation", "test", "bolivia_holdout"}:
        raise TerraMindSen1ConfigError(
            "prediction_split must be validation, test, bolivia_holdout, or omitted for fit."
        )
    if prediction_split and probability_output_dir is None:
        raise TerraMindSen1ConfigError("Prediction configs require probability_output_dir.")
    if int(checkpoint_mirror_every_n_epochs) <= 0:
        raise TerraMindSen1ConfigError("checkpoint_mirror_every_n_epochs must be positive.")
    if diagnostic_batch_limit is not None and int(diagnostic_batch_limit) < 2:
        raise TerraMindSen1ConfigError(
            "diagnostic_batch_limit must be at least 2. Lightning treats 1/1.0 "
            "ambiguously as 100% of batches; an integer >=2 is unambiguously a "
            "fixed, small batch count."
        )
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
    elif prediction_split in {"test", "bolivia_holdout"}:
        datamodule = "rsfm_fairness_audit.terratorch_exports.LabeledTestAsPredictDataModule"
    else:
        datamodule = "rsfm_fairness_audit.terratorch_exports.GeoBWERSen1DataModule"
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
    trainer: dict[str, Any] = {
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
    }
    if diagnostic_batch_limit is not None:
        # Unlike Lightning fast_dev_run, this bounded diagnostic keeps
        # checkpointing enabled so the real predict + probability-writer path
        # can be exercised after the one-epoch fit.
        trainer.update(
            {
                "limit_train_batches": int(diagnostic_batch_limit),
                "limit_val_batches": int(diagnostic_batch_limit),
                "limit_predict_batches": int(diagnostic_batch_limit),
                "num_sanity_val_steps": 0,
            }
        )
    return {
        "seed_everything": int(seed),
        "trainer": trainer,
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
                # Scalar raw-zero replacement would become extreme after
                # standardisation. The repository datamodule instead replaces
                # every non-finite value with its frozen per-band mean.
                "no_data_replace": None,
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
            "class_path": "lightning.pytorch.cli.ReduceLROnPlateau",
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modality_prefixes(root: Path, suffix: str) -> set[str]:
    return {path.name[: -len(suffix)] for path in root.glob(f"*{suffix}") if path.is_file()}


def _validate_source_inventory(values: Mapping[str, str | Path]) -> tuple[dict[str, Path], dict[str, set[str]]]:
    required = (
        "s1_root",
        "s2_root",
        "label_root",
        "train_split",
        "val_split",
        "test_split",
        "bolivia_split",
    )
    missing = [name for name in required if name not in values or not Path(values[name]).exists()]
    if missing:
        raise TerraMindSen1ConfigError(f"Missing TerraMind Sen1Floods11 source paths: {missing}")
    paths = {name: Path(values[name]) for name in required}
    for root_name in ("s1_root", "s2_root", "label_root"):
        if not paths[root_name].is_dir():
            raise TerraMindSen1ConfigError(f"{root_name} must be a directory: {paths[root_name]}")
    for split_name in ("train_split", "val_split", "test_split", "bolivia_split"):
        if not paths[split_name].is_file():
            raise TerraMindSen1ConfigError(f"{split_name} must be a file: {paths[split_name]}")
    prefixes = {
        "s1": _modality_prefixes(paths["s1_root"], SEN1FLOODS11_SUFFIXES["s1"]),
        "s2": _modality_prefixes(paths["s2_root"], SEN1FLOODS11_SUFFIXES["s2"]),
        "label": _modality_prefixes(paths["label_root"], SEN1FLOODS11_SUFFIXES["label"]),
    }
    counts = {name: len(items) for name, items in prefixes.items()}
    if min(counts.values()) == 0 or len(set(counts.values())) != 1:
        raise TerraMindSen1ConfigError(f"S1/S2/label file counts are empty or unequal: {counts}")
    if not (prefixes["s1"] == prefixes["s2"] == prefixes["label"]):
        mismatch = {
            "missing_s1": sorted((prefixes["s2"] | prefixes["label"]) - prefixes["s1"])[:10],
            "missing_s2": sorted((prefixes["s1"] | prefixes["label"]) - prefixes["s2"])[:10],
            "missing_label": sorted((prefixes["s1"] | prefixes["s2"]) - prefixes["label"])[:10],
        }
        raise TerraMindSen1ConfigError(
            f"S1/S2/Label inventories do not represent the same sample prefixes: {mismatch}"
        )
    return paths, prefixes


def _is_official_split_header(row: Sequence[str]) -> bool:
    if len(row) != 2:
        return False
    s1_header = row[0].strip().casefold().replace(" ", "_")
    label_header = row[1].strip().casefold().replace(" ", "_")
    return s1_header in {"s1", "s1_file", "s1_filename", "image", "image_file"} and label_header in {
        "label",
        "label_file",
        "label_filename",
        "mask",
        "mask_file",
    }


def _split_prefix(row: Sequence[str], *, source: Path, line_number: int) -> str:
    cells = [cell.strip() for cell in row]
    while cells and cells[-1] == "":
        cells.pop()
    if len(cells) not in {1, 2} or not cells[0]:
        raise TerraMindSen1ConfigError(
            f"Split row must contain one prefix/S1 filename or an S1,Label pair: "
            f"{source}:{line_number}: {list(row)!r}"
        )
    s1_value = cells[0]
    s1_suffix = SEN1FLOODS11_SUFFIXES["s1"]
    label_suffix = SEN1FLOODS11_SUFFIXES["label"]
    if s1_value.endswith(s1_suffix):
        prefix = s1_value[: -len(s1_suffix)]
    elif len(cells) == 1 and not s1_value.lower().endswith((".tif", ".tiff")):
        prefix = s1_value
    else:
        raise TerraMindSen1ConfigError(
            f"Split S1 entry must end with {s1_suffix!r} or be a bare prefix: "
            f"{source}:{line_number}: {s1_value!r}"
        )
    if not prefix or "/" in prefix or "\\" in prefix or "*" in prefix:
        raise TerraMindSen1ConfigError(
            f"Split sample prefix must be a flat, non-wildcard filename prefix: "
            f"{source}:{line_number}: {prefix!r}"
        )
    if len(cells) == 2:
        expected_label = f"{prefix}{label_suffix}"
        if cells[1] != expected_label:
            raise TerraMindSen1ConfigError(
                f"S1 and Label entries do not identify the same sample at {source}:{line_number}: "
                f"expected {expected_label!r}, observed {cells[1]!r}."
            )
    return prefix


def read_sen1floods11_split_prefixes(path: str | Path) -> list[str]:
    """Read an official two-column split or a TerraTorch one-prefix-per-line split."""

    source = Path(path)
    if not source.is_file():
        raise TerraMindSen1ConfigError(f"Split file does not exist: {source}")
    prefixes: list[str] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if not prefixes and _is_official_split_header(row):
                continue
            prefixes.append(_split_prefix(row, source=source, line_number=line_number))
    if not prefixes:
        raise TerraMindSen1ConfigError(f"Split file contains no sample members: {source}")
    duplicates = sorted(prefix for prefix, count in Counter(prefixes).items() if count > 1)
    if duplicates:
        raise TerraMindSen1ConfigError(f"Split file contains duplicate sample prefixes: {source}: {duplicates[:10]}")
    return prefixes


def prepare_terramind_sen1_splits(
    values: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    expected_split_counts: Mapping[str, int] | None = OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS,
) -> dict[str, Any]:
    """Materialize prefix-only TerraTorch splits without changing official membership."""

    paths, modality_prefixes = _validate_source_inventory(values)
    source_split_paths = {
        "train": paths["train_split"],
        "validation": paths["val_split"],
        "test": paths["test_split"],
        "bolivia_holdout": paths["bolivia_split"],
    }
    split_members = {
        split: read_sen1floods11_split_prefixes(path) for split, path in source_split_paths.items()
    }
    if expected_split_counts is not None:
        expected = {name: int(expected_split_counts[name]) for name in source_split_paths}
        observed = {name: len(members) for name, members in split_members.items()}
        if observed != expected:
            raise TerraMindSen1ConfigError(
                f"Official Sen1Floods11 split counts changed: expected={expected}, observed={observed}."
            )
    split_names = tuple(split_members)
    overlaps: dict[str, list[str]] = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            shared = sorted(set(split_members[left]) & set(split_members[right]))
            if shared:
                overlaps[f"{left}__{right}"] = shared[:10]
    if overlaps:
        raise TerraMindSen1ConfigError(f"Sen1Floods11 split members overlap: {overlaps}")
    complete_prefixes = modality_prefixes["s1"]
    unresolved: dict[str, list[str]] = {}
    for split, members in split_members.items():
        missing = sorted(set(members) - complete_prefixes)
        if missing:
            unresolved[split] = missing[:10]
    if unresolved:
        raise TerraMindSen1ConfigError(
            f"Split members do not resolve simultaneously to S1, S2, and Label files: {unresolved}"
        )
    selected = set().union(*(set(members) for members in split_members.values()))
    formal_four_split_contract = (
        expected_split_counts is not None
        and {
            name: int(expected_split_counts[name]) for name in source_split_paths
        }
        == OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS
    )
    if formal_four_split_contract and selected != complete_prefixes:
        raise TerraMindSen1ConfigError(
            "The four official Sen1Floods11 sets must partition the complete "
            f"{OFFICIAL_SEN1FLOODS11_SAMPLE_COUNT}-chip hand-labeled inventory: "
            f"selected={len(selected)}, inventory={len(complete_prefixes)}, "
            f"missing={sorted(complete_prefixes - selected)[:10]}, "
            f"extra={sorted(selected - complete_prefixes)[:10]}."
        )
    if formal_four_split_contract and len(selected) != OFFICIAL_SEN1FLOODS11_SAMPLE_COUNT:
        raise TerraMindSen1ConfigError(
            "The complete Sen1Floods11 hand-labeled partition must contain exactly "
            f"{OFFICIAL_SEN1FLOODS11_SAMPLE_COUNT} samples, observed={len(selected)}."
        )
    event = lambda prefix: str(prefix).split("_", 1)[0]
    complete_events = {event(prefix) for prefix in complete_prefixes}
    bolivia_events = {event(prefix) for prefix in split_members["bolivia_holdout"]}
    if formal_four_split_contract and bolivia_events != {"Bolivia"}:
        raise TerraMindSen1ConfigError(
            "The independent Bolivia holdout must contain exactly event={'Bolivia'}, "
            f"observed={sorted(bolivia_events)}."
        )
    core_bolivia = {
        split: sorted(prefix for prefix in members if event(prefix) == "Bolivia")[:10]
        for split, members in split_members.items()
        if split != "bolivia_holdout"
    }
    core_bolivia = {split: values for split, values in core_bolivia.items() if values}
    if formal_four_split_contract and core_bolivia:
        raise TerraMindSen1ConfigError(
            f"Bolivia leaked into train/validation/standard test: {core_bolivia}."
        )
    if formal_four_split_contract and len(complete_events) != OFFICIAL_SEN1FLOODS11_EVENT_COUNT:
        raise TerraMindSen1ConfigError(
            "The complete hand-labeled inventory must represent exactly "
            f"{OFFICIAL_SEN1FLOODS11_EVENT_COUNT} events, observed={sorted(complete_events)}."
        )
    standard_test_events = {event(prefix) for prefix in split_members["test"]}
    expected_non_bolivia_events = complete_events - {"Bolivia"}
    if formal_four_split_contract and standard_test_events != expected_non_bolivia_events:
        raise TerraMindSen1ConfigError(
            "Standard test must cover every one of the 10 non-Bolivia events before "
            "the combined evaluation can claim the fixed 11-event universe: "
            f"expected={sorted(expected_non_bolivia_events)}, "
            f"observed={sorted(standard_test_events)}."
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated_paths: dict[str, Path] = {}
    for split, members in split_members.items():
        generated = destination / f"{split}_prefixes.txt"
        temporary = generated.with_suffix(f"{generated.suffix}.tmp")
        temporary.write_text("".join(f"{prefix}\n" for prefix in members), encoding="utf-8")
        temporary.replace(generated)
        generated_paths[split] = generated
    return {
        "status": "ready",
        "schema": "geobwer.sen1floods11.terratorch_split_adapter.v2",
        "counts": {name: len(prefixes) for name, prefixes in modality_prefixes.items()},
        "split_counts": {name: len(members) for name, members in split_members.items()},
        "selected_unique_samples": len(selected),
        "unused_complete_samples": len(complete_prefixes - selected),
        "complete_event_count": len(complete_events),
        "complete_events": sorted(complete_events),
        "standard_test_events": sorted(standard_test_events),
        "bolivia_holdout_events": sorted(bolivia_events),
        "combined_evaluation_sample_count": (
            len(split_members["test"]) + len(split_members["bolivia_holdout"])
        ),
        "combined_evaluation_event_count": len(
            standard_test_events | bolivia_events
        ),
        "no_training_or_calibration_leakage": True,
        "source_split_paths": {name: str(path) for name, path in source_split_paths.items()},
        "source_split_sha256": {name: _file_sha256(path) for name, path in source_split_paths.items()},
        "terratorch_split_paths": {name: str(path) for name, path in generated_paths.items()},
        "terratorch_split_sha256": {name: _file_sha256(path) for name, path in generated_paths.items()},
        "roots": {
            "s1": str(paths["s1_root"]),
            "s2": str(paths["s2_root"]),
            "label": str(paths["label_root"]),
        },
    }


def validate_terramind_sen1_source_layout(values: Mapping[str, str | Path]) -> dict[str, Any]:
    paths, prefixes = _validate_source_inventory(values)
    return {
        "status": "ready",
        "counts": {name: len(items) for name, items in prefixes.items()},
        "paths": {key: str(path) for key, path in paths.items()},
    }


__all__ = [
    "TerraMindSen1ConfigError",
    "OFFICIAL_SEN1FLOODS11_CORE_SPLIT_COUNTS",
    "OFFICIAL_SEN1FLOODS11_EVENT_COUNT",
    "OFFICIAL_SEN1FLOODS11_SAMPLE_COUNT",
    "OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS",
    "build_terramind_sen1floods11_config",
    "prepare_terramind_sen1_splits",
    "read_sen1floods11_split_prefixes",
    "validate_terramind_sen1_source_layout",
    "write_terramind_sen1floods11_config",
]
