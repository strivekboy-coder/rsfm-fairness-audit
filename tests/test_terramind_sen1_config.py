from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rsfm_fairness_audit.adapters.terramind import validate_terratorch_runtime
from rsfm_fairness_audit.terramind_sen1_config import (
    OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS,
    TerraMindSen1ConfigError,
    build_terramind_sen1floods11_config,
    prepare_terramind_sen1_splits,
    read_sen1floods11_split_prefixes,
    write_terramind_sen1floods11_config,
)


def _build(mode: str, prediction_split=None):
    return build_terramind_sen1floods11_config(
        sensor_mode=mode,
        s1_root="/content/sen1/S1GRDHand",
        s2_root="/content/sen1/S2L1CHand",
        label_root="/content/sen1/LabelHand",
        train_split="/content/sen1/splits/flood_train_data.txt",
        val_split="/content/sen1/splits/flood_valid_data.txt",
        test_split="/content/sen1/splits/flood_test_data.txt",
        run_dir=f"/content/outputs/{mode}",
        backbone_checkpoint_path="/content/models/TerraMind_v1_base.pt",
        prediction_split=prediction_split,
        probability_output_dir=None if prediction_split is None else f"/content/outputs/{mode}/{prediction_split}",
    )


def _sen1_source_fixture(tmp_path, split_members, *, unused=()):
    roots = {
        "s1_root": tmp_path / "S1GRDHand",
        "s2_root": tmp_path / "S2L1CHand",
        "label_root": tmp_path / "LabelHand",
    }
    for root in roots.values():
        root.mkdir(parents=True)
    members = list(dict.fromkeys([*split_members["train"], *split_members["validation"], *split_members["test"], *unused]))
    for prefix in members:
        (roots["s1_root"] / f"{prefix}_S1Hand.tif").touch()
        (roots["s2_root"] / f"{prefix}_S2Hand.tif").touch()
        (roots["label_root"] / f"{prefix}_LabelHand.tif").touch()
    split_dir = tmp_path / "official_splits"
    split_dir.mkdir()
    split_paths = {}
    for split, prefixes in split_members.items():
        name = "val" if split == "validation" else split
        path = split_dir / f"{name}.csv"
        path.write_text(
            "".join(f"{prefix}_S1Hand.tif,{prefix}_LabelHand.tif\n" for prefix in prefixes),
            encoding="utf-8",
        )
        split_paths[split] = path
    return {
        **roots,
        "train_split": split_paths["train"],
        "val_split": split_paths["validation"],
        "test_split": split_paths["test"],
    }


@pytest.mark.parametrize(
    ("mode", "modalities"),
    [("S1", ["S1GRD"]), ("S2", ["S2L1C"]), ("S1+S2", ["S2L1C", "S1GRD"])],
)
def test_sensor_modes_change_only_the_controlled_modalities(mode, modalities):
    config = _build(mode)
    assert config["data"]["class_path"] == "terratorch.datamodules.GenericMultiModalDataModule"
    assert config["data"]["init_args"]["modalities"] == modalities
    assert config["data"]["init_args"]["concat_bands"] is False
    assert config["data"]["init_args"]["shared_transforms"] is True
    expected_bands = {item: list(range(13 if item == "S2L1C" else 2)) for item in modalities}
    assert config["data"]["init_args"]["dataset_bands"] == expected_bands
    assert config["data"]["init_args"]["output_bands"] == expected_bands
    assert config["model"]["init_args"]["model_args"]["backbone_modalities"] == modalities
    assert config["model"]["init_args"]["model_args"]["backbone_bands"] == expected_bands
    assert config["model"]["init_args"]["model_args"]["backbone_pretrained"] is False
    assert config["model"]["init_args"]["model_args"]["backbone_ckpt_path"] == "/content/models/TerraMind_v1_base.pt"
    assert config["model"]["init_args"]["model_args"]["backbone_merge_method"] == "mean"
    assert config["trainer"]["max_epochs"] == 100
    assert config["trainer"]["fast_dev_run"] is False


def test_uses_released_terramind_pretraining_standardisation_and_recipe():
    config = _build("S1+S2")
    args = config["data"]["init_args"]
    assert args["means"]["S1GRD"] == [-12.599, -20.293]
    assert args["means"]["S2L1C"][:3] == [2357.089, 2137.385, 2018.788]
    assert config["model"]["init_args"]["loss"] == "dice"
    assert config["optimizer"]["init_args"]["lr"] == 2e-5


def test_fit_scheduler_uses_lightning_cli_plateau_wrapper():
    config = _build("S1+S2")
    assert config["lr_scheduler"] == {
        "class_path": "lightning.pytorch.cli.ReduceLROnPlateau",
        "init_args": {"monitor": "val/loss", "factor": 0.5, "patience": 5},
    }


def test_official_split_adapter_preserves_252_89_90_membership(tmp_path):
    members = [f"Event{index % 11:02d}_{index:03d}" for index in range(431)]
    splits = {
        "train": members[:252],
        "validation": members[252:341],
        "test": members[341:431],
    }
    source = _sen1_source_fixture(
        tmp_path,
        splits,
        unused=[f"Unused_{index:02d}" for index in range(15)],
    )
    report = prepare_terramind_sen1_splits(source, tmp_path / "terratorch_splits")

    assert OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS == {"train": 252, "validation": 89, "test": 90}
    assert report["counts"] == {"s1": 446, "s2": 446, "label": 446}
    assert report["split_counts"] == OFFICIAL_SEN1FLOODS11_SPLIT_COUNTS
    assert report["selected_unique_samples"] == 431
    assert report["unused_complete_samples"] == 15
    for split, expected_members in splits.items():
        generated = Path(report["terratorch_split_paths"][split])
        assert generated.read_text(encoding="utf-8").splitlines() == expected_members
        assert read_sen1floods11_split_prefixes(generated) == expected_members


def test_split_preflight_requires_matching_s1_s2_and_label_members(tmp_path):
    splits = {"train": ["A"], "validation": ["B"], "test": ["C"]}
    source = _sen1_source_fixture(tmp_path, splits)
    Path(source["s2_root"], "B_S2Hand.tif").unlink()
    Path(source["s2_root"], "Different_S2Hand.tif").touch()
    with pytest.raises(TerraMindSen1ConfigError, match="same sample prefixes"):
        prepare_terramind_sen1_splits(
            source,
            tmp_path / "terratorch_splits",
            expected_split_counts={"train": 1, "validation": 1, "test": 1},
        )


def test_split_preflight_rejects_mismatched_label_pair_and_overlap(tmp_path):
    splits = {"train": ["A"], "validation": ["B"], "test": ["C"]}
    source = _sen1_source_fixture(tmp_path, splits)
    Path(source["train_split"]).write_text("A_S1Hand.tif,B_LabelHand.tif\n", encoding="utf-8")
    with pytest.raises(TerraMindSen1ConfigError, match="do not identify the same sample"):
        prepare_terramind_sen1_splits(
            source,
            tmp_path / "bad_pair",
            expected_split_counts={"train": 1, "validation": 1, "test": 1},
        )

    Path(source["train_split"]).write_text("A_S1Hand.tif,A_LabelHand.tif\n", encoding="utf-8")
    Path(source["val_split"]).write_text("A_S1Hand.tif,A_LabelHand.tif\n", encoding="utf-8")
    with pytest.raises(TerraMindSen1ConfigError, match="overlap"):
        prepare_terramind_sen1_splits(
            source,
            tmp_path / "overlap",
            expected_split_counts={"train": 1, "validation": 1, "test": 1},
        )


def test_final_runner_dry_run_rewires_all_modes_to_prefix_splits(tmp_path):
    import yaml

    members = [f"Event{index % 11:02d}_{index:03d}" for index in range(431)]
    splits = {
        "train": members[:252],
        "validation": members[252:341],
        "test": members[341:431],
    }
    source = _sen1_source_fixture(
        tmp_path,
        splits,
        unused=[f"Unused_{index:02d}" for index in range(15)],
    )
    output_dir = tmp_path / "runner_output"
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "colab" / "run_terramind_sen1floods11_final_colab.py"),
            "--s1-root",
            str(source["s1_root"]),
            "--s2-root",
            str(source["s2_root"]),
            "--label-root",
            str(source["label_root"]),
            "--train-split",
            str(source["train_split"]),
            "--val-split",
            str(source["val_split"]),
            "--test-split",
            str(source["test_split"]),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(tmp_path / "TerraMind_v1_base.pt"),
            "--dry-run",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    expected_paths = {
        "train_split": (output_dir / "terratorch_splits" / "train_prefixes.txt").as_posix(),
        "val_split": (output_dir / "terratorch_splits" / "validation_prefixes.txt").as_posix(),
        "test_split": (output_dir / "terratorch_splits" / "test_prefixes.txt").as_posix(),
    }
    for slug in ("s1", "s2", "s1_plus_s2"):
        for seed in (42, 73, 101):
            config = yaml.safe_load(
                (
                    output_dir / slug / f"seed_{seed}" / "configs" / "fit.yaml"
                ).read_text(encoding="utf-8")
            )
            data_args = config["data"]["init_args"]
            assert data_args["test_split"] == expected_paths["test_split"]
            assert "model_selection_seed_" in data_args["train_split"]
            assert "model_selection_seed_" in data_args["val_split"]
            assert config["seed_everything"] == seed
            fit_members = set(
                Path(data_args["train_split"]).read_text(encoding="utf-8").split()
            )
            selection_members = set(
                Path(data_args["val_split"]).read_text(encoding="utf-8").split()
            )
            assert fit_members | selection_members == set(splits["train"])
            assert fit_members.isdisjoint(selection_members)
            assert {
                value.split("_", 1)[0] for value in fit_members
            }.isdisjoint(
                {value.split("_", 1)[0] for value in selection_members}
            )
            prediction_config = yaml.safe_load(
                (
                    output_dir
                    / slug
                    / f"seed_{seed}"
                    / "configs"
                    / "predict_validation.yaml"
                ).read_text(encoding="utf-8")
            )
            assert (
                prediction_config["data"]["init_args"]["val_split"]
                == expected_paths["val_split"]
            )
    preflight = (output_dir / "source_preflight.json").read_text(encoding="utf-8")
    assert '"split_counts"' in preflight
    assert '"unused_complete_samples": 15' in preflight


@pytest.mark.skipif(
    importlib.util.find_spec("terratorch") is None,
    reason="TerraTorch CLI integration test requires the frozen TerraTorch runtime.",
)
def test_generated_fit_yaml_parses_with_frozen_terratorch_cli(tmp_path):
    validate_terratorch_runtime()
    config_path = write_terramind_sen1floods11_config(
        tmp_path / "fit.yaml",
        sensor_mode="S1+S2",
        s1_root="/content/sen1/S1GRDHand",
        s2_root="/content/sen1/S2L1CHand",
        label_root="/content/sen1/LabelHand",
        train_split="/content/sen1/splits/flood_train_data.txt",
        val_split="/content/sen1/splits/flood_valid_data.txt",
        test_split="/content/sen1/splits/flood_test_data.txt",
        run_dir="/content/outputs/terramind_sen1_smoke",
        backbone_checkpoint_path="/content/models/TerraMind_v1_base.pt",
        diagnostic_batch_limit=2,
    )
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(project_root / "src"), env.get("PYTHONPATH", ""))
        if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "terratorch",
            "fit",
            "--config",
            str(config_path),
            "--print_config",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        "Frozen TerraTorch CLI rejected the generated fit config.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "lightning.pytorch.cli.ReduceLROnPlateau" in result.stdout
    assert "monitor: val/loss" in result.stdout
    assert "limit_train_batches: 2" in result.stdout
    assert "limit_val_batches: 2" in result.stdout
    assert "limit_predict_batches: 2" in result.stdout


@pytest.mark.skipif(
    importlib.util.find_spec("terratorch") is None or importlib.util.find_spec("rasterio") is None,
    reason="Real TerraTorch dataset construction requires TerraTorch and rasterio.",
)
def test_real_terratorch_dataset_builds_for_all_sensor_modes(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from terratorch.datamodules import GenericMultiModalDataModule

    validate_terratorch_runtime()
    splits = {"train": ["A", "B"], "validation": ["C"], "test": ["D"]}
    source = _sen1_source_fixture(tmp_path, splits)
    transform = from_origin(0, 16, 1, 1)
    for prefix in ("A", "B", "C", "D"):
        for root_key, suffix, count in (
            ("s1_root", "_S1Hand.tif", 2),
            ("s2_root", "_S2Hand.tif", 13),
        ):
            with rasterio.open(
                Path(source[root_key], f"{prefix}{suffix}"),
                "w",
                driver="GTiff",
                height=16,
                width=16,
                count=count,
                dtype="float32",
                transform=transform,
            ) as dataset:
                dataset.write(np.ones((count, 16, 16), dtype=np.float32))
        with rasterio.open(
            Path(source["label_root"], f"{prefix}_LabelHand.tif"),
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=1,
            dtype="uint8",
            transform=transform,
        ) as dataset:
            dataset.write(np.zeros((1, 16, 16), dtype=np.uint8))
    report = prepare_terramind_sen1_splits(
        source,
        tmp_path / "terratorch_splits",
        expected_split_counts={"train": 2, "validation": 1, "test": 1},
    )
    generated = report["terratorch_split_paths"]
    for mode, expected_modalities in (
        ("S1", {"S1GRD"}),
        ("S2", {"S2L1C"}),
        ("S1+S2", {"S1GRD", "S2L1C"}),
    ):
        config = build_terramind_sen1floods11_config(
            sensor_mode=mode,
            s1_root=source["s1_root"],
            s2_root=source["s2_root"],
            label_root=source["label_root"],
            train_split=generated["train"],
            val_split=generated["validation"],
            test_split=generated["test"],
            run_dir=tmp_path / f"run_{mode}",
            backbone_checkpoint_path=tmp_path / "TerraMind_v1_base.pt",
            batch_size=1,
            num_workers=0,
            fast_dev_run=True,
        )
        data_args = dict(config["data"]["init_args"])
        data_args.pop("train_transform")
        datamodule = GenericMultiModalDataModule(**data_args)
        datamodule.setup("fit")
        assert len(datamodule.train_dataset) == 2
        assert len(datamodule.val_dataset) == 1
        batch = next(iter(datamodule.train_dataloader()))
        assert set(batch["image"]) == expected_modalities
        assert "mask" in batch


@pytest.mark.skipif(
    os.environ.get("RSFM_RUN_TERRAMIND_GPU_INTEGRATION") != "1",
    reason="Set RSFM_RUN_TERRAMIND_GPU_INTEGRATION=1 with real Sen1 paths to run the GPU fast-dev regression.",
)
def test_real_terratorch_fast_dev_run_uses_adapted_split(tmp_path):
    validate_terratorch_runtime()
    environment_names = {
        "s1_root": "RSFM_SEN1_S1_ROOT",
        "s2_root": "RSFM_SEN1_S2_ROOT",
        "label_root": "RSFM_SEN1_LABEL_ROOT",
        "train_split": "RSFM_SEN1_TRAIN_SPLIT",
        "val_split": "RSFM_SEN1_VAL_SPLIT",
        "test_split": "RSFM_SEN1_TEST_SPLIT",
    }
    missing = [name for name in [*environment_names.values(), "RSFM_TERRAMIND_CHECKPOINT"] if not os.environ.get(name)]
    assert not missing, f"Missing GPU integration environment variables: {missing}"
    source = {key: Path(os.environ[env_name]) for key, env_name in environment_names.items()}
    report = prepare_terramind_sen1_splits(source, tmp_path / "terratorch_splits")
    split_paths = report["terratorch_split_paths"]
    config_path = write_terramind_sen1floods11_config(
        tmp_path / "fit.yaml",
        sensor_mode="S1+S2",
        s1_root=source["s1_root"],
        s2_root=source["s2_root"],
        label_root=source["label_root"],
        train_split=split_paths["train"],
        val_split=split_paths["validation"],
        test_split=split_paths["test"],
        run_dir=tmp_path / "fast_dev_run",
        backbone_checkpoint_path=os.environ["RSFM_TERRAMIND_CHECKPOINT"],
        batch_size=1,
        num_workers=0,
        fast_dev_run=True,
    )
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(project_root / "src"), env.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [sys.executable, "-m", "terratorch", "fit", "--config", str(config_path)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, (
        "Frozen TerraTorch failed the real adapted-split fast_dev_run.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_prediction_uses_labeled_validation_adapter_and_probability_writer():
    config = _build("S2", "validation")
    assert config["data"]["class_path"].endswith("LabeledValidationAsPredictDataModule")
    assert any(callback["class_path"].endswith("GeoBWERProbabilityWriter") for callback in config["trainer"]["callbacks"])


def test_prediction_output_is_mandatory():
    with pytest.raises(TerraMindSen1ConfigError, match="probability_output_dir"):
        build_terramind_sen1floods11_config(
            sensor_mode="S2",
            s1_root="s1",
            s2_root="s2",
            label_root="labels",
            train_split="train",
            val_split="val",
            test_split="test",
            run_dir="run",
            backbone_checkpoint_path="/content/models/TerraMind_v1_base.pt",
            prediction_split="test",
        )


def test_fit_can_mirror_checkpoints_without_using_drive_as_live_run_dir():
    config = build_terramind_sen1floods11_config(
        sensor_mode="S2",
        s1_root="/content/sen1/S1GRDHand",
        s2_root="/content/sen1/S2L1CHand",
        label_root="/content/sen1/LabelHand",
        train_split="/content/sen1/splits/train.txt",
        val_split="/content/sen1/splits/val.txt",
        test_split="/content/sen1/splits/test.txt",
        run_dir="/content/runs/s2",
        backbone_checkpoint_path="/content/models/TerraMind_v1_base.pt",
        persistent_checkpoint_dir="/content/drive/MyDrive/rsfm_fairness_audit/outputs/s2/checkpoints",
    )
    callbacks = config["trainer"]["callbacks"]
    mirror = next(item for item in callbacks if item["class_path"].endswith("PersistentCheckpointMirror"))
    assert mirror["init_args"]["source_dir"] == "/content/runs/s2/checkpoints"
    assert mirror["init_args"]["every_n_epochs"] == 5


def test_real_gpu_smoke_uses_lightning_fast_dev_run():
    config = build_terramind_sen1floods11_config(
        sensor_mode="S1",
        s1_root="/content/sen1/S1GRDHand",
        s2_root="/content/sen1/S2L1CHand",
        label_root="/content/sen1/LabelHand",
        train_split="/content/sen1/splits/train.txt",
        val_split="/content/sen1/splits/val.txt",
        test_split="/content/sen1/splits/test.txt",
        run_dir="/content/runs/s1_smoke",
        backbone_checkpoint_path="/content/models/TerraMind_v1_base.pt",
        fast_dev_run=True,
    )
    assert config["trainer"]["fast_dev_run"] is True


def test_bounded_end_to_end_smoke_keeps_checkpointing_and_limits_all_stages():
    import yaml

    config = build_terramind_sen1floods11_config(
        sensor_mode="S1+S2",
        s1_root="/content/sen1/S1GRDHand",
        s2_root="/content/sen1/S2L1CHand",
        label_root="/content/sen1/LabelHand",
        train_split="/content/sen1/splits/train.txt",
        val_split="/content/sen1/splits/val.txt",
        test_split="/content/sen1/splits/test.txt",
        run_dir="/content/runs/smoke",
        backbone_checkpoint_path="/content/models/TerraMind_v1_base.pt",
        max_epochs=1,
        diagnostic_batch_limit=2,
    )
    trainer = config["trainer"]
    assert trainer["fast_dev_run"] is False
    assert trainer["max_epochs"] == 1
    assert trainer["limit_train_batches"] == 2
    assert trainer["limit_val_batches"] == 2
    assert trainer["limit_predict_batches"] == 2
    assert isinstance(trainer["limit_train_batches"], int)
    assert isinstance(trainer["limit_val_batches"], int)
    assert isinstance(trainer["limit_predict_batches"], int)
    assert trainer["num_sanity_val_steps"] == 0
    assert any(
        callback["class_path"].endswith("ModelCheckpoint")
        for callback in trainer["callbacks"]
    )
    round_trip = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
    assert round_trip["trainer"]["limit_train_batches"] == 2
    assert isinstance(round_trip["trainer"]["limit_train_batches"], int)


def test_diagnostic_batch_limit_must_be_positive():
    with pytest.raises(TerraMindSen1ConfigError, match="diagnostic_batch_limit"):
        build_terramind_sen1floods11_config(
            sensor_mode="S1",
            s1_root="s1",
            s2_root="s2",
            label_root="labels",
            train_split="train",
            val_split="val",
            test_split="test",
            run_dir="run",
            backbone_checkpoint_path="checkpoint",
            diagnostic_batch_limit=0,
        )


def test_diagnostic_batch_limit_rejects_lightning_ambiguous_one():
    with pytest.raises(TerraMindSen1ConfigError, match="ambiguously"):
        build_terramind_sen1floods11_config(
            sensor_mode="S1",
            s1_root="s1",
            s2_root="s2",
            label_root="labels",
            train_split="train",
            val_split="val",
            test_split="test",
            run_dir="run",
            backbone_checkpoint_path="checkpoint",
            diagnostic_batch_limit=1,
        )
