from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rsfm_fairness_audit.adapters.terramind import validate_terratorch_runtime
from rsfm_fairness_audit.terramind_sen1_config import (
    TerraMindSen1ConfigError,
    build_terramind_sen1floods11_config,
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
        fast_dev_run=True,
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
