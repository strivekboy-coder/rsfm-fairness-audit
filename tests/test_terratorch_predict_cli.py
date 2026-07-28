from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rsfm_fairness_audit.terratorch_predict_cli import (
    TerraTorchPredictCLIError,
    _geobwer_cli_class,
    filter_geobwer_predict_callbacks,
    main,
)


def _callback(module: str, name: str):
    callback_type = type(name, (), {})
    callback_type.__module__ = module
    return callback_type()


def _callback_id(callback) -> tuple[str, str]:
    return callback.__class__.__module__, callback.__class__.__name__


def _callback_fixture() -> tuple[list[object], object, object, object]:
    rich = _callback("lightning.pytorch.callbacks.progress.rich_progress", "RichProgressBar")
    checkpoint = _callback("lightning.pytorch.callbacks.model_checkpoint", "ModelCheckpoint")
    default = _callback("terratorch.cli_tools", "CustomWriter")
    geobwer = _callback(
        "rsfm_fairness_audit.terratorch_exports",
        "GeoBWERProbabilityWriter",
    )
    return [rich, checkpoint, geobwer, default], rich, checkpoint, geobwer


def test_filter_removes_only_exact_terratorch_writer_and_retains_geobwer():
    callbacks, rich, checkpoint, geobwer = _callback_fixture()
    original = tuple(callbacks)
    retained, report = filter_geobwer_predict_callbacks(callbacks)

    assert callbacks == list(original)
    assert retained == [rich, checkpoint, geobwer]
    assert retained is not callbacks
    assert sum(
        _callback_id(callback)
        == ("terratorch.cli_tools", "CustomWriter")
        for callback in retained
    ) == 0
    assert sum(
        _callback_id(callback)
        == (
            "rsfm_fairness_audit.terratorch_exports",
            "GeoBWERProbabilityWriter",
        )
        for callback in retained
    ) == 1
    assert report["lifecycle"] == "after_cli_instantiation_before_trainer_predict"
    assert report["terratorch_custom_writer_count"] == 0
    assert report["geobwer_probability_writer_count"] == 1


@pytest.mark.parametrize(
    "callbacks, match",
    [
        (
            [
                _callback(
                    "rsfm_fairness_audit.terratorch_exports",
                    "GeoBWERProbabilityWriter",
                )
            ],
            "default CustomWriter",
        ),
        (
            [
                _callback("terratorch.cli_tools", "CustomWriter"),
                _callback("terratorch.cli_tools", "CustomWriter"),
                _callback(
                    "rsfm_fairness_audit.terratorch_exports",
                    "GeoBWERProbabilityWriter",
                ),
            ],
            "default CustomWriter",
        ),
        (
            [_callback("terratorch.cli_tools", "CustomWriter")],
            "exactly one GeoBWERProbabilityWriter",
        ),
    ],
)
def test_filter_hard_fails_on_callback_contract_drift(callbacks, match):
    with pytest.raises(TerraTorchPredictCLIError, match=match):
        filter_geobwer_predict_callbacks(callbacks)


def test_before_predict_filters_after_instantiation_without_in_loop_mutation(capsys):
    callbacks, rich, checkpoint, geobwer = _callback_fixture()

    class BaseCLI:
        def before_predict(self) -> None:
            self.parent_before_predict_called = True

    cli = _geobwer_cli_class(BaseCLI)()
    cli.trainer = SimpleNamespace(callbacks=callbacks)
    original_list = cli.trainer.callbacks
    cli.before_predict()

    assert cli.parent_before_predict_called is True
    assert cli.trainer.callbacks is not original_list
    assert cli.trainer.callbacks == [rich, checkpoint, geobwer]
    output = capsys.readouterr().out
    assert '"terratorch_custom_writer_count": 0' in output
    assert '"geobwer_probability_writer_count": 1' in output


def test_main_temporarily_installs_wrapper_then_restores_terratorch_cli(monkeypatch):
    callbacks, rich, checkpoint, geobwer = _callback_fixture()

    class OriginalCLI:
        pass

    observed: dict[str, object] = {}

    def build_lightning_cli(*, args, run):
        observed["args"] = args
        observed["run"] = run
        observed["patched_cli"] = cli_tools.MyLightningCLI
        cli = cli_tools.MyLightningCLI()
        cli.trainer = SimpleNamespace(callbacks=callbacks)
        cli.before_predict()
        observed["callbacks"] = cli.trainer.callbacks
        return cli

    cli_tools = SimpleNamespace(
        MyLightningCLI=OriginalCLI,
        build_lightning_cli=build_lightning_cli,
    )
    terratorch = ModuleType("terratorch")
    terratorch.cli_tools = cli_tools
    monkeypatch.setitem(sys.modules, "terratorch", terratorch)

    main(["predict", "-c", "predict.yaml", "--ckpt_path", "model.ckpt"])

    assert observed["args"][0] == "predict"
    assert observed["run"] is True
    assert observed["patched_cli"] is not OriginalCLI
    assert observed["callbacks"] == [rich, checkpoint, geobwer]
    assert cli_tools.MyLightningCLI is OriginalCLI


@pytest.mark.skipif(
    importlib.util.find_spec("terratorch") is None
    or importlib.util.find_spec("lightning") is None,
    reason="Exact TerraTorch callback integration requires the frozen Colab runtime.",
)
def test_exact_terratorch_1210_callback_and_mapping_writer_path(tmp_path: Path):
    import terratorch
    from terratorch.cli_tools import CustomWriter

    from rsfm_fairness_audit.terratorch_exports import GeoBWERProbabilityWriter

    assert terratorch.__version__ == "1.2.10"
    output = tmp_path / "probabilities"
    geobwer = GeoBWERProbabilityWriter(output_dir=str(output), write_interval="batch")
    callbacks = [
        _callback("lightning.pytorch.callbacks.progress.rich_progress", "RichProgressBar"),
        geobwer,
        CustomWriter(write_interval="batch"),
    ]
    retained, _ = filter_geobwer_predict_callbacks(callbacks)
    assert sum(isinstance(callback, CustomWriter) for callback in retained) == 0
    assert sum(isinstance(callback, GeoBWERProbabilityWriter) for callback in retained) == 1

    trainer = SimpleNamespace(global_rank=0, world_size=1)

    class FakeModel:
        pass

    geobwer.on_predict_start(trainer, FakeModel())
    probabilities = np.asarray(
        [[[[0.8, 0.2], [0.7, 0.1]], [[0.2, 0.8], [0.3, 0.9]]]],
        dtype=np.float32,
    )
    target = np.asarray([[[0, 1], [0, 1]]], dtype=np.int64)
    prediction = {
        "probabilities": probabilities,
        "target": target,
        "filename": ["event-a.tif"],
    }
    geobwer.write_on_batch_end(
        trainer,
        FakeModel(),
        prediction,
        None,
        {"mask": target, "filename": ["event-a.tif"]},
        0,
        0,
    )

    assert list(output.glob("writer_manifest_rank_*.json"))
    [index_path] = list((output / "index_parts").glob("*.jsonl"))
    row = json.loads(index_path.read_text(encoding="utf-8").strip())
    with np.load(output / row["probability_path"]) as artifact:
        assert artifact["probabilities"].shape == (2, 2, 2)
        np.testing.assert_array_equal(artifact["target"], target[0])
