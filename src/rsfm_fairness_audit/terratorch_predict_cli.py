from __future__ import annotations

import json
import sys
from typing import Any, Sequence


TERRATORCH_DEFAULT_WRITER = ("terratorch.cli_tools", "CustomWriter")
GEOBWER_PROBABILITY_WRITER = (
    "rsfm_fairness_audit.terratorch_exports",
    "GeoBWERProbabilityWriter",
)


class TerraTorchPredictCLIError(RuntimeError):
    """Raised when the frozen TerraTorch prediction callback contract drifts."""


def _callback_identity(callback: Any) -> tuple[str, str]:
    callback_type = callback.__class__
    return callback_type.__module__, callback_type.__name__


def filter_geobwer_predict_callbacks(
    callbacks: Sequence[Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Remove only TerraTorch's incompatible default prediction writer.

    TerraTorch 1.2.10 injects ``terratorch.cli_tools.CustomWriter`` through
    ``trainer_defaults`` after parsing the configured callbacks.  Its writer
    accepts tensor/tuple predictions, whereas GeoBWER deliberately returns a
    mapping containing full probabilities, targets, and filenames.

    This function constructs a new list before ``trainer.predict`` starts.  It
    never mutates the callback collection while Lightning is iterating it.
    """

    before = list(callbacks)
    default_writers = [
        callback
        for callback in before
        if _callback_identity(callback) == TERRATORCH_DEFAULT_WRITER
    ]
    retained = [
        callback
        for callback in before
        if _callback_identity(callback) != TERRATORCH_DEFAULT_WRITER
    ]
    geobwer_writers = [
        callback
        for callback in retained
        if _callback_identity(callback) == GEOBWER_PROBABILITY_WRITER
    ]
    if len(default_writers) != 1:
        raise TerraTorchPredictCLIError(
            "Expected exactly one TerraTorch 1.2.10 default CustomWriter before "
            f"GeoBWER prediction, observed={len(default_writers)}. Refusing an "
            "unverified callback lifecycle."
        )
    if len(geobwer_writers) != 1:
        raise TerraTorchPredictCLIError(
            "GeoBWER prediction requires exactly one GeoBWERProbabilityWriter "
            f"after filtering, observed={len(geobwer_writers)}."
        )
    report = {
        "schema": "geobwer.terratorch_predict_callback_filter.v1",
        "lifecycle": "after_cli_instantiation_before_trainer_predict",
        "removed": [
            f"{callback.__class__.__module__}.{callback.__class__.__name__}"
            for callback in default_writers
        ],
        "retained": [
            f"{callback.__class__.__module__}.{callback.__class__.__name__}"
            for callback in retained
        ],
        "terratorch_custom_writer_count": 0,
        "geobwer_probability_writer_count": 1,
    }
    return retained, report


def _geobwer_cli_class(base_cli: type) -> type:
    class GeoBWERPredictLightningCLI(base_cli):
        def before_predict(self) -> None:
            parent = getattr(super(), "before_predict", None)
            if callable(parent):
                parent()
            retained, report = filter_geobwer_predict_callbacks(
                tuple(self.trainer.callbacks)
            )
            self.trainer.callbacks = retained
            print(
                "[geobwer:terratorch-predict] " + json.dumps(report, sort_keys=True),
                flush=True,
            )

    GeoBWERPredictLightningCLI.__name__ = "GeoBWERPredictLightningCLI"
    GeoBWERPredictLightningCLI.__qualname__ = "GeoBWERPredictLightningCLI"
    return GeoBWERPredictLightningCLI


def main(args: Sequence[str] | None = None) -> None:
    """Run TerraTorch with a GeoBWER-safe predict callback lifecycle."""

    try:
        from terratorch import cli_tools
    except ImportError as exc:  # pragma: no cover - exercised in the Colab runtime
        raise TerraTorchPredictCLIError(
            "The frozen TerraTorch runtime is required for GeoBWER prediction."
        ) from exc

    original_cli = cli_tools.MyLightningCLI
    cli_tools.MyLightningCLI = _geobwer_cli_class(original_cli)
    try:
        cli_tools.build_lightning_cli(
            args=list(sys.argv[1:] if args is None else args),
            run=True,
        )
    finally:
        cli_tools.MyLightningCLI = original_cli


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    main()
