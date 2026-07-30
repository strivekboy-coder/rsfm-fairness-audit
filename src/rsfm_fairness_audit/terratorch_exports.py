from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np


class TerraTorchExportError(RuntimeError):
    """Raised when a formal TerraTorch probability export is incomplete."""


def segmentation_probabilities_from_logits(logits: Any) -> Any:
    """Validate raw segmentation logits and retain every class probability.

    TerraTorch's ``select_classes`` is an inference presentation hook.  With
    ``output_on_inference='prediction'`` it returns an argmax tuple rather than
    logits, so it must never participate in a lossless audit export.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Colab runtime requires torch
        raise TerraTorchExportError(
            "PyTorch is required to convert segmentation logits to probabilities."
        ) from exc
    if not torch.is_tensor(logits):
        raise TerraTorchExportError(
            "Segmentation model output must be the raw logits tensor [N,K,H,W]; "
            f"got {type(logits).__name__}."
        )
    if logits.ndim != 4 or logits.shape[1] < 2:
        raise TerraTorchExportError(
            f"Segmentation logits must be [N,K,H,W] with K>=2, got {tuple(logits.shape)}."
        )
    return torch.softmax(logits, dim=1)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _batch_values(value: Any, batch_size: int) -> list[Any]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, (str, Path)):
        return [str(value)] * batch_size
    if isinstance(value, Mapping):
        columns = {str(key): _batch_values(item, batch_size) for key, item in value.items()}
        return [{key: values[index] for key, values in columns.items()} for index in range(batch_size)]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()] * batch_size
        if len(value) != batch_size:
            raise TerraTorchExportError(
                f"Batch metadata has {len(value)} rows but probability output has {batch_size}."
            )
        return [value[index] for index in range(batch_size)]
    if isinstance(value, Sequence):
        if len(value) != batch_size:
            raise TerraTorchExportError(
                f"Batch metadata has {len(value)} rows but probability output has {batch_size}."
            )
        return list(value)
    return [value] * batch_size


def _canonical_filename(raw: Any) -> str:
    if isinstance(raw, Mapping):
        return json.dumps({str(key): str(value) for key, value in raw.items()}, sort_keys=True)
    return str(raw) if raw not in (None, "") else ""


def _stable_sample_id(raw: Any, batch_idx: int, row_idx: int) -> str:
    source = _canonical_filename(raw) or f"batch-{batch_idx:06d}-row-{row_idx:04d}"
    stem = Path(source).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-.") or "sample"
    suffix = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:80]}-{suffix}"


def write_probability_batch(
    output_dir: str | Path,
    *,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    batch_idx: int,
    dataloader_idx: int = 0,
    metadata_keys: Sequence[str] = (
        "event_id",
        "location_id",
        "country",
        "country_code",
        "region",
        "spatial_block",
    ),
) -> list[dict[str, Any]]:
    """Write one lossless NPZ per independent prediction unit.

    The function is framework-neutral and is used by the Lightning callback
    below.  One file per unit avoids keeping dense segmentation probability
    maps in RAM and makes interrupted Colab prediction resumable.
    """

    if "probabilities" not in outputs:
        raise TerraTorchExportError("Prediction output has no 'probabilities' tensor.")
    probabilities = _as_numpy(outputs["probabilities"]).astype(np.float32, copy=False)
    if probabilities.ndim < 2:
        raise TerraTorchExportError(f"Expected probabilities [N,...], got {probabilities.shape}.")
    if not np.isfinite(probabilities).all():
        raise TerraTorchExportError("Probability output contains NaN or infinity.")
    if np.min(probabilities) < -1e-6 or np.max(probabilities) > 1.0 + 1e-6:
        raise TerraTorchExportError("Probability output is outside [0,1]; logits were likely exported by mistake.")

    batch_size = int(probabilities.shape[0])
    filenames = _batch_values(outputs.get("filename", batch.get("filename")), batch_size)
    target_source = outputs.get("target")
    if target_source is None:
        target_source = batch.get("mask", batch.get("label"))
    targets = _batch_values(target_source, batch_size)
    if all(target is None for target in targets):
        raise TerraTorchExportError(
            "Formal export requires labels/masks in the prediction batch; use the labeled test split, not an unlabeled predict root."
        )

    root = Path(output_dir)
    sample_dir = root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata_columns = {
        key: _batch_values(outputs.get(key, batch.get(key)), batch_size)
        for key in metadata_keys
        if outputs.get(key, batch.get(key)) is not None
    }
    rows: list[dict[str, Any]] = []
    for row_idx in range(batch_size):
        sample_id = _stable_sample_id(filenames[row_idx], batch_idx, row_idx)
        canonical_filename = _canonical_filename(filenames[row_idx])
        path = sample_dir / f"{sample_id}.npz"
        target = _as_numpy(targets[row_idx])
        if target.size == 0:
            raise TerraTorchExportError(f"Empty target for sample {sample_id}.")
        np.savez_compressed(
            path,
            probabilities=probabilities[row_idx],
            target=target,
            filename=np.asarray(canonical_filename),
            batch_idx=np.asarray(int(batch_idx)),
            dataloader_idx=np.asarray(int(dataloader_idx)),
        )
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "filename": canonical_filename,
            # A relative path is deliberate: Colab writes to /content and the
            # completed export is then mirrored to Drive.  Absolute /content
            # paths would make an otherwise valid archive non-portable.
            "probability_path": path.relative_to(root).as_posix(),
            "probability_shape": json.dumps(list(probabilities[row_idx].shape)),
            "target_shape": json.dumps(list(target.shape)),
            "batch_idx": int(batch_idx),
            "dataloader_idx": int(dataloader_idx),
        }
        for key, values in metadata_columns.items():
            value = values[row_idx]
            if isinstance(value, np.ndarray):
                value = value.item() if value.ndim == 0 else json.dumps(value.tolist())
            row[key] = value
        rows.append(row)
    return rows


def mean_impute_and_normalize_tensor(
    image: Any,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> Any:
    """Apply frozen-mean imputation to one TerraTorch modality tensor."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime requires torch
        raise TerraTorchExportError(
            "PyTorch is required for TerraMind input normalization."
        ) from exc
    if not torch.is_tensor(image) or image.ndim not in {3, 4, 5}:
        raise TerraTorchExportError(
            "TerraMind modality input must be a tensor [C,H,W], [B,C,H,W], "
            f"or [B,C,T,H,W], got {type(image).__name__} "
            f"{getattr(image, 'shape', None)}."
        )
    channel_axis = 0 if image.ndim == 3 else 1
    if int(image.shape[channel_axis]) != len(mean) or len(mean) != len(std):
        raise TerraTorchExportError(
            "Frozen TerraMind statistics do not match the modality channels: "
            f"shape={tuple(image.shape)}, mean={len(mean)}, std={len(std)}."
        )
    shape = [1] * int(image.ndim)
    shape[channel_axis] = len(mean)
    mean_tensor = torch.as_tensor(
        mean,
        dtype=image.dtype,
        device=image.device,
    ).reshape(shape)
    std_tensor = torch.clamp(
        torch.as_tensor(
            std,
            dtype=image.dtype,
            device=image.device,
        ).reshape(shape),
        min=1e-6,
    )
    imputed = torch.where(torch.isfinite(image), image, mean_tensor)
    output = (imputed - mean_tensor) / std_tensor
    if not bool(torch.isfinite(output).all().item()):
        raise TerraTorchExportError(
            "TerraMind modality remains non-finite after frozen-mean imputation."
        )
    return output


try:  # Optional Colab-only runtime; pure helpers above remain locally testable.
    import torch
    from lightning.pytorch.callbacks import BasePredictionWriter, Callback
    from terratorch.datamodules import GenericMultiModalDataModule
    from terratorch.tasks import ClassificationTask, MultiLabelClassificationTask, SemanticSegmentationTask

except ImportError:  # pragma: no cover - exercised when TerraTorch is not installed

    class _MissingTerraTorch:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise TerraTorchExportError(
                "This class requires torch, lightning, and the official TerraTorch runtime. "
                "Install the frozen Colab dependencies before constructing it."
            )

    GeoBWERClassificationTask = _MissingTerraTorch
    GeoBWERMultiLabelClassificationTask = _MissingTerraTorch
    GeoBWERSemanticSegmentationTask = _MissingTerraTorch
    GeoBWERProbabilityWriter = _MissingTerraTorch
    PersistentCheckpointMirror = _MissingTerraTorch
    TerraMindOperationalMonitor = _MissingTerraTorch
    GeoBWERSen1DataModule = _MissingTerraTorch
    LabeledTestAsPredictDataModule = _MissingTerraTorch
    LabeledValidationAsPredictDataModule = _MissingTerraTorch

else:  # pragma: no cover - executed in the Colab model runtime

    class _MeanImputingMultimodalNormalize:
        """Replace non-finite EO values with frozen means before normalization."""

        def __init__(
            self,
            means: Mapping[str, Sequence[float]],
            stds: Mapping[str, Sequence[float]],
        ) -> None:
            self.means = {
                str(key): tuple(float(value) for value in values)
                for key, values in means.items()
            }
            self.stds = {
                str(key): tuple(float(value) for value in values)
                for key, values in stds.items()
            }

        def __call__(self, batch: Any, denormalize: bool = False) -> Any:
            for modality in self.means:
                if modality in batch:
                    image = batch[modality]
                    container = batch
                elif (
                    "image" in batch
                    and isinstance(batch["image"], Mapping)
                    and modality in batch["image"]
                ):
                    image = batch["image"][modality]
                    container = batch["image"]
                else:
                    continue
                if denormalize:
                    channel_axis = 1 if image.ndim >= 4 else 0
                    shape = [1] * int(image.ndim)
                    shape[channel_axis] = len(self.means[modality])
                    mean = torch.as_tensor(
                        self.means[modality],
                        dtype=image.dtype,
                        device=image.device,
                    ).reshape(shape)
                    std = torch.as_tensor(
                        self.stds[modality],
                        dtype=image.dtype,
                        device=image.device,
                    ).reshape(shape)
                    output = image * std + mean
                else:
                    # TerraTorch 1.2.10's dataset-level no_data_replace is a
                    # scalar raw-value replacement. Keep it disabled and apply
                    # the frozen per-band mean here so every imputed normalized
                    # value is exactly zero.
                    output = mean_impute_and_normalize_tensor(
                        image,
                        mean=self.means[modality],
                        std=self.stds[modality],
                    )
                if not bool(torch.isfinite(output).all().item()):
                    raise TerraTorchExportError(
                        "TerraMind input remains non-finite after frozen-mean "
                        f"imputation for modality={modality}."
                    )
                container[modality] = output
            return batch


    class GeoBWERSen1DataModule(GenericMultiModalDataModule):
        """TerraTorch datamodule with the frozen Sen1 mean-imputation contract."""

        def __init__(
            self,
            *args: Any,
            means: Mapping[str, Sequence[float]],
            stds: Mapping[str, Sequence[float]],
            persistent_workers: bool = False,
            prefetch_factor: int = 2,
            **kwargs: Any,
        ) -> None:
            if kwargs.get("no_data_replace") is not None:
                raise TerraTorchExportError(
                    "GeoBWERSen1DataModule requires no_data_replace=None; "
                    "scalar raw-value replacement would violate the frozen "
                    "per-band mean-imputation contract."
                )
            num_workers = int(kwargs.get("num_workers", 0))
            if int(prefetch_factor) <= 0:
                raise TerraTorchExportError(
                    "prefetch_factor must be positive."
                )
            self.geobwer_persistent_workers = bool(persistent_workers)
            self.geobwer_prefetch_factor = int(prefetch_factor)
            super().__init__(*args, means=means, stds=stds, **kwargs)
            self.aug = _MeanImputingMultimodalNormalize(means, stds)

        def _dataloader_factory(self, split: str):
            """Add loader-only tuning absent from TerraTorch 1.2.10.

            The upstream loader has already frozen the dataset, deterministic
            sampler, batch sampler, and collate function. Rebuilding only the
            DataLoader wrapper avoids copying the upstream dataset-selection
            implementation or changing any scientific sample semantics.
            """

            upstream = super()._dataloader_factory(split)
            kwargs: dict[str, Any] = {
                "dataset": upstream.dataset,
                "batch_sampler": upstream.batch_sampler,
                "num_workers": int(self.num_workers),
                "collate_fn": upstream.collate_fn,
                "pin_memory": bool(self.pin_memory),
                "persistent_workers": bool(
                    self.geobwer_persistent_workers
                    and int(self.num_workers) > 0
                ),
            }
            if int(self.num_workers) > 0:
                kwargs["prefetch_factor"] = int(
                    self.geobwer_prefetch_factor
                )
            return torch.utils.data.DataLoader(**kwargs)

    def _tensor_devices(value: Any) -> set[str]:
        if isinstance(value, Mapping):
            devices: set[str] = set()
            for item in value.values():
                devices.update(_tensor_devices(item))
            return devices
        if isinstance(value, (list, tuple)):
            devices = set()
            for item in value:
                devices.update(_tensor_devices(item))
            return devices
        device = getattr(value, "device", None)
        return {str(device)} if device is not None else set()


    class _RuntimeDeviceAuditMixin:
        """Print the actual model/input device once per Lightning stage."""

        def _geobwer_log_runtime(self, stage: str, batch: Mapping[str, Any] | None = None) -> None:
            logged = getattr(self, "_geobwer_logged_device_stages", set())
            if stage in logged:
                return
            first_parameter = next(iter(self.parameters()), None)
            parameter_device = str(first_parameter.device) if first_parameter is not None else "no_parameters"
            input_devices = sorted(_tensor_devices(batch.get("image"))) if batch is not None else []
            cuda_name = ""
            if torch.cuda.is_available():
                cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(
                "[geobwer:runtime] "
                f"stage={stage} cuda_available={torch.cuda.is_available()} gpu={cuda_name or 'none'} "
                f"parameter_device={parameter_device} input_devices={input_devices or ['not_yet_available']}",
                flush=True,
            )
            logged = set(logged)
            logged.add(stage)
            self._geobwer_logged_device_stages = logged

        def on_fit_start(self) -> None:
            parent = getattr(super(), "on_fit_start", None)
            if callable(parent):
                parent()
            self._geobwer_log_runtime("fit")

        def training_step(self, batch: Mapping[str, Any], *args: Any, **kwargs: Any):
            self._geobwer_log_runtime("train_batch", batch)
            return super().training_step(batch, *args, **kwargs)

    class PersistentCheckpointMirror(Callback):
        """Mirror resumable checkpoints from local Colab storage to Drive.

        Training remains on the fast ephemeral filesystem.  At a bounded epoch
        interval the completed checkpoint files are copied atomically to the
        persistent cache.  The campaign hydrates these files before a resumed
        fit, so Drive is never used as Lightning's live checkpoint directory.
        """

        def __init__(
            self,
            source_dir: str,
            persistent_dir: str,
            *,
            every_n_epochs: int = 10,
        ) -> None:
            super().__init__()
            if int(every_n_epochs) <= 0:
                raise TerraTorchExportError("every_n_epochs must be positive.")
            self.source_dir = Path(source_dir)
            self.persistent_dir = Path(persistent_dir)
            self.every_n_epochs = int(every_n_epochs)

        def _mirror(self, trainer: Any, *, include_best: bool) -> None:
            if not bool(getattr(trainer, "is_global_zero", True)):
                return
            started = time.perf_counter()
            self.persistent_dir.mkdir(parents=True, exist_ok=True)
            sources = [self.source_dir / "last.ckpt"]
            if include_best:
                sources.extend(sorted(self.source_dir.glob("best-*.ckpt")))
            copied = 0
            for source in sources:
                if not source.is_file():
                    continue
                destination = self.persistent_dir / source.name
                if destination.exists():
                    source_stat = source.stat()
                    destination_stat = destination.stat()
                    if (
                        destination_stat.st_size == source_stat.st_size
                        and destination_stat.st_mtime_ns >= source_stat.st_mtime_ns
                    ):
                        continue
                temporary = destination.with_suffix(destination.suffix + ".partial")
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
                copied += 1
            removed = 0
            if include_best:
                active_best = {
                    source.name
                    for source in sources
                    if source.is_file() and source.name.startswith("best-")
                }
                for stale in self.persistent_dir.glob("best-*.ckpt"):
                    if stale.name not in active_best:
                        stale.unlink()
                        removed += 1
            print(
                f"[terramind:checkpoint-mirror] epoch={int(getattr(trainer, 'current_epoch', -1))} "
                f"include_best={include_best} copied={copied} removed_stale_best={removed} "
                f"seconds={time.perf_counter() - started:.3f} "
                f"persistent_dir={self.persistent_dir}",
                flush=True,
            )

        def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
            epoch = int(getattr(trainer, "current_epoch", 0)) + 1
            if epoch % self.every_n_epochs == 0:
                # A resumable interval needs only last.ckpt. Copying every
                # transient best checkpoint wastes Drive bandwidth and leaves
                # multiple ambiguous best-* files after hydration.
                self._mirror(trainer, include_best=False)

        def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
            self._mirror(trainer, include_best=True)


    class TerraMindOperationalMonitor(Callback):
        """Low-overhead timing, throughput, and GPU observability callback."""

        def __init__(
            self,
            *,
            sensor_mode: str,
            seed: int,
            stage: str,
            log_path: str,
            gpu_log_every_n_steps: int = 20,
        ) -> None:
            super().__init__()
            if int(gpu_log_every_n_steps) <= 0:
                raise TerraTorchExportError(
                    "gpu_log_every_n_steps must be positive."
                )
            self.sensor_mode = str(sensor_mode)
            self.seed = int(seed)
            self.stage = str(stage)
            self.log_path = Path(log_path)
            self.gpu_log_every_n_steps = int(gpu_log_every_n_steps)
            self._stage_started: float | None = None
            self._epoch_started: float | None = None
            self._epoch_samples = 0
            self._validation_started: float | None = None
            self._validation_samples = 0
            self._predict_samples = 0

        @staticmethod
        def _batch_size(batch: Any) -> int:
            if isinstance(batch, Mapping):
                for key in ("mask", "label"):
                    value = batch.get(key)
                    if hasattr(value, "shape") and len(value.shape):
                        return int(value.shape[0])
                image = batch.get("image")
                if isinstance(image, Mapping):
                    for value in image.values():
                        if hasattr(value, "shape") and len(value.shape):
                            return int(value.shape[0])
                if hasattr(image, "shape") and len(image.shape):
                    return int(image.shape[0])
            return 0

        @staticmethod
        def _gpu_snapshot() -> dict[str, Any]:
            if not torch.cuda.is_available():
                return {"cuda_available": False}
            device = torch.cuda.current_device()
            snapshot: dict[str, Any] = {
                "cuda_available": True,
                "device": f"cuda:{device}",
                "gpu_name": torch.cuda.get_device_name(device),
                "memory_allocated_bytes": int(
                    torch.cuda.memory_allocated(device)
                ),
                "memory_reserved_bytes": int(
                    torch.cuda.memory_reserved(device)
                ),
            }
            utilization = getattr(torch.cuda, "utilization", None)
            if callable(utilization):
                try:
                    snapshot["utilization_percent"] = int(
                        utilization(device)
                    )
                except Exception:
                    snapshot["utilization_percent"] = None
            return snapshot

        def _record(self, event: str, **values: Any) -> None:
            payload = {
                "schema": "geobwer.terramind.operational_event.v1",
                "event": event,
                "sensor_mode": self.sensor_mode,
                "seed": self.seed,
                "stage": self.stage,
                "monotonic_seconds": time.perf_counter(),
                **values,
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                "[terramind:operational] "
                + " ".join(
                    f"{key}={value}"
                    for key, value in payload.items()
                    if key not in {"schema", "monotonic_seconds"}
                ),
                flush=True,
            )

        def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
            self._stage_started = time.perf_counter()
            self._record("fit_start", gpu=self._gpu_snapshot())

        def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
            elapsed = (
                time.perf_counter() - self._stage_started
                if self._stage_started is not None
                else None
            )
            self._record("fit_end", elapsed_seconds=elapsed, gpu=self._gpu_snapshot())

        def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
            self._epoch_started = time.perf_counter()
            self._epoch_samples = 0
            self._record(
                "train_epoch_start",
                epoch=int(getattr(trainer, "current_epoch", -1)) + 1,
            )

        def on_train_batch_end(
            self,
            trainer: Any,
            pl_module: Any,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            self._epoch_samples += self._batch_size(batch)
            if (int(batch_idx) + 1) % self.gpu_log_every_n_steps == 0:
                self._record(
                    "gpu_snapshot",
                    epoch=int(getattr(trainer, "current_epoch", -1)) + 1,
                    batch_index=int(batch_idx),
                    gpu=self._gpu_snapshot(),
                )

        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            elapsed = (
                time.perf_counter() - self._epoch_started
                if self._epoch_started is not None
                else 0.0
            )
            self._record(
                "train_epoch_end",
                epoch=int(getattr(trainer, "current_epoch", -1)) + 1,
                elapsed_seconds=elapsed,
                samples=self._epoch_samples,
                samples_per_second=(
                    self._epoch_samples / elapsed if elapsed > 0 else None
                ),
                gpu=self._gpu_snapshot(),
            )

        def on_validation_epoch_start(
            self, trainer: Any, pl_module: Any
        ) -> None:
            self._validation_started = time.perf_counter()
            self._validation_samples = 0
            self._record(
                "validation_epoch_start",
                epoch=int(getattr(trainer, "current_epoch", -1)) + 1,
            )

        def on_validation_batch_end(
            self,
            trainer: Any,
            pl_module: Any,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> None:
            self._validation_samples += self._batch_size(batch)

        def on_validation_epoch_end(
            self, trainer: Any, pl_module: Any
        ) -> None:
            elapsed = (
                time.perf_counter() - self._validation_started
                if self._validation_started is not None
                else 0.0
            )
            self._record(
                "validation_epoch_end",
                epoch=int(getattr(trainer, "current_epoch", -1)) + 1,
                elapsed_seconds=elapsed,
                samples=self._validation_samples,
                samples_per_second=(
                    self._validation_samples / elapsed
                    if elapsed > 0
                    else None
                ),
            )

        def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
            self._stage_started = time.perf_counter()
            self._predict_samples = 0
            self._record("predict_start", gpu=self._gpu_snapshot())

        def on_predict_batch_end(
            self,
            trainer: Any,
            pl_module: Any,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> None:
            self._predict_samples += self._batch_size(batch)

        def on_predict_end(self, trainer: Any, pl_module: Any) -> None:
            elapsed = (
                time.perf_counter() - self._stage_started
                if self._stage_started is not None
                else 0.0
            )
            self._record(
                "predict_end",
                elapsed_seconds=elapsed,
                samples=self._predict_samples,
                samples_per_second=(
                    self._predict_samples / elapsed if elapsed > 0 else None
                ),
                gpu=self._gpu_snapshot(),
            )

    class LabeledTestAsPredictDataModule(GeoBWERSen1DataModule):
        """Expose the labeled, frozen test split to Lightning's predict loop.

        TerraTorch's generic predict dataset intentionally drops labels.  That
        is appropriate for deployment, but invalid for an audit export.  This
        datamodule reuses the exact test dataset (including its test split file
        and label root) during ``predict`` without changing the training or test
        implementations.
        """

        def setup(self, stage: str) -> None:
            if stage == "predict":
                super().setup("test")
                self.predict_dataset = self.test_dataset
                return
            super().setup(stage)


    class LabeledValidationAsPredictDataModule(GeoBWERSen1DataModule):
        """Expose the labeled validation split to predict for block calibration."""

        def setup(self, stage: str) -> None:
            if stage == "predict":
                super().setup("validate")
                self.predict_dataset = self.val_dataset
                return
            super().setup(stage)

    class GeoBWERClassificationTask(_RuntimeDeviceAuditMixin, ClassificationTask):
        """TerraTorch classification task whose predict step returns full softmax."""

        def predict_step(self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0):
            self._geobwer_log_runtime("predict_batch", batch)
            x = batch["image"]
            file_names = batch.get("filename")
            rest = {key: batch[key] for key in batch.keys() - {"image", "label", "filename"}}
            logits = self(x, **rest).output
            if logits.ndim != 2 or logits.shape[1] < 2:
                raise TerraTorchExportError(f"Classification logits must be [N,K] with K>=2, got {tuple(logits.shape)}.")
            return {
                "probabilities": torch.softmax(logits, dim=1),
                "target": batch.get("label"),
                "filename": file_names,
            }


    class GeoBWERMultiLabelClassificationTask(_RuntimeDeviceAuditMixin, MultiLabelClassificationTask):
        """TerraTorch multilabel task whose predict step returns full sigmoid."""

        def predict_step(self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0):
            self._geobwer_log_runtime("predict_batch", batch)
            x = batch["image"]
            file_names = batch.get("filename")
            rest = {key: batch[key] for key in batch.keys() - {"image", "label", "filename"}}
            logits = self(x, **rest).output
            if logits.ndim != 2:
                raise TerraTorchExportError(f"Multilabel logits must be [N,K], got {tuple(logits.shape)}.")
            return {
                "probabilities": torch.sigmoid(logits),
                "target": batch.get("label"),
                "filename": file_names,
            }


    class GeoBWERSemanticSegmentationTask(_RuntimeDeviceAuditMixin, SemanticSegmentationTask):
        """TerraTorch segmentation task whose predict step returns class probabilities."""

        def predict_step(self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0):
            self._geobwer_log_runtime("predict_batch", batch)
            x = batch["image"]
            file_names = batch.get("filename")
            rest = {key: batch[key] for key in batch.keys() - {"image", "mask", "filename"}}

            def model_forward(inputs: Any, **kwargs: Any):
                return self(inputs, **kwargs).output

            if self.tiled_inference_parameters:
                from terratorch.tasks.segmentation_tasks import tiled_inference

                logits = tiled_inference(model_forward, x, **self.tiled_inference_parameters, **rest)
            else:
                logits = model_forward(x, **rest)
            return {
                "probabilities": segmentation_probabilities_from_logits(logits),
                "target": batch.get("mask"),
                "filename": file_names,
            }


    class GeoBWERProbabilityWriter(BasePredictionWriter):
        """Streaming, resumable Lightning writer for formal GeoBWER inputs."""

        def __init__(
            self,
            output_dir: str,
            *,
            write_interval: str = "batch",
            metadata_keys: Sequence[str] = (
                "event_id",
                "location_id",
                "country",
                "country_code",
                "region",
                "spatial_block",
            ),
        ) -> None:
            if write_interval != "batch":
                raise TerraTorchExportError("GeoBWERProbabilityWriter requires write_interval='batch'.")
            super().__init__(write_interval=write_interval)
            self.output_dir = Path(output_dir)
            self.metadata_keys = tuple(metadata_keys)

        def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema": "geobwer.terratorch_probability_export.v2",
                "writer": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "task_class": f"{pl_module.__class__.__module__}.{pl_module.__class__.__name__}",
                "world_size": int(getattr(trainer, "world_size", 1)),
                "global_rank": int(getattr(trainer, "global_rank", 0)),
                "metadata_keys": list(self.metadata_keys),
                "probability_path_policy": "relative_to_export_root",
            }
            (self.output_dir / f"writer_manifest_rank_{manifest['global_rank']:03d}.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

        def write_on_batch_end(
            self,
            trainer: Any,
            pl_module: Any,
            prediction: Mapping[str, Any],
            batch_indices: Any,
            batch: Mapping[str, Any],
            batch_idx: int,
            dataloader_idx: int,
        ) -> None:
            rank = int(getattr(trainer, "global_rank", 0))
            rows = write_probability_batch(
                self.output_dir,
                outputs=prediction,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
                metadata_keys=self.metadata_keys,
            )
            index_dir = self.output_dir / "index_parts"
            index_dir.mkdir(parents=True, exist_ok=True)
            index_path = index_dir / (
                f"rank_{rank:03d}_loader_{int(dataloader_idx):02d}_batch_{int(batch_idx):08d}.jsonl"
            )
            # One deterministic file per batch makes prediction reruns idempotent:
            # a completed batch is overwritten rather than duplicated in an append-only index.
            with index_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
