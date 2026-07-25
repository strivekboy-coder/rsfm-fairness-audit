from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.fmow_sentinel_classification import (
    build_resnet50_multiband,
)
from rsfm_fairness_audit.reben_resnet50_campaign import (
    MODE_CHANNELS,
    RebenResNet50Config,
    _TorchDataset,
    _device,
    _loader,
    _require_torch,
)


class RebenDataBenchmarkError(RuntimeError):
    """Raised when a loader candidate changes the deterministic input stream."""


def _update_array_digest(digest: Any, value: Any) -> None:
    array = value.detach().cpu().contiguous().numpy()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))


def benchmark_reben_loader_workers(
    adapter: Any,
    contract: Mapping[str, Any],
    config: RebenResNet50Config,
    *,
    mode: str,
    worker_counts: Sequence[int] = (0, 2, 4, 8),
    max_batches: int = 150,
    checksum_batches: int = 100,
    warmup_batches: int = 5,
    seed: int = 42,
    reference_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark worker counts and certify that their input streams agree.

    This is intentionally a bounded forward-only workload. It does not update
    model weights, choose thresholds, or run GeoBWER.
    """

    if mode not in MODE_CHANNELS:
        raise ValueError(f"Unsupported mode: {mode}")
    if max_batches <= 0 or checksum_batches <= 0:
        raise ValueError("max_batches and checksum_batches must be positive.")
    if checksum_batches > max_batches:
        raise ValueError("checksum_batches cannot exceed max_batches.")
    workers = tuple(dict.fromkeys(int(value) for value in worker_counts))
    if not workers or workers[0] != 0 or any(value < 0 for value in workers):
        raise ValueError("worker_counts must start with 0 and be non-negative.")

    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(config.device)
    model = build_resnet50_multiband(
        19,
        in_channels=MODE_CHANNELS[mode],
        pretrained=False,
    ).to(device)
    checkpoint_source = "deterministic_benchmark_reference"
    checkpoint_path = (
        Path(reference_checkpoint) if reference_checkpoint is not None else None
    )
    if checkpoint_path is not None and checkpoint_path.is_file():
        try:
            checkpoint_payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:  # pragma: no cover - older Colab torch
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        state = (
            checkpoint_payload.get("model_state_dict")
            if isinstance(checkpoint_payload, Mapping)
            else None
        )
        if not isinstance(state, Mapping):
            raise RebenDataBenchmarkError(
                f"Reference checkpoint has no model_state_dict: {checkpoint_path}"
            )
        model.load_state_dict(state, strict=True)
        checkpoint_source = "existing_fixed_checkpoint"
    elif checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "geobwer.reben.loader_benchmark_reference.v1",
                "formal_evidence": False,
                "sensor_mode": mode,
                "seed": seed,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        try:
            checkpoint_payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:  # pragma: no cover - older Colab torch
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(
            checkpoint_payload["model_state_dict"],
            strict=True,
        )
        checkpoint_source = "persisted_deterministic_benchmark_reference"
    model.eval()
    model_state_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        model_state_digest.update(name.encode("utf-8"))
        _update_array_digest(model_state_digest, tensor)

    dataset = _TorchDataset(adapter, mode, contract)
    baseline: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    for num_workers in workers:
        candidate_config = replace(config, num_workers=num_workers)
        image_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        sample_digest = hashlib.sha256()
        first_logits: np.ndarray | None = None
        correctness_loader = _loader(
            dataset, candidate_config, shuffle=False, seed=seed
        )
        correctness_iterator = iter(correctness_loader)
        correctness_batches_seen = 0
        try:
            with torch.inference_mode():
                while correctness_batches_seen < checksum_batches:
                    try:
                        images, labels, indices = next(correctness_iterator)
                    except StopIteration:
                        break
                    _update_array_digest(image_digest, images)
                    _update_array_digest(label_digest, labels)
                    for index in indices.tolist():
                        sample_digest.update(
                            str(dataset.rows[int(index)]["sample_id"]).encode(
                                "utf-8"
                            )
                        )
                        sample_digest.update(b"\0")
                    if first_logits is None:
                        first_logits = (
                            model(
                                images.to(
                                    device,
                                    non_blocking=(
                                        candidate_config.host_to_device_non_blocking
                                    ),
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    correctness_batches_seen += 1
        finally:
            shutdown = getattr(correctness_iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()

        loader = _loader(dataset, candidate_config, shuffle=False, seed=seed)
        iterator = iter(loader)
        total_samples = 0
        timed_samples = 0
        data_wait_seconds = 0.0
        step_seconds = 0.0
        batches = 0
        try:
            with torch.inference_mode():
                while batches < max_batches:
                    wait_started = time.perf_counter()
                    try:
                        images, labels, indices = next(iterator)
                    except StopIteration:
                        break
                    waited = time.perf_counter() - wait_started
                    compute_started = time.perf_counter()
                    images_device = images.to(
                        device,
                        non_blocking=candidate_config.host_to_device_non_blocking,
                    )
                    logits = model(images_device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    computed = time.perf_counter() - compute_started
                    batch_samples = int(images.shape[0])
                    total_samples += batch_samples
                    if batches >= warmup_batches:
                        timed_samples += batch_samples
                        data_wait_seconds += waited
                        step_seconds += waited + computed
                    batches += 1
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()

        if (
            batches <= warmup_batches
            or first_logits is None
            or correctness_batches_seen < checksum_batches
        ):
            raise RebenDataBenchmarkError(
                "Not enough batches for benchmark/checksum: "
                f"timing={batches}, checksum={correctness_batches_seen}."
            )
        observation = {
            "num_workers": num_workers,
            "batches": batches,
            "checksum_batches": correctness_batches_seen,
            "samples": total_samples,
            "timed_samples": timed_samples,
            "samples_per_second": timed_samples / step_seconds,
            "data_wait_seconds": data_wait_seconds,
            "data_wait_seconds_per_batch": data_wait_seconds
            / (batches - warmup_batches),
            "step_seconds": step_seconds,
            "step_seconds_per_batch": step_seconds / (batches - warmup_batches),
            "image_checksum": image_digest.hexdigest(),
            "label_checksum": label_digest.hexdigest(),
            "sample_id_checksum": sample_digest.hexdigest(),
            "first_logits": first_logits,
        }
        if baseline is None:
            baseline = observation
            observation["correctness"] = {
                "sample_order_equal": True,
                "labels_equal": True,
                "inputs_equal": True,
                "forward_allclose": True,
                "max_abs_forward_difference": 0.0,
            }
        else:
            max_abs = float(
                np.max(
                    np.abs(
                        observation["first_logits"]
                        - baseline["first_logits"]
                    )
                )
            )
            correctness = {
                "sample_order_equal": (
                    observation["sample_id_checksum"]
                    == baseline["sample_id_checksum"]
                ),
                "labels_equal": (
                    observation["label_checksum"]
                    == baseline["label_checksum"]
                ),
                "inputs_equal": (
                    observation["image_checksum"]
                    == baseline["image_checksum"]
                ),
                "forward_allclose": bool(
                    np.allclose(
                        observation["first_logits"],
                        baseline["first_logits"],
                        rtol=1e-5,
                        atol=1e-6,
                    )
                ),
                "max_abs_forward_difference": max_abs,
            }
            observation["correctness"] = correctness
            if not all(
                correctness[key]
                for key in (
                    "sample_order_equal",
                    "labels_equal",
                    "inputs_equal",
                    "forward_allclose",
                )
            ):
                raise RebenDataBenchmarkError(
                    f"num_workers={num_workers} changed the input or forward stream: "
                    f"{correctness}"
                )
        observation.pop("first_logits")
        results.append(observation)

    peak_throughput = max(float(row["samples_per_second"]) for row in results)
    # Prefer the smallest worker pool within 5% of the observed peak. This
    # avoids selecting eight workers for noise-level gains and limits RAM/file
    # descriptor pressure in a long Colab campaign.
    recommended = min(
        (
            row
            for row in results
            if float(row["samples_per_second"]) >= 0.95 * peak_throughput
        ),
        key=lambda row: int(row["num_workers"]),
    )
    return {
        "schema": "geobwer.reben.loader_benchmark.v1",
        "formal_evidence": False,
        "training_performed": False,
        "sensor_mode": mode,
        "device": str(device),
        "model_state_sha256": model_state_digest.hexdigest(),
        "reference_checkpoint": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        "reference_checkpoint_source": checkpoint_source,
        "max_batches": max_batches,
        "checksum_batches": checksum_batches,
        "warmup_batches": warmup_batches,
        "results": results,
        "recommended_num_workers": int(recommended["num_workers"]),
        "recommendation_policy": "smallest_worker_count_within_5pct_of_peak",
        "recommended_samples_per_second": float(
            recommended["samples_per_second"]
        ),
    }


__all__ = ["RebenDataBenchmarkError", "benchmark_reben_loader_workers"]
