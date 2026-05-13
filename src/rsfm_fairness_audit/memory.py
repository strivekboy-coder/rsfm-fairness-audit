from __future__ import annotations

import gc
import os
from typing import Any


def memory_snapshot(stage: str) -> dict[str, Any]:
    """Return a lightweight process/GPU memory snapshot without hard dependencies."""
    snapshot: dict[str, Any] = {"stage": stage}
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        snapshot["rss_mb"] = process.memory_info().rss / (1024**2)
    except Exception:
        snapshot["rss_mb"] = ""
    try:
        import torch

        if torch.cuda.is_available():
            snapshot["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            snapshot["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
    except Exception:
        pass
    return snapshot


def log_memory(stage: str) -> None:
    snapshot = memory_snapshot(stage)
    parts = [f"[mem] {stage}"]
    if snapshot.get("rss_mb") != "":
        parts.append(f"rss={float(snapshot['rss_mb']):.1f}MB")
    if "cuda_allocated_mb" in snapshot:
        parts.append(f"cuda_alloc={float(snapshot['cuda_allocated_mb']):.1f}MB")
        parts.append(f"cuda_reserved={float(snapshot['cuda_reserved_mb']):.1f}MB")
    print(" ".join(parts), flush=True)


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
