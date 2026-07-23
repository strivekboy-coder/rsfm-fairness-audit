from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil


class PersistentCacheError(RuntimeError):
    """Raised when a Colab live-output/persistent-cache contract is unsafe."""


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_colab_drive_path(path: str | Path) -> bool:
    text = Path(path).resolve().as_posix().lower().rstrip("/") + "/"
    return text.startswith("/content/drive/")


def validate_storage_contract(
    live_output_dir: str | Path,
    persistent_output_dir: str | Path | None,
) -> None:
    """Require fast local Colab storage for live work and a distinct mirror."""

    live = Path(live_output_dir).resolve()
    if is_colab_drive_path(live):
        raise PersistentCacheError(
            "Live output may not be under /content/drive. Use a local /content path and pass Drive as the "
            "persistent output directory."
        )
    if persistent_output_dir is None:
        return
    persistent = Path(persistent_output_dir).resolve()
    if live == persistent:
        raise PersistentCacheError("Live and persistent output directories must differ.")
    if live in persistent.parents or persistent in live.parents:
        raise PersistentCacheError("Live and persistent output directories may not contain one another.")


def copy_changed_tree(
    source: str | Path,
    destination: str | Path,
    *,
    label: str,
    small_hash_limit_bytes: int = 8 * 1024 * 1024,
) -> int:
    """Atomically copy newer/changed artifacts without deleting either tree."""

    source_root = Path(source)
    destination_root = Path(destination)
    if not source_root.exists():
        return 0
    destination_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        target = destination_root / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            source_stat = path.stat()
            target_stat = target.stat()
            if source_stat.st_size == target_stat.st_size:
                if source_stat.st_size <= int(small_hash_limit_bytes):
                    if _sha256(path) == _sha256(target):
                        continue
                elif target_stat.st_mtime_ns >= source_stat.st_mtime_ns:
                    continue
            if target_stat.st_mtime_ns > source_stat.st_mtime_ns:
                print(f"[persistent-cache] keeping newer destination artifact {target}", flush=True)
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        shutil.copy2(path, temporary)
        os.replace(temporary, target)
        copied += 1
    print(
        f"[persistent-cache] {label}: copied={copied} source={source_root} destination={destination_root}",
        flush=True,
    )
    return copied


def hydrate_output(live_output_dir: str | Path, persistent_output_dir: str | Path | None) -> int:
    validate_storage_contract(live_output_dir, persistent_output_dir)
    if persistent_output_dir is None or not Path(persistent_output_dir).exists():
        return 0
    return copy_changed_tree(persistent_output_dir, live_output_dir, label="hydrate")


def persist_output(
    live_output_dir: str | Path,
    persistent_output_dir: str | Path | None,
    *,
    label: str,
) -> int:
    validate_storage_contract(live_output_dir, persistent_output_dir)
    if persistent_output_dir is None:
        return 0
    return copy_changed_tree(live_output_dir, persistent_output_dir, label=label)


__all__ = [
    "PersistentCacheError",
    "copy_changed_tree",
    "hydrate_output",
    "is_colab_drive_path",
    "persist_output",
    "validate_storage_contract",
]
