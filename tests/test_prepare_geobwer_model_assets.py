from __future__ import annotations

from pathlib import Path
import os
import shutil
import stat
import subprocess
import time
import uuid

import pytest

from scripts.colab.prepare_geobwer_model_assets_colab import (
    AssetPreparationError,
    _ensure_drive_cache,
    _prepare_repo,
)


def _git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_repository(root: Path) -> tuple[Path, str]:
    source = root / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "geobwer-test@example.invalid")
    _git(source, "config", "user.name", "GeoBWER Test")
    tracked = source / "tracked.py"
    tracked.write_text("VERSION = 1\n", encoding="utf-8")
    _git(source, "add", "tracked.py")
    _git(source, "commit", "-m", "pinned revision")
    pinned = _git(source, "rev-parse", "HEAD").lower()
    tracked.write_text("VERSION = 2\n", encoding="utf-8")
    _git(source, "commit", "-am", "newer default revision")
    return source, pinned


@pytest.fixture
def repo_test_root() -> Path:
    root = Path("work") / "test_runs" / f"asset_repo_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        def remove_readonly(function: object, path: str, _: object) -> None:
            os.chmod(path, stat.S_IWRITE)
            function(path)  # type: ignore[operator]

        for attempt in range(10):
            try:
                shutil.rmtree(root, onexc=remove_readonly)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1)


def test_prepare_repo_fresh_clone_checks_out_pinned_revision(repo_test_root: Path) -> None:
    source, pinned = _source_repository(repo_test_root)
    destination = _prepare_repo(
        "demo",
        {"url": str(source), "revision": pinned},
        repo_test_root / "repos",
    )
    assert _git(destination, "rev-parse", "HEAD").lower() == pinned
    assert (destination / "tracked.py").read_text(encoding="utf-8") == "VERSION = 1\n"
    assert _git(destination, "status", "--porcelain", "--untracked-files=no") == ""


def test_prepare_repo_second_run_reuses_clean_checkout(repo_test_root: Path) -> None:
    source, pinned = _source_repository(repo_test_root)
    specification = {"url": str(source), "revision": pinned}
    first = _prepare_repo("demo", specification, repo_test_root / "repos")
    second = _prepare_repo("demo", specification, repo_test_root / "repos")
    assert second == first
    assert _git(second, "rev-parse", "HEAD").lower() == pinned


def test_prepare_repo_rejects_real_tracked_modification(repo_test_root: Path) -> None:
    source, pinned = _source_repository(repo_test_root)
    specification = {"url": str(source), "revision": pinned}
    destination = _prepare_repo("demo", specification, repo_test_root / "repos")
    (destination / "tracked.py").write_text("VERSION = 'locally modified'\n", encoding="utf-8")
    with pytest.raises(AssetPreparationError, match="Tracked files are modified"):
        _prepare_repo("demo", specification, repo_test_root / "repos")


def test_ensure_drive_cache_creates_nested_project_cache(repo_test_root: Path) -> None:
    mount_root = repo_test_root / "mounted_drive"
    mount_root.mkdir()
    cache = mount_root / "project" / "cache" / "model_assets"
    assert _ensure_drive_cache(cache, drive_mount_root=mount_root) == cache
    assert cache.is_dir()


def test_ensure_drive_cache_refuses_unmounted_drive(repo_test_root: Path) -> None:
    mount_root = repo_test_root / "missing_drive"
    cache = mount_root / "project" / "cache" / "model_assets"
    with pytest.raises(AssetPreparationError, match="mount is unavailable"):
        _ensure_drive_cache(cache, drive_mount_root=mount_root)
