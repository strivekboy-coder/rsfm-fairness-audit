from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


ASSETS = {
    "dofav2": {
        "repo_id": "earthflow/DOFA",
        "filename": "dofav2_vit_base_e150.pth",
        "revision": "67e355727ca732ff0d6ca3ebcd86d399cd6b3c15",
        "sha256": "e1be9d50fb3e4e3640e337d098b92d67797eaf2a579de3b7a1e363095885314d",
    },
    "croma": {
        "repo_id": "antofuller/CROMA",
        "filename": "CROMA_base.pt",
        "revision": "0dd28e3d633bd6715856ae9890e8c49360040598",
        "sha256": "0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63",
    },
    "terramind": {
        "repo_id": "ibm-esa-geospatial/TerraMind-1.0-base",
        "filename": "TerraMind_v1_base.pt",
        "revision": "fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a",
        "sha256": "83c3a0938067c83867a46e564443c2fa38383bf4f966d931b11cb025b847d7ec",
    },
}

REPOSITORIES = {
    "dofa": {
        "url": "https://github.com/zhu-xlab/DOFA.git",
        "revision": "0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9",
    },
    "croma": {
        "url": "https://github.com/antofuller/CROMA.git",
        "revision": "59505a6bcadbf36ba20767270154bf9f3067c5e7",
    },
}


class AssetPreparationError(RuntimeError):
    """Raised instead of silently accepting a mutable or corrupt model asset."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> str:
    if not path.is_file():
        raise AssetPreparationError(f"Missing checkpoint: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise AssetPreparationError(
            f"Checkpoint hash mismatch for {path}: expected={expected}, observed={observed}. "
            "Do not overwrite the file in place; inspect the source and use a clean cache directory."
        )
    return observed


def _download_to_drive(name: str, specification: dict[str, str], drive_cache: Path) -> Path:
    destination = drive_cache / "checkpoints" / specification["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify(destination, specification["sha256"])
        print(f"[assets] reuse verified Drive checkpoint {destination}", flush=True)
        return destination
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise AssetPreparationError("Install huggingface_hub before preparing pinned checkpoints.") from exc
    print(
        f"[assets] download {name}: {specification['repo_id']}@{specification['revision']}/{specification['filename']}",
        flush=True,
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=specification["repo_id"],
            filename=specification["filename"],
            revision=specification["revision"],
            cache_dir=str(drive_cache / "huggingface_cache"),
        )
    )
    _verify(downloaded, specification["sha256"])
    shutil.copy2(downloaded, destination)
    _verify(destination, specification["sha256"])
    return destination


def _copy_local(source: Path, destination: Path, expected: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify(destination, expected)
        print(f"[assets] reuse verified local checkpoint {destination}", flush=True)
        return destination
    print(f"[assets] copy checkpoint Drive -> local: {destination.name}", flush=True)
    shutil.copy2(source, destination)
    _verify(destination, expected)
    return destination


def _run_git(arguments: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(arguments, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssetPreparationError(
            f"Git command failed ({' '.join(arguments)}): {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _prepare_repo(name: str, specification: dict[str, str], repo_root: Path) -> Path:
    destination = repo_root / name
    if destination.exists() and not (destination / ".git").is_dir():
        raise AssetPreparationError(f"Repository destination exists but is not a Git checkout: {destination}")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[assets] clone {specification['url']} -> {destination}", flush=True)
        _run_git(["git", "clone", "--filter=blob:none", "--no-checkout", specification["url"], str(destination)])
    dirty = _run_git(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=destination,
    )
    if dirty:
        raise AssetPreparationError(f"Tracked files are modified under {destination}; refusing to replace them.")
    observed = _run_git(["git", "rev-parse", "HEAD"], cwd=destination).lower()
    if observed != specification["revision"]:
        print(f"[assets] fetch and checkout pinned {name} revision {specification['revision']}", flush=True)
        _run_git(["git", "fetch", "origin", specification["revision"], "--depth", "1"], cwd=destination)
        _run_git(["git", "checkout", "--detach", specification["revision"]], cwd=destination)
    observed = _run_git(["git", "rev-parse", "HEAD"], cwd=destination).lower()
    if observed != specification["revision"]:
        raise AssetPreparationError(
            f"Repository revision mismatch for {name}: expected={specification['revision']}, observed={observed}."
        )
    return destination


def prepare_assets(drive_cache: Path, local_root: Path, repo_root: Path) -> dict[str, Any]:
    if not drive_cache.parent.exists():
        raise AssetPreparationError(
            f"Drive cache parent is unavailable: {drive_cache.parent}. Mount Google Drive before running this script."
        )
    checkpoints: dict[str, Any] = {}
    for name, specification in ASSETS.items():
        drive_path = _download_to_drive(name, specification, drive_cache)
        local_path = _copy_local(
            drive_path,
            local_root / specification["filename"],
            specification["sha256"],
        )
        checkpoints[name] = {
            **specification,
            "drive_path": str(drive_path),
            "local_path": str(local_path),
            "observed_sha256": _verify(local_path, specification["sha256"]),
        }
    repositories = {
        name: {
            **specification,
            "local_path": str(_prepare_repo(name, specification, repo_root)),
        }
        for name, specification in REPOSITORIES.items()
    }
    manifest = {
        "schema": "geobwer.model_assets.v1",
        "checkpoints": checkpoints,
        "repositories": repositories,
        "execution_policy": "Drive is persistent cache; formal model loading uses local /content copies.",
    }
    local_manifest = local_root / "geobwer_model_assets_manifest.json"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    local_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    drive_manifest = drive_cache / "geobwer_model_assets_manifest.json"
    drive_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "local_manifest": str(local_manifest), "drive_manifest": str(drive_manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, cache, copy, and verify the exact DOFAv2/CROMA/TerraMind assets for formal campaigns."
    )
    parser.add_argument(
        "--drive-cache",
        type=Path,
        default=Path("/content/drive/MyDrive/rsfm_fairness_audit/cache/model_assets/geobwer_final_v1"),
    )
    parser.add_argument("--local-root", type=Path, default=Path("/content/rsfm_model_assets"))
    parser.add_argument("--repo-root", type=Path, default=Path("/content/rsfm_model_repos"))
    args = parser.parse_args()
    manifest = prepare_assets(args.drive_cache, args.local_root, args.repo_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
