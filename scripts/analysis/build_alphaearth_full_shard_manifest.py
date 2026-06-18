from __future__ import annotations

import argparse
from pathlib import Path

from rsfm_fairness_audit.io import ensure_dir, write_csv


DEFAULT_SHARD_DIR = Path("outputs/alphaearth_gee_full_v1")
DEFAULT_MANIFEST = Path("outputs/alphaearth_gee_full_v1/alphaearth_worldcover_full_export_manifest.csv")


def build_alphaearth_full_shard_manifest(shard_dir: Path = DEFAULT_SHARD_DIR, manifest: Path = DEFAULT_MANIFEST) -> Path:
    ensure_dir(manifest.parent)
    shards = sorted(
        path
        for path in shard_dir.glob("alphaearth_worldcover_full_2021_*_shard.csv")
        if path.is_file()
    )
    rows = [
        {
            "shard_id": path.stem.replace("alphaearth_worldcover_full_2021_", "").replace("_shard", ""),
            "path": path.name,
            "status": "available",
            "bytes": path.stat().st_size,
        }
        for path in shards
    ]
    if not rows:
        rows = [{"shard_id": "", "path": "", "status": "missing_no_shards_found", "bytes": ""}]
    write_csv(manifest, rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest for AlphaEarth full CSV shards.")
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(build_alphaearth_full_shard_manifest(args.shard_dir, args.out))


if __name__ == "__main__":
    main()
