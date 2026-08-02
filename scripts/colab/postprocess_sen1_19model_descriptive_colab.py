from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rsfm_fairness_audit import __version__  # noqa: E402
from rsfm_fairness_audit.formal_outputs import file_sha256  # noqa: E402
from rsfm_fairness_audit.sen1_19model_descriptive import (  # noqa: E402
    expected_model_specs,
    run_sen1_19model_descriptive_postprocess,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only unified descriptive postprocess for the frozen Sen1 19-model panel."
    )
    parser.add_argument("--unet-root", type=Path, required=True)
    parser.add_argument("--prithvi-root", type=Path, required=True)
    parser.add_argument("--terramind-root", type=Path, required=True)
    parser.add_argument("--core-metadata", type=Path, required=True)
    parser.add_argument("--bolivia-metadata", type=Path, required=True)
    parser.add_argument("--unet-audit", type=Path, required=True)
    parser.add_argument("--prithvi-audit", type=Path, required=True)
    parser.add_argument("--terramind-audit", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/content/sen1_19model_descriptive_work"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/sen1_19model_descriptive_v1"))
    parser.add_argument("--persistent-output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _completion_valid(root: Path) -> bool:
    contract = root / "completion_contract.json"
    if not contract.is_file():
        return False
    payload = json.loads(contract.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or int(payload.get("model_count", -1)) != 19:
        return False
    for record in payload.get("artifacts", []):
        path = root / str(record.get("path", ""))
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            return False
    return True


def _copy_tree(source: Path, target: Path) -> dict[str, object]:
    if not source.is_dir():
        raise RuntimeError(f"Frozen probability export is missing: {source}")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, copy_function=shutil.copy2)
    index_records = []
    for source_index in sorted((source / "index_parts").glob("*.jsonl")):
        target_index = target / "index_parts" / source_index.name
        rows = []
        for line in target_index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["probability_path"] = f"samples/{Path(str(row['probability_path'])).name}"
            rows.append(row)
        target_index.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        index_records.append(
            {
                "name": source_index.name,
                "source_sha256": file_sha256(source_index),
                "portable_staged_sha256": file_sha256(target_index),
                "row_count": len(rows),
            }
        )
    return {"source": str(source), "target": str(target), "indexes": index_records}


def _stage_probability_panel(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    staged = args.work_root / "staged_probability_sources"
    roots = {
        "unet": staged / "unet",
        "prithvi": staged / "prithvi",
        "terramind": staged / "terramind",
    }
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    source_specs = expected_model_specs(
        unet_root=args.unet_root,
        prithvi_root=args.prithvi_root,
        terramind_root=args.terramind_root,
    )
    target_specs = expected_model_specs(
        unet_root=roots["unet"],
        prithvi_root=roots["prithvi"],
        terramind_root=roots["terramind"],
    )
    records = []
    for index, (source, target) in enumerate(zip(source_specs, target_specs), start=1):
        print(f"[sen1:stage] model={index}/19 name={source.model_name}", flush=True)
        for split in ("validation", "standard_test", "bolivia_holdout"):
            record = _copy_tree(source.export(split), target.export(split))
            records.append({"model": source.model_name, "split": split, **record})
    manifest = args.work_root / "staging_provenance.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.19model_local_staging.v1",
                "policy": "copy_probability_exports_to_local_scratch_and_rewrite_only_staged_index_paths",
                "frozen_sources_modified": False,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return roots["unet"], roots["prithvi"], roots["terramind"], manifest


def main() -> int:
    args = _args()
    if _completion_valid(args.persistent_output_dir):
        print("SEN1_19MODEL_DESCRIPTIVE_POSTPROCESS=PASS")
        print(f"OUTPUT={args.persistent_output_dir}")
        return 0
    if args.persistent_output_dir.exists() and any(args.persistent_output_dir.iterdir()):
        raise RuntimeError(
            "Persistent output is non-empty but lacks a valid completion contract; "
            f"use a new versioned directory: {args.persistent_output_dir}"
        )
    args.work_root.mkdir(parents=True, exist_ok=True)
    if args.output_dir.exists():
        if _completion_valid(args.output_dir):
            shutil.copytree(args.output_dir, args.persistent_output_dir)
            print("SEN1_19MODEL_DESCRIPTIVE_POSTPROCESS=PASS")
            print(f"OUTPUT={args.persistent_output_dir}")
            return 0
        if any(args.output_dir.iterdir()):
            raise RuntimeError(f"Incomplete local output exists: {args.output_dir}")
    staged_unet, staged_prithvi, staged_terramind, staging_manifest = _stage_probability_panel(args)
    metadata_root = args.work_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    staged_core = metadata_root / "core431_metadata.csv"
    staged_bolivia = metadata_root / "bolivia15_metadata.csv"
    shutil.copy2(args.core_metadata, staged_core)
    shutil.copy2(args.bolivia_metadata, staged_bolivia)
    run_sen1_19model_descriptive_postprocess(
        unet_root=staged_unet,
        prithvi_root=staged_prithvi,
        terramind_root=staged_terramind,
        metadata_csvs=[staged_core, staged_bolivia],
        audit_evidence=[
            args.unet_audit,
            args.prithvi_audit,
            args.terramind_audit,
            staging_manifest,
        ],
        output_dir=args.output_dir,
        code_commit=_git_head(),
        package_version=__version__,
    )
    shutil.copytree(args.output_dir, args.persistent_output_dir)
    if not _completion_valid(args.persistent_output_dir):
        raise RuntimeError("Persistent copy failed completion-contract verification.")
    print("SEN1_19MODEL_DESCRIPTIVE_POSTPROCESS=PASS")
    print(f"OUTPUT={args.persistent_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
