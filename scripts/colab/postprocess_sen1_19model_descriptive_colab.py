from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--geospatial-metadata", type=Path, required=True)
    parser.add_argument("--unet-audit", type=Path, required=True)
    parser.add_argument("--prithvi-audit", type=Path, required=True)
    parser.add_argument("--terramind-audit", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/content/sen1_19model_descriptive_work"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/sen1_19model_descriptive_v2"))
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
        raise RuntimeError(f"Refusing to overwrite an existing staged export: {target}")
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


def _index_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise RuntimeError(f"Missing probability index: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not str(row.get("sample_id", "")).strip() for row in rows):
        raise RuntimeError(f"Invalid or empty probability index: {path}")
    return rows


def _indexed_artifact(export: Path, row: dict[str, object]) -> Path:
    raw = Path(str(row.get("probability_path", "")))
    frozen_local = export / "samples" / raw.name
    if frozen_local.is_file():
        candidate = frozen_local
    else:
        candidate = raw if raw.is_absolute() else export / raw
    if not candidate.is_file():
        raise RuntimeError(f"Missing indexed probability artifact: {candidate}")
    return candidate


def _validate_staged_export(
    source: Path,
    target: Path,
    recorded_indexes: list[dict[str, object]],
) -> dict[str, object]:
    source_indexes = sorted((source / "index_parts").glob("*.jsonl"))
    target_indexes = sorted((target / "index_parts").glob("*.jsonl"))
    recorded = {str(item.get("name", "")): item for item in recorded_indexes}
    names = [path.name for path in source_indexes]
    if not names or names != [path.name for path in target_indexes] or set(names) != set(recorded):
        raise RuntimeError(
            f"Staging index inventory does not match its frozen source: source={source}, target={target}."
        )
    collection = hashlib.sha256()
    artifact_count = 0
    indexes = []
    for source_index, target_index in zip(source_indexes, target_indexes):
        record = recorded[source_index.name]
        source_sha = file_sha256(source_index)
        target_sha = file_sha256(target_index)
        if source_sha != record.get("source_sha256") or target_sha != record.get("portable_staged_sha256"):
            raise RuntimeError(f"Staging index SHA mismatch: {source_index.name}")
        source_rows = _index_rows(source_index)
        target_rows = _index_rows(target_index)
        if len(source_rows) != len(target_rows) or len(source_rows) != int(record.get("row_count", -1)):
            raise RuntimeError(f"Staging index row-count mismatch: {source_index.name}")
        for source_row, target_row in zip(source_rows, target_rows):
            sample_id = str(source_row.get("sample_id", ""))
            if sample_id != str(target_row.get("sample_id", "")):
                raise RuntimeError(f"Staging sample order mismatch: {source_index.name}")
            source_artifact = _indexed_artifact(source, source_row)
            target_artifact = _indexed_artifact(target, target_row)
            if source_artifact.name != target_artifact.name:
                raise RuntimeError(f"Staging probability basename mismatch: sample_id={sample_id}")
            if source_artifact.stat().st_size != target_artifact.stat().st_size:
                raise RuntimeError(f"Staging probability size mismatch: sample_id={sample_id}")
            source_artifact_sha = file_sha256(source_artifact)
            if source_artifact_sha != file_sha256(target_artifact):
                raise RuntimeError(f"Staging probability SHA mismatch: sample_id={sample_id}")
            collection.update(sample_id.encode("utf-8"))
            collection.update(b"\0")
            collection.update(source_artifact_sha.encode("ascii"))
            collection.update(b"\0")
            artifact_count += 1
        indexes.append(
            {
                "name": source_index.name,
                "source_sha256": source_sha,
                "portable_staged_sha256": target_sha,
                "row_count": len(source_rows),
            }
        )
    return {
        "source": str(source),
        "target": str(target),
        "indexes": indexes,
        "probability_artifact_count": artifact_count,
        "source_staged_probability_collection_sha256": collection.hexdigest(),
    }


def _validate_existing_staging(
    *,
    source_specs: list,
    target_specs: list,
    manifest: Path,
) -> Path:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid staging provenance: {manifest}") from exc
    if payload.get("schema") not in {
        "geobwer.sen1floods11.19model_local_staging.v1",
        "geobwer.sen1floods11.19model_local_staging.v2",
    }:
        raise RuntimeError(f"Unsupported staging provenance schema: {manifest}")
    records = {
        (str(row.get("model", "")), str(row.get("split", ""))): row
        for row in payload.get("records", [])
    }
    expected_keys = {
        (spec.model_name, split)
        for spec in source_specs
        for split in ("validation", "standard_test", "bolivia_holdout")
    }
    if len(source_specs) != 19 or set(records) != expected_keys:
        raise RuntimeError(
            "Staging provenance does not bind the exact frozen 19-model/57-export panel."
        )
    validated_records = []
    for source, target in zip(source_specs, target_specs):
        for split in ("validation", "standard_test", "bolivia_holdout"):
            record = records[(source.model_name, split)]
            source_export = source.export(split)
            target_export = target.export(split)
            if Path(str(record.get("source", ""))).resolve() != source_export.resolve():
                raise RuntimeError(f"Staging source path drift: {source.model_name}/{split}")
            if Path(str(record.get("target", ""))).resolve() != target_export.resolve():
                raise RuntimeError(f"Staging target path drift: {source.model_name}/{split}")
            validated = _validate_staged_export(
                source_export,
                target_export,
                list(record.get("indexes", [])),
            )
            validated_records.append({"model": source.model_name, "split": split, **validated})
    manifest.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.19model_local_staging.v2",
                "status": "validated_for_reuse",
                "policy": "copy_probability_exports_to_local_scratch_and_rewrite_only_staged_index_paths",
                "frozen_sources_modified": False,
                "exact_model_count": 19,
                "exact_export_count": 57,
                "records": validated_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def _stage_probability_panel(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    staged = args.work_root / "staged_probability_sources"
    roots = {
        "unet": staged / "unet",
        "prithvi": staged / "prithvi",
        "terramind": staged / "terramind",
    }
    manifest = args.work_root / "staging_provenance.json"
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
    if staged.exists() or manifest.exists():
        if not staged.is_dir() or not manifest.is_file():
            raise RuntimeError(
                "Partial staging state exists. Refusing to recopy or repair 57 exports; "
                f"staged={staged}, provenance={manifest}."
            )
        validated = _validate_existing_staging(
            source_specs=source_specs,
            target_specs=target_specs,
            manifest=manifest,
        )
        print("[sen1:stage] existing 57-export staging strictly validated and reused", flush=True)
        return roots["unet"], roots["prithvi"], roots["terramind"], validated
    staged.mkdir(parents=True)
    records = []
    for index, (source, target) in enumerate(zip(source_specs, target_specs), start=1):
        print(f"[sen1:stage] model={index}/19 name={source.model_name}", flush=True)
        for split in ("validation", "standard_test", "bolivia_holdout"):
            record = _copy_tree(source.export(split), target.export(split))
            records.append({"model": source.model_name, "split": split, **record})
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
    validated = _validate_existing_staging(
        source_specs=source_specs,
        target_specs=target_specs,
        manifest=manifest,
    )
    return roots["unet"], roots["prithvi"], roots["terramind"], validated


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
    staged_geospatial = metadata_root / "geospatial446_metadata.csv"
    shutil.copy2(args.core_metadata, staged_core)
    shutil.copy2(args.bolivia_metadata, staged_bolivia)
    shutil.copy2(args.geospatial_metadata, staged_geospatial)
    metadata_provenance = metadata_root / "metadata_source_provenance.json"
    metadata_provenance.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.19model_metadata_sources.v1",
                "coordinate_authority": {
                    "source": str(args.geospatial_metadata),
                    "source_sha256": file_sha256(args.geospatial_metadata),
                    "staged": str(staged_geospatial),
                    "staged_sha256": file_sha256(staged_geospatial),
                    "role": "exclusive_latitude_longitude_source",
                },
                "attribute_sources": [
                    {
                        "source": str(source),
                        "source_sha256": file_sha256(source),
                        "staged": str(staged),
                        "staged_sha256": file_sha256(staged),
                        "role": "event_split_and_noncoordinate_attributes_only",
                    }
                    for source, staged in (
                        (args.core_metadata, staged_core),
                        (args.bolivia_metadata, staged_bolivia),
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    run_sen1_19model_descriptive_postprocess(
        unet_root=staged_unet,
        prithvi_root=staged_prithvi,
        terramind_root=staged_terramind,
        core_metadata_csv=staged_core,
        bolivia_metadata_csv=staged_bolivia,
        geospatial_metadata_csv=staged_geospatial,
        audit_evidence=[
            args.unet_audit,
            args.prithvi_audit,
            args.terramind_audit,
            staging_manifest,
            metadata_provenance,
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
