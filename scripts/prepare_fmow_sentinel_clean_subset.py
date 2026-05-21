from __future__ import annotations

import argparse
import heapq
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


SUPPORTED_STRATIFY_FIELDS = {
    "split",
    "category",
    "country",
    "region",
    "un_region",
    "continent",
    "season",
    "latitude_band",
    "location_id",
}

SUPPORT_AUGMENT_FIELDS = (
    "season",
    "latitude_band",
    "un_region",
    "region",
    "country",
    "category__region",
    "country__category",
)


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).lower() in {"nan", "none", "null"}


def _write_csv_union(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        write_csv(path, [])
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    normalized = [{key: row.get(key, "") for key in fieldnames} for row in rows]
    write_csv(path, normalized)


def archive_path_for_row(row: Mapping[str, Any]) -> str:
    """Return the official fMoW-Sentinel member path for one metadata row."""
    existing = str(row.get("archive_path", "") or "")
    if existing:
        return existing.replace("\\", "/").lstrip("/")
    split = str(row.get("split", "") or "").strip()
    category = str(row.get("category", row.get("label", "")) or "").strip()
    location_id = str(row.get("location_id", "") or "").strip()
    image_id = str(row.get("image_id", "") or "").strip()
    if not split or not category or not location_id or not image_id:
        missing = [key for key, value in {"split": split, "category": category, "location_id": location_id, "image_id": image_id}.items() if not value]
        raise ValueError(f"Cannot construct archive path; missing {', '.join(missing)}.")
    location_name = f"{category}_{location_id}"
    file_name = f"{category}_{location_id}_{image_id}.tif"
    return f"fmow-sentinel/{split}/{category}/{location_name}/{file_name}"


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("sample_id", "image_id"):
        if key in row and not _is_missing(row[key]):
            return str(row[key])
    return str(index)


def _load_rows(path: Path, splits: set[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(read_csv_rows(path)):
        item = dict(row)
        item["sample_id"] = _sample_id(item, index)
        item["category"] = item.get("category") or item.get("label") or item.get("class_label") or ""
        item["split"] = item.get("split") or "all"
        if splits and str(item["split"]) not in splits:
            continue
        try:
            item["archive_path"] = archive_path_for_row(item)
        except ValueError as exc:
            item["archive_path_error"] = str(exc)
        rows.append(item)
    return rows


def _infer_split_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    if stem in {"train", "val", "test"}:
        return stem
    for split in ("train", "val", "test"):
        if split in stem:
            return split
    return ""


def _build_manifest_from_sources(
    satmae_csvs: Sequence[Path],
    location_geography_csv: Path,
    output: Path,
    splits: set[str] | None,
) -> Path:
    geography_rows = read_csv_rows(location_geography_csv)
    geography_by_location = {
        (str(row.get("category", "")), str(row.get("location_id", ""))): row
        for row in geography_rows
    }
    merged: list[dict[str, Any]] = []
    for csv_path in satmae_csvs:
        inferred_split = _infer_split_from_filename(csv_path)
        for row in read_csv_rows(csv_path):
            item = dict(row)
            item["category"] = item.get("category") or item.get("label") or item.get("class_label") or ""
            item["split"] = item.get("split") or inferred_split or "all"
            if splits and str(item["split"]) not in splits:
                continue
            geo = geography_by_location.get((str(item.get("category", "")), str(item.get("location_id", ""))), {})
            for key, value in geo.items():
                if key not in item or _is_missing(item.get(key)):
                    item[key] = value
            item["sample_id"] = _sample_id(item, len(merged))
            item["archive_path"] = archive_path_for_row(item)
            item["image_path"] = item["archive_path"]
            item["metadata_provenance"] = item.get("metadata_provenance", "location_level_geography_enrichment")
            merged.append(item)
    manifest = output / "fmow_sentinel_enriched_sample_manifest_final_v1.csv"
    write_csv(manifest, merged)
    return manifest


def _group_key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "missing") or "missing") for field in fields)


def _support_value(row: Mapping[str, Any], field: str) -> str:
    if "__" not in field:
        return str(row.get(field, "missing") or "missing")
    parts = field.split("__")
    return "__".join(str(row.get(part, "missing") or "missing") for part in parts)


def _support_counts(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Counter[str]]:
    return {field: Counter(_support_value(row, field) for row in rows) for field in fields}


def _write_support_by_value(rows: Sequence[Mapping[str, Any]], output_path: Path, fields: Sequence[str]) -> Path:
    records: list[dict[str, Any]] = []
    counts = _support_counts(rows, fields)
    for field in fields:
        for value, support in sorted(counts[field].items()):
            records.append({"field": field, "slice_value": value, "support": support})
    write_csv(output_path, records)
    return output_path


def select_rows(
    rows: Sequence[dict[str, Any]],
    stratify_fields: Sequence[str],
    max_samples_per_split: int,
    max_total: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministic quota-aware round-robin sampling over split and requested strata."""
    usable = [row for row in rows if not row.get("archive_path_error")]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_split[str(row.get("split", "all"))].append(row)
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    per_split_limit = max_samples_per_split if max_samples_per_split > 0 else max(len(items) for items in by_split.values())
    for split in sorted(by_split):
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in by_split[split]:
            groups[_group_key(row, stratify_fields)].append(row)
        for group_rows in groups.values():
            group_rows.sort(key=lambda row: (str(row.get("category", "")), str(row.get("location_id", "")), str(row.get("image_id", ""))))
            rng.shuffle(group_rows)
        group_keys = sorted(groups)
        split_selected: list[dict[str, Any]] = []
        cursor = 0
        while len(split_selected) < per_split_limit and group_keys:
            key = group_keys[cursor % len(group_keys)]
            bucket = groups[key]
            if bucket:
                split_selected.append(bucket.pop())
            group_keys = [item for item in group_keys if groups[item]]
            cursor += 1
        split_selected.sort(key=lambda row: str(row["archive_path"]))
        selected.extend(split_selected)
    selected.sort(key=lambda row: (str(row.get("split", "")), str(row["archive_path"])))
    if max_total and max_total > 0 and len(selected) > max_total:
        indices = sorted(rng.choice(len(selected), size=max_total, replace=False).tolist())
        selected = [selected[index] for index in indices]
    return selected


def _split_targets(args: argparse.Namespace) -> dict[str, int]:
    targets: dict[str, int] = {}
    if args.target_train is not None:
        targets["train"] = int(args.target_train)
    if args.target_val is not None:
        targets["val"] = int(args.target_val)
    if targets:
        return targets
    splits = tuple(args.split or ("train", "val"))
    if args.target_total and splits:
        base = int(args.target_total) // len(splits)
        remainder = int(args.target_total) % len(splits)
        for index, split in enumerate(splits):
            targets[str(split)] = base + (1 if index < remainder else 0)
    return targets


def _row_identity(row: Mapping[str, Any]) -> str:
    for key in ("archive_path", "sample_id", "image_id"):
        if key in row and not _is_missing(row.get(key)):
            return str(row[key])
    return ""


def select_augmentation_rows(
    existing_rows: Sequence[dict[str, Any]],
    metadata_rows: Sequence[dict[str, Any]],
    target_by_split: Mapping[str, int],
    seed: int,
    support_fields: Sequence[str] = SUPPORT_AUGMENT_FIELDS,
    batch_size: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select additional samples by prioritizing weak slice support.

    The first implementation used one-sample-at-a-time greedy selection. That is
    methodologically clean but too slow for 20k+ fMoW augmentation because it
    rescans hundreds of thousands of candidates after every chosen sample. This
    batched version keeps the same support-aware scoring objective while
    updating supports after small batches, which is deterministic and practical
    for Colab-scale subset preparation.
    """
    existing_ids = {_row_identity(row) for row in existing_rows if _row_identity(row)}
    selected: list[dict[str, Any]] = []
    current_rows = [dict(row) for row in existing_rows]
    rng = np.random.default_rng(seed)
    batch_size = max(1, int(batch_size))
    for split in sorted(target_by_split):
        current_split = [row for row in current_rows if str(row.get("split", "")) == split]
        needed = max(0, int(target_by_split[split]) - len(current_split))
        if needed <= 0:
            continue
        candidates = [
            dict(row)
            for row in metadata_rows
            if str(row.get("split", "")) == split
            and not row.get("archive_path_error")
            and _row_identity(row) not in existing_ids
        ]
        for row in candidates:
            # Stable seeded jitter only affects exact ties.
            row["_jitter"] = float(rng.random() * 1e-9)
        print(
            f"[selection] split={split} existing={len(current_split)} "
            f"target={target_by_split[split]} need={needed} candidates={len(candidates)}"
        )
        selected_for_split = 0
        while needed > 0 and candidates:
            if not candidates:
                break
            counts = _support_counts(current_rows, support_fields)

            def score(row: Mapping[str, Any]) -> tuple[float, str]:
                value = 0.0
                for field in support_fields:
                    support = counts[field].get(_support_value(row, field), 0)
                    weight = 2.0 if field in {"season", "latitude_band", "country"} else 1.0
                    value += weight / (support + 1.0)
                value += float(row.get("_jitter", 0.0))
                return value, str(row.get("archive_path", ""))

            take = min(needed, batch_size, len(candidates))
            top = heapq.nlargest(take, enumerate(candidates), key=lambda item: score(item[1]))
            top_indices = {index for index, _row in top}
            for _index, best in top:
                chosen = dict(best)
                chosen.pop("_jitter", None)
                selected.append(chosen)
                current_rows.append(chosen)
                existing_ids.add(_row_identity(chosen))
            candidates = [row for index, row in enumerate(candidates) if index not in top_indices]
            needed -= take
            selected_for_split += take
            print(
                f"[selection] split={split} selected={selected_for_split} "
                f"remaining_need={needed} remaining_candidates={len(candidates)}"
            )
    summary = {
        "target_by_split": dict(target_by_split),
        "existing_by_split": dict(Counter(str(row.get("split", "")) for row in existing_rows)),
        "selected_by_split": dict(Counter(str(row.get("split", "")) for row in selected)),
        "final_by_split": dict(Counter(str(row.get("split", "")) for row in current_rows)),
        "support_priority_fields": list(support_fields),
        "selection_batch_size": batch_size,
    }
    return selected, summary


def _safe_member_path(member_name: str, output_root: Path) -> Path:
    relative = Path(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return output_root / relative


def _read_tif(path: Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("tifffile is required to validate extracted fMoW-Sentinel TIFFs.") from exc
    return np.asarray(tifffile.imread(path))


def _band_axis_and_count(array: np.ndarray) -> tuple[int, int]:
    if array.ndim != 3:
        return -1, 0
    if array.shape[0] == 13:
        return 0, 13
    if array.shape[-1] == 13:
        return 2, 13
    return -1, int(max(array.shape))


def validate_raster(path: Path) -> dict[str, Any]:
    try:
        array = _read_tif(path)
        band_axis, band_count = _band_axis_and_count(array)
        valid = array.size > 0 and band_count == 13
        finite = np.asarray(array, dtype=np.float32)
        return {
            "path": str(path),
            "valid": valid,
            "reason": "" if valid else f"expected 13 bands, got shape={array.shape}",
            "shape": "x".join(str(dim) for dim in array.shape),
            "band_count": band_count,
            "band_axis": band_axis,
            "dtype": str(array.dtype),
            "min": float(np.nanmin(finite)) if finite.size else "",
            "max": float(np.nanmax(finite)) if finite.size else "",
            "mean": float(np.nanmean(finite)) if finite.size else "",
        }
    except Exception as exc:
        return {
            "path": str(path),
            "valid": False,
            "reason": str(exc),
            "shape": "",
            "band_count": "",
            "band_axis": "",
            "dtype": "",
            "min": "",
            "max": "",
            "mean": "",
        }


def _write_support_summary(rows: Sequence[Mapping[str, Any]], output: Path, fields: Sequence[str]) -> Path:
    summary: list[dict[str, Any]] = []
    for field in fields:
        counts = Counter(str(row.get(field, "missing") or "missing") for row in rows)
        values = list(counts.values())
        summary.append(
            {
                "field": field,
                "slice_count": len(counts),
                "min_support": min(values) if values else 0,
                "median_support": float(np.median(values)) if values else 0,
                "max_support": max(values) if values else 0,
                "missing_count": counts.get("missing", 0) + counts.get("", 0),
            }
        )
    path = output / "support_summary.csv"
    write_csv(path, summary)
    return path


def prepare_subset(args: argparse.Namespace) -> dict[str, Path]:
    output = ensure_dir(args.output_dir)
    splits = set(args.split or ()) or None
    stratify_fields = tuple(dict.fromkeys(["split", *(args.stratify_field or ("category", "country"))]))
    invalid_fields = [field for field in stratify_fields if field not in SUPPORTED_STRATIFY_FIELDS]
    if invalid_fields:
        raise ValueError(f"Unsupported stratify fields: {', '.join(invalid_fields)}")

    metadata_csv = args.metadata_csv
    if metadata_csv is None:
        if not args.satmae_csv or args.location_geography_csv is None:
            raise ValueError("Provide --metadata-csv, or provide --satmae-csv plus --location-geography-csv to build the sample manifest.")
        metadata_csv = _build_manifest_from_sources(tuple(args.satmae_csv), args.location_geography_csv, output, splits)
    rows = _load_rows(metadata_csv, splits)
    selected = select_rows(rows, stratify_fields, args.max_samples_per_split, args.max_total, args.seed)
    for row in selected:
        row["target_path"] = row["archive_path"]
        row["extracted_image_path"] = row["archive_path"]

    artifacts = {
        "target_paths": output / "target_paths.csv",
        "include_list": output / "include_list.txt",
        "clean_subset_manifest": output / "clean_subset_manifest.csv",
        "support_summary": output / "support_summary.csv",
        "extraction_summary": output / "extraction_summary.csv",
        "raster_validation_report": output / "raster_validation_report.csv",
        "warnings": output / "warnings.json",
    }
    write_csv(artifacts["target_paths"], selected)
    artifacts["include_list"].write_text("\n".join(row["archive_path"] for row in selected) + "\n", encoding="utf-8")

    warnings: list[str] = []
    validation_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    selected_by_path = {str(row["archive_path"]): row for row in selected}
    found_paths: set[str] = set()
    with tarfile.open(args.archive, "r:gz") as tar:
        for member in tar:
            member_name = member.name[2:] if member.name.startswith("./") else member.name
            if not member.isfile() or member_name not in selected_by_path:
                continue
            found_paths.add(member_name)
            destination = _safe_member_path(member_name, output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                warnings.append(f"Could not extract member: {member_name}")
                continue
            with source, destination.open("wb") as handle:
                handle.write(source.read())
            validation = validate_raster(destination)
            validation["archive_path"] = member_name
            validation_rows.append(validation)
            if validation["valid"]:
                row = dict(selected_by_path[member_name])
                row["image_path"] = str(Path(member_name))
                row["extracted_path"] = str(destination)
                clean_rows.append(row)
            else:
                warnings.append(f"Invalid raster excluded: {member_name}: {validation['reason']}")
    missing = sorted(set(selected_by_path) - found_paths)
    warnings.extend(f"Selected archive path not found: {path}" for path in missing[:500])

    write_csv(artifacts["clean_subset_manifest"], clean_rows)
    _write_support_summary(clean_rows, output, stratify_fields)
    write_csv(artifacts["raster_validation_report"], validation_rows)
    summary = [
        {
            "archive": str(args.archive),
            "metadata_csv": str(metadata_csv),
            "output_dir": str(output),
            "seed": args.seed,
            "stratify_fields": ";".join(stratify_fields),
            "requested_rows": len(selected),
            "extracted_members": len(found_paths),
            "valid_rasters": len(clean_rows),
            "missing_members": len(missing),
            "max_samples_per_split": args.max_samples_per_split,
            "max_total": args.max_total or "",
            "geography_metadata_usage": "audit_slicing_only_not_model_input",
            "geography_join_level": "location_level",
        }
    ]
    write_csv(artifacts["extraction_summary"], summary)
    artifacts["warnings"].write_text(json.dumps({"warnings": warnings}, indent=2), encoding="utf-8")
    return artifacts


def _extract_selected_rows(
    archive_path: Path,
    selected: Sequence[dict[str, Any]],
    output: Path,
    validation_report: Path,
    warnings_path: Path,
    progress_every_members: int = 10000,
    progress_every_extracted: int = 500,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int, int]:
    warnings: list[str] = []
    validation_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    selected_by_path = {str(row["archive_path"]): row for row in selected}
    found_paths: set[str] = set()
    scanned_members = 0
    print(f"[extract] target_count={len(selected_by_path)} archive={archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            scanned_members += 1
            member_name = member.name[2:] if member.name.startswith("./") else member.name
            if not member.isfile() or member_name not in selected_by_path:
                if progress_every_members and scanned_members % progress_every_members == 0:
                    print(
                        f"[extract] scanned_members={scanned_members} "
                        f"found={len(found_paths)}/{len(selected_by_path)} valid={len(clean_rows)}"
                    )
                continue
            found_paths.add(member_name)
            destination = _safe_member_path(member_name, output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                warnings.append(f"Could not extract member: {member_name}")
                continue
            with source, destination.open("wb") as handle:
                handle.write(source.read())
            validation = validate_raster(destination)
            validation["archive_path"] = member_name
            validation_rows.append(validation)
            if validation["valid"]:
                row = dict(selected_by_path[member_name])
                row["image_path"] = str(Path(member_name))
                row["extracted_path"] = str(destination)
                clean_rows.append(row)
            else:
                warnings.append(f"Invalid raster excluded: {member_name}: {validation['reason']}")
            if progress_every_extracted and len(found_paths) % progress_every_extracted == 0:
                print(
                    f"[extract] extracted_targets={len(found_paths)}/{len(selected_by_path)} "
                    f"valid={len(clean_rows)} latest={member_name}"
                )
    missing = sorted(set(selected_by_path) - found_paths)
    warnings.extend(f"Selected archive path not found: {path}" for path in missing[:500])
    write_csv(validation_report, validation_rows)
    warnings_path.write_text(json.dumps({"warnings": warnings}, indent=2), encoding="utf-8")
    return clean_rows, validation_rows, warnings, len(found_paths), len(missing)


def augment_subset(args: argparse.Namespace) -> dict[str, Path]:
    output = ensure_dir(args.output_dir)
    splits = set(args.split or ()) or None
    metadata_csv = args.metadata_csv
    if metadata_csv is None:
        if not args.satmae_csv or args.location_geography_csv is None:
            raise ValueError("Provide --metadata-csv, or provide --satmae-csv plus --location-geography-csv to build the sample manifest.")
        metadata_csv = _build_manifest_from_sources(tuple(args.satmae_csv), args.location_geography_csv, output, splits)
    existing_rows = _load_rows(args.augment_existing_manifest, splits)
    metadata_rows = _load_rows(metadata_csv, splits)
    target_by_split = _split_targets(args)
    if not target_by_split:
        raise ValueError("Augmentation requires --target-total or --target-train/--target-val.")
    selected, selection_summary = select_augmentation_rows(
        existing_rows,
        metadata_rows,
        target_by_split,
        args.seed,
        batch_size=args.augmentation_batch_size,
    )
    for row in selected:
        row["target_path"] = row["archive_path"]
        row["extracted_image_path"] = row["archive_path"]

    artifacts = {
        "augmented_clean_subset_manifest": output / "augmented_clean_subset_manifest.csv",
        "augmentation_target_paths": output / "augmentation_target_paths.csv",
        "augmentation_support_before": output / "augmentation_support_before.csv",
        "augmentation_support_after": output / "augmentation_support_after.csv",
        "augmentation_summary": output / "augmentation_summary.json",
        "raster_validation_report_augmented": output / "raster_validation_report_augmented.csv",
        "warnings_augmented": output / "warnings_augmented.json",
    }
    _write_support_by_value(existing_rows, artifacts["augmentation_support_before"], SUPPORT_AUGMENT_FIELDS)
    write_csv(artifacts["augmentation_target_paths"], selected)
    clean_added, _validation_rows, warnings, found_count, missing_count = _extract_selected_rows(
        args.archive,
        selected,
        output,
        artifacts["raster_validation_report_augmented"],
        artifacts["warnings_augmented"],
        progress_every_members=args.progress_every_members,
        progress_every_extracted=args.progress_every_extracted,
    )
    combined = [dict(row) for row in existing_rows] + clean_added
    combined.sort(key=lambda row: (str(row.get("split", "")), str(row.get("archive_path", row.get("image_path", "")))))
    _write_csv_union(artifacts["augmented_clean_subset_manifest"], combined)
    _write_support_by_value(combined, artifacts["augmentation_support_after"], SUPPORT_AUGMENT_FIELDS)
    final_by_split = Counter(str(row.get("split", "")) for row in combined)
    summary = {
        "mode": "augment_existing_clean_subset",
        "archive": str(args.archive),
        "metadata_csv": str(metadata_csv),
        "existing_manifest": str(args.augment_existing_manifest),
        "output_dir": str(output),
        "seed": args.seed,
        "target_total": args.target_total or "",
        "target_by_split": target_by_split,
        "existing_total": len(existing_rows),
        "selected_additional_targets": len(selected),
        "extracted_additional_members": found_count,
        "valid_added_samples": len(clean_added),
        "final_total": len(combined),
        "final_by_split": dict(final_by_split),
        "missing_members": missing_count,
        "warnings_count": len(warnings),
        **selection_summary,
    }
    artifacts["augmentation_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a clean fMoW-Sentinel subset from the official tar.gz without full extraction.")
    parser.add_argument("--archive", type=Path, required=True, help="Local fmow-sentinel.tar.gz archive.")
    parser.add_argument("--metadata-csv", type=Path, help="Final enriched sample manifest CSV.")
    parser.add_argument("--augment-existing-manifest", type=Path, help="Existing clean_subset_manifest.csv to preserve and augment.")
    parser.add_argument("--satmae-csv", action="append", type=Path, default=[], help="SatMAE train/val/test CSV used only when --metadata-csv is missing.")
    parser.add_argument("--location-geography-csv", type=Path, help="Final location-level geography metadata used only when --metadata-csv is missing.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", default=[], help="Split to include, e.g. train or val. Repeat as needed.")
    parser.add_argument("--max-samples-per-split", type=int, default=500, help="Quota cap after split filtering.")
    parser.add_argument("--max-total", type=int, help="Optional cap across all selected splits.")
    parser.add_argument("--target-total", type=int, help="Final target size for augmentation mode, e.g. 30000.")
    parser.add_argument("--target-train", type=int, help="Final train target for augmentation mode.")
    parser.add_argument("--target-val", type=int, help="Final val target for augmentation mode.")
    parser.add_argument("--augmentation-batch-size", type=int, default=1000, help="Support-aware augmentation selection batch size.")
    parser.add_argument("--progress-every-members", type=int, default=10000, help="Print tar scan progress every N archive members during augmentation extraction.")
    parser.add_argument("--progress-every-extracted", type=int, default=500, help="Print extraction progress every N matched targets during augmentation extraction.")
    parser.add_argument("--stratify-field", action="append", default=[], help="Additional stratification field. Defaults to category and country.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = augment_subset(args) if args.augment_existing_manifest else prepare_subset(args)
    print("fMoW-Sentinel clean subset prepared.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
