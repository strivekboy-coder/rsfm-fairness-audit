from __future__ import annotations

import argparse
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


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).lower() in {"nan", "none", "null"}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a clean fMoW-Sentinel subset from the official tar.gz without full extraction.")
    parser.add_argument("--archive", type=Path, required=True, help="Local fmow-sentinel.tar.gz archive.")
    parser.add_argument("--metadata-csv", type=Path, help="Final enriched sample manifest CSV.")
    parser.add_argument("--satmae-csv", action="append", type=Path, default=[], help="SatMAE train/val/test CSV used only when --metadata-csv is missing.")
    parser.add_argument("--location-geography-csv", type=Path, help="Final location-level geography metadata used only when --metadata-csv is missing.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", default=[], help="Split to include, e.g. train or val. Repeat as needed.")
    parser.add_argument("--max-samples-per-split", type=int, default=500, help="Quota cap after split filtering.")
    parser.add_argument("--max-total", type=int, help="Optional cap across all selected splits.")
    parser.add_argument("--stratify-field", action="append", default=[], help="Additional stratification field. Defaults to category and country.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    artifacts = prepare_subset(build_parser().parse_args())
    print("fMoW-Sentinel clean subset prepared.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
