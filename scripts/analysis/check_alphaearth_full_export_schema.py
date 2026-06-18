from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


DEFAULT_GEE_ROOT = Path("outputs/alphaearth_gee_full_v1")
DEFAULT_EXPORT = DEFAULT_GEE_ROOT / "alphaearth_worldcover_full_export.csv"
DEFAULT_MANIFEST = DEFAULT_GEE_ROOT / "alphaearth_worldcover_full_export_manifest.csv"
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
REQUIRED_COLUMNS = [
    "sample_id",
    "lon",
    "lat",
    "year",
    "country_iso3",
    "spatial_block_id",
    "split",
    "worldcover_label",
    "worldcover_class_name",
    *EMBEDDING_BANDS,
]
RECOMMENDED_COLUMNS = ["region", "income_group", "biome_or_ecoregion", "urban_rural_or_built_proxy"]
OPTIONAL_COLUMNS = ["dynamic_world_label", "dynamic_world_confidence"]


def read_alphaearth_full_export(input_csv: Path = DEFAULT_EXPORT, manifest_csv: Path = DEFAULT_MANIFEST) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    source_rows: list[dict[str, Any]] = []
    if input_csv.exists():
        rows = read_csv_rows(input_csv)
        source_rows.append({"source_type": "merged_csv", "path": str(input_csv), "status": "available", "rows": len(rows)})
        return rows, source_rows
    if manifest_csv.exists():
        rows: list[dict[str, str]] = []
        for shard in read_csv_rows(manifest_csv):
            path = Path(shard.get("path", ""))
            if not path.is_absolute():
                path = manifest_csv.parent / path
            if shard.get("status", "available") not in {"available", "complete", "completed"}:
                source_rows.append({**shard, "source_type": "shard", "resolved_path": str(path), "rows": 0})
                continue
            if not path.exists():
                source_rows.append({**shard, "source_type": "shard", "resolved_path": str(path), "status": "missing", "rows": 0})
                continue
            shard_rows = read_csv_rows(path)
            rows.extend(shard_rows)
            source_rows.append({**shard, "source_type": "shard", "resolved_path": str(path), "status": "available", "rows": len(shard_rows)})
        return rows, source_rows
    return [], [{"source_type": "missing", "path": str(input_csv), "manifest": str(manifest_csv), "status": "missing", "rows": 0}]


def _missing_columns(columns: Sequence[str]) -> list[str]:
    present = set(columns)
    return [column for column in REQUIRED_COLUMNS if column not in present]


def _support_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    dimensions = ["split", "country_iso3", "region", "worldcover_class_name", "country_iso3|worldcover_class_name", "region|worldcover_class_name", "biome_or_ecoregion", "urban_rural_or_built_proxy", "income_group"]
    for dim in dimensions:
        counts: dict[str, int] = {}
        for row in rows:
            if "|" in dim:
                parts = dim.split("|")
                if any(str(row.get(part, "")).strip() == "" for part in parts):
                    continue
                value = "|".join(str(row.get(part)) for part in parts)
            else:
                if dim not in row or str(row.get(dim, "")).strip() == "":
                    continue
                value = str(row.get(dim))
            counts[value] = counts.get(value, 0) + 1
        for value, count in sorted(counts.items()):
            target = 50 if "|" in dim else 200 if dim == "country_iso3" else 1
            output.append(
                {
                    "slice_variable": dim,
                    "slice_value": value,
                    "sample_count": count,
                    "target_support": target,
                    "support_status": "ok" if count >= target else "below_target",
                }
            )
    return output


def check_alphaearth_full_export_schema(
    input_csv: Path = DEFAULT_EXPORT,
    manifest_csv: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_GEE_ROOT,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    rows, source_rows = read_alphaearth_full_export(input_csv, manifest_csv)
    columns = list(rows[0].keys()) if rows else []
    missing = _missing_columns(columns)
    optional_present = [column for column in OPTIONAL_COLUMNS if column in columns]
    recommended_present = [column for column in RECOMMENDED_COLUMNS if column in columns]
    status = "ok" if rows and not missing else "missing_export" if not rows else "invalid_schema"
    support_rows = _support_rows(rows)
    status_rows = [
        {
            "input_csv": str(input_csv),
            "manifest_csv": str(manifest_csv),
            "schema_status": status,
            "n_rows": len(rows),
            "n_countries": len({row.get("country_iso3") for row in rows if row.get("country_iso3")}),
            "n_classes": len({row.get("worldcover_class_name") for row in rows if row.get("worldcover_class_name")}),
            "missing_required_columns": ";".join(missing),
            "recommended_columns_present": ";".join(recommended_present),
            "optional_columns_present": ";".join(optional_present),
        }
    ]
    artifacts = {
        "alphaearth_worldcover_full_export_manifest": output / "alphaearth_worldcover_full_export_manifest.csv",
        "alphaearth_full_export_schema_report": output / "alphaearth_full_export_schema_report.md",
        "alphaearth_full_support_preflight": output / "alphaearth_full_support_preflight.csv",
        "alphaearth_full_sampling_report": output / "alphaearth_full_sampling_report.md",
        "alphaearth_full_export_schema_status": output / "alphaearth_full_export_schema_status.csv",
    }
    if not artifacts["alphaearth_worldcover_full_export_manifest"].exists():
        write_csv(artifacts["alphaearth_worldcover_full_export_manifest"], source_rows)
    write_csv(artifacts["alphaearth_full_support_preflight"], support_rows or [{"status": "unavailable_missing_export"}])
    write_csv(artifacts["alphaearth_full_export_schema_status"], status_rows)
    artifacts["alphaearth_full_export_schema_report"].write_text(
        "# AlphaEarth full export schema report\n\n"
        f"- Schema status: {status}\n"
        f"- Rows: {len(rows)}\n"
        f"- Countries: {status_rows[0]['n_countries']}\n"
        f"- Classes: {status_rows[0]['n_classes']}\n"
        f"- Missing required columns: {', '.join(missing) if missing else 'none'}\n"
        f"- Recommended columns present: {', '.join(recommended_present) if recommended_present else 'none'}\n"
        f"- Optional confidence columns present: {', '.join(optional_present) if optional_present else 'none'}\n\n"
        "Full formal claims require sufficient support; see `alphaearth_full_support_preflight.csv`.\n",
        encoding="utf-8",
    )
    blockers = []
    if len(rows) < 100000:
        blockers.append(f"sample_count {len(rows)} below minimum formal target 100000")
    countries = len({row.get("country_iso3") for row in rows if row.get("country_iso3")})
    if countries < 100:
        blockers.append(f"country_count {countries} below target 100-120")
    artifacts["alphaearth_full_sampling_report"].write_text(
        "# AlphaEarth full sampling report\n\n"
        f"- Export source rows: {len(source_rows)}\n"
        f"- Total sample rows: {len(rows)}\n"
        f"- Quota/support blockers: {'; '.join(blockers) if blockers else 'none at schema stage'}\n"
        "- Do not silently shrink the run; report actual support and quota blockers.\n",
        encoding="utf-8",
    )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Check AlphaEarth full export schema and support preflight.")
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_GEE_ROOT)
    args = parser.parse_args()
    artifacts = check_alphaearth_full_export_schema(args.input, args.manifest, args.out)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
