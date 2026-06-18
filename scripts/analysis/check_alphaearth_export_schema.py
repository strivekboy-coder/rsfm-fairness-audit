from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


DEFAULT_INPUT = Path("outputs/alphaearth_gee_pilot_v1/alphaearth_worldcover_pilot_export.csv")
DEFAULT_OUTPUT = Path("outputs/alphaearth_gee_pilot_v1")
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
OPTIONAL_COLUMNS = [
    "region",
    "income_group",
    "biome_or_ecoregion",
    "urban_rural_or_built_proxy",
    "dynamic_world_label",
    "dynamic_world_confidence",
]


def _missing_columns(columns: Sequence[str]) -> list[str]:
    present = set(columns)
    return [column for column in REQUIRED_COLUMNS if column not in present]


def _empty_required_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {column: 0 for column in REQUIRED_COLUMNS}
    for row in rows:
        for column in REQUIRED_COLUMNS:
            if str(row.get(column, "")).strip() == "":
                counts[column] += 1
    return counts


def check_alphaearth_export_schema(input_csv: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> tuple[dict[str, Any], Path]:
    output = ensure_dir(output_dir)
    report_path = output / "alphaearth_export_schema_report.md"
    if not input_csv.exists():
        status = {
            "input_csv": str(input_csv),
            "schema_status": "missing_export_table",
            "n_rows": 0,
            "missing_required_columns": ";".join(REQUIRED_COLUMNS),
            "optional_columns_present": "",
        }
        report_path.write_text(
            "# AlphaEarth export schema report\n\n"
            f"Export table is missing: `{input_csv}`.\n\n"
            "Required columns:\n\n"
            + "\n".join(f"- `{column}`" for column in REQUIRED_COLUMNS)
            + "\n\nNo model training or BWER audit was run.\n",
            encoding="utf-8",
        )
        return status, report_path
    rows = read_csv_rows(input_csv)
    columns = list(rows[0].keys()) if rows else []
    missing = _missing_columns(columns)
    optional_present = [column for column in OPTIONAL_COLUMNS if column in columns]
    empty_counts = _empty_required_counts(rows) if rows else {column: 0 for column in REQUIRED_COLUMNS}
    empty_required = [f"{column}:{count}" for column, count in empty_counts.items() if count > 0]
    status_name = "ok" if rows and not missing and not empty_required else "invalid"
    status = {
        "input_csv": str(input_csv),
        "schema_status": status_name,
        "n_rows": len(rows),
        "missing_required_columns": ";".join(missing),
        "optional_columns_present": ";".join(optional_present),
        "empty_required_counts": ";".join(empty_required),
    }
    report_path.write_text(
        "# AlphaEarth export schema report\n\n"
        f"- Input table: `{input_csv}`\n"
        f"- Rows: {len(rows)}\n"
        f"- Schema status: {status_name}\n"
        f"- Missing required columns: {', '.join(missing) if missing else 'none'}\n"
        f"- Empty required field counts: {', '.join(empty_required) if empty_required else 'none'}\n"
        f"- Optional columns present: {', '.join(optional_present) if optional_present else 'none'}\n\n"
        "This checker validates table shape only; it does not train models or make empirical AlphaEarth claims.\n",
        encoding="utf-8",
    )
    write_csv(output / "alphaearth_export_schema_status.csv", [status])
    return status, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check AlphaEarth/GEE pilot export table schema.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    status, report = check_alphaearth_export_schema(args.input, args.out)
    print(f"schema_status: {status['schema_status']}")
    print(f"schema_report: {report}")
    if status["schema_status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
