from __future__ import annotations

import csv
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


def raise_csv_field_size_limit() -> int:
    """Raise the CSV parser field limit for long warning/provenance columns."""
    limit = sys.maxsize
    while True:
        try:
            return csv.field_size_limit(limit)
        except OverflowError:
            limit //= 10


raise_csv_field_size_limit()


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_csv(path: str | Path, rows: Sequence[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    materialized = [asdict(row) if is_dataclass(row) else dict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    raise_csv_field_size_limit()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
