from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rsfm_fairness_audit.io import read_csv_rows


def test_read_csv_rows_accepts_long_warning_field() -> None:
    root = Path("outputs") / f"test_io_csv_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "long.csv"
    long_warning = "warning-" + ("x" * 200_000)
    path.write_text(f"slice_variable,warnings\ncountry,{long_warning}\n", encoding="utf-8")

    rows = read_csv_rows(path)

    assert rows[0]["slice_variable"] == "country"
    assert rows[0]["warnings"] == long_warning

