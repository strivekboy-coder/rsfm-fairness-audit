from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rsfm_fairness_audit.io import read_csv_rows, write_csv


def test_read_csv_rows_accepts_long_warning_field() -> None:
    root = Path("outputs") / f"test_io_csv_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "long.csv"
    long_warning = "warning-" + ("x" * 200_000)
    path.write_text(f"slice_variable,warnings\ncountry,{long_warning}\n", encoding="utf-8")

    rows = read_csv_rows(path)

    assert rows[0]["slice_variable"] == "country"
    assert rows[0]["warnings"] == long_warning


def test_write_csv_accepts_rows_with_late_extra_fields() -> None:
    root = Path("outputs") / f"test_io_csv_{uuid4().hex}"
    path = root / "ragged.csv"
    write_csv(path, [{"a": 1}, {"a": 2, "cross_run_mode_bwer": 0.3}])

    rows = read_csv_rows(path)

    assert rows[0]["cross_run_mode_bwer"] == ""
    assert rows[1]["cross_run_mode_bwer"] == "0.3"
