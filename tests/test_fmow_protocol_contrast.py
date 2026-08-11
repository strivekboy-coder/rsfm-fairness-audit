from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rsfm_fairness_audit.io import read_csv_rows
from scripts.analysis.build_fmow_protocol_contrast_v1 import build_fmow_protocol_contrast


def test_protocol_contrast_schema_smoke() -> None:
    out = Path("outputs") / f"test_fmow_protocol_contrast_{uuid4().hex}"
    artifacts = build_fmow_protocol_contrast(out)
    rows = read_csv_rows(artifacts["fmow_protocol_contrast_bwer_summary"])
    assert rows
    assert {"run_id", "model_family", "protocol", "protocol_status", "accuracy", "country_raw_bwer"}.issubset(rows[0])
    assert {row["protocol_status"] for row in rows} == {
        "valid_benchmark_formal_partial", "sanity_protocol_contrast"
    }


def test_protocol_contrast_figure_smoke() -> None:
    out = Path("outputs") / f"test_fmow_protocol_contrast_fig_{uuid4().hex}"
    artifacts = build_fmow_protocol_contrast(out)
    assert artifacts["figure_random_vs_location_accuracy_png"].exists()
    assert artifacts["figure_random_vs_location_bwer_pdf"].exists()
