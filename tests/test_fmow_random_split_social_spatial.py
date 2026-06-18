from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rsfm_fairness_audit.io import read_csv_rows
from scripts.analysis.build_fmow_random_split_social_spatial_v1 import build_fmow_random_split_social_spatial


def test_random_split_social_spatial_association_schema_smoke() -> None:
    out = Path("outputs") / f"test_fmow_random_split_social_{uuid4().hex}"
    artifacts = build_fmow_random_split_social_spatial(out, unified_v3_dir=None)
    rows = read_csv_rows(artifacts["fmow_random_split_risk_indicator_association"])
    assert rows
    assert {"run_id", "protocol", "indicator", "n_countries", "pearson_r", "claim_scope"}.issubset(rows[0])
    assert {row["protocol"] for row in rows} == {"random_split_sanity"}
    assert all("not deployment evidence" in row["claim_scope"] for row in rows)


def test_random_split_social_spatial_figure_smoke() -> None:
    out = Path("outputs") / f"test_fmow_random_split_social_fig_{uuid4().hex}"
    artifacts = build_fmow_random_split_social_spatial(out, unified_v3_dir=None)
    assert artifacts["figure_random_split_bwer_vs_gdp_per_capita_scatter_png"].exists()
    assert artifacts["figure_random_split_indicator_association_summary_pdf"].exists()
