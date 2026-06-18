from __future__ import annotations

import uuid
from pathlib import Path

from rsfm_fairness_audit.io import read_csv_rows
from scripts.analysis.build_fmow_social_spatial_interpretation_v1_1 import (
    association_rows,
    build_fmow_social_spatial_v1_1,
    join_indicators,
)


def test_indicator_join_v1_1_smoke() -> None:
    rows = [{"country": "AAA", "support_count": 20, "mean_risk": 0.5, "scenario": "baseline", "run_id": "r"}]
    indicators = {"AAA": {"gdp_per_capita": 1000, "population_density": 10, "urban_population_share": 50, "income_group": "Low income"}}

    joined = join_indicators(rows, indicators, "available")

    assert joined[0]["indicator_status"] == "matched"
    assert joined[0]["gdp_per_capita"] == 1000


def test_association_table_v1_1_smoke() -> None:
    rows = [
        {"run_id": "r", "scenario": "baseline", "support_count": 20, "mean_risk": 0.5, "gdp_per_capita": 1000, "population_density": 10, "urban_population_share": 50},
        {"run_id": "r", "scenario": "baseline", "support_count": 20, "mean_risk": 0.4, "gdp_per_capita": 2000, "population_density": 20, "urban_population_share": 60},
        {"run_id": "r", "scenario": "baseline", "support_count": 20, "mean_risk": 0.3, "gdp_per_capita": 3000, "population_density": 30, "urban_population_share": 70},
    ]

    assoc = association_rows(rows, min_support=20)

    assert {row["indicator"] for row in assoc} == {"gdp_per_capita", "population_density", "urban_population_share"}
    assert all(row["association_status"] == "available" for row in assoc)


def test_v1_1_figure_generation_smoke() -> None:
    root = Path("outputs") / f"test_fmow_social_v1_1_{uuid.uuid4().hex}"
    input_dir = root / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "fmow_country_risk_summary.csv").write_text(
        "run_id,scenario,country,slice_value,support_count,mean_risk,risk_excess_vs_overall,support_ok\n"
        "resnet50_13band,baseline,AAA,AAA,25,0.5,0.1,True\n"
        "resnet50_13band,baseline,BBB,BBB,25,0.4,0.0,True\n"
        "resnet50_13band,baseline,CCC,CCC,25,0.3,-0.1,True\n",
        encoding="utf-8",
    )
    indicators = root / "indicators.csv"
    indicators.write_text(
        "iso3,gdp_per_capita,population_density,urban_population_share,income_group\n"
        "AAA,1000,10,50,Low\n"
        "BBB,2000,20,60,Middle\n"
        "CCC,3000,30,70,High\n",
        encoding="utf-8",
    )

    artifacts = build_fmow_social_spatial_v1_1(input_dir=input_dir, output_dir=root / "out", indicator_csv=indicators, min_support=20, unified_v2_dir=root / "unified")

    assert artifacts["fmow_social_indicator_join_v1_1"].exists()
    assert artifacts["figure_bwer_vs_gdp_per_capita_scatter_png"].exists()
    assert artifacts["unified_v2_fmow_social_spatial_indicator_association_v1_1_summary"].exists()
    rows = read_csv_rows(artifacts["fmow_risk_indicator_association_v1_1"])
    assert any(row["association_status"] == "available" for row in rows)
