from __future__ import annotations

import uuid
from pathlib import Path

from rsfm_fairness_audit.io import read_csv_rows
from scripts.analysis.build_fmow_social_spatial_interpretation_v1 import (
    _make_country_plot,
    build_indicator_join,
    build_fmow_social_spatial_interpretation,
    support_filter_rows,
)


def _write_minimal_selective_outputs(root: Path) -> Path:
    selective = root / "selective"
    selective.mkdir(parents=True)
    (selective / "fmow_selective_bwer_summary.csv").write_text(
        "run_id,selector,coverage_target,analysis_type,slice_variable,balance_variable,bwer,mean_slice_risk,tail_risk,worst_slice,tail_slices,n_valid_slices,note\n"
        "resnet50_13band,baseline_all_test,1.0,raw,country,,0.2,0.7,0.9,B, B,2,n\n",
        encoding="utf-8",
    )
    (selective / "fmow_conformal_bwer_summary.csv").write_text(
        "run_id,selector,coverage_target,analysis_type,slice_variable,balance_variable,bwer,mean_slice_risk,tail_risk,worst_slice,tail_slices,n_valid_slices,note\n",
        encoding="utf-8",
    )
    (selective / "fmow_selective_risk_summary.csv").write_text(
        "run_id,selector,coverage_target,mean_risk\n"
        "resnet50_13band,baseline_all_test,1.0,0.5\n"
        "resnet50_13band,confidence_topk_test,0.7,0.4\n"
        "resnet50_13band,calibrated_confidence_threshold_diagnostic,0.7,0.45\n",
        encoding="utf-8",
    )
    (selective / "rank_divergence_under_selective_audit.csv").write_text(
        "scenario,selector,coverage_target,analysis_type,aggregate_best_run,aggregate_best_mean_risk,bwer_best_run,bwer_best,rank_diverges,compared_runs,interpretation\n"
        "tiny,baseline_all_test,1.0,raw,resnet50_13band,0.5,dofa_scaled10000,0.1,True,dofa_scaled10000;resnet50_13band,diverges\n",
        encoding="utf-8",
    )
    return selective


def test_support_filter_rows_smoke() -> None:
    rows = [{"support_count": 5}, {"support_count": 25}]
    assert len(support_filter_rows(rows, min_support=20)) == 1


def test_indicator_join_smoke() -> None:
    root = Path("outputs") / f"test_fmow_indicator_join_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    indicators = root / "indicators.csv"
    indicators.write_text("iso3,gdp_per_capita,population_density,income_group\nAAA,1000,20,Low\n", encoding="utf-8")
    country_rows = [{"slice_variable": "country", "country": "AAA", "scenario": "baseline", "support_ok": True, "mean_risk": 0.5}]

    joined, assoc, status = build_indicator_join(country_rows, indicators)

    assert status == "available_user_supplied_indicator_csv"
    assert joined[0]["gdp_per_capita"] == "1000"
    assert assoc


def test_country_plot_generation_smoke() -> None:
    root = Path("outputs") / f"test_fmow_country_plot_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    rows = [{"country": "AAA", "slice_value": "AAA", "support_ok": True, "mean_risk": 0.5, "is_bwer_tail_slice": True}]
    path = root / "fig" / "country_bwer_choropleth"

    _make_country_plot(rows, path, title="Tiny")

    assert path.with_suffix(".png").exists()
    assert path.with_suffix(".pdf").exists()


def test_social_spatial_builder_schema_smoke(monkeypatch) -> None:
    root = Path("outputs") / f"test_fmow_social_spatial_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    audit = root / "audit.csv"
    audit.write_text(
        "sample_id,location_id,country,region,class_label,risk,confidence,max_probability,correct\n"
        "a,l1,A,R1,x,0,0.95,0.95,True\n"
        "b,l1,A,R1,y,1,0.20,0.20,False\n"
        "c,l2,B,R2,x,1,0.30,0.30,False\n"
        "d,l2,B,R2,y,0,0.90,0.90,True\n",
        encoding="utf-8",
    )
    selective = _write_minimal_selective_outputs(root)

    monkeypatch.setattr("scripts.analysis.build_fmow_social_spatial_interpretation_v1.discover_fmow_audit_tables", lambda: {"resnet50_13band": audit})
    artifacts = build_fmow_social_spatial_interpretation(
        output_dir=root / "out",
        fmow_selective_dir=selective,
        unified_v2_dir=root / "unified",
        min_support=1,
    )

    rows = read_csv_rows(artifacts["fmow_country_risk_summary"])
    assert {"run_id", "scenario", "country", "support_count", "mean_risk", "support_ok"}.issubset(rows[0])
    assert artifacts["figure_country_bwer_choropleth_png"].exists()
