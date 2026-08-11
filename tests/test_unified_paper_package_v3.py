from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rsfm_fairness_audit.io import read_csv_rows
from scripts.analysis.build_fmow_protocol_contrast_v1 import build_fmow_protocol_contrast
from scripts.analysis.build_unified_paper_package_v3 import build_unified_paper_package
from scripts.analysis.build_unified_paper_package_v3 import _main_result_rows, _evidence_policy


def test_unified_package_output_existence_smoke() -> None:
    protocol_out = Path("outputs") / f"test_protocol_for_unified_v3_{uuid4().hex}"
    build_fmow_protocol_contrast(protocol_out)
    out = Path("outputs") / f"test_unified_paper_package_v3_{uuid4().hex}"
    artifacts = build_unified_paper_package(out, protocol_root=protocol_out)
    required = [
        "paper_ready_main_findings_v3",
        "claim_support_table_v3",
        "experiment_status_matrix_v3",
        "metric_scope_and_caveat_matrix_v3",
        "unified_results_narrative_table_v3",
        "bwer_robustness_summary_v3",
        "fmow_selective_social_spatial_summary_v3",
        "fmow_protocol_contrast_summary_v3",
        "fmow_random_split_social_spatial_summary_v3",
        "next_experiment_readiness_alphaearth_gee",
        "manuscript_outline_v3",
        "thesis_chapter_outline_v3",
    ]
    for key in required:
        assert artifacts[key].exists(), key


def test_unified_package_tables_and_figures_smoke() -> None:
    protocol_out = Path("outputs") / f"test_protocol_for_unified_fig_v3_{uuid4().hex}"
    build_fmow_protocol_contrast(protocol_out)
    out = Path("outputs") / f"test_unified_paper_package_fig_v3_{uuid4().hex}"
    artifacts = build_unified_paper_package(out, protocol_root=protocol_out)
    claims = read_csv_rows(artifacts["claim_support_table_v3"])
    assert any(row["claim_id"] == "C2" and "protocol" in row["formal_status"] for row in claims)
    assert artifacts["figure_fmow_protocol_contrast_v3_png"].exists()
    assert artifacts["figure_claim_strength_matrix_v3_pdf"].exists()


def test_current_evidence_policy_retains_valid_rankings_without_overclaiming() -> None:
    rows = _main_result_rows()
    assert rows
    assert all(row["estimate_validity"] == "valid" for row in rows)
    assert all(row["presentation_role"] == "main_benchmark" for row in rows)
    policy = _evidence_policy()
    valid = [row for row in policy if row.get("estimate_validity") == "valid"]
    assert valid and all(row.get("retain_result") is True for row in valid)
    alpha = [row for row in policy if row["task"] == "AlphaEarth"]
    assert any(row["presentation_role"] == "main_benchmark" for row in alpha)
