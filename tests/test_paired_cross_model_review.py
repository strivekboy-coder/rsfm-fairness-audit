from __future__ import annotations

import csv
import json
from pathlib import Path

from rsfm_fairness_audit.paired_cross_model_review import build_cross_model_review, compare_paired_results, read_paired_result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _result(root: Path, *, delta_geo: float, ppr: float, signature: str) -> None:
    root.mkdir(parents=True)
    (root / "paired_shift_result_audit.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    _write_csv(root / "paired_shift_delta_seed_panel.csv", [{
        "seed": 42, "delta_mean_risk": .2, "delta_tail_risk": .25,
        "delta_geobwer": delta_geo, "tail_acceleration": delta_geo,
    }])
    _write_csv(root / "paired_probability_seed_summary.csv", [{
        "seed": 42, "mean_delta_auroc": -.3, "mean_delta_ap": -.4,
        "mean_delta_f1_at_locked_threshold": -.35, "mean_probability_wasserstein_1": .2,
        "mean_threshold_crossing_rate": .3, "mean_mean_absolute_paired_probability_shift": .31,
    }])
    _write_csv(root / "paired_probability_label_summary.csv", [{
        "class_label": "label_a", "mean_delta_auroc": -.3, "mean_delta_ap": -.4,
        "mean_delta_predicted_positive_rate": ppr, "mean_threshold_crossing_rate": .3,
        "modal_diagnostic_signature": signature,
    }])


def test_cross_model_failure_geometry_is_scoped_and_detected(tmp_path: Path) -> None:
    _result(tmp_path / "a", delta_geo=.05, ppr=.8, signature="representation_collapse_signature")
    _result(tmp_path / "b", delta_geo=.00, ppr=-.2, signature="mixed_or_partial_degradation")
    result = compare_paired_results([
        read_paired_result("A", tmp_path / "a"), read_paired_result("B", tmp_path / "b")
    ])
    assert result["claim_assessment"]["supported"] is True
    assert result["claim_assessment"]["opposite_score_transport_label_count"] == 1
    assert "not causal" in result["claim_assessment"]["scope"]
    built = build_cross_model_review({"A": tmp_path / "a", "B": tmp_path / "b"}, tmp_path / "review")
    assert built["status"] == "pass"
    assert (tmp_path / "review" / "paired_cross_model_metric_comparison.png").is_file()
