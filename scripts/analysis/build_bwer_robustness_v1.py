from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, write_csv  # noqa: E402


DEFAULT_REGISTRY = Path("configs/analysis/unified_audit_registry.yaml")
DEFAULT_ALPHAS = (0.05, 0.10, 0.20)
DEFAULT_SUPPORT_THRESHOLDS = (10, 20, 30)
DEFAULT_MISSING_POLICIES = ("renormalize", "overlap", "invalidate")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load the audit registry.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "experiments" not in data:
        raise ValueError(f"Invalid audit registry: {path}")
    return data


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _score_to_risk(score: Any, metric_family: str) -> float:
    value = _float(score)
    if math.isnan(value):
        return value
    if metric_family in {"iou_risk", "classification_error"}:
        return 1.0 - value
    return value


def _experiment_runs(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in registry.get("experiments", []):
        metric_family = str(exp.get("primary_metric_family", ""))
        for run in exp.get("formal_runs", []) or []:
            aggregate_score = _float(run.get("aggregate_score"))
            aggregate_risk = _score_to_risk(aggregate_score, metric_family)
            if metric_family == "bce_risk" and not math.isnan(_float(run.get("mean_bce_risk"))):
                aggregate_risk = _float(run.get("mean_bce_risk"))
            raw_bwer = _float(run.get("raw_bwer"))
            std_bwer = _float(run.get("standardised_bwer"))
            rows.append(
                {
                    "experiment_id": exp.get("experiment_id", ""),
                    "dataset": exp.get("dataset", ""),
                    "task_type": exp.get("task_type", ""),
                    "deployment_axis": exp.get("deployment_axis", ""),
                    "run_id": run.get("run_id", ""),
                    "model_family": run.get("model_family", ""),
                    "metric_family": metric_family,
                    "aggregate_score": "" if math.isnan(aggregate_score) else aggregate_score,
                    "aggregate_risk": "" if math.isnan(aggregate_risk) else aggregate_risk,
                    "raw_bwer": "" if math.isnan(raw_bwer) else raw_bwer,
                    "standardised_bwer": "" if math.isnan(std_bwer) else std_bwer,
                    "primary_bwer_slice": run.get("raw_bwer_slice", exp.get("primary_bwer_slice", "")),
                    "standardised_balance": exp.get("standardised_balance", ""),
                    "worst_slice": run.get("worst_slice", ""),
                    "tail_slices": run.get("tail_slices", ""),
                    "source": run.get("data_source", "registry"),
                }
            )
    return rows


def _comparison_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        raw = _float(row.get("raw_bwer"))
        std = _float(row.get("standardised_bwer"))
        aggregate_risk = _float(row.get("aggregate_risk"))
        has_raw = not math.isnan(raw)
        has_std = not math.isnan(std)
        rows.append(
            {
                **dict(row),
                "has_raw_bwer": has_raw,
                "has_standardised_bwer": has_std,
                "stabilised_bwer": "",
                "mean_slice_risk": "",
                "tail_risk": "",
                "max_min_gap": "",
                "comparison_status": "available_from_registry_summary" if has_raw or has_std else "missing_bwer_summary",
                "caveat": "Per-slice risks are unavailable in the registry; robustness rows are summary-level unless source artifacts are added.",
                "aggregate_risk_for_context": "" if math.isnan(aggregate_risk) else aggregate_risk,
            }
        )
    return rows


def _traditional_metric_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        aggregate_risk = _float(row.get("aggregate_risk"))
        raw = _float(row.get("raw_bwer"))
        rows.append(
            {
                "experiment_id": row.get("experiment_id", ""),
                "run_id": row.get("run_id", ""),
                "aggregate_risk": "" if math.isnan(aggregate_risk) else aggregate_risk,
                "mean_slice_risk": "",
                "worst_slice_risk": "",
                "max_min_gap": "",
                "group_std": "",
                "coefficient_of_variation": "",
                "worst_group_accuracy_or_risk": "",
                "cvar_tail_risk": "",
                "raw_bwer_for_context": "" if math.isnan(raw) else raw,
                "status": "requires_slice_risk_table" if math.isnan(raw) else "summary_context_only",
            }
        )
    return rows


def _alpha_rows(run_rows: Sequence[Mapping[str, Any]], alphas: Sequence[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        raw = _float(row.get("raw_bwer"))
        for alpha in alphas:
            rows.append(
                {
                    "experiment_id": row.get("experiment_id", ""),
                    "run_id": row.get("run_id", ""),
                    "alpha": alpha,
                    "bwer": "" if math.isnan(raw) else raw if abs(alpha - 0.10) < 1e-9 else "",
                    "status": "available_primary_alpha" if not math.isnan(raw) and abs(alpha - 0.10) < 1e-9 else "requires_recomputable_slice_table",
                }
            )
    return rows


def _support_rows(run_rows: Sequence[Mapping[str, Any]], thresholds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        raw = _float(row.get("raw_bwer"))
        for threshold in thresholds:
            rows.append(
                {
                    "experiment_id": row.get("experiment_id", ""),
                    "run_id": row.get("run_id", ""),
                    "min_support": threshold,
                    "bwer": "" if math.isnan(raw) else raw if threshold == 20 else "",
                    "status": "available_default_threshold_or_documented" if not math.isnan(raw) and threshold == 20 else "requires_support_and_slice_risk_table",
                }
            )
    return rows


def _missing_policy_rows(run_rows: Sequence[Mapping[str, Any]], policies: Sequence[str], default_policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        std = _float(row.get("standardised_bwer"))
        for policy in policies:
            rows.append(
                {
                    "experiment_id": row.get("experiment_id", ""),
                    "run_id": row.get("run_id", ""),
                    "missing_policy": policy,
                    "standardised_bwer": "" if math.isnan(std) else std if policy == default_policy else "",
                    "status": "available_default_policy" if not math.isnan(std) and policy == default_policy else "requires_balance_cell_table",
                }
            )
    return rows


def _bootstrap_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": row.get("experiment_id", ""),
            "run_id": row.get("run_id", ""),
            "bootstrap_ci_low": "",
            "bootstrap_ci_high": "",
            "status": "requires_row_or_cluster_level_audit_table",
        }
        for row in run_rows
    ]


def _caveat_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        if row.get("raw_bwer") == "" and row.get("standardised_bwer") == "":
            caveat = "BWER summary is missing from registry and no source artifact was recomputed."
        else:
            caveat = "Summary-level robustness only; add per-slice or row-level artifacts for bootstrap, alpha, support, and missing-policy recomputation."
        rows.append(
            {
                "experiment_id": row.get("experiment_id", ""),
                "run_id": row.get("run_id", ""),
                "caveat_type": "artifact_granularity",
                "caveat": caveat,
            }
        )
    return rows


def _write_report(path: Path, comparison_rows: Sequence[Mapping[str, Any]], caveats: Sequence[Mapping[str, Any]]) -> None:
    available = sum(1 for row in comparison_rows if row.get("comparison_status") == "available_from_registry_summary")
    lines = [
        "# BWER Robustness v1 Report",
        "",
        f"Runs with registry-level BWER summaries: {available} / {len(comparison_rows)}.",
        "",
        "This report keeps segmentation, single-label classification, and multi-label classification separate via `metric_family` and `task_type` columns.",
        "",
        "## Caveats",
        "",
    ]
    for row in caveats:
        lines.append(f"- {row.get('experiment_id', '')} / {row.get('run_id', '')}: {row.get('caveat', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_bwer_robustness(registry_path: Path = DEFAULT_REGISTRY, output_dir: Path | None = None, missing_policy: str = "renormalize") -> dict[str, Path]:
    registry = _load_yaml(registry_path)
    out = ensure_dir(output_dir or "outputs/bwer_robustness_v1")
    defaults = registry.get("defaults", {}) or {}
    alphas = defaults.get("alphas", DEFAULT_ALPHAS)
    thresholds = defaults.get("support_thresholds", DEFAULT_SUPPORT_THRESHOLDS)
    run_rows = _experiment_runs(registry)
    comparison = _comparison_rows(run_rows)
    caveats = _caveat_rows(run_rows)
    artifacts = {
        "bwer_metric_comparison": out / "bwer_metric_comparison.csv",
        "bwer_bootstrap_ci_summary": out / "bwer_bootstrap_ci_summary.csv",
        "alpha_sensitivity_summary": out / "alpha_sensitivity_summary.csv",
        "support_threshold_sensitivity_summary": out / "support_threshold_sensitivity_summary.csv",
        "missing_policy_sensitivity_summary": out / "missing_policy_sensitivity_summary.csv",
        "traditional_subgroup_metrics": out / "traditional_subgroup_metrics.csv",
        "robustness_caveats": out / "robustness_caveats.csv",
        "bwer_robustness_report": out / "bwer_robustness_report.md",
    }
    write_csv(artifacts["bwer_metric_comparison"], comparison)
    write_csv(artifacts["bwer_bootstrap_ci_summary"], _bootstrap_rows(run_rows))
    write_csv(artifacts["alpha_sensitivity_summary"], _alpha_rows(run_rows, alphas))
    write_csv(artifacts["support_threshold_sensitivity_summary"], _support_rows(run_rows, thresholds))
    write_csv(artifacts["missing_policy_sensitivity_summary"], _missing_policy_rows(run_rows, DEFAULT_MISSING_POLICIES, missing_policy))
    write_csv(artifacts["traditional_subgroup_metrics"], _traditional_metric_rows(run_rows))
    write_csv(artifacts["robustness_caveats"], caveats)
    _write_report(artifacts["bwer_robustness_report"], comparison, caveats)
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--missing-policy", default="renormalize")
    args = parser.parse_args(argv)
    artifacts = build_bwer_robustness(args.registry, args.out, args.missing_policy)
    for path in artifacts.values():
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
