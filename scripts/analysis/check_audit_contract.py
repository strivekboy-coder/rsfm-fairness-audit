from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv  # noqa: E402


DEFAULT_REGISTRY = Path("configs/analysis/unified_audit_registry.yaml")


TASK_CONTRACTS: dict[str, dict[str, list[set[str]]]] = {
    "single_label_classification": {
        "core": [
            {"sample_id", "unit_id"},
            {"split", "eval_scope"},
            {"dataset"},
            {"task_type", "task"},
            {"model", "model_family", "run_id"},
            {"y_true", "label", "class_label"},
            {"y_pred", "prediction", "predicted_label"},
        ],
        "risk": [{"risk", "score", "correct", "loss"}],
        "score": [{"confidence", "probability", "prob_true", "max_probability", "logit", "logits"}],
        "support": [{"sample_count", "n_samples", "support", "unit_id", "sample_id"}],
        "conformal": [{"calibration", "is_calibration", "calibration_split", "split"}],
    },
    "multi_label_classification": {
        "core": [
            {"sample_id", "unit_id"},
            {"split", "eval_scope"},
            {"dataset"},
            {"task_type", "task"},
            {"model", "model_family", "run_id"},
            {"label", "class_label"},
            {"y_true", "target"},
            {"y_pred", "prediction", "binary_prediction"},
        ],
        "risk": [{"risk_bce", "bce", "loss", "risk", "binary_error"}],
        "score": [{"confidence", "probability", "label_probability", "prob", "logit", "logits"}],
        "support": [{"sample_count", "n_samples", "support", "label_support", "sample_id"}],
        "conformal": [{"calibration", "is_calibration", "calibration_split", "split"}],
    },
    "segmentation": {
        "core": [
            {"sample_id", "chip_id", "unit_id", "event_id"},
            {"split", "eval_scope"},
            {"dataset"},
            {"task_type", "task"},
            {"model", "model_family", "run_id"},
            {"event_id", "slice_value", "unit_id"},
        ],
        "risk": [{"risk", "micro_iou", "iou", "score", "TP", "tp"}],
        "score": [{"confidence", "probability", "probability_path", "logit", "logits", "score"}],
        "support": [
            {"valid_pixel_count", "valid_pixels", "TP", "tp"},
            {"positive_pixel_count", "positive_pixels", "FN", "fn"},
        ],
        "conformal": [{"calibration", "is_calibration", "calibration_split", "split"}],
    },
    "tabular_embedding_classification": {
        "core": [
            {"sample_id", "unit_id"},
            {"split", "eval_scope"},
            {"dataset"},
            {"task_type", "task"},
            {"model", "model_family", "run_id"},
            {"y_true", "label", "class_label"},
            {"y_pred", "prediction", "predicted_label"},
            {"latitude", "lat", "longitude", "lon", "country", "region"},
            {"timestamp", "year", "date"},
        ],
        "risk": [{"risk", "score", "correct", "loss"}],
        "score": [{"confidence", "probability", "prob_true", "max_probability", "logit", "logits"}],
        "support": [{"sample_count", "n_samples", "support", "unit_id", "sample_id"}],
        "conformal": [{"calibration", "is_calibration", "calibration_split", "split"}],
    },
}


TASK_ALIASES = {
    "classification": "single_label_classification",
    "single_label": "single_label_classification",
    "single-label classification": "single_label_classification",
    "multilabel": "multi_label_classification",
    "multi-label classification": "multi_label_classification",
    "multi_label": "multi_label_classification",
    "segmentation": "segmentation",
    "tabular_embedding": "tabular_embedding_classification",
}


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


def _canonical_task(task_type: Any) -> str:
    text = str(task_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    return TASK_ALIASES.get(text, text)


def _first_existing(paths: Sequence[str | Path] | None) -> Path | None:
    for value in paths or []:
        path = Path(value)
        if path.exists():
            return path
    return None


def _candidate_search_roots(path: Path) -> list[Path]:
    roots: list[Path] = []
    if path.exists() and path.is_dir():
        roots.append(path)
    if path.parent.exists() and path.parent.name.lower() not in {"outputs", "content"}:
        roots.append(path.parent)
    else:
        for ancestor in path.parents:
            if ancestor.exists() and ancestor != ancestor.parent:
                if ancestor.name.lower() in {"outputs", "content"}:
                    break
                roots.append(ancestor)
                break
    output: list[Path] = []
    for root in roots:
        if root.exists() and root.is_dir() and root not in output:
            output.append(root)
    return output


def _path_tokens(path: Path) -> list[str]:
    tokens: list[str] = []
    for part in path.parts:
        if part.lower() in {"", ".", "outputs", "content"}:
            continue
        if part.endswith(".csv"):
            continue
        if part not in tokens:
            tokens.append(part)
    return tokens


def _discover_csv(paths: Sequence[str | Path] | None, filenames: Sequence[str]) -> Path | None:
    existing = _first_existing(paths)
    if existing is not None and existing.is_file() and existing.suffix.lower() == ".csv":
        return existing
    for value in paths or []:
        path = Path(value)
        tokens = _path_tokens(path)
        for root in _candidate_search_roots(path):
            for filename in filenames:
                try:
                    matches = sorted(root.rglob(filename), key=lambda item: (len(item.parts), str(item)))
                except OSError:
                    continue
                for match in matches:
                    if any(part.startswith("test_") or part.startswith("pytest") for part in match.parts):
                        continue
                    match_text = str(match)
                    if tokens and not any(token in match_text for token in tokens):
                        continue
                    return match
    return None


def _columns(path: Path | None) -> set[str]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        rows = read_csv_rows(path)
    except Exception:
        return set()
    if not rows:
        return set()
    return set().union(*(row.keys() for row in rows))


def _has_any(columns: set[str], alternatives: set[str]) -> bool:
    lower = {col.lower() for col in columns}
    return any(value.lower() in lower for value in alternatives)


def _missing_groups(columns: set[str], groups: Sequence[set[str]]) -> list[str]:
    missing: list[str] = []
    for group in groups:
        if not _has_any(columns, group):
            missing.append("|".join(sorted(group)))
    return missing


def _collect_candidate_paths(exp: Mapping[str, Any], run: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("prediction_table_candidates", "source_summary_candidates"):
        value = run.get(key)
        if isinstance(value, list):
            paths.extend(str(item) for item in value)
    selective = exp.get("selective_risk", {}) or {}
    pred_map = selective.get("prediction_table_candidates") or {}
    run_id = str(run.get("run_id", ""))
    if isinstance(pred_map, Mapping):
        for key in (run_id, str(run.get("model_family", "")), str(run.get("sensor_mode", ""))):
            value = pred_map.get(key)
            if isinstance(value, list):
                paths.extend(str(item) for item in value)
    run_summary = selective.get("source_run_summary_candidates") or {}
    if isinstance(run_summary, Mapping) and run_id in run_summary:
        spec = run_summary[run_id]
        if isinstance(spec, Mapping):
            paths.extend(str(item) for item in spec.get("paths", []) or [])
    for key in ("output_dir_candidates", "comparison_dir_candidates"):
        value = exp.get(key)
        if isinstance(value, list):
            paths.extend(str(item) for item in value)
    return paths


def _artifact_for_run(exp: Mapping[str, Any], run: Mapping[str, Any]) -> Path | None:
    filenames = [
        "audit_table.csv",
        "predictions.csv",
        "segmentation_metrics.csv",
        "event_segmentation_metrics.csv",
        "selective_risk_summary.csv",
        "selective_risk_comparison.csv",
        "closure_comparison_summary.csv",
        "aggregate_sensor_mode_comparison.csv",
    ]
    return _discover_csv(_collect_candidate_paths(exp, run), filenames)


def _run_support_from_registry(run: Mapping[str, Any], exp: Mapping[str, Any], task_type: str) -> dict[str, bool]:
    has_risk = any(run.get(key) not in {None, ""} for key in ("raw_bwer", "standardised_bwer", "mean_bce_risk", "aggregate_score"))
    selective = exp.get("selective_risk", {}) or {}
    availability = str(selective.get("availability", "")).lower()
    has_score = availability == "available" or "prediction_tables_present" in availability
    has_conf = "calibration" in selective or "conformal" in selective
    return {
        "core": True,
        "risk": has_risk,
        "score": has_score,
        "support": True,
        "conformal": has_conf,
    }


def _coverage_for_run(exp: Mapping[str, Any], run: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    task_type = _canonical_task(exp.get("task_type", ""))
    contract = TASK_CONTRACTS.get(task_type, TASK_CONTRACTS["single_label_classification"])
    artifact = _artifact_for_run(exp, run)
    cols = _columns(artifact)
    registry_support = _run_support_from_registry(run, exp, task_type)
    categories: dict[str, bool] = {}
    missing_rows: list[dict[str, Any]] = []
    for category, groups in contract.items():
        missing = _missing_groups(cols, groups) if cols else list("|".join(sorted(group)) for group in groups)
        if cols:
            ok = not missing
        else:
            ok = registry_support.get(category, False)
        categories[category] = ok
        if not ok:
            for group in missing:
                missing_rows.append(
                    {
                        "experiment_id": exp.get("experiment_id", ""),
                        "run_id": run.get("run_id", ""),
                        "task_type": task_type,
                        "category": category,
                        "missing_alternatives": group,
                        "severity": "blocking" if category in {"core", "risk"} else "analysis_limited",
                        "artifact_path": str(artifact or ""),
                    }
                )
    supports_raw = categories.get("core", False) and categories.get("risk", False) and categories.get("support", False)
    supports_standardised = supports_raw and bool(exp.get("standardised_balance"))
    supports_selective = supports_raw and categories.get("score", False)
    supports_conformal = supports_selective and categories.get("conformal", False)
    row = {
        "experiment_id": exp.get("experiment_id", ""),
        "dataset": exp.get("dataset", ""),
        "task_type": task_type,
        "run_id": run.get("run_id", ""),
        "model_family": run.get("model_family", ""),
        "artifact_status": "found" if artifact else "documented_record_only",
        "artifact_path": str(artifact or ""),
        "core_fields_ok": categories.get("core", False),
        "risk_fields_ok": categories.get("risk", False),
        "score_fields_ok": categories.get("score", False),
        "support_fields_ok": categories.get("support", False),
        "calibration_fields_ok": categories.get("conformal", False),
        "supports_raw_bwer": supports_raw,
        "supports_standardised_bwer": supports_standardised,
        "supports_selective_risk": supports_selective,
        "supports_conformal_selective_risk": supports_conformal,
    }
    return row, missing_rows, {"artifact": artifact, "categories": categories}


def _rerun_row(coverage: Mapping[str, Any], missing_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing_categories = sorted({str(row.get("category", "")) for row in missing_rows if row.get("run_id") == coverage.get("run_id") and row.get("experiment_id") == coverage.get("experiment_id")})
    if coverage.get("supports_conformal_selective_risk") == "True" or coverage.get("supports_conformal_selective_risk") is True:
        requirement = "posthoc_only"
        reason = "All required fields for current contract checks are available."
    elif coverage.get("supports_selective_risk") == "True" or coverage.get("supports_selective_risk") is True:
        requirement = "requires_calibration_split_for_conformal"
        reason = "Selective-risk scores are available, but calibration split/indicator is missing for conformal selective audit."
    elif coverage.get("supports_raw_bwer") == "True" or coverage.get("supports_raw_bwer") is True:
        requirement = "requires_probability_or_logit_export"
        reason = "Raw/standardised BWER is supported, but probability/logit/confidence fields are missing for selective/conformal audit."
    elif coverage.get("artifact_status") == "documented_record_only":
        requirement = "posthoc_summary_only_or_inference_rerun_if_scores_needed"
        reason = "Registry contains documented summary metrics, but no prediction/audit artifact was found locally."
    else:
        requirement = "missing_artifact_or_core_fields"
        reason = "Core or risk fields are missing from the resolved artifact."
    return {
        "experiment_id": coverage.get("experiment_id", ""),
        "run_id": coverage.get("run_id", ""),
        "requirement": requirement,
        "reason": reason,
        "missing_categories": ";".join(missing_categories),
    }


def _write_report(path: Path, coverage_rows: Sequence[Mapping[str, Any]], missing_rows: Sequence[Mapping[str, Any]], rerun_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Audit Contract Coverage Report",
        "",
        "This report is generated from the unified audit registry and locally available CSV artifacts.",
        "",
        "## Summary",
        "",
    ]
    total = len(coverage_rows)
    raw = sum(1 for row in coverage_rows if row.get("supports_raw_bwer") is True)
    sel = sum(1 for row in coverage_rows if row.get("supports_selective_risk") is True)
    conf = sum(1 for row in coverage_rows if row.get("supports_conformal_selective_risk") is True)
    lines.extend(
        [
            f"- Runs checked: {total}",
            f"- Supports Raw/Standardised BWER from current evidence: {raw}",
            f"- Supports selective risk: {sel}",
            f"- Supports conformal selective risk: {conf}",
            "",
            "## Rerun Requirements",
            "",
            "| Experiment | Run | Requirement | Reason |",
            "|---|---|---|---|",
        ]
    )
    for row in rerun_rows:
        lines.append(f"| {row.get('experiment_id', '')} | {row.get('run_id', '')} | {row.get('requirement', '')} | {row.get('reason', '')} |")
    lines.extend(["", "## Missing Fields", "", "| Experiment | Run | Category | Missing alternatives | Severity |", "|---|---|---|---|---|"])
    for row in missing_rows:
        lines.append(f"| {row.get('experiment_id', '')} | {row.get('run_id', '')} | {row.get('category', '')} | {row.get('missing_alternatives', '')} | {row.get('severity', '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit_contract_coverage(registry_path: Path = DEFAULT_REGISTRY, output_dir: Path | None = None) -> dict[str, Path]:
    registry = _load_yaml(registry_path)
    defaults = registry.get("defaults", {}) or {}
    out = ensure_dir(output_dir or defaults.get("output_audit_contract", "outputs/audit_contract_v1"))
    coverage_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for exp in registry.get("experiments", []):
        runs = exp.get("formal_runs") or [{"run_id": "experiment"}]
        for run in runs:
            coverage, missing, _ = _coverage_for_run(exp, run)
            coverage_rows.append(coverage)
            missing_rows.extend(missing)
    rerun_rows = [_rerun_row(row, missing_rows) for row in coverage_rows]

    artifacts = {
        "audit_contract_coverage": out / "audit_contract_coverage.csv",
        "missing_fields_by_experiment": out / "missing_fields_by_experiment.csv",
        "rerun_requirements": out / "rerun_requirements.csv",
        "audit_contract_report": out / "audit_contract_report.md",
    }
    write_csv(artifacts["audit_contract_coverage"], coverage_rows)
    write_csv(artifacts["missing_fields_by_experiment"], missing_rows)
    write_csv(artifacts["rerun_requirements"], rerun_rows)
    _write_report(artifacts["audit_contract_report"], coverage_rows, missing_rows, rerun_rows)
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    artifacts = build_audit_contract_coverage(args.registry, args.out)
    for path in artifacts.values():
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
