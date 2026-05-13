from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.audit_pipeline import _dataset_taxonomy, _interaction_columns
from rsfm_fairness_audit.audit_table import (
    build_audit_table_from_predictions,
    build_audit_table_from_segmentation_metrics,
    read_audit_table,
    write_audit_table,
)
from rsfm_fairness_audit.bwer import BWERConfig, compute_slice_scores
from rsfm_fairness_audit.io import ensure_dir, write_csv


def _parse_candidate(value: str) -> tuple[str, str | None]:
    cleaned = value.strip()
    if "|" in cleaned:
        left, right = cleaned.split("|", 1)
        balance = right.strip() or None
    else:
        left, balance = cleaned, None
    return left.strip(), balance


def _candidate_label(slice_variable: str, balance_variable: str | None) -> str:
    return f"BWER({slice_variable} | {balance_variable})" if balance_variable else f"BWER({slice_variable})"


def _unique_existing(values: Sequence[str | None]) -> list[str | None]:
    seen: set[str | None] = set()
    output: list[str | None] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _candidate_pairs(
    rows: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
    candidates: Sequence[str] | None,
) -> list[tuple[str, str | None]]:
    if candidates:
        return [_parse_candidate(candidate) for candidate in candidates]
    columns = set(rows[0]) if rows else set()
    slices = list(taxonomy.get("primary_slices", []) or []) + list(taxonomy.get("secondary_slices", []) or [])
    slices = [value for value in _unique_existing(slices) if value and value in columns]
    balances = [value for value in taxonomy.get("balance_variables", []) or [] if value in columns]
    pairs: list[tuple[str, str | None]] = []
    for slice_variable in slices:
        pairs.append((str(slice_variable), None))
        for balance_variable in balances:
            if balance_variable != slice_variable:
                pairs.append((str(slice_variable), str(balance_variable)))
    return pairs


def _support_ratio_for_valid_slices(support_rows: Sequence[Mapping[str, Any]], valid_slices: set[str]) -> tuple[int, int, float]:
    relevant = [row for row in support_rows if str(row.get("slice_value")) in valid_slices]
    if not relevant:
        return 0, 0, 0.0
    missing = sum(str(row.get("has_support")).lower() not in {"true", "1"} for row in relevant)
    return missing, len(relevant), float(missing / len(relevant))


def _recommend_raw(valid_slice_count: int, min_slices_required: int) -> tuple[bool, str]:
    if valid_slice_count < min_slices_required:
        return False, f"raw BWER has only {valid_slice_count} valid slices; requires at least {min_slices_required}"
    return True, "raw BWER has enough valid slices"


def _recommend_balanced(
    valid_slice_count: int,
    min_slices_required: int,
    n_balance_levels: int,
    missing_ratio: float,
) -> tuple[bool, str, str]:
    if valid_slice_count < min_slices_required:
        return False, "not_recommended", f"balanced BWER has only {valid_slice_count} valid slices; requires at least {min_slices_required}"
    if n_balance_levels < 2:
        return False, "not_recommended", "balanced BWER needs at least two observed balance levels"
    if missing_ratio > 0.70:
        return False, "not_recommended", f"slice x balance support is too sparse ({missing_ratio:.1%} missing)"
    if missing_ratio > 0.30:
        return True, "caution", f"slice x balance support is sparse ({missing_ratio:.1%} missing); run sensitivity checks"
    return True, "recommended", f"slice x balance support is adequate ({missing_ratio:.1%} missing)"


def _summarize_slice_rows(slice_variable: str, slice_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = [int(row.get("n_units", 0)) for row in slice_rows]
    valid = [row for row in slice_rows if str(row.get("is_valid_slice")).lower() in {"true", "1"}]
    return {
        "slice_variable": slice_variable,
        "n_slices_total": len(slice_rows),
        "n_slices_valid": len(valid),
        "min_units_per_slice": min(counts) if counts else 0,
        "median_units_per_slice": float(median(counts)) if counts else 0.0,
        "max_units_per_slice": max(counts) if counts else 0,
    }


def evaluate_slice_support(
    rows: list[dict[str, Any]],
    dataset: str,
    model: str,
    task: str,
    output_dir: str | Path,
    slice_config: str | Path = "configs/slice_taxonomy.yaml",
    candidates: Sequence[str] | None = None,
    min_samples_per_slice: int | None = None,
    min_units_required: int | None = None,
    min_slices_required: int | None = None,
    score_column: str | None = None,
    risk_column: str | None = None,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    taxonomy, warnings = _dataset_taxonomy(slice_config, dataset)
    warnings = list(warnings)
    working_rows = _interaction_columns([dict(row) for row in rows], taxonomy) if rows else []
    columns = set(working_rows[0]) if working_rows else set()
    pairs = _candidate_pairs(working_rows, taxonomy, candidates)
    if not pairs:
        warnings.append("No BWER candidate slice variables were available.")
    config = BWERConfig(
        dataset=dataset,
        model=model,
        task=task,
        split=str(working_rows[0].get("split", "all")) if working_rows else "all",
        min_samples_per_slice=int(min_samples_per_slice if min_samples_per_slice is not None else taxonomy.get("min_samples_per_slice", 1) or 1),
        min_positive_support=taxonomy.get("min_positive_support"),
        min_units_required=int(min_units_required if min_units_required is not None else taxonomy.get("min_units_required", 1) or 1),
        min_slices_required=int(min_slices_required if min_slices_required is not None else taxonomy.get("min_slices_required", 2) or 2),
    )
    recommendation_rows: list[dict[str, Any]] = []
    summary_by_slice: dict[str, dict[str, Any]] = {}
    for slice_variable, balance_variable in pairs:
        label = _candidate_label(slice_variable, balance_variable)
        if slice_variable not in columns:
            recommendation_rows.append(
                {
                    "candidate": label,
                    "slice_variable": slice_variable,
                    "balance_variable": balance_variable or "",
                    "recommendation": "not_recommended",
                    "reason": f"missing slice column: {slice_variable}",
                    "n_units": len(working_rows),
                    "n_slices_total": 0,
                    "n_slices_valid": 0,
                    "min_units_per_slice": 0,
                    "median_units_per_slice": 0,
                    "max_units_per_slice": 0,
                    "n_balance_levels": 0,
                    "missing_slice_balance_count": "",
                    "slice_balance_cell_count": "",
                    "missing_slice_balance_ratio": "",
                    "raw_bwer_appropriate": False,
                    "balanced_bwer_appropriate": False,
                    "formal_bwer_runnable": False,
                    "preferred_bwer": "none",
                }
            )
            continue
        if balance_variable and balance_variable not in columns:
            recommendation_rows.append(
                {
                    "candidate": label,
                    "slice_variable": slice_variable,
                    "balance_variable": balance_variable,
                    "recommendation": "not_recommended",
                    "reason": f"missing balance column: {balance_variable}",
                    "n_units": len(working_rows),
                    "n_slices_total": 0,
                    "n_slices_valid": 0,
                    "min_units_per_slice": 0,
                    "median_units_per_slice": 0,
                    "max_units_per_slice": 0,
                    "n_balance_levels": 0,
                    "missing_slice_balance_count": "",
                    "slice_balance_cell_count": "",
                    "missing_slice_balance_ratio": "",
                    "raw_bwer_appropriate": False,
                    "balanced_bwer_appropriate": False,
                    "formal_bwer_runnable": False,
                    "preferred_bwer": "none",
                }
            )
            continue
        slice_rows, support_rows, support_warnings, balance_stats = compute_slice_scores(
            working_rows,
            config,
            slice_variable,
            balance_variable,
            score_column=score_column,
            risk_column=risk_column,
        )
        warnings.extend(support_warnings)
        summary_by_slice.setdefault(slice_variable, _summarize_slice_rows(slice_variable, slice_rows))
        valid_slices = {str(row.get("slice_value")) for row in slice_rows if str(row.get("is_valid_slice")).lower() in {"true", "1"}}
        valid_slice_count = len(valid_slices)
        raw_ok, raw_reason = _recommend_raw(valid_slice_count, config.min_slices_required)
        missing_count = ""
        cell_count = ""
        missing_ratio_value = float("nan")
        balanced_ok = False
        balanced_status = "not_recommended"
        balanced_reason = "balanced BWER was not requested"
        n_balance_levels = int(balance_stats.get("n_total_balance_levels", 0) or 0)
        if balance_variable:
            missing, total, missing_ratio_value = _support_ratio_for_valid_slices(support_rows, valid_slices)
            missing_count = missing
            cell_count = total
            balanced_ok, balanced_status, balanced_reason = _recommend_balanced(
                valid_slice_count,
                config.min_slices_required,
                n_balance_levels,
                missing_ratio_value,
            )
        if balance_variable:
            recommendation = balanced_status
            reason = balanced_reason
            preferred = "balanced" if balanced_ok else "raw" if raw_ok else "none"
            if not balanced_ok and raw_ok:
                reason = f"{balanced_reason}; raw BWER may still be usable"
        else:
            recommendation = "recommended" if raw_ok else "not_recommended"
            reason = raw_reason
            preferred = "raw" if raw_ok else "none"
        formal_bwer_runnable = bool(raw_ok if not balance_variable else balanced_ok)
        if not formal_bwer_runnable and recommendation == "recommended":
            recommendation = "not_recommended"
        counts = [int(row.get("n_units", 0)) for row in slice_rows]
        recommendation_rows.append(
            {
                "candidate": label,
                "slice_variable": slice_variable,
                "balance_variable": balance_variable or "",
                "recommendation": recommendation,
                "reason": reason,
                "n_units": len(working_rows),
                "n_slices_total": len(slice_rows),
                "n_slices_valid": valid_slice_count,
                "min_units_per_slice": min(counts) if counts else 0,
                "median_units_per_slice": float(median(counts)) if counts else 0.0,
                "max_units_per_slice": max(counts) if counts else 0,
                "n_balance_levels": n_balance_levels,
                "missing_slice_balance_count": missing_count,
                "slice_balance_cell_count": cell_count,
                "missing_slice_balance_ratio": "" if math.isnan(missing_ratio_value) else missing_ratio_value,
                "raw_bwer_appropriate": raw_ok,
                "balanced_bwer_appropriate": balanced_ok,
                "formal_bwer_runnable": formal_bwer_runnable,
                "preferred_bwer": preferred,
            }
        )
    artifacts = {
        "audit_table": output / "audit_table.csv",
        "recommendations": output / "slice_support_recommendations.csv",
        "summary": output / "slice_support_summary.csv",
        "report": output / "slice_support_report.md",
        "warnings": output / "warnings.json",
    }
    write_audit_table(artifacts["audit_table"], working_rows)
    write_csv(artifacts["recommendations"], recommendation_rows)
    write_csv(artifacts["summary"], list(summary_by_slice.values()))
    _write_slice_support_report(artifacts["report"], dataset, model, task, recommendation_rows, warnings)
    artifacts["warnings"].write_text(json.dumps({"warnings": sorted(set(str(warning) for warning in warnings))}, indent=2), encoding="utf-8")
    return artifacts


def evaluate_slice_support_from_files(
    output_dir: str | Path,
    dataset: str,
    model: str,
    task: str,
    audit_table: str | Path | None = None,
    predictions: str | Path | None = None,
    metadata: str | Path | None = None,
    segmentation_metrics: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Path]:
    if audit_table:
        rows = read_audit_table(audit_table)
    elif segmentation_metrics:
        rows = build_audit_table_from_segmentation_metrics(segmentation_metrics, metadata, dataset=dataset, model=model, task=task)
    elif predictions:
        rows = build_audit_table_from_predictions(predictions, metadata, dataset=dataset, model=model, task=task)
    else:
        raise ValueError("BWER support preflight requires --audit-table, --predictions, or --segmentation-metrics.")
    return evaluate_slice_support(rows, dataset, model, task, output_dir, **kwargs)


def _write_slice_support_report(
    path: str | Path,
    dataset: str,
    model: str,
    task: str,
    recommendation_rows: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BWER Candidate Slice-Support Preflight",
        "",
        f"- Dataset: `{dataset}`",
        f"- Model: `{model}`",
        f"- Task: `{task}`",
        "",
        "This report checks whether candidate raw or balanced BWER configurations have enough slice support before paper-grade BWER runs. It is a design diagnostic, not a model-performance result.",
        "",
        "## Recommendations",
        "",
        "| candidate | recommendation | runnable | preferred | valid slices | missing slice x balance | reason |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in recommendation_rows:
        ratio = row.get("missing_slice_balance_ratio", "")
        ratio_text = "" if ratio == "" else f"{float(ratio):.1%}"
        lines.append(
            f"| {row.get('candidate', '')} | {row.get('recommendation', '')} | {row.get('formal_bwer_runnable', '')} | {row.get('preferred_bwer', '')} | {row.get('n_slices_valid', '')} | {ratio_text} | {row.get('reason', '')} |"
        )
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in sorted(set(str(warning) for warning in warnings)):
            lines.append(f"- {warning}")
    else:
        lines.append("- No support preflight warnings were emitted.")
    lines.extend(
        [
            "",
            "## Files Produced",
            "",
            "- `audit_table.csv`",
            "- `slice_support_recommendations.csv`",
            "- `slice_support_summary.csv`",
            "- `slice_support_report.md`",
            "- `warnings.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
