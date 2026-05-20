from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.audit_table import (
    build_audit_table_from_predictions,
    build_audit_table_from_segmentation_metrics,
    read_audit_table,
    write_audit_table,
)
from rsfm_fairness_audit.bwer import BWERConfig, compute_bwer_family, create_interaction_slice
from rsfm_fairness_audit.bwer_plots import write_bwer_figures
from rsfm_fairness_audit.bwer_report import write_bwer_report, write_warnings_json
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.io import ensure_dir, write_csv


TAXONOMY_ALIASES = {
    "ben_ge": "ben_ge_800",
    "bigearthnet": "bigearthnet_lccol",
}


def _dataset_taxonomy(slice_config: str | Path, dataset: str, task: str | None = None) -> tuple[dict[str, Any], list[str]]:
    config = load_yaml(slice_config)
    datasets = config.get("datasets", {})
    task_key = f"{dataset}_{task}" if task else ""
    if task_key in datasets:
        return dict(datasets[task_key]), [f"Using task-aware slice taxonomy: dataset={dataset}, task={task} -> {task_key}."]
    if dataset in datasets:
        return dict(datasets[dataset]), []
    canonical = TAXONOMY_ALIASES.get(dataset)
    if canonical and canonical in datasets:
        return dict(datasets[canonical]), [f"Using slice taxonomy alias: dataset={dataset} -> {canonical}."]
    known = ", ".join(sorted(datasets)) or "none"
    return {}, [f"No taxonomy entry found for dataset={dataset}. Known dataset taxonomy names: {known}."]


def _available(columns: set[str], values: Sequence[str]) -> list[str]:
    return [value for value in values if value in columns]


def _interaction_columns(rows: list[dict[str, Any]], taxonomy: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = rows
    for interaction in taxonomy.get("secondary_slices", []) or []:
        parts = str(interaction).split("__")
        if len(parts) >= 2 and all(part in output[0] for part in parts):
            output = create_interaction_slice(output, parts, str(interaction))
    return output


def _balance_variants(columns: set[str], taxonomy: Mapping[str, Any], override: str | None) -> list[str | None]:
    if override:
        if override.lower() in {"raw", "none", "null"}:
            return [None]
        return [override]
    variants: list[str | None] = [None]
    for value in taxonomy.get("balance_variables", []) or []:
        if value in columns:
            variants.append(value)
    return variants


def _slice_variables(columns: set[str], taxonomy: Mapping[str, Any], override: str | None) -> list[str]:
    if override:
        return [override]
    configured = list(taxonomy.get("primary_slices", []) or []) + list(taxonomy.get("secondary_slices", []) or [])
    return _available(columns, configured)


def evaluate_bwer_table(
    audit_rows: list[dict[str, Any]],
    dataset: str,
    model: str,
    task: str,
    output_dir: str | Path,
    slice_config: str | Path = "configs/slice_taxonomy.yaml",
    slice_variable: str | None = None,
    balance_variable: str | None = None,
    tail_fraction: float | None = None,
    bootstrap: int = 0,
    cluster_key: str | None = None,
    seed: int = 42,
    weighting: str = "uniform",
    missing_balance_policy: str = "renormalize",
    score_column: str | None = None,
    risk_column: str | None = None,
    audit_level: str = "pilot",
    selective_coverage: float | None = None,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    taxonomy, warnings = _dataset_taxonomy(slice_config, dataset, task)
    warnings = list(warnings)
    if not taxonomy:
        warnings.append("No taxonomy entry was available; provide --slice-variable and optional --balance-variable explicitly.")
    rows = _interaction_columns([dict(row) for row in audit_rows], taxonomy) if audit_rows else []
    columns = set(rows[0].keys()) if rows else set()
    slices = _slice_variables(columns, taxonomy, slice_variable)
    if not slices and slice_variable:
        slices = [slice_variable]
    if not slices:
        raise ValueError("No available slice variables. Provide --slice-variable or update configs/slice_taxonomy.yaml.")
    balances = _balance_variants(columns, taxonomy, balance_variable)
    config = BWERConfig(
        dataset=dataset,
        model=model,
        task=task,
        split=str(rows[0].get("split", "all")) if rows else "all",
        tail_fraction=float(tail_fraction if tail_fraction is not None else taxonomy.get("default_tail_fraction", 0.10)),
        weighting=weighting,
        missing_balance_policy=missing_balance_policy,
        min_samples_per_slice=int(taxonomy.get("min_samples_per_slice", 1) or 1),
        min_positive_support=taxonomy.get("min_positive_support"),
        min_valid_pixel_support=taxonomy.get("min_valid_pixel_support"),
        min_slices_required=int(taxonomy.get("min_slices_required", 2) or 2),
        min_units_required=int(taxonomy.get("min_units_required", 1) or 1),
        bootstrap_n=bootstrap,
        bootstrap_method="cluster" if cluster_key else "ordinary" if bootstrap else "none",
        cluster_key=cluster_key or taxonomy.get("bootstrap_cluster_key"),
        selective_coverage=selective_coverage,
        seed=seed,
    )
    if any("confidence" in row for row in rows):
        warnings.append("Confidence field detected; formal BWER records selective_risk hooks. Use the post-hoc run-selective-risk command for confidence-conditioned retention diagnostics.")
    effective_cluster = cluster_key or taxonomy.get("bootstrap_cluster_key")
    if effective_cluster and effective_cluster not in columns:
        warnings.append(f"Skipping cluster bootstrap key {effective_cluster}: column missing.")
        effective_cluster = None
    summary_rows, slice_rows, support_rows, ci_rows, bwer_warnings = compute_bwer_family(
        rows,
        config,
        slices,
        balances,
        n_bootstrap=bootstrap,
        cluster_key=effective_cluster,
        score_column=score_column,
        risk_column=risk_column,
    )
    warnings.extend(bwer_warnings)
    artifacts = {
        "audit_table": output / "audit_table.csv",
        "bwer_summary": output / "bwer_summary.csv",
        "bwer_by_slice": output / "bwer_by_slice.csv",
        "support_diagnostics": output / "support_diagnostics.csv",
        "bootstrap_ci": output / "bootstrap_ci.csv",
        "warnings": output / "warnings.json",
        "report": output / "report.md",
    }
    if not summary_rows:
        fatal = "no valid BWER variants produced"
        warnings.append(fatal)
        write_audit_table(artifacts["audit_table"], rows)
        write_csv(artifacts["bwer_summary"], summary_rows)
        write_csv(artifacts["bwer_by_slice"], slice_rows)
        write_csv(artifacts["support_diagnostics"], support_rows)
        write_csv(artifacts["bootstrap_ci"], ci_rows)
        write_warnings_json(artifacts["warnings"], warnings)
        write_bwer_report(artifacts["report"], summary_rows, slice_rows, warnings, audit_level=audit_level)
        raise ValueError(fatal)
    write_audit_table(artifacts["audit_table"], rows)
    write_csv(artifacts["bwer_summary"], summary_rows)
    write_csv(artifacts["bwer_by_slice"], slice_rows)
    write_csv(artifacts["support_diagnostics"], support_rows)
    write_csv(artifacts["bootstrap_ci"], ci_rows)
    write_warnings_json(artifacts["warnings"], warnings)
    figure_paths = write_bwer_figures(summary_rows, slice_rows, output)
    write_bwer_report(artifacts["report"], summary_rows, slice_rows, warnings, audit_level=audit_level)
    artifacts.update(figure_paths)
    return artifacts


def evaluate_bwer_from_file(
    audit_table: str | Path,
    dataset: str,
    model: str,
    task: str,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Path]:
    rows = read_audit_table(audit_table)
    return evaluate_bwer_table(rows, dataset, model, task, output_dir, **kwargs)


def run_audit_from_outputs(
    output_dir: str | Path,
    dataset: str,
    model: str,
    task: str,
    predictions: str | Path | None = None,
    metadata: str | Path | None = None,
    segmentation_metrics: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Path]:
    if segmentation_metrics:
        rows = build_audit_table_from_segmentation_metrics(segmentation_metrics, metadata, dataset=dataset, model=model, task=task)
    elif predictions:
        rows = build_audit_table_from_predictions(predictions, metadata, dataset=dataset, model=model, task=task)
    else:
        raise ValueError("run-audit requires --predictions or --segmentation-metrics.")
    return evaluate_bwer_table(rows, dataset, model, task, output_dir, **kwargs)
