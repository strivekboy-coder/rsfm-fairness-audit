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


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency issue
        raise RuntimeError("PyYAML is required to load the unified audit registry.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "experiments" not in data:
        raise ValueError(f"Invalid unified audit registry: {path}")
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


def _first_existing(paths: Sequence[str | Path] | None) -> Path | None:
    for value in paths or []:
        path = Path(value)
        if path.exists():
            return path
    return None


def _row_value(row: Mapping[str, Any] | None, names: Sequence[str], default: Any = "") -> Any:
    if not row:
        return default
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return value
    return default


def _sensor_mode_alias(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    aliases = {
        "s1": "S1",
        "croma_s1": "S1",
        "sar": "S1",
        "s2": "S2",
        "croma_s2": "S2",
        "optical": "S2",
        "s1+s2": "S1+S2",
        "s1_plus_s2": "S1+S2",
        "croma_s1_plus_s2": "S1+S2",
        "both": "S1+S2",
        "fusion": "S1+S2",
    }
    return aliases.get(text, str(value or "").strip())


def _metric_score_to_risk(score: Any, metric_family: str) -> float:
    value = _float(score)
    if math.isnan(value):
        return value
    if metric_family in {"iou_risk", "classification_error"}:
        return 1.0 - value
    if metric_family == "bce_risk":
        return value
    return value


def _registry_experiment_rows(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in registry.get("experiments", []):
        rows.append(
            {
                "experiment_id": exp.get("experiment_id", ""),
                "dataset": exp.get("dataset", ""),
                "deployment_axis": exp.get("deployment_axis", ""),
                "task_type": exp.get("task_type", ""),
                "formal_status": exp.get("formal_status", ""),
                "result_level": exp.get("result_level", ""),
                "protocol_summary": exp.get("protocol_summary", ""),
                "primary_metric_family": exp.get("primary_metric_family", ""),
                "aggregate_metric_name": exp.get("aggregate_metric_name", ""),
                "risk_metric_name": exp.get("risk_metric_name", ""),
                "primary_bwer_slice": exp.get("primary_bwer_slice", ""),
                "standardised_balance": exp.get("standardised_balance", ""),
                "resolved_output_dir": str(_first_existing(exp.get("output_dir_candidates")) or ""),
            }
        )
    return rows


def _registry_run_rows(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in registry.get("experiments", []):
        metric_family = str(exp.get("primary_metric_family", ""))
        for run in exp.get("formal_runs", []):
            aggregate_score = _float(run.get("aggregate_score"))
            aggregate_risk = _metric_score_to_risk(aggregate_score, metric_family)
            if metric_family == "bce_risk" and not math.isnan(_float(run.get("mean_bce_risk"))):
                aggregate_risk = _float(run.get("mean_bce_risk"))
            row = {
                "experiment_id": exp.get("experiment_id", ""),
                "dataset": exp.get("dataset", ""),
                "deployment_axis": exp.get("deployment_axis", ""),
                "task_type": exp.get("task_type", ""),
                "result_level": run.get("result_level", exp.get("result_level", "formal_result")),
                "formal_status": exp.get("formal_status", ""),
                "run_id": run.get("run_id", ""),
                "model_family": run.get("model_family", ""),
                "model_variant": run.get("model_variant", ""),
                "sensor_mode": run.get("sensor_mode", ""),
                "input_mode": run.get("input_mode", ""),
                "split_protocol": run.get("split_protocol", ""),
                "eval_scope": run.get("eval_scope", ""),
                "metric_family": metric_family,
                "aggregate_metric_name": exp.get("aggregate_metric_name", ""),
                "aggregate_score": aggregate_score if not math.isnan(aggregate_score) else "",
                "aggregate_risk": aggregate_risk,
                "risk_metric_name": exp.get("risk_metric_name", ""),
                "raw_bwer_slice": run.get("raw_bwer_slice", exp.get("primary_bwer_slice", "")),
                "raw_bwer": run.get("raw_bwer", ""),
                "standardised_balance": exp.get("standardised_balance", ""),
                "standardised_bwer": run.get("standardised_bwer", ""),
                "worst_slice": run.get("worst_slice", ""),
                "best_slice": run.get("best_slice", ""),
                "tail_slices": run.get("tail_slices", ""),
                "data_source": run.get("data_source", "registry_documented_record"),
            }
            for extra in ("macro_f1", "micro_f1", "micro_ap", "balanced_accuracy", "top5_accuracy", "mean_bce_risk", "aggregate_dice"):
                if extra in run:
                    row[extra] = run.get(extra, "")
            if exp.get("experiment_id") == "reben_croma_sensor_mode" and (
                row.get("micro_ap", "") in {"", None} or row.get("micro_f1", "") in {"", None}
            ):
                row["metric_availability_note"] = "micro_ap and/or micro_f1 unavailable in registry record; loaded from aggregate_sensor_mode_comparison.csv when present."
            rows.append(row)
    return _enrich_rows_from_available_outputs(registry, rows)


def _enrich_rows_from_available_outputs(registry: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use completed output CSVs when present, while keeping registry records as fallback."""
    reben_exp = None
    for exp in registry.get("experiments", []):
        if exp.get("experiment_id") == "reben_croma_sensor_mode":
            reben_exp = exp
            break
    if not reben_exp:
        return rows
    output_dir = _first_existing(reben_exp.get("output_dir_candidates"))
    if output_dir is None:
        return rows
    aggregate_path = output_dir / "aggregate_sensor_mode_comparison.csv"
    bce_path = output_dir / "bce_bwer_sensor_mode_comparison.csv"
    binary_path = output_dir / "binary_error_bwer_sensor_mode_comparison.csv"
    if not aggregate_path.exists():
        return rows
    aggregate_rows = read_csv_rows(aggregate_path)
    bwer_rows = []
    if bce_path.exists():
        bwer_rows.extend(read_csv_rows(bce_path))
    if binary_path.exists():
        bwer_rows.extend(read_csv_rows(binary_path))
    by_mode = {_sensor_mode_alias(row.get("sensor_mode", row.get("run_name", ""))): row for row in aggregate_rows}
    bwer_by_mode = {
        _sensor_mode_alias(row.get("sensor_mode", row.get("run_name", ""))): row
        for row in bwer_rows
        if str(_row_value(row, ["risk_name", "risk_metric", "risk_column"], "")).strip() in {"risk_bce", "bce", "labelwise_bce"}
        and str(_row_value(row, ["slice_variable", "slice", "slice_name"], "")).strip() == "country"
        and str(_row_value(row, ["balance_variable", "balance", "standardised_balance"], "")).strip() == ""
    }
    std_by_mode = {
        _sensor_mode_alias(row.get("sensor_mode", row.get("run_name", ""))): row
        for row in bwer_rows
        if str(_row_value(row, ["risk_name", "risk_metric", "risk_column"], "")).strip() in {"risk_bce", "bce", "labelwise_bce"}
        and str(_row_value(row, ["slice_variable", "slice", "slice_name"], "")).strip() == "country"
        and str(_row_value(row, ["balance_variable", "balance", "standardised_balance"], "")).strip() == "class_label"
    }
    for row in rows:
        if row.get("experiment_id") != "reben_croma_sensor_mode":
            continue
        mode = _sensor_mode_alias(row.get("sensor_mode", row.get("run_id", "")))
        aggregate = by_mode.get(mode)
        if aggregate:
            row["aggregate_score"] = _row_value(aggregate, ["macro_ap", "aggregate_score"], row.get("aggregate_score", ""))
            row["micro_ap"] = _row_value(aggregate, ["micro_ap"], row.get("micro_ap", ""))
            row["macro_f1"] = _row_value(aggregate, ["macro_f1"], row.get("macro_f1", ""))
            row["micro_f1"] = _row_value(aggregate, ["micro_f1"], row.get("micro_f1", ""))
            row["mean_bce_risk"] = _row_value(aggregate, ["mean_bce_risk", "aggregate_risk", "mean_risk"], row.get("mean_bce_risk", ""))
            if row.get("micro_ap", "") in {"", None} or row.get("micro_f1", "") in {"", None}:
                row["metric_availability_note"] = "micro_ap and/or micro_f1 not present in aggregate_sensor_mode_comparison.csv."
            else:
                row["metric_availability_note"] = ""
            row["data_source"] = f"file_read:{aggregate_path}"
        bwer = bwer_by_mode.get(mode)
        if bwer:
            row["raw_bwer"] = _row_value(bwer, ["bwer", "raw_bwer", "cross_run_mode_bwer"], row.get("raw_bwer", ""))
            row["worst_slice"] = _row_value(bwer, ["worst_slice"], row.get("worst_slice", ""))
            row["tail_slices"] = _row_value(bwer, ["tail_slices"], row.get("tail_slices", ""))
            row["raw_bwer_slice"] = "country"
        std = std_by_mode.get(mode)
        if std:
            row["standardised_bwer"] = _row_value(std, ["bwer", "standardised_bwer", "cross_run_mode_bwer"], row.get("standardised_bwer", ""))
        if row.get("metric_family") == "bce_risk":
            row["aggregate_risk"] = row.get("mean_bce_risk", row.get("aggregate_risk", ""))
    return rows


def _registry_slice_rows(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in registry.get("experiments", []):
        for slice_name in [exp.get("primary_bwer_slice", ""), exp.get("standardised_balance", "")]:
            if slice_name:
                rows.append(
                    {
                        "experiment_id": exp.get("experiment_id", ""),
                        "dataset": exp.get("dataset", ""),
                        "deployment_axis": exp.get("deployment_axis", ""),
                        "slice_or_balance_variable": slice_name,
                        "role": "primary_slice" if slice_name == exp.get("primary_bwer_slice") else "standardisation_balance",
                        "task_type": exp.get("task_type", ""),
                    }
                )
    return rows


def _list_rows(registry: Mapping[str, Any], key: str, output_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in registry.get("experiments", []):
        for index, value in enumerate(exp.get(key, []), start=1):
            rows.append(
                {
                    "experiment_id": exp.get("experiment_id", ""),
                    "dataset": exp.get("dataset", ""),
                    "deployment_axis": exp.get("deployment_axis", ""),
                    "index": index,
                    output_column: value,
                }
            )
    return rows


def _claim_rows(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in registry.get("experiments", []):
        for item in exp.get("claim_support", []):
            rows.append(
                {
                    "experiment_id": exp.get("experiment_id", ""),
                    "dataset": exp.get("dataset", ""),
                    "deployment_axis": exp.get("deployment_axis", ""),
                    "claim": item.get("claim", ""),
                    "support": item.get("support", ""),
                    "caveat": item.get("caveat", ""),
                }
            )
    return rows


def _average_vs_bwer(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_exp: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("result_level", "")) == "formal_result":
            by_exp.setdefault(str(row.get("experiment_id", "")), []).append(row)
    for exp_id, items in by_exp.items():
        valid_agg = [(row, _float(row.get("aggregate_score"))) for row in items if not math.isnan(_float(row.get("aggregate_score")))]
        valid_bwer = [(row, _float(row.get("raw_bwer"))) for row in items if not math.isnan(_float(row.get("raw_bwer")))]
        aggregate_best = max(valid_agg, key=lambda item: item[1])[0] if valid_agg else {}
        bwer_best = min(valid_bwer, key=lambda item: item[1])[0] if valid_bwer else {}
        for row in items:
            output.append(
                {
                    **dict(row),
                    "aggregate_best_run_id": aggregate_best.get("run_id", ""),
                    "raw_bwer_best_run_id": bwer_best.get("run_id", ""),
                    "aggregate_best_equals_bwer_best": str(bool(aggregate_best and bwer_best and aggregate_best.get("run_id") == bwer_best.get("run_id"))),
                }
            )
    return output


def _worst_slice_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": row.get("experiment_id", ""),
            "dataset": row.get("dataset", ""),
            "deployment_axis": row.get("deployment_axis", ""),
            "run_id": row.get("run_id", ""),
            "metric_family": row.get("metric_family", ""),
            "bwer_slice": row.get("raw_bwer_slice", ""),
            "raw_bwer": row.get("raw_bwer", ""),
            "standardised_bwer": row.get("standardised_bwer", ""),
            "worst_slice": row.get("worst_slice", ""),
            "tail_slices": row.get("tail_slices", ""),
        }
        for row in rows
    ]


def _support_rows(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _list_rows(registry, "support_notes", "support_note")


def _sensitivity_rows(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    defaults = registry.get("defaults", {})
    for exp in registry.get("experiments", []):
        rows.append(
            {
                "experiment_id": exp.get("experiment_id", ""),
                "dataset": exp.get("dataset", ""),
                "deployment_axis": exp.get("deployment_axis", ""),
                "tail_fraction_main": defaults.get("tail_fraction", ""),
                "min_samples_per_slice_main": defaults.get("min_samples_per_slice", ""),
                "standardised_balance": exp.get("standardised_balance", ""),
                "sensitivity_note": "Use task-specific support and missing-cell diagnostics; do not compare raw metric magnitudes across task types.",
            }
        )
    return rows


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig: Any, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"))
    fig.savefig(path_stem.with_suffix(".pdf"))


def _barh(ax: Any, labels: Sequence[str], values: Sequence[float], title: str, xlabel: str) -> None:
    y = list(range(len(labels)))
    ax.barh(y, values, color="#0072B2")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)


def _figures(output: Path, rows: Sequence[Mapping[str, Any]], claims: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    plt = _configure_matplotlib()
    figures = ensure_dir(output / "figures")
    for stale_name in ("worst_slice_heatmap_by_dataset.png", "worst_slice_heatmap_by_dataset.pdf"):
        stale = figures / stale_name
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass
    paths = {
        "framework_overview": figures / "framework_overview",
        "deployment_axis_matrix": figures / "deployment_axis_matrix",
        "average_vs_bwer_cross_dataset": figures / "average_vs_bwer_cross_dataset",
        "worst_slice_barplot_by_run": figures / "worst_slice_barplot_by_run",
        "reben_sensor_mode_summary": figures / "reben_sensor_mode_summary",
        "claim_support_caveat_matrix": figures / "claim_support_caveat_matrix",
    }

    # Framework overview.
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")
    boxes = [
        ("Average metrics", 0.08, 0.55),
        ("BWER by slice", 0.35, 0.55),
        ("Caveats/support", 0.62, 0.55),
        ("Paper claims", 0.35, 0.18),
    ]
    for text, x, y in boxes:
        ax.text(x, y, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#F2F2F2", "ec": "#555555"})
    for x0, y0, x1, y1 in [(0.18, 0.55, 0.27, 0.55), (0.45, 0.55, 0.54, 0.55), (0.62, 0.45, 0.45, 0.25), (0.35, 0.45, 0.35, 0.30)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("BWER-Audit synthesis workflow")
    _save(fig, paths["framework_overview"])
    plt.close(fig)

    # Deployment axis matrix.
    axis_counts: dict[str, int] = {}
    for row in rows:
        axis_counts[str(row.get("deployment_axis", ""))] = axis_counts.get(str(row.get("deployment_axis", "")), 0) + 1
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    _barh(ax, list(axis_counts), list(axis_counts.values()), "Deployment axes represented", "formal run count")
    _save(fig, paths["deployment_axis_matrix"])
    plt.close(fig)

    # Average vs BWER, faceted by experiment to avoid cross-task numeric equivalence.
    valid_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if math.isnan(_float(row.get("aggregate_score"))) or math.isnan(_float(row.get("raw_bwer"))):
            continue
        valid_groups.setdefault(str(row.get("experiment_id", "")), []).append(row)
    n_panels = max(1, len(valid_groups))
    fig, axes = plt.subplots(1, n_panels, figsize=(max(4.0, 3.6 * n_panels), 3.4), squeeze=False)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    if not valid_groups:
        axes[0][0].axis("off")
        axes[0][0].text(0.5, 0.5, "No formal rows with both aggregate score and Raw-BWER.", ha="center", va="center")
    for ax, (exp_id, items) in zip(axes[0], valid_groups.items()):
        for idx, row in enumerate(items):
            agg = _float(row.get("aggregate_score"))
            bwer = _float(row.get("raw_bwer"))
            ax.scatter(agg, bwer, color=palette[idx % len(palette)], s=52, edgecolor="white", linewidth=0.7)
            ax.text(agg, bwer, str(row.get("run_id")), fontsize=6.5, ha="left", va="bottom")
        first = items[0]
        ax.set_title(f"{first.get('dataset')}\n{first.get('metric_family')}", fontsize=9)
        ax.set_xlabel(str(first.get("aggregate_metric_name", "aggregate score")))
        ax.set_ylabel("Raw-BWER")
        ax.grid(alpha=0.22)
    for ax in axes[0][len(valid_groups):]:
        ax.axis("off")
    fig.suptitle("Average score vs Raw-BWER, faceted by task family", y=1.02, fontsize=10)
    _save(fig, paths["average_vs_bwer_cross_dataset"])
    plt.close(fig)

    # Worst-slice bar plot by run.
    valid_worst = [row for row in rows if not math.isnan(_float(row.get("raw_bwer")))]
    valid_worst = sorted(valid_worst, key=lambda row: _float(row.get("raw_bwer")), reverse=True)
    fig, ax = plt.subplots(figsize=(7.6, max(2.8, 0.34 * len(valid_worst))))
    if valid_worst:
        labels = [f"{row.get('dataset')} | {row.get('run_id')}" for row in valid_worst]
        values = [_float(row.get("raw_bwer"), 0.0) for row in valid_worst]
        _barh(ax, labels, values, "Raw-BWER by formal run", "Raw-BWER")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid Raw-BWER rows available.", ha="center", va="center")
    _save(fig, paths["worst_slice_barplot_by_run"])
    plt.close(fig)

    # reBEN sensor mode summary.
    reben = [row for row in rows if row.get("experiment_id") == "reben_croma_sensor_mode"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), squeeze=False)
    metrics = [
        ("aggregate_score", "macro-AP", "higher is better"),
        ("mean_bce_risk", "mean BCE risk", "lower is better"),
        ("raw_bwer", "country Raw-BWER", "lower is better"),
        ("standardised_bwer", "country | class BWER", "lower is better"),
    ]
    labels = [str(row.get("sensor_mode") or row.get("run_id")) for row in reben]
    for ax, (column, title, subtitle) in zip(axes.ravel(), metrics):
        values = [_float(row.get(column)) for row in reben]
        clean_values = [0.0 if math.isnan(value) else value for value in values]
        ax.bar(labels, clean_values, color=["#0072B2", "#D55E00", "#009E73"][: len(labels)])
        for idx, value in enumerate(values):
            label = "NA" if math.isnan(value) else f"{value:.3f}"
            ax.text(idx, clean_values[idx] + (max(clean_values or [0.0]) * 0.025 + 0.005), label, ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{title}\n{subtitle}", fontsize=9)
        ax.set_ylim(0, max(clean_values + [0.01]) * 1.22)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("reBEN/CROMA sensor-mode summary", y=1.02, fontsize=10)
    _save(fig, paths["reben_sensor_mode_summary"])
    plt.close(fig)

    # Claim support matrix.
    support_order = ["supported", "supported_croma_only", "supported_case_study", "supported_protocol_aware", "supported_by_sanity"]
    counts = {key: 0 for key in support_order}
    for row in claims:
        key = str(row.get("support", ""))
        counts[key] = counts.get(key, 0) + 1
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    _barh(ax, list(counts), list(counts.values()), "Claim support caveat matrix", "claim count")
    _save(fig, paths["claim_support_caveat_matrix"])
    plt.close(fig)
    return {key: value.with_suffix(".png") for key, value in paths.items()}


def _write_reports(output: Path, registry: Mapping[str, Any], run_rows: Sequence[Mapping[str, Any]]) -> None:
    reports = ensure_dir(output / "reports")
    matrix_report = [
        "# Unified Audit Matrix v1 Report",
        "",
        "This report synthesizes completed BWER-Audit evidence across event/disaster, geography/location, and sensor/modality axes.",
        "",
        "The synthesis is post-hoc. It does not train models, rerun inference, recompute embeddings/logits/probabilities, or modify raw experiment outputs.",
        "",
        "Cross-task numeric values are not directly interchangeable: segmentation IoU-risk, single-label classification error, and multi-label BCE risk are separated by `metric_family`.",
    ]
    missing_metric_notes = [
        f"- `{row.get('run_id')}`: {row.get('metric_availability_note')}"
        for row in run_rows
        if str(row.get("metric_availability_note", "")).strip()
    ]
    if missing_metric_notes:
        matrix_report.extend(["", "## Metric Availability Notes", "", *missing_metric_notes])
    (reports / "unified_audit_matrix_report.md").write_text("\n".join(matrix_report) + "\n", encoding="utf-8")
    summary = [
        "# Paper-Ready Summary",
        "",
        "Average performance alone is insufficient to describe deployment reliability.",
        "BWER provides a common audit language for event/disaster, geography/location, and sensor/modality axes while preserving task-specific risk definitions.",
        "",
        "Main guarded claim: BWER exposes residual or redistributed tail risk that aggregate metrics alone do not summarize.",
        "",
        "Do not overclaim global fairness, causal bias, or direct numerical equivalence across metric families.",
    ]
    (reports / "paper_ready_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    outline = [
        "# Paper Outline",
        "",
        "1. Motivation: deployment reliability is slice-structured.",
        "2. Method: BWER-Audit and standardised BWER.",
        "3. Event/disaster case: Sen1Floods11.",
        "4. Geography/location case: fMoW-Sentinel.",
        "5. Sensor/modality case: BigEarthNet v2 / reBEN with CROMA.",
        "6. Selective risk and confidence availability.",
        "7. Caveats, blocked components, and reproducibility.",
    ]
    (output / "paper_outline.md").write_text("\n".join(outline) + "\n", encoding="utf-8")


def _figure_notes(output: Path) -> None:
    notes = [
        "# Figure Notes",
        "",
        "- `framework_overview`: supports the synthesis workflow only; it is not result evidence.",
        "- `deployment_axis_matrix`: shows coverage of event, geography, and sensor axes.",
        "- `average_vs_bwer_cross_dataset`: shows aggregate-vs-BWER relationships with metric-family caveats; do not compare raw y-values across tasks as equivalent risks.",
        "- `worst_slice_barplot_by_run`: bar plot of formal-run Raw-BWER; it is intentionally not called a heatmap.",
        "- `reben_sensor_mode_summary`: CROMA-only sensor-mode comparison across macro-AP, mean BCE risk, country Raw-BWER, and country | class BWER; BIFOLD is blocked.",
        "- `selective_risk_curves_cross_dataset` and `retained_coverage_by_slice_heatmap` are produced by the selective-risk script. The main selective curve uses `slice_variable == all` only.",
        "- `claim_support_caveat_matrix`: summarizes claim support categories and caveat burden.",
    ]
    (output / "figures" / "figure_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def build_unified_matrix(registry_path: Path, output_dir: Path | None = None) -> dict[str, Path]:
    registry = _load_yaml(registry_path)
    output = ensure_dir(output_dir or Path(registry.get("defaults", {}).get("output_unified", "outputs/unified_audit_matrix_v1")))
    figures = ensure_dir(output / "figures")
    reports = ensure_dir(output / "reports")
    experiment_rows = _registry_experiment_rows(registry)
    run_rows = _registry_run_rows(registry)
    slice_rows = _registry_slice_rows(registry)
    caveats = _list_rows(registry, "caveats", "caveat")
    protocol_risks = _list_rows(registry, "protocol_risks", "protocol_risk")
    claims = _claim_rows(registry)
    avg_vs_bwer = _average_vs_bwer(run_rows)
    worst = _worst_slice_summary(run_rows)
    support = _support_rows(registry)
    sensitivity = _sensitivity_rows(registry)

    artifacts = {
        "unified_experiment_matrix": output / "unified_experiment_matrix.csv",
        "unified_main_results_table": output / "unified_main_results_table.csv",
        "unified_slice_registry": output / "unified_slice_registry.csv",
        "unified_caveats_table": output / "unified_caveats_table.csv",
        "unified_protocol_risk_table": output / "unified_protocol_risk_table.csv",
        "average_vs_bwer_cross_dataset": output / "average_vs_bwer_cross_dataset.csv",
        "worst_slice_summary_cross_dataset": output / "worst_slice_summary_cross_dataset.csv",
        "support_diagnostics_summary": output / "support_diagnostics_summary.csv",
        "sensitivity_summary": output / "sensitivity_summary.csv",
        "claim_support_table": output / "claim_support_table.csv",
        "scientific_findings_unified_audit_matrix_v1": output / "scientific_findings_unified_audit_matrix_v1.md",
        "paper_outline": output / "paper_outline.md",
        "unified_audit_matrix_report": reports / "unified_audit_matrix_report.md",
        "paper_ready_summary": reports / "paper_ready_summary.md",
        "figure_notes": figures / "figure_notes.md",
    }
    write_csv(artifacts["unified_experiment_matrix"], experiment_rows)
    write_csv(artifacts["unified_main_results_table"], run_rows)
    write_csv(artifacts["unified_slice_registry"], slice_rows)
    write_csv(artifacts["unified_caveats_table"], caveats)
    write_csv(artifacts["unified_protocol_risk_table"], protocol_risks)
    write_csv(artifacts["average_vs_bwer_cross_dataset"], avg_vs_bwer)
    write_csv(artifacts["worst_slice_summary_cross_dataset"], worst)
    write_csv(artifacts["support_diagnostics_summary"], support)
    write_csv(artifacts["sensitivity_summary"], sensitivity)
    write_csv(artifacts["claim_support_table"], claims)
    findings = [
        "# Unified Audit Matrix v1",
        "",
        "Recorded: 2026-06-05.",
        "",
        "This post-hoc synthesis records BWER-Audit evidence across event/disaster, geography/location, and sensor/modality deployment axes.",
        "",
        "Scientific framing: average performance alone is insufficient to describe deployment reliability, and BWER provides a unified audit language while preserving task-specific risk definitions.",
        "",
        "Caveat: cross-task plots use metric-family labels, ranks, and within-task annotations. They must not imply numerical equivalence between IoU-risk, classification error, and BCE risk.",
    ]
    artifacts["scientific_findings_unified_audit_matrix_v1"].write_text("\n".join(findings) + "\n", encoding="utf-8")
    _write_reports(output, registry, run_rows)
    figure_paths = _figures(output, run_rows, claims)
    _figure_notes(output)
    artifacts.update({f"figure_{key}": path for key, path in figure_paths.items()})
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Unified Audit Matrix v1 from registry and completed outputs.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = build_unified_matrix(args.registry, args.output_dir)
    print(f"[unified] output_dir={args.output_dir or 'registry default'}")
    for name, path in artifacts.items():
        print(f"[unified] {name}: {path}")


if __name__ == "__main__":
    main()
