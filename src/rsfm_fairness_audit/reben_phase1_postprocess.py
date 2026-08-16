from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


LABEL_BUDGET_SCHEMA = "geobwer.reben.label_budget.postprocess.v1"
PAIRED_SHIFT_SCHEMA = "geobwer.reben.paired_sensor_shift.postprocess.v1"
FINAL_EVIDENCE_SCHEMA = "geobwer.optimization_1_7.final_evidence.v1"
DEFAULT_BUDGETS = (0.05, 0.10, 0.25, 0.50, 1.00)
DEFAULT_SEEDS = (42, 73, 101)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value: Any) -> int:
    return int(round(_float(value)))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bool:
    return bool(rows) and all(math.isfinite(_float(row.get(column))) for row in rows for column in columns)


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _write_figure(fig: Any, stem: Path) -> list[Path]:
    paths = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    return paths


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    return plt


def _label_budget_figures(rows: Sequence[Mapping[str, Any]], output: Path) -> list[Path]:
    plt = _matplotlib()
    figure_dir = ensure_dir(output / "figures")
    metrics = (
        ("macro_ap", "Macro AP", True),
        ("macro_f1", "Macro F1", True),
        ("mean_risk", "Mean country risk", False),
        ("geobwer_beta_0_10", "GeoBWER (beta=0.10)", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), squeeze=False)
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    styles = (("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"))
    seeds = sorted({_int(row["seed"]) for row in rows})
    budgets = sorted({_float(row["budget_fraction"]) for row in rows})
    for panel_index, (ax, (column, label, _)) in enumerate(zip(axes.flat, metrics)):
        seed_series = []
        for seed_index, seed in enumerate(seeds):
            selected = sorted(
                (row for row in rows if _int(row["seed"]) == seed),
                key=lambda row: _float(row["budget_fraction"]),
            )
            values = [_float(row.get(column)) for row in selected]
            seed_series.append(values)
            ax.plot(
                [100.0 * _float(row["budget_fraction"]) for row in selected],
                values, color=palette[seed_index % len(palette)],
                linestyle=styles[seed_index % len(styles)][0], marker=styles[seed_index % len(styles)][1],
                linewidth=1.1, alpha=0.78, label=f"seed {seed}",
            )
        matrix = np.asarray(seed_series, dtype=float)
        if matrix.shape == (len(seeds), len(budgets)):
            x_values = 100.0 * np.asarray(budgets)
            ax.fill_between(x_values, np.min(matrix, axis=0), np.max(matrix, axis=0), color="#777777", alpha=0.13, label="seed range")
            ax.plot(x_values, np.mean(matrix, axis=0), color="#000000", linewidth=1.7, marker="x", label="seed mean")
        ax.set_xlabel("Labelled independent units (%)")
        ax.set_ylabel(label)
        ax.text(-0.14, 1.04, chr(ord("A") + panel_index), transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("reBEN TerraMind S2: nested label-budget sensitivity", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    paths = _write_figure(fig, figure_dir / "label_budget_curves")
    plt.close(fig)
    return paths


def postprocess_label_budget(
    result_dir: str | Path,
    *,
    expected_budgets: Sequence[float] = DEFAULT_BUDGETS,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Path]:
    output = Path(result_dir)
    summary_path = output / "label_budget_curves.csv"
    selections_path = output / "nested_budget_unit_selections.csv"
    manifest_path = output / "label_budget_manifest.json"
    for path in (summary_path, selections_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required label-budget result is missing: {path}")
    rows = read_csv_rows(summary_path)
    selections = read_csv_rows(selections_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {(int(seed), float(budget)) for seed in expected_seeds for budget in expected_budgets}
    observed_keys = {(_int(row.get("seed")), _float(row.get("budget_fraction"))) for row in rows}
    unique_rows = len(observed_keys) == len(rows)

    selected_by_key: dict[tuple[int, float], set[str]] = defaultdict(set)
    for row in selections:
        selected_by_key[(_int(row.get("seed")), _float(row.get("budget_fraction")))].add(
            str(row.get("independent_unit_id", ""))
        )
    nested = True
    full_sets: list[set[str]] = []
    selection_counts_match = True
    for seed in expected_seeds:
        previous: set[str] = set()
        for budget in sorted(float(value) for value in expected_budgets):
            current = selected_by_key.get((int(seed), budget), set())
            nested = nested and bool(current) and previous.issubset(current)
            previous = current
            matched = [row for row in rows if _int(row.get("seed")) == int(seed) and math.isclose(_float(row.get("budget_fraction")), budget)]
            selection_counts_match = selection_counts_match and len(matched) == 1 and len(current) == _int(matched[0].get("selected_independent_units"))
        full_sets.append(previous)
    full_budget_common_universe = bool(full_sets) and all(values == full_sets[0] for values in full_sets[1:])
    fixed_test = len({_int(row.get("test_samples")) for row in rows}) == 1
    primary_columns = ("macro_ap", "macro_f1", "mean_risk", "tail_risk_beta_0_10", "geobwer_beta_0_10")

    aggregate_rows: list[dict[str, Any]] = []
    for budget in sorted(float(value) for value in expected_budgets):
        selected = [row for row in rows if math.isclose(_float(row.get("budget_fraction")), budget)]
        for metric in primary_columns:
            aggregate_rows.append({"budget_fraction": budget, "metric": metric, **_summary(_float(row.get(metric)) for row in selected)})
    aggregate_path = output / "label_budget_seed_aggregate.csv"
    write_csv(aggregate_path, aggregate_rows)

    endpoint_rows: list[dict[str, Any]] = []
    low, high = min(expected_budgets), max(expected_budgets)
    for seed in expected_seeds:
        keyed = {
            _float(row.get("budget_fraction")): row
            for row in rows
            if _int(row.get("seed")) == int(seed)
        }
        if float(low) not in keyed or float(high) not in keyed:
            continue
        endpoint_rows.append({
            "seed": int(seed), "from_budget": float(low), "to_budget": float(high),
            "delta_macro_ap": _float(keyed[float(high)].get("macro_ap")) - _float(keyed[float(low)].get("macro_ap")),
            "delta_macro_f1": _float(keyed[float(high)].get("macro_f1")) - _float(keyed[float(low)].get("macro_f1")),
            "delta_mean_risk": _float(keyed[float(high)].get("mean_risk")) - _float(keyed[float(low)].get("mean_risk")),
            "delta_tail_risk": _float(keyed[float(high)].get("tail_risk_beta_0_10")) - _float(keyed[float(low)].get("tail_risk_beta_0_10")),
            "delta_geobwer": _float(keyed[float(high)].get("geobwer_beta_0_10")) - _float(keyed[float(low)].get("geobwer_beta_0_10")),
        })
    endpoint_path = output / "label_budget_endpoint_changes.csv"
    write_csv(endpoint_path, endpoint_rows)
    figures = _label_budget_figures(rows, output)
    gates = {
        "campaign_manifest_complete": manifest.get("status") == "complete",
        "expected_seed_budget_grid_complete": observed_keys == expected_keys and unique_rows,
        "primary_metrics_finite": _finite_rows(rows, primary_columns),
        "nested_independent_unit_selections": nested,
        "selection_counts_match_summary": selection_counts_match,
        "full_budget_uses_common_unit_universe": full_budget_common_universe,
        "validation_and_test_fixed_by_protocol": bool(manifest.get("validation_and_test_fixed")),
        "test_not_used_for_selection": manifest.get("test_used_for_selection") is False,
        "test_sample_count_fixed": fixed_test,
        "endpoint_contrasts_complete": len(endpoint_rows) == len(expected_seeds),
        "figures_include_vector_and_raster": {path.suffix for path in figures} == {".png", ".pdf"},
    }
    warnings = []
    if not all("selected_positive_labels" in row for row in rows):
        warnings.append("label_coverage_columns_absent_in_legacy_runner_output")
    audit = output / "label_budget_result_audit.json"
    audit.write_text(json.dumps({
        "schema": LABEL_BUDGET_SCHEMA, "status": "pass" if all(gates.values()) else "fail",
        "gates": gates, "warnings": warnings, "row_count": len(rows),
        "inference_role": "descriptive_seed_sensitivity_no_null_hypothesis_significance_claim",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report = output / "label_budget_result_report.md"
    report.write_text("\n".join([
        "# reBEN label-budget result audit", "", "## Material Passport", "",
        f"- Verification Status: `{'VERIFIED' if all(gates.values()) else 'FAILED'}`",
        "- Evidence role: descriptive seed sensitivity; three seeds do not justify population-level significance claims.",
        "- Frozen design: nested independent units; validation and test sets fixed; test excluded from selection.", "",
        "## Audit", "", *[f"- `{name}`: `{passed}`" for name, passed in gates.items()], "",
        "## Interpretation boundary", "",
        "The curves estimate how aggregate performance, mean geographic risk, tail risk, and GeoBWER change with labelled independent-unit budget. Non-monotonic seed paths remain reported rather than smoothed away.",
    ]) + "\n", encoding="utf-8")
    if not all(gates.values()):
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"Label-budget result audit failed: {failed}")
    return {"audit": audit, "aggregate": aggregate_path, "endpoints": endpoint_path, "report": report, "figures": figures[0].parent}


def _paired_shift_figures(
    summaries: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    burdens: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    plt = _matplotlib()
    figure_dir = ensure_dir(output / "figures")
    paths: list[Path] = []
    metrics = (
        ("macro_ap", "Macro AP"), ("mean_risk", "Mean country risk"),
        ("tail_risk_beta_0_10", "Tail country risk"), ("geobwer_beta_0_10", "GeoBWER"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), squeeze=False)
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    styles = (("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"))
    seeds = sorted({_int(row["seed"]) for row in summaries})
    for panel_index, (ax, (column, label)) in enumerate(zip(axes.flat, metrics)):
        seed_series = []
        for seed_index, seed in enumerate(seeds):
            pair = {str(row["domain"]): _float(row.get(column)) for row in summaries if _int(row["seed"]) == seed}
            values = [pair.get("ID", np.nan), pair.get("OOD", np.nan)]
            seed_series.append(values)
            ax.plot(
                [0, 1], values, color=palette[seed_index % len(palette)],
                linestyle=styles[seed_index % len(styles)][0], marker=styles[seed_index % len(styles)][1],
                alpha=0.78, label=f"seed {seed}",
            )
        matrix = np.asarray(seed_series, dtype=float)
        ax.fill_between([0, 1], np.min(matrix, axis=0), np.max(matrix, axis=0), color="#777777", alpha=0.13, label="seed range")
        ax.plot([0, 1], np.mean(matrix, axis=0), color="#000000", linewidth=1.7, marker="x", label="seed mean")
        ax.set_xticks([0, 1], ["S2 ID", "S1 OOD"])
        ax.set_ylabel(label)
        ax.text(-0.14, 1.04, chr(ord("A") + panel_index), transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("reBEN TerraMind: locked-head paired sensor shift", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    paths.extend(_write_figure(fig, figure_dir / "paired_sensor_shift_id_ood"))
    plt.close(fig)

    delta_columns = (
        ("delta_mean_risk", "Mean risk"), ("delta_tail_risk", "Tail risk"),
        ("delta_geobwer", "GeoBWER"), ("tail_acceleration_minus_mean", "Tail acceleration"),
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    seeds = [_int(row["seed"]) for row in deltas]
    width = 0.18
    x = np.arange(len(seeds), dtype=float)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    hatches = ("", "//", "xx", "..")
    for index, ((column, display_name), color) in enumerate(zip(delta_columns, colors)):
        ax.bar(
            x + (index - 1.5) * width, [_float(row.get(column)) for row in deltas], width,
            label=display_name, color=color, hatch=hatches[index], edgecolor="#222222", linewidth=0.4,
        )
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("Probe seed")
    ax.set_ylabel("S1 OOD minus S2 ID")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    paths.extend(_write_figure(fig, figure_dir / "paired_sensor_shift_deltas"))
    plt.close(fig)

    if burdens:
        ranked = sorted(burdens, key=lambda row: abs(_float(row.get("mean_delta_risk"))), reverse=True)[:20]
        fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.28 * len(ranked))))
        labels = [f"{row.get('slice_axis')}: {row.get('slice_value')}" for row in reversed(ranked)]
        values = [_float(row.get("mean_delta_risk")) for row in reversed(ranked)]
        colors = ["#D55E00" if value > 0 else "#0072B2" for value in values]
        bars = ax.barh(labels, values, color=colors, edgecolor="#222222", linewidth=0.4)
        for bar, value in zip(bars, values):
            bar.set_hatch("//" if value > 0 else "..")
        ax.axvline(0.0, color="#222222", linewidth=0.9)
        ax.set_xlabel("Mean paired risk change (S1 OOD - S2 ID)")
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        paths.extend(_write_figure(fig, figure_dir / "paired_sensor_shift_burden_carriers"))
        plt.close(fig)
    return paths


def postprocess_paired_sensor_shift(
    result_dir: str | Path,
    *,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Path]:
    output = Path(result_dir)
    summary_path = output / "paired_shift_seed_panel.csv"
    delta_path = output / "paired_shift_delta_seed_panel.csv"
    manifest_path = output / "paired_shift_panel_manifest.json"
    preflight_path = output / "paired_shift_preflight.json"
    for path in (summary_path, delta_path, manifest_path, preflight_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required paired-shift result is missing: {path}")
    summaries = read_csv_rows(summary_path)
    deltas = read_csv_rows(delta_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    expected = {(int(seed), domain) for seed in expected_seeds for domain in ("ID", "OOD")}
    observed = {(_int(row.get("seed")), str(row.get("domain"))) for row in summaries}
    delta_seeds = {_int(row.get("seed")) for row in deltas}
    primary = ("macro_ap", "macro_f1", "mean_risk", "tail_risk_beta_0_10", "geobwer_beta_0_10")
    delta_columns = ("delta_mean_risk", "delta_tail_risk", "delta_geobwer", "tail_acceleration_minus_mean")

    aggregate_rows = []
    for column in delta_columns:
        aggregate_rows.append({"metric": column, **_summary(_float(row.get(column)) for row in deltas)})
    aggregate_path = output / "paired_shift_delta_seed_aggregate.csv"
    write_csv(aggregate_path, aggregate_rows)

    slice_rows: list[dict[str, str]] = []
    for filename in ("paired_shift_label_deltas.csv", "paired_shift_country_deltas.csv"):
        path = output / filename
        if path.is_file():
            slice_rows.extend(read_csv_rows(path))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in slice_rows:
        grouped[(str(row.get("slice_axis")), str(row.get("slice_value")))].append(row)
    burden_rows = []
    for (axis, value), group in grouped.items():
        values = [_float(row.get("delta_risk")) for row in group]
        burden_rows.append({
            "slice_axis": axis, "slice_value": value, "seed_count": len(group),
            "ood_worse_seed_count": sum(value > 0 for value in values),
            "mean_delta_risk": float(np.mean(values)), "min_delta_risk": float(np.min(values)),
            "max_delta_risk": float(np.max(values)),
        })
    burden_path = output / "paired_shift_burden_recurrence.csv"
    write_csv(burden_path, burden_rows)
    figures = _paired_shift_figures(summaries, deltas, burden_rows, output)
    preflight_ready = preflight.get("status") in {"ready", "formal_ready"}
    gates = {
        "formal_preflight_passed": preflight_ready,
        "panel_manifest_complete": manifest.get("status") == "complete",
        "same_s2_trained_head_within_seed": manifest.get("same_s2_trained_head_within_seed") is True,
        "test_not_used_for_selection": manifest.get("test_used_for_selection") is False,
        "expected_seed_domain_grid_complete": observed == expected and len(summaries) == len(expected),
        "expected_delta_seed_grid_complete": delta_seeds == {int(seed) for seed in expected_seeds} and len(deltas) == len(expected_seeds),
        "primary_metrics_finite": _finite_rows(summaries, primary),
        "delta_metrics_finite": _finite_rows(deltas, delta_columns),
        "paired_common_support": bool(preflight.get("paired_sample_ids_targets_and_metadata")),
        "effective_robustness_not_claimed": manifest.get("effective_robustness_claimed", False) is False,
        "slice_burden_tables_present": bool(slice_rows),
        "figures_include_vector_and_raster": ".png" in {path.suffix for path in figures} and ".pdf" in {path.suffix for path in figures},
    }
    audit_path = output / "paired_shift_result_audit.json"
    audit_path.write_text(json.dumps({
        "schema": PAIRED_SHIFT_SCHEMA, "status": "pass" if all(gates.values()) else "fail",
        "gates": gates, "summary_rows": len(summaries), "delta_rows": len(deltas),
        "burden_rows": len(burden_rows),
        "inference_role": "paired_external_validity_descriptive_seed_panel",
        "terminology": "OOD degradation; effective robustness is not claimed",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report = output / "paired_shift_result_report.md"
    report.write_text("\n".join([
        "# reBEN paired sensor-shift result audit", "", "## Material Passport", "",
        f"- Verification Status: `{'VERIFIED' if all(gates.values()) else 'FAILED'}`",
        "- Estimand: unchanged S2-trained head evaluated on paired S2 ID and S1 OOD test representations.",
        "- Terminology: OOD degradation. EarthShift effective robustness is not claimed without a reference regression.", "",
        "## Audit", "", *[f"- `{name}`: `{passed}`" for name, passed in gates.items()], "",
        "## Interpretation boundary", "",
        "Positive deltas mean larger risk under S1 OOD. Tail acceleration is delta tail risk minus delta mean risk. Slice burden rows are descriptive unless uncertainty and multiplicity controls are added.",
    ]) + "\n", encoding="utf-8")
    if not all(gates.values()):
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"Paired sensor-shift result audit failed: {failed}")
    return {"audit": audit_path, "aggregate": aggregate_path, "burdens": burden_path, "report": report, "figures": figures[0].parent}


def _git_value(root: Path, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _evidence_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    allowed = {".csv", ".json", ".md", ".png", ".pdf", ".yaml", ".yml"}
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def build_final_optimization_evidence(
    project_root: str | Path,
    base_result_dir: str | Path,
    label_budget_dir: str | Path,
    paired_shift_dir: str | Path,
    output_dir: str | Path,
    *,
    allow_pending: bool = False,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    base = Path(base_result_dir)
    label = Path(label_budget_dir)
    paired = Path(paired_shift_dir)
    output = ensure_dir(output_dir)
    base_validation_path = base / "optimization_1_7_validation.json"
    base_validation = json.loads(base_validation_path.read_text(encoding="utf-8")) if base_validation_path.is_file() else {}
    item_audits: dict[int, dict[str, Any]] = {}
    for item, directory, name in ((6, label, "label_budget_result_audit.json"), (7, paired, "paired_shift_result_audit.json")):
        path = directory / name
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        item_audits[item] = {"path": str(path), "status": payload.get("status", "pending"), "gates": payload.get("gates", {})}
    base_pass = base_validation.get("passes") is True
    complete = base_pass and all(item_audits[item]["status"] == "pass" for item in (6, 7))
    evidence_rows: list[dict[str, Any]] = []
    for item_range, directory in (("1-5", base), ("6", label), ("7", paired)):
        for path in _evidence_files(directory):
            if output in path.parents:
                continue
            evidence_rows.append({
                "items": item_range, "path": str(path.resolve()), "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    inventory_path = output / "optimization_1_7_final_evidence_inventory.csv"
    write_csv(inventory_path, evidence_rows)
    payload = {
        "schema": FINAL_EVIDENCE_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if complete else "pending_required_results",
        "finality": complete,
        "git_head": _git_value(root, ("rev-parse", "HEAD")),
        "git_status_at_generation": _git_value(root, ("status", "--short")),
        "scope": {"included": list(range(1, 8)), "not_started": list(range(8, 18))},
        "scope_guard": "Items 8-17 have no formal experiment, training, new data engineering, preflight, result audit, or figure in this workflow.",
        "items": {
            "1-5": {"status": "pass" if base_pass else "pending", "validation": str(base_validation_path)},
            "6": item_audits[6], "7": item_audits[7],
        },
        "evidence_file_count": len(evidence_rows),
        "inventory": str(inventory_path),
    }
    manifest_path = output / "optimization_1_7_final_evidence_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output / "optimization_1_7_final_evidence_report.md"
    report_path.write_text("\n".join([
        "# Optimization 1–7 final evidence status", "", "## Material Passport", "",
        f"- Verification Status: `{'VERIFIED' if complete else 'PENDING'}`",
        f"- Evidence files inventoried: `{len(evidence_rows)}`", "- Included scope: items 1–7.",
        "- Items 8–17: not started; no formal experiment, training, new data engineering, preflight, audit, or figure was launched.", "",
        "## Item gates", "",
        f"- Items 1–5: `{'pass' if base_pass else 'pending'}`",
        f"- Item 6: `{item_audits[6]['status']}`", f"- Item 7: `{item_audits[7]['status']}`", "",
        "A manifest marked `pending_required_results` is a readiness record, not final empirical evidence. It becomes final only when all three item gates pass.",
    ]) + "\n", encoding="utf-8")
    if not complete and not allow_pending:
        raise RuntimeError("Final evidence is incomplete: items 1-5, 6, and 7 must all pass their audits.")
    return {"manifest": manifest_path, "inventory": inventory_path, "report": report_path}


__all__ = [
    "build_final_optimization_evidence",
    "postprocess_label_budget",
    "postprocess_paired_sensor_shift",
]
