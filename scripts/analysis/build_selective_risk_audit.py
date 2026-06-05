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
    except ImportError as exc:  # pragma: no cover
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


def _normalize_source_path(path: str | Path) -> str:
    text = str(path)
    replacements = {
        "/content/outputs/content/outputs/": "/content/outputs/",
        "\\content\\outputs\\content\\outputs\\": "\\content\\outputs\\",
    }
    changed = True
    while changed:
        changed = False
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
                changed = True
    return text


def _path_tokens(path: Path, extra_tokens: Sequence[str] | None = None) -> list[str]:
    tokens = [str(token) for token in extra_tokens or [] if str(token)]
    for part in path.parts:
        if "reben_croma_sensor_mode_audit" in part:
            tokens.append(part)
        elif part in {"croma_s1", "croma_s2", "croma_s1_plus_s2"}:
            tokens.append(part)
    output: list[str] = []
    for token in tokens:
        if token not in output:
            output.append(token)
    return output


def _search_roots_for_candidate(path: Path) -> list[Path]:
    roots: list[Path] = []
    if path.exists() and path.is_dir():
        roots.append(path)
    if path.parent.exists():
        roots.append(path.parent)
    else:
        for ancestor in path.parents:
            if ancestor.exists() and ancestor != ancestor.parent:
                roots.append(ancestor)
                break
    output: list[Path] = []
    for root in roots:
        if root not in output and root.exists() and root.is_dir():
            output.append(root)
    return output


def _discover_existing_csv(
    paths: Sequence[str | Path] | None,
    *,
    filename: str,
    match_tokens: Sequence[str] | None = None,
) -> Path | None:
    exact = _first_existing(paths)
    if exact is not None:
        return exact
    candidates: list[Path] = []
    allow_test_dirs = any("test_" in str(Path(value)) or "pytest" in str(Path(value)) for value in paths or [])
    for value in paths or []:
        path = Path(value)
        tokens = _path_tokens(path, match_tokens)
        for root in _search_roots_for_candidate(path):
            try:
                matches = list(root.rglob(filename))
            except OSError:
                continue
            for match in matches:
                text = str(match)
                if not allow_test_dirs and any(part.startswith("test_") or part.startswith("pytest") for part in match.parts):
                    continue
                required_dir_tokens = [token for token in tokens if "reben_croma_sensor_mode_audit" in token]
                if required_dir_tokens and not all(token in text for token in required_dir_tokens):
                    continue
                if not required_dir_tokens and tokens and not any(token in text for token in tokens):
                    continue
                candidates.append(match)
    if not candidates:
        return None
    candidates = sorted(
        set(candidates),
        key=lambda item: (
            len(item.parts),
            0 if "bwer" not in item.parts else 1,
            str(item),
        ),
    )
    return candidates[0]


def _sensor_mode_alias(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "s1": "S1",
        "croma_s1": "S1",
        "s2": "S2",
        "croma_s2": "S2",
        "s1+s2": "S1+S2",
        "s1_plus_s2": "S1+S2",
        "croma_s1_plus_s2": "S1+S2",
    }
    return aliases.get(text, str(value or "").strip())


def _run_id_for_mode(mode: str) -> str:
    canonical = _sensor_mode_alias(mode)
    if canonical == "S1":
        return "croma_s1"
    if canonical == "S2":
        return "croma_s2"
    if canonical == "S1+S2":
        return "croma_s1_plus_s2"
    return str(mode or "")


def _display_label(value: Any) -> str:
    text = str(value or "").strip()
    labels = {
        "croma_s1": "CROMA S1",
        "croma_s2": "CROMA S2",
        "croma_s1_plus_s2": "CROMA S1+S2",
        "resnet50_13band": "ResNet-50",
        "dofa_scaled10000": "DOFA scaled",
        "sen1floods11_closure": "Sen1Floods11",
        "fmow_sentinel_step3": "fMoW-Sentinel",
        "reben_croma_sensor_mode": "BigEarthNet v2 / reBEN",
        "event_disaster": "Event/disaster",
        "geography_location": "Geography/location",
        "sensor_modality": "Sensor/modality",
    }
    return labels.get(text, text.replace("_", " ").strip().title() if text else "")


def _shorten(text: Any, max_chars: int = 60) -> str:
    value = str(text or "")
    return value if len(value) <= max_chars else value[: max_chars - 1].rstrip() + "…"


def _modes_in_selective_summary(path: Path | None) -> set[str]:
    if path is None:
        return set()
    modes: set[str] = set()
    try:
        rows = read_csv_rows(path)
    except Exception:
        return modes
    for row in rows:
        mode = _sensor_mode_alias(row.get("sensor_mode", row.get("run_name", "")))
        if not mode:
            mode = _sensor_mode_alias(row.get("run_id", ""))
        if mode:
            modes.add(mode)
    return modes


def _unavailable_row(
    *,
    experiment_id: str,
    dataset: str,
    deployment_axis: str,
    run_id: str,
    reason: str,
    source_path: str = "",
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "deployment_axis": deployment_axis,
        "run_id": run_id,
        "sensor_mode": "",
        "coverage_target": "",
        "slice_variable": "unavailable",
        "slice_value": "unavailable",
        "confidence_threshold": "",
        "retained_count": "",
        "total_count": "",
        "retained_coverage": "",
        "abstention_rate": "",
        "mean_risk": "",
        "status": "unavailable",
        "reason": reason,
        "source_path": _normalize_source_path(source_path),
    }


def _availability_rows(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in registry.get("experiments", []):
        selective = exp.get("selective_risk", {}) or {}
        base = {
            "experiment_id": exp.get("experiment_id", ""),
            "dataset": exp.get("dataset", ""),
            "deployment_axis": exp.get("deployment_axis", ""),
            "task_type": exp.get("task_type", ""),
            "confidence_definition": selective.get("confidence_definition", ""),
            "availability_policy": selective.get("availability", ""),
        }
        source_summary = _discover_existing_csv(
            selective.get("source_summary_candidates"),
            filename="selective_risk_comparison.csv",
            match_tokens=["reben_croma_sensor_mode_audit_croma_comparison"],
        )
        if source_summary is not None:
            rows.append({**base, "run_id": "comparison_summary", "status": "available", "source_path": _normalize_source_path(source_summary), "reason": "existing_selective_summary"})
        source_summary_modes = _modes_in_selective_summary(source_summary)
        for run_id, spec in (selective.get("source_run_summary_candidates") or {}).items():
            paths = spec.get("paths", []) if isinstance(spec, Mapping) else spec
            mode = _sensor_mode_alias(spec.get("sensor_mode", "")) if isinstance(spec, Mapping) else ""
            table = _discover_existing_csv(paths, filename="selective_risk_summary.csv", match_tokens=[str(run_id)])
            source_path = table
            status = "available" if table else "unavailable_missing_selective_summary"
            reason = "existing_per_run_selective_summary" if table else "no per-run selective risk summary found at registry candidates"
            if table is None and mode in source_summary_modes and source_summary is not None:
                source_path = source_summary
                status = "available"
                reason = "existing_comparison_selective_summary_contains_run"
            rows.append(
                {
                    **base,
                    "run_id": run_id,
                    "status": status,
                    "source_path": _normalize_source_path(source_path or ""),
                    "reason": reason,
                }
            )
        prediction_map = selective.get("prediction_table_candidates") or {}
        if prediction_map:
            for run_id, paths in prediction_map.items():
                table = _first_existing(paths)
                rows.append(
                    {
                        **base,
                        "run_id": run_id,
                        "status": "available" if table else "unavailable_missing_prediction_table",
                        "source_path": _normalize_source_path(table or ""),
                        "reason": "prediction_table_with_confidence" if table else "no existing prediction/audit table found at registry candidates",
                    }
                )
        if not source_summary and not selective.get("source_run_summary_candidates") and not prediction_map:
            rows.append({**base, "run_id": "", "status": selective.get("availability", "unavailable"), "source_path": "", "reason": selective.get("reason", "")})
    return rows


def _confidence_registry(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in registry.get("experiments", []):
        selective = exp.get("selective_risk", {}) or {}
        rows.append(
            {
                "experiment_id": exp.get("experiment_id", ""),
                "dataset": exp.get("dataset", ""),
                "deployment_axis": exp.get("deployment_axis", ""),
                "task_type": exp.get("task_type", ""),
                "confidence_definition": selective.get("confidence_definition", ""),
                "confidence_column": selective.get("confidence_column", ""),
                "risk_column": selective.get("risk_column", ""),
                "slice_columns": ";".join(selective.get("slice_columns", []) or []),
                "availability_policy": selective.get("availability", ""),
            }
        )
    return rows


def _compute_selective_from_prediction_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    dataset: str,
    deployment_axis: str,
    run_id: str,
    risk_column: str,
    confidence_column: str,
    slice_columns: Sequence[str],
    coverages: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    usable = [dict(row) for row in rows if confidence_column in row and risk_column in row]
    if not usable:
        return [], [], [], []
    confidences = [_float(row.get(confidence_column)) for row in usable]
    risks = [_float(row.get(risk_column)) for row in usable]
    paired = [(row, conf, risk) for row, conf, risk in zip(usable, confidences, risks) if not math.isnan(conf) and not math.isnan(risk)]
    if not paired:
        return [], [], [], []
    summary: list[dict[str, Any]] = []
    retained_by_slice: list[dict[str, Any]] = []
    high_conf_errors: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    sorted_conf = sorted(conf for _, conf, _ in paired)
    for coverage in coverages:
        if not 0.0 < float(coverage) <= 1.0:
            continue
        q_index = max(0, min(len(sorted_conf) - 1, int(math.floor((1.0 - float(coverage)) * (len(sorted_conf) - 1)))))
        threshold = sorted_conf[q_index]
        retained = [(row, conf, risk) for row, conf, risk in paired if conf >= threshold]
        mean_risk = sum(risk for _, _, risk in retained) / max(1, len(retained))
        summary.append(
            {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "deployment_axis": deployment_axis,
                "run_id": run_id,
                "coverage_target": coverage,
                "confidence_threshold": threshold,
                "retained_count": len(retained),
                "total_count": len(paired),
                "retained_coverage": len(retained) / max(1, len(paired)),
                "abstention_rate": 1.0 - (len(retained) / max(1, len(paired))),
                "mean_risk": mean_risk,
                "risk_column": risk_column,
                "confidence_column": confidence_column,
            }
        )
        curve.append(summary[-1])
        for column in slice_columns:
            if not paired or column not in paired[0][0]:
                continue
            values = sorted({str(row.get(column)) for row, _, _ in paired if str(row.get(column, "")).strip()})
            for value in values:
                all_slice = [(row, conf, risk) for row, conf, risk in paired if str(row.get(column)) == value]
                kept = [(row, conf, risk) for row, conf, risk in retained if str(row.get(column)) == value]
                slice_risk = sum(risk for _, _, risk in kept) / max(1, len(kept)) if kept else float("nan")
                retained_by_slice.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset": dataset,
                        "deployment_axis": deployment_axis,
                        "run_id": run_id,
                        "coverage_target": coverage,
                        "slice_variable": column,
                        "slice_value": value,
                        "retained_count": len(kept),
                        "total_count": len(all_slice),
                        "retained_coverage": len(kept) / max(1, len(all_slice)),
                        "mean_risk": slice_risk,
                    }
                )
                high_conf_errors.append(
                    {
                        **retained_by_slice[-1],
                        "high_confidence_error": slice_risk,
                    }
                )
    return summary, retained_by_slice, high_conf_errors, curve


def _simple_bwer(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("experiment_id")), str(row.get("run_id")), str(row.get("coverage_target")), str(row.get("slice_variable"))), []).append(row)
    for (experiment_id, run_id, coverage, slice_variable), items in grouped.items():
        risks = [_float(row.get("mean_risk")) for row in items]
        risks = [value for value in risks if not math.isnan(value)]
        if not risks:
            continue
        mean = sum(risks) / len(risks)
        worst = max(risks)
        output.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "coverage_target": coverage,
                "slice_variable": slice_variable,
                "mean_slice_risk": mean,
                "worst_slice_risk": worst,
                "selective_bwer_proxy": worst - mean,
                "note": "Post-hoc selective BWER proxy over retained slice means; compare within metric_family only.",
            }
        )
    return output


def _read_existing_selective_summary(
    exp: Mapping[str, Any],
    path: Path,
    *,
    run_id: str = "",
    sensor_mode: str = "",
) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    output = []
    for row in rows:
        mode = _sensor_mode_alias(row.get("sensor_mode", sensor_mode))
        resolved_run_id = str(row.get("run_id", row.get("run_name", run_id or _run_id_for_mode(mode))))
        output.append(
            {
                "experiment_id": exp.get("experiment_id", ""),
                "dataset": exp.get("dataset", ""),
                "deployment_axis": exp.get("deployment_axis", ""),
                **dict(row),
                "run_id": resolved_run_id,
                "sensor_mode": mode,
                "status": row.get("status", "available"),
                "source_path": _normalize_source_path(path),
            }
        )
    return output


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
        }
    )
    return plt


def _save(fig: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.25)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def _figures(output: Path, summary: Sequence[Mapping[str, Any]], retained: Sequence[Mapping[str, Any]], caveats: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    plt = _configure_matplotlib()
    figures = ensure_dir(output / "figures")
    curve_stem = figures / "selective_risk_curves_cross_dataset"
    heat_stem = figures / "retained_coverage_by_slice_heatmap"
    caveat_stem = figures / "claim_support_caveat_matrix"

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    all_rows = [row for row in summary if str(row.get("slice_variable", "")) == "all" and str(row.get("status", "available")) != "unavailable"]
    for key in sorted({(str(row.get("experiment_id")), str(row.get("run_id", row.get("sensor_mode", "")))) for row in all_rows}):
        items = [row for row in all_rows if (str(row.get("experiment_id")), str(row.get("run_id", row.get("sensor_mode", "")))) == key]
        x = [_float(row.get("retained_coverage", row.get("coverage_target"))) for row in items]
        y = [_float(row.get("mean_risk")) for row in items]
        pairs = sorted((a, b) for a, b in zip(x, y) if not math.isnan(a) and not math.isnan(b))
        if pairs:
            ax.plot([a for a, _ in pairs], [b for _, b in pairs], marker="o", linewidth=1.8, label=" / ".join(_display_label(part) for part in key if part))
    if not ax.lines:
        ax.text(0.5, 0.5, "No overall selective-risk rows available.", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Overall risk under confidence retention", pad=12)
    ax.set_xlabel("Retained coverage")
    ax.set_ylabel("Mean retained risk")
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend(frameon=False, fontsize=8)
    _save(fig, curve_stem)
    plt.close(fig)

    slice_rows = [
        row
        for row in retained
        if str(row.get("slice_variable", "")) not in {"", "all", "unavailable"}
        and not math.isnan(_float(row.get("mean_risk")))
    ]
    slice_rows = sorted(
        slice_rows,
        key=lambda row: (
            _float(row.get("mean_risk"), 0.0),
            abs(_float(row.get("retained_coverage"), 1.0) - 1.0),
            _float(row.get("total_count"), 0.0),
        ),
        reverse=True,
    )[:20]
    fig, ax = plt.subplots(figsize=(9.8, max(5.0, 0.42 * len(slice_rows))))
    if slice_rows:
        labels = [
            _shorten(
                f"{_display_label(row.get('run_id', row.get('sensor_mode', '')))} | "
                f"{_display_label(row.get('slice_variable'))}: {row.get('slice_value')} | "
                f"coverage { _float(row.get('coverage_target')):.2g}",
                78,
            )
            for row in slice_rows
        ]
        matrix = [
            [
                _float(row.get("mean_risk"), 0.0),
                _float(row.get("retained_coverage"), 0.0),
            ]
            for row in slice_rows
        ]
        image = ax.imshow(matrix, aspect="auto", cmap="cividis", vmin=0.0, vmax=max(1.0, max(max(row) for row in matrix)))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Mean risk", "Retained coverage"])
        ax.set_title("Highest retained-slice risks", pad=12)
        for y, row_values in enumerate(matrix):
            for x, value in enumerate(row_values):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color="white" if value > 0.55 else "black")
        fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02, label="Value")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No slice-level selective-risk rows available.", ha="center", va="center")
    _save(fig, heat_stem)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    counts: dict[str, int] = {}
    for row in caveats:
        counts[str(row.get("status", row.get("availability_policy", "unknown")))] = counts.get(str(row.get("status", row.get("availability_policy", "unknown"))), 0) + 1
    labels = list(counts)
    values = list(counts.values())
    ax.barh(range(len(labels)), values, color="#D55E00")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([_display_label(label) for label in labels])
    ax.invert_yaxis()
    ax.set_title("Confidence availability and caveats", pad=12)
    ax.set_xlabel("Count")
    _save(fig, caveat_stem)
    plt.close(fig)
    return {
        "selective_risk_curves_cross_dataset": curve_stem.with_suffix(".png"),
        "retained_coverage_by_slice_heatmap": heat_stem.with_suffix(".png"),
        "claim_support_caveat_matrix": caveat_stem.with_suffix(".png"),
    }


def build_selective_risk_audit(registry_path: Path, output_dir: Path | None = None) -> dict[str, Path]:
    registry = _load_yaml(registry_path)
    output = ensure_dir(output_dir or Path(registry.get("defaults", {}).get("output_selective", "outputs/selective_risk_audit_v1")))
    reports = ensure_dir(output / "reports")
    ensure_dir(output / "figures")
    coverages = [float(value) for value in registry.get("defaults", {}).get("coverages", [0.7, 0.8, 0.9])]

    availability = _availability_rows(registry)
    confidence_registry = _confidence_registry(registry)
    summary: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    high_conf: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []

    for exp in registry.get("experiments", []):
        selective = exp.get("selective_risk", {}) or {}
        exp_id = str(exp.get("experiment_id", ""))
        dataset = str(exp.get("dataset", ""))
        axis = str(exp.get("deployment_axis", ""))
        source_summary = _discover_existing_csv(
            selective.get("source_summary_candidates"),
            filename="selective_risk_comparison.csv",
            match_tokens=["reben_croma_sensor_mode_audit_croma_comparison"],
        )
        if source_summary is not None:
            existing = _read_existing_selective_summary(exp, source_summary)
            summary.extend(existing)
            curve.extend(existing)
            retained.extend([row for row in existing if str(row.get("slice_variable", "")) not in {"all", ""}])
            high_conf.extend([row for row in existing if str(row.get("slice_variable", "")) not in {"all", ""}])
            if existing:
                continue
        for run_id, spec in (selective.get("source_run_summary_candidates") or {}).items():
            paths = spec.get("paths", []) if isinstance(spec, Mapping) else spec
            mode = str(spec.get("sensor_mode", "")) if isinstance(spec, Mapping) else ""
            table = _discover_existing_csv(paths, filename="selective_risk_summary.csv", match_tokens=[str(run_id)])
            if table is None:
                unavailable = _unavailable_row(
                    experiment_id=exp_id,
                    dataset=dataset,
                    deployment_axis=axis,
                    run_id=str(run_id),
                    reason="no existing per-run selective_risk_summary.csv found at registry candidates",
                )
                summary.append(unavailable)
                curve.append(unavailable)
                continue
            existing = _read_existing_selective_summary(exp, table, run_id=str(run_id), sensor_mode=mode)
            summary.extend(existing)
            curve.extend(existing)
            retained.extend([row for row in existing if str(row.get("slice_variable", "")) not in {"all", "", "unavailable"}])
            high_conf.extend([row for row in existing if str(row.get("slice_variable", "")) not in {"all", "", "unavailable"}])
        for run_id, paths in (selective.get("prediction_table_candidates") or {}).items():
            table = _first_existing(paths)
            if table is None:
                unavailable = _unavailable_row(
                    experiment_id=exp_id,
                    dataset=dataset,
                    deployment_axis=axis,
                    run_id=str(run_id),
                    reason="no existing prediction/audit table found at registry candidates",
                )
                summary.append(unavailable)
                curve.append(unavailable)
                continue
            rows = read_csv_rows(table)
            s, r, h, c = _compute_selective_from_prediction_table(
                rows,
                experiment_id=str(exp.get("experiment_id", "")),
                dataset=str(exp.get("dataset", "")),
                deployment_axis=str(exp.get("deployment_axis", "")),
                run_id=str(run_id),
                risk_column=str(selective.get("risk_column", "risk")),
                confidence_column=str(selective.get("confidence_column", "confidence")),
                slice_columns=selective.get("slice_columns", []) or [],
                coverages=coverages,
            )
            summary.extend(s)
            retained.extend(r)
            high_conf.extend(h)
            curve.extend(c)
            if not s:
                unavailable = _unavailable_row(
                    experiment_id=exp_id,
                    dataset=dataset,
                    deployment_axis=axis,
                    run_id=str(run_id),
                    reason="prediction/audit table exists but confidence/risk columns were not usable",
                    source_path=str(table),
                )
                summary.append(unavailable)
                curve.append(unavailable)

    caveats = []
    for row in availability:
        if str(row.get("status", "")).startswith("unavailable") or str(row.get("status", "")) == "unavailable":
            caveats.append({**row, "caveat": "Selective risk unavailable; no confidence/probability/logit table is used or fabricated."})
        else:
            caveats.append({**row, "caveat": "Selective risk uses existing saved confidence/probability outputs only."})
    if not summary:
        summary = [
            _unavailable_row(
                experiment_id=str(row.get("experiment_id", "")),
                dataset=str(row.get("dataset", "")),
                deployment_axis=str(row.get("deployment_axis", "")),
                run_id=str(row.get("run_id", "")),
                reason=str(row.get("reason", "no existing confidence/probability/logit output found")),
                source_path=str(row.get("source_path", "")),
            )
            for row in availability
        ] or [
            _unavailable_row(
                experiment_id="",
                dataset="",
                deployment_axis="",
                run_id="",
                reason="no selective risk registry entries were available",
            )
        ]
    selective_bwer = _simple_bwer(retained)
    if not retained:
        retained = [
            {
                "experiment_id": row.get("experiment_id", ""),
                "dataset": row.get("dataset", ""),
                "deployment_axis": row.get("deployment_axis", ""),
                "run_id": row.get("run_id", ""),
                "coverage_target": "",
                "slice_variable": "unavailable",
                "slice_value": "unavailable",
                "retained_count": "",
                "total_count": "",
                "retained_coverage": "",
                "mean_risk": "",
                "status": "unavailable",
                "reason": row.get("reason", "no retained slice rows available"),
            }
            for row in summary
            if str(row.get("status", "")) == "unavailable"
        ] or [
            {
                "experiment_id": "",
                "dataset": "",
                "deployment_axis": "",
                "run_id": "",
                "coverage_target": "",
                "slice_variable": "unavailable",
                "slice_value": "unavailable",
                "retained_count": "",
                "total_count": "",
                "retained_coverage": "",
                "mean_risk": "",
                "status": "unavailable",
                "reason": "no selective risk rows available",
            }
        ]
    if not high_conf:
        high_conf = [{**dict(row), "high_confidence_error": row.get("mean_risk", "")} for row in retained]
    if not selective_bwer:
        selective_bwer = [
            {
                "experiment_id": row.get("experiment_id", ""),
                "run_id": row.get("run_id", ""),
                "coverage_target": row.get("coverage_target", ""),
                "slice_variable": row.get("slice_variable", "unavailable"),
                "mean_slice_risk": "",
                "worst_slice_risk": "",
                "selective_bwer_proxy": "",
                "status": "unavailable",
                "note": row.get("reason", "Selective BWER unavailable because no retained slice support rows were available."),
            }
            for row in summary
            if str(row.get("status", "")) == "unavailable"
        ] or [
            {
                "experiment_id": "",
                "run_id": "",
                "coverage_target": "",
                "slice_variable": "unavailable",
                "mean_slice_risk": "",
                "worst_slice_risk": "",
                "selective_bwer_proxy": "",
                "status": "unavailable",
                "note": "Selective BWER unavailable because no retained slice support rows were available.",
            }
        ]

    artifacts = {
        "confidence_availability_audit": output / "confidence_availability_audit.csv",
        "confidence_definition_registry": output / "confidence_definition_registry.csv",
        "selective_risk_summary_cross_dataset": output / "selective_risk_summary_cross_dataset.csv",
        "retained_coverage_by_slice": output / "retained_coverage_by_slice.csv",
        "high_confidence_error_by_slice": output / "high_confidence_error_by_slice.csv",
        "selective_bwer_summary": output / "selective_bwer_summary.csv",
        "risk_coverage_curve_points": output / "risk_coverage_curve_points.csv",
        "selective_risk_caveats": output / "selective_risk_caveats.csv",
        "selective_risk_report": output / "selective_risk_report.md",
        "selective_risk_audit_report": reports / "selective_risk_audit_report.md",
    }
    write_csv(artifacts["confidence_availability_audit"], availability)
    write_csv(artifacts["confidence_definition_registry"], confidence_registry)
    write_csv(artifacts["selective_risk_summary_cross_dataset"], summary)
    write_csv(artifacts["retained_coverage_by_slice"], retained)
    write_csv(artifacts["high_confidence_error_by_slice"], high_conf)
    write_csv(artifacts["selective_bwer_summary"], selective_bwer)
    write_csv(artifacts["risk_coverage_curve_points"], curve)
    write_csv(artifacts["selective_risk_caveats"], caveats)
    report = [
        "# Selective Risk Audit",
        "",
        "This report uses only existing saved confidence/probability/logit outputs.",
        "Unavailable runs are recorded explicitly; no confidence values are fabricated.",
        "",
        "Selective-risk curves should be interpreted within task and metric family. They do not establish direct numerical equivalence between segmentation IoU-risk, classification error, and multi-label BCE risk.",
        "",
        "Figure captions:",
        "- `selective_risk_curves_cross_dataset` reports overall retained-risk curves only; it should be read as a coverage-risk summary, not as a slice-specific trajectory.",
        "- `retained_coverage_by_slice_heatmap` shows the highest retained-slice risks among available slice rows, limited to a readable subset.",
        "- `claim_support_caveat_matrix` summarizes confidence availability and caveat burden.",
    ]
    artifacts["selective_risk_report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    artifacts["selective_risk_audit_report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    figure_paths = _figures(output, summary, retained, caveats)
    artifacts.update({f"figure_{key}": path for key, path in figure_paths.items()})
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Selective Risk Audit v1 from existing confidence outputs.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = build_selective_risk_audit(args.registry, args.output_dir)
    print(f"[selective] output_dir={args.output_dir or 'registry default'}")
    for name, path in artifacts.items():
        print(f"[selective] {name}: {path}")


if __name__ == "__main__":
    main()
