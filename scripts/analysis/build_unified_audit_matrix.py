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


def _path_tokens(path: Path, extra_tokens: Sequence[str] | None = None) -> list[str]:
    tokens = [str(token) for token in extra_tokens or [] if str(token)]
    for part in path.parts:
        if "reben_croma_sensor_mode_audit" in part:
            tokens.append(part)
        if "sen1floods11_closure" in part:
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
    if exact is not None and exact.is_file():
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
                required_dir_tokens = [
                    token
                    for token in tokens
                    if "reben_croma_sensor_mode_audit" in token or "sen1floods11_closure" in token
                ]
                if required_dir_tokens and not all(token in text for token in required_dir_tokens):
                    continue
                if not required_dir_tokens and tokens and not any(token in text for token in tokens):
                    continue
                candidates.append(match)
    if not candidates:
        return None
    return sorted(set(candidates), key=lambda item: (len(item.parts), str(item)))[0]


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


def _normalized_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _sen1_run_alias(value: Any) -> str:
    key = _normalized_key(value)
    if not key:
        return ""
    if "resnet34" in key or "albunet" in key:
        return "s2_resnet34_unet"
    if "prithvi" in key:
        return "prithvi_tl"
    if "spectral" in key or "mndwi" in key:
        return "spectral_mndwi"
    if "vanilla" in key or key.startswith("unet") or key in {"unet", "vanillaunet"}:
        return "vanilla_unet"
    return key


def _is_empty_balance(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "nan", "none", "null"}


def _display_label(value: Any) -> str:
    text = str(value or "").strip()
    labels = {
        "prithvi_tl": "Prithvi TL",
        "vanilla_unet": "Vanilla U-Net",
        "spectral_mndwi": "MNDWI rule",
        "s2_resnet34_unet": "S2 ResNet34 U-Net",
        "resnet50_13band": "ResNet-50",
        "dofa_scaled10000": "DOFA scaled",
        "croma_s1": "CROMA S1",
        "croma_s2": "CROMA S2",
        "croma_s1_plus_s2": "CROMA S1+S2",
        "S1": "S1",
        "S2": "S2",
        "S1+S2": "S1+S2",
        "event_disaster": "Event/disaster",
        "geography_location": "Geography/location",
        "sensor_modality": "Sensor/modality",
        "aggregate_iou": "Aggregate IoU",
        "aggregate_score": "Aggregate score",
        "macro_ap": "Macro-AP",
        "iou_risk": "IoU risk",
        "classification_error": "Classification error",
        "bce_risk": "BCE risk",
    }
    return labels.get(text, text.replace("_", " ").strip().title() if text else "")


def _shorten(text: Any, max_chars: int = 46) -> str:
    value = str(text or "")
    return value if len(value) <= max_chars else value[: max_chars - 1].rstrip() + "…"


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


def _merge_sen1_closure_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    average_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source_row in summary_rows:
        alias = _sen1_run_alias(source_row.get("run_name", source_row.get("model", "")))
        if not alias:
            continue
        merged.setdefault(alias, {}).update(dict(source_row))
    for source_row in average_rows:
        alias = _sen1_run_alias(source_row.get("run_name", source_row.get("model_label", "")))
        if not alias:
            continue
        row = merged.setdefault(alias, {})
        for target, names in {
            "aggregate_iou": ["aggregate_iou", "average_score"],
            "raw_bwer_event_id": ["raw_bwer_event_id", "raw_bwer"],
            "standardised_bwer_event_id_flood_extent_bin": [
                "standardised_bwer_event_id_flood_extent_bin",
                "standardised_bwer",
            ],
        }.items():
            value = _row_value(source_row, names, "")
            if value not in {"", None}:
                row[target] = value
    return merged


def _enrich_sen1floods11_rows(registry: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sen1_exp = next((exp for exp in registry.get("experiments", []) if exp.get("experiment_id") == "sen1floods11_closure"), None)
    if not sen1_exp:
        return rows
    output_candidates = sen1_exp.get("output_dir_candidates") or []
    summary_path = _discover_existing_csv(
        [*(Path(value) / "closure_comparison_summary.csv" for value in output_candidates), *output_candidates],
        filename="closure_comparison_summary.csv",
        match_tokens=["sen1floods11_closure"],
    ) or _discover_existing_csv(
        [*(Path(value) / "comparison_summary.csv" for value in output_candidates), *output_candidates],
        filename="comparison_summary.csv",
        match_tokens=["sen1floods11_closure"],
    )
    average_path = _discover_existing_csv(
        [*(Path(value) / "closure_average_vs_bwer.csv" for value in output_candidates), *output_candidates],
        filename="closure_average_vs_bwer.csv",
        match_tokens=["sen1floods11_closure"],
    ) or _discover_existing_csv(
        [*(Path(value) / "average_vs_bwer.csv" for value in output_candidates), *output_candidates],
        filename="average_vs_bwer.csv",
        match_tokens=["sen1floods11_closure"],
    )
    if summary_path is None and average_path is None:
        return rows
    summary_rows = read_csv_rows(summary_path) if summary_path and summary_path.exists() else []
    average_rows = read_csv_rows(average_path) if average_path and average_path.exists() else []
    by_run = _merge_sen1_closure_rows(summary_rows, average_rows)
    source_parts = [str(path) for path in (summary_path, average_path) if path]
    source_note = "file_read:" + ";".join(source_parts)
    for row in rows:
        if row.get("experiment_id") != "sen1floods11_closure":
            continue
        source = by_run.get(_sen1_run_alias(row.get("run_id", "")))
        if not source:
            row["metric_availability_note"] = "No matching Sen1Floods11 closure CSV row found for this run."
            continue
        aggregate_iou = _row_value(source, ["aggregate_iou", "average_score"], "")
        raw_bwer = _row_value(source, ["raw_bwer_event_id", "raw_bwer"], "")
        standardised_bwer = _row_value(
            source,
            ["standardised_bwer_event_id_flood_extent_bin", "standardised_bwer"],
            "",
        )
        aggregate_risk = _metric_score_to_risk(aggregate_iou, "iou_risk")
        row["aggregate_score"] = aggregate_iou or row.get("aggregate_score", "")
        row["aggregate_risk"] = aggregate_risk if not math.isnan(aggregate_risk) else row.get("aggregate_risk", "")
        row["aggregate_dice"] = _row_value(source, ["aggregate_dice"], row.get("aggregate_dice", ""))
        row["raw_bwer"] = raw_bwer or row.get("raw_bwer", "")
        row["standardised_bwer"] = standardised_bwer or row.get("standardised_bwer", "")
        row["worst_slice"] = _row_value(source, ["worst_event", "worst_slice"], row.get("worst_slice", ""))
        row["best_slice"] = _row_value(source, ["best_event", "best_slice"], row.get("best_slice", ""))
        row["tail_slices"] = _row_value(source, ["tail_events", "tail_slices"], row.get("tail_slices", ""))
        row["data_source"] = source_note
        missing = [
            name
            for name, value in (
                ("aggregate_iou", aggregate_iou),
                ("raw_bwer_event_id", raw_bwer),
                ("standardised_bwer_event_id_flood_extent_bin", standardised_bwer),
            )
            if value in {"", None}
        ]
        row["metric_availability_note"] = "" if not missing else "Missing in Sen1Floods11 closure CSV: " + "; ".join(missing)
    return rows


def _enrich_rows_from_available_outputs(registry: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use completed output CSVs when present, while keeping registry records as fallback."""
    rows = _enrich_sen1floods11_rows(registry, rows)
    reben_exp = None
    for exp in registry.get("experiments", []):
        if exp.get("experiment_id") == "reben_croma_sensor_mode":
            reben_exp = exp
            break
    if not reben_exp:
        return rows
    output_candidates = reben_exp.get("output_dir_candidates") or []
    aggregate_path = _discover_existing_csv(
        output_candidates,
        filename="aggregate_sensor_mode_comparison.csv",
        match_tokens=["reben_croma_sensor_mode_audit_croma_comparison"],
    )
    if aggregate_path is None or not aggregate_path.exists():
        return rows
    comparison_dir = aggregate_path.parent
    bce_path = _discover_existing_csv(
        [comparison_dir / "bce_bwer_sensor_mode_comparison.csv", *output_candidates],
        filename="bce_bwer_sensor_mode_comparison.csv",
        match_tokens=["reben_croma_sensor_mode_audit_croma_comparison"],
    )
    binary_path = _discover_existing_csv(
        [comparison_dir / "binary_error_bwer_sensor_mode_comparison.csv", *output_candidates],
        filename="binary_error_bwer_sensor_mode_comparison.csv",
        match_tokens=["reben_croma_sensor_mode_audit_croma_comparison"],
    )
    aggregate_rows = read_csv_rows(aggregate_path)
    bwer_rows = []
    if bce_path and bce_path.exists():
        bwer_rows.extend(read_csv_rows(bce_path))
    if binary_path and binary_path.exists():
        bwer_rows.extend(read_csv_rows(binary_path))
    by_mode = {_sensor_mode_alias(row.get("sensor_mode", row.get("run_name", ""))): row for row in aggregate_rows}
    bwer_by_mode = {
        _sensor_mode_alias(row.get("sensor_mode", row.get("run_name", ""))): row
        for row in bwer_rows
        if str(_row_value(row, ["risk_name", "risk_metric", "risk_column"], "")).strip() in {"risk_bce", "bce", "labelwise_bce"}
        and str(_row_value(row, ["slice_variable", "slice", "slice_name"], "")).strip() == "country"
        and _is_empty_balance(_row_value(row, ["balance_variable", "balance", "standardised_balance"], ""))
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


def _save(fig: Any, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.25)
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")


def _barh(ax: Any, labels: Sequence[str], values: Sequence[float], title: str, xlabel: str) -> None:
    y = list(range(len(labels)))
    ax.barh(y, values, color="#0072B2")
    ax.set_yticks(y)
    ax.set_yticklabels([_shorten(label, 62) for label in labels])
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
        ("Aggregate performance", 0.10, 0.55),
        ("Slice-tail risk", 0.36, 0.55),
        ("Support and caveats", 0.64, 0.55),
        ("Guarded claims", 0.36, 0.18),
    ]
    for text, x, y in boxes:
        ax.text(x, y, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#F2F2F2", "ec": "#555555"})
    for x0, y0, x1, y1 in [(0.18, 0.55, 0.27, 0.55), (0.45, 0.55, 0.54, 0.55), (0.62, 0.45, 0.45, 0.25), (0.35, 0.45, 0.35, 0.30)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("Audit evidence synthesis workflow", pad=12)
    _save(fig, paths["framework_overview"])
    plt.close(fig)

    # Deployment axis matrix.
    axis_counts: dict[str, int] = {}
    for row in rows:
        axis_counts[str(row.get("deployment_axis", ""))] = axis_counts.get(str(row.get("deployment_axis", "")), 0) + 1
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    _barh(ax, [_display_label(label) for label in axis_counts], list(axis_counts.values()), "Deployment axes represented", "audited run count")
    _save(fig, paths["deployment_axis_matrix"])
    plt.close(fig)

    # Average vs BWER, faceted by experiment to avoid cross-task numeric equivalence.
    valid_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if math.isnan(_float(row.get("aggregate_score"))) or math.isnan(_float(row.get("raw_bwer"))):
            continue
        valid_groups.setdefault(str(row.get("experiment_id", "")), []).append(row)
    n_panels = max(1, len(valid_groups))
    fig, axes = plt.subplots(1, n_panels, figsize=(max(4.4, 4.1 * n_panels), 3.9), squeeze=False)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    if not valid_groups:
        axes[0][0].axis("off")
        axes[0][0].text(0.5, 0.5, "No formal rows with both aggregate score and Raw-BWER.", ha="center", va="center")
    for ax, (exp_id, items) in zip(axes[0], valid_groups.items()):
        for idx, row in enumerate(items):
            agg = _float(row.get("aggregate_score"))
            bwer = _float(row.get("raw_bwer"))
            ax.scatter(agg, bwer, color=palette[idx % len(palette)], s=52, edgecolor="white", linewidth=0.7)
            ax.text(agg, bwer, _display_label(row.get("run_id")), fontsize=7.5, ha="left", va="bottom")
        first = items[0]
        ax.set_title(f"{first.get('dataset')}\n{_display_label(first.get('metric_family'))}", fontsize=10, pad=8)
        ax.set_xlabel(_display_label(first.get("aggregate_metric_name", "aggregate score")))
        ax.set_ylabel("Raw-BWER")
        ax.grid(alpha=0.22)
    for ax in axes[0][len(valid_groups):]:
        ax.axis("off")
    fig.suptitle("Aggregate performance and slice-tail risk", y=1.04, fontsize=12)
    _save(fig, paths["average_vs_bwer_cross_dataset"])
    plt.close(fig)

    # Worst-slice bar plot by run.
    valid_worst = [row for row in rows if not math.isnan(_float(row.get("raw_bwer")))]
    valid_worst = sorted(valid_worst, key=lambda row: _float(row.get("raw_bwer")), reverse=True)
    fig, ax = plt.subplots(figsize=(8.2, max(3.1, 0.42 * len(valid_worst))))
    if valid_worst:
        labels = [f"{row.get('dataset')} | {_display_label(row.get('run_id'))}" for row in valid_worst]
        values = [_float(row.get("raw_bwer"), 0.0) for row in valid_worst]
        _barh(ax, labels, values, "Slice-tail risk across audited runs", "Raw-BWER")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid Raw-BWER rows available.", ha="center", va="center")
    _save(fig, paths["worst_slice_barplot_by_run"])
    plt.close(fig)

    # reBEN sensor mode summary.
    reben = [row for row in rows if row.get("experiment_id") == "reben_croma_sensor_mode"]
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.8), squeeze=False)
    metrics = [
        ("aggregate_score", "macro-AP", "higher is better"),
        ("mean_bce_risk", "mean BCE risk", "lower is better"),
        ("raw_bwer", "country Raw-BWER", "lower is better"),
        ("standardised_bwer", "country | class BWER", "lower is better"),
    ]
    labels = [_display_label(row.get("sensor_mode") or row.get("run_id")) for row in reben]
    for ax, (column, title, subtitle) in zip(axes.ravel(), metrics):
        values = [_float(row.get(column)) for row in reben]
        clean_values = [0.0 if math.isnan(value) else value for value in values]
        if all(math.isnan(value) for value in values):
            ax.axis("off")
            ax.text(0.5, 0.52, "BWER rows not available\nin current inputs", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title}\n{subtitle}", fontsize=10, pad=8)
            continue
        ax.bar(labels, clean_values, color=["#0072B2", "#D55E00", "#009E73"][: len(labels)])
        for idx, value in enumerate(values):
            label = "NA" if math.isnan(value) else f"{value:.3f}"
            ax.text(idx, clean_values[idx] + (max(clean_values or [0.0]) * 0.025 + 0.005), label, ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, pad=8)
        ax.set_ylim(0, max(clean_values + [0.01]) * 1.22)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Sensor-mode comparison on BigEarthNet v2 / reBEN", y=1.04, fontsize=12)
    _save(fig, paths["reben_sensor_mode_summary"])
    plt.close(fig)

    # Claim support matrix.
    support_order = ["supported", "supported_croma_only", "supported_case_study", "supported_protocol_aware", "supported_by_sanity"]
    counts = {key: 0 for key in support_order}
    for row in claims:
        key = str(row.get("support", ""))
        counts[key] = counts.get(key, 0) + 1
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _barh(ax, [_display_label(label) for label in counts], list(counts.values()), "Claim support and caveat burden", "claim count")
    _save(fig, paths["claim_support_caveat_matrix"])
    plt.close(fig)
    return {key: value.with_suffix(".png") for key, value in paths.items()}


def _write_reports(output: Path, registry: Mapping[str, Any], run_rows: Sequence[Mapping[str, Any]]) -> None:
    reports = ensure_dir(output / "reports")
    matrix_report = [
        "# Unified Audit Matrix Report",
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
        "- `framework_overview`: conceptual map from aggregate performance to slice-tail risk, support diagnostics, caveats, and guarded claims. It is a workflow figure, not empirical evidence.",
        "- `deployment_axis_matrix`: coverage of the three deployment axes represented in the synthesis: event/disaster, geography/location, and sensor/modality.",
        "- `average_vs_bwer_cross_dataset`: faceted comparison of aggregate score and Raw-BWER within each task family. The facets are intentionally separated because IoU risk, classification error, and BCE risk are not numerically interchangeable.",
        "- `worst_slice_barplot_by_run`: Raw-BWER across audited runs. It summarizes tail-risk pressure but should be read together with support diagnostics and caveats.",
        "- `reben_sensor_mode_summary`: CROMA sensor-mode comparison across macro-AP, mean BCE risk, country Raw-BWER, and country | class BWER. The blocked supervised reference is not included.",
        "- `selective_risk_curves_cross_dataset`: overall retained-risk curves only; slice-level points are not connected as trajectories.",
        "- `retained_coverage_by_slice_heatmap`: top retained-slice risks and coverage values, limited to a readable subset of slices.",
        "- `claim_support_caveat_matrix`: count of claims by support category, intended as a caveat burden summary rather than a result ranking.",
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
        "# Unified Audit Matrix",
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
