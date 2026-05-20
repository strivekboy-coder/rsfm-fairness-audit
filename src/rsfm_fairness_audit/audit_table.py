from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.io import read_csv_rows, write_csv


class AuditTableError(RuntimeError):
    """Raised when an audit table cannot be built or validated."""


ALLOWED_ADAPTATION_PROTOCOLS = {
    "",
    "frozen_probe",
    "frozen_encoder_lightweight_head",
    "task_adapted_decoder",
    "full_finetune",
    "supervised_baseline",
    "diagnostic_spectral_rule",
}

ALLOWED_SPLIT_PROTOCOLS = {
    "",
    "standard_split",
    "random_chip_split",
    "event_held_out",
    "leave_one_event_out",
    "spatial_split",
}


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or str(value).lower() in {"nan", "none", "null"}


def _coalesce(row: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in row and not _is_missing(row.get(key)):
            return row.get(key)
    return default


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AuditTableError(f"Expected numeric value, got {value!r}.") from exc


def _risk_from_score(score: Any) -> float | str:
    if _is_missing(score):
        return ""
    return 1.0 - _as_float(score)


def _metric_from_counts(row: Mapping[str, Any], metric: str = "iou") -> float | str:
    if not all(key in row and not _is_missing(row.get(key)) for key in ["TP", "FP", "FN"]):
        return ""
    tp = _as_float(row.get("TP"))
    fp = _as_float(row.get("FP"))
    fn = _as_float(row.get("FN"))
    if metric.lower() in {"dice", "f1"}:
        denom = (2.0 * tp) + fp + fn
        return 1.0 if denom == 0 else (2.0 * tp) / denom
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def _read_optional(path: str | Path | None) -> list[dict[str, str]]:
    return read_csv_rows(path) if path else []


def _index_by_sample_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("sample_id", row.get("unit_id", index))): row for index, row in enumerate(rows)}


def infer_task_type(rows: Sequence[Mapping[str, Any]], explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if rows and ("water_iou" in rows[0] or "mean_water_iou" in rows[0]):
        return "segmentation"
    if rows and ("prediction" in rows[0] or "y_pred" in rows[0]):
        return "classification"
    return "tabular_score"


def build_audit_table_from_predictions(
    predictions_path: str | Path,
    metadata_path: str | Path | None = None,
    dataset: str = "",
    model: str = "",
    task: str = "",
    split: str = "all",
) -> list[dict[str, Any]]:
    predictions = read_csv_rows(predictions_path)
    metadata = _index_by_sample_id(_read_optional(metadata_path))
    rows: list[dict[str, Any]] = []
    for index, pred in enumerate(predictions):
        sample_id = str(_coalesce(pred, "sample_id", "unit_id", default=index))
        meta = metadata.get(sample_id, {})
        label = _coalesce(pred, "label", "y_true", default=meta.get("label", ""))
        prediction = _coalesce(pred, "prediction", "y_pred", default="")
        score = _coalesce(pred, "score", "correct", default="")
        if _is_missing(score) and not _is_missing(label) and not _is_missing(prediction):
            score = float(str(label) == str(prediction))
        row: dict[str, Any] = {
            "dataset": dataset or _coalesce(pred, "dataset", default=_coalesce(meta, "dataset", default="")),
            "model": model or _coalesce(pred, "model", default=_coalesce(meta, "model", default="")),
            "task": task or _coalesce(pred, "task", default=_coalesce(meta, "task", default=infer_task_type([pred]))),
            "split": _coalesce(pred, "split", default=_coalesce(meta, "split", default=split)),
            "unit_id": sample_id,
            "sample_id": sample_id,
            "input_mode": _coalesce(pred, "input_mode", default=_coalesce(meta, "input_mode", default="")),
            "adaptation_protocol": _coalesce(pred, "adaptation_protocol", default=_coalesce(meta, "adaptation_protocol", default="")),
            "training_budget": _coalesce(pred, "training_budget", "training_setup", default=_coalesce(meta, "training_budget", "training_setup", default="")),
            "split_protocol": _coalesce(pred, "split_protocol", default=_coalesce(meta, "split_protocol", default="")),
            "label": label,
            "y_true": label,
            "y_pred": prediction,
            "score": score,
            "risk": _risk_from_score(score),
            "class_label": _coalesce(pred, "class_label", default=label),
            "aggregation_level": "sample",
        }
        for source in [meta, pred]:
            for key, value in source.items():
                if key not in row:
                    row[key] = value
        rows.append(row)
    validate_audit_table(rows)
    return rows


def build_audit_table_from_segmentation_metrics(
    metrics_path: str | Path,
    metadata_path: str | Path | None = None,
    dataset: str = "",
    model: str = "",
    task: str = "segmentation",
    split: str = "all",
    score_column: str = "water_iou",
) -> list[dict[str, Any]]:
    metrics = read_csv_rows(metrics_path)
    metadata = _index_by_sample_id(_read_optional(metadata_path))
    rows: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics):
        sample_id = str(_coalesce(metric, "sample_id", "unit_id", default=index))
        meta = metadata.get(sample_id, {})
        score = _coalesce(metric, score_column, "score", "micro_iou", "iou", "water_iou", "mean_water_iou", default="")
        if _is_missing(score):
            score = _metric_from_counts(metric, "iou")
        aggregation_level = _coalesce(metric, "aggregation_level", default="sample")
        unit_id = _coalesce(metric, "unit_id", default="")
        if _is_missing(unit_id):
            unit_id = _coalesce(metric, "event_id", "event", default=sample_id) if str(aggregation_level) in {"event", "slice"} else sample_id
        row: dict[str, Any] = {
            "dataset": dataset or _coalesce(metric, "dataset", default=_coalesce(meta, "dataset", default="")),
            "model": model or _coalesce(metric, "model", default=_coalesce(meta, "model", default="")),
            "task": task or _coalesce(metric, "task", default=_coalesce(meta, "task", default="segmentation")),
            "split": _coalesce(metric, "split", default=_coalesce(meta, "split", default=split)),
            "unit_id": unit_id,
            "sample_id": sample_id,
            "input_mode": _coalesce(metric, "input_mode", default=_coalesce(meta, "input_mode", default="")),
            "adaptation_protocol": _coalesce(metric, "adaptation_protocol", default=_coalesce(meta, "adaptation_protocol", default="")),
            "training_budget": _coalesce(metric, "training_budget", "training_setup", default=_coalesce(meta, "training_budget", "training_setup", default="")),
            "split_protocol": _coalesce(metric, "split_protocol", default=_coalesce(meta, "split_protocol", default="")),
            "score": score,
            "risk": _risk_from_score(score),
            "class_label": _coalesce(metric, "class_label", default="water"),
            "aggregation_level": aggregation_level,
        }
        for source in [meta, metric]:
            for key, value in source.items():
                if key not in row:
                    row[key] = value
        rows.append(row)
    validate_audit_table(rows)
    return rows


def build_audit_table_from_classwise_metrics(
    metrics_path: str | Path,
    dataset: str = "",
    model: str = "",
    task: str = "classification",
    score_column: str = "f1",
) -> list[dict[str, Any]]:
    rows = []
    for index, metric in enumerate(read_csv_rows(metrics_path)):
        score = _coalesce(metric, score_column, "score", default="")
        class_label = _coalesce(metric, "class_label", "class_id", default=index)
        rows.append(
            {
                "dataset": dataset or _coalesce(metric, "dataset", default=""),
                "model": model or _coalesce(metric, "model", default=""),
                "task": task or _coalesce(metric, "task", default="classification"),
                "split": _coalesce(metric, "split", default="all"),
                "unit_id": f"class-{class_label}",
                "input_mode": _coalesce(metric, "input_mode", default=""),
                "adaptation_protocol": _coalesce(metric, "adaptation_protocol", default=""),
                "training_budget": _coalesce(metric, "training_budget", "training_setup", default=""),
                "split_protocol": _coalesce(metric, "split_protocol", default=""),
                "score": score,
                "risk": _risk_from_score(score),
                "class_label": class_label,
                "n_positive": _coalesce(metric, "support", default=""),
                "aggregation_level": "class",
            }
        )
    validate_audit_table(rows)
    return rows


def validate_audit_table(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AuditTableError("Audit table is empty.")
    required_any = [("score",), ("risk",), ("correct",), ("label", "prediction"), ("y_true", "y_pred"), ("TP", "FP", "FN")]
    score_like_columns = ["score", "correct", "water_iou", "iou", "dice", "f1", "accuracy"]
    for row_index, row in enumerate(rows, start=1):
        for key in ["dataset", "model", "task", "split", "unit_id"]:
            if key not in row:
                raise AuditTableError(f"Audit table row {row_index} is missing required column: {key}")
        if not any(all(key in row and not _is_missing(row.get(key)) for key in option) for option in required_any):
            raise AuditTableError(f"Audit table row {row_index} requires score/risk or label/prediction columns.")
        for key in score_like_columns:
            if key in row and not _is_missing(row.get(key)):
                try:
                    value = _as_float(row.get(key))
                except AuditTableError as exc:
                    raise AuditTableError(f"Audit table row {row_index} column {key}: {exc}") from exc
                if not math.isfinite(value):
                    raise AuditTableError(f"Audit table row {row_index} column {key} must be finite.")
                if value < 0.0 or value > 1.0:
                    raise AuditTableError(f"Audit table row {row_index} column {key} must be in [0, 1].")
        if "risk" in row and not _is_missing(row.get("risk")):
            try:
                value = _as_float(row.get("risk"))
            except AuditTableError as exc:
                raise AuditTableError(f"Audit table row {row_index} column risk: {exc}") from exc
            if not math.isfinite(value):
                raise AuditTableError(f"Audit table row {row_index} column risk must be finite.")
            if value < 0.0 or value > 1.0:
                raise AuditTableError(f"Audit table row {row_index} column risk must be in [0, 1].")
        protocol = str(row.get("adaptation_protocol", "") or "")
        if protocol not in ALLOWED_ADAPTATION_PROTOCOLS:
            allowed = ", ".join(sorted(value for value in ALLOWED_ADAPTATION_PROTOCOLS if value))
            raise AuditTableError(f"Audit table row {row_index} has unsupported adaptation_protocol={protocol!r}. Allowed: {allowed}.")
        split_protocol = str(row.get("split_protocol", "") or "")
        if split_protocol not in ALLOWED_SPLIT_PROTOCOLS:
            allowed = ", ".join(sorted(value for value in ALLOWED_SPLIT_PROTOCOLS if value))
            raise AuditTableError(f"Audit table row {row_index} has unsupported split_protocol={split_protocol!r}. Allowed: {allowed}.")
        if "confidence" in row and not _is_missing(row.get("confidence")):
            value = _as_float(row.get("confidence"))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise AuditTableError(f"Audit table row {row_index} column confidence must be finite and in [0, 1].")


def write_audit_table(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(path, rows)


def read_audit_table(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    validate_audit_table(rows)
    return rows


def attach_metadata_columns(rows: Sequence[Mapping[str, Any]], metadata_path: str | Path, output_path: str | Path | None = None) -> list[dict[str, Any]]:
    metadata = _index_by_sample_id(read_csv_rows(metadata_path))
    merged = []
    for row in rows:
        item = dict(row)
        meta = metadata.get(str(item.get("sample_id") or item.get("unit_id")), {})
        for key, value in meta.items():
            item.setdefault(key, value)
        merged.append(item)
    if output_path:
        write_csv(output_path, merged)
    return merged
