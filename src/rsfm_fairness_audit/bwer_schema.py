from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.bwer_protocol import METRIC_VERSION, Validity


BASE_FORMAL_COLUMNS = (
    "dataset",
    "model",
    "task",
    "split",
    "sample_id",
    "independent_unit_id",
    "risk",
    "split_role",
    "model_signature",
    "dataset_signature",
    "protocol_hash",
    "metric_version",
)


TASK_ALTERNATIVES = {
    "multiclass": (("probabilities_path",), ("probability_vector",)),
    "multilabel": (("probabilities_path",), ("probability_vector",)),
    "segmentation": (("probability_map_path",), ("TP", "FP", "FN", "TN")),
    "conformal": (("prediction_set",), ("set_size", "covered")),
}


@dataclass(frozen=True)
class SchemaValidation:
    validity: Validity
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    columns: tuple[str, ...]
    row_count: int

    @property
    def ok(self) -> bool:
        return self.validity == Validity.VALID


def _missing(value: Any) -> bool:
    return value is None or value == "" or str(value).lower() in {"nan", "none", "null"}


def validate_formal_audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_adapter: str,
    require_spatial_block: bool = False,
    required_cluster_column: str | None = None,
    require_probabilities: bool = True,
    expected_protocol_hash: str | None = None,
    expected_metric_version: str = METRIC_VERSION,
) -> SchemaValidation:
    if not rows:
        return SchemaValidation(Validity.INVALID_PROTOCOL, ("audit table is empty",), (), (), 0)
    columns = tuple(sorted(set().union(*(row.keys() for row in rows))))
    errors: list[str] = []
    warnings: list[str] = []
    for column in BASE_FORMAL_COLUMNS:
        missing_rows = sum(1 for row in rows if column not in row or _missing(row.get(column)))
        if missing_rows:
            errors.append(f"{column}: missing in {missing_rows}/{len(rows)} rows")
    for identifier in ("sample_id", "independent_unit_id"):
        values = [str(row.get(identifier)) for row in rows if not _missing(row.get(identifier))]
        duplicate_count = len(values) - len(set(values))
        if duplicate_count:
            errors.append(
                f"{identifier}: {duplicate_count} duplicate rows; formal tables require one task loss per independent unit"
            )
    if require_spatial_block:
        missing_rows = sum(1 for row in rows if _missing(row.get("spatial_block_id")))
        if missing_rows:
            errors.append(f"spatial_block_id: missing in {missing_rows}/{len(rows)} rows")
    if required_cluster_column:
        missing_rows = sum(1 for row in rows if _missing(row.get(required_cluster_column)))
        if missing_rows:
            errors.append(f"{required_cluster_column}: missing in {missing_rows}/{len(rows)} rows")
    alternatives = TASK_ALTERNATIVES.get(task_adapter, ()) if require_probabilities else ()
    if alternatives:
        invalid_rows = sum(
            1
            for row in rows
            if not any(all(column in row and not _missing(row.get(column)) for column in option) for option in alternatives)
        )
        if invalid_rows:
            errors.append(f"{task_adapter}: {invalid_rows}/{len(rows)} rows do not satisfy one of {alternatives}")
    if task_adapter in {"multiclass", "multilabel"} and require_probabilities:
        missing_hash = sum(1 for row in rows if _missing(row.get("class_mapping_hash")))
        if missing_hash:
            errors.append(f"class_mapping_hash: missing in {missing_hash}/{len(rows)} rows")
        mapping_hashes = {str(row.get("class_mapping_hash")) for row in rows if not _missing(row.get("class_mapping_hash"))}
        if len(mapping_hashes) > 1:
            errors.append(f"class_mapping_hash is inconsistent across rows: {sorted(mapping_hashes)}")
    roles = {str(row.get("split_role", "")) for row in rows}
    forbidden_roles = roles & {"train", "training", "calibration", "calibrate"}
    if forbidden_roles:
        errors.append(f"Formal evaluation table contains non-evaluation split roles: {sorted(forbidden_roles)}")
    if "test" not in roles and "evaluation" not in roles:
        warnings.append("No explicit test/evaluation split_role was found.")
    metric_versions = {str(row.get("metric_version", "")) for row in rows if not _missing(row.get("metric_version"))}
    if metric_versions != {expected_metric_version}:
        errors.append(f"Expected metric_version={expected_metric_version}, found {sorted(metric_versions)}")
    protocol_hashes = {str(row.get("protocol_hash", "")) for row in rows if not _missing(row.get("protocol_hash"))}
    if len(protocol_hashes) > 1:
        errors.append(f"protocol_hash is inconsistent across rows: {sorted(protocol_hashes)}")
    if expected_protocol_hash is not None and protocol_hashes != {expected_protocol_hash}:
        errors.append(f"Expected protocol_hash={expected_protocol_hash}, found {sorted(protocol_hashes)}")
    for signature_column in ("model_signature", "dataset_signature"):
        signatures = {str(row.get(signature_column, "")) for row in rows if not _missing(row.get(signature_column))}
        if len(signatures) > 1:
            errors.append(f"{signature_column} is inconsistent across rows: {sorted(signatures)}")
    if errors:
        if any(error.startswith("independent_unit_id") for error in errors):
            validity = Validity.MISSING_INDEPENDENT_UNIT
        elif forbidden_roles:
            validity = Validity.CALIBRATION_LEAKAGE
        elif alternatives and any(error.startswith(task_adapter) for error in errors):
            validity = Validity.MISSING_PROBABILITY_OUTPUT
        else:
            validity = Validity.INVALID_PROTOCOL
    else:
        validity = Validity.VALID
    return SchemaValidation(validity, tuple(errors), tuple(warnings), columns, len(rows))


def class_mapping_hash(class_names: Sequence[Any]) -> str:
    payload = json.dumps([str(value) for value in class_names], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_signature(values: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(values), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
