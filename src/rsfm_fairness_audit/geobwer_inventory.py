from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.bwer_schema import BASE_FORMAL_COLUMNS, TASK_ALTERNATIVES
from rsfm_fairness_audit.io import ensure_dir, write_csv


AUDIT_NAME_HINTS = (
    "audit_table",
    "prediction",
    "segmentation_metrics",
    "event_metrics",
    "conformal",
    "selective",
)


@dataclass(frozen=True)
class InventoryRecord:
    source: str
    artifact_kind: str
    columns: tuple[str, ...]
    missing_formal_columns: tuple[str, ...]
    probability_status: str
    raw_bwer_status: str
    standardised_bwer_status: str
    selective_bwer_status: str
    conformal_bwer_status: str
    inference_status: str
    recommended_action: str


def _kind(name: str, columns: Sequence[str]) -> str:
    lowered = name.lower()
    values = set(columns)
    if "conformal" in lowered or {"covered", "set_size"}.issubset(values):
        return "conformal"
    if "segmentation" in lowered or {"TP", "FP", "FN"}.issubset(values):
        return "segmentation"
    if "selective" in lowered or "confidence" in values:
        return "selective_or_predictions"
    if "audit" in lowered or "risk" in values:
        return "audit_table"
    if "prediction" in lowered or {"label", "prediction"}.issubset(values):
        return "predictions"
    return "other_csv"


def _record(source: str, columns: Sequence[str], protocol: BWERProtocol) -> InventoryRecord:
    values = set(columns)
    missing = tuple(column for column in BASE_FORMAL_COLUMNS if column not in values)
    alternatives = TASK_ALTERNATIVES.get(protocol.task_adapter, ())
    probability_ok = not alternatives or any(all(column in values for column in option) for option in alternatives)
    raw_ok = "risk" in values or "score" in values or {"label", "prediction"}.issubset(values) or {"TP", "FP", "FN"}.issubset(values)
    standard_ok = raw_ok and bool(protocol.balance_variable and protocol.balance_variable in values)
    selective_ok = raw_ok and "confidence" in values
    conformal_ok = {"covered", "set_size"}.issubset(values) or "prediction_set" in values
    cluster_candidates = {protocol.cluster_column, protocol.spatial_block_column, "cluster_id", "spatial_block_id"}
    cluster_ok = any(value and value in values for value in cluster_candidates)
    if not raw_ok:
        action = "not_an_audit_source"
    elif missing:
        action = "upgrade_schema_or_reexport"
    elif not probability_ok:
        action = "reexport_full_probabilities"
    elif protocol.inference_method != "none" and not cluster_ok:
        action = "derive_or_export_independent_clusters"
    else:
        action = "ready_for_geobwer"
    return InventoryRecord(
        source=source,
        artifact_kind=_kind(source, columns),
        columns=tuple(columns),
        missing_formal_columns=missing,
        probability_status="available" if probability_ok else "missing",
        raw_bwer_status="recomputable" if raw_ok else "unavailable",
        standardised_bwer_status="recomputable" if standard_ok else "missing_balance_axis_or_risk",
        selective_bwer_status="recomputable" if selective_ok else "missing_confidence_or_risk",
        conformal_bwer_status="recomputable" if conformal_ok else "missing_prediction_set_outputs",
        inference_status="cluster_ready" if cluster_ok else "missing_cluster_or_spatial_block",
        recommended_action=action,
    )


def _csv_header(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        return tuple(next(reader, ()))


def _zip_csv_headers(path: Path) -> Iterable[tuple[str, tuple[str, ...]]]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            lowered = name.lower()
            if not lowered.endswith(".csv") or not any(hint in lowered for hint in AUDIT_NAME_HINTS):
                continue
            with archive.open(name) as raw:
                first_line = raw.readline().decode("utf-8-sig", errors="replace")
            yield name, tuple(next(csv.reader([first_line]), ()))


def inventory_artifacts(paths: Sequence[str | Path], protocol: BWERProtocol) -> list[InventoryRecord]:
    records: list[InventoryRecord] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw)
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(
                item
                for item in path.rglob("*")
                if item.is_file()
                and (
                    item.suffix.lower() == ".zip"
                    or (item.suffix.lower() == ".csv" and any(hint in item.name.lower() for hint in AUDIT_NAME_HINTS))
                )
            )
        elif path.is_file():
            candidates = [path]
        else:
            continue
        for candidate in candidates:
            if candidate.suffix.lower() == ".zip":
                try:
                    for entry, header in _zip_csv_headers(candidate):
                        source = f"{candidate}::{entry}"
                        if source not in seen:
                            records.append(_record(source, header, protocol))
                            seen.add(source)
                except zipfile.BadZipFile:
                    continue
            elif candidate.suffix.lower() == ".csv":
                source = str(candidate)
                if source not in seen:
                    records.append(_record(source, _csv_header(candidate), protocol))
                    seen.add(source)
    return records


def write_inventory_report(
    records: Sequence[InventoryRecord],
    protocol: BWERProtocol,
    output_dir: str | Path,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    csv_path = output / "geobwer_artifact_inventory.csv"
    json_path = output / "geobwer_artifact_inventory.json"
    report_path = output / "geobwer_artifact_inventory.md"
    rows = []
    for record in records:
        row = dict(record.__dict__)
        row["columns"] = ";".join(record.columns)
        row["missing_formal_columns"] = ";".join(record.missing_formal_columns)
        row["protocol_hash"] = protocol.signature
        rows.append(row)
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps({"protocol": protocol.to_dict(), "protocol_hash": protocol.signature, "records": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for record in records:
        counts[record.recommended_action] = counts.get(record.recommended_action, 0) + 1
    lines = [
        "# GeoBWER Artifact Inventory",
        "",
        f"- Protocol hash: `{protocol.signature}`",
        f"- Candidate artifacts: {len(records)}",
        "",
        "## Actions",
        "",
    ]
    lines.extend(f"- `{action}`: {count}" for action, count in sorted(counts.items()))
    if not records:
        lines.append("- No local candidate CSV/ZIP artifacts were accessible. Run this inventory in Colab against the Drive output roots.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"inventory_csv": csv_path, "inventory_json": json_path, "report": report_path}

