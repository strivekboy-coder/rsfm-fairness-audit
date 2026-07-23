from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence
import zipfile

from rsfm_fairness_audit.io import ensure_dir, write_csv


CSV_HINTS = ("audit", "prediction", "metric", "conformal", "selective", "probab", "label")
ARRAY_HINTS = ("probab", "logit", "score_map", "prob_map", "prediction")


@dataclass(frozen=True)
class ArchiveMember:
    archive: str
    member: str
    uncompressed_bytes: int
    compressed_bytes: int
    artifact_kind: str


@dataclass(frozen=True)
class HeaderRecord:
    source: str
    columns: tuple[str, ...]
    full_probability_columns: int
    has_serialized_probability_vector: bool
    has_split: bool
    has_sample_id: bool
    has_cluster: bool
    has_coordinates: bool


@dataclass(frozen=True)
class ExecutionDecision:
    task: str
    existing_evidence: str
    decision: str
    reason: str
    blocks_formal_run: bool


def _artifact_kind(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".csv"):
        return "csv"
    if lowered.endswith(".npz"):
        return "npz"
    if lowered.endswith(".npy"):
        return "npy"
    if lowered.endswith((".tif", ".tiff")):
        return "geotiff"
    if lowered.endswith(".json"):
        return "json"
    if lowered.endswith((".pt", ".pth", ".ckpt")):
        return "checkpoint"
    return "other"


def _header_record(source: str, columns: Sequence[str]) -> HeaderRecord:
    normalized = tuple(str(column).strip() for column in columns)
    lower = {column.lower() for column in normalized}
    probability_columns = sum(column.startswith("prob_") for column in lower)
    vector_fields = {"probabilities", "probability_vector", "prob_vector", "class_probabilities", "probs"}
    return HeaderRecord(
        source=source,
        columns=normalized,
        full_probability_columns=probability_columns,
        has_serialized_probability_vector=bool(lower & vector_fields),
        has_split=bool(lower & {"split", "split_role", "partition"}),
        has_sample_id=bool(lower & {"sample_id", "chip_id", "image_id", "independent_unit_id"}),
        has_cluster=bool(lower & {"site_id", "location_id", "event_id", "spatial_block_id", "cluster_id"}),
        has_coordinates=bool(lower & {"latitude", "lat"}) and bool(lower & {"longitude", "lon", "lng"}),
    )


def _read_csv_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(next(csv.reader(handle), ()))


def inspect_drive_artifacts(
    roots: Sequence[str | Path],
) -> tuple[list[ArchiveMember], list[HeaderRecord], list[str]]:
    members: list[ArchiveMember] = []
    headers: list[HeaderRecord] = []
    warnings: list[str] = []
    candidates: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            warnings.append(f"missing_root:{root}")
            continue
        if root.is_file():
            candidates.append(root)
            continue
        print(f"[drive-preflight] scanning root={root}")
        for index, path in enumerate(root.rglob("*"), start=1):
            if path.is_file() and path.suffix.lower() in {".zip", ".csv", ".npz", ".npy"}:
                candidates.append(path)
            if index % 5000 == 0:
                print(f"[drive-preflight] scanned entries={index} root={root}")
    seen: set[str] = set()
    for index, path in enumerate(sorted(set(candidates)), start=1):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        if index % 100 == 0 or index == 1:
            print(f"[drive-preflight] inspecting candidate={index}/{len(candidates)} path={path}")
        if path.suffix.lower() == ".csv":
            if any(hint in path.name.lower() for hint in CSV_HINTS):
                try:
                    headers.append(_header_record(str(path), _read_csv_header(path)))
                except (OSError, UnicodeError, csv.Error) as exc:
                    warnings.append(f"csv_header_error:{path}:{type(exc).__name__}:{exc}")
            continue
        if path.suffix.lower() != ".zip":
            members.append(
                ArchiveMember(str(path.parent), path.name, path.stat().st_size, path.stat().st_size, _artifact_kind(path.name))
            )
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    kind = _artifact_kind(info.filename)
                    lowered = info.filename.lower()
                    if kind in {"csv", "npz", "npy", "geotiff", "json", "checkpoint"}:
                        members.append(
                            ArchiveMember(str(path), info.filename, info.file_size, info.compress_size, kind)
                        )
                    if kind == "csv" and any(hint in lowered for hint in CSV_HINTS):
                        try:
                            with archive.open(info) as raw:
                                first_line = raw.readline().decode("utf-8-sig", errors="replace")
                            headers.append(_header_record(f"{path}::{info.filename}", tuple(next(csv.reader([first_line]), ()))))
                        except (OSError, UnicodeError, csv.Error, RuntimeError) as exc:
                            warnings.append(f"zip_csv_header_error:{path}::{info.filename}:{type(exc).__name__}:{exc}")
        except (OSError, zipfile.BadZipFile) as exc:
            warnings.append(f"zip_error:{path}:{type(exc).__name__}:{exc}")
    return members, headers, warnings


def _sources(headers: Iterable[HeaderRecord], tokens: Sequence[str]) -> list[HeaderRecord]:
    lowered = tuple(token.lower() for token in tokens)
    return [record for record in headers if all(token in record.source.lower() for token in lowered)]


def _member_sources(members: Iterable[ArchiveMember], tokens: Sequence[str]) -> list[ArchiveMember]:
    lowered = tuple(token.lower() for token in tokens)
    return [record for record in members if all(token in f"{record.archive}/{record.member}".lower() for token in lowered)]


def decide_execution(
    members: Sequence[ArchiveMember],
    headers: Sequence[HeaderRecord],
) -> list[ExecutionDecision]:
    decisions: list[ExecutionDecision] = []
    alpha_headers = _sources(headers, ("alphaearth",))
    alpha_ready = [
        record
        for record in alpha_headers
        if record.full_probability_columns >= 2
        and record.has_split
        and record.has_sample_id
        and record.has_coordinates
    ]
    decisions.append(
        ExecutionDecision(
            task="AlphaEarth/WorldCover",
            existing_evidence=";".join(record.source for record in alpha_ready[:3]) or "none",
            decision="postprocess_existing_probabilities" if alpha_ready else "repair_or_reexport_source_table",
            reason=(
                "Existing all-split table contains full class probabilities, split labels, sample IDs and coordinates."
                if alpha_ready
                else "No single inspected table proves the complete no-training contract."
            ),
            blocks_formal_run=not bool(alpha_ready),
        )
    )
    fmow_headers = _sources(headers, ("fmow",))
    fmow_ready = [
        record
        for record in fmow_headers
        if (record.full_probability_columns >= 62 or record.has_serialized_probability_vector)
        and record.has_split
        and record.has_sample_id
        and record.has_cluster
    ]
    decisions.append(
        ExecutionDecision(
            task="fMoW-Sentinel/DOFAv2",
            existing_evidence=";".join(record.source for record in fmow_ready[:3]) or "none",
            decision="reuse_complete_predictions" if fmow_ready else "run_dofav2_inference_and_export_62d",
            reason=(
                "A complete 62-class probability/split/location contract is present."
                if fmow_ready
                else "Old packages do not prove a full 62-dimensional probability matrix with disjoint calibration/test roles."
            ),
            blocks_formal_run=False,
        )
    )
    reben_headers = _sources(headers, ("reben",)) + _sources(headers, ("bigearthnet",))
    reben_ready = [
        record
        for record in reben_headers
        if (record.full_probability_columns >= 19 or record.has_serialized_probability_vector)
        and record.has_split
        and record.has_sample_id
    ]
    reben_label_expanded = [
        record
        for record in reben_headers
        if {"sample_id", "class_label"}.issubset({column.lower() for column in record.columns})
        and bool({"probability", "score", "confidence"} & {column.lower() for column in record.columns})
        and record.has_split
    ]
    reben_arrays = [
        record for record in _member_sources(members, ("reben",)) if record.artifact_kind in {"npz", "npy"} and any(hint in record.member.lower() for hint in ARRAY_HINTS)
    ]
    reben_reusable = bool(reben_ready or reben_label_expanded or reben_arrays)
    decisions.append(
        ExecutionDecision(
            task="reBEN/CROMA historical modes",
            existing_evidence=";".join(
                [record.source for record in (*reben_ready[:2], *reben_label_expanded[:2])]
                + [f"{record.archive}::{record.member}" for record in reben_arrays[:2]]
            ) or "none",
            decision="schema_smoke_then_reuse" if reben_reusable else "rerun_croma_probability_export",
            reason=(
                "Candidate multi-label probabilities exist, but class completeness and calibration/test separation must be schema-checked."
                if reben_reusable
                else "No inspected artifact proves reusable multi-label probabilities."
            ),
            blocks_formal_run=not reben_reusable,
        )
    )
    decisions.append(
        ExecutionDecision(
            task="reBEN/TerraMind S1,S2,S1+S2",
            existing_evidence="none",
            decision="run_new_terramind_campaign",
            reason="This is the pre-registered cross-architecture modality replication, not a post-processing task.",
            blocks_formal_run=False,
        )
    )
    sen1_members = _member_sources(members, ("sen1",))
    sen1_probability = [
        record
        for record in sen1_members
        if record.artifact_kind in {"npz", "npy", "geotiff"}
        and any(hint in record.member.lower() for hint in ARRAY_HINTS)
        and any(split in record.member.lower() for split in ("val", "calibration", "test"))
    ]
    decisions.append(
        ExecutionDecision(
            task="Sen1Floods11 historical/formal models",
            existing_evidence=";".join(f"{record.archive}::{record.member}" for record in sen1_probability[:4]) or "none",
            decision="schema_smoke_then_reuse" if sen1_probability else "rerun_probability_map_export",
            reason=(
                "Candidate validation/test probability maps exist; verify event/split/mask alignment before reuse."
                if sen1_probability
                else "Old summaries alone cannot support conformal/CRC or pixel-level uncertainty auditing."
            ),
            blocks_formal_run=not bool(sen1_probability),
        )
    )
    decisions.append(
        ExecutionDecision(
            task="Sen1Floods11/TerraMind S1,S2,S1+S2",
            existing_evidence="none",
            decision="run_new_terramind_campaign",
            reason="This supplies the missing native-SAR and controlled multimodal anchor panel.",
            blocks_formal_run=False,
        )
    )
    return decisions


def run_drive_preflight(roots: Sequence[str | Path], output_dir: str | Path) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    members, headers, warnings = inspect_drive_artifacts(roots)
    decisions = decide_execution(members, headers)
    members_path = output / "drive_archive_members.csv"
    headers_path = output / "drive_csv_headers.csv"
    decisions_path = output / "execution_decisions.csv"
    summary_path = output / "drive_preflight_summary.json"
    write_csv(members_path, members)
    write_csv(
        headers_path,
        [
            {
                **asdict(record),
                "columns": ";".join(record.columns),
            }
            for record in headers
        ],
    )
    write_csv(decisions_path, decisions)
    summary_path.write_text(
        json.dumps(
            {
                "schema": "geobwer.drive_preflight.v1",
                "roots": [str(Path(root)) for root in roots],
                "archive_member_count": len(members),
                "csv_header_count": len(headers),
                "warnings": warnings,
                "decisions": [asdict(record) for record in decisions],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[drive-preflight] complete members={len(members)} headers={len(headers)} warnings={len(warnings)}")
    return {
        "archive_members": members_path,
        "csv_headers": headers_path,
        "execution_decisions": decisions_path,
        "summary": summary_path,
    }


__all__ = [
    "ArchiveMember",
    "ExecutionDecision",
    "HeaderRecord",
    "decide_execution",
    "inspect_drive_artifacts",
    "run_drive_preflight",
]
