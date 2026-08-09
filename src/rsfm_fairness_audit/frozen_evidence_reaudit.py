from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit import __version__
from rsfm_fairness_audit.bwer_protocol import BWERProtocol, CERTIFICATION_VERSION
from rsfm_fairness_audit.evidence_registry import CanonicalEvidenceRegistry
from rsfm_fairness_audit.geobwer import audit_rows
from rsfm_fairness_audit.risk_spec import RiskSpec


REAUDIT_SCHEMA = "geobwer.frozen_evidence_reaudit.v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _task_risk_spec(task: str, protocol: BWERProtocol) -> RiskSpec:
    values: dict[str, dict[str, str]] = {
        "alphaearth": {
            "reference": "ESA_WorldCover_reference_map_agreement",
            "threshold_source": "argmax_frozen_probabilities",
            "task_adapter": "classification",
        },
        "fmow_sentinel": {
            "reference": "fMoW_Sentinel_class_label",
            "threshold_source": "argmax_frozen_probabilities",
            "task_adapter": "classification",
        },
        "reben": {
            "reference": "reBEN_multilabel_reference",
            "threshold_source": "validation_locked_per_label_thresholds",
            "task_adapter": "multilabel",
        },
    }
    selected = values.get(task, {})
    return RiskSpec(
        name=protocol.loss_name,
        lower_bound=0.0,
        upper_bound=1.0,
        unit="independent_deployment_unit",
        aggregation="mean_within_registered_slice",
        reference=selected.get("reference", "frozen_task_reference"),
        ignore_policy="frozen_source_contract",
        threshold_source=selected.get("threshold_source", "frozen_source_contract"),
        task_adapter=protocol.task_adapter,
        metadata=(("task", task),),
    )


def certification_protocol(
    source: BWERProtocol,
    *,
    task: str,
    calibration_signature: str,
    min_clusters_for_inference: int = 75,
) -> BWERProtocol:
    risk_spec = _task_risk_spec(task, source)
    return replace(
        source,
        certification_version=CERTIFICATION_VERSION,
        min_clusters_for_inference=int(min_clusters_for_inference),
        cluster_eligibility_calibration_signature=str(calibration_signature),
        risk_spec=risk_spec,
    )


def reaudit_frozen_table(
    *,
    source_table: str | Path,
    output_dir: str | Path,
    protocol: BWERProtocol,
    group_columns: Sequence[str],
    cluster_column: str,
    registry: CanonicalEvidenceRegistry,
    registry_asset_id: str,
    balance_column: str | None = None,
    required_group_universe: Mapping[str, Sequence[Any]] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Path]:
    source = Path(source_table).resolve()
    output = Path(output_dir).resolve()
    if output == source.parent or source in output.parents:
        raise ValueError("Re-audit output must not overlap the frozen source table.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite re-audit output: {output}")
    assets = {asset.asset_id: asset for asset in registry.assets}
    if registry_asset_id not in assets or assets[registry_asset_id].status not in {
        "canonical_source", "canonical_derived", "descriptive_only"
    }:
        raise ValueError("Re-audit requires one usable canonical registry asset.")
    source_sha_before = sha256_file(source)
    rows = _read_csv(source)
    if not rows:
        raise ValueError("Frozen audit table is empty.")
    missing = sorted((set(group_columns) | {cluster_column, "risk"}) - set(rows[0]))
    if missing:
        raise ValueError(f"Frozen audit table is missing required columns: {missing}")
    old_protocol_hashes = sorted({str(row.get("protocol_hash", "")) for row in rows})
    overlay = [
        {
            **row,
            "protocol_hash": protocol.signature,
            "metric_version": protocol.metric_version,
        }
        for row in rows
    ]
    result = audit_rows(
        overlay,
        group_columns=tuple(group_columns),
        protocol=protocol,
        cluster_column=cluster_column,
        balance_column=balance_column,
        formal=True,
        required_group_universe=required_group_universe,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    output.mkdir(parents=True, exist_ok=False)
    artifacts = result.to_report(output)
    source_sha_after = sha256_file(source)
    if source_sha_after != source_sha_before:
        raise RuntimeError("Frozen source table changed during read-only re-audit.")
    manifest_path = output / "reaudit_manifest.json"
    manifest = {
        "schema": REAUDIT_SCHEMA,
        "package_version": __version__,
        "task": assets[registry_asset_id].task,
        "registry_asset_id": registry_asset_id,
        "registry_signature": registry.signature,
        "source_table": str(source),
        "source_sha256": source_sha_before,
        "source_immutable": True,
        "source_protocol_hashes": old_protocol_hashes,
        "protocol_hash": protocol.signature,
        "metric_version": protocol.metric_version,
        "certification_version": protocol.certification_version,
        "risk_spec_signature": protocol.risk_spec.signature,
        "group_columns": list(group_columns),
        "cluster_column": cluster_column,
        "balance_column": balance_column,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "evidence_status_by_axis": {
            axis.axis: axis.evidence_status().value for axis in result.axes
        },
        "material_passport": {
            "inputs": [str(source)],
            "transform": "in-memory protocol overlay and CPU-only certification 1.2",
            "model_training_or_inference": False,
            "test_used_for_selection": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    hashes = _artifact_hashes(output)
    completion_path = output / "completion_contract.json"
    completion_path.write_text(
        json.dumps(
            {
                "schema": "geobwer.frozen_evidence_reaudit_completion.v1",
                "status": "complete",
                "source_sha256": source_sha_before,
                "protocol_hash": protocol.signature,
                "artifacts": hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {**artifacts, "manifest": manifest_path, "completion": completion_path}
