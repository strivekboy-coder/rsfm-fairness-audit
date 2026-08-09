from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REGISTRY_SCHEMA = "geobwer.canonical_evidence_registry.v1"
ALLOWED_STATUSES = {
    "canonical_source",
    "canonical_derived",
    "descriptive_only",
    "revoked",
    "debug_only",
}
USABLE_STATUSES = {"canonical_source", "canonical_derived", "descriptive_only"}


@dataclass(frozen=True)
class EvidenceAsset:
    asset_id: str
    task: str
    role: str
    status: str
    drive_path: str
    drive_url: str
    immutable: bool
    allowed_use: tuple[str, ...]
    completion_artifact: str = ""
    replaced_by: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CanonicalEvidenceRegistry:
    schema: str
    registry_version: str
    assets: tuple[EvidenceAsset, ...]
    signature: str

    def resolve(
        self,
        *,
        task: str,
        role: str,
        include_descriptive: bool = True,
    ) -> EvidenceAsset:
        allowed = {"canonical_source", "canonical_derived"}
        if include_descriptive:
            allowed.add("descriptive_only")
        matches = [asset for asset in self.assets if asset.task == task and asset.role == role and asset.status in allowed]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one usable asset for task={task!r}, role={role!r}; found {len(matches)}."
            )
        return matches[0]

    def revoked(self) -> tuple[EvidenceAsset, ...]:
        return tuple(asset for asset in self.assets if asset.status in {"revoked", "debug_only"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "registry_version": self.registry_version,
            "signature": self.signature,
            "assets": [asset.__dict__ for asset in self.assets],
        }


def _signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_canonical_evidence_registry(path: str | Path) -> CanonicalEvidenceRegistry:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"Expected schema={REGISTRY_SCHEMA!r}.")
    raw_assets: Sequence[Mapping[str, Any]] = payload.get("assets") or ()
    if not raw_assets:
        raise ValueError("Canonical evidence registry contains no assets.")
    assets: list[EvidenceAsset] = []
    ids: set[str] = set()
    for raw in raw_assets:
        asset_id = str(raw.get("asset_id", "")).strip()
        if not asset_id or asset_id in ids:
            raise ValueError(f"Evidence asset IDs must be non-empty and unique: {asset_id!r}.")
        ids.add(asset_id)
        status = str(raw.get("status", "")).strip()
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Unsupported evidence status={status!r} for asset={asset_id}.")
        drive_path = str(raw.get("drive_path", "")).strip()
        drive_url = str(raw.get("drive_url", "")).strip()
        if not drive_path.startswith("/content/drive/MyDrive/rsfm_fairness_audit/"):
            raise ValueError(f"Asset {asset_id} is outside the canonical Drive project root.")
        if not drive_url.startswith("https://drive.google.com/"):
            raise ValueError(f"Asset {asset_id} requires a verified Google Drive URL.")
        immutable = bool(raw.get("immutable", False))
        if status in USABLE_STATUSES and not immutable:
            raise ValueError(f"Usable canonical asset {asset_id} must be immutable.")
        allowed_use = tuple(str(value) for value in (raw.get("allowed_use") or ()))
        if not allowed_use:
            raise ValueError(f"Asset {asset_id} must declare allowed_use.")
        replaced_by = str(raw.get("replaced_by", "")).strip()
        if status == "revoked" and not replaced_by:
            raise ValueError(f"Revoked asset {asset_id} must name replaced_by.")
        assets.append(
            EvidenceAsset(
                asset_id=asset_id,
                task=str(raw.get("task", "")).strip(),
                role=str(raw.get("role", "")).strip(),
                status=status,
                drive_path=drive_path,
                drive_url=drive_url,
                immutable=immutable,
                allowed_use=allowed_use,
                completion_artifact=str(raw.get("completion_artifact", "")).strip(),
                replaced_by=replaced_by,
                notes=str(raw.get("notes", "")).strip(),
            )
        )
    for asset in assets:
        if asset.replaced_by and asset.replaced_by not in ids:
            raise ValueError(f"Asset {asset.asset_id} references unknown replacement={asset.replaced_by!r}.")
    identity_counts: dict[tuple[str, str], int] = {}
    for asset in assets:
        if asset.status in USABLE_STATUSES:
            key = (asset.task, asset.role)
            identity_counts[key] = identity_counts.get(key, 0) + 1
    duplicates = {key: count for key, count in identity_counts.items() if count > 1}
    if duplicates:
        raise ValueError(f"Canonical task/role identities are ambiguous: {duplicates}.")
    canonical_payload = {
        "schema": payload["schema"],
        "registry_version": str(payload.get("registry_version", "")),
        "assets": [asset.__dict__ for asset in assets],
    }
    return CanonicalEvidenceRegistry(
        schema=payload["schema"],
        registry_version=str(payload.get("registry_version", "")),
        assets=tuple(assets),
        signature=_signature(canonical_payload),
    )
