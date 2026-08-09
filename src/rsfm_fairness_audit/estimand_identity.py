from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.bwer_core import normalize_deployment_weights


ESTIMAND_IDENTITY_VERSION = "geobwer_estimand_identity_v1"


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EstimandIdentity:
    metric_version: str
    certification_version: str
    risk_spec_signature: str
    axis: str
    group_universe_hash: str
    deployment_measure_hash: str
    beta: float
    partition_rule: str
    estimand_scope: str
    standardization_target: str
    missingness_rule: str
    dependence_design: str
    comparison_support_hash: str = ""
    version: str = ESTIMAND_IDENTITY_VERSION

    @property
    def signature(self) -> str:
        return _hash_json(asdict(self))

    def comparable_with(self, other: "EstimandIdentity") -> bool:
        return self.signature == other.signature


def build_estimand_identity(
    protocol: BWERProtocol,
    *,
    axis: str,
    group_universe: Sequence[Any],
    deployment_weights: Mapping[Any, float] | None = None,
    comparison_support_hash: str = "",
) -> EstimandIdentity:
    groups = tuple(sorted(set(str(value) for value in group_universe)))
    if not groups:
        raise ValueError("Estimand identity requires a non-empty fixed group universe.")
    weights = normalize_deployment_weights(groups, deployment_weights)
    return EstimandIdentity(
        metric_version=protocol.metric_version,
        certification_version=protocol.certification_version,
        risk_spec_signature=protocol.risk_spec.signature,
        axis=str(axis),
        group_universe_hash=_hash_json(groups),
        deployment_measure_hash=_hash_json(tuple(sorted(weights.items()))),
        beta=float(protocol.beta),
        partition_rule=protocol.partition_rule,
        estimand_scope=protocol.estimand_scope,
        standardization_target=protocol.standardization_target,
        missingness_rule=protocol.missingness_rule,
        dependence_design=protocol.dependence_design,
        comparison_support_hash=str(comparison_support_hash),
    )
