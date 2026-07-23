from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


METRIC_VERSION = "geobwer_fractional_1.1"


class Validity(str, Enum):
    VALID = "valid"
    DESCRIPTIVE_ONLY = "descriptive_only"
    INSUFFICIENT_SLICES = "insufficient_slices"
    INSUFFICIENT_INDEPENDENT_UNITS = "insufficient_independent_units"
    INSUFFICIENT_TAIL_EFFECTIVE_GROUPS = "insufficient_tail_effective_groups"
    NOT_IDENTIFIED = "not_identified_missing_standardization_cells"
    NOT_IDENTIFIED_SELECTIVE_COVERAGE = "not_identified_zero_selective_coverage"
    NO_COMMON_SUPPORT = "no_common_support"
    MISSING_PROBABILITY_OUTPUT = "missing_probability_output"
    MISSING_INDEPENDENT_UNIT = "missing_independent_unit"
    CALIBRATION_LEAKAGE = "calibration_leakage"
    INVALID_INFERENCE_TARGET = "invalid_inference_target"
    REFERENCE_PRODUCT_NOT_COMPARABLE = "reference_product_not_comparable"
    INFERENCE_NOT_CERTIFIED = "inference_not_certified"
    SPATIAL_BLOCK_NOT_CALIBRATED = "spatial_block_not_calibrated"
    INVALID_PROTOCOL = "invalid_protocol"


@dataclass(frozen=True)
class BWERProtocol:
    """Versioned estimand and inference contract for a GeoBWER audit."""

    beta: float = 0.10
    beta_profile: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    deployment_weighting: str = "equal"
    audit_measure: str = "balanced"
    partition_rule: str = "one_axis_at_a_time"
    missingness_rule: str = "strict"
    standardization_target: str = "uniform"
    standardization_weights: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    support_rule: str = "preflight"
    inference_target: str = "fixed_slice_universe"
    estimand_scope: str = "fixed_slice_universe"
    inference_method: str = "cluster_maxt"
    dependence_design: str = "independent_clusters"
    confidence_level: float = 0.95
    min_slices: int = 2
    min_units_per_slice: int = 1
    min_clusters_per_slice: int = 2
    min_clusters_for_default: int = 30
    group_variable: str = "group"
    balance_variable: str = ""
    loss_name: str = "risk"
    task_adapter: str = "generic"
    independent_unit_column: str = "independent_unit_id"
    cluster_column: str = "cluster_id"
    spatial_block_column: str = "spatial_block_id"
    metric_version: str = METRIC_VERSION
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.beta) <= 1.0:
            raise ValueError("beta must be in (0, 1].")
        if not self.beta_profile:
            raise ValueError("beta_profile must contain at least one value.")
        if any(not 0.0 < float(value) <= 1.0 for value in self.beta_profile):
            raise ValueError("Every beta_profile value must be in (0, 1].")
        if self.deployment_weighting not in {"equal", "empirical", "custom"}:
            raise ValueError("deployment_weighting must be equal, empirical, or custom.")
        measure_for_weighting = {
            "equal": "balanced",
            "empirical": "observed",
            "custom": "external",
        }
        if self.audit_measure not in {"balanced", "observed", "external"}:
            raise ValueError("audit_measure must be balanced, observed, or external.")
        if self.audit_measure != measure_for_weighting[self.deployment_weighting]:
            raise ValueError(
                "audit_measure and deployment_weighting disagree: use balanced/equal, "
                "observed/empirical, or external/custom."
            )
        if self.partition_rule not in {"one_axis_at_a_time", "explicit_intersection"}:
            raise ValueError("partition_rule must be one_axis_at_a_time or explicit_intersection.")
        if self.missingness_rule not in {"strict", "overlap", "partial_bounds"}:
            raise ValueError("missingness_rule must be strict, overlap, or partial_bounds.")
        if self.standardization_target not in {"uniform", "custom"}:
            raise ValueError("standardization_target must be uniform or custom.")
        if self.standardization_target == "custom" and not self.standardization_weights:
            raise ValueError("custom standardization_target requires standardization_weights.")
        if self.standardization_target == "uniform" and self.standardization_weights:
            raise ValueError("uniform standardization_target may not include custom weights.")
        if self.standardization_weights:
            values = [float(value) for _, value in self.standardization_weights]
            if any(not math.isfinite(value) or value < 0.0 for value in values) or sum(values) <= 0.0:
                raise ValueError("standardization_weights must be finite, non-negative, and have positive mass.")
        if self.inference_target not in {"fixed_slice_universe", "slice_superpopulation"}:
            raise ValueError("Unsupported inference_target.")
        if self.estimand_scope not in {
            "fixed_slice_universe",
            "fixed_event_universe",
            "sampled_unit_superpopulation",
            "slice_superpopulation",
        }:
            raise ValueError("Unsupported estimand_scope.")
        if (self.inference_target == "slice_superpopulation") != (
            self.estimand_scope == "slice_superpopulation"
        ):
            raise ValueError("inference_target and estimand_scope must agree on slice-superpopulation inference.")
        if self.inference_method not in {"none", "cluster_maxt", "spatial_maxt", "strict_bound"}:
            raise ValueError("Unsupported inference_method.")
        if self.dependence_design not in {
            "independent_clusters",
            "spatial_blocks",
            "event_clusters",
            "fixed_universe_descriptive",
        }:
            raise ValueError("Unsupported dependence_design.")
        if self.inference_method == "spatial_maxt" and self.dependence_design != "spatial_blocks":
            raise ValueError("spatial_maxt requires dependence_design=spatial_blocks.")
        if self.inference_method == "cluster_maxt" and self.dependence_design not in {
            "independent_clusters",
            "event_clusters",
        }:
            raise ValueError("cluster_maxt requires independent_clusters or event_clusters.")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be in (0, 1).")
        if self.min_slices < 2:
            raise ValueError("min_slices must be at least 2.")
        if self.min_units_per_slice < 1:
            raise ValueError("min_units_per_slice must be positive.")
        if self.min_clusters_per_slice < 1:
            raise ValueError("min_clusters_per_slice must be positive.")
        if self.min_clusters_for_default < 2:
            raise ValueError("min_clusters_for_default must be at least 2.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["beta_profile"] = list(self.beta_profile)
        payload["metadata"] = {key: value for key, value in self.metadata}
        payload["standardization_weights"] = {key: value for key, value in self.standardization_weights}
        return payload

    @property
    def signature(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BWERProtocol":
        data = dict(values)
        if "beta_profile" in data:
            data["beta_profile"] = tuple(float(value) for value in data["beta_profile"])
        metadata = data.get("metadata", ())
        if isinstance(metadata, Mapping):
            data["metadata"] = tuple(sorted((str(key), str(value)) for key, value in metadata.items()))
        standardization_weights = data.get("standardization_weights", ())
        if isinstance(standardization_weights, Mapping):
            data["standardization_weights"] = tuple(
                sorted((str(key), float(value)) for key, value in standardization_weights.items())
            )
        return cls(**data)


Protocol = BWERProtocol
