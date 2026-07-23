from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_core import BWERPointEstimate, bwer_from_arrays, compute_geobwer_profile
from rsfm_fairness_audit.bwer_inference import (
    CertifiedBWER,
    HonestConfirmedBWER,
    PairedBWERComparison,
    certified_geobwer,
    certify_geobwer_from_band,
    honest_confirmed_bwer,
    paired_bwer_comparison,
    simultaneous_standardized_risk_band,
)
from rsfm_fairness_audit.bwer_protocol import BWERProtocol, Protocol, Validity
from rsfm_fairness_audit.bwer_schema import SchemaValidation, validate_formal_audit_rows
from rsfm_fairness_audit.bwer_standardization import (
    PartialBWERBounds,
    StandardizationResult,
    partial_bwer_bounds,
    standardize_group_risks,
)
from rsfm_fairness_audit.io import ensure_dir, write_csv


def _native(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return _native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Validity):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    return value


@dataclass(frozen=True)
class GeoBWERAxisResult:
    axis: str
    validity: Validity
    point: BWERPointEstimate | None
    profile: tuple[BWERPointEstimate, ...]
    certified: CertifiedBWER | None
    group_risks: tuple[tuple[str, float], ...]
    group_support: tuple[tuple[str, int], ...]
    group_cluster_support: tuple[tuple[str, int], ...]
    excluded_groups: tuple[str, ...]
    standardization: StandardizationResult | None = None
    partial_bounds: PartialBWERBounds | None = None
    message: str = ""

    def summary_dict(self, protocol: BWERProtocol) -> dict[str, Any]:
        output: dict[str, Any] = {
            "axis": self.axis,
            "validity": self.validity.value,
            "protocol_hash": protocol.signature,
            "metric_version": protocol.metric_version,
            "loss_name": protocol.loss_name,
            "deployment_weighting": protocol.deployment_weighting,
            "audit_measure": protocol.audit_measure,
            "partition_rule": protocol.partition_rule,
            "estimand_scope": protocol.estimand_scope,
            "dependence_design": protocol.dependence_design,
            "excluded_groups": ";".join(self.excluded_groups),
            "message": self.message,
            "standardization_target": protocol.standardization_target,
        }
        if self.point is not None:
            output.update(self.point.to_dict())
        if self.certified is not None:
            output.update(
                {
                    "ci_low": self.certified.ci_low,
                    "ci_high": self.certified.ci_high,
                    "lower_confidence_bound": self.certified.lower_confidence_bound,
                    "certification_radius": self.certified.radius,
                    "weighted_sum_radius": self.certified.weighted_sum_radius,
                    "total_variation_radius": self.certified.total_variation_radius,
                    "certification_radius_method": self.certified.radius_method,
                    "parameter_upper_bound": self.certified.parameter_upper_bound,
                    "cluster_count": self.certified.band.cluster_count,
                    "critical_value": self.certified.band.critical_value,
                }
            )
        else:
            output.update(
                {
                    "ci_low": "",
                    "ci_high": "",
                    "lower_confidence_bound": "",
                    "certification_radius": "",
                    "weighted_sum_radius": "",
                    "total_variation_radius": "",
                    "certification_radius_method": "",
                    "parameter_upper_bound": "",
                }
            )
        if self.partial_bounds is not None:
            output.update(
                {
                    "partial_bwer_lower": self.partial_bounds.lower,
                    "partial_bwer_upper": self.partial_bounds.upper,
                }
            )
        else:
            output.update({"partial_bwer_lower": "", "partial_bwer_upper": ""})
        return output


@dataclass(frozen=True)
class GeoBWERAudit:
    protocol: BWERProtocol
    axes: tuple[GeoBWERAxisResult, ...]
    schema: SchemaValidation | None = None

    @property
    def ok(self) -> bool:
        return bool(self.axes) and all(axis.validity in {Validity.VALID, Validity.DESCRIPTIVE_ONLY} for axis in self.axes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict(),
            "protocol_hash": self.protocol.signature,
            "schema": None
            if self.schema is None
            else {
                "validity": self.schema.validity.value,
                "errors": list(self.schema.errors),
                "warnings": list(self.schema.warnings),
                "columns": list(self.schema.columns),
                "row_count": self.schema.row_count,
            },
            "axes": [
                {
                    "summary": axis.summary_dict(self.protocol),
                    "group_risks": dict(axis.group_risks),
                    "group_support": dict(axis.group_support),
                    "group_cluster_support": dict(axis.group_cluster_support),
                    "profile": [point.to_dict() for point in axis.profile],
                    "standardization": None
                    if axis.standardization is None
                    else {
                        "validity": axis.standardization.validity.value,
                        "target_weights": dict(axis.standardization.target_weights),
                        "risk_lower": dict(axis.standardization.risk_lower),
                        "risk_upper": dict(axis.standardization.risk_upper),
                        "support": list(axis.standardization.support),
                        "missing_cells": list(axis.standardization.missing_cells),
                        "message": axis.standardization.message,
                    },
                    "partial_bounds": None
                    if axis.partial_bounds is None
                    else {
                        "lower": axis.partial_bounds.lower,
                        "upper": axis.partial_bounds.upper,
                        "point_if_identified": axis.partial_bounds.point_if_identified,
                        "validity": axis.partial_bounds.validity.value,
                    },
                }
                for axis in self.axes
            ],
        }

    def to_report(self, output_dir: str | Path) -> dict[str, Path]:
        output = ensure_dir(output_dir)
        artifacts = {
            "protocol": output / "geobwer_protocol.json",
            "summary": output / "geobwer_summary.csv",
            "by_group": output / "geobwer_by_group.csv",
            "profile": output / "geobwer_profile.csv",
            "report_card": output / "geobwer_report_card.json",
            "report": output / "geobwer_report.md",
        }
        artifacts["protocol"].write_text(json.dumps(self.protocol.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        payload = _native(self.to_dict())
        artifacts["report_card"].write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        write_csv(artifacts["summary"], [axis.summary_dict(self.protocol) for axis in self.axes])
        by_group: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        for axis in self.axes:
            supports = dict(axis.group_support)
            cluster_supports = dict(axis.group_cluster_support)
            selected = dict(axis.point.allocation.selected_mass) if axis.point is not None else {}
            for group, risk in axis.group_risks:
                by_group.append(
                    {
                        "axis": axis.axis,
                        "group": group,
                        "risk": risk,
                        "support": supports.get(group, 0),
                        "cluster_support": cluster_supports.get(group, 0),
                        "selected_tail_mass": selected.get(group, 0.0),
                        "protocol_hash": self.protocol.signature,
                    }
                )
            for point in axis.profile:
                row = point.to_dict()
                row.update({"axis": axis.axis, "validity": axis.validity.value, "protocol_hash": self.protocol.signature})
                profile_rows.append(row)
        write_csv(artifacts["by_group"], by_group)
        write_csv(artifacts["profile"], profile_rows)
        lines = [
            "# GeoBWER Audit Report",
            "",
            f"- Metric version: `{self.protocol.metric_version}`",
            f"- Protocol hash: `{self.protocol.signature}`",
            f"- Primary beta: `{self.protocol.beta}`",
            f"- Audit measure: `{self.protocol.audit_measure}`",
            f"- Partition rule: `{self.protocol.partition_rule}`",
            f"- Estimand scope: `{self.protocol.estimand_scope}`",
            f"- Dependence design: `{self.protocol.dependence_design}`",
            f"- Inference: `{self.protocol.inference_method}`",
            "",
            "| Axis | Validity | Mean risk | Tail risk | GeoBWER | 95% CI | LCB |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for axis in self.axes:
            if axis.point is None:
                lines.append(f"| {axis.axis} | {axis.validity.value} | — | — | — | — | — |")
                continue
            ci = "—" if axis.certified is None else f"[{axis.certified.ci_low:.6f}, {axis.certified.ci_high:.6f}]"
            lcb = "—" if axis.certified is None else f"{axis.certified.lower_confidence_bound:.6f}"
            lines.append(
                f"| {axis.axis} | {axis.validity.value} | {axis.point.mean_risk:.6f} | {axis.point.tail_risk:.6f} | {axis.point.bwer:.6f} | {ci} | {lcb} |"
            )
        lines.extend(
            [
                "",
                "`apparent` point estimates describe the observed audit table. A positive LCB is required for a certified positive disparity claim.",
            ]
        )
        artifacts["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
        return artifacts


def _custom_axis_weights(
    deployment_weights: Mapping[Any, Any] | None,
    axis: str,
) -> Mapping[Any, float] | None:
    if deployment_weights is None:
        return None
    if axis in deployment_weights and isinstance(deployment_weights[axis], Mapping):
        return deployment_weights[axis]
    return deployment_weights  # type: ignore[return-value]


def audit(
    *,
    loss: Sequence[float],
    groups: Sequence[Any] | Mapping[str, Sequence[Any]],
    unit_id: Sequence[Any],
    protocol: BWERProtocol | None = None,
    cluster_id: Sequence[Any] | None = None,
    spatial_block_id: Sequence[Any] | None = None,
    deployment_weights: Mapping[Any, Any] | None = None,
    required_group_universe: Mapping[str, Sequence[Any]] | None = None,
    balance: Sequence[Any] | None = None,
    formal: bool = True,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> GeoBWERAudit:
    protocol = protocol or BWERProtocol()
    if formal and protocol.inference_target == "slice_superpopulation":
        raise ValueError(
            "Formal slice_superpopulation inference is not implemented by the fixed-slice simultaneous-band engine. "
            "Use inference_target=fixed_slice_universe or a separately validated clone-safe slice-resampling protocol."
        )
    y = np.asarray(loss, dtype=float)
    units = np.asarray([str(value) for value in unit_id], dtype=object)
    if len(y) == 0 or len(y) != len(units) or not np.all(np.isfinite(y)):
        raise ValueError("loss and unit_id must be non-empty, aligned, and finite.")
    axes = {protocol.group_variable: groups} if not isinstance(groups, Mapping) else dict(groups)
    if any(len(values) != len(y) for values in axes.values()):
        raise ValueError("Every group axis must align with loss.")
    if protocol.partition_rule == "explicit_intersection" and len(axes) != 1:
        raise ValueError(
            "partition_rule=explicit_intersection requires one pre-composed intersection axis; "
            "multiple overlapping axes are not a partition."
        )
    if formal:
        missing_by_axis = {
            str(axis): sum(
                str(value).strip().lower() in {"", "nan", "none", "null"}
                for value in values
            )
            for axis, values in axes.items()
        }
        missing_by_axis = {axis: count for axis, count in missing_by_axis.items() if count}
        if missing_by_axis:
            raise ValueError(f"Formal group axes contain missing values: {missing_by_axis}")
    if len(set(units.tolist())) != len(units):
        # Repeated rows per independent unit are legitimate, but the caller must
        # then supply a cluster/block ID that carries the dependence structure.
        if formal and cluster_id is None and spatial_block_id is None:
            raise ValueError("Repeated unit_id values require cluster_id or spatial_block_id in formal mode.")
    if (
        formal
        and protocol.inference_method != "none"
        and protocol.dependence_design == "spatial_blocks"
        and spatial_block_id is None
    ):
        raise ValueError("dependence_design=spatial_blocks requires spatial_block_id in formal mode.")
    if (
        formal
        and protocol.inference_method != "none"
        and protocol.dependence_design in {"independent_clusters", "event_clusters"}
        and cluster_id is None
    ):
        raise ValueError(f"dependence_design={protocol.dependence_design} requires cluster_id in formal mode.")
    clusters_raw = spatial_block_id if spatial_block_id is not None else cluster_id
    clusters = None if clusters_raw is None else np.asarray([str(value) for value in clusters_raw], dtype=object)
    if clusters is not None and len(clusters) != len(y):
        raise ValueError("cluster_id/spatial_block_id must align with loss.")
    if formal and protocol.inference_method != "none" and clusters is None:
        raise ValueError("Formal GeoBWER inference requires cluster_id or spatial_block_id; no i.i.d. fallback is permitted.")
    balance_values = None if balance is None else np.asarray([str(value) for value in balance], dtype=object)
    if balance_values is not None and len(balance_values) != len(y):
        raise ValueError("balance must align with loss.")
    results: list[GeoBWERAxisResult] = []
    metadata = dict(protocol.metadata)
    for axis, raw_values in axes.items():
        group_values = np.asarray([str(value) for value in raw_values], dtype=object)
        grouped_support = {group: int(np.sum(group_values == group)) for group in sorted(set(group_values.tolist()))}
        expected_groups = (
            set()
            if required_group_universe is None or str(axis) not in required_group_universe
            else {str(value) for value in required_group_universe[str(axis)]}
        )
        missing_expected = tuple(sorted(expected_groups - set(grouped_support)))
        if missing_expected:
            support_with_missing = dict(grouped_support)
            support_with_missing.update({group: 0 for group in missing_expected})
            cluster_support_with_missing = {
                group: (
                    len(set(clusters[group_values == group].tolist()))
                    if clusters is not None and group in grouped_support
                    else support_with_missing[group] if group in grouped_support else 0
                )
                for group in support_with_missing
            }
            results.append(
                GeoBWERAxisResult(
                    axis=str(axis),
                    validity=Validity.NOT_IDENTIFIED_SELECTIVE_COVERAGE,
                    point=None,
                    profile=(),
                    certified=None,
                    group_risks=(),
                    group_support=tuple(sorted(support_with_missing.items())),
                    group_cluster_support=tuple(sorted(cluster_support_with_missing.items())),
                    excluded_groups=missing_expected,
                    message=(
                        "Conditional selective risk is not identified because these pre-registered groups "
                        f"have zero accepted units: {', '.join(missing_expected)}."
                    ),
                )
            )
            continue
        grouped_cluster_support = {
            group: len(set(clusters[group_values == group].tolist())) if clusters is not None else grouped_support[group]
            for group in grouped_support
        }
        require_cluster_support = formal and protocol.inference_method != "none"
        included = tuple(
            group
            for group, count in grouped_support.items()
            if count >= protocol.min_units_per_slice
            and (
                not require_cluster_support
                or grouped_cluster_support[group] >= protocol.min_clusters_per_slice
            )
        )
        excluded = tuple(group for group in grouped_support if group not in included)
        mask = np.isin(group_values, included)
        custom = _custom_axis_weights(deployment_weights, str(axis))
        if len(included) < protocol.min_slices:
            results.append(
                GeoBWERAxisResult(
                    axis=str(axis),
                    validity=Validity.INSUFFICIENT_SLICES,
                    point=None,
                    profile=(),
                    certified=None,
                    group_risks=(),
                    group_support=tuple(sorted(grouped_support.items())),
                    group_cluster_support=tuple(sorted(grouped_cluster_support.items())),
                    excluded_groups=excluded,
                    message=(
                        f"Only {len(included)} groups pass min_units_per_slice={protocol.min_units_per_slice}"
                        + (
                            f" and min_clusters_per_slice={protocol.min_clusters_per_slice}."
                            if require_cluster_support
                            else "."
                        )
                    ),
                )
            )
            continue
        standardization: StandardizationResult | None = None
        if balance_values is not None:
            rows = [
                {"group": group_values[index], "balance": balance_values[index], "risk": float(y[index])}
                for index in np.flatnonzero(mask)
            ]
            standardization = standardize_group_risks(
                rows,
                group_column="group",
                balance_column="balance",
                loss_column="risk",
                target_weights=(
                    None
                    if protocol.standardization_target == "uniform"
                    else dict(protocol.standardization_weights)
                ),
                missingness_rule=protocol.missingness_rule,
            )
            if standardization.validity != Validity.VALID:
                bounds: PartialBWERBounds | None = None
                if protocol.missingness_rule == "partial_bounds":
                    if protocol.deployment_weighting == "equal":
                        bound_weights = None
                    elif protocol.deployment_weighting == "empirical":
                        bound_weights = {
                            group: grouped_support[group]
                            for group in standardization.lower_dict()
                        }
                    else:
                        if custom is None:
                            raise ValueError("A custom deployment weight mapping is required by the protocol.")
                        bound_weights = custom
                    bounds = partial_bwer_bounds(
                        standardization.lower_dict(),
                        standardization.upper_dict(),
                        beta=protocol.beta,
                        deployment_weights=bound_weights,
                    )
                results.append(
                    GeoBWERAxisResult(
                        axis=str(axis),
                        validity=standardization.validity,
                        point=None,
                        profile=(),
                        certified=None,
                        group_risks=standardization.group_risks,
                        group_support=tuple(sorted(grouped_support.items())),
                        group_cluster_support=tuple(sorted(grouped_cluster_support.items())),
                        excluded_groups=excluded,
                        standardization=standardization,
                        partial_bounds=bounds,
                        message=standardization.message,
                    )
                )
                continue
            group_risks = standardization.group_risk_dict()
            supports = {group: grouped_support[group] for group in group_risks}
        else:
            _, group_risks, supports = bwer_from_arrays(y[mask], group_values[mask], beta=protocol.beta)
        if protocol.deployment_weighting == "equal":
            weights = None
        elif protocol.deployment_weighting == "empirical":
            weights = {group: supports[group] for group in group_risks}
        else:
            if custom is None:
                raise ValueError("A custom deployment weight mapping is required by the protocol.")
            weights = custom
        point = compute_geobwer_profile(group_risks, (protocol.beta,), weights)[0]
        profile = tuple(compute_geobwer_profile(group_risks, protocol.beta_profile, weights))
        certified: CertifiedBWER | None = None
        validity = Validity.VALID if not excluded else Validity.DESCRIPTIVE_ONLY
        message = "" if not excluded else "Groups below the pre-registered support threshold were excluded; inspect valid deployment mass."
        if formal and protocol.inference_method != "none":
            unique_clusters = len(set(clusters[mask].tolist())) if clusters is not None else 0
            small_calibrated = str(metadata.get("small_cluster_calibrated", "false")).lower() == "true"
            if unique_clusters < protocol.min_clusters_for_default and not small_calibrated:
                validity = Validity.INFERENCE_NOT_CERTIFIED
                message = (
                    f"Only {unique_clusters} clusters; protocol requires {protocol.min_clusters_for_default} "
                    "unless a task-specific small-cluster calibration has passed."
                )
            elif balance_values is not None:
                standardized_band = simultaneous_standardized_risk_band(
                    y[mask],
                    group_values[mask],
                    balance_values[mask],
                    clusters[mask],
                    target_weights=dict(standardization.target_weights) if standardization is not None else {},
                    confidence_level=protocol.confidence_level,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    min_clusters_per_group=protocol.min_clusters_per_slice,
                )
                if standardized_band.validity == Validity.VALID:
                    certified = certify_geobwer_from_band(standardized_band, beta=protocol.beta, deployment_weights=weights)
                    validity = certified.validity if certified.validity != Validity.VALID else validity
                else:
                    validity = standardized_band.validity
                    message = standardized_band.message
            else:
                certified = certified_geobwer(
                    y[mask],
                    group_values[mask],
                    clusters[mask],
                    beta=protocol.beta,
                    deployment_weights=weights,
                    confidence_level=protocol.confidence_level,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    min_clusters_per_group=protocol.min_clusters_per_slice,
                )
                validity = certified.validity if certified.validity != Validity.VALID else validity
                if certified.message:
                    message = certified.message
        results.append(
            GeoBWERAxisResult(
                axis=str(axis),
                validity=validity,
                point=point,
                profile=profile,
                certified=certified,
                group_risks=tuple(sorted(group_risks.items())),
                group_support=tuple(sorted(supports.items())),
                group_cluster_support=tuple(
                    sorted((group, grouped_cluster_support[group]) for group in supports)
                ),
                excluded_groups=excluded,
                standardization=standardization,
                message=message,
            )
        )
    return GeoBWERAudit(protocol=protocol, axes=tuple(results))


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_columns: Sequence[str],
    protocol: BWERProtocol,
    loss_column: str = "risk",
    unit_column: str = "independent_unit_id",
    cluster_column: str | None = None,
    balance_column: str | None = None,
    formal: bool = True,
    require_probabilities: bool = False,
    required_group_universe: Mapping[str, Sequence[Any]] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> GeoBWERAudit:
    if not rows:
        raise ValueError("Formal audit table is empty.")
    if not group_columns:
        raise ValueError("At least one pre-registered group column is required.")
    missing_group_fields = {
        column: sum(
            1
            for row in rows
            if column not in row or str(row.get(column, "")).strip().lower() in {"", "nan", "none", "null"}
        )
        for column in group_columns
    }
    missing_group_fields = {key: value for key, value in missing_group_fields.items() if value}
    if formal and missing_group_fields:
        raise ValueError(f"Formal audit group columns contain missing values: {missing_group_fields}")
    cluster_name = cluster_column or (
        protocol.spatial_block_column if protocol.inference_method == "spatial_maxt" else protocol.cluster_column
    )
    schema = validate_formal_audit_rows(
        rows,
        task_adapter=protocol.task_adapter,
        require_spatial_block=protocol.inference_method == "spatial_maxt",
        required_cluster_column=cluster_name if formal and protocol.inference_method != "none" else None,
        require_probabilities=require_probabilities,
        expected_protocol_hash=protocol.signature,
        expected_metric_version=protocol.metric_version,
    )
    if formal and not schema.ok:
        raise ValueError("Formal audit schema failed: " + " | ".join(schema.errors))
    result = audit(
        loss=[float(row[loss_column]) for row in rows],
        groups={column: [row[column] for row in rows] for column in group_columns},
        unit_id=[row.get(unit_column, row.get("unit_id", row.get("sample_id"))) for row in rows],
        cluster_id=(
            None
            if protocol.dependence_design == "spatial_blocks" or cluster_name not in rows[0]
            else [row[cluster_name] for row in rows]
        ),
        spatial_block_id=(
            [row[cluster_name] for row in rows]
            if protocol.dependence_design == "spatial_blocks" and cluster_name in rows[0]
            else None
        ),
        protocol=protocol,
        balance=None if balance_column is None else [row[balance_column] for row in rows],
        required_group_universe=required_group_universe,
        formal=formal,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return GeoBWERAudit(protocol=result.protocol, axes=result.axes, schema=schema)


def compare(
    *,
    loss_a: Sequence[float],
    loss_b: Sequence[float],
    groups: Sequence[Any],
    unit_id: Sequence[Any],
    cluster_id: Sequence[Any],
    protocol: BWERProtocol | None = None,
    model_a: str = "model_a",
    model_b: str = "model_b",
    deployment_weights: Mapping[Any, float] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> PairedBWERComparison:
    protocol = protocol or BWERProtocol()
    if len(set(str(value) for value in unit_id)) != len(unit_id):
        raise ValueError("compare requires one aligned loss per independent unit; aggregate repeated rows first.")
    if len(loss_a) != len(unit_id):
        raise ValueError("Paired arrays must align with unit_id.")
    return paired_bwer_comparison(
        loss_a,
        loss_b,
        groups,
        cluster_id,
        model_a=model_a,
        model_b=model_b,
        beta=protocol.beta,
        deployment_weights=deployment_weights,
        confidence_level=protocol.confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def confirm(
    *,
    loss: Sequence[float],
    groups: Sequence[Any],
    cluster_id: Sequence[Any],
    protocol: BWERProtocol | None = None,
    deployment_weights: Mapping[Any, float] | None = None,
    seed: int = 42,
    min_clusters_per_group_per_partition: int = 2,
) -> HonestConfirmedBWER:
    """Run the A→B/B→A honest tail-confirmation diagnostic."""

    protocol = protocol or BWERProtocol()
    if protocol.inference_target != "fixed_slice_universe":
        raise ValueError("Honest tail confirmation currently targets a fixed pre-specified slice universe.")
    return honest_confirmed_bwer(
        loss,
        groups,
        cluster_id,
        beta=protocol.beta,
        deployment_weights=deployment_weights,
        seed=seed,
        min_clusters_per_group_per_partition=min_clusters_per_group_per_partition,
    )


__all__ = [
    "BWERProtocol",
    "Protocol",
    "Validity",
    "GeoBWERAudit",
    "GeoBWERAxisResult",
    "audit",
    "audit_rows",
    "compare",
    "confirm",
]
