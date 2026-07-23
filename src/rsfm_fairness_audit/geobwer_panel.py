from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_inference import PairedBWERComparison
from rsfm_fairness_audit.bwer_protocol import BWERProtocol, Validity
from rsfm_fairness_audit.geobwer import audit_rows, compare
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


class GeoBWERPanelError(RuntimeError):
    """Raised when model outputs cannot support a paired common-unit panel."""


@dataclass(frozen=True)
class ModelPanelArtifacts:
    output_dir: Path
    model_summary: Path
    paired_comparisons: Path
    common_support: Path
    report: Path


def _rows_by_unit(
    rows: Sequence[Mapping[str, Any]],
    *,
    unit_column: str,
    model_name: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in rows:
        unit = str(source.get(unit_column, "")).strip()
        if not unit:
            raise GeoBWERPanelError(f"{model_name}: missing {unit_column}.")
        if unit in output:
            raise GeoBWERPanelError(f"{model_name}: duplicate {unit_column}={unit!r}.")
        output[unit] = dict(source)
    return output


def _native_comparison(value: PairedBWERComparison) -> dict[str, Any]:
    payload = asdict(value)
    payload["validity"] = value.validity.value
    payload["common_groups"] = ";".join(value.common_groups)
    return payload


def run_geobwer_model_panel(
    model_tables: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    protocol: BWERProtocol,
    group_column: str | None = None,
    loss_column: str = "risk",
    unit_column: str | None = None,
    cluster_column: str | None = None,
    comparison_pairs: Sequence[tuple[str, str]] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> ModelPanelArtifacts:
    """Compare two or more models on exactly the same deployment units.

    The panel reports both sample-weighted mean risk and equal-deployment
    GeoBWER.  All paired intervals are calculated only after intersecting
    physical units and verifying that their slice and dependence-cluster labels
    agree across models.
    """

    if len(model_tables) < 2:
        raise GeoBWERPanelError("At least two model tables are required.")
    axis = group_column or protocol.group_variable
    unit = unit_column or protocol.independent_unit_column
    cluster = cluster_column or (
        protocol.spatial_block_column if protocol.inference_method == "spatial_maxt" else protocol.cluster_column
    )
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    total_units: dict[str, int] = {}
    dataset_signatures: dict[str, set[str]] = {}
    for model_name, table in model_tables.items():
        rows = read_csv_rows(table)
        indexed = _rows_by_unit(rows, unit_column=unit, model_name=str(model_name))
        loaded[str(model_name)] = indexed
        total_units[str(model_name)] = len(indexed)
        dataset_signatures[str(model_name)] = {
            str(row.get("dataset_signature", "")) for row in rows if str(row.get("dataset_signature", ""))
        }
        if len(dataset_signatures[str(model_name)]) != 1:
            raise GeoBWERPanelError(
                f"{model_name}: formal comparison requires exactly one non-empty dataset_signature."
            )
    signature_values = {next(iter(values)) for values in dataset_signatures.values()}
    if len(signature_values) != 1:
        raise GeoBWERPanelError(
            "Models do not share one model-independent dataset signature. Regenerate portable dataset lineage "
            "from semantic identifiers and content hashes, not model-specific output paths."
        )
    common_units = tuple(sorted(set.intersection(*(set(rows) for rows in loaded.values()))))
    if len(common_units) < 2:
        raise GeoBWERPanelError("Fewer than two physical units are common to every model.")
    reference_model = next(iter(loaded))
    for sample in common_units:
        reference = loaded[reference_model][sample]
        for model_name, rows in loaded.items():
            candidate = rows[sample]
            invariant_columns = (
                axis,
                cluster,
                "dataset",
                "task",
                "split",
                "split_role",
                "protocol_hash",
                "metric_version",
                "class_mapping_hash",
                "label",
                "valid_pixels",
                "positive_pixels",
            )
            for column in invariant_columns:
                if column not in reference and column not in candidate:
                    continue
                if str(candidate.get(column, "")) != str(reference.get(column, "")):
                    raise GeoBWERPanelError(
                        f"Common-unit metadata drift for unit={sample!r}, column={column!r}, model={model_name!r}."
                    )
    common_fraction = {
        model: len(common_units) / total for model, total in total_units.items()
    }
    if min(common_fraction.values()) < 0.95:
        raise GeoBWERPanelError(
            f"Common support retains less than 95% for at least one model: {common_fraction}. "
            "Repair prediction completeness instead of comparing selective supports."
        )

    summaries: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for model_name, indexed in loaded.items():
        rows = [indexed[sample] for sample in common_units]
        audit = audit_rows(
            rows,
            group_columns=(axis,),
            protocol=protocol,
            loss_column=loss_column,
            unit_column=unit,
            cluster_column=cluster,
            formal=True,
            require_probabilities=True,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        result = audit.axes[0]
        if result.point is None or result.validity not in {Validity.VALID, Validity.DESCRIPTIVE_ONLY}:
            raise GeoBWERPanelError(
                f"{model_name}: common-support GeoBWER is not estimable ({result.validity.value}: {result.message})."
            )
        audits[model_name] = result
        risks = np.asarray([float(row[loss_column]) for row in rows], dtype=float)
        summaries.append(
            {
                "model": model_name,
                "common_units": len(common_units),
                "common_fraction": common_fraction[model_name],
                "sample_mean_risk": float(np.mean(risks)),
                "deployment_mean_risk": result.point.mean_risk,
                "tail_risk": result.point.tail_risk,
                "geobwer": result.point.bwer,
                "geobwer_ci_low": "" if result.certified is None else result.certified.ci_low,
                "geobwer_ci_high": "" if result.certified is None else result.certified.ci_high,
                "validity": result.validity.value,
                "protocol_hash": protocol.signature,
            }
        )
    mean_order = {
        row["model"]: rank + 1
        for rank, row in enumerate(sorted(summaries, key=lambda value: (value["sample_mean_risk"], value["model"])))
    }
    tail_order = {
        row["model"]: rank + 1
        for rank, row in enumerate(sorted(summaries, key=lambda value: (value["tail_risk"], value["model"])))
    }
    for row in summaries:
        row["mean_risk_rank"] = mean_order[row["model"]]
        row["tail_risk_rank"] = tail_order[row["model"]]
        row["rank_shift_tail_minus_mean"] = tail_order[row["model"]] - mean_order[row["model"]]

    if comparison_pairs is None:
        pairs = tuple(itertools.combinations(sorted(loaded), 2))
    else:
        normalized_pairs: list[tuple[str, str]] = []
        seen_pairs: set[frozenset[str]] = set()
        for left, right in comparison_pairs:
            pair = (str(left), str(right))
            if pair[0] == pair[1] or pair[0] not in loaded or pair[1] not in loaded:
                raise GeoBWERPanelError(f"Invalid pre-specified comparison pair: {pair}.")
            identity = frozenset(pair)
            if identity in seen_pairs:
                raise GeoBWERPanelError(f"Duplicate pre-specified comparison pair: {pair}.")
            seen_pairs.add(identity)
            normalized_pairs.append(pair)
        pairs = tuple(normalized_pairs)
        if not pairs:
            raise GeoBWERPanelError("comparison_pairs cannot be empty when supplied.")
    familywise_alpha = 1.0 - protocol.confidence_level
    pairwise_confidence = 1.0 - familywise_alpha / len(pairs)
    pairwise_protocol = replace(protocol, confidence_level=pairwise_confidence)
    pairwise: list[dict[str, Any]] = []
    for pair_index, (model_a, model_b) in enumerate(pairs):
        eligible_groups = set(dict(audits[model_a].group_risks)) & set(dict(audits[model_b].group_risks))
        pair_units = [
            sample for sample in common_units if str(loaded[model_a][sample][axis]) in eligible_groups
        ]
        comparison = compare(
            loss_a=[float(loaded[model_a][sample][loss_column]) for sample in pair_units],
            loss_b=[float(loaded[model_b][sample][loss_column]) for sample in pair_units],
            groups=[loaded[model_a][sample][axis] for sample in pair_units],
            unit_id=pair_units,
            cluster_id=[loaded[model_a][sample][cluster] for sample in pair_units],
            protocol=pairwise_protocol,
            model_a=model_a,
            model_b=model_b,
            n_bootstrap=n_bootstrap,
            seed=seed + pair_index,
        )
        comparison_row = _native_comparison(comparison)
        comparison_row.update(
            {
                "multiplicity_method": "bonferroni_familywise",
                "comparison_family_size": len(pairs),
                "familywise_confidence_level": protocol.confidence_level,
                "pairwise_adjusted_confidence_level": pairwise_confidence,
            }
        )
        pairwise.append(comparison_row)

    output = ensure_dir(output_dir)
    summary_path = output / "model_panel_summary.csv"
    pairwise_path = output / "paired_geobwer_comparisons.csv"
    support_path = output / "common_support.json"
    report_path = output / "model_panel_report.md"
    write_csv(summary_path, summaries)
    write_csv(pairwise_path, pairwise)
    support_path.write_text(
        json.dumps(
            {
                "schema": "geobwer.model_panel_common_support.v1",
                "protocol_hash": protocol.signature,
                "group_column": axis,
                "unit_column": unit,
                "cluster_column": cluster,
                "common_units": len(common_units),
                "total_units": total_units,
                "common_fraction": common_fraction,
                "dataset_signatures": {key: sorted(value) for key, value in dataset_signatures.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ranking_reversals = [row for row in summaries if row["mean_risk_rank"] != row["tail_risk_rank"]]
    report_path.write_text(
        "\n".join(
            [
                "# GeoBWER Common-Unit Model Panel",
                "",
                f"- Common physical units: {len(common_units)}",
                f"- Slice axis: `{axis}`",
                f"- Dependence cluster: `{cluster}`",
                f"- Models: {', '.join(sorted(loaded))}",
                f"- Models whose average-risk and tail-risk ranks differ: {len(ranking_reversals)}",
                f"- Pairwise inference: Bonferroni family-wise {100 * protocol.confidence_level:.1f}% "
                f"over {len(pairs)} comparisons",
                "",
                "All comparisons use the same physical units, slice labels, clusters, deployment measure, and protocol hash.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return ModelPanelArtifacts(output, summary_path, pairwise_path, support_path, report_path)


__all__ = ["GeoBWERPanelError", "ModelPanelArtifacts", "run_geobwer_model_panel"]
