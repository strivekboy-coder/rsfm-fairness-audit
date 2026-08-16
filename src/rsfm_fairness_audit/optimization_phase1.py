from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer_core import compute_geobwer, compute_geobwer_profile
from rsfm_fairness_audit.io import ensure_dir, write_csv


SCHEMA = "geobwer.optimization_1_7.v1"
BETAS = (0.05, 0.10, 0.20, 0.30)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: Sequence[str], root: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    mass = weights[order]
    cumulative = np.cumsum(mass) / float(np.sum(mass))
    return float(ordered[min(int(np.searchsorted(cumulative, quantile, side="left")), len(ordered) - 1)])


def distribution_metrics(
    risks: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    betas: Sequence[float] = BETAS,
) -> dict[str, Any]:
    clean = {str(key): float(value) for key, value in risks.items() if math.isfinite(float(value))}
    if not clean:
        raise ValueError("At least one finite slice risk is required.")
    if weights is None:
        normalized = {key: 1.0 / len(clean) for key in clean}
    else:
        raw = {key: float(weights[key]) for key in clean}
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("Deployment weights must have positive mass.")
        normalized = {key: value / total for key, value in raw.items()}
    values = np.asarray([clean[key] for key in clean], dtype=float)
    masses = np.asarray([normalized[key] for key in clean], dtype=float)
    mean = float(np.sum(values * masses))
    variance = float(np.sum(masses * np.square(values - mean)))
    median = _weighted_quantile(values, masses, 0.5)
    mad = _weighted_quantile(np.abs(values - median), masses, 0.5)
    q25 = _weighted_quantile(values, masses, 0.25)
    q75 = _weighted_quantile(values, masses, 0.75)
    profile = compute_geobwer_profile(clean, betas=betas, deployment_weights=normalized)
    primary = min(profile, key=lambda item: abs(item.beta - 0.10))
    return {
        "slice_count": len(clean),
        "mean_risk": mean,
        "weighted_variance": variance,
        "weighted_std": math.sqrt(max(variance, 0.0)),
        "weighted_sd": math.sqrt(max(variance, 0.0)),
        "weighted_median": median,
        "weighted_mad": mad,
        "weighted_iqr": q75 - q25,
        "min_risk": float(np.min(values)),
        "max_risk": float(np.max(values)),
        "max_min_gap": float(np.max(values) - np.min(values)),
        "worst_mean_gap": primary.worst_group_gap,
        "worst_minus_mean": primary.worst_group_gap,
        "tail_risk_beta_0_10": primary.tail_risk,
        "geobwer_beta_0_10": primary.bwer,
        "geobwer": primary.bwer,
        "tail_effective_groups_beta_0_10": primary.allocation.tail_effective_groups,
        "tail_regime_beta_0_10": primary.allocation.tail_regime,
        "profile": profile,
        "weights": normalized,
    }


def _aggregate(
    rows: Iterable[Mapping[str, Any]], group_columns: Sequence[str], risk_column: str
) -> tuple[dict[str, float], dict[str, int]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        risk = _float(row.get(risk_column))
        parts = [str(row.get(column, "")).strip() for column in group_columns]
        if math.isfinite(risk) and all(parts):
            values[" | ".join(parts)].append(risk)
    return (
        {key: float(np.mean(group)) for key, group in values.items()},
        {key: len(group) for key, group in values.items()},
    )


def _append_panel(
    slice_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    dataset: str,
    run_id: str,
    model_family: str,
    mode: str,
    axis: str,
    risks: Mapping[str, float],
    supports: Mapping[str, int],
    min_support: int,
    evidence_role: str = "valid_descriptive",
) -> None:
    eligible = {key: value for key, value in risks.items() if supports.get(key, 0) >= min_support}
    if not eligible:
        return
    metrics = distribution_metrics(eligible)
    profile = metrics.pop("profile")
    weights = metrics.pop("weights")
    primary = min(profile, key=lambda item: abs(item.beta - 0.10))
    selected = primary.allocation.selected_mass_dict()
    summary_rows.append(
        {
            "dataset": dataset,
            "run_id": run_id,
            "model_family": model_family,
            "mode": mode,
            "slice_axis": axis,
            "min_support": min_support,
            "evidence_role": evidence_role,
            "deployment_weighting": "equal_fixed_slice_universe",
            **metrics,
        }
    )
    for point in profile:
        summary_rows.append(
            {
                "dataset": dataset,
                "run_id": run_id,
                "model_family": model_family,
                "mode": mode,
                "slice_axis": axis,
                "min_support": min_support,
                "evidence_role": "beta_profile",
                "deployment_weighting": "equal_fixed_slice_universe",
                "beta": point.beta,
                "mean_risk": point.mean_risk,
                "tail_risk": point.tail_risk,
                "geobwer": point.bwer,
                "tail_effective_groups": point.allocation.tail_effective_groups,
                "tail_regime": point.allocation.tail_regime,
            }
        )
    ranked = sorted(risks, key=lambda key: (-risks[key], key))
    for rank, key in enumerate(ranked, start=1):
        support = int(supports.get(key, 0))
        valid = key in eligible
        slice_rows.append(
            {
                "dataset": dataset,
                "run_id": run_id,
                "model_family": model_family,
                "mode": mode,
                "slice_axis": axis,
                "slice_value": key,
                "risk": risks[key],
                "support": support,
                "min_support": min_support,
                "eligible_for_primary_metric": valid,
                "descriptive_rank_all_slices": rank,
                "deployment_weight": weights.get(key, ""),
                "tail_selected_mass_beta_0_10": selected.get(key, "") if valid else "",
                "is_tail_beta_0_10": bool(valid and selected.get(key, 0.0) > 0.0),
                "tail_contribution_beta_0_10": (
                    selected.get(key, 0.0) / 0.10 * risks[key] if valid else ""
                ),
                "presentation_role": "primary_eligible" if valid else "descriptive_low_support",
            }
        )


def _load_fmow(root: Path, slices: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    sources = [("dofav2", root / "fmow/dofa/formal_audit_table.csv")]
    sources.extend(
        (f"resnet50_seed_{seed}", root / f"fmow/resnet/seed_{seed}/formal_audit_table.csv")
        for seed in (42, 73, 101)
    )
    for run_id, path in sources:
        if not path.exists():
            continue
        rows = [row for row in _read_csv(path) if str(row.get("split", "")).lower() == "test"]
        for axis, columns in (
            ("country", ("country",)),
            ("country_x_class", ("country", "class_label")),
            ("region_x_class", ("region", "class_label")),
        ):
            risks, supports = _aggregate(rows, columns, "risk")
            _append_panel(
                slices, summaries, dataset="fMoW-Sentinel", run_id=run_id,
                model_family="dofav2" if run_id == "dofav2" else "resnet50",
                mode="S2", axis=axis, risks=risks, supports=supports,
                min_support=20,
            )


def _load_sen1(root: Path, slices: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    path = root / "sen1/event_level_metrics.csv"
    if not path.exists():
        return
    rows = [row for row in _read_csv(path) if row.get("split") == "combined_held_out"]
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[str(row["model"])].append(row)
    for run_id, items in by_run.items():
        risks = {str(row["event_id"]): _float(row["mean_chip_iou_risk"]) for row in items}
        supports = {str(row["event_id"]): int(float(row["auditable_sample_count"])) for row in items}
        first = items[0]
        _append_panel(
            slices, summaries, dataset="Sen1Floods11", run_id=run_id,
            model_family=str(first.get("family", "")), mode=str(first.get("mode", "")),
            axis="event", risks=risks, supports=supports, min_support=1,
        )


def _npz_label_panel(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    with np.load(path, allow_pickle=False) as archive:
        probabilities = np.asarray(archive["probabilities"], dtype=float)
        targets = np.asarray(archive["targets"], dtype=np.int8)
        thresholds = np.asarray(archive["thresholds"], dtype=float)
        names = [str(value) for value in archive["class_names"]]
    predictions = probabilities >= thresholds.reshape(1, -1)
    errors = np.mean(predictions != targets, axis=0)
    return dict(zip(names, errors.astype(float))), {name: len(targets) for name in names}


def _parse_reben_name(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    family, mode, seed = stem.split("__")
    return family, mode, seed.replace("seed_", "")


def _load_reben(root: Path, slices: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    if not root.exists():
        return
    for path in sorted(root.glob("*.npz")):
        family, mode, seed = _parse_reben_name(path)
        risks, supports = _npz_label_panel(path)
        _append_panel(
            slices, summaries, dataset="reBEN", run_id=path.stem,
            model_family=family, mode=mode, axis="label",
            risks=risks, supports=supports, min_support=1,
        )


def _load_alphaearth(root: Path, slices: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    metadata_path = root / "alphaearth/alphaearth_eval_metadata_compact.csv"
    encoded_path = root / "alphaearth/test_probabilities.npz.b64"
    if not metadata_path.exists() or not encoded_path.exists():
        return
    metadata = {row["sample_id"]: row for row in _read_csv(metadata_path)}
    archive = np.load(io.BytesIO(base64.b64decode(encoded_path.read_bytes())), allow_pickle=False)
    sample_ids = [str(value) for value in archive["sample_id"]]
    probabilities = np.asarray(archive["probabilities"], dtype=float)
    targets = np.asarray(archive["targets"], dtype=int)
    predictions = np.argmax(probabilities, axis=1)
    rows: list[dict[str, Any]] = []
    for sample_id, target, prediction in zip(sample_ids, targets, predictions):
        row = metadata.get(sample_id)
        if row is None:
            continue
        rows.append({**row, "risk": float(int(target != prediction))})
    for axis, columns in (
        ("spatial_block_x_landcover", ("spatial_block_id", "worldcover_class_name")),
        ("country_x_landcover", ("country_iso3", "worldcover_class_name")),
    ):
        risks, supports = _aggregate(rows, columns, "risk")
        _append_panel(
            slices, summaries, dataset="AlphaEarth", run_id="alphaearth_linear_probe",
            model_family="alphaearth", mode="embedding_probe", axis=axis,
            risks=risks, supports=supports, min_support=5,
            evidence_role="reference_map_agreement_descriptive",
        )


def build_slice_distribution_panel(snapshot_v050: Path, reben_npz: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slices: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    _load_fmow(snapshot_v050, slices, summaries)
    _load_sen1(snapshot_v050, slices, summaries)
    _load_reben(reben_npz, slices, summaries)
    _load_alphaearth(snapshot_v050, slices, summaries)
    return slices, summaries


def synthetic_counterexamples() -> list[dict[str, Any]]:
    scenarios = {
        "same_variance_different_upper_tail_A": [0.10, 0.10, 0.10, 0.10, 0.30],
        "same_variance_different_upper_tail_B": [0.00, 0.20, 0.20, 0.20, 0.20],
        "same_geobwer_different_body_A": [0.10, 0.10, 0.10, 0.10, 0.30],
        "same_geobwer_different_body_B": [0.20, 0.20, 0.20, 0.20, 0.40],
        "exceptionally_good_slice": [0.00, 0.30, 0.30, 0.30, 0.30],
        "uniform_reference": [0.20, 0.20, 0.20, 0.20, 0.20],
        "levelling_down_before": [0.10, 0.10, 0.10, 0.10, 0.40],
        "levelling_down_after": [0.35, 0.35, 0.35, 0.35, 0.40],
    }
    rows: list[dict[str, Any]] = []
    for name, values in scenarios.items():
        metrics = distribution_metrics({f"g{i}": value for i, value in enumerate(values)})
        metrics.pop("profile")
        metrics.pop("weights")
        rows.append({"scenario": name, "risks": json.dumps(values), **metrics})
    return rows


def _rank(values: Sequence[float], *, lower_is_better: bool) -> list[float]:
    array = np.asarray(values, dtype=float)
    key = array if lower_is_better else -array
    order = np.argsort(key, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and key[order[end]] == key[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return ranks.tolist()


def build_terramind_cross_task(snapshot_v050: Path, distribution: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    reben_path = snapshot_v050 / "reben/unified_27run_metrics.csv"
    if reben_path.exists():
        rows = _read_csv(reben_path)
        by_peer: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_peer[(row["mode"], row["seed"])].append(row)
        for (mode, seed), peers in by_peer.items():
            mean_values = [_float(row["deployment_mean_risk"]) for row in peers]
            bwer_values = [_float(row["geobwer"]) for row in peers]
            mean_ranks = _rank(mean_values, lower_is_better=True)
            bwer_ranks = _rank(bwer_values, lower_is_better=True)
            for row, mean_rank, bwer_rank in zip(peers, mean_ranks, bwer_ranks):
                if row["family"] == "terramind":
                    output.append({
                        "dataset": "reBEN", "task": "multi_label_classification",
                        "run_id": row["run_id"], "mode": mode, "seed": seed,
                        "mean_risk": row["deployment_mean_risk"], "tail_risk": row["tail_risk"],
                        "geobwer": row["geobwer"], "aggregate_score": row["macro_ap"],
                        "mean_risk_rank_within_task_mode_seed": mean_rank,
                        "geobwer_rank_within_task_mode_seed": bwer_rank,
                        "peer_count": len(peers), "comparison_role": "within_task_rank_only",
                    })
    sen1_metrics_path = snapshot_v050 / "sen1/unified_19model_metrics.csv"
    if sen1_metrics_path.exists():
        metrics = [row for row in _read_csv(sen1_metrics_path) if row["split"] == "combined_held_out" and row["comparison_role"] == "same_grid_primary_panel"]
        event_summary = {
            (str(row["run_id"]), str(row["slice_axis"])): row
            for row in distribution
            if row.get("dataset") == "Sen1Floods11" and row.get("evidence_role") == "valid_descriptive"
        }
        by_peer: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in metrics:
            by_peer[(row["mode"], row["seed"])].append(row)
        for (mode, seed), peers in by_peer.items():
            means = [_float(row["mean_chip_iou_risk"]) for row in peers]
            gaps = [_float(event_summary.get((row["model"], "event"), {}).get("geobwer_beta_0_10")) for row in peers]
            mean_ranks = _rank(means, lower_is_better=True)
            gap_ranks = _rank(gaps, lower_is_better=True)
            for row, mean_rank, gap_rank, gap in zip(peers, mean_ranks, gap_ranks, gaps):
                if row["family"] == "terramind_v1_base":
                    summary = event_summary.get((row["model"], "event"), {})
                    output.append({
                        "dataset": "Sen1Floods11", "task": "segmentation",
                        "run_id": row["model"], "mode": mode, "seed": seed,
                        "mean_risk": row["mean_chip_iou_risk"],
                        "tail_risk": summary.get("tail_risk_beta_0_10", ""),
                        "geobwer": gap, "aggregate_score": row["mean_chip_flood_iou"],
                        "mean_risk_rank_within_task_mode_seed": mean_rank,
                        "geobwer_rank_within_task_mode_seed": gap_rank,
                        "peer_count": len(peers), "comparison_role": "within_task_rank_only",
                    })
    return output


def _plots(slice_rows: Sequence[Mapping[str, Any]], output: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figure_dir = ensure_dir(output / "figures")
    paths: list[Path] = []
    primary_axes = {
        "fMoW-Sentinel": "country",
        "Sen1Floods11": "event",
        "reBEN": "label",
        "AlphaEarth": "country_x_landcover",
    }
    def render(dataset: str, axis: str, relevant: list[Mapping[str, Any]], suffix: str = "") -> list[Path]:
        from string import ascii_uppercase

        generated: list[Path] = []
        runs = sorted({str(row["run_id"]) for row in relevant})
        if not runs:
            return generated
        columns = min(3, len(runs))
        rows_n = int(math.ceil(len(runs) / columns))
        fig, axes = plt.subplots(rows_n, columns, figsize=(5.0 * columns, 3.25 * rows_n), squeeze=False)
        for panel_index, (ax, run_id) in enumerate(zip(axes.flat, runs)):
            items = sorted([row for row in relevant if row["run_id"] == run_id], key=lambda row: _float(row["risk"]))
            categories = (
                ("descriptive_low_support", "#999999", "x", "low support"),
                ("primary_eligible", "#0072B2", "o", "eligible"),
                ("tail", "#D55E00", "^", "GeoBWER tail"),
            )
            for category, color, marker, label in categories:
                selected = [
                    (index, row) for index, row in enumerate(items)
                    if (category == "tail" and row["is_tail_beta_0_10"])
                    or (category == "primary_eligible" and row["eligible_for_primary_metric"] and not row["is_tail_beta_0_10"])
                    or (category == "descriptive_low_support" and not row["eligible_for_primary_metric"])
                ]
                if selected:
                    ax.scatter(
                        [index for index, _ in selected],
                        [_float(row["risk"]) for _, row in selected],
                        c=color, marker=marker,
                        s=[16 + 7 * math.log1p(int(row["support"])) for _, row in selected],
                        alpha=0.85, label=label,
                    )
            eligible = [_float(row["risk"]) for row in items if row["eligible_for_primary_metric"]]
            if eligible:
                ax.axhline(float(np.mean(eligible)), color="#222222", linewidth=1, linestyle="--", label="eligible mean")
            ax.set_title(run_id, fontsize=8)
            ax.set_xlabel("Slices ordered by risk")
            ax.set_ylabel("Risk")
            ax.text(-0.12, 1.04, ascii_uppercase[panel_index % len(ascii_uppercase)], transform=ax.transAxes, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.16)
        for ax in axes.flat[len(runs):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.982))
        title_suffix = f" — {suffix}" if suffix else ""
        fig.suptitle(f"{dataset}: full {axis} slice distribution{title_suffix}", y=0.999, fontsize=12)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
        safe_suffix = f"_{suffix.lower().replace(' ', '_').replace('+', 'plus')}" if suffix else ""
        stem = figure_dir / f"{dataset.lower().replace('-', '_')}_{axis}_distribution{safe_suffix}"
        for extension in ("png", "pdf"):
            path = stem.with_suffix(f".{extension}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            generated.append(path)
        plt.close(fig)
        return generated

    for dataset, axis in primary_axes.items():
        relevant = [row for row in slice_rows if row["dataset"] == dataset and row["slice_axis"] == axis]
        paths.extend(render(dataset, axis, relevant))
        families = sorted({str(row["model_family"]) for row in relevant})
        if len({str(row["run_id"]) for row in relevant}) > 9:
            for family in families:
                family_rows = [row for row in relevant if str(row["model_family"]) == family]
                paths.extend(render(dataset, axis, family_rows, family))
    return paths


def audit_four_task_distribution_coverage(
    summaries: Sequence[Mapping[str, Any]],
    slices: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Path]:
    expected = {
        "fMoW-Sentinel": "classification",
        "Sen1Floods11": "segmentation",
        "reBEN": "multilabel_classification",
        "AlphaEarth": "landcover_agreement",
    }
    primary = [row for row in summaries if row.get("evidence_role") != "beta_profile"]
    coverage_rows = []
    for dataset, task in expected.items():
        panels = [row for row in primary if row.get("dataset") == dataset]
        distribution = [row for row in slices if row.get("dataset") == dataset]
        metric_complete = bool(panels) and all(
            math.isfinite(_float(row.get(column)))
            for row in panels
            for column in ("weighted_sd", "worst_mean_gap", "geobwer_beta_0_10")
        )
        aliases_complete = bool(panels) and all(
            math.isfinite(_float(row.get(column)))
            for row in panels
            for column in ("weighted_sd", "weighted_sd", "worst_minus_mean", "geobwer")
        )
        coverage_rows.append({
            "dataset": dataset, "task": task, "panel_count": len(panels),
            "slice_row_count": len(distribution),
            "slice_axes": ";".join(sorted({str(row.get("slice_axis")) for row in distribution})),
            "weighted_sd_complete": metric_complete,
            "worst_minus_mean_complete": metric_complete,
            "geobwer_complete": metric_complete,
            "canonical_aliases_complete": aliases_complete,
            "full_slice_distribution_complete": bool(distribution),
            "primary_eligible_slice_count": sum(str(row.get("presentation_role")) == "primary_eligible" for row in distribution),
            "descriptive_low_support_slice_count": sum(str(row.get("presentation_role")) == "descriptive_low_support" for row in distribution),
            "status": "pass" if metric_complete and aliases_complete and bool(distribution) else "fail",
        })
    coverage_path = output / "four_task_distribution_coverage.csv"
    write_csv(coverage_path, coverage_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = {"fMoW-Sentinel": "#0072B2", "Sen1Floods11": "#D55E00", "reBEN": "#009E73", "AlphaEarth": "#CC79A7"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for dataset in expected:
        rows = [row for row in primary if row.get("dataset") == dataset]
        axes[0].scatter([_float(row["weighted_sd"]) for row in rows], [_float(row["geobwer_beta_0_10"]) for row in rows], color=palette[dataset], label=dataset, alpha=0.72, s=24)
        axes[1].scatter([_float(row["worst_mean_gap"]) for row in rows], [_float(row["geobwer_beta_0_10"]) for row in rows], color=palette[dataset], label=dataset, alpha=0.72, s=24)
    axes[0].set(xlabel="Weighted SD", ylabel="GeoBWER (beta=0.10)")
    axes[1].set(xlabel="Worst slice − mean risk", ylabel="GeoBWER (beta=0.10)")
    for index, ax in enumerate(axes):
        ax.text(-0.13, 1.04, chr(ord("A") + index), transform=ax.transAxes, fontweight="bold")
        ax.grid(alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    figure_dir = ensure_dir(output / "figures")
    figures = []
    for suffix in (".png", ".pdf"):
        path = figure_dir / f"four_task_dispersion_comparator{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        figures.append(path)
    plt.close(fig)
    audit_path = output / "four_task_distribution_coverage.json"
    audit_path.write_text(json.dumps({
        "schema": f"{SCHEMA}.four_task_distribution_coverage.v1",
        "status": "pass" if all(row["status"] == "pass" for row in coverage_rows) else "fail",
        "expected_tasks": expected, "rows": coverage_rows,
        "interpretation": "Weighted SD, worst-minus-mean, and GeoBWER are complementary descriptive summaries; the full slice table remains the distribution-level evidence.",
    }, indent=2), encoding="utf-8")
    return {"table": coverage_path, "audit": audit_path, "figures": figures[0].parent}


def freeze_evidence(root: Path, output: Path, source_paths: Sequence[Path]) -> dict[str, Path]:
    inventory = []
    for path in source_paths:
        absolute = path if path.is_absolute() else root / path
        inventory.append({
            "path": str(path).replace("\\", "/"),
            "exists": absolute.exists(),
            "size_bytes": absolute.stat().st_size if absolute.exists() else "",
            "sha256": _sha256(absolute) if absolute.exists() and absolute.is_file() else "",
        })
    git_status = _git_value(("status", "--short"), root)
    payload = {
        "schema": f"{SCHEMA}.evidence_freeze",
        "git_head": _git_value(("rev-parse", "HEAD"), root),
        "git_status_at_generation": git_status,
        "dirty_worktree_disclosure": bool(git_status and git_status != "unavailable"),
        "evidence_policy": "limitations on inference do not suppress valid point estimates",
        "source_inventory": inventory,
        "scope": [1, 2, 3, 4, 5, 6, 7],
        "out_of_scope_not_started": list(range(8, 18)),
    }
    manifest = output / "evidence_freeze_manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory_path = output / "evidence_freeze_inventory.csv"
    write_csv(inventory_path, inventory)
    return {"manifest": manifest, "inventory": inventory_path}


def _write_report(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
    cross_task: Sequence[Mapping[str, Any]],
    figure_paths: Sequence[Path],
) -> Path:
    primary = [row for row in summaries if row.get("evidence_role") != "beta_profile"]
    lines = [
        "# Optimization 1–7 CPU post-processing report",
        "",
        f"- Schema: `{SCHEMA}`",
        f"- Recomputed distribution panels: `{len(primary)}`",
        f"- TerraMind cross-task rows: `{len(cross_task)}`",
        f"- Figures: `{len(figure_paths)}`",
        "- Inference policy: valid descriptive estimates and ranks are retained; unsupported formal inference is labelled, not deleted.",
        "",
        "## Metric interpretation",
        "",
        "Weighted SD is a symmetric full-distribution comparator. GeoBWER is the deployment-weighted upper-tail excess risk. They are reported together and are not interchangeable.",
        "",
        "## Local completeness",
        "",
        "- fMoW country/interactions, Sen1 event interactions, reBEN label interactions, and AlphaEarth block/class interactions were recomputed from the frozen local snapshot.",
        "- reBEN country×label requires the frozen metadata JSONL, which is intentionally a required input of the large-asset runner.",
        "- Label-budget training requires frozen TerraMind embeddings; no prediction-only artifact is misrepresented as an embedding cache.",
        "- Paired S2→S1 shift requires one S2-trained head applied unchanged to aligned S2 and S1 test embeddings; separately trained modality heads are explicitly rejected.",
    ]
    path = output / "optimization_1_7_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_optimization_1_7(
    root: str | Path,
    output_dir: str | Path,
    *,
    snapshot_v050: str | Path = "work/drive_snapshot_v050",
    reben_npz: str | Path = "work/drive_snapshot_v060/reben_probability_npz",
) -> dict[str, Path]:
    project = Path(root).resolve()
    output = ensure_dir(output_dir)
    snapshot = project / snapshot_v050
    probabilities = project / reben_npz
    source_paths = [
        Path("configs/analysis/evidence_status_v060.json"),
        Path("configs/analysis/canonical_evidence_registry_v1.yaml"),
        Path("configs/analysis/unified_audit_registry.yaml"),
        Path("configs/analysis/optimization_1_7_v1.yaml"),
        Path("configs/analysis/optimization_1_7_drive_reconciliation_v1.json"),
        Path("src/rsfm_fairness_audit/optimization_phase1.py"),
        Path("src/rsfm_fairness_audit/reben_phase1_runners.py"),
        Path("src/rsfm_fairness_audit/reben_phase1_postprocess.py"),
        Path("src/rsfm_fairness_audit/paired_probability_diagnostics.py"),
        Path("src/rsfm_fairness_audit/reben_terramind_campaign.py"),
        Path("scripts/analysis/finalize_optimization_1_7.py"),
        Path("docs/reproduction/optimization_1_7.md"),
        Path("reports/optimization_1_7_scientific_review_2026_08_16.md"),
        Path(snapshot_v050) / "sen1/unified_19model_metrics.csv",
        Path(snapshot_v050) / "sen1/event_level_metrics.csv",
        Path(snapshot_v050) / "sen1/source_contract.json",
        Path(snapshot_v050) / "sen1/postprocess_manifest.json",
        Path(snapshot_v050) / "reben/unified_27run_metrics.csv",
        Path(snapshot_v050) / "fmow/dofa/formal_audit_table.csv",
        Path(snapshot_v050) / "fmow/resnet/seed_42/formal_audit_table.csv",
        Path(snapshot_v050) / "fmow/resnet/seed_73/formal_audit_table.csv",
        Path(snapshot_v050) / "fmow/resnet/seed_101/formal_audit_table.csv",
        Path(snapshot_v050) / "alphaearth/alphaearth_eval_metadata_compact.csv",
        Path(snapshot_v050) / "alphaearth/test_probabilities.npz.b64",
    ]
    source_paths.extend(
        path.relative_to(project) if path.is_relative_to(project) else path
        for path in sorted(probabilities.glob("*.npz"))
    )
    artifacts = freeze_evidence(project, output, source_paths)
    slices, summaries = build_slice_distribution_panel(snapshot, probabilities)
    cross_task = build_terramind_cross_task(snapshot, summaries)
    interactions = [
        row for row in slices
        if row["slice_axis"] in {"country_x_class", "region_x_class", "event", "label", "spatial_block_x_landcover", "country_x_landcover"}
    ]
    artifact_paths = {
        "slice_distribution": output / "full_slice_distribution.csv",
        "distribution_metrics": output / "distribution_metric_comparison.csv",
        "synthetic_counterexamples": output / "metric_counterexamples.csv",
        "terramind_cross_task": output / "terramind_cross_task_analysis.csv",
        "compound_interactions": output / "compound_interaction_atlas.csv",
        "execution_status": output / "optimization_1_7_execution_status.csv",
    }
    write_csv(artifact_paths["slice_distribution"], slices)
    write_csv(artifact_paths["distribution_metrics"], summaries)
    counterexamples = synthetic_counterexamples()
    write_csv(artifact_paths["synthetic_counterexamples"], counterexamples)
    write_csv(artifact_paths["terramind_cross_task"], cross_task)
    write_csv(artifact_paths["compound_interactions"], interactions)
    coverage = audit_four_task_distribution_coverage(summaries, slices, output)
    status = [
        {"item": 1, "name": "evidence_freeze", "code_status": "implemented", "execution_status": "completed_local"},
        {"item": 2, "name": "distribution_and_variance_comparator", "code_status": "implemented", "execution_status": "completed_local"},
        {"item": 3, "name": "full_slice_visualization", "code_status": "implemented", "execution_status": "completed_local"},
        {"item": 4, "name": "terramind_cross_task", "code_status": "implemented", "execution_status": "completed_local"},
        {"item": 5, "name": "compound_interaction_atlas", "code_status": "implemented", "execution_status": "completed_local_with_reben_country_label_pending_metadata"},
        {"item": 6, "name": "nested_label_budget", "code_status": "runner_postprocess_audit_figures_ready", "execution_status": "pending_or_running_large_embedding_cache"},
        {"item": 7, "name": "paired_sensor_shift", "code_status": "formal_preflight_runner_postprocess_audit_figures_ready", "execution_status": "pending_aligned_s1_s2_embedding_cache"},
    ]
    write_csv(artifact_paths["execution_status"], status)
    figures = _plots(slices, output)
    figures.extend(sorted(coverage["figures"].glob("four_task_dispersion_comparator.*")))
    report = _write_report(output, summaries, cross_task, figures)
    primary = [row for row in summaries if row.get("evidence_role") != "beta_profile"]
    examples = {row["scenario"]: row for row in counterexamples}
    gates = {
        "all_frozen_sources_exist": all(row["exists"] for row in json.loads(artifacts["manifest"].read_text(encoding="utf-8"))["source_inventory"]),
        "distribution_panels_nonempty": len(primary) > 0,
        "all_primary_distribution_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in primary
            for key in ("mean_risk", "weighted_std", "worst_mean_gap", "geobwer_beta_0_10")
        ),
        "four_task_metric_and_full_distribution_coverage": json.loads(coverage["audit"].read_text(encoding="utf-8"))["status"] == "pass",
        "full_slice_table_retains_primary_and_low_support": (
            any(row["presentation_role"] == "primary_eligible" for row in slices)
            and any(row["presentation_role"] == "descriptive_low_support" for row in slices)
        ),
        "terramind_cross_task_has_both_tasks": {row["dataset"] for row in cross_task} == {"reBEN", "Sen1Floods11"},
        "same_variance_counterexample_exact": abs(
            examples["same_variance_different_upper_tail_A"]["weighted_variance"]
            - examples["same_variance_different_upper_tail_B"]["weighted_variance"]
        ) <= 1e-12,
        "same_variance_counterexample_changes_geobwer": abs(
            examples["same_variance_different_upper_tail_A"]["geobwer_beta_0_10"]
            - examples["same_variance_different_upper_tail_B"]["geobwer_beta_0_10"]
        ) > 0.05,
        "translation_counterexample_preserves_geobwer": abs(
            examples["same_geobwer_different_body_A"]["geobwer_beta_0_10"]
            - examples["same_geobwer_different_body_B"]["geobwer_beta_0_10"]
        ) <= 1e-12,
        "levelling_down_is_detected": (
            examples["levelling_down_after"]["mean_risk"] > examples["levelling_down_before"]["mean_risk"]
            and examples["levelling_down_after"]["geobwer_beta_0_10"] < examples["levelling_down_before"]["geobwer_beta_0_10"]
        ),
        "publication_figures_include_vector_and_raster": (
            any(path.suffix == ".pdf" for path in figures) and any(path.suffix == ".png" for path in figures)
        ),
    }
    validation_path = output / "optimization_1_7_validation.json"
    validation_path.write_text(json.dumps({
        "schema": f"{SCHEMA}.validation", "passes": all(gates.values()),
        "gates": gates, "counts": {
            "primary_distribution_panels": len(primary), "slice_rows": len(slices),
            "cross_task_rows": len(cross_task), "interaction_rows": len(interactions),
            "figure_files_generated": len(figures), "frozen_source_files": len(source_paths),
        },
    }, indent=2), encoding="utf-8")
    if not all(gates.values()):
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"Optimization 1-7 validation failed: {failed}")
    return {**artifacts, **artifact_paths, "report": report, "validation": validation_path, "coverage_audit": coverage["audit"], "coverage_table": coverage["table"]}


__all__ = [
    "BETAS",
    "audit_four_task_distribution_coverage",
    "build_slice_distribution_panel",
    "build_terramind_cross_task",
    "distribution_metrics",
    "run_optimization_1_7",
    "synthetic_counterexamples",
]
