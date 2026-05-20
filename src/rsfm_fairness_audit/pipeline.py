from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Mapping

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter, ModelAdapter
from rsfm_fairness_audit.adapters.ben_ge import BenGEDatasetAdapter
from rsfm_fairness_audit.adapters.bigearthnet import BigEarthNetDatasetAdapter
from rsfm_fairness_audit.adapters.croma import CROMAAdapter
from rsfm_fairness_audit.adapters.dofa import DOFAAdapter
from rsfm_fairness_audit.adapters.dummy import DummyDatasetConfig, DummyEODataset, DummyModelAdapter
from rsfm_fairness_audit.adapters.prithvi import PrithviAdapter, PrithviSen1Floods11TLAdapter
from rsfm_fairness_audit.adapters.sen1floods11 import Sen1Floods11DatasetAdapter
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.embedding import extract_embeddings, extract_embeddings_to_chunks
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from rsfm_fairness_audit.memory import log_memory, release_memory
from rsfm_fairness_audit.metrics import classwise_metrics, group_metrics, raw_vs_balanced_gap, summarize_gap
from rsfm_fairness_audit.probes import NearestCentroidProbe, evaluate_probe_suite
from rsfm_fairness_audit.report import write_real_report, write_static_report
from rsfm_fairness_audit.sampling import balanced_indices
from rsfm_fairness_audit.viz import (
    plot_average_vs_worst,
    plot_fairness_map,
    plot_raw_vs_balanced_gap,
    plot_representation_shift,
    plot_sensor_heatmap,
)


def run_dummy_pipeline(output_dir: str | Path, num_samples: int = 240, seed: int = 7) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    dataset = DummyEODataset(DummyDatasetConfig(num_samples=num_samples, seed=seed))
    model = DummyModelAdapter()
    embeddings, labels, metadata = extract_embeddings(dataset, model)

    probe = NearestCentroidProbe().fit(embeddings, labels)
    predictions = probe.predict(embeddings)

    region_rows = group_metrics(labels, predictions, metadata, "region")
    sensor_rows = group_metrics(labels, predictions, metadata, "sensor")
    task_rows = group_metrics(labels, predictions, metadata, "task")

    balanced_idx = balanced_indices(metadata, keys=("region", "label"), seed=seed)
    balanced_metadata = [metadata[index] for index in balanced_idx]
    balanced_labels = labels[balanced_idx]
    balanced_predictions = predictions[balanced_idx]
    balanced_region_rows = group_metrics(balanced_labels, balanced_predictions, balanced_metadata, "region")

    summary_rows = [
        summarize_gap(region_rows, "raw_region_gap"),
        summarize_gap(balanced_region_rows, "balanced_region_gap"),
        summarize_gap(sensor_rows, "raw_sensor_gap"),
        summarize_gap(task_rows, "raw_task_gap"),
    ]
    gap_rows = [raw_vs_balanced_gap(region_rows, balanced_region_rows, "region")]

    artifacts = {
        "region_matrix": output / "fairness_matrix_region.csv",
        "sensor_matrix": output / "fairness_matrix_sensor.csv",
        "task_matrix": output / "fairness_matrix_task.csv",
        "summary": output / "fairness_summary.csv",
        "gap_table": output / "raw_vs_balanced_gap.csv",
        "classwise_metrics": output / "classwise_metrics.csv",
        "probe_comparison": output / "probe_comparison.csv",
        "average_vs_worst": output / "average_vs_worst.png",
        "raw_vs_balanced_gap_plot": output / "raw_vs_balanced_gap.png",
        "sensor_heatmap": output / "sensor_fairness_heatmap.png",
        "representation_shift": output / "representation_shift.png",
        "report": output / "report.md",
    }

    write_csv(artifacts["region_matrix"], region_rows)
    write_csv(artifacts["sensor_matrix"], sensor_rows)
    write_csv(artifacts["task_matrix"], task_rows)
    write_csv(artifacts["summary"], summary_rows)
    write_csv(artifacts["gap_table"], gap_rows)
    write_csv(artifacts["classwise_metrics"], classwise_metrics(labels, predictions))
    write_csv(artifacts["probe_comparison"], evaluate_probe_suite(embeddings, labels))
    plot_average_vs_worst(summary_rows, artifacts["average_vs_worst"])
    plot_raw_vs_balanced_gap(gap_rows, artifacts["raw_vs_balanced_gap_plot"])
    plot_sensor_heatmap(sensor_rows, artifacts["sensor_heatmap"])
    plot_representation_shift(embeddings, metadata, artifacts["representation_shift"])
    write_static_report(output, summary_rows, gap_rows)
    return artifacts


def _prediction_rows(metadata: list[dict], labels: np.ndarray, predictions: np.ndarray) -> list[dict]:
    rows = []
    for index, row in enumerate(metadata):
        rows.append(
            {
                "sample_id": row.get("sample_id", index),
                "label": int(labels[index]),
                "prediction": int(predictions[index]),
                "region": row.get("region", "to_verify"),
                "sensor": row.get("sensor", "to_verify"),
                "correct": int(labels[index] == predictions[index]),
            }
        )
    return rows


def run_real_pipeline(
    dataset: DatasetAdapter,
    model: ModelAdapter,
    output_dir: str | Path,
    dataset_name: str,
    model_name: str,
    seed: int = 13,
    chunk_size: int = 256,
    streaming_embeddings: bool = False,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    batch_size = int(getattr(model, "batch_size", 32))
    if streaming_embeddings:
        embeddings, labels, metadata = extract_embeddings_to_chunks(
            dataset,
            model,
            output / "embedding_chunks",
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
    else:
        embeddings, labels, metadata = extract_embeddings(dataset, model, batch_size=batch_size)
    log_memory("probe training:start")
    probe = NearestCentroidProbe().fit(embeddings, labels)
    log_memory("probe training:done")
    log_memory("probe prediction:start")
    predictions = probe.predict(embeddings)
    log_memory("probe prediction:done")

    log_memory("fairness metrics:start")
    region_rows = group_metrics(labels, predictions, metadata, "region")
    sensor_rows = group_metrics(labels, predictions, metadata, "sensor")
    task_rows = group_metrics(labels, predictions, metadata, "task")

    balanced_idx = balanced_indices(metadata, keys=("region", "label"), seed=seed)
    balanced_metadata = [metadata[index] for index in balanced_idx]
    balanced_region_rows = group_metrics(labels[balanced_idx], predictions[balanced_idx], balanced_metadata, "region")

    summary_rows = [
        summarize_gap(region_rows, "raw_region_gap"),
        summarize_gap(balanced_region_rows, "balanced_region_gap"),
        summarize_gap(sensor_rows, "raw_sensor_gap"),
        summarize_gap(task_rows, "raw_task_gap"),
    ]
    gap_rows = [raw_vs_balanced_gap(region_rows, balanced_region_rows, "region")]
    log_memory("fairness metrics:done")

    artifacts = {
        "embeddings": output / "embeddings.npz",
        "predictions": output / "predictions.csv",
        "region_matrix": output / "fairness_matrix_region.csv",
        "sensor_matrix": output / "fairness_matrix_sensor.csv",
        "task_matrix": output / "fairness_matrix_task.csv",
        "summary": output / "fairness_summary.csv",
        "gap_table": output / "raw_vs_balanced_gap.csv",
        "classwise_metrics": output / "classwise_metrics.csv",
        "probe_comparison": output / "probe_comparison.csv",
        "average_vs_worst": output / "average_vs_worst.png",
        "raw_vs_balanced_gap_plot": output / "raw_vs_balanced_gap.png",
        "sensor_heatmap": output / "sensor_fairness_heatmap.png",
        "representation_shift": output / "representation_shift.png",
        "fairness_map": output / "fairness_map.png",
        "report": output / "report.md",
    }

    log_memory("embeddings.npz writing:start")
    np.savez_compressed(
        artifacts["embeddings"],
        embeddings=embeddings,
        labels=labels,
        predictions=predictions,
        sample_ids=np.asarray([str(row.get("sample_id", index)) for index, row in enumerate(metadata)]),
    )
    log_memory("embeddings.npz writing:done")
    log_memory("predictions.csv writing:start")
    write_csv(artifacts["predictions"], _prediction_rows(metadata, labels, predictions))
    log_memory("predictions.csv writing:done")
    write_csv(artifacts["region_matrix"], region_rows)
    write_csv(artifacts["sensor_matrix"], sensor_rows)
    write_csv(artifacts["task_matrix"], task_rows)
    write_csv(artifacts["summary"], summary_rows)
    write_csv(artifacts["gap_table"], gap_rows)
    write_csv(artifacts["classwise_metrics"], classwise_metrics(labels, predictions))
    log_memory("probe comparison:start")
    write_csv(artifacts["probe_comparison"], evaluate_probe_suite(embeddings, labels))
    log_memory("probe comparison:done")
    log_memory("report/figure generation:start")
    plot_average_vs_worst(summary_rows, artifacts["average_vs_worst"])
    plot_raw_vs_balanced_gap(gap_rows, artifacts["raw_vs_balanced_gap_plot"])
    plot_sensor_heatmap(sensor_rows, artifacts["sensor_heatmap"])
    plot_representation_shift(embeddings, metadata, artifacts["representation_shift"])
    map_generated = plot_fairness_map(metadata, predictions, labels, artifacts["fairness_map"])
    if not map_generated:
        artifacts.pop("fairness_map")
    write_real_report(output, dataset_name, model_name, summary_rows, gap_rows, map_generated)
    log_memory("report/figure generation:done")
    release_memory()
    tables_dir = ensure_dir(output / "tables")
    figures_dir = ensure_dir(output / "figures")
    artifacts["tables_fairness_matrix"] = tables_dir / "fairness_matrix.csv"
    artifacts["tables_raw_vs_balanced_gap"] = tables_dir / "raw_vs_balanced_gap.csv"
    artifacts["tables_classwise_metrics"] = tables_dir / "classwise_metrics.csv"
    artifacts["tables_probe_comparison"] = tables_dir / "probe_comparison.csv"
    artifacts["figures_average_vs_worst_group"] = figures_dir / "average_vs_worst_group.png"
    artifacts["figures_raw_vs_balanced_gap"] = figures_dir / "raw_vs_balanced_gap.png"
    artifacts["figures_fairness_map"] = figures_dir / "fairness_map.png"
    copyfile(artifacts["region_matrix"], artifacts["tables_fairness_matrix"])
    copyfile(artifacts["gap_table"], artifacts["tables_raw_vs_balanced_gap"])
    copyfile(artifacts["classwise_metrics"], artifacts["tables_classwise_metrics"])
    copyfile(artifacts["probe_comparison"], artifacts["tables_probe_comparison"])
    copyfile(artifacts["average_vs_worst"], artifacts["figures_average_vs_worst_group"])
    copyfile(artifacts["raw_vs_balanced_gap_plot"], artifacts["figures_raw_vs_balanced_gap"])
    if "fairness_map" in artifacts:
        copyfile(artifacts["fairness_map"], artifacts["figures_fairness_map"])
    return artifacts


def build_real_adapters(
    dataset_name: str,
    model_name: str,
    data_root: str | Path,
    metadata_path: str | Path | None,
    subset_manifest_path: str | Path | None,
    subset_size: int | None,
    split: str,
    sensor_mode: str,
    dofa_wavelengths: list[float] | None = None,
    allow_torch_hub_download: bool = False,
    model_config: str | Path | None = None,
) -> tuple[DatasetAdapter, ModelAdapter]:
    if dataset_name not in {"bigearthnet", "ben_ge", "sen1floods11"}:
        raise ValueError("Real smoke runs currently implement dataset='bigearthnet', dataset='ben_ge', or dataset='sen1floods11'.")
    if model_name not in {"dofa", "croma", "prithvi", "prithvi_tl_sen1floods11"}:
        raise ValueError(
            "Real smoke runs currently implement model='dofa', model='croma', model='prithvi', "
            "or model='prithvi_tl_sen1floods11'."
        )
    if dataset_name == "bigearthnet":
        dataset = BigEarthNetDatasetAdapter(
            data_root=data_root,
            metadata_path=metadata_path,
            subset_manifest_path=subset_manifest_path,
            subset_size=subset_size,
            split=split,
            sensor_mode=sensor_mode,
        )
    elif dataset_name == "ben_ge":
        dataset = BenGEDatasetAdapter(
            data_root=data_root,
            metadata_path=metadata_path,
            subset_size=subset_size,
            split=split,
            sensor_mode=sensor_mode,
        )
    else:
        dataset = Sen1Floods11DatasetAdapter(
            data_root=data_root,
            metadata_path=metadata_path,
            subset_size=subset_size,
            split=split,
        )
    if model_name == "dofa":
        if model_config is not None:
            config = load_yaml(model_config)
            config.setdefault("input_modality", sensor_mode)
            if dofa_wavelengths is not None:
                config["wavelength_list"] = dofa_wavelengths
            if allow_torch_hub_download:
                config["allow_torch_hub_download"] = True
            model = DOFAAdapter.from_config(config)
        else:
            model = DOFAAdapter(
                sensor_mode=sensor_mode,
                wavelengths=dofa_wavelengths,
                allow_torch_hub_download=allow_torch_hub_download,
            )
    elif model_name == "croma" and model_config is not None:
        config = load_yaml(model_config)
        model = CROMAAdapter.from_config(config)
    elif model_name == "croma":
        model = CROMAAdapter(device="auto", allow_hf_download=False)
    elif model_name == "prithvi_tl_sen1floods11" and model_config is not None:
        config = load_yaml(model_config)
        model = PrithviSen1Floods11TLAdapter.from_config(config)
    elif model_name == "prithvi_tl_sen1floods11":
        model = PrithviSen1Floods11TLAdapter(device="auto", allow_hf_download=False)
    elif model_config is not None:
        config = load_yaml(model_config)
        model = PrithviAdapter.from_config(config)
    else:
        model = PrithviAdapter(device="auto", allow_hf_download=False)
    return dataset, model


def _float_or_nan(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _gap_value(rows: list[dict[str, str]], name: str) -> float:
    if name in {"raw_fairness_gap", "balanced_fairness_gap", "residual_gap_after_balancing"} and rows:
        return _float_or_nan(rows[0], name)
    for row in rows:
        if row.get("gap_name") == name:
            return _float_or_nan(row, "gap")
    return float("nan")


def compare_model_runs(
    runs: Mapping[str, str | Path],
    output_dir: str | Path,
    dataset_name: str = "bigearthnet",
    closure: bool = False,
) -> dict[str, Path]:
    if any((Path(run_dir_value) / "event_segmentation_metrics.csv").exists() for run_dir_value in runs.values()):
        from rsfm_fairness_audit.segmentation_comparison import compare_segmentation_runs

        return compare_segmentation_runs(runs, output_dir, dataset_name=dataset_name, closure=closure)
    output = ensure_dir(output_dir)
    tables_dir = ensure_dir(output / "tables")
    figures_dir = ensure_dir(output / "figures")
    rows: list[dict[str, object]] = []
    for model_name, run_dir_value in runs.items():
        run_dir = Path(run_dir_value)
        summary_path = run_dir / "fairness_summary.csv"
        gap_path = run_dir / "raw_vs_balanced_gap.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing fairness summary for {model_name}: {summary_path}")
        summary_rows = read_csv_rows(summary_path)
        gap_rows = read_csv_rows(gap_path) if gap_path.exists() else []
        raw_region = next((row for row in summary_rows if row.get("gap_name") == "raw_region_gap"), {})
        rows.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "run_dir": str(run_dir),
                "subset_size": "",
                "average_performance": _float_or_nan(raw_region, "average_performance"),
                "worst_group": raw_region.get("worst_group", ""),
                "worst_group_performance": _float_or_nan(raw_region, "worst_region_performance"),
                "best_worst_gap": _float_or_nan(raw_region, "best_worst_gap"),
                "balanced_gap": _gap_value(gap_rows, "balanced_fairness_gap"),
                "sensor_group_notes": "lc-col BigEarthNet Phase 2A uses S2-only optical inputs; true CROMA sensor fairness is Phase 2B.",
            }
        )
    comparison_path = tables_dir / "model_comparison.csv"
    write_csv(comparison_path, rows)
    figure_path = figures_dir / "dofa_vs_croma_average_vs_worst.png"
    _plot_model_comparison(rows, figure_path)
    return {"model_comparison": comparison_path, "comparison_figure": figure_path}


def _plot_model_comparison(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        ax.scatter(row["average_performance"], row["worst_group_performance"], s=80)
        ax.annotate(str(row["model"]), (row["average_performance"], row["worst_group_performance"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Average performance")
    ax.set_ylabel("Worst-group performance")
    ax.set_title("DOFA vs CROMA Average vs Worst Group")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compare_sensor_mode_runs(
    runs: Mapping[str, str | Path],
    output_dir: str | Path,
    dataset_name: str = "ben_ge",
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    tables_dir = ensure_dir(output / "tables")
    figures_dir = ensure_dir(output / "figures")
    rows: list[dict[str, object]] = []
    for sensor_mode, run_dir_value in runs.items():
        run_dir = Path(run_dir_value)
        summary_rows = read_csv_rows(run_dir / "fairness_summary.csv")
        gap_rows = read_csv_rows(run_dir / "raw_vs_balanced_gap.csv")
        raw_region = next((row for row in summary_rows if row.get("gap_name") == "raw_region_gap"), {})
        rows.append(
            {
                "sensor_mode": sensor_mode,
                "dataset": dataset_name,
                "run_dir": str(run_dir),
                "average_performance": _float_or_nan(raw_region, "average_performance"),
                "worst_group": raw_region.get("worst_group", ""),
                "worst_group_performance": _float_or_nan(raw_region, "worst_region_performance"),
                "best_worst_gap": _float_or_nan(raw_region, "best_worst_gap"),
                "balanced_gap": _gap_value(gap_rows, "balanced_fairness_gap"),
            }
        )
    table_path = tables_dir / "sensor_mode_comparison.csv"
    write_csv(table_path, rows)
    heatmap_path = figures_dir / "sensor_fairness_heatmap.png"
    scatter_path = figures_dir / "average_vs_worst_sensor_mode.png"
    _plot_sensor_mode_heatmap(rows, heatmap_path)
    _plot_sensor_mode_scatter(rows, scatter_path)
    report_path = output / "report.md"
    _write_sensor_mode_report(report_path, rows)
    return {
        "sensor_mode_comparison": table_path,
        "sensor_fairness_heatmap": heatmap_path,
        "average_vs_worst_sensor_mode": scatter_path,
        "report": report_path,
    }


def _plot_sensor_mode_heatmap(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    modes = [str(row["sensor_mode"]) for row in rows]
    values = np.asarray(
        [[float(row["average_performance"]), float(row["worst_group_performance"]), float(row["best_worst_gap"])] for row in rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(rows))))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(modes)), labels=modes)
    ax.set_xticks(np.arange(3), labels=["average", "worst", "gap"])
    fig.colorbar(image, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_sensor_mode_scatter(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    used_offsets: list[tuple[float, float]] = []
    offsets = [(6, 6), (6, 18), (18, 6), (-42, 10), (10, -18)]
    for index, row in enumerate(rows):
        x = float(row["average_performance"])
        y = float(row["worst_group_performance"])
        ax.scatter(x, y, s=80)
        nearby = sum(1 for px, py in used_offsets if abs(px - x) < 0.035 and abs(py - y) < 0.035)
        offset = offsets[(index + nearby) % len(offsets)]
        ax.annotate(str(row["sensor_mode"]), (x, y), xytext=offset, textcoords="offset points")
        used_offsets.append((x, y))
    ax.plot([0, 1], [0, 1], color="0.7", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Average performance")
    ax.set_ylabel("Worst-group performance")
    ax.set_title("CROMA Sensor-Mode Fairness")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_sensor_mode_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CROMA BEN-GE-800 Sensor-Mode Comparison",
        "",
        "Phase 2B compares CROMA SAR-only, optical-only, and S1+S2 fusion on paired BEN-GE-800 samples.",
        "",
        "This 64-sample BEN-GE-800 run is a smoke validation only. Extreme worst-group or gap values should not be interpreted as paper-grade fairness conclusions.",
        "",
        "| sensor_mode | average | worst_group | worst | gap | balanced_gap |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {sensor_mode} | {average_performance:.4f} | {worst_group} | {worst_group_performance:.4f} | {best_worst_gap:.4f} | {balanced_gap:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
