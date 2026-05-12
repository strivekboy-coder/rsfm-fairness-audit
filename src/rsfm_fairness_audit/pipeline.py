from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import numpy as np

from rsfm_fairness_audit.adapters.base import DatasetAdapter, ModelAdapter
from rsfm_fairness_audit.adapters.bigearthnet import BigEarthNetDatasetAdapter
from rsfm_fairness_audit.adapters.dofa import DOFAAdapter
from rsfm_fairness_audit.adapters.dummy import DummyDatasetConfig, DummyEODataset, DummyModelAdapter
from rsfm_fairness_audit.config import load_yaml
from rsfm_fairness_audit.embedding import extract_embeddings, extract_embeddings_to_chunks
from rsfm_fairness_audit.io import ensure_dir, write_csv
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
    probe = NearestCentroidProbe().fit(embeddings, labels)
    predictions = probe.predict(embeddings)

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

    np.savez_compressed(
        artifacts["embeddings"],
        embeddings=embeddings,
        labels=labels,
        predictions=predictions,
        sample_ids=np.asarray([str(row.get("sample_id", index)) for index, row in enumerate(metadata)]),
    )
    write_csv(artifacts["predictions"], _prediction_rows(metadata, labels, predictions))
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
    map_generated = plot_fairness_map(metadata, predictions, labels, artifacts["fairness_map"])
    if not map_generated:
        artifacts.pop("fairness_map")
    write_real_report(output, dataset_name, model_name, summary_rows, gap_rows, map_generated)
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
    if dataset_name != "bigearthnet":
        raise ValueError("Milestone 3 only implements dataset='bigearthnet'.")
    if model_name != "dofa":
        raise ValueError("Milestone 3 only implements model='dofa'.")
    dataset = BigEarthNetDatasetAdapter(
        data_root=data_root,
        metadata_path=metadata_path,
        subset_manifest_path=subset_manifest_path,
        subset_size=subset_size,
        split=split,
        sensor_mode=sensor_mode,
    )
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
    return dataset, model
