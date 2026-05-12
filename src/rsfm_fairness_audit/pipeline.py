from __future__ import annotations

from pathlib import Path

from rsfm_fairness_audit.adapters.dummy import DummyDatasetConfig, DummyEODataset, DummyModelAdapter
from rsfm_fairness_audit.embedding import extract_embeddings
from rsfm_fairness_audit.io import ensure_dir, write_csv
from rsfm_fairness_audit.metrics import group_metrics, raw_vs_balanced_gap, summarize_gap
from rsfm_fairness_audit.probes import NearestCentroidProbe
from rsfm_fairness_audit.report import write_static_report
from rsfm_fairness_audit.sampling import balanced_indices
from rsfm_fairness_audit.viz import plot_average_vs_worst, plot_representation_shift, plot_sensor_heatmap


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
        "average_vs_worst": output / "average_vs_worst.png",
        "sensor_heatmap": output / "sensor_fairness_heatmap.png",
        "representation_shift": output / "representation_shift.png",
        "report": output / "report.md",
    }

    write_csv(artifacts["region_matrix"], region_rows)
    write_csv(artifacts["sensor_matrix"], sensor_rows)
    write_csv(artifacts["task_matrix"], task_rows)
    write_csv(artifacts["summary"], summary_rows)
    write_csv(artifacts["gap_table"], gap_rows)
    plot_average_vs_worst(summary_rows, artifacts["average_vs_worst"])
    plot_sensor_heatmap(sensor_rows, artifacts["sensor_heatmap"])
    plot_representation_shift(embeddings, metadata, artifacts["representation_shift"])
    write_static_report(output, summary_rows, gap_rows)
    return artifacts
