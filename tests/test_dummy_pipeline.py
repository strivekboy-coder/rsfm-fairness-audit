from __future__ import annotations

from rsfm_fairness_audit.adapters.dummy import DummyEODataset
from rsfm_fairness_audit.io import read_csv_rows
from rsfm_fairness_audit.pipeline import run_dummy_pipeline
from rsfm_fairness_audit.sampling import balanced_indices


def test_dummy_dataset_contains_required_imbalance() -> None:
    dataset = DummyEODataset()
    metadata = dataset.load_metadata()
    regions = {row["region"] for row in metadata}
    sensors = {row["sensor"] for row in metadata}
    labels = {row["label"] for row in metadata}
    region_class = {(row["region"], row["label"]) for row in metadata}

    assert len(regions) >= 4
    assert len(sensors) >= 3
    assert len(labels) >= 4
    assert len(region_class) > len(regions)

    region_counts = {region: sum(row["region"] == region for row in metadata) for region in regions}
    assert max(region_counts.values()) / min(region_counts.values()) >= 4


def test_balanced_indices_equalize_region_label_groups() -> None:
    dataset = DummyEODataset()
    metadata = dataset.load_metadata()
    indices = balanced_indices(metadata, keys=("region", "label"), seed=3)
    counts: dict[tuple[str, int], int] = {}
    for index in indices:
        row = metadata[int(index)]
        key = (row["region"], row["label"])
        counts[key] = counts.get(key, 0) + 1

    assert counts
    assert len(set(counts.values())) == 1


def test_dummy_pipeline_writes_core_artifacts() -> None:
    artifacts = run_dummy_pipeline("outputs/test_dummy_pipeline", num_samples=180, seed=9)

    for path in artifacts.values():
        assert path.exists(), path

    region_rows = read_csv_rows(artifacts["region_matrix"])
    gap_rows = read_csv_rows(artifacts["gap_table"])
    assert len(region_rows) >= 4
    assert float(gap_rows[0]["raw_fairness_gap"]) >= 0
    assert "balanced_fairness_gap" in gap_rows[0]
