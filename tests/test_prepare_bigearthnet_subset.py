from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

from rsfm_fairness_audit.adapters.bigearthnet import BigEarthNetDatasetAdapter


def _load_prepare_module():
    script_path = Path("scripts/prepare_bigearthnet_subset.py")
    spec = importlib.util.spec_from_file_location("prepare_bigearthnet_subset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_real_like_fixture(root: Path, count: int = 12) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        region = "germany" if index % 2 == 0 else "spain"
        label = index % 3
        s1 = np.full((2, 6, 6), fill_value=index, dtype=np.float32)
        s2 = np.full((9, 6, 6), fill_value=index, dtype=np.float32)
        s1_path = root / f"real_{index:03d}_s1.npy"
        s2_path = root / f"real_{index:03d}_s2.npy"
        np.save(s1_path, s1)
        np.save(s2_path, s2)
        rows.append(
            {
                "patch_id": f"REAL-BEN-{index:03d}",
                "primary_label": label,
                "multi_hot": "[1, 0, 0]" if label == 0 else ("[0, 1, 0]" if label == 1 else "[0, 0, 1]"),
                "label_names": f"class_{label}",
                "country_name": region,
                "split_name": "train",
                "s1_rel": s1_path.name,
                "s2_rel": s2_path.name,
            }
        )
    metadata_path = root / "real_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def test_prepare_real_subset_stratified_s2_is_adapter_compatible() -> None:
    module = _load_prepare_module()
    source = Path("outputs/test_prepare_real_source")
    metadata_path = _write_real_like_fixture(source, count=18)
    output = Path("outputs/test_prepare_real_subset")

    manifest = module.prepare_subset(
        source_root=source,
        metadata_path=metadata_path,
        output_root=output,
        subset_size=12,
        sensor_mode="S2",
        seed=3,
        stratify_by="region_class",
        sample_id_column="patch_id",
        label_column="primary_label",
        label_vector_column="multi_hot",
        region_column="country_name",
        country_column="country_name",
        split_column="split_name",
        s2_path_column="s2_rel",
        progress_every=1000,
    )

    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    assert len(rows) == 12
    assert {row["region"] for row in rows} == {"germany", "spain"}
    assert {row["label"] for row in rows} == {"0", "1", "2"}
    assert all((output / row["s2_path"]).exists() for row in rows)

    adapter = BigEarthNetDatasetAdapter(output, subset_size=4, sensor_mode="S2")
    assert len(adapter.load_metadata()) == 4
    assert adapter.load_sample(0)["image"].shape == (9, 6, 6)


def test_prepare_real_subset_s1_s2_and_resume_skip_existing() -> None:
    module = _load_prepare_module()
    source = Path("outputs/test_prepare_real_source_s1s2")
    metadata_path = _write_real_like_fixture(source, count=6)
    output = Path("outputs/test_prepare_real_subset_s1s2")

    manifest = module.prepare_subset(
        source_root=source,
        metadata_path=metadata_path,
        output_root=output,
        subset_size=6,
        sensor_mode="S1+S2",
        seed=4,
        stratify_by="class",
        sample_id_column="patch_id",
        label_column="primary_label",
        label_vector_column="multi_hot",
        region_column="country_name",
        country_column="country_name",
        split_column="split_name",
        s1_path_column="s1_rel",
        s2_path_column="s2_rel",
        progress_every=1000,
    )
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    first_s2 = output / rows[0]["s2_path"]
    before = first_s2.stat().st_mtime_ns

    module.prepare_subset(
        source_root=source,
        metadata_path=metadata_path,
        output_root=output,
        subset_size=6,
        sensor_mode="S1+S2",
        seed=4,
        stratify_by="class",
        sample_id_column="patch_id",
        label_column="primary_label",
        label_vector_column="multi_hot",
        region_column="country_name",
        country_column="country_name",
        split_column="split_name",
        s1_path_column="s1_rel",
        s2_path_column="s2_rel",
        progress_every=1000,
    )

    assert first_s2.stat().st_mtime_ns == before
    assert all((output / row["s1_path"]).exists() for row in rows)
    assert all((output / row["s2_path"]).exists() for row in rows)
