from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from rsfm_fairness_audit.sen1_19model_descriptive import (
    Sen119ModelDescriptiveError,
    expected_model_specs,
    run_sen1_19model_descriptive_postprocess,
)
from rsfm_fairness_audit.sen1floods11_formal import write_sen1_probability_export


def _metadata(tmp_path: Path) -> tuple[Path, Path, dict[str, list[str]]]:
    split_ids = {
        "validation": [f"Ghana_val_{index}" for index in range(89)],
        "standard_test": [f"Pakistan_test_{index}" for index in range(90)],
        "bolivia_holdout": [f"Bolivia_holdout_{index}" for index in range(15)],
    }
    train_ids = [f"USA_train_{index}" for index in range(252)]
    fields = ["sample_id", "event_id", "latitude", "longitude", "split"]
    core = tmp_path / "core.csv"
    with core.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in train_ids:
            writer.writerow({"sample_id": sample_id, "event_id": "USA", "latitude": 1, "longitude": 2, "split": "train"})
        for split in ("validation", "standard_test"):
            for sample_id in split_ids[split]:
                writer.writerow({"sample_id": sample_id, "event_id": sample_id.split("_", 1)[0], "latitude": 1, "longitude": 2, "split": split})
    bolivia = tmp_path / "bolivia.csv"
    with bolivia.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in split_ids["bolivia_holdout"]:
            writer.writerow({"sample_id": sample_id, "event_id": "Bolivia", "latitude": -17, "longitude": -65, "split": "bolivia_holdout"})
    return core, bolivia, split_ids


def _base_exports(tmp_path: Path, split_ids: dict[str, list[str]]) -> dict[str, Path]:
    result = {}
    for split, ids in split_ids.items():
        probabilities = []
        targets = []
        filenames = []
        for index, sample_id in enumerate(ids):
            probabilities.append(np.full((2, 2), 0.1, dtype=np.float32))
            target = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
            if split == "validation" and index == 0:
                target = np.full((2, 2), -1, dtype=np.int64)
            targets.append(target)
            filenames.append({"S1GRD": f"{sample_id}_S1Hand.tif"})
        result[split] = write_sen1_probability_export(
            tmp_path / "base" / split,
            probabilities=probabilities,
            targets=targets,
            filenames=filenames,
        )
    return result


def _panel(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, list[Path]]:
    core, bolivia, ids = _metadata(tmp_path)
    base = _base_exports(tmp_path, ids)
    unet = tmp_path / "unet"
    prithvi = tmp_path / "prithvi"
    terramind = tmp_path / "terramind"
    for spec in expected_model_specs(unet_root=unet, prithvi_root=prithvi, terramind_root=terramind):
        for split in ("validation", "standard_test", "bolivia_holdout"):
            shutil.copytree(base[split], spec.export(split))
    audits = []
    for name in ("unet", "prithvi", "terramind"):
        path = tmp_path / f"{name}_audit.json"
        path.write_text(json.dumps({"status": "pass", "source": name}), encoding="utf-8")
        audits.append(path)
    return unet, prithvi, terramind, core, bolivia, audits


def test_unified_19_model_metrics_are_comparable_and_resumable(tmp_path: Path) -> None:
    unet, prithvi, terramind, core, bolivia, audits = _panel(tmp_path)
    output = tmp_path / "output"
    run_sen1_19model_descriptive_postprocess(
        unet_root=unet,
        prithvi_root=prithvi,
        terramind_root=terramind,
        metadata_csvs=[core, bolivia],
        audit_evidence=audits,
        output_dir=output,
        code_commit="a" * 40,
        package_version="0.4.36",
    )
    rows = list(csv.DictReader((output / "unified_19model_metrics.csv").open(encoding="utf-8")))
    assert len(rows) == 19 * 4
    assert {row["risk_definition"] for row in rows} == {
        "per_chip_one_minus_flood_iou_at_probability_0.5"
    }
    assert {int(row["source_split_sample_count"]) for row in rows if row["split"] == "combined_held_out"} == {105}
    assert {float(row["all_nonflood_prediction_chip_rate"]) for row in rows} == {1.0}
    assert {float(row["near_constant_probability_chip_rate"]) for row in rows} == {1.0}
    assert len(list(csv.DictReader((output / "three_seed_architecture_modality_summary.csv").open(encoding="utf-8")))) == 28
    assert len(list(csv.DictReader((output / "same_seed_modality_rankings.csv").open(encoding="utf-8")))) == 72
    assert len(list(csv.DictReader((output / "modality_ranking_stability.csv").open(encoding="utf-8")))) == 8
    prithvi_rows = [row for row in rows if row["model"] == "prithvi_eo_v2_300_tl_s2"]
    assert {row["comparison_role"] for row in prithvi_rows} == {
        "task_specific_external_resolution_reference"
    }
    contract = json.loads((output / "completion_contract.json").read_text(encoding="utf-8"))
    assert contract["status"] == "complete"
    assert contract["formal_evidence"] is False
    assert contract["model_count"] == 19
    # A complete immutable derivation is safely reusable.
    assert run_sen1_19model_descriptive_postprocess(
        unet_root=unet,
        prithvi_root=prithvi,
        terramind_root=terramind,
        metadata_csvs=[core, bolivia],
        audit_evidence=audits,
        output_dir=output,
        code_commit="a" * 40,
        package_version="0.4.36",
    ) == output


def test_output_overlap_and_partial_output_fail(tmp_path: Path) -> None:
    unet, prithvi, terramind, core, bolivia, audits = _panel(tmp_path)
    with pytest.raises(Sen119ModelDescriptiveError, match="must not overlap"):
        run_sen1_19model_descriptive_postprocess(
            unet_root=unet,
            prithvi_root=prithvi,
            terramind_root=terramind,
            metadata_csvs=[core, bolivia],
            audit_evidence=audits,
            output_dir=unet / "derived",
            code_commit="a" * 40,
            package_version="0.4.36",
        )
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "orphan.txt").write_text("partial", encoding="utf-8")
    with pytest.raises(Sen119ModelDescriptiveError, match="Non-empty incomplete"):
        run_sen1_19model_descriptive_postprocess(
            unet_root=unet,
            prithvi_root=prithvi,
            terramind_root=terramind,
            metadata_csvs=[core, bolivia],
            audit_evidence=audits,
            output_dir=partial,
            code_commit="a" * 40,
            package_version="0.4.36",
        )
