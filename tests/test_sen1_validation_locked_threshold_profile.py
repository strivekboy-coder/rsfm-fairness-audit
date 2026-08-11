from __future__ import annotations

import csv
import json
from pathlib import Path
import uuid

import numpy as np
import pytest

from rsfm_fairness_audit.sen1floods11_formal import write_sen1_probability_export
from scripts.colab.run_sen1_validation_locked_threshold_profile_colab import run_profile


def _case() -> Path:
    root = Path("work") / f"sen1_threshold_test_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


def _write_metadata(path: Path) -> None:
    rows = [
        ("Ghana_1", "Ghana", 6.1, -1.1),
        ("Pakistan_1", "Pakistan", 30.1, 70.1),
        ("Ghana_2", "Ghana", 6.2, -1.2),
        ("Pakistan_2", "Pakistan", 30.2, 70.2),
        ("Bolivia_1", "Bolivia", -17.1, -65.1),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sample_id", "event_id", "latitude", "longitude", "country"]
        )
        writer.writeheader()
        for sample_id, event, latitude, longitude in rows:
            writer.writerow({
                "sample_id": sample_id, "event_id": event, "latitude": latitude,
                "longitude": longitude, "country": event,
            })


def _export(path: Path, sample_ids: list[str]) -> None:
    truth = np.asarray([[0, 1], [1, 0]], dtype=np.int16)
    probability = np.asarray([[0.1, 0.8], [0.6, 0.2]], dtype=np.float32)
    write_sen1_probability_export(
        path,
        probabilities=[probability for _ in sample_ids],
        targets=[truth for _ in sample_ids],
        filenames=[f"{sample_id}_S2Hand.tif" for sample_id in sample_ids],
    )


def test_sen1_profile_selects_on_validation_only_and_seals_outputs() -> None:
    drive = _case()
    final = drive / "outputs" / "geobwer_final_v3"
    metadata = final / "sen1_19model_descriptive_v2" / "official_446_metadata_binding.csv"
    _write_metadata(metadata)
    model_roots = [
        final / "sen1_geobwer_v0428" / "supervised" / "s1" / "seed_42" / "probabilities",
        final / "sen1_geobwer_v0434" / "terramind_final" / "s1" / "seed_42" / "probabilities",
        final / "sen1_geobwer_v0432" / "prithvi_final" / "probabilities",
    ]
    split_ids = {
        "validation": ["Ghana_1", "Pakistan_1"],
        "test": ["Ghana_2", "Pakistan_2"],
        "bolivia_holdout": ["Bolivia_1"],
    }
    for model_root in model_roots:
        for split, ids in split_ids.items():
            _export(model_root / split, ids)
    # The frozen TerraMind panel pads its writer rank to three digits, unlike
    # U-Net. Discovery must rely on the stable index_parts export contract.
    for manifest in model_roots[1].rglob("writer_manifest_rank_0.json"):
        manifest.rename(manifest.with_name("writer_manifest_rank_000.json"))
    output = drive / "result"
    paths = run_profile(
        drive_root=drive, output_dir=output, thresholds=(0.3, 0.5, 0.7),
        expected_models=3, expected_exports=9,
        expected_counts={"validation": 2, "standard_test": 2, "bolivia_holdout": 1},
        expected_events={"validation": 2, "standard_test": 2, "bolivia_holdout": 1},
    )
    selections = list(csv.DictReader(paths["selection"].open(encoding="utf-8")))
    assert len(selections) == 3
    assert {row["selected_threshold"] for row in selections} == {"0.5"}
    assert {row["test_or_bolivia_used_for_selection"] for row in selections} == {"False"}
    profile = list(csv.DictReader(paths["profile"].open(encoding="utf-8")))
    assert {row["split"] for row in profile} == {
        "validation", "standard_test", "bolivia_holdout", "combined_held_out"
    }
    assert all(row["spatial_inference_valid"] == "False" for row in profile)
    bolivia = [row for row in profile if row["split"] == "bolivia_holdout"]
    assert bolivia
    assert {row["event_geobwer_identified"] for row in bolivia} == {"False"}
    assert {row["event_gap_interpretation"] for row in bolivia} == {
        "single_event_structural_zero_not_between_event_disparity"
    }
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["test_or_bolivia_used_for_selection"] is False
    assert manifest["model_training_or_inference"] is False
    completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
    assert completion["status"] == "complete"
    reused = run_profile(
        drive_root=drive, output_dir=output, thresholds=(0.3, 0.5, 0.7),
        expected_models=3, expected_exports=9,
        expected_counts={"validation": 2, "standard_test": 2, "bolivia_holdout": 1},
        expected_events={"validation": 2, "standard_test": 2, "bolivia_holdout": 1},
    )
    assert reused["profile"] == paths["profile"]

    paths["profile"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or changed"):
        run_profile(
            drive_root=drive, output_dir=output, thresholds=(0.3, 0.5, 0.7),
            expected_models=3, expected_exports=9,
            expected_counts={"validation": 2, "standard_test": 2, "bolivia_holdout": 1},
            expected_events={"validation": 2, "standard_test": 2, "bolivia_holdout": 1},
        )
