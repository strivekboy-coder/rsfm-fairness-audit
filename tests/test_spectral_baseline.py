from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.spectral_baseline import (
    SpectralBaselineConfig,
    run_spectral_sen1floods11,
    spectral_indices,
    threshold_spectral_index,
)


def _write_tiny_prepared_dataset(root: Path, n: int = 6) -> None:
    (root / "chips").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    events = ["Bolivia", "Pakistan", "Mekong"]
    for index in range(n):
        event = events[index % len(events)]
        image = np.zeros((1, 6, 16, 16), dtype=np.float32)
        mask = np.zeros((16, 16), dtype=np.int16)
        mask[4:12, 4:12] = 1
        image[0, 1] = 0.20
        image[0, 3] = 0.30
        image[0, 4] = 0.30
        image[0, 1, 4:12, 4:12] = 0.70
        image[0, 3, 4:12, 4:12] = 0.10
        image[0, 4, 4:12, 4:12] = 0.05
        if index == n - 1:
            mask[:2, :2] = -1
        chip = root / "chips" / f"{event}_{index}_prithvi_s2.npz"
        label = root / "masks" / f"{event}_{index}_qc.npz"
        np.savez_compressed(chip, image=image)
        np.savez_compressed(label, mask=mask)
        rows.append(
            {
                "sample_id": f"{event}_{index}",
                "event_id": event,
                "event": event,
                "region": event,
                "country": event,
                "chip_path": str(chip.relative_to(root)),
                "mask_path": str(label.relative_to(root)),
                "band_profile": "prithvi_tl_sen1floods11",
                "band_names": "BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2",
                "target_size": "512",
            }
        )
    write_csv(root / "metadata.csv", rows)


def test_spectral_indices_and_thresholding_handle_6band_s2() -> None:
    image = np.zeros((6, 4, 4), dtype=np.float32)
    image[1] = 0.7
    image[3] = 0.1
    image[4] = 0.05
    scores = spectral_indices(image)
    assert set(scores) == {"ndwi", "mndwi", "nir_darkness"}
    assert float(scores["ndwi"].mean()) > 0
    assert float(scores["mndwi"].mean()) > float(scores["ndwi"].mean())
    assert int(threshold_spectral_index(scores["mndwi"], "mndwi", 0.0).sum()) == 16
    assert int(threshold_spectral_index(scores["nir_darkness"], "nir_darkness", 0.2).sum()) == 16


def test_spectral_baseline_writes_bwer_compatible_outputs() -> None:
    root = Path("outputs") / f"test_spectral_{uuid.uuid4().hex}"
    data_root = root / "data"
    output = root / "run"
    _write_tiny_prepared_dataset(data_root)
    artifacts = run_spectral_sen1floods11(
        SpectralBaselineConfig(
            data_root=data_root,
            output_dir=output,
            index="mndwi",
            threshold=0.0,
            max_samples=6,
            run_bwer_v2=True,
        )
    )
    for key in ["segmentation_metrics", "event_segmentation_metrics", "audit_table", "bwer_summary", "support_diagnostics"]:
        assert artifacts[key].exists(), key
    chip_rows = read_csv_rows(output / "segmentation_metrics.csv")
    assert chip_rows[0]["adaptation_protocol"] == "diagnostic_spectral_rule"
    assert chip_rows[0]["model_family"] == "spectral_rule"
    assert chip_rows[0]["input_mode"] == "s2_6band_image_only"
    event_rows = read_csv_rows(output / "event_segmentation_metrics.csv")
    assert {row["event_id"] for row in event_rows} == {"Bolivia", "Pakistan", "Mekong"}
    bwer_rows = read_csv_rows(output / "bwer_v2" / "bwer_v2_summary.csv")
    assert any(row["analysis_type"] == "raw" for row in bwer_rows)
    assert any(row["analysis_type"] == "standardised" for row in bwer_rows)
    shutil.rmtree(root, ignore_errors=True)
