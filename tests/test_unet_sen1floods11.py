from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from scripts.run_unet_sen1floods11_colab import _validate_output_dir
from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.bwer_v2 import run_bwer_v2_posthoc
from rsfm_fairness_audit.unet_baseline import (
    Sen1Floods11TorchDataset,
    S2ResNet34UNet,
    UNetSmall,
    UnetConfig,
    _prepare_image,
    _read_metadata,
    masked_bce_dice_loss,
    run_unet_sen1floods11,
    split_metadata,
)


def _write_tiny_prepared_dataset(root: Path, n: int = 8) -> None:
    (root / "chips").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    events = ["Bolivia", "Pakistan", "Mekong", "USA"]
    for index in range(n):
        event = events[index % len(events)]
        image = np.zeros((1, 6, 32, 32), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.int64)
        start = 4 + (index % 3) * 4
        mask[start : start + 12, start : start + 12] = 1
        image[0, 2, :, :] = mask.astype(np.float32)
        image[0, 3, :, :] = mask.astype(np.float32) * 0.8
        image[0, 0, :, :] = 0.1
        if index == n - 1:
            mask[:4, :4] = -1
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
                "band_indices": "1,2,3,8,11,12",
                "band_names": "BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2",
                "target_size": "512",
            }
        )
    write_csv(root / "metadata.csv", rows)


def _write_fake_unet_completed_output(root: Path) -> None:
    chip_rows = []
    event_rows = []
    event_specs = {
        "Bolivia": [(120, 30, 40), (180, 40, 60), (260, 50, 80)],
        "Pakistan": [(100, 45, 70), (170, 55, 90), (240, 70, 120)],
        "Mekong": [(140, 15, 20), (210, 20, 25), (300, 25, 30)],
    }
    for event_id, chips in event_specs.items():
        totals = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "valid_pixel_count": 0, "positive_pixel_count": 0, "predicted_positive_pixel_count": 0}
        for index, (tp, fp, fn) in enumerate(chips):
            valid = 1000
            positive = tp + fn
            predicted = tp + fp
            tn = valid - tp - fp - fn
            iou = tp / (tp + fp + fn)
            row = {
                "sample_id": f"{event_id}_{index}",
                "event_id": event_id,
                "event": event_id,
                "country": event_id,
                "dataset": "sen1floods11",
                "model": "unet_sen1floods11_s2_512",
                "model_family": "unet",
                "task": "segmentation",
                "adaptation_protocol": "supervised_baseline",
                "split_protocol": "random_chip_split",
                "input_mode": "S2",
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "valid_pixel_count": valid,
                "positive_pixel_count": positive,
                "predicted_positive_pixel_count": predicted,
                "ground_truth_positive_pixel_ratio": positive / valid,
                "micro_iou": iou,
                "risk": 1.0 - iou,
            }
            chip_rows.append(row)
            for key in totals:
                totals[key] += int(row[key])
        tp, fp, fn = totals["TP"], totals["FP"], totals["FN"]
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        event_rows.append(
            {
                "dataset": "sen1floods11",
                "model": "unet_sen1floods11_s2_512",
                "model_family": "unet",
                "task": "segmentation",
                "event_id": event_id,
                "event": event_id,
                "country": event_id,
                "sample_count": len(chips),
                "adaptation_protocol": "supervised_baseline",
                "split_protocol": "random_chip_split",
                "input_mode": "S2",
                **totals,
                "TP_plus_FN_support": totals["positive_pixel_count"],
                "micro_iou": iou,
                "micro_dice": dice,
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
                "risk": 1.0 - iou,
                "score": iou,
            }
        )
    write_csv(root / "segmentation_metrics.csv", chip_rows)
    write_csv(root / "event_segmentation_metrics.csv", event_rows)
    write_csv(root / "bwer_summary.csv", [{"dataset": "sen1floods11", "model": "unet_sen1floods11_s2_512", "bwer": 0.1}])
    (root / "model_debug.json").write_text('{"resolution": 512, "model_family": "unet"}\n', encoding="utf-8")


def test_unet_metadata_and_image_preparation_without_torch() -> None:
    root = Path("outputs") / f"test_unet_metadata_{uuid.uuid4().hex}" / "data"
    _write_tiny_prepared_dataset(root, n=5)
    rows = _read_metadata(root)
    assert len(rows) == 5
    assert rows[0]["event_id"] == "Bolivia"
    assert rows[0]["sample_id"] == "Bolivia_0"
    image = np.ones((1, 6, 8, 8), dtype=np.float32) * 10000
    prepared = _prepare_image(image)
    assert prepared.shape == (6, 8, 8)
    assert float(prepared.max()) == pytest.approx(1.0)
    shutil.rmtree(root.parent, ignore_errors=True)


def test_unet_split_metadata_records_random_and_event_held_out_protocols() -> None:
    root = Path("outputs") / f"test_unet_split_{uuid.uuid4().hex}" / "data"
    _write_tiny_prepared_dataset(root, n=8)
    rows = _read_metadata(root)
    random_splits = split_metadata(rows, UnetConfig(data_root=root, output_dir=root.parent / "run", seed=1))
    assert random_splits["train"]
    assert random_splits["val"]
    assert random_splits["test"]
    held = split_metadata(
        rows,
        UnetConfig(
            data_root=root,
            output_dir=root.parent / "run_event",
            split_protocol="event_held_out",
            held_out_events=("Pakistan",),
            seed=1,
        ),
    )
    assert {row["event_id"] for row in held["test"]} == {"Pakistan"}
    assert all(row["event_id"] != "Pakistan" for row in held["train"] + held["val"])
    shutil.rmtree(root.parent, ignore_errors=True)


def test_bwer_v2_runs_on_unet_output_without_torch() -> None:
    root = Path("outputs") / f"test_unet_bwer_v2_{uuid.uuid4().hex}" / "run"
    root.mkdir(parents=True)
    _write_fake_unet_completed_output(root)
    out = root / "bwer_v2"
    artifacts = run_bwer_v2_posthoc(root, out, bootstrap=5, seed=11)
    assert artifacts["bwer_v2_summary"].exists()
    assert artifacts["standardised_bwer"].exists()
    summary = read_csv_rows(out / "bwer_v2_summary.csv")
    assert {row["analysis_type"] for row in summary} >= {"raw", "standardised"}
    assert any(row["balance_variable"] == "flood_extent_bin" for row in summary)
    assert all(row["adaptation_protocol"] == "supervised_baseline" for row in summary)
    assert all(row["model_family"] == "unet" for row in summary)
    adaptation_report = (out / "adaptation_protocol_report.md").read_text(encoding="utf-8")
    assert "model_family: unet" in adaptation_report
    assert "supervised classical baseline" in adaptation_report
    assert "official Sen1Floods11 task-adapted decoder route" not in adaptation_report
    split_report = (out / "split_diagnostics_report.md").read_text(encoding="utf-8")
    assert "event leakage is possible" in split_report
    shutil.rmtree(root.parent, ignore_errors=True)


def test_colab_helper_output_validation_accepts_complete_unet_run() -> None:
    root = Path("outputs") / f"test_unet_zip_validation_{uuid.uuid4().hex}" / "run"
    bwer_v2 = root / "bwer_v2"
    bwer_v2.mkdir(parents=True)
    for rel in [
        "segmentation_metrics.csv",
        "event_segmentation_metrics.csv",
        "audit_table.csv",
        "bwer_summary.csv",
        "bwer_by_slice.csv",
        "support_diagnostics.csv",
    ]:
        write_csv(root / rel, [{"dataset": "sen1floods11", "model": "unet_sen1floods11_s2_512"}])
    for rel in ["bwer_v2_summary.csv", "standardised_bwer.csv", "event_failure_analysis.csv"]:
        write_csv(bwer_v2 / rel, [{"model_family": "unet", "adaptation_protocol": "supervised_baseline"}])
    (root / "warnings.json").write_text('{"warnings": []}\n', encoding="utf-8")
    (root / "report.md").write_text("U-Net Protocol Note\n", encoding="utf-8")
    (root / "model_debug.json").write_text('{"model_family": "unet"}\n', encoding="utf-8")
    (root / "run_metadata.json").write_text('{"adaptation_protocol": "supervised_baseline"}\n', encoding="utf-8")
    (bwer_v2 / "bwer_audit_report.md").write_text("BWER v2\n", encoding="utf-8")
    _validate_output_dir(root)
    shutil.rmtree(root.parent, ignore_errors=True)


def test_colab_helper_output_validation_rejects_incomplete_unet_run() -> None:
    root = Path("outputs") / f"test_unet_zip_validation_bad_{uuid.uuid4().hex}" / "run"
    root.mkdir(parents=True)
    write_csv(root / "segmentation_metrics.csv", [{"dataset": "sen1floods11"}])
    with pytest.raises(RuntimeError, match="missing files"):
        _validate_output_dir(root)
    shutil.rmtree(root.parent, ignore_errors=True)


def test_unet_dataset_loads_prepared_npz_and_ignore_label() -> None:
    torch = pytest.importorskip("torch")
    root = Path("outputs") / f"test_unet_dataset_{uuid.uuid4().hex}" / "data"
    _write_tiny_prepared_dataset(root, n=4)
    rows = read_csv_rows(root / "metadata.csv")
    dataset = Sen1Floods11TorchDataset(root, rows)
    item = dataset[len(dataset) - 1]
    assert tuple(item["image"].shape) == (6, 32, 32)
    assert tuple(item["mask"].shape) == (32, 32)
    assert int((item["mask"] == -1).sum().item()) > 0
    shutil.rmtree(root.parent, ignore_errors=True)


def test_unet_forward_and_ignore_loss_are_finite() -> None:
    torch = pytest.importorskip("torch")
    model = UNetSmall(in_channels=6, base_channels=4)
    logits = model(torch.randn(2, 6, 32, 32))
    assert tuple(logits.shape) == (2, 32, 32)
    masks = torch.zeros(2, 32, 32, dtype=torch.long)
    masks[0, :4, :4] = -1
    loss = masked_bce_dice_loss(logits, masks)
    assert torch.isfinite(loss)
    ignored = torch.full((1, 32, 32), -1, dtype=torch.long)
    zero_loss = masked_bce_dice_loss(logits[:1], ignored)
    assert torch.isfinite(zero_loss)
    assert float(zero_loss.detach()) == pytest.approx(0.0)


def test_s2_resnet34_unet_forward_smoke_when_torchvision_available() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    model = S2ResNet34UNet(in_channels=6, pretrained_encoder=False)
    logits = model(torch.randn(1, 6, 64, 64))
    assert tuple(logits.shape) == (1, 64, 64)


def test_unet_smoke_run_writes_bwer_compatible_outputs() -> None:
    pytest.importorskip("torch")
    root = Path("outputs") / f"test_unet_smoke_{uuid.uuid4().hex}"
    data_root = root / "data"
    output_dir = root / "run"
    _write_tiny_prepared_dataset(data_root, n=8)
    artifacts = run_unet_sen1floods11(
        UnetConfig(
            data_root=data_root,
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            base_channels=4,
            max_samples=8,
            eval_split="all",
            run_bwer_v2=True,
            seed=3,
            amp=False,
        )
    )
    for key in ["segmentation_metrics", "event_segmentation_metrics", "audit_table", "bwer_summary", "support_diagnostics"]:
        assert artifacts[key].exists(), key
    assert (output_dir / "bwer_v2" / "bwer_v2_summary.csv").exists()
    chip_rows = read_csv_rows(output_dir / "segmentation_metrics.csv")
    assert chip_rows
    assert chip_rows[0]["adaptation_protocol"] == "supervised_baseline"
    assert chip_rows[0]["split_protocol"] == "random_chip_split"
    assert "valid_pixel_count" in chip_rows[0]
    event_rows = read_csv_rows(output_dir / "event_segmentation_metrics.csv")
    assert {row["event_id"] for row in event_rows} >= {"Bolivia", "Pakistan"}
    bwer_v2 = read_csv_rows(output_dir / "bwer_v2" / "bwer_v2_summary.csv")
    assert any(row["analysis_type"] == "raw" for row in bwer_v2)
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "U-Net Protocol Note" in report
    assert "supervised_baseline" in report
    shutil.rmtree(root, ignore_errors=True)
