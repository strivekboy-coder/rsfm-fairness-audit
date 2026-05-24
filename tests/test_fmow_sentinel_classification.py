from __future__ import annotations

import gc
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.cli import main
from rsfm_fairness_audit.adapters.dofa import DOFAAdapter
from rsfm_fairness_audit.fmow_sentinel_classification import (
    FmowClassificationConfig,
    _prediction_rows,
    build_resnet50_13band,
    compare_fmow_runs,
    load_fmow_sentinel_image,
    run_fmow_geography_bwer,
    run_fmow_sentinel_classification,
    FmowBwerConfig,
)
from rsfm_fairness_audit.io import read_csv_rows, write_csv


def _cleanup(path: Path) -> None:
    gc.collect()
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.1)
    shutil.rmtree(path, ignore_errors=True)


def _write_fmow_fixture(root: Path, n_train_per_class: int = 12, n_val_per_class_country: int = 10) -> Path:
    chips = root / "chips"
    chips.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    index = 0
    for split, repeats in [("train", n_train_per_class), ("val", n_val_per_class_country)]:
        countries = ["CountryA", "CountryB"] if split == "val" else ["CountryA"]
        for country in countries:
            for category, value in [("airport", 10.0), ("port", 100.0)]:
                for local in range(repeats):
                    image_id = f"{split}_{country}_{category}_{local}"
                    chip = np.full((13, 8, 8), value, dtype=np.float32)
                    chip += np.arange(13, dtype=np.float32)[:, None, None]
                    np.save(chips / f"{image_id}.npy", chip)
                    rows.append(
                        {
                            "sample_id": image_id,
                            "image_id": image_id,
                            "image_path": f"chips/{image_id}.npy",
                            "category": category,
                            "split": split,
                            "timestamp": "2020-01-15",
                            "location_id": f"{country}_{local}",
                            "latitude": "42.0" if country == "CountryA" else "-12.0",
                            "longitude": "8.0",
                            "country": country,
                            "continent": "ContinentA" if country == "CountryA" else "ContinentB",
                            "un_region": "RegionA" if country == "CountryA" else "RegionB",
                            "region": "SubregionA" if country == "CountryA" else "SubregionB",
                            "latitude_band": "north_mid_latitude" if country == "CountryA" else "south_tropics",
                            "metadata_provenance": "location_level_geography_enrichment",
                        }
                    )
                    index += 1
    metadata = root / "metadata.csv"
    write_csv(metadata, rows)
    return metadata


def test_fmow_13band_loader_accepts_channels_first_npy() -> None:
    root = Path("outputs") / f"test_fmow_loader_{uuid.uuid4().hex}"
    chips = root / "chips"
    chips.mkdir(parents=True)
    np.save(chips / "img.npy", np.ones((13, 8, 8), dtype=np.float32))
    chip = load_fmow_sentinel_image({"image_path": "chips/img.npy"}, data_root=root, image_size=4)
    assert chip.shape == (13, 4, 4)
    _cleanup(root)


def test_dofa_fmow_config_is_13band_and_download_is_explicit() -> None:
    adapter = DOFAAdapter.from_config_file("configs/models/dofa_fmow_sentinel.yaml")
    assert adapter.band_profile == "sentinel2_13band_fmow"
    assert adapter.expected_bands == 13
    assert adapter.wavelengths is not None
    assert len(adapter.wavelengths) == 13
    assert adapter.image_size == 224
    assert adapter.allow_torch_hub_download is False


def test_fmow_supervised_stats_run_writes_bwer_compatible_outputs() -> None:
    root = Path("outputs") / f"test_fmow_cls_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    metadata = _write_fmow_fixture(root)
    out = root / "run"
    artifacts = run_fmow_sentinel_classification(
        FmowClassificationConfig(
            metadata_csv=metadata,
            data_root=root,
            output_dir=out,
            image_size=8,
            run_bwer=True,
        )
    )
    assert artifacts["predictions"].exists()
    assert artifacts["audit_table"].exists()
    assert (out / "bwer" / "bwer_summary.csv").exists()
    audit = read_csv_rows(out / "audit_table.csv")
    assert audit[0]["dataset"] == "fmow_sentinel"
    assert audit[0]["input_mode"] == "s2_13band_image_only"
    assert audit[0]["adaptation_protocol"] == "supervised_baseline"
    assert audit[0]["split_protocol"] == "official_split"
    assert audit[0]["country"] in {"CountryA", "CountryB"}
    bwer = read_csv_rows(out / "bwer" / "bwer_summary.csv")
    assert any(row["slice_variable"] == "country" for row in bwer)
    _cleanup(root)


def test_fmow_geography_bwer_cli_and_compare_two_runs(monkeypatch) -> None:
    root = Path("outputs") / f"test_fmow_cli_compare_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    metadata = _write_fmow_fixture(root)
    out = root / "run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "run-fmow-sentinel-classification",
            "--metadata-csv",
            str(metadata),
            "--data-root",
            str(root),
            "--output-dir",
            str(out),
            "--image-size",
            "8",
        ],
    )
    main()
    run_fmow_geography_bwer(FmowBwerConfig(input_dir=out, output_dir=out / "bwer"))
    other = root / "run_copy"
    shutil.copytree(out, other)
    artifacts = compare_fmow_runs({"supervised": out, "copy": other}, root / "comparison")
    assert artifacts["comparison_summary"].exists()
    summary = read_csv_rows(root / "comparison" / "comparison_summary.csv")
    assert len(summary) == 2
    assert "raw_bwer_country" in summary[0]
    _cleanup(root)


def test_fmow_max_samples_is_applied_after_train_eval_split() -> None:
    root = Path("outputs") / f"test_fmow_max_samples_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    metadata = _write_fmow_fixture(root, n_train_per_class=4, n_val_per_class_country=4)
    out = root / "run"
    run_fmow_sentinel_classification(
        FmowClassificationConfig(
            metadata_csv=metadata,
            data_root=root,
            output_dir=out,
            image_size=8,
            max_samples=2,
        )
    )
    run_metadata = (out / "run_metadata.json").read_text(encoding="utf-8")
    assert '"train_rows_readable": 2' in run_metadata
    assert '"eval_rows_readable": 2' in run_metadata
    _cleanup(root)


def test_resnet50_first_conv_accepts_13_channels() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    model = build_resnet50_13band(num_classes=62)
    assert model.conv1.in_channels == 13
    assert model.fc.out_features == 62


def test_resnet50_prediction_rows_preserve_audit_schema_and_geography() -> None:
    row = {
        "sample_id": "s1",
        "image_id": "img1",
        "image_path": "fmow-sentinel/val/airport/airport_1/airport_1_1.tif",
        "extracted_path": "/content/data/subset/fmow-sentinel/val/airport/airport_1/airport_1_1.tif",
        "category": "airport",
        "split": "val",
        "timestamp": "2020-07-10",
        "year": "2020",
        "month": "7",
        "season": "JJA",
        "location_id": "1",
        "latitude": "45.0",
        "longitude": "7.0",
        "country": "ITA",
        "continent": "Europe",
        "un_region": "Southern Europe",
        "region": "Europe",
        "latitude_band": "north_mid_latitude",
        "metadata_provenance": "location_level_geography_enrichment",
    }
    rows = _prediction_rows(
        [row],
        ["port"],
        FmowClassificationConfig(
            metadata_csv=Path("metadata.csv"),
            output_dir=Path("outputs/unused"),
            model="resnet50",
            split_protocol="location_disjoint",
            eval_scope="val",
        ),
        confidences=[0.73],
        top5_correct=[1.0],
    )
    out = rows[0]
    required = {
        "sample_id",
        "image_id",
        "image_path",
        "extracted_path",
        "dataset",
        "task",
        "split",
        "label",
        "category",
        "prediction",
        "predicted_category",
        "correct",
        "risk",
        "confidence",
        "max_probability",
        "model_family",
        "model_variant",
        "input_mode",
        "adaptation_protocol",
        "split_protocol",
        "eval_scope",
        "resolution",
        "band_profile",
        "timestamp",
        "year",
        "month",
        "season",
        "location_id",
        "latitude",
        "longitude",
        "country",
        "continent",
        "un_region",
        "region",
        "latitude_band",
    }
    assert required.issubset(out)
    assert out["dataset"] == "fmow_sentinel"
    assert out["task"] == "scene_classification"
    assert out["model_family"] == "resnet"
    assert out["model_variant"] == "resnet50_13band_from_scratch"
    assert out["input_mode"] == "s2_13band_image_only"
    assert out["adaptation_protocol"] == "supervised_baseline"
    assert out["split_protocol"] == "location_disjoint"
    assert out["country"] == "ITA"
    assert out["risk"] == 1.0


def test_dofa_linear_probe_writes_formal_protocol_and_cache(monkeypatch) -> None:
    pytest.importorskip("torch")

    class FakeDOFAAdapter:
        checkpoint_path = None
        repo_path = None
        torch_hub_repo = "fake/dofa"
        model_variant = "vit_base_dofa"
        allow_torch_hub_download = False
        image_size = 8
        band_profile = "sentinel2_13band_fmow"
        embedding_layer = "forward_features"
        embedding_pooling = "flatten"
        wavelengths = [0.443, 0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 0.945, 1.373, 1.61, 2.19]
        normalization_mean = [0.0] * 13
        normalization_std = [1.0] * 13
        saw_image_only_samples = False

        def load_model(self) -> None:
            return None

        def preprocess(self, batch):
            assert all(set(sample) == {"image"} for sample in batch["samples"])
            self.saw_image_only_samples = True
            return batch

        def extract_embeddings(self, batch):
            images = np.stack([sample["image"] for sample in batch["samples"]]).astype(np.float32)
            flat = images.reshape(images.shape[0], images.shape[1], -1)
            return np.concatenate([flat.mean(axis=2), flat.std(axis=2)], axis=1).astype(np.float32)

    fake_adapter = FakeDOFAAdapter()
    monkeypatch.setattr(
        "rsfm_fairness_audit.fmow_sentinel_classification.DOFAAdapter.from_config_file",
        lambda _path: fake_adapter,
    )
    root = Path("outputs") / f"test_fmow_dofa_linear_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    metadata = _write_fmow_fixture(root, n_train_per_class=6, n_val_per_class_country=3)
    out = root / "run"
    run_fmow_sentinel_classification(
        FmowClassificationConfig(
            metadata_csv=metadata,
            data_root=root,
            output_dir=out,
            model="dofa",
            model_config=Path("fake.yaml"),
            probe="linear",
            probe_epochs=2,
            dofa_embedding_pooling="mean_tokens",
            batch_size=4,
            image_size=8,
            split_protocol="location_disjoint",
        )
    )
    assert fake_adapter.saw_image_only_samples
    predictions = read_csv_rows(out / "predictions.csv")
    assert predictions
    first = predictions[0]
    assert first["model_family"] == "dofa"
    assert first["model_variant"] == "dofa_vit_base"
    assert first["adaptation_protocol"] == "frozen_encoder_linear_probe"
    assert first["input_mode"] == "s2_13band_image_only"
    assert first["split_protocol"] == "location_disjoint"
    assert first["country"] in {"CountryA", "CountryB"}
    assert first["confidence"] != ""
    assert first["max_probability"] != ""
    metadata_payload = (out / "run_metadata.json").read_text(encoding="utf-8")
    assert '"adaptation_protocol": "frozen_encoder_linear_probe"' in metadata_payload
    assert '"model_variant": "dofa_vit_base"' in metadata_payload
    assert '"wavelength_list"' in metadata_payload
    assert '"input_scale"' in metadata_payload
    assert '"embedding_pooling": "mean_tokens"' in metadata_payload
    assert '"embedding_cache_path"' in metadata_payload
    assert (out / "embedding_cache").exists()
    _cleanup(root)


def test_dofa_cache_key_ignores_manifest_path_and_split_label() -> None:
    from rsfm_fairness_audit.adapters.dofa import DOFAAdapter
    from rsfm_fairness_audit.fmow_sentinel_classification import _dofa_cache_key

    rows = [
        {"sample_id": "s1", "image_id": "i1", "image_path": "fmow-sentinel/train/a/a_1/a_1_0.tif"},
        {"sample_id": "s2", "image_id": "i2", "image_path": "fmow-sentinel/train/b/b_2/b_2_0.tif"},
    ]
    adapter = DOFAAdapter(
        checkpoint_path="checkpoint.pt",
        model_variant="vit_base_dofa",
        band_profile="sentinel2_13band_fmow",
        image_size=224,
        embedding_pooling="flatten",
        input_scale=10000,
    )
    first = FmowClassificationConfig(
        metadata_csv=Path("random_split_manifest.csv"),
        output_dir=Path("unused-a"),
        model="dofa",
        image_size=224,
        band_profile="sentinel2_13band_fmow",
    )
    second = FmowClassificationConfig(
        metadata_csv=Path("another_manifest_path.csv"),
        output_dir=Path("unused-b"),
        model="dofa",
        image_size=224,
        band_profile="sentinel2_13band_fmow",
    )

    assert _dofa_cache_key(rows, "train", first, adapter) == _dofa_cache_key(rows, "val", second, adapter)
