from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.adapters.croma import CROMAAdapter
from rsfm_fairness_audit.adapters.reben import (
    ConfigILMRebenDatasetAdapter,
    LmdbSafetensorsRebenDatasetAdapter,
    _LMDB_ENV_CACHE,
    import_configilm_reben_dataset_class,
    reben_labels_to_multihot,
    resolve_reben_root_dir,
)
from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.reben_sensor_audit import (
    BifoldResNet101Runner,
    REBEN_LABEL_COUNT,
    SOURCE_VERIFICATION_URLS,
    RebenRunLabels,
    croma_embedding_from_output,
    collect_reben_sensor_audit_outputs,
    compute_multilabel_metrics,
    default_reben_class_names,
    expand_predictions_to_label_audit_rows,
    run_reben_dataset_preflight,
    run_reben_multilabel_bwer,
    run_croma_reben_frozen_probe,
    run_bifold_resnet101_reben_inference,
    select_thresholds_from_validation,
    validate_bifold_resnet101_refs,
    validate_reben_sensor_audit_contract,
    validate_multilabel_audit_rows,
    write_reben_source_verification_report,
    write_run_outputs,
)


class _FakeRebenDataset:
    def __init__(self, count: int = 4) -> None:
        self.metadata = [{"patch_id": f"p{i}", "country": "DE" if i % 2 == 0 else "FR", "split": "train"} for i in range(count)]
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        image = np.zeros((14, 16, 16), dtype=np.float32)
        image += index
        image += np.arange(14, dtype=np.float32)[:, None, None] * 0.01
        label = np.zeros(19, dtype=np.float32)
        label[index % 3] = 1.0
        label[(index + 1) % 3] = 1.0
        return image, label


class _MockCromaModel:
    def extract_embeddings(self, *inputs) -> np.ndarray:
        arrays = []
        for value in inputs:
            if isinstance(value, dict):
                arrays.extend(value.values())
            else:
                arrays.append(value)
        merged = np.concatenate(arrays, axis=1)
        return np.concatenate([merged.mean(axis=(2, 3)), merged.std(axis=(2, 3))], axis=1).astype(np.float32)


class _MockBifoldModel:
    def eval(self):
        return self

    def __call__(self, images):
        import torch

        pooled = images.mean(dim=(2, 3))
        out = torch.zeros((images.shape[0], 19), dtype=torch.float32, device=images.device)
        width = min(pooled.shape[1], 19)
        out[:, :width] = pooled[:, :width]
        return {"logits": out}


def test_source_verification_smoke_report_exists() -> None:
    report = Path("outputs/reben_croma_sensor_mode_audit/source_verification_report.md")
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Proceed with implementation" in text
    assert "BigEarthNet v2.0 / reBEN" in text
    assert len(SOURCE_VERIFICATION_URLS) == 5
    assert REBEN_LABEL_COUNT == 19


def test_source_verification_report_can_be_written_to_run_dir() -> None:
    root = Path("outputs/test_reben_source_report")
    path = write_reben_source_verification_report(root)
    text = path.read_text(encoding="utf-8")
    assert "BigEarthNet v2.0 / reBEN" in text
    assert "sensor_mode as cross-run" in text


def test_default_reben_class_names_are_official_19_names() -> None:
    names = default_reben_class_names()
    assert len(names) == 19
    assert names[0] == "Urban fabric"
    assert "Agro-forestry areas" in names
    assert names[-1] == "Marine waters"


def test_threshold_selection_uses_validation_only() -> None:
    y_true_val = np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=int)
    y_prob_val = np.asarray([[0.9, 0.1], [0.8, 0.6], [0.2, 0.7], [0.1, 0.4]], dtype=float)
    thresholds_a = select_thresholds_from_validation(y_true_val, y_prob_val, grid=[0.25, 0.5, 0.75])
    # Eval arrays are intentionally different; the threshold API does not accept
    # them, so mutating eval data cannot leak into selected thresholds.
    y_true_eval = np.asarray([[0, 1], [0, 1]], dtype=int)
    y_prob_eval = np.asarray([[0.99, 0.01], [0.98, 0.02]], dtype=float)
    del y_true_eval, y_prob_eval
    thresholds_b = select_thresholds_from_validation(y_true_val, y_prob_val, grid=[0.25, 0.5, 0.75])
    assert np.allclose(thresholds_a, thresholds_b)
    assert thresholds_a.shape == (2,)


def test_multilabel_metrics_and_label_expanded_rows() -> None:
    labels = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=int)
    probs = np.asarray([[0.9, 0.2], [0.3, 0.8], [0.7, 0.6]], dtype=float)
    thresholds = np.asarray([0.5, 0.5])
    summary, per_class = compute_multilabel_metrics(labels, probs, thresholds, ["forest", "urban"])
    assert summary["macro_ap"] == pytest.approx(1.0)
    assert summary["micro_f1"] == pytest.approx(1.0)
    assert [row["class_label"] for row in per_class] == ["forest", "urban"]

    sample_rows = [
        {"sample_id": "a", "patch_id": "a", "country": "DE", "split": "val"},
        {"sample_id": "b", "patch_id": "b", "country": "FR", "split": "val"},
        {"sample_id": "c", "patch_id": "c", "country": "DE", "split": "val"},
    ]
    audit_rows = expand_predictions_to_label_audit_rows(
        sample_rows,
        labels,
        probs,
        thresholds,
        RebenRunLabels(
            model_family="croma",
            model_variant="croma_base",
            sensor_mode="S2",
            adaptation_protocol="frozen_encoder_linear_probe",
        ),
        ["forest", "urban"],
    )
    assert len(audit_rows) == 6
    assert audit_rows[0]["task"] == "multilabel_scene_classification"
    assert audit_rows[0]["risk_bce"] >= 0.0
    assert {row["class_label"] for row in audit_rows} == {"forest", "urban"}


def test_reben_bwer_rejects_single_label_primitive() -> None:
    with pytest.raises(ValueError, match="multi-label BWER requires"):
        validate_multilabel_audit_rows(
            [
                {
                    "sample_id": "x",
                    "prediction": "forest",
                    "risk": "0",
                }
            ]
        )


def test_reben_bwer_outputs_support_and_missing_policy() -> None:
    labels = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=int)
    probs = np.asarray([[0.9, 0.2], [0.2, 0.8], [0.6, 0.55], [0.4, 0.45]], dtype=float)
    rows = expand_predictions_to_label_audit_rows(
        [
            {"sample_id": "a", "country": "DE"},
            {"sample_id": "b", "country": "FR"},
            {"sample_id": "c", "country": "DE"},
            {"sample_id": "d", "country": "IT"},
        ],
        labels,
        probs,
        [0.5, 0.5],
        RebenRunLabels("croma", "croma_base", "S2", "frozen_encoder_linear_probe"),
        ["forest", "urban"],
    )
    out = Path("outputs/test_reben_bwer_outputs")
    artifacts = run_reben_multilabel_bwer(rows, out, model_name="croma_s2", min_support=1)
    summary = read_csv_rows(artifacts["bwer_summary"])
    assert {row["slice_variable"] for row in summary} >= {"country", "class_label"}
    assert any(row["slice_variable"] == "country" and row["balance_variable"] == "class_label" for row in summary)
    assert read_csv_rows(artifacts["support_sensitivity"])
    assert read_csv_rows(artifacts["missing_policy_sensitivity"])


def test_reben_dataset_preflight_writes_support_tables() -> None:
    root = Path("outputs/test_reben_dataset_preflight")
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "metadata.csv"
    rows = []
    for index in range(19):
        rows.append(
            {
                "patch_id": f"patch_{index}",
                "split": "train" if index < 10 else "validation",
                "country": "DE" if index % 2 == 0 else "FR",
                "labels": f"class_{index}",
            }
        )
    write_csv(metadata, rows)
    artifacts = run_reben_dataset_preflight(metadata, root / "preflight")
    preflight = json.loads(artifacts["dataset_preflight"].read_text(encoding="utf-8"))
    assert preflight["label_count_observed"] == 19
    assert preflight["status"] == "ok"
    assert len(read_csv_rows(artifacts["class_support"])) == 19
    assert {row["split"] for row in read_csv_rows(artifacts["split_summary"])} == {"train", "validation"}


def test_write_run_outputs_creates_prediction_metrics_and_bwer() -> None:
    root = Path("outputs/test_reben_write_run_outputs")
    labels = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=int)
    probs = np.asarray([[0.9, 0.2], [0.2, 0.8], [0.6, 0.55], [0.4, 0.45]], dtype=float)
    artifacts = write_run_outputs(
        root,
        run_name="croma_s2",
        sample_rows=[
            {"sample_id": "a", "country": "DE"},
            {"sample_id": "b", "country": "FR"},
            {"sample_id": "c", "country": "DE"},
            {"sample_id": "d", "country": "IT"},
        ],
        y_true=labels,
        y_prob=probs,
        thresholds=[0.5, 0.5],
        class_names=["forest", "urban"],
        run_labels=RebenRunLabels("croma", "croma_base", "S2", "frozen_encoder_linear_probe"),
        run_bwer=True,
    )
    assert artifacts["predictions"].exists()
    assert artifacts["thresholds"].exists()
    assert artifacts["thresholds_json"].exists()
    assert artifacts["bwer_bwer_summary"].exists()
    assert artifacts["bwer_binary_error_bwer_summary"].exists()
    collected = collect_reben_sensor_audit_outputs(root)
    assert collected["aggregate_metrics"].exists()
    assert collected["bwer_summary"].exists()
    assert collected["audit_report"].exists()
    assert collected["sensor_mode_summary"].exists()
    risk_names = {row["risk_name"] for row in read_csv_rows(collected["bwer_summary"])}
    assert {"risk_bce", "risk_binary_error"} <= risk_names


def test_croma_feature_shape_handling() -> None:
    outputs = {
        "SAR_GAP": np.ones((2, 768), dtype=np.float32),
        "optical_GAP": np.ones((2, 4, 3), dtype=np.float32),
        "joint_GAP": np.ones((2, 1536), dtype=np.float32),
    }
    assert croma_embedding_from_output(outputs, "S1").shape == (2, 768)
    assert croma_embedding_from_output(outputs, "S2").shape == (2, 12)
    assert croma_embedding_from_output(outputs, "S1+S2").shape == (2, 1536)
    with pytest.raises(ValueError, match="missing"):
        croma_embedding_from_output({}, "S2")


def test_configilm_reben_adapter_splits_s1_s2_channels() -> None:
    fake = _FakeRebenDataset(count=2)
    fake.metadata[0]["contains_seasonal_snow"] = True
    s1 = ConfigILMRebenDatasetAdapter("lmdb", "meta.parquet", "snow.parquet", split="train", sensor_mode="S1", dataset=fake)
    s2 = ConfigILMRebenDatasetAdapter("lmdb", "meta.parquet", "snow.parquet", split="train", sensor_mode="S2", dataset=fake)
    both = ConfigILMRebenDatasetAdapter("lmdb", "meta.parquet", "snow.parquet", split="train", sensor_mode="S1+S2", dataset=fake)
    assert s1.load_sample(0)["image"].shape == (2, 16, 16)
    assert s1.load_sample(0)["metadata"]["cloud_snow_shadow"] == "cloud_snow_shadow"
    assert s2.load_sample(0)["image"].shape == (12, 16, 16)
    paired = both.load_sample(0)["image"]
    assert paired["S1"].shape == (2, 16, 16)
    assert paired["S2"].shape == (12, 16, 16)
    assert both.get_label_vector(0).shape == (19,)
    bifold_s2 = ConfigILMRebenDatasetAdapter(
        "lmdb",
        "meta.parquet",
        "snow.parquet",
        split="val",
        sensor_mode="S2",
        channel_profile="bifold_resnet101",
        dataset=_FakeRebenDataset(count=2),
    )
    # BIFOLD tests use a fake 14-channel fixture, but formal BIFOLD S2 expects
    # ConfigILM to return 10 channels. The adapter should reject mismatches.
    with pytest.raises(Exception, match="Expected BIFOLD S2 10-channel"):
        bifold_s2.load_sample(0)


def test_configilm_reben_adapter_uses_ben2_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeBEN2DataSet:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.metadata = [{"patch_id": "patch_a", "country": "DE", "split": "val"}]

        def __len__(self):
            return 1

        def __getitem__(self, index: int):
            image = np.zeros((12, 16, 16), dtype=np.float32)
            label = np.zeros(19, dtype=np.float32)
            label[0] = 1.0
            return image, label, "patch_a"

    tmp_root = Path(".pytest_tmp/test_reben_ben2_signature")
    tmp_root.mkdir(parents=True, exist_ok=True)
    root = tmp_root / "root"
    root.mkdir(exist_ok=True)
    (root / "BigEarthNetEncoded.lmdb").write_text("", encoding="utf-8")
    labels = tmp_root / "labels.parquet"
    snow = tmp_root / "snow.parquet"
    labels.write_text("", encoding="utf-8")
    snow.write_text("", encoding="utf-8")
    monkeypatch.setattr("rsfm_fairness_audit.adapters.reben.import_configilm_reben_dataset_class", lambda: (_FakeBEN2DataSet, {"module": "m", "class": "BEN2DataSet"}))
    adapter = ConfigILMRebenDatasetAdapter(
        root / "BigEarthNetEncoded.lmdb",
        labels,
        snow,
        split="val",
        sensor_mode="S2",
        max_samples=5,
    )
    sample = adapter.load_sample(0)
    assert captured["root_dir"] == root
    assert captured["split"] == "val"
    assert captured["img_size"] == (12, 120, 120)
    assert captured["return_patchname"] is True
    assert captured["new_label_file"] == labels
    assert captured["max_img_idx"] == 5
    assert sample["metadata"]["patch_id"] == "patch_a"


def test_configilm_reben_import_prefers_ben2_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeBEN2DataSet:
        pass

    module = types.SimpleNamespace(BEN2DataSet=_FakeBEN2DataSet)

    def fake_import(name: str):
        if name == "configilm.extra.DataSets.BEN2_DataSet":
            return module
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fake_import)
    dataset_class, info = import_configilm_reben_dataset_class()
    assert dataset_class is _FakeBEN2DataSet
    assert info["qualified_name"] == "configilm.extra.DataSets.BEN2_DataSet.BEN2DataSet"


def test_resolve_reben_root_dir_handles_nested_lmdb() -> None:
    root, lmdb, notes = resolve_reben_root_dir(Path("/content/data/reben/BigEarthNetEncoded.lmdb"))
    assert root == Path("/content/data/reben")
    assert lmdb == Path("/content/data/reben/BigEarthNetEncoded.lmdb")
    assert notes

    nested_base = Path(".pytest_tmp/test_reben_nested/BigEarthNetEncoded.lmdb")
    (nested_base / "BigEarthNetEncoded.lmdb").mkdir(parents=True, exist_ok=True)
    root_nested, lmdb_nested, notes_nested = resolve_reben_root_dir(nested_base)
    assert root_nested == nested_base
    assert lmdb_nested == nested_base / "BigEarthNetEncoded.lmdb"
    assert any("nested" in note for note in notes_nested)


def test_lmdb_safetensors_adapter_loads_metadata_and_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "patch_id": "patch_a",
            "s1_name": "s1_a",
            "s2v1_name": "s2_a",
            "split": "train",
            "country": "DE",
            "labels": ["Broad-leaved forest"],
        }
    ]

    monkeypatch.setattr("rsfm_fairness_audit.adapters.reben.prepare_lmdb_safetensors_metadata", lambda *args, **kwargs: rows)

    def fake_load_key(self, key: str):
        if key == "s1_a":
            return {"VV": np.ones((4, 4), dtype=np.float32), "VH": np.ones((4, 4), dtype=np.float32) * 2}
        if key == "s2_a":
            return {band: np.ones((4, 4), dtype=np.float32) * index for index, band in enumerate(("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"))}
        raise KeyError(key)

    monkeypatch.setattr(LmdbSafetensorsRebenDatasetAdapter, "_load_key", fake_load_key)
    adapter = LmdbSafetensorsRebenDatasetAdapter("lmdb", "meta.parquet", split="train", sensor_mode="S1+S2")
    sample = adapter.load_sample(0)
    assert sample["image"]["S1"].shape == (2, 4, 4)
    assert sample["image"]["S2"].shape == (12, 4, 4)
    assert sample["metadata"]["label_vector"][8] == 1
    assert adapter.loader_info()["payload_format"] == "safetensors"


def test_reben_labels_to_multihot_handles_class_names_and_one_hot() -> None:
    by_name = reben_labels_to_multihot(["Broad-leaved forest", "Marine waters"])
    assert by_name.shape == (19,)
    assert by_name[8] == 1
    assert by_name[18] == 1
    one_hot = reben_labels_to_multihot([1] + [0] * 18)
    assert one_hot[0] == 1
    assert int(one_hot.sum()) == 1


def test_lmdb_safetensors_adapter_reuses_env_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lmdb")
    opened = []

    class _FakeEnv:
        pass

    def fake_open(path, **kwargs):
        opened.append((path, kwargs))
        return _FakeEnv()

    monkeypatch.setattr("lmdb.open", fake_open)
    _LMDB_ENV_CACHE.clear()
    a = LmdbSafetensorsRebenDatasetAdapter(".pytest_tmp/reben_cache_test", "meta.parquet", split="train", sensor_mode="S1")
    b = LmdbSafetensorsRebenDatasetAdapter(".pytest_tmp/reben_cache_test", "meta.parquet", split="val", sensor_mode="S1")
    assert a._lmdb_env() is b._lmdb_env()
    assert len(opened) == 1
    assert opened[0][1]["max_readers"] == 512
    _LMDB_ENV_CACHE.clear()


def test_bifold_resnet101_runner_with_mock_model() -> None:
    pytest.importorskip("torch")

    class _FakeBifoldDataset(_FakeRebenDataset):
        def __getitem__(self, index: int):
            image, label = super().__getitem__(index)
            return image[:2], label

    root = Path("outputs/test_reben_bifold_runner")
    eval_dataset = ConfigILMRebenDatasetAdapter(
        "lmdb",
        "meta.parquet",
        "snow.parquet",
        split="val",
        sensor_mode="S1",
        channel_profile="bifold_resnet101",
        dataset=_FakeBifoldDataset(count=4),
    )
    artifacts = run_bifold_resnet101_reben_inference(
        eval_dataset=eval_dataset,
        model_runner=BifoldResNet101Runner("BIFOLD-BigEarthNetv2-0/resnet101-s1-v0.2.0", model=_MockBifoldModel()),
        output_dir=root,
        run_name="bifold_resnet101_s1",
        run_labels=RebenRunLabels("bifold_resnet101", "resnet101_v0.2.0", "S1", "official_supervised_reference"),
        class_names=[f"class_{index}" for index in range(19)],
        batch_size=2,
        run_bwer=False,
    )
    assert artifacts["predictions"].exists()
    assert artifacts["logits"].exists()
    assert len(read_csv_rows(artifacts["predictions"])) == 4 * 19


def test_bifold_reference_validation_accepts_official_ids_or_existing_local_path() -> None:
    local = Path("outputs/test_reben_local_bifold_model")
    local.mkdir(parents=True, exist_ok=True)
    rows = validate_bifold_resnet101_refs(
        {
            "S1": "BIFOLD-BigEarthNetv2-0/resnet101-s1-v0.2.0",
            "S2": str(local),
            "S1+S2": "BIFOLD-BigEarthNetv2-0/resnet101-all-v0.2.0",
        }
    )
    assert [row["status"] for row in rows] == ["ok", "ok", "ok"]
    bad = validate_bifold_resnet101_refs({"S1": "BIFOLD-BigEarthNetv2-0/resnet101-s1-v0.1.1"})
    assert bad[0]["status"] == "invalid"


def test_reben_contract_validation_reports_missing_and_manifest() -> None:
    root = Path("outputs/test_reben_contract_validation")
    root.mkdir(parents=True, exist_ok=True)
    (root / "source_verification_report.md").write_text("verified", encoding="utf-8")
    artifacts = validate_reben_sensor_audit_contract(root)
    assert artifacts["contract_validation"].exists()
    assert artifacts["contract_report"].exists()
    payload = json.loads(artifacts["archive_manifest"].read_text(encoding="utf-8"))
    assert payload["ready_for_interpretation"] is False
    assert "dataset_preflight.json" in payload["missing_artifacts"]


def test_croma_reben_frozen_probe_runner_with_mock_model() -> None:
    pytest.importorskip("torch")
    root = Path("outputs/test_reben_croma_runner")
    train = ConfigILMRebenDatasetAdapter("lmdb", "meta.parquet", "snow.parquet", split="train", sensor_mode="S2", dataset=_FakeRebenDataset(count=6))
    val = ConfigILMRebenDatasetAdapter("lmdb", "meta.parquet", "snow.parquet", split="val", sensor_mode="S2", dataset=_FakeRebenDataset(count=4))
    adapter = CROMAAdapter(model=_MockCromaModel(), image_size=16, batch_size=2)
    artifacts = run_croma_reben_frozen_probe(
        train_dataset=train,
        eval_dataset=val,
        croma_adapter=adapter,
        output_dir=root,
        run_name="croma_s2",
        run_labels=RebenRunLabels("croma", "croma_base", "S2", "frozen_encoder_linear_probe"),
        class_names=[f"class_{index}" for index in range(19)],
        batch_size=2,
        probe_epochs=2,
        run_bwer=False,
    )
    assert artifacts["predictions"].exists()
    assert artifacts["run_metadata"].exists()
    rows = read_csv_rows(artifacts["predictions"])
    assert len(rows) == 4 * 19
    assert rows[0]["sensor_mode"] == "S2"


def test_reben_multilabel_bwer_cli_accepts_long_label_expanded_predictions() -> None:
    rows = [
        {
            "sample_id": "a",
            "class_label": "forest",
            "label_true": 1,
            "label_probability": 0.9,
            "risk_bce": 0.1,
            "risk_binary_error": 0,
            "correct": 1,
            "country": "DE",
            "warnings": "x" * 140000,
        },
        {
            "sample_id": "b",
            "class_label": "forest",
            "label_true": 0,
            "label_probability": 0.4,
            "risk_bce": 0.5,
            "risk_binary_error": 0,
            "correct": 1,
            "country": "FR",
            "warnings": "x" * 140000,
        },
    ]
    root = Path("outputs/test_reben_cli_long_fields")
    root.mkdir(parents=True, exist_ok=True)
    predictions = root / "predictions.csv"
    write_csv(predictions, rows)
    out = root / "bwer"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rsfm_fairness_audit.cli",
            "run-reben-multilabel-bwer",
            "--predictions",
            str(predictions),
            "--output-dir",
            str(out),
            "--model-name",
            "croma_s2",
            "--min-support",
            "1",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
    )
    assert (out / "bwer_summary.csv").exists()
    assert json.loads((out / "warnings.json").read_text(encoding="utf-8")) is not None
