from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.io import write_csv
from rsfm_fairness_audit.optimization_phase1 import distribution_metrics, synthetic_counterexamples
from rsfm_fairness_audit.paired_probability_diagnostics import paired_label_probability_diagnostics
from rsfm_fairness_audit.reben_phase1_postprocess import (
    build_final_optimization_evidence,
    postprocess_label_budget,
    postprocess_paired_sensor_shift,
)
from rsfm_fairness_audit.reben_phase1_runners import validate_cache_contract, validate_paired_cache_contract
from rsfm_fairness_audit.reben_terramind_campaign import train_streaming_multilabel_probe


def test_distribution_metrics_reports_variance_and_tail_separately() -> None:
    result = distribution_metrics({"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.4})
    assert result["weighted_std"] > 0
    assert result["weighted_variance"] == pytest.approx(result["weighted_std"] ** 2)
    assert result["geobwer_beta_0_10"] > 0
    assert len(result["profile"]) == 4
    assert result["weighted_sd"] == result["weighted_std"]
    assert result["worst_minus_mean"] == result["worst_mean_gap"]
    assert result["geobwer"] == result["geobwer_beta_0_10"]


def test_probability_diagnostics_separate_threshold_and_collapse_signatures() -> None:
    targets = np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int8)
    id_prob = np.asarray([[0.10, 0.10], [0.20, 0.20], [0.80, 0.80], [0.90, 0.90]])
    ood_prob = np.asarray([[0.55, 0.50], [0.60, 0.50], [0.70, 0.50], [0.75, 0.50]])
    rows = paired_label_probability_diagnostics(
        targets, id_prob, ood_prob, np.asarray([0.5, 0.5]), ["threshold", "collapse"], seed=42,
    )
    assert rows[0]["diagnostic_signature"] == "threshold_shift_dominant"
    assert rows[0]["delta_auroc"] == pytest.approx(0.0)
    assert rows[1]["diagnostic_signature"] == "representation_collapse_signature"
    assert rows[1]["diagnostic_is_causal_attribution"] is False


def test_counterexamples_include_levelling_down() -> None:
    rows = {row["scenario"]: row for row in synthetic_counterexamples()}
    before = rows["levelling_down_before"]
    after = rows["levelling_down_after"]
    assert after["mean_risk"] > before["mean_risk"]
    assert after["geobwer_beta_0_10"] < before["geobwer_beta_0_10"]


def _write_split(root: Path, split: str, ids: list[str]) -> None:
    target = root / split
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "embeddings.npy", np.zeros((len(ids), 4), dtype=np.float32))
    np.save(target / "labels.npy", np.zeros((len(ids), 19), dtype=np.int8))
    rows = [
        {
            "sample_id": sample_id,
            "country": "DEU",
            "source_tile_id": f"tile_{sample_id}",
            "independent_unit_id": f"unit_{sample_id}",
        }
        for sample_id in ids
    ]
    (target / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    sensor = "S1" if root.name.lower() == "s1" else "S2"
    (target / "embedding_cache_manifest.json").write_text(json.dumps({
        "schema": "geobwer.reben.embedding_cache.v1",
        "cache_signature": f"{sensor}:{split}",
        "lineage": {"adapter": {
            "model_name": "terramind_v1_base", "model_release": "v1",
            "checkpoint_expected_sha256": "a" * 64, "checkpoint_actual_sha256": "a" * 64,
            "sensor_mode": sensor, "modalities": [sensor], "input_profile": f"reben_{sensor.lower()}",
            "s1_unit_policy": "db" if sensor == "S1" else "none",
        }},
        "embedding_shape": [len(ids), 4], "labels_shape": [len(ids), 19],
    }), encoding="utf-8")


def test_embedding_cache_contract_rejects_sample_leakage() -> None:
    test_root = Path("work/test_optimization_phase1_contract")
    _write_split(test_root, "train", ["a", "b"])
    _write_split(test_root, "val", ["c"])
    _write_split(test_root, "test", ["d"])
    contract = validate_cache_contract(test_root)
    assert contract["sample_overlap_counts"] == {"train__val": 0, "train__test": 0, "val__test": 0}

    _write_split(test_root, "test", ["a"])
    try:
        validate_cache_contract(test_root)
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected leakage rejection")


def test_paired_cache_contract_accepts_aligned_test_samples() -> None:
    root = Path("work/test_optimization_phase1_paired")
    s2 = root / "s2"
    s1 = root / "s1"
    for cache_root, prefix in ((s2, "s2"), (s1, "s1")):
        _write_split(cache_root, "train", [f"{prefix}_train"])
        _write_split(cache_root, "val", [f"{prefix}_val"])
        _write_split(cache_root, "test", ["paired_a", "paired_b"])
    contract = validate_paired_cache_contract(s2, s1)
    assert contract["status"] == "formal_ready"
    assert contract["paired_test_sample_count"] == 2


def test_indexed_probe_reuses_full_embedding_cache() -> None:
    pytest.importorskip("torch")
    root = Path("work/test_optimization_phase1_indexed_probe")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(12, 5)).astype(np.float32)
    train_y = (rng.random((12, 19)) > 0.75).astype(np.int8)
    val_x = rng.normal(size=(4, 5)).astype(np.float32)
    test_x = rng.normal(size=(5, 5)).astype(np.float32)
    np.save(root / "train_x.npy", train_x)
    np.save(root / "train_y.npy", train_y)
    np.save(root / "val_x.npy", val_x)
    np.save(root / "test_x.npy", test_x)
    probabilities, checkpoint = train_streaming_multilabel_probe(
        root / "train_x.npy",
        root / "train_y.npy",
        {"validation": root / "val_x.npy", "test": root / "test_x.npy"},
        root / "probe",
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=3,
        device="cpu",
        seed=7,
        cache_signature="indexed-test-v1",
        train_indices=np.asarray([0, 2, 4, 6, 8, 10]),
    )
    assert checkpoint.exists()
    assert np.load(probabilities["validation"]).shape == (4, 19)
    assert np.load(probabilities["test"]).shape == (5, 19)


def test_label_budget_postprocess_audits_nested_grid_and_writes_figures(tmp_path: Path) -> None:
    output = tmp_path / "label_budget"
    rows = []
    selections = []
    units = ["u1", "u2", "u3", "u4"]
    for seed in (42, 73):
        for budget, count in ((0.5, 2), (1.0, 4)):
            rows.append({
                "seed": seed, "budget_fraction": budget, "selected_independent_units": count,
                "total_independent_units": 4, "selected_samples": count, "test_samples": 8,
                "macro_ap": 0.6 + 0.1 * budget, "macro_f1": 0.5 + 0.1 * budget,
                "mean_risk": 0.3 - 0.1 * budget, "tail_risk_beta_0_10": 0.4 - 0.1 * budget,
                "geobwer_beta_0_10": 0.1,
            })
            selections.extend({"seed": seed, "budget_fraction": budget, "independent_unit_id": unit} for unit in units[:count])
    write_csv(output / "label_budget_curves.csv", rows)
    write_csv(output / "nested_budget_unit_selections.csv", selections)
    (output / "label_budget_manifest.json").write_text(json.dumps({
        "status": "complete", "validation_and_test_fixed": True, "test_used_for_selection": False,
    }), encoding="utf-8")
    artifacts = postprocess_label_budget(output, expected_budgets=(0.5, 1.0), expected_seeds=(42, 73))
    audit = json.loads(artifacts["audit"].read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert (output / "figures/label_budget_curves.png").is_file()
    assert (output / "figures/label_budget_curves.pdf").is_file()


def test_paired_shift_postprocess_requires_complete_paired_outputs(tmp_path: Path) -> None:
    output = tmp_path / "paired"
    cache_root = tmp_path / "s2_cache"
    (cache_root / "test").mkdir(parents=True)
    rng = np.random.default_rng(11)
    targets = (rng.random((40, 19)) > 0.65).astype(np.int8)
    np.save(cache_root / "test" / "labels.npy", targets)
    summaries = []
    deltas = []
    label_deltas = []
    country_deltas = []
    for seed in (42, 73):
        seed_dir = output / f"seed_{seed}"
        probe_dir = seed_dir / "s2_trained_probe"
        probe_dir.mkdir(parents=True)
        id_probability = np.clip(0.15 + 0.70 * targets + rng.normal(0, 0.08, targets.shape), 0, 1)
        ood_probability = np.clip(id_probability - 0.12 + rng.normal(0, 0.05, targets.shape), 0, 1)
        np.save(probe_dir / "s2_id_test_probabilities.npy", id_probability)
        np.save(probe_dir / "s1_ood_test_probabilities.npy", ood_probability)
        write_csv(seed_dir / "s2_validation_locked_thresholds.csv", [
            {"class_index": index, "class_label": f"label_{index}", "threshold": 0.5}
            for index in range(19)
        ])
        for domain, offset in (("ID", 0.0), ("OOD", 0.1)):
            summaries.append({
                "seed": seed, "domain": domain, "macro_ap": 0.8 - offset,
                "macro_f1": 0.7 - offset, "mean_risk": 0.2 + offset,
                "tail_risk_beta_0_10": 0.3 + offset, "geobwer_beta_0_10": 0.1,
            })
        deltas.append({
            "seed": seed, "delta_mean_risk": 0.1, "delta_tail_risk": 0.1,
            "delta_geobwer": 0.0, "tail_acceleration_minus_mean": 0.0,
            "levelling_down_flag": False,
        })
        label_deltas.append({"seed": seed, "slice_axis": "label", "slice_value": "forest", "delta_risk": 0.05})
        country_deltas.append({"seed": seed, "slice_axis": "country", "slice_value": "DEU", "delta_risk": 0.1})
    write_csv(output / "paired_shift_seed_panel.csv", summaries)
    write_csv(output / "paired_shift_delta_seed_panel.csv", deltas)
    write_csv(output / "paired_shift_label_deltas.csv", label_deltas)
    write_csv(output / "paired_shift_country_deltas.csv", country_deltas)
    (output / "paired_shift_panel_manifest.json").write_text(json.dumps({
        "status": "complete", "same_s2_trained_head_within_seed": True,
        "test_used_for_selection": False, "effective_robustness_claimed": False,
    }), encoding="utf-8")
    (output / "paired_shift_preflight.json").write_text(json.dumps({
        "status": "formal_ready", "paired_sample_ids_targets_and_metadata": True,
        "s2_contract": {"root": str(cache_root)},
    }), encoding="utf-8")
    artifacts = postprocess_paired_sensor_shift(output, expected_seeds=(42, 73))
    assert json.loads(artifacts["audit"].read_text(encoding="utf-8"))["status"] == "pass"
    assert (output / "figures/paired_sensor_shift_burden_carriers.pdf").is_file()
    assert (output / "paired_shift_probability_diagnostics_label_summary.csv").is_file()
    assert (output / "figures/paired_probability_diagnostics.pdf").is_file()


def test_final_evidence_manifest_stays_nonfinal_when_results_are_pending(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "optimization_1_7_validation.json").write_text(json.dumps({"passes": True}), encoding="utf-8")
    artifacts = build_final_optimization_evidence(
        Path.cwd(), base, tmp_path / "missing_label", tmp_path / "missing_paired",
        tmp_path / "final", allow_pending=True,
    )
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["status"] == "pending_required_results"
    assert manifest["finality"] is False
    assert manifest["scope"]["not_started"] == list(range(8, 18))
    label = tmp_path / "label_complete"
    paired = tmp_path / "paired_complete"
    label.mkdir()
    paired.mkdir()
    (label / "label_budget_result_audit.json").write_text(json.dumps({"status": "pass", "gates": {"ok": True}}), encoding="utf-8")
    (paired / "paired_shift_result_audit.json").write_text(json.dumps({"status": "pass", "gates": {
        "ok": True, "probability_diagnostics_complete": True,
        "probability_diagnostic_sources_complete": True,
    }}), encoding="utf-8")
    complete = build_final_optimization_evidence(
        Path.cwd(), base, label, paired, tmp_path / "complete_final",
    )
    complete_manifest = json.loads(complete["manifest"].read_text(encoding="utf-8"))
    assert complete_manifest["status"] == "complete"
    assert complete_manifest["finality"] is True


def test_final_evidence_distinguishes_completed_experiment_from_pending_diagnostics(tmp_path: Path) -> None:
    base = tmp_path / "base"
    label = tmp_path / "label"
    paired = tmp_path / "paired"
    for directory in (base, label, paired):
        directory.mkdir()
    (base / "optimization_1_7_validation.json").write_text(json.dumps({"passes": True}), encoding="utf-8")
    (label / "label_budget_result_audit.json").write_text(json.dumps({"status": "pass", "gates": {"legacy": True}}), encoding="utf-8")
    (paired / "paired_shift_result_audit.json").write_text(json.dumps({"status": "pass", "gates": {"legacy": True}}), encoding="utf-8")
    artifacts = build_final_optimization_evidence(
        Path.cwd(), base, label, paired, tmp_path / "final", allow_pending=True,
    )
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["formal_experiments_complete"] is True
    assert manifest["paired_probability_diagnostics_complete"] is False
    assert manifest["status"] == "pending_no_retraining_probability_diagnostics"
    assert manifest["finality"] is False
    assert "experiment_pass_diagnostics_pending" in artifacts["status"].read_text(encoding="utf-8")
