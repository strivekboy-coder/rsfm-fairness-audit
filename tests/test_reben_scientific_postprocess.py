from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.reben_scientific_postprocess import (
    FAMILIES,
    MODES,
    SEEDS,
    discover_reben_panel,
    run_reben_scientific_postprocess,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> list[Path]:
    sample_id: list[str] = []
    country: list[str] = []
    cluster: list[str] = []
    cardinality: list[int] = []
    for group, cluster_count in (("A", 5), ("B", 5), ("C", 1)):
        for cluster_index in range(cluster_count):
            for replicate in range(2):
                sample_id.append(f"{group}_{cluster_index}_{replicate}")
                country.append(group)
                cluster.append(f"{group}_tile_{cluster_index}")
                cardinality.append(1 + (cluster_index + replicate) % 5)
    n = len(sample_id)
    targets = np.zeros((n, 3), dtype=np.int8)
    targets[:, 0] = 1
    targets[np.asarray(cardinality) >= 3, 1] = 1
    targets[np.asarray(cardinality) >= 5, 2] = 1
    source_files: list[Path] = []
    family_offset = {"supervised_resnet50": 0.025, "croma": 0.015, "terramind": 0.0}
    mode_offset = {"s1": 0.04, "s2": 0.01, "s1_plus_s2": 0.0}
    for family in FAMILIES:
        for mode in MODES:
            for seed in SEEDS:
                run = root / family / mode / f"seed_{seed}"
                formal = run / "formal_outputs"
                formal.mkdir(parents=True)
                seed_offset = (seed % 7) * 0.0005
                probabilities = np.clip(
                    targets * 0.78 + (1 - targets) * 0.18
                    - family_offset[family]
                    - mode_offset[mode]
                    + seed_offset,
                    0.02,
                    0.98,
                )
                predictions = probabilities >= 0.5
                risks = np.mean(predictions != targets, axis=1).astype(float)
                # Preserve model/mode ordering even when hard labels are equal.
                risks += family_offset[family] + mode_offset[mode] + seed_offset
                risks += np.asarray([0.02 if value == "C" else 0.0 for value in country])
                rows = []
                for index in range(n):
                    rows.append(
                        {
                            "sample_id": sample_id[index],
                            "country": country[index],
                            "source_tile_id": cluster[index],
                            "risk": risks[index],
                            "label_cardinality": cardinality[index],
                            "snow_cloud_status": "clear" if index % 2 == 0 else "affected",
                            "protocol_hash": "shared_protocol",
                            "metric_version": "geobwer_fractional_1.1",
                        }
                    )
                write_csv(formal / "formal_audit_table.csv", rows)
                np.savez_compressed(
                    formal / "probabilities.npz",
                    sample_id=np.asarray(sample_id),
                    probabilities=probabilities,
                    targets=targets,
                    class_names=np.asarray(["x", "y", "z"]),
                    thresholds=np.asarray([0.5, 0.5, 0.5]),
                )
                (formal / "formal_output_manifest.json").write_text(
                    json.dumps({"schema": "test", "rows": n}),
                    encoding="utf-8",
                )
                write_csv(
                    run / "metrics_summary.csv",
                    [
                        {
                            "macro_ap": 0.8,
                            "micro_ap": 0.81,
                            "macro_f1": 0.7,
                            "micro_f1": 0.71,
                        }
                    ],
                )
                write_csv(
                    run / "geobwer" / "geobwer_summary.csv",
                    [{"validity": "descriptive_only", "geobwer": 0.1}],
                )
                write_csv(
                    run / "uncertainty_extensions" / "uncertainty_summary.csv",
                    [{"method": "crc", "coverage": 0.9}],
                )
                source_files.extend(
                    [
                        formal / "formal_audit_table.csv",
                        formal / "probabilities.npz",
                        formal / "formal_output_manifest.json",
                    ]
                )
    return source_files


def test_discovers_and_postprocesses_full_panel_without_mutating_sources(tmp_path: Path) -> None:
    panel = tmp_path / "review" / "nested" / "reben_full_panel"
    source_files = _fixture(panel)
    before = {path: _sha(path) for path in source_files}

    discovered_root, runs = discover_reben_panel(tmp_path / "review")
    assert discovered_root == panel
    assert len(runs) == 27

    artifacts = run_reben_scientific_postprocess(
        tmp_path / "review",
        tmp_path / "postprocessed",
        min_units=2,
        cluster_thresholds=(2, 3, 5),
        n_bootstrap=100,
    )

    assert len(read_csv_rows(artifacts["unified_metrics"])) == 27
    assert len(read_csv_rows(artifacts["three_seed_summary"])) == 9
    comparisons = read_csv_rows(artifacts["paired_comparisons"])
    assert len(comparisons) == 54
    assert {row["interpretation_scope"] for row in comparisons} == {
        "adapted_model_pipeline_under_common_evaluation_contract"
    }
    assert {row["causal_backbone_attribution"] for row in comparisons} == {"False"}
    assert len(read_csv_rows(artifacts["beta_profile"])) == 81
    assert len(read_csv_rows(artifacts["support_sensitivity"])) == 81
    assert len(read_csv_rows(artifacts["composition_sensitivity"])) == 54
    partial = read_csv_rows(artifacts["partial_identification"])
    assert len(partial) == 27
    assert {row["excluded_countries"] for row in partial} == {"C"}
    assert all(float(row["fixed_universe_partial_upper"]) >= float(row["fixed_universe_partial_lower"]) for row in partial)
    support = json.loads(artifacts["support_contract"].read_text(encoding="utf-8"))
    assert support["fixed_countries"] == ["A", "B", "C"]
    assert support["supported_countries"] == ["A", "B"]
    assert before == {path: _sha(path) for path in source_files}


def test_metrics_are_recomputed_when_legacy_summary_is_absent(tmp_path: Path) -> None:
    panel = tmp_path / "panel"
    _fixture(panel)
    missing = panel / "croma" / "s1" / "seed_42" / "metrics_summary.csv"
    missing.unlink()
    artifacts = run_reben_scientific_postprocess(
        panel,
        tmp_path / "out",
        min_units=2,
        cluster_thresholds=(2,),
        n_bootstrap=100,
    )
    row = next(
        item
        for item in read_csv_rows(artifacts["unified_metrics"])
        if item["run_id"] == "croma__s1__seed_42"
    )
    assert np.isfinite(float(row["macro_f1"]))
    assert np.isfinite(float(row["micro_ap"]))
