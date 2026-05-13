from __future__ import annotations

from pathlib import Path
import json

from rsfm_fairness_audit.cli import main
from rsfm_fairness_audit.io import read_csv_rows, write_csv
from rsfm_fairness_audit.slice_support import evaluate_slice_support


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(10):
        rows.append(
            {
                "dataset": "ben_ge_800",
                "model": "croma",
                "task": "classification",
                "split": "all",
                "unit_id": f"a-{index}",
                "sensor_mode": "sar" if index < 5 else "optical",
                "climatezone": "climatezone_0",
                "class_label": "0" if index < 5 else "1",
                "score": 1.0,
            }
        )
    rows.extend(
        [
            {
                "dataset": "ben_ge_800",
                "model": "croma",
                "task": "classification",
                "split": "all",
                "unit_id": "a-extra",
                "sensor_mode": "optical",
                "climatezone": "climatezone_0",
                "class_label": "2",
                "score": 1.0,
            },
            {
                "dataset": "ben_ge_800",
                "model": "croma",
                "task": "classification",
                "split": "all",
                "unit_id": "b-0",
                "sensor_mode": "both",
                "climatezone": "climatezone_26",
                "class_label": "0",
                "score": 0.0,
            },
            {
                "dataset": "ben_ge_800",
                "model": "croma",
                "task": "classification",
                "split": "all",
                "unit_id": "b-1",
                "sensor_mode": "both",
                "climatezone": "climatezone_26",
                "class_label": "0",
                "score": 0.0,
            },
        ]
    )
    return rows


def test_slice_support_recommends_against_sparse_balanced_candidate() -> None:
    artifacts = evaluate_slice_support(
        _rows(),
        "ben_ge_800",
        "croma",
        "classification",
        "outputs/test_slice_support_sparse",
        candidates=["climatezone|class_label", "sensor_mode|class_label"],
        min_samples_per_slice=1,
        min_units_required=1,
        min_slices_required=2,
    )
    recommendations = {row["candidate"]: row for row in read_csv_rows(artifacts["recommendations"])}
    climate = recommendations["BWER(climatezone | class_label)"]
    assert climate["recommendation"] == "caution"
    assert float(climate["missing_slice_balance_ratio"]) > 0.30
    assert climate["preferred_bwer"] == "balanced"
    assert Path(artifacts["report"]).exists()


def test_slice_support_marks_missing_candidate_not_recommended() -> None:
    artifacts = evaluate_slice_support(
        _rows(),
        "ben_ge_800",
        "croma",
        "classification",
        "outputs/test_slice_support_missing",
        candidates=["country|class_label"],
    )
    row = read_csv_rows(artifacts["recommendations"])[0]
    assert row["recommendation"] == "not_recommended"
    assert "missing slice column" in row["reason"]


def test_preflight_bwer_cli_reads_predictions(monkeypatch) -> None:
    root = Path("outputs/test_slice_support_cli")
    root.mkdir(parents=True, exist_ok=True)
    predictions = root / "predictions.csv"
    write_csv(
        predictions,
        [
            {"sample_id": "s1", "label": "0", "prediction": "0", "region": "A", "correct": 1},
            {"sample_id": "s2", "label": "1", "prediction": "0", "region": "B", "correct": 0},
        ],
    )
    output = root / "support"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "preflight-bwer",
            "--predictions",
            str(predictions),
            "--dataset",
            "bigearthnet",
            "--model",
            "dofa",
            "--task",
            "classification",
            "--candidate",
            "region|class_label",
            "--output-dir",
            str(output),
            "--min-samples-per-slice",
            "1",
            "--min-units-required",
            "1",
        ],
    )
    main()
    assert (output / "slice_support_recommendations.csv").exists()
    assert read_csv_rows(output / "slice_support_recommendations.csv")[0]["candidate"] == "BWER(region | class_label)"


def test_preflight_bwer_cli_reads_segmentation_metrics(monkeypatch) -> None:
    root = Path("outputs/test_slice_support_cli_segmentation")
    root.mkdir(parents=True, exist_ok=True)
    metrics = root / "segmentation_metrics.csv"
    write_csv(
        metrics,
        [
            {"sample_id": "s1", "event_id": "e1", "country": "C1", "water_iou": 0.7, "positive_pixels": 1500},
            {"sample_id": "s2", "event_id": "e2", "country": "C2", "water_iou": 0.2, "positive_pixels": 1600},
        ],
    )
    output = root / "support"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rsfm-audit",
            "preflight-bwer",
            "--segmentation-metrics",
            str(metrics),
            "--dataset",
            "sen1floods11",
            "--model",
            "prithvi",
            "--task",
            "segmentation",
            "--candidate",
            "event_id|class_label",
            "--output-dir",
            str(output),
            "--min-samples-per-slice",
            "1",
            "--min-units-required",
            "1",
        ],
    )
    main()
    rows = read_csv_rows(output / "slice_support_recommendations.csv")
    assert rows[0]["candidate"] == "BWER(event_id | class_label)"


def test_prithvi_bwer_colab_notebook_parses() -> None:
    path = Path("notebooks/prithvi_sen1floods11_bwer_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "preflight-bwer" in source
    assert "evaluate-bwer" in source
    assert "prithvi_sen1floods11_bwer" in source
