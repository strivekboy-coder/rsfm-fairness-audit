from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from fmow_sanity_common import ensure_extracted_dataset, python_module_cmd, read_csv, run_command, write_csv, write_json


def _tiny_manifest(rows: list[dict[str, str]], classes: int, samples_per_class: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = row.get("category", "")
        if label:
            grouped[label].append(row)
    selected_labels = sorted(label for label, items in grouped.items() if len(items) >= samples_per_class)[:classes]
    if len(selected_labels) < max(2, classes):
        raise ValueError(f"Not enough classes with at least {samples_per_class} samples for tiny overfit.")
    out: list[dict[str, str]] = []
    for label in selected_labels:
        for idx, row in enumerate(grouped[label][:samples_per_class]):
            train = dict(row)
            train.setdefault("split_original", train.get("split", ""))
            train["split"] = "train"
            train["split_protocol"] = "tiny_overfit_sanity"
            train["sample_id"] = f"tiny_train_{label}_{idx}_{train.get('sample_id', train.get('image_id', idx))}"
            out.append(train)

            val = dict(row)
            val.setdefault("split_original", val.get("split", ""))
            val["split"] = "val"
            val["split_protocol"] = "tiny_overfit_sanity"
            val["sample_id"] = f"tiny_val_{label}_{idx}_{val.get('sample_id', val.get('image_id', idx))}"
            out.append(val)
    return out


def _history_to_csv(output_dir: Path) -> None:
    debug_path = output_dir / "model_debug.json"
    if not debug_path.exists():
        return
    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    if isinstance(history, list) and history:
        write_csv(output_dir / "tiny_overfit_history.csv", [dict(row) for row in history])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny fMoW-Sentinel ResNet overfit sanity check in Colab.")
    parser.add_argument("--prepared-dataset-zip", type=Path)
    parser.add_argument("--extract-dir", type=Path, default=Path("/content/data/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/outputs/fmow_tiny_overfit_sanity_resnet50"))
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = ensure_extracted_dataset(args.prepared_dataset_zip, args.extract_dir, args.manifest)
    data_root = args.data_root or manifest.parent
    tiny_rows = _tiny_manifest(read_csv(manifest), args.classes, args.samples_per_class)
    tiny_manifest = args.output_dir / "tiny_overfit_manifest.csv"
    write_csv(tiny_manifest, tiny_rows)

    run_command(
        python_module_cmd(
            "rsfm_fairness_audit.cli",
            [
                "run-fmow-sentinel-classification",
                "--metadata-csv",
                str(tiny_manifest),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(args.output_dir),
                "--model",
                "resnet50",
                "--train-split",
                "train",
                "--eval-split",
                "val",
                "--split-protocol",
                "custom_stratified_subset",
                "--eval-scope",
                "tiny_overfit_eval",
                "--image-size",
                "96",
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(args.learning_rate),
                "--seed",
                str(args.seed),
            ],
        )
    )
    _history_to_csv(args.output_dir)
    metrics_path = args.output_dir / "metrics_summary.csv"
    metrics = read_csv(metrics_path)[0] if metrics_path.exists() and read_csv(metrics_path) else {}
    write_json(
        args.output_dir / "tiny_overfit_metadata.json",
        {
            "sanity_type": "tiny_overfit_sanity",
            "formal_result": False,
            "source_manifest": str(manifest),
            "tiny_manifest": str(tiny_manifest),
            "data_root": str(data_root),
            "classes": args.classes,
            "samples_per_class": args.samples_per_class,
            "epochs": args.epochs,
            "seed": args.seed,
            "final_accuracy": metrics.get("accuracy", ""),
            "final_macro_f1": metrics.get("macro_f1", ""),
        },
    )
    report = [
        "# fMoW-Sentinel Tiny Overfit Sanity",
        "",
        "This diagnostic checks the ResNet training loop, label mapping, and loss plumbing on a tiny repeated train/eval subset.",
        "It is not a scientific result and should not be compared to the formal location-disjoint runs.",
        "",
        f"- source manifest: `{manifest}`",
        f"- tiny manifest: `{tiny_manifest}`",
        f"- classes: {args.classes}",
        f"- samples per class: {args.samples_per_class}",
        f"- epochs: {args.epochs}",
        f"- final accuracy: {metrics.get('accuracy', '')}",
        f"- final macro-F1: {metrics.get('macro_f1', '')}",
        "",
        "See `tiny_overfit_history.csv` when available.",
    ]
    (args.output_dir / "tiny_overfit_report.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

