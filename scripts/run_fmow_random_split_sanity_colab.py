from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fmow_sanity_common import (
    ensure_extracted_dataset,
    first_row,
    python_module_cmd,
    read_csv,
    read_csv_from_zip,
    run_command,
    write_csv,
    write_json,
)


PRIMARY_SLICES = ("country", "region", "latitude_band", "season", "category")


def _random_split(rows: list[dict[str, str]], val_fraction: float, seed: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    n_val = int(round(len(rows) * val_fraction))
    val_indices = set(int(i) for i in order[:n_val])
    out: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item.setdefault("split_original", item.get("split", ""))
        item["split"] = "val" if idx in val_indices else "train"
        item["split_protocol"] = "random_split_sanity"
        out.append(item)
    return out


def _bwer_slice_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path) if path.exists() else []
    return [row for row in rows if row.get("slice_variable") in PRIMARY_SLICES and not row.get("balance_variable")]


def _summary_rows(random_dir: Path, location_zip: Path | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    random_metrics = first_row(random_dir / "metrics_summary.csv")
    random_bwer = _bwer_slice_rows(random_dir / "bwer" / "bwer_summary.csv")
    rows.append(
        {
            "run_name": "random_split_sanity",
            "protocol": "random_split_sanity",
            "accuracy": random_metrics.get("accuracy", ""),
            "balanced_accuracy": random_metrics.get("balanced_accuracy", ""),
            "macro_f1": random_metrics.get("macro_f1", ""),
            "top5_accuracy": random_metrics.get("top5_accuracy", ""),
            "bwer_slices": ";".join(f"{row.get('slice_variable')}={row.get('bwer')}" for row in random_bwer),
        }
    )
    if location_zip is not None and location_zip.exists():
        loc_metrics = read_csv_from_zip(location_zip, "metrics_summary.csv")
        loc_bwer = [
            row
            for row in read_csv_from_zip(location_zip, "bwer_summary.csv")
            if row.get("slice_variable") in PRIMARY_SLICES and not row.get("balance_variable")
        ]
        first = loc_metrics[0] if loc_metrics else {}
        rows.append(
            {
                "run_name": "location_disjoint_formal_resnet50",
                "protocol": "location_disjoint",
                "accuracy": first.get("accuracy", ""),
                "balanced_accuracy": first.get("balanced_accuracy", ""),
                "macro_f1": first.get("macro_f1", ""),
                "top5_accuracy": first.get("top5_accuracy", ""),
                "bwer_slices": ";".join(f"{row.get('slice_variable')}={row.get('bwer')}" for row in loc_bwer),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fMoW-Sentinel random split sanity contrast in Colab.")
    parser.add_argument("--prepared-dataset-zip", type=Path)
    parser.add_argument("--extract-dir", type=Path, default=Path("/content/data/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/outputs/fmow_random_split_sanity_resnet50"))
    parser.add_argument("--location-disjoint-resnet-zip", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    manifest = ensure_extracted_dataset(args.prepared_dataset_zip, args.extract_dir, args.manifest)
    data_root = args.data_root or manifest.parent
    rows = _random_split(read_csv(manifest), args.val_fraction, args.seed)
    split_manifest = args.output_dir / "random_split_manifest.csv"
    write_csv(split_manifest, rows)
    train_count = sum(1 for row in rows if row["split"] == "train")
    val_count = sum(1 for row in rows if row["split"] == "val")

    run_command(
        python_module_cmd(
            "rsfm_fairness_audit.cli",
            [
                "run-fmow-sentinel-classification",
                "--metadata-csv",
                str(split_manifest),
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
                "random_split_sanity",
                "--eval-scope",
                "random_split_val",
                "--image-size",
                "96",
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--run-bwer",
            ],
        )
    )
    comparison_rows = _summary_rows(args.output_dir, args.location_disjoint_resnet_zip)
    write_csv(args.output_dir / "random_split_vs_location_disjoint_summary.csv", comparison_rows)
    metadata = {
        "sanity_type": "random_split_sanity",
        "formal_result": False,
        "prepared_dataset_zip": str(args.prepared_dataset_zip or ""),
        "source_manifest": str(manifest),
        "random_split_manifest": str(split_manifest),
        "data_root": str(data_root),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_rows": train_count,
        "val_rows": val_count,
        "model": "resnet50_13band_from_scratch",
        "protocol_note": "Random sample-level split sanity contrast only; not the formal deployment protocol.",
    }
    write_json(args.output_dir / "random_split_sanity_metadata.json", metadata)
    report = [
        "# fMoW-Sentinel Random Split Sanity",
        "",
        "This is a sanity contrast only, not a formal deployment protocol.",
        "",
        f"- source manifest: `{manifest}`",
        f"- random split manifest: `{split_manifest}`",
        f"- seed: `{args.seed}`",
        f"- train / val rows: {train_count} / {val_count}",
        f"- epochs: {args.epochs}",
        "",
        "Outputs include `predictions.csv`, `audit_table.csv`, `metrics_summary.csv`, `bwer/`, and `random_split_vs_location_disjoint_summary.csv`.",
    ]
    (args.output_dir / "random_split_sanity_report.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

