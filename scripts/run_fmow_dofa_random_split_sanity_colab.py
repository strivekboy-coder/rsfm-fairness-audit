from __future__ import annotations

import argparse
from pathlib import Path

from fmow_sanity_common import ensure_extracted_dataset, first_row, python_module_cmd, read_csv, run_command, write_json


def _count_split(rows: list[dict[str, str]], split: str) -> int:
    return sum(1 for row in rows if row.get("split") == split)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fMoW-Sentinel DOFA scaled10000 random-split sanity using an existing random_split_manifest.csv."
    )
    parser.add_argument(
        "--random-split-manifest",
        type=Path,
        default=Path("/content/outputs/baseline_closure_sanity/random_split_resnet50_16epoch/random_split_manifest.csv"),
        help="Random split manifest produced by the final 16-epoch ResNet sanity run.",
    )
    parser.add_argument(
        "--prepared-dataset-zip",
        type=Path,
        help="Optional final v3_merged dataset zip. Used only to extract the data tree if --data-root is not already present.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("/content/data/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged"),
        help="Data root for extracted fMoW-Sentinel 30k v3_merged imagery.",
    )
    parser.add_argument("--data-root", type=Path, help="Override data root used to resolve relative image paths.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/outputs/baseline_closure_sanity/dofa_random_split_sanity"),
    )
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/dofa_fmow_sentinel.yaml"))
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.random_split_manifest.exists():
        raise FileNotFoundError(
            f"Random split manifest not found: {args.random_split_manifest}. "
            "Run the final 16-epoch ResNet random-split sanity first, or pass --random-split-manifest."
        )

    data_root = args.data_root or args.extract_dir
    if not data_root.exists() and args.prepared_dataset_zip:
        ensure_extracted_dataset(args.prepared_dataset_zip, args.extract_dir, None)
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root}. Pass --prepared-dataset-zip to extract the v3_merged dataset, "
            "or pass --data-root to the existing extracted data tree."
        )

    rows = read_csv(args.random_split_manifest)
    train_rows = _count_split(rows, "train")
    val_rows = _count_split(rows, "val")
    if train_rows == 0 or val_rows == 0:
        raise ValueError(
            f"Random split manifest must contain train and val rows; found train={train_rows}, val={val_rows}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.embedding_cache_dir or args.output_dir / "embedding_cache"
    run_command(
        python_module_cmd(
            "rsfm_fairness_audit.cli",
            [
                "run-fmow-sentinel-classification",
                "--metadata-csv",
                str(args.random_split_manifest),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(args.output_dir),
                "--model",
                "dofa",
                "--model-config",
                str(args.model_config),
                "--probe",
                "linear",
                "--dofa-input-scale",
                "10000",
                "--dofa-embedding-pooling",
                "flatten",
                "--embedding-cache-dir",
                str(cache_dir),
                "--train-split",
                "train",
                "--eval-split",
                "val",
                "--split-protocol",
                "random_split_sanity",
                "--eval-scope",
                "random_split_val",
                "--image-size",
                str(args.image_size),
                "--batch-size",
                str(args.batch_size),
                "--probe-epochs",
                str(args.probe_epochs),
                "--probe-learning-rate",
                str(args.probe_learning_rate),
                "--seed",
                str(args.seed),
                "--allow-torch-hub-download",
                "--run-bwer",
            ],
        )
    )

    metrics = first_row(args.output_dir / "metrics_summary.csv")
    metadata = {
        "sanity_type": "dofa_random_split_sanity",
        "formal_result": False,
        "random_split_manifest": str(args.random_split_manifest),
        "reuses_resnet_random_split_manifest": True,
        "data_root": str(data_root),
        "model": "dofa_vit_base",
        "adaptation_protocol": "frozen_encoder_linear_probe",
        "split_protocol": "random_split_sanity",
        "eval_scope": "random_split_val",
        "input_scale": 10000,
        "embedding_pooling": "flatten",
        "train_rows": train_rows,
        "val_rows": val_rows,
        "seed": args.seed,
        "embedding_cache_dir": str(cache_dir),
        "cache_key_note": "DOFA cache key is based on sample identity and protocol-changing settings, not the random split label itself.",
        "protocol_note": "Random sample-level split sanity contrast only; not the formal deployment protocol.",
    }
    write_json(args.output_dir / "dofa_random_split_sanity_metadata.json", metadata)

    report = [
        "# fMoW-Sentinel DOFA Random Split Sanity",
        "",
        "This is an optional baseline-closure sanity contrast only. It is not the formal deployment protocol.",
        "",
        f"- random split manifest: `{args.random_split_manifest}`",
        f"- reused ResNet 16-epoch random split manifest: `true`",
        f"- data root: `{data_root}`",
        f"- train / val rows: {train_rows} / {val_rows}",
        "- model: `dofa_vit_base`",
        "- adaptation_protocol: `frozen_encoder_linear_probe`",
        "- split_protocol: `random_split_sanity`",
        "- eval_scope: `random_split_val`",
        "- input_scale: `10000`",
        "- pooling: `flatten`",
        f"- accuracy: {metrics.get('accuracy', '')}",
        f"- balanced_accuracy: {metrics.get('balanced_accuracy', '')}",
        f"- macro_f1: {metrics.get('macro_f1', '')}",
        "",
        "Outputs include `predictions.csv`, `audit_table.csv`, `metrics_summary.csv`, `run_metadata.json`, `bwer/`, and this report.",
        "Do not compare this as a formal deployment result; use it only to contrast random sample-level split behavior with the formal location-disjoint DOFA result.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
