from __future__ import annotations

import argparse
from pathlib import Path

from fmow_sanity_common import ensure_extracted_dataset, python_module_cmd, read_csv, run_command, write_csv, write_json


def _metrics_row(run_dir: Path, pooling: str, status: str = "completed", note: str = "") -> dict[str, str]:
    metrics_path = run_dir / "metrics_summary.csv"
    metrics = read_csv(metrics_path)[0] if metrics_path.exists() and read_csv(metrics_path) else {}
    return {
        "pooling": pooling,
        "status": status,
        "accuracy": metrics.get("accuracy", ""),
        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "top5_accuracy": metrics.get("top5_accuracy", ""),
        "note": note,
    }


def _run_pooling(args: argparse.Namespace, manifest: Path, data_root: Path, pooling: str) -> Path:
    out = args.output_dir / pooling
    cmd = python_module_cmd(
        "rsfm_fairness_audit.cli",
        [
            "run-fmow-sentinel-classification",
            "--metadata-csv",
            str(manifest),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(out),
            "--model",
            "dofa",
            "--model-config",
            str(args.model_config),
            "--probe",
            "linear",
            "--dofa-input-scale",
            "10000",
            "--dofa-embedding-pooling",
            pooling,
            "--embedding-cache-dir",
            str(args.output_dir / "embedding_cache"),
            "--train-split",
            "train",
            "--eval-split",
            "val",
            "--split-protocol",
            "location_disjoint",
            "--eval-scope",
            "val",
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
    if args.max_samples_per_split:
        cmd.extend(["--max-samples-per-split", str(args.max_samples_per_split)])
    run_command(cmd)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DOFA flatten vs mean-token pooling sanity ablation in Colab.")
    parser.add_argument("--prepared-dataset-zip", type=Path)
    parser.add_argument("--extract-dir", type=Path, default=Path("/content/data/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/outputs/fmow_dofa_pooling_ablation_scaled10000"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/dofa_fmow_sentinel.yaml"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = ensure_extracted_dataset(args.prepared_dataset_zip, args.extract_dir, args.manifest)
    data_root = args.data_root or manifest.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pooling in ("flatten", "mean_tokens"):
        run_dir = _run_pooling(args, manifest, data_root, pooling)
        rows.append(_metrics_row(run_dir, pooling))
    rows.append(
        {
            "pooling": "cls",
            "status": "unavailable",
            "accuracy": "",
            "balanced_accuracy": "",
            "macro_f1": "",
            "top5_accuracy": "",
            "note": "Current completed adapter path does not expose a named CLS representation separately from forward_features output.",
        }
    )
    write_csv(args.output_dir / "pooling_ablation_summary.csv", rows)
    run_command(
        python_module_cmd(
            "rsfm_fairness_audit.cli",
            [
                "compare-fmow-runs",
                "--run",
                f"flatten={args.output_dir / 'flatten'}",
                "--run",
                f"mean_tokens={args.output_dir / 'mean_tokens'}",
                "--output-dir",
                str(args.output_dir / "comparison"),
            ],
        )
    )
    write_json(
        args.output_dir / "pooling_ablation_metadata.json",
        {
            "sanity_type": "dofa_pooling_ablation",
            "formal_result": False,
            "source_manifest": str(manifest),
            "data_root": str(data_root),
            "input_scale": 10000,
            "normalization_guidance": "DOFA author confirmed inputs should be normalized to [0,1] or [-1,1].",
            "band_wavelength_note": "Band order itself is not critical; wavelength-band correspondence is critical.",
            "cls_status": "unavailable",
            "seed": args.seed,
        },
    )
    report = [
        "# DOFA Pooling Ablation Sanity",
        "",
        "This is a sanity ablation for the scaled DOFA frozen linear-probe protocol. It is not a new formal model result.",
        "",
        "- input_scale: `10000`",
        "- compared: `flatten` vs `mean_tokens`",
        "- CLS: unavailable in the current adapter/output contract; no CLS result is fabricated.",
        "- DOFA author guidance: inputs should be normalized to `[0,1]` or `[-1,1]`; band-wavelength correspondence is the critical requirement.",
        "",
        "Outputs include `pooling_ablation_summary.csv`, per-pooling run directories, BWER outputs, and `comparison/`.",
    ]
    (args.output_dir / "pooling_ablation_report.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

