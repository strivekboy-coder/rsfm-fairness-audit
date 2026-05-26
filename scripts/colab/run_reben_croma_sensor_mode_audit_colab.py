from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir  # noqa: E402
from rsfm_fairness_audit.io import write_csv  # noqa: E402
from rsfm_fairness_audit.adapters.croma import CROMAAdapter  # noqa: E402
from rsfm_fairness_audit.adapters.reben import ConfigILMRebenDatasetAdapter, check_reben_configilm_dependency_chain  # noqa: E402
from rsfm_fairness_audit.reben_sensor_audit import (  # noqa: E402
    REBEN_BIFOLD_RESNET101_IDS,
    REBEN_CROMA_EMBEDDING_KEYS,
    SOURCE_VERIFICATION_URLS,
    BifoldResNet101Runner,
    RebenRunLabels,
    collect_reben_sensor_audit_outputs,
    run_bifold_resnet101_reben_inference,
    run_croma_reben_frozen_probe,
    run_reben_dataset_preflight,
    validate_reben_sensor_audit_contract,
    validate_bifold_resnet101_refs,
    default_reben_class_names,
    write_reben_source_verification_report,
    write_reben_blocked_report,
)


REQUIRED_BIFOLD_IDS = {
    "s1": "BIFOLD-BigEarthNetv2-0/resnet101-s1-v0.2.0",
    "s2": "BIFOLD-BigEarthNetv2-0/resnet101-s2-v0.2.0",
    "all": "BIFOLD-BigEarthNetv2-0/resnet101-all-v0.2.0",
}


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--label-expanded-predictions must use name=path")
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Colab-oriented reBEN/CROMA sensor-mode audit handoff runner. "
            "Run this after scripts/colab/prepare_reben_croma_sensor_audit_colab.py "
            "has verified the reBEN LMDB/parquet files, CROMA repo/checkpoint, and "
            "official BIFOLD model refs. This script performs protocol/data readiness "
            "checks and can run the post-hoc multi-label BWER stage on completed "
            "label-expanded predictions."
        )
    )
    parser.add_argument("--data-root", type=Path, help="Optional extracted BigEarthNet v2/reBEN root.")
    parser.add_argument("--lmdb-root", type=Path, help="ConfigILM/reBEN LMDB root or images_lmdb path.")
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--metadata-snow-cloud-parquet", type=Path)
    parser.add_argument("--croma-checkpoint", type=Path, required=True)
    parser.add_argument("--croma-repo", type=Path, help="Local clone of https://github.com/antofuller/CROMA containing use_croma.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/outputs/reben_croma_sensor_mode_audit"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, help="Smoke-test cap only; omit for formal runs.")
    parser.add_argument("--bifold-resnet101-s1", default=REQUIRED_BIFOLD_IDS["s1"], help="Official HF id or local official v0.2.0 model folder for S1.")
    parser.add_argument("--bifold-resnet101-s2", default=REQUIRED_BIFOLD_IDS["s2"], help="Official HF id or local official v0.2.0 model folder for S2.")
    parser.add_argument("--bifold-resnet101-all", default=REQUIRED_BIFOLD_IDS["all"], help="Official HF id or local official v0.2.0 model folder for S1+S2.")
    parser.add_argument("--label-expanded-predictions", action="append", type=_parse_run, default=[], help="Optional completed run in name=predictions.csv form for post-hoc BWER.")
    parser.add_argument("--run-croma", action="store_true", help="Run CROMA frozen encoder + linear multi-label probe rows for S1/S2/S1+S2.")
    parser.add_argument("--run-bifold", action="store_true", help="Run official BIFOLD ResNet101 v0.2.0 inference rows for S1/S2/S1+S2.")
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--class-names-json", type=Path, help="Optional JSON list of the 19 reBEN class names.")
    parser.add_argument("--package", action="store_true", help="Zip the output directory after readiness/post-hoc stages.")
    return parser


def _exists(path: Path | None) -> bool:
    return bool(path and path.exists())


def _write_reports(args: argparse.Namespace, out: Path) -> None:
    reports = ensure_dir(out / "reports")
    protocol = [
        "# reBEN / CROMA Sensor-Mode Audit Protocol",
        "",
        "Workflow: first run `scripts/colab/prepare_reben_croma_sensor_audit_colab.py`, then a smoke run with `--max-samples`, then the full run without `--max-samples`.",
        "",
        "This runner is for BigEarthNet v2.0 / reBEN, not BigEarthNet v1.",
        "",
        "Main rows:",
        "- CROMA frozen encoder + linear multi-label probe: S1, S2, S1+S2.",
        "- Official BIFOLD ResNet101 v0.2.0 supervised references: S1, S2, S1+S2.",
        "",
        "Sensor mode is a cross-run experimental condition, not a per-sample metadata slice.",
        "Within-run geography/class/cloud-snow slices are audited post hoc from label-expanded predictions.",
    ]
    (reports / "model_protocol.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")
    risk = [
        "# Protocol Risk",
        "",
        "CROMA and BIFOLD ResNet101 use official but not identical input conventions.",
        "CROMA expects 2-channel Sentinel-1, 12-channel Sentinel-2, and 120x120 inputs.",
        "BIFOLD ResNet101 v0.2.0 uses S1 VV/VH, S2 10-band 10m/20m order, or their 12-band S1+S2 concatenation.",
        "Cross-family comparisons are therefore protocol-aware, not pure architecture-only controls.",
    ]
    (reports / "protocol_risk.md").write_text("\n".join(risk) + "\n", encoding="utf-8")
    primitives = [
        "# Metric Primitives",
        "",
        "Primary multi-label risk primitive: label-wise binary cross-entropy when probabilities/logits are available.",
        "Secondary diagnostic primitive: thresholded label-wise binary error.",
        "Thresholds must be selected from validation predictions only and saved before final evaluation.",
        "Single-label accuracy-style risk is invalid for the main reBEN multi-label audit.",
    ]
    (reports / "metric_primitives.md").write_text("\n".join(primitives) + "\n", encoding="utf-8")
    support = [
        "# Support Preflight",
        "",
        "Main support threshold: min_samples_per_slice=20.",
        "Sensitivity thresholds: 10 and 30.",
        "Primary slices: class, country, country | class, and country x class diagnostic.",
        "Cloud/snow sensitivity is only formal when metadata and support are present.",
    ]
    (reports / "support_preflight.md").write_text("\n".join(support) + "\n", encoding="utf-8")


def _write_preflight(args: argparse.Namespace, out: Path) -> None:
    write_reben_source_verification_report(out)
    try:
        run_reben_dataset_preflight(args.metadata_parquet, out, max_rows=args.max_samples)
    except Exception as exc:
        (out / "dataset_preflight_error.txt").write_text(str(exc), encoding="utf-8")
    existing: dict[str, object] = {}
    preflight_path = out / "dataset_preflight.json"
    if preflight_path.exists():
        try:
            existing = json.loads(preflight_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    data = {
        **existing,
        "dataset": "bigearthnet_v2_reben",
        "metadata_parquet": str(args.metadata_parquet),
        "metadata_parquet_exists": _exists(args.metadata_parquet),
        "metadata_snow_cloud_parquet": str(args.metadata_snow_cloud_parquet or ""),
        "metadata_snow_cloud_parquet_exists": _exists(args.metadata_snow_cloud_parquet),
        "data_root": str(args.data_root or ""),
        "data_root_exists": _exists(args.data_root),
        "lmdb_root": str(args.lmdb_root or ""),
        "lmdb_root_exists": _exists(args.lmdb_root),
        "croma_checkpoint": str(args.croma_checkpoint),
        "croma_checkpoint_exists": _exists(args.croma_checkpoint),
        "croma_repo": str(args.croma_repo or ""),
        "croma_use_croma_exists": bool(args.croma_repo and (args.croma_repo / "use_croma.py").exists()),
        "batch_size": args.batch_size,
        "max_samples": args.max_samples or "",
        "bifold_resnet101_ids": {
            "s1": args.bifold_resnet101_s1,
            "s2": args.bifold_resnet101_s2,
            "all": args.bifold_resnet101_all,
        },
        "source_verification_urls": SOURCE_VERIFICATION_URLS,
        "reben_configilm_dependency_check": check_reben_configilm_dependency_chain(),
        "status": "ready_for_colab_model_stage" if _exists(args.metadata_parquet) and _exists(args.croma_checkpoint) else "missing_required_runtime_inputs",
    }
    (out / "dataset_preflight.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_csv(
        out / "bifold_resnet101_readiness.csv",
        validate_bifold_resnet101_refs(
            {
                "S1": args.bifold_resnet101_s1,
                "S2": args.bifold_resnet101_s2,
                "S1+S2": args.bifold_resnet101_all,
            }
        ),
    )


def _run_posthoc_bwer(prediction_runs: list[tuple[str, Path]], out: Path) -> None:
    if not prediction_runs:
        return
    for name, predictions in prediction_runs:
        if not predictions.exists():
            raise FileNotFoundError(f"Prediction table for {name} does not exist: {predictions}")
        run_out = ensure_dir(out / "bwer" / name)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rsfm_fairness_audit.cli",
                "run-reben-multilabel-bwer",
                "--predictions",
                str(predictions),
                "--output-dir",
                str(run_out),
                "--model-name",
                name,
                "--selective-risk",
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )


def _class_names(path: Path | None) -> list[str]:
    if path is None:
        return default_reben_class_names()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 19:
        raise ValueError("--class-names-json must be a JSON list with 19 names.")
    return [str(item) for item in data]


def _croma_mode_config(mode: str) -> dict[str, object]:
    if mode == "S1":
        return {"input_modality": "SAR", "embedding_key": REBEN_CROMA_EMBEDDING_KEYS[mode], "expected_s1_bands": 2, "expected_s2_bands": 12}
    if mode == "S2":
        return {"input_modality": "optical", "embedding_key": REBEN_CROMA_EMBEDDING_KEYS[mode], "expected_s1_bands": 2, "expected_s2_bands": 12}
    if mode == "S1+S2":
        return {"input_modality": "both", "embedding_key": REBEN_CROMA_EMBEDDING_KEYS[mode], "expected_s1_bands": 2, "expected_s2_bands": 12}
    raise ValueError(f"Unsupported CROMA sensor mode: {mode}")


def _run_croma_rows(args: argparse.Namespace, out: Path) -> None:
    if not args.run_croma:
        return
    if not args.lmdb_root:
        write_reben_blocked_report(out, "--run-croma missing --lmdb-root")
        raise ValueError("--run-croma requires --lmdb-root pointing to ConfigILM/reBEN images_lmdb.")
    if not args.metadata_snow_cloud_parquet:
        write_reben_blocked_report(out, "--run-croma missing --metadata-snow-cloud-parquet")
        raise ValueError("--run-croma requires --metadata-snow-cloud-parquet.")
    classes = _class_names(args.class_names_json)
    for mode in ["S1", "S2", "S1+S2"]:
        run_name = f"croma_{mode.lower().replace('+', '_plus_')}"
        print(f"[reben:croma] starting {run_name}")
        train_dataset = ConfigILMRebenDatasetAdapter(
            args.lmdb_root,
            args.metadata_parquet,
            args.metadata_snow_cloud_parquet,
            split="train",
            sensor_mode=mode,
            max_samples=args.max_samples,
        )
        eval_dataset = ConfigILMRebenDatasetAdapter(
            args.lmdb_root,
            args.metadata_parquet,
            args.metadata_snow_cloud_parquet,
            split="val",
            sensor_mode=mode,
            max_samples=args.max_samples,
        )
        adapter = CROMAAdapter(
            checkpoint_path=args.croma_checkpoint,
            repo_path=args.croma_repo,
            image_size=120,
            batch_size=args.batch_size,
            **_croma_mode_config(mode),
        )
        run_croma_reben_frozen_probe(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            croma_adapter=adapter,
            output_dir=out,
            run_name=run_name,
            run_labels=RebenRunLabels(
                model_family="croma",
                model_variant="croma_base",
                sensor_mode=mode,
                adaptation_protocol="frozen_encoder_linear_probe",
                eval_scope="validation",
                band_profile="croma_official_s1_2_s2_12",
            ),
            class_names=classes,
            batch_size=args.batch_size,
            probe_epochs=args.probe_epochs,
            probe_learning_rate=args.probe_learning_rate,
            probe_weight_decay=args.probe_weight_decay,
            seed=42,
            device=args.device,
        )
        print(f"[reben:croma] finished {run_name}")


def _run_bifold_rows(args: argparse.Namespace, out: Path) -> None:
    if not args.run_bifold:
        return
    if not args.lmdb_root:
        write_reben_blocked_report(out, "--run-bifold missing --lmdb-root")
        raise ValueError("--run-bifold requires --lmdb-root pointing to ConfigILM/reBEN images_lmdb.")
    if not args.metadata_snow_cloud_parquet:
        write_reben_blocked_report(out, "--run-bifold missing --metadata-snow-cloud-parquet")
        raise ValueError("--run-bifold requires --metadata-snow-cloud-parquet.")
    configured = {
        "S1": args.bifold_resnet101_s1,
        "S2": args.bifold_resnet101_s2,
        "S1+S2": args.bifold_resnet101_all,
    }
    readiness = validate_bifold_resnet101_refs(configured)
    bad = [row for row in readiness if row["status"] != "ok"]
    if bad:
        write_reben_blocked_report(out, "non-official BIFOLD ResNet101 refs", {"bad_refs": bad})
        raise ValueError(f"Refusing non-official BIFOLD ResNet101 ids: {bad}")
    classes = _class_names(args.class_names_json)
    for mode in ["S1", "S2", "S1+S2"]:
        run_name = f"bifold_resnet101_{mode.lower().replace('+', '_plus_')}"
        print(f"[reben:bifold] starting {run_name}")
        eval_dataset = ConfigILMRebenDatasetAdapter(
            args.lmdb_root,
            args.metadata_parquet,
            args.metadata_snow_cloud_parquet,
            split="val",
            sensor_mode=mode,
            max_samples=args.max_samples,
            channel_profile="bifold_resnet101",
        )
        run_bifold_resnet101_reben_inference(
            eval_dataset=eval_dataset,
            model_runner=BifoldResNet101Runner(configured[mode], device=args.device),
            output_dir=out,
            run_name=run_name,
            run_labels=RebenRunLabels(
                model_family="bifold_resnet101",
                model_variant="resnet101_v0.2.0",
                sensor_mode=mode,
                adaptation_protocol="official_supervised_reference",
                eval_scope="validation",
                band_profile="bifold_reben_v0.2.0_official",
            ),
            class_names=classes,
            batch_size=args.batch_size,
        )
        print(f"[reben:bifold] finished {run_name}")


def _package(out: Path) -> Path:
    archive = out.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=out)
    return archive


def main() -> None:
    args = build_parser().parse_args()
    out = ensure_dir(args.output_dir)
    _write_preflight(args, out)
    _write_reports(args, out)
    _run_croma_rows(args, out)
    _run_bifold_rows(args, out)
    _run_posthoc_bwer(args.label_expanded_predictions, out)
    collect_reben_sensor_audit_outputs(out)
    validate_reben_sensor_audit_contract(out)
    if args.package:
        archive = _package(out)
        print(f"[reben] packaged: {archive}")
    print(f"[reben] output_dir={out}")
    print("[reben] readiness report: dataset_preflight.json")


if __name__ == "__main__":
    main()
