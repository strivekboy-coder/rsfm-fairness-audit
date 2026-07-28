from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rsfm_fairness_audit.formal_outputs import file_sha256  # noqa: E402
from rsfm_fairness_audit import __version__  # noqa: E402


MODES = ("s1", "s2", "s1_plus_s2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-command, non-formal real-GPU gate for the complete Sen1Floods11 "
            "panel: TerraMind S1/S2/S1+S2, ResNet34-U-Net S1/S2/S1+S2, and "
            "the official Prithvi S2 task checkpoint."
        )
    )
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--s2-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--terramind-checkpoint", type=Path, required=True)
    parser.add_argument("--prithvi-prepared-data-root", type=Path, required=True)
    parser.add_argument("--prithvi-prepared-metadata-csv", type=Path)
    parser.add_argument("--prithvi-model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic-max-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def _run(command: list[str]) -> None:
    print("[sen1:gpu-smoke]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _resolve_artifact(root: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    direct = root / candidate
    if direct.is_file():
        return direct
    by_name = root / "samples" / candidate.name
    if by_name.is_file():
        return by_name
    raise RuntimeError(f"Missing probability artifact under {root}: {value}")


def validate_probability_export(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    index_parts = sorted((root / "index_parts").glob("*.jsonl"))
    if not index_parts:
        raise RuntimeError(f"Probability export has no index parts: {root}")
    rows: list[dict[str, Any]] = []
    for index_part in index_parts:
        rows.extend(
            json.loads(line)
            for line in index_part.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"Probability export has empty or duplicate sample IDs: {root}")
    shapes: list[list[int]] = []
    valid_pixel_counts: list[int] = []
    observed_target_values: set[int] = set()
    aggregate_valid_pixel_count = 0
    all_ignore_row_count = 0
    for row in rows:
        artifact_path = _resolve_artifact(root, row["probability_path"])
        with np.load(artifact_path) as artifact:
            probabilities = np.asarray(artifact["probabilities"])
            target = np.asarray(artifact["target"]).squeeze()
        if probabilities.ndim != 3 or probabilities.shape[0] != 2:
            raise RuntimeError(
                f"Expected [2,H,W] probabilities, observed={probabilities.shape}: {artifact_path}"
            )
        if target.shape != probabilities.shape[1:]:
            raise RuntimeError(
                f"Probability/target shape mismatch at {artifact_path}: "
                f"{probabilities.shape} vs {target.shape}"
            )
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(f"NaN/Inf probability values: {artifact_path}")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise RuntimeError(f"Probability values outside [0,1]: {artifact_path}")
        if not np.allclose(probabilities.sum(axis=0), 1.0, atol=2e-4, rtol=2e-4):
            raise RuntimeError(f"Class probabilities do not sum to one: {artifact_path}")
        raw_target_values = np.unique(target).tolist()
        try:
            numeric_target_values = [float(value) for value in raw_target_values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Target contains non-numeric values: {artifact_path}"
            ) from exc
        invalid_target_values = sorted(
            value
            for value in numeric_target_values
            if not np.isfinite(value) or value not in {-1.0, 0.0, 1.0}
        )
        if invalid_target_values:
            raise RuntimeError(
                "Target contains values outside the formal {-1,0,1} contract: "
                f"{artifact_path}, invalid={invalid_target_values}"
            )
        target_values = {int(value) for value in numeric_target_values}
        observed_target_values.update(target_values)
        valid_pixel_count = int(np.isin(target, [0, 1]).sum())
        valid_pixel_counts.append(valid_pixel_count)
        aggregate_valid_pixel_count += valid_pixel_count
        if valid_pixel_count == 0:
            all_ignore_row_count += 1
        shapes.append(list(target.shape))
    if aggregate_valid_pixel_count <= 0:
        raise RuntimeError(
            f"Probability export contains no valid hand-labeled pixels across any row: {root}"
        )
    return {
        "row_count": len(rows),
        "sample_ids": sample_ids,
        "target_shapes": shapes,
        "valid_pixel_counts": valid_pixel_counts,
        "all_ignore_row_count": all_ignore_row_count,
        "valid_row_count": len(rows) - all_ignore_row_count,
        "aggregate_valid_pixel_count": aggregate_valid_pixel_count,
        "observed_target_values": sorted(observed_target_values),
    }


def _validate_pair(validation: Path, test: Path) -> dict[str, Any]:
    validation_report = validate_probability_export(validation)
    test_report = validate_probability_export(test)
    overlap = sorted(
        set(validation_report["sample_ids"]) & set(test_report["sample_ids"])
    )
    if overlap:
        raise RuntimeError(
            "Smoke validation/test sample IDs overlap: " + ", ".join(overlap[:10])
        )
    return {
        "validation": validation_report,
        "test": test_report,
        "validation_test_overlap": 0,
    }


def _require_cuda() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the GPU smoke.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required; CPU fallback is not accepted.")
    device = torch.device("cuda:0")
    probe = torch.zeros(1, device=device)
    report = {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "probe_tensor_device": str(probe.device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    print("[sen1:gpu-smoke] runtime:", json.dumps(report), flush=True)
    return report


def _source_paths(args: argparse.Namespace) -> list[Path]:
    paths = [
        args.s1_root,
        args.s2_root,
        args.label_root,
        args.train_split,
        args.val_split,
        args.test_split,
        args.terramind_checkpoint,
        args.prithvi_prepared_data_root,
        args.prithvi_model_config,
    ]
    if args.prithvi_prepared_metadata_csv is not None:
        paths.append(args.prithvi_prepared_metadata_csv)
    return paths


def _completion_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = _json(path)
    if payload.get("status") != "pass" or payload.get("formal_evidence") is not False:
        return False
    for artifact in payload.get("artifacts", []):
        candidate = Path(str(artifact["path"]))
        if not candidate.is_file() or file_sha256(candidate) != artifact["sha256"]:
            return False
    return True


def main() -> None:
    args = build_parser().parse_args()
    if min(args.diagnostic_max_samples, args.batch_size, args.num_workers + 1) <= 0:
        raise ValueError("diagnostic-max-samples and batch-size must be positive; num-workers must be non-negative.")
    for path in _source_paths(args):
        if not path.exists():
            raise FileNotFoundError(f"Required smoke asset is missing: {path}")
    if args.persistent_output_dir is not None:
        try:
            args.output_dir.resolve().relative_to(args.persistent_output_dir.resolve())
        except ValueError:
            pass
        else:
            raise RuntimeError("Live smoke output must be local /content, not inside the Drive mirror.")
    completion = args.output_dir / "completion_contract.json"
    if _completion_is_valid(completion):
        print(f"[sen1:gpu-smoke] reusing verified completion contract: {completion}")
        print("SEN1_GPU_SMOKE=PASS")
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            f"Non-empty partial smoke directory found: {args.output_dir}. "
            "Preserve it for diagnosis and use a new versioned output directory."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _require_cuda()
    python = sys.executable

    terramind_root = args.output_dir / "terramind"
    terramind_command = [
        python,
        str(PROJECT_ROOT / "scripts/colab/run_terramind_sen1floods11_final_colab.py"),
        "--s1-root", str(args.s1_root),
        "--s2-root", str(args.s2_root),
        "--label-root", str(args.label_root),
        "--train-split", str(args.train_split),
        "--val-split", str(args.val_split),
        "--test-split", str(args.test_split),
        "--checkpoint", str(args.terramind_checkpoint),
        "--output-dir", str(terramind_root),
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--smoke-only",
    ]
    if args.persistent_output_dir is not None:
        terramind_command += [
            "--persistent-output-dir",
            str(args.persistent_output_dir / "terramind"),
        ]
    _run(terramind_command)

    supervised_root = args.output_dir / "supervised"
    supervised_command = [
        python,
        str(PROJECT_ROOT / "scripts/colab/run_sen1_supervised_panel_colab.py"),
        "--s1-root", str(args.s1_root),
        "--s2-root", str(args.s2_root),
        "--label-root", str(args.label_root),
        "--train-split", str(args.train_split),
        "--val-split", str(args.val_split),
        "--test-split", str(args.test_split),
        "--output-dir", str(supervised_root),
        "--seeds", str(args.seed),
        "--max-epochs", "1",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", "cuda",
        "--diagnostic-max-samples", str(args.diagnostic_max_samples),
    ]
    if args.persistent_output_dir is not None:
        supervised_command += [
            "--persistent-output-dir",
            str(args.persistent_output_dir / "supervised"),
        ]
    _run(supervised_command)

    prithvi_root = args.output_dir / "prithvi"
    prithvi_command = [
        python,
        str(PROJECT_ROOT / "scripts/colab/run_prithvi_sen1_geobwer_migration_colab.py"),
        "--prepared-data-root", str(args.prithvi_prepared_data_root),
        "--model-config", str(args.prithvi_model_config),
        "--val-split", str(args.val_split),
        "--test-split", str(args.test_split),
        "--output-dir", str(prithvi_root),
        "--batch-size", "1",
        "--device", "cuda",
        "--diagnostic-max-samples", str(min(args.diagnostic_max_samples, 4)),
    ]
    if args.prithvi_prepared_metadata_csv is not None:
        prithvi_command += [
            "--prepared-metadata-csv",
            str(args.prithvi_prepared_metadata_csv),
        ]
    if args.persistent_output_dir is not None:
        prithvi_command += [
            "--persistent-output-dir",
            str(args.persistent_output_dir / "prithvi"),
        ]
    _run(prithvi_command)

    checks: dict[str, Any] = {"terramind": {}, "supervised": {}, "prithvi": {}}
    terramind_panel = _json(terramind_root / "diagnostic_panel_manifest.json")
    if terramind_panel.get("formal_evidence") is not False:
        raise RuntimeError("TerraMind smoke was not marked non-formal.")
    for mode in MODES:
        manifest_path = terramind_root / mode / f"seed_{args.seed}" / "diagnostic_manifest.json"
        manifest = _json(manifest_path)
        if (
            manifest.get("schema") != "geobwer.sen1floods11.terramind_diagnostic.v3"
            or manifest.get("formal_evidence") is not False
            or int(manifest.get("validation_test_sample_overlap", -1)) != 0
        ):
            raise RuntimeError(f"Invalid TerraMind end-to-end smoke manifest: {manifest_path}")
        checks["terramind"][mode] = {
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "validation_rows": manifest["validation_export"]["row_count"],
            "test_rows": manifest["test_export"]["row_count"],
        }

    supervised_panel = _json(supervised_root / "campaign_manifest.json")
    if supervised_panel.get("formal_evidence") is not False:
        raise RuntimeError("Supervised smoke was not marked non-formal.")
    for mode in MODES:
        run_root = supervised_root / mode / f"seed_{args.seed}"
        checks["supervised"][mode] = _validate_pair(
            run_root / "probabilities/validation",
            run_root / "probabilities/test",
        )

    prithvi_manifest_path = prithvi_root / "campaign_manifest.json"
    prithvi_manifest = _json(prithvi_manifest_path)
    if prithvi_manifest.get("formal_evidence") is not False:
        raise RuntimeError("Prithvi smoke was not marked non-formal.")
    checks["prithvi"]["s2"] = _validate_pair(
        prithvi_root / "probabilities/validation",
        prithvi_root / "probabilities/test",
    )

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    artifact_paths = [
        terramind_root / "diagnostic_panel_manifest.json",
        supervised_root / "campaign_manifest.json",
        prithvi_manifest_path,
    ]
    payload = {
        "schema": "geobwer.sen1floods11.complete_gpu_smoke.v1",
        "status": "pass",
        "formal_evidence": False,
        "code_commit": git_head,
        "package_version": __version__,
        "runtime": runtime,
        "seed": args.seed,
        "checks": checks,
        "commands": {
            "terramind": terramind_command,
            "supervised": supervised_command,
            "prithvi": prithvi_command,
        },
        "artifacts": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in artifact_paths
        ],
    }
    completion.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.persistent_output_dir is not None:
        args.persistent_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(completion, args.persistent_output_dir / completion.name)
    print(f"[sen1:gpu-smoke] completion contract: {completion}")
    print("SEN1_GPU_SMOKE=PASS")


if __name__ == "__main__":
    main()
