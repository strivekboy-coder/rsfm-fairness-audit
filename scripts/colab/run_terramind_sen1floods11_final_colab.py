from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.formal_outputs import file_sha256  # noqa: E402
from rsfm_fairness_audit import __version__  # noqa: E402
from rsfm_fairness_audit.adapters.terramind import (  # noqa: E402
    INPUT_PROFILES,
    S1_MEAN,
    S1_STD,
    TERRAMIND_OFFICIAL_REVISION,
    TERRAMIND_OFFICIAL_SHA256,
    validate_terratorch_runtime,
    validate_terramind_checkpoint,
)
from rsfm_fairness_audit.bwer_protocol import BWERProtocol  # noqa: E402
from rsfm_fairness_audit.geobwer_extensions import run_segmentation_uncertainty_suite  # noqa: E402
from rsfm_fairness_audit.geobwer_panel import run_geobwer_model_panel  # noqa: E402
from rsfm_fairness_audit.persistent_cache import (  # noqa: E402
    copy_changed_tree,
    hydrate_output,
    persist_output,
)
from rsfm_fairness_audit.probe_selection import group_disjoint_inner_split  # noqa: E402
from rsfm_fairness_audit.sen1floods11_formal import (  # noqa: E402
    calibrate_common_sen1_spatial_blocks,
    combine_sen1_evaluation_exports,
    finalize_sen1floods11_segmentation,
    load_sen1_probability_units,
    write_sen1_descriptive_probability_report,
    write_sen1_evaluation_split_report,
)
from rsfm_fairness_audit.sen1_input_quality import (  # noqa: E402
    FORMAL_COMPLETE_MODALITY_MISSING_EXPECTATION,
    SEN1_IMPUTATION_POLICY,
    input_quality_summary,
    normalize_named_modalities,
)
from rsfm_fairness_audit.terramind_sen1_config import (  # noqa: E402
    prepare_terramind_sen1_splits,
    write_terramind_sen1floods11_config,
)


MODES = ("S1", "S2", "S1+S2")
SMOKE_BATCH_LIMIT = 2
CALIBRATION_PANEL_SCOPE = "all_19_models_unet9_terramind9_prithvi1"


def _expected_calibration_model_names() -> tuple[str, ...]:
    names = [
        f"{prefix}_{mode.lower().replace('+', '_plus_')}_seed_{seed}"
        for prefix in ("resnet34_unet", "terramind_v1_base")
        for mode in MODES
        for seed in (42, 73, 101)
    ]
    names.append("prithvi_eo_v2_300_tl_s2")
    return tuple(sorted(names))


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Expected unique comma-separated integer seeds.")
    return result


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=/absolute/path.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("Expected non-empty NAME=/absolute/path.")
    return name, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Final, resumable TerraMind S1/S2/S1+S2 Sen1Floods11 campaign with formal probability exports."
    )
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--s2-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument(
        "--bolivia-split",
        "--heldout-event-split",
        dest="bolivia_split",
        type=Path,
        required=True,
        help="Independent official 15-chip Bolivia holdout; never used for fitting or calibration.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Pinned official TerraMind_v1_base.pt; SHA-256 is verified before any formal fit.",
    )
    parser.add_argument(
        "--persistent-output-dir",
        type=Path,
        help="Drive mirror for completed stages. Keep --output-dir on /content; never train directly on Drive.",
    )
    parser.add_argument(
        "--resume-source-root",
        type=Path,
        help=(
            "Read-only frozen Drive root from a prior calibration-failed run. "
            "Its verified fit/checkpoint and validation prediction contracts "
            "are copied to local scratch; all new evidence is persisted only "
            "to --persistent-output-dir."
        ),
    )
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/geobwer/sen1floods11.yaml")
    parser.add_argument("--mode", action="append", choices=MODES, help="Repeat to restrict modes; default is all three.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--gpu-log-every-n-steps", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--seeds",
        type=_csv_ints,
        default=(42, 73, 101),
        help="Formal training seeds. All are independently trained and retain complete probabilities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Legacy single-seed diagnostic override; not accepted for a formal campaign.",
    )
    parser.add_argument(
        "--additional-validation-export",
        action="append",
        type=_named_path,
        default=[],
        metavar="NAME=PATH",
        help=(
            "Diagnostic-only validation probability export from another Sen1 "
            "model. Formal runs instead require the audited U-Net panel and "
            "the explicit Prithvi manifest/export pair."
        ),
    )
    parser.add_argument("--calibration-simulations", type=int, default=200)
    parser.add_argument("--calibration-bootstrap", type=int, default=500)
    parser.add_argument("--audit-bootstrap", type=int, default=2000)
    parser.add_argument("--crc-alpha", type=float, default=0.10)
    parser.add_argument("--minimum-moderate-tail-power", type=float, default=0.80)
    parser.add_argument("--checkpoint-mirror-every-n-epochs", type=int, default=10)
    parser.add_argument(
        "--supervised-campaign-manifest",
        type=Path,
        help=(
            "Read-only v0.4.28 U-Net campaign manifest. Required for a formal "
            "panel so all nine validation exports enter common calibration."
        ),
    )
    parser.add_argument(
        "--supervised-audit-json",
        type=Path,
        help=(
            "PASS evidence from audit_sen1_unet_v0428_artifacts_colab.py. "
            "Required with --supervised-campaign-manifest."
        ),
    )
    parser.add_argument(
        "--prithvi-validation-export",
        type=Path,
        help=(
            "Read-only formal Prithvi validation probability export. Required "
            "before a formal all-model spatial calibration."
        ),
    )
    parser.add_argument(
        "--prithvi-campaign-manifest",
        type=Path,
        help=(
            "Read-only formal Prithvi probability-migration manifest. Required "
            "with --prithvi-validation-export so the 89-unit export is bound "
            "to the official 252/89/90+15 contract."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate configs without training/inference.")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help=(
            "Run a bounded end-to-end GPU diagnostic per mode: one-epoch fit with one "
            "train/validation batch, checkpoint reload, and one validation/test prediction "
            "batch through the formal probability writer. Output is non-formal."
        ),
    )
    return parser


def _terratorch_command() -> list[str]:
    validate_terratorch_runtime()
    executable = shutil.which("terratorch")
    if executable:
        return [executable]
    raise RuntimeError("The official `terratorch` CLI is unavailable. Install terratorch>=1.2.5,<1.3 in Colab.")


def _terratorch_predict_command() -> list[str]:
    """Use the repository wrapper that filters only TerraTorch's default writer."""

    validate_terratorch_runtime()
    return [
        sys.executable,
        "-m",
        "rsfm_fairness_audit.terratorch_predict_cli",
    ]


def _stage_log(
    *,
    stage: str,
    phase: str,
    mode: str | None = None,
    seed: int | None = None,
    elapsed_seconds: float | None = None,
    detail: str | None = None,
) -> None:
    values = [
        f"stage={stage}",
        f"phase={phase}",
        f"mode={mode or 'panel'}",
        f"seed={seed if seed is not None else 'panel'}",
    ]
    if elapsed_seconds is not None:
        values.append(f"elapsed_seconds={elapsed_seconds:.3f}")
    if detail:
        values.append(f"detail={detail}")
    print("[terramind:stage] " + " ".join(values), flush=True)


def _run(
    command: list[str],
    *,
    stage: str,
    mode: str,
    seed: int,
) -> None:
    _stage_log(stage=stage, phase="start", mode=mode, seed=seed)
    print("[terramind:campaign]", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    _stage_log(
        stage=stage,
        phase="end",
        mode=mode,
        seed=seed,
        elapsed_seconds=time.perf_counter() - started,
    )


def _persist_with_log(
    source: Path,
    destination: Path | None,
    *,
    label: str,
    mode: str | None = None,
    seed: int | None = None,
) -> None:
    _stage_log(stage="persist", phase="start", mode=mode, seed=seed, detail=label)
    started = time.perf_counter()
    persist_output(source, destination, label=label)
    _stage_log(
        stage="persist",
        phase="end",
        mode=mode,
        seed=seed,
        elapsed_seconds=time.perf_counter() - started,
        detail=label,
    )


def _split_count(path: Path) -> int:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Split file is empty: {path}")
    return len(lines)


def _split_prefixes(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise RuntimeError(f"Split file is empty: {path}")
    return values


def _prefix_sha256(prefixes: list[str]) -> str:
    return hashlib.sha256(
        ("".join(f"{value}\n" for value in prefixes)).encode("utf-8")
    ).hexdigest()


def _read_raster(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is required for the Sen1 input-quality preflight."
        ) from exc
    with rasterio.open(path) as dataset:
        return np.asarray(dataset.read(), dtype=np.float32)


def _terramind_mode_statistics(
    mode: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    profile = INPUT_PROFILES["sen1floods11_l1c"]
    means: dict[str, list[float]] = {}
    stds: dict[str, list[float]] = {}
    if "S2" in mode:
        means["S2L1C"] = list(profile.s2_mean)
        stds["S2L1C"] = list(profile.s2_std)
    if "S1" in mode:
        means["S1GRD"] = list(S1_MEAN)
        stds["S1GRD"] = list(S1_STD)
    return means, stds


def _build_terramind_input_quality_contracts(
    *,
    s1_root: Path,
    s2_root: Path,
    split_paths: dict[str, Path],
    modes: tuple[str, ...],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Scan all official inputs once and freeze modality-aware availability."""

    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {split: [] for split in split_paths} for mode in modes
    }
    prefix_sets = {
        split: _split_prefixes(path) for split, path in split_paths.items()
    }
    for split, prefixes in prefix_sets.items():
        split_role = "standard_test" if split == "test" else split
        for index, prefix in enumerate(prefixes, start=1):
            s1 = _read_raster(s1_root / f"{prefix}_S1Hand.tif")
            s2 = _read_raster(s2_root / f"{prefix}_S2Hand.tif")
            for mode in modes:
                arrays: dict[str, np.ndarray] = {}
                # Preserve the actual TerraTorch modality order.
                if "S2" in mode:
                    arrays["S2L1C"] = s2
                if "S1" in mode:
                    arrays["S1GRD"] = s1
                means, stds = _terramind_mode_statistics(mode)
                _normalized, quality = normalize_named_modalities(
                    arrays,
                    means=means,
                    stds=stds,
                    prefix=prefix,
                    split_role=split_role,
                )
                quality["split"] = split
                records[mode][split].append(quality)
            if index % 50 == 0:
                print(
                    "[terramind:input-quality] "
                    f"split={split} files={index}/{len(prefixes)}",
                    flush=True,
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        split_contracts = {
            split: {
                "prefix_sha256": _prefix_sha256(prefix_sets[split]),
                "summary": input_quality_summary(
                    records[mode][split],
                    mode=mode,
                ),
                "records": records[mode][split],
            }
            for split in split_paths
        }
        observed = {
            split: dict(
                split_contracts[split]["summary"][
                    "fully_missing_modalities_by_sample"
                ]
            )
            for split in split_paths
        }
        expected = FORMAL_COMPLETE_MODALITY_MISSING_EXPECTATION[mode]
        if observed != expected:
            raise RuntimeError(
                "Official TerraMind complete-modality-missing contract changed: "
                f"mode={mode}, expected={expected}, observed={observed}."
            )
        means, stds = _terramind_mode_statistics(mode)
        all_records = [
            record
            for split in split_paths
            for record in records[mode][split]
        ]
        payload = {
            "schema": "geobwer.sen1floods11.terramind_input_quality.v1",
            "sensor_mode": mode,
            "imputation_policy": SEN1_IMPUTATION_POLICY,
            "normalization_source": "frozen_terramind_pretraining_statistics",
            "means": means,
            "stds": stds,
            "source_artifacts_modified": False,
            "summary": input_quality_summary(all_records, mode=mode),
            "splits": split_contracts,
        }
        path = (
            output_dir
            / f"{mode.lower().replace('+', '_plus_')}.json"
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        results[mode] = {
            "path": path,
            "sha256": file_sha256(path),
            "contract": payload,
        }
    return results


def _bind_input_quality(
    export_root: Path,
    *,
    split: str,
    mode: str,
    contract: Mapping[str, Any],
    contract_path: Path,
    contract_sha256: str,
) -> Path:
    split_contract = contract["splits"][split]
    path = export_root / "input_quality_binding.json"
    path.write_text(
        json.dumps(
            {
                "schema": "geobwer.sen1floods11.input_quality_binding.v1",
                "split": split,
                "split_role": (
                    "standard_test" if split == "test" else split
                ),
                "sensor_mode": mode,
                "imputation_policy": SEN1_IMPUTATION_POLICY,
                "input_quality_contract": str(contract_path),
                "input_quality_contract_sha256": contract_sha256,
                "prefix_sha256": split_contract["prefix_sha256"],
                "summary": split_contract["summary"],
                "fully_missing_modality_records": [
                    record
                    for record in split_contract["records"]
                    if record["fully_missing_modality"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _probability_artifact(export_root: Path, value: Any) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        if raw.exists():
            return raw
        return export_root / "samples" / raw.name
    return export_root / raw


def _export_complete(path: Path, expected: int) -> bool:
    index_parts = sorted((path / "index_parts").glob("*.jsonl"))
    if not index_parts or not list(path.glob("writer_manifest_rank_*.json")):
        return False
    rows: dict[str, dict[str, Any]] = {}
    for part in index_parts:
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                sample_id = str(item["sample_id"])
                if sample_id in rows and rows[sample_id] != item:
                    raise RuntimeError(f"Conflicting duplicate probability index for sample_id={sample_id}.")
                rows[sample_id] = item
    return len(rows) == expected and all(
        _probability_artifact(path, row["probability_path"]).exists() for row in rows.values()
    )


def _audited_supervised_validation_exports(
    campaign_manifest_path: Path,
    audit_json_path: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Resolve nine immutable U-Net validation exports from PASS evidence."""

    if not campaign_manifest_path.is_file() or not audit_json_path.is_file():
        raise RuntimeError(
            "Formal TerraMind calibration requires the audited v0.4.28 U-Net "
            "campaign manifest and its external PASS audit JSON."
        )
    campaign = json.loads(
        campaign_manifest_path.read_text(encoding="utf-8")
    )
    audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    if (
        campaign.get("schema")
        != "geobwer.sen1floods11.supervised_panel.v6"
        or campaign.get("formal_evidence") is not True
        or campaign.get("package_version") != "0.4.28"
        or campaign.get("code_commit")
        != "60cff004057c99799ae3c9523a0eab5de4070f59"
        or audit.get("schema")
        != "geobwer.sen1floods11.unet_artifact_audit.v1"
        or audit.get("status") != "pass"
        or int(audit.get("model_count", -1)) != 9
        or audit.get("cross_model_sample_and_target_identity") != "exact"
        or audit.get("target", {}).get("campaign_manifest_sha256")
        != file_sha256(campaign_manifest_path)
        or audit.get("target", {}).get("code_commit")
        != campaign.get("code_commit")
    ):
        raise RuntimeError(
            "The supplied U-Net campaign/audit pair is not the frozen, "
            "successfully audited v0.4.28 nine-model panel."
        )
    exports: dict[str, Path] = {}
    for mode in MODES:
        slug = mode.lower().replace("+", "_plus_")
        for seed in (42, 73, 101):
            name = f"resnet34_unet_{slug}_seed_{seed}"
            run = campaign.get("runs", {}).get(name)
            if not isinstance(run, Mapping):
                raise RuntimeError(
                    f"Audited U-Net campaign is missing run={name}."
                )
            export = Path(str(run.get("validation_export", "")))
            if not _export_complete(export, 89):
                # Persistent campaign manifests intentionally retain their
                # original /content lineage. Resolve the audited Drive copy by
                # the frozen panel layout without rewriting that manifest.
                persistent_export = (
                    campaign_manifest_path.parent
                    / slug
                    / f"seed_{seed}"
                    / "probabilities"
                    / "validation"
                )
                if _export_complete(persistent_export, 89):
                    export = persistent_export
            if not _export_complete(export, 89):
                raise RuntimeError(
                    f"Audited U-Net validation export is unavailable or "
                    f"incomplete: run={name}, path={export}."
                )
            exports[name] = export
    return exports, {
        "campaign_manifest": str(campaign_manifest_path),
        "campaign_manifest_sha256": file_sha256(campaign_manifest_path),
        "audit_json": str(audit_json_path),
        "audit_json_sha256": file_sha256(audit_json_path),
        "validation_export_count": len(exports),
        "read_only": True,
    }


def _validated_prithvi_validation_export(
    campaign_manifest_path: Path,
    validation_export: Path,
) -> dict[str, Any]:
    """Bind the immutable Prithvi validation export to its formal manifest."""

    if not campaign_manifest_path.is_file():
        raise RuntimeError(
            "Formal TerraMind calibration requires the Prithvi campaign "
            f"manifest: {campaign_manifest_path}."
        )
    manifest = json.loads(
        campaign_manifest_path.read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema")
        != "geobwer.sen1floods11.prithvi_tl_probability_migration.v3"
        or manifest.get("formal_evidence") is not True
        or manifest.get("split_protocol")
        != "official_252_89_90_plus_15_bolivia_holdout"
        or int(manifest.get("train_count", -1)) != 252
        or int(manifest.get("validation_count", -1)) != 89
        or int(manifest.get("test_count", -1)) != 90
        or int(manifest.get("bolivia_holdout_count", -1)) != 15
        or int(manifest.get("combined_evaluation_count", -1)) != 105
        or manifest.get("bolivia_holdout_used_for_training_or_calibration")
        is not False
        or manifest.get("no_training_or_calibration_leakage") is not True
    ):
        raise RuntimeError(
            "The supplied Prithvi manifest is not a formal official "
            "252/89/90+15 probability migration."
        )
    declared = Path(
        str(manifest.get("probability_exports", {}).get("validation", ""))
    )
    if declared.name != validation_export.name or declared.parent.name != "probabilities":
        # In persistent copies the frozen manifest can retain its original
        # /content path, so compare the stable terminal layout rather than
        # requiring that disposable absolute path to still exist.
        if declared.parts[-2:] != ("probabilities", "validation"):
            raise RuntimeError(
                "Prithvi manifest does not declare the expected validation "
                f"export: declared={declared}, supplied={validation_export}."
            )
    if not _export_complete(validation_export, 89):
        raise RuntimeError(
            "Prithvi validation export is incomplete; expected 89 frozen "
            f"validation units at {validation_export}."
        )
    return {
        "campaign_manifest": str(campaign_manifest_path),
        "campaign_manifest_sha256": file_sha256(campaign_manifest_path),
        "validation_export": str(validation_export),
        "validation_row_count": 89,
        "read_only": True,
    }


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise RuntimeError(f"Could not resolve a frozen code commit: {value!r}.")
    return value


def _assert_frozen_checkout() -> str:
    commit = _git_head()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "Formal TerraMind execution requires an unmodified frozen "
            f"checkout; tracked changes were found:\n{status}"
        )
    return commit


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required completion artifact is missing: {path}.")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_diagnostic_export(
    path: Path,
    *,
    maximum_rows: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index_part in sorted((path / "index_parts").glob("*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in index_part.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not 1 <= len(rows) <= int(maximum_rows):
        raise RuntimeError(
            f"Diagnostic probability export must contain 1..{maximum_rows} rows, "
            f"observed={len(rows)} at {path}."
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"Diagnostic probability export has duplicate sample IDs: {path}.")
    shape_pairs: list[dict[str, Any]] = []
    observed_target_values: set[int] = set()
    aggregate_valid_pixel_count = 0
    all_ignore_row_count = 0
    for index, row in enumerate(rows):
        artifact_path = _probability_artifact(path, row["probability_path"])
        if not artifact_path.is_file():
            raise RuntimeError(f"Missing diagnostic probability artifact: {artifact_path}")
        with np.load(artifact_path) as artifact:
            probability_array = np.asarray(artifact["probabilities"])
            target_array = np.asarray(artifact["target"]).squeeze()
        if probability_array.ndim != 3 or probability_array.shape[0] != 2:
            raise RuntimeError(
                f"Diagnostic probability map must be [2,H,W], got "
                f"{probability_array.shape}: {artifact_path}."
            )
        if probability_array.shape[1:] != target_array.shape:
            raise RuntimeError(
                "Diagnostic probability/target shape mismatch at "
                f"{path}, row={index}: probability={probability_array.shape}, "
                f"target={target_array.shape}."
            )
        if not np.all(np.isfinite(probability_array)):
            raise RuntimeError(f"Diagnostic probability map contains NaN/Inf: {path}, row={index}.")
        if np.any((probability_array < 0.0) | (probability_array > 1.0)):
            raise RuntimeError(f"Diagnostic probability map is outside [0,1]: {path}, row={index}.")
        if not np.allclose(
            probability_array.sum(axis=0),
            1.0,
            atol=2e-4,
            rtol=2e-4,
        ):
            raise RuntimeError(
                f"Diagnostic class probabilities do not sum to one: {path}, row={index}."
            )
        raw_target_values = np.unique(target_array).tolist()
        try:
            numeric_target_values = [float(value) for value in raw_target_values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Diagnostic target contains non-numeric values: {path}, row={index}."
            ) from exc
        invalid_target_values = sorted(
            value
            for value in numeric_target_values
            if not np.isfinite(value) or value not in {-1.0, 0.0, 1.0}
        )
        if invalid_target_values:
            raise RuntimeError(
                "Diagnostic target contains values outside the formal "
                f"{{-1,0,1}} contract: {path}, row={index}, "
                f"invalid={invalid_target_values}."
            )
        target_values = {int(value) for value in numeric_target_values}
        observed_target_values.update(target_values)
        valid_array = np.isin(target_array, [0, 1])
        valid_pixel_count = int(valid_array.sum())
        aggregate_valid_pixel_count += valid_pixel_count
        if valid_pixel_count == 0:
            all_ignore_row_count += 1
        shape_pairs.append(
            {
                "sample_id": sample_ids[index],
                "probability_shape": list(probability_array.shape[1:]),
                "valid_pixel_count": valid_pixel_count,
                "observed_target_values": sorted(target_values),
            }
        )
    if aggregate_valid_pixel_count <= 0:
        raise RuntimeError(
            "Diagnostic probability export contains no valid hand-labeled "
            f"pixels across any row: {path}."
        )
    return {
        "row_count": len(rows),
        "sample_ids": sample_ids,
        "samples": shape_pairs,
        "all_ignore_row_count": all_ignore_row_count,
        "valid_row_count": len(rows) - all_ignore_row_count,
        "aggregate_valid_pixel_count": aggregate_valid_pixel_count,
        "observed_target_values": sorted(observed_target_values),
    }


def _checkpoint(run_dir: Path) -> Path | None:
    best = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    if len(best) > 1:
        raise RuntimeError(f"More than one best checkpoint exists under {run_dir}; provenance is ambiguous: {best}")
    return best[0] if best else None


def _fit_if_needed(
    config: Path,
    run_dir: Path,
    *,
    mode: str,
    seed: int,
    backbone_checkpoint_sha256: str,
    dry_run: bool,
    reuse_only: bool = False,
) -> Path | None:
    protocol_path = run_dir / "fit_protocol.json"
    current_protocol = {
        "schema": "geobwer.terramind.fit_protocol.v2",
        "config_sha256": file_sha256(config),
        "backbone_checkpoint_sha256": backbone_checkpoint_sha256,
        "backbone_revision": TERRAMIND_OFFICIAL_REVISION,
        "training_length_policy": "fixed_100_epochs_no_early_stopping",
        "early_stopping_enabled": False,
    }
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != current_protocol:
            raise RuntimeError(
                f"Fit protocol changed under an existing output directory: {run_dir}. "
                "Use a new output directory; do not resume a checkpoint produced by another protocol."
            )
    else:
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(current_protocol, indent=2), encoding="utf-8")
    best = _checkpoint(run_dir)
    marker = run_dir / "fit_complete.json"
    if best is not None and marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if (
            payload.get("checkpoint_sha256") == file_sha256(best)
            and payload.get("fit_protocol_sha256") == file_sha256(protocol_path)
        ):
            print(f"[terramind:campaign] reusing complete checkpoint {best}")
            return best
    if reuse_only:
        raise RuntimeError(
            "--resume-source-root forbids fitting, but the hydrated "
            f"fit_complete/checkpoint contract is missing or incompatible: {run_dir}."
        )
    if dry_run:
        return best
    command = _terratorch_command() + ["fit", "-c", str(config)]
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        command += ["--ckpt_path", str(last)]
    _run(command, stage="fit", mode=mode, seed=seed)
    best = _checkpoint(run_dir)
    if best is None:
        raise RuntimeError(f"TerraTorch fit finished without one best checkpoint under {run_dir}.")
    marker.write_text(
        json.dumps(
            {
                "checkpoint": str(best),
                "checkpoint_sha256": file_sha256(best),
                "fit_protocol": str(protocol_path),
                "fit_protocol_sha256": file_sha256(protocol_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best


def _predict_if_needed(
    config: Path,
    checkpoint: Path | None,
    output: Path,
    *,
    mode: str,
    seed: int,
    split: str,
    expected: int,
    dry_run: bool,
    input_quality_contract_sha256: str,
    reuse_only: bool = False,
) -> None:
    prediction_protocol = {
        "schema": "geobwer.sen1floods11.terramind_prediction_protocol.v1",
        "config_sha256": file_sha256(config),
        "checkpoint_sha256": (
            file_sha256(checkpoint)
            if checkpoint is not None and checkpoint.is_file()
            else None
        ),
        "input_quality_contract_sha256": str(
            input_quality_contract_sha256
        ),
        "imputation_policy": SEN1_IMPUTATION_POLICY,
        "expected_row_count": int(expected),
    }
    completion = output / "prediction_completion_contract.json"
    if _export_complete(output, expected):
        if not completion.is_file():
            raise RuntimeError(
                "A complete legacy TerraMind probability export has no v0.4.27 "
                f"preprocessing completion contract: {output}. Use a new output "
                "directory; do not relabel raw-zero predictions as mean-imputed."
            )
        previous = json.loads(completion.read_text(encoding="utf-8"))
        if previous != prediction_protocol:
            raise RuntimeError(
                f"TerraMind prediction completion contract drifted: {output}."
            )
        print(f"[terramind:campaign] reusing complete probability export {output}")
        return
    if reuse_only:
        raise RuntimeError(
            "--resume-source-root forbids regenerating the frozen validation "
            f"probabilities, but their completion contract is missing or incompatible: {output}."
        )
    if dry_run:
        return
    if checkpoint is None:
        raise RuntimeError("Prediction requires a completed checkpoint.")
    _run(
        _terratorch_predict_command()
        + ["predict", "-c", str(config), "--ckpt_path", str(checkpoint)],
        stage=f"predict_{split}",
        mode=mode,
        seed=seed,
    )
    if not _export_complete(output, expected):
        raise RuntimeError(
            f"Probability export failed completeness check: expected={expected}, path={output}. "
            "Do not proceed to BWER with partial predictions."
        )
    completion.write_text(
        json.dumps(prediction_protocol, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    formal_run = not (args.dry_run or args.smoke_only)
    if args.dry_run and args.smoke_only:
        raise ValueError("--dry-run and --smoke-only are mutually exclusive.")
    if formal_run:
        frozen_code_commit = _assert_frozen_checkout()
        runtime_version = validate_terratorch_runtime()
        if runtime_version != "1.2.10":
            raise RuntimeError(
                "The frozen formal TerraMind campaign requires "
                f"TerraTorch 1.2.10; observed={runtime_version}."
            )
        local_text = args.output_dir.as_posix().rstrip("/")
        persistent_text = (
            args.persistent_output_dir.as_posix().rstrip("/")
            if args.persistent_output_dir is not None
            else ""
        )
        if (
            not local_text.startswith("/content/")
            or local_text.startswith("/content/drive/")
            or not persistent_text.startswith(
                "/content/drive/MyDrive/rsfm_fairness_audit/"
            )
        ):
            raise RuntimeError(
                "Formal TerraMind fitting must use a non-Drive /content "
                "output directory and a persistent mirror under "
                "/content/drive/MyDrive/rsfm_fairness_audit/."
            )
        if args.resume_source_root is not None:
            resume_text = args.resume_source_root.as_posix().rstrip("/")
            if (
                not resume_text.startswith(
                    "/content/drive/MyDrive/rsfm_fairness_audit/"
                )
                or resume_text == persistent_text
            ):
                raise RuntimeError(
                    "--resume-source-root must be a distinct, read-only Drive "
                    "root under the project directory. New artifacts must use "
                    "a different --persistent-output-dir."
                )
    if args.dry_run and not args.checkpoint.is_file():
        backbone_checkpoint_sha256 = TERRAMIND_OFFICIAL_SHA256
        print(
            "[terramind:campaign] dry-run only: checkpoint bytes are absent; generated configs retain the "
            "official expected checkpoint identity.",
            flush=True,
        )
    else:
        _, backbone_checkpoint_sha256 = validate_terramind_checkpoint(args.checkpoint)
    seeds = (int(args.seed),) if args.seed is not None else tuple(map(int, args.seeds))
    if formal_run and (
        seeds != (42, 73, 101)
        or int(args.batch_size) != 8
        or int(args.max_epochs) != 100
        or int(args.num_workers) != 8
        or args.persistent_workers is not True
        or int(args.prefetch_factor) != 4
    ):
        raise RuntimeError(
            "The frozen formal TerraMind campaign requires seeds=42,73,101, "
            "batch_size=8, max_epochs=100, num_workers=8, "
            "persistent_workers=true, and prefetch_factor=4. Operational "
            "alternatives are diagnostic-only and require --dry-run or "
            "--smoke-only."
        )
    external_validation_exports: dict[str, Path] = {}
    audited_supervised_lineage: dict[str, Any] | None = None
    prithvi_lineage: dict[str, Any] | None = None
    if formal_run:
        if (
            args.supervised_campaign_manifest is None
            or args.supervised_audit_json is None
            or args.prithvi_validation_export is None
            or args.prithvi_campaign_manifest is None
        ):
            raise RuntimeError(
                "Formal all-model spatial calibration requires "
                "--supervised-campaign-manifest, --supervised-audit-json, "
                "--prithvi-campaign-manifest, and "
                "--prithvi-validation-export."
            )
        (
            external_validation_exports,
            audited_supervised_lineage,
        ) = _audited_supervised_validation_exports(
            args.supervised_campaign_manifest,
            args.supervised_audit_json,
        )
        prithvi_lineage = _validated_prithvi_validation_export(
            args.prithvi_campaign_manifest,
            args.prithvi_validation_export,
        )
        external_validation_exports[
            "prithvi_eo_v2_300_tl_s2"
        ] = args.prithvi_validation_export
    for name, path in args.additional_validation_export:
        if formal_run:
            raise RuntimeError(
                "Ad-hoc --additional-validation-export values are not accepted "
                "in the frozen formal panel. Use the audited U-Net and explicit "
                "Prithvi inputs."
            )
        if name in external_validation_exports:
            raise RuntimeError(f"Duplicate --additional-validation-export name: {name}")
        external_validation_exports[name] = path
    hydrate_output(args.output_dir, args.persistent_output_dir)
    if args.resume_source_root is not None:
        if not args.resume_source_root.is_dir():
            raise RuntimeError(
                f"Frozen resume source does not exist: {args.resume_source_root}."
            )
        copy_changed_tree(
            args.resume_source_root,
            args.output_dir,
            label="hydrate-read-only-calibration-failed-source",
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_report = prepare_terramind_sen1_splits(
        {
            "s1_root": args.s1_root,
            "s2_root": args.s2_root,
            "label_root": args.label_root,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "bolivia_split": args.bolivia_split,
        },
        args.output_dir / "terratorch_splits",
    )
    terratorch_splits = {
        name: Path(path) for name, path in source_report["terratorch_split_paths"].items()
    }
    modes = tuple(args.mode or MODES)
    if set(modes) != set(MODES) and not (args.dry_run or args.smoke_only):
        raise RuntimeError(
            "The final primary campaign requires S1, S2, and S1+S2 together so block calibration and comparisons "
            "use a common model set. Use --dry-run to inspect a subset of configs."
        )
    input_quality_contracts = (
        {}
        if args.dry_run
        else _build_terramind_input_quality_contracts(
            s1_root=args.s1_root,
            s2_root=args.s2_root,
            split_paths={
                key: terratorch_splits[key]
                for key in ("train", "validation", "test", "bolivia_holdout")
            },
            modes=modes,
            output_dir=args.output_dir / "input_quality",
        )
    )
    official_train_prefixes = [
        line.strip()
        for line in terratorch_splits["train"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inner_splits: dict[int, dict[str, Path]] = {}
    for seed in seeds:
        fit_indices, selection_indices = group_disjoint_inner_split(
            [prefix.split("_", 1)[0] for prefix in official_train_prefixes],
            validation_fraction=0.18,
            seed=seed,
        )
        seed_root = args.output_dir / "terratorch_splits" / f"model_selection_seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        fit_path = seed_root / "fit_prefixes.txt"
        selection_path = seed_root / "selection_prefixes.txt"
        fit_path.write_text(
            "".join(official_train_prefixes[index] + "\n" for index in fit_indices),
            encoding="utf-8",
        )
        selection_path.write_text(
            "".join(
                official_train_prefixes[index] + "\n" for index in selection_indices
            ),
            encoding="utf-8",
        )
        inner_splits[seed] = {"fit": fit_path, "selection": selection_path}
    (args.output_dir / "source_preflight.json").write_text(
        json.dumps(
            {
                **source_report,
                "terramind_checkpoint": str(args.checkpoint),
                "terramind_checkpoint_sha256": backbone_checkpoint_sha256,
                "terramind_revision": TERRAMIND_OFFICIAL_REVISION,
                "package_version": __version__,
                "code_commit": (
                    frozen_code_commit if formal_run else "diagnostic"
                ),
                "operational_contract": {
                    "num_workers": int(args.num_workers),
                    "pin_memory": True,
                    "persistent_workers": bool(args.persistent_workers),
                    "prefetch_factor": int(args.prefetch_factor),
                    "checkpoint_mirror_every_n_epochs": int(
                        args.checkpoint_mirror_every_n_epochs
                    ),
                    "checkpoint_interval_payload": "last.ckpt_only",
                    "checkpoint_fit_end_payload": "last_plus_unique_best",
                    "gpu_log_every_n_steps": int(
                        args.gpu_log_every_n_steps
                    ),
                    "live_training_filesystem": "/content",
                    "drive_role": "checkpoint_and_completed_artifact_mirror",
                },
                "training_length_contract": {
                    "max_epochs": int(args.max_epochs),
                    "early_stopping_enabled": False,
                    "policy": "fixed_100_epochs_no_early_stopping",
                    "reason": (
                        "Frozen cross-seed/cross-modality protocol; best "
                        "validation-loss checkpoint is selected after the same "
                        "100-epoch opportunity for every run."
                    ),
                },
                "audited_supervised_lineage": audited_supervised_lineage,
                "prithvi_lineage": prithvi_lineage,
                "calibration_panel_scope": CALIBRATION_PANEL_SCOPE,
                "expected_calibration_model_names": list(
                    _expected_calibration_model_names()
                ),
                "resume_source": (
                    {
                        "path": str(args.resume_source_root),
                        "read_only": True,
                        "new_persistent_output_is_distinct": True,
                        "fit_and_validation_policy": (
                            "completion_contract_match_or_hard_fail_no_reexecution"
                        ),
                    }
                    if args.resume_source_root is not None
                    else None
                ),
                "model_selection": {
                    str(seed): {
                        "policy": "official_train_inner_event_disjoint",
                        "fit_split": str(paths["fit"]),
                        "selection_split": str(paths["selection"]),
                        "fit_count": _split_count(paths["fit"]),
                        "selection_count": _split_count(paths["selection"]),
                        "outer_validation_used_for_model_selection": False,
                    }
                    for seed, paths in inner_splits.items()
                },
                "input_quality_contracts": {
                    mode: {
                        "path": str(artifacts["path"]),
                        "sha256": str(artifacts["sha256"]),
                        "imputation_policy": SEN1_IMPUTATION_POLICY,
                        "summary": artifacts["contract"]["summary"],
                    }
                    for mode, artifacts in input_quality_contracts.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _persist_with_log(
        args.output_dir,
        args.persistent_output_dir,
        label="source-preflight",
    )
    checkpoints: dict[str, Path | None] = {}
    validation_exports: dict[str, Path] = dict(external_validation_exports)
    test_exports: dict[str, Path] = {}
    bolivia_exports: dict[str, Path] = {}
    for mode in modes:
        slug = mode.lower().replace("+", "_plus_")
        for seed in seeds:
            run_name = f"terramind_v1_base_{slug}_seed_{seed}"
            run_dir = args.output_dir / slug / f"seed_{seed}"
            config_dir = run_dir / "configs"
            validation_output = run_dir / "probabilities" / "validation"
            test_output = run_dir / "probabilities" / "test"
            bolivia_output = run_dir / "probabilities" / "bolivia_holdout"
            prediction_common = {
                "sensor_mode": mode,
                "s1_root": args.s1_root,
                "s2_root": args.s2_root,
                "label_root": args.label_root,
                "train_split": terratorch_splits["train"],
                "val_split": terratorch_splits["validation"],
                "test_split": terratorch_splits["test"],
                "run_dir": run_dir,
                "backbone_checkpoint_path": args.checkpoint,
                "seed": seed,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "persistent_workers": args.persistent_workers,
                "prefetch_factor": args.prefetch_factor,
                "gpu_log_every_n_steps": args.gpu_log_every_n_steps,
                "max_epochs": 1 if args.smoke_only else args.max_epochs,
                "fast_dev_run": False,
                "diagnostic_batch_limit": SMOKE_BATCH_LIMIT if args.smoke_only else None,
                "persistent_checkpoint_dir": (
                    args.persistent_output_dir / slug / f"seed_{seed}" / "checkpoints"
                    if args.persistent_output_dir is not None
                    else None
                ),
                "checkpoint_mirror_every_n_epochs": args.checkpoint_mirror_every_n_epochs,
            }
            fit_common = {
                **prediction_common,
                "train_split": inner_splits[seed]["fit"],
                "val_split": inner_splits[seed]["selection"],
            }
            fit_config = write_terramind_sen1floods11_config(
                config_dir / "fit.yaml", **fit_common
            )
            if args.smoke_only:
                validation_config = write_terramind_sen1floods11_config(
                    config_dir / "predict_validation.yaml",
                    **prediction_common,
                    prediction_split="validation",
                    probability_output_dir=validation_output,
                )
                test_config = write_terramind_sen1floods11_config(
                    config_dir / "predict_test.yaml",
                    **prediction_common,
                    prediction_split="test",
                    probability_output_dir=test_output,
                )
                checkpoint = _fit_if_needed(
                    fit_config,
                    run_dir,
                    mode=mode,
                    seed=seed,
                    backbone_checkpoint_sha256=backbone_checkpoint_sha256,
                    dry_run=False,
                )
                assert checkpoint is not None
                _run(
                    _terratorch_predict_command()
                    + ["predict", "-c", str(validation_config), "--ckpt_path", str(checkpoint)],
                    stage="predict_validation",
                    mode=mode,
                    seed=seed,
                )
                _run(
                    _terratorch_predict_command()
                    + ["predict", "-c", str(test_config), "--ckpt_path", str(checkpoint)],
                    stage="predict_test",
                    mode=mode,
                    seed=seed,
                )
                validation_quality_binding = _bind_input_quality(
                    validation_output,
                    split="validation",
                    mode=mode,
                    contract=input_quality_contracts[mode]["contract"],
                    contract_path=input_quality_contracts[mode]["path"],
                    contract_sha256=input_quality_contracts[mode]["sha256"],
                )
                test_quality_binding = _bind_input_quality(
                    test_output,
                    split="test",
                    mode=mode,
                    contract=input_quality_contracts[mode]["contract"],
                    contract_path=input_quality_contracts[mode]["path"],
                    contract_sha256=input_quality_contracts[mode]["sha256"],
                )
                validation_diagnostics = _validate_diagnostic_export(
                    validation_output,
                    maximum_rows=args.batch_size * SMOKE_BATCH_LIMIT,
                )
                test_diagnostics = _validate_diagnostic_export(
                    test_output,
                    maximum_rows=args.batch_size * SMOKE_BATCH_LIMIT,
                )
                overlap = sorted(
                    set(validation_diagnostics["sample_ids"])
                    & set(test_diagnostics["sample_ids"])
                )
                if overlap:
                    raise RuntimeError(
                        "TerraMind diagnostic validation/test sample IDs overlap: "
                        + ", ".join(overlap[:10])
                    )
                smoke_manifest = run_dir / "diagnostic_manifest.json"
                smoke_manifest.write_text(
                    json.dumps(
                        {
                            "schema": "geobwer.sen1floods11.terramind_diagnostic.v3",
                            "formal_evidence": False,
                            "reason": "bounded_end_to_end_real_gpu_smoke",
                            "sensor_mode": mode,
                            "seed": seed,
                            "fit_config": str(fit_config),
                            "fit_config_sha256": file_sha256(fit_config),
                            "pretraining_checkpoint_sha256": backbone_checkpoint_sha256,
                            "trained_checkpoint": str(checkpoint),
                            "trained_checkpoint_sha256": file_sha256(checkpoint),
                            "validation_export": validation_diagnostics,
                            "test_export": test_diagnostics,
                            "input_quality_contract": {
                                "path": str(
                                    input_quality_contracts[mode]["path"]
                                ),
                                "sha256": str(
                                    input_quality_contracts[mode]["sha256"]
                                ),
                                "imputation_policy": SEN1_IMPUTATION_POLICY,
                            },
                            "validation_input_quality_binding": {
                                "path": str(validation_quality_binding),
                                "sha256": file_sha256(
                                    validation_quality_binding
                                ),
                            },
                            "test_input_quality_binding": {
                                "path": str(test_quality_binding),
                                "sha256": file_sha256(test_quality_binding),
                            },
                            "validation_test_sample_overlap": 0,
                            "bounded_batch_count": SMOKE_BATCH_LIMIT,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                _persist_with_log(
                    run_dir,
                    (
                        args.persistent_output_dir / slug / f"seed_{seed}"
                        if args.persistent_output_dir
                        else None
                    ),
                    label=f"{run_name}-diagnostic",
                    mode=mode,
                    seed=seed,
                )
                continue
            validation_config = write_terramind_sen1floods11_config(
                config_dir / "predict_validation.yaml",
                **prediction_common,
                prediction_split="validation",
                probability_output_dir=validation_output,
            )
            test_config = write_terramind_sen1floods11_config(
                config_dir / "predict_test.yaml",
                **prediction_common,
                prediction_split="test",
                probability_output_dir=test_output,
            )
            bolivia_config = write_terramind_sen1floods11_config(
                config_dir / "predict_bolivia_holdout.yaml",
                **{
                    **prediction_common,
                    "test_split": terratorch_splits["bolivia_holdout"],
                },
                prediction_split="bolivia_holdout",
                probability_output_dir=bolivia_output,
            )
            checkpoint = _fit_if_needed(
                fit_config,
                run_dir,
                mode=mode,
                seed=seed,
                backbone_checkpoint_sha256=backbone_checkpoint_sha256,
                dry_run=args.dry_run,
                reuse_only=args.resume_source_root is not None,
            )
            _persist_with_log(
                run_dir,
                (
                    args.persistent_output_dir / slug / f"seed_{seed}"
                    if args.persistent_output_dir
                    else None
                ),
                label=f"{run_name}-fit",
                mode=mode,
                seed=seed,
            )
            _predict_if_needed(
                validation_config,
                checkpoint,
                validation_output,
                mode=mode,
                seed=seed,
                split="validation",
                expected=_split_count(terratorch_splits["validation"]),
                dry_run=args.dry_run,
                input_quality_contract_sha256=(
                    input_quality_contracts[mode]["sha256"]
                    if not args.dry_run
                    else "dry_run_not_executed"
                ),
                reuse_only=args.resume_source_root is not None,
            )
            if not args.dry_run:
                _bind_input_quality(
                    validation_output,
                    split="validation",
                    mode=mode,
                    contract=input_quality_contracts[mode]["contract"],
                    contract_path=input_quality_contracts[mode]["path"],
                    contract_sha256=input_quality_contracts[mode]["sha256"],
                )
            _persist_with_log(
                validation_output,
                (
                    args.persistent_output_dir
                    / slug
                    / f"seed_{seed}"
                    / "probabilities"
                    / "validation"
                    if args.persistent_output_dir
                    else None
                ),
                label=f"{run_name}-validation-predictions",
                mode=mode,
                seed=seed,
            )
            checkpoints[run_name] = checkpoint
            validation_exports[run_name] = validation_output
            test_exports[run_name] = test_output
            bolivia_exports[run_name] = bolivia_output
            # Materialize test config before the validation-only scale is chosen.
            assert test_config.exists()
            assert bolivia_config.exists()
    if args.smoke_only:
        diagnostic_manifest = args.output_dir / "diagnostic_panel_manifest.json"
        diagnostic_manifest.write_text(
            json.dumps(
                {
                    "schema": "geobwer.sen1floods11.terramind_panel_diagnostic.v2",
                    "formal_evidence": False,
                    "reason": "bounded_end_to_end_real_gpu_smoke",
                    "modes": list(modes),
                    "seeds": list(seeds),
                    "pretraining_checkpoint_sha256": backbone_checkpoint_sha256,
                    "bounded_batch_count_per_stage": SMOKE_BATCH_LIMIT,
                    "checks": [
                        "real_train_batch",
                        "real_validation_batch",
                        "checkpoint_saved_and_reloaded",
                        "validation_probability_writer",
                        "test_probability_writer",
                        "finite_probability_maps_in_unit_interval",
                        "validation_test_sample_disjointness",
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _persist_with_log(
            args.output_dir,
            args.persistent_output_dir,
            label="sen1-terramind-diagnostic-complete",
        )
        print(f"[terramind:campaign] diagnostic complete: {diagnostic_manifest}")
        return
    if args.dry_run:
        _persist_with_log(
            args.output_dir,
            args.persistent_output_dir,
            label="dry-run-configs",
        )
        print(f"[terramind:campaign] dry-run configs ready under {args.output_dir}")
        return

    expected_calibration_models = _expected_calibration_model_names()
    observed_calibration_models = tuple(sorted(validation_exports))
    if observed_calibration_models != expected_calibration_models:
        raise RuntimeError(
            "Frozen 19-model calibration panel mismatch before calibration: "
            f"expected={list(expected_calibration_models)}, "
            f"observed={list(observed_calibration_models)}."
        )
    calibration_path = args.output_dir / "common_spatial_block_calibration.json"
    calibration_failure_path = (
        args.output_dir / "calibration_failure_report.json"
    )
    calibration = calibrate_common_sen1_spatial_blocks(
        validation_exports,
        calibration_path,
        metadata_csv=args.metadata_csv,
        n_simulations=args.calibration_simulations,
        n_bootstrap=args.calibration_bootstrap,
        seed=seeds[0],
        minimum_moderate_tail_power=args.minimum_moderate_tail_power,
        calibration_panel_scope=CALIBRATION_PANEL_SCOPE,
        expected_model_names=expected_calibration_models,
        failure_output_json=calibration_failure_path,
        return_invalid=True,
    )
    calibration_valid = (
        calibration.get("validity") == "valid"
        and calibration.get("all_models_passed") is True
    )
    if not calibration_valid:
        invalid_contract = args.output_dir / "calibration_invalid_contract.json"
        invalid_contract.write_text(
            json.dumps(
                {
                    "schema": (
                        "geobwer.sen1floods11."
                        "terramind_calibration_invalid.v1"
                    ),
                    "status": "calibration_invalid",
                    "formal_evidence": True,
                    "validation_only": True,
                    "calibration_panel_scope": CALIBRATION_PANEL_SCOPE,
                    "model_count": len(expected_calibration_models),
                    "model_names": list(expected_calibration_models),
                    "calibration_failure_report": _artifact_record(
                        calibration_failure_path, args.output_dir
                    ),
                    "test_or_bolivia_used_for_calibration": False,
                    "permitted_next_stage": (
                        "descriptive_test_and_bolivia_probability_export_only"
                    ),
                    "prohibited": [
                        "scale_dependent_inferential_geobwer",
                        "bootstrap_significance",
                        "model_panel_inference",
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    _persist_with_log(
        args.output_dir,
        args.persistent_output_dir,
        label=(
            "spatial-calibration"
            if calibration_valid
            else "spatial-calibration-invalid-evidence"
        ),
    )
    terratorch_version = importlib.metadata.version("terratorch")
    panel_tables: dict[str, Path] = {}
    panel_protocol: BWERProtocol | None = None
    completed_run_artifacts: dict[str, dict[str, Any]] = {}
    for mode in modes:
        slug = mode.lower().replace("+", "_plus_")
        for seed in seeds:
            run_name = f"terramind_v1_base_{slug}_seed_{seed}"
            run_dir = args.output_dir / slug / f"seed_{seed}"
            test_config = run_dir / "configs" / "predict_test.yaml"
            bolivia_config = run_dir / "configs" / "predict_bolivia_holdout.yaml"
            _predict_if_needed(
                test_config,
                checkpoints[run_name],
                test_exports[run_name],
                mode=mode,
                seed=seed,
                split="test",
                expected=_split_count(terratorch_splits["test"]),
                dry_run=False,
                input_quality_contract_sha256=input_quality_contracts[mode][
                    "sha256"
                ],
            )
            test_quality_binding = _bind_input_quality(
                test_exports[run_name],
                split="test",
                mode=mode,
                contract=input_quality_contracts[mode]["contract"],
                contract_path=input_quality_contracts[mode]["path"],
                contract_sha256=input_quality_contracts[mode]["sha256"],
            )
            _persist_with_log(
                test_exports[run_name],
                (
                    args.persistent_output_dir
                    / slug
                    / f"seed_{seed}"
                    / "probabilities"
                    / "test"
                    if args.persistent_output_dir
                    else None
                ),
                label=f"{run_name}-test-predictions",
                mode=mode,
                seed=seed,
            )
            _predict_if_needed(
                bolivia_config,
                checkpoints[run_name],
                bolivia_exports[run_name],
                mode=mode,
                seed=seed,
                split="bolivia_holdout",
                expected=_split_count(terratorch_splits["bolivia_holdout"]),
                dry_run=False,
                input_quality_contract_sha256=input_quality_contracts[mode][
                    "sha256"
                ],
            )
            bolivia_quality_binding = _bind_input_quality(
                bolivia_exports[run_name],
                split="bolivia_holdout",
                mode=mode,
                contract=input_quality_contracts[mode]["contract"],
                contract_path=input_quality_contracts[mode]["path"],
                contract_sha256=input_quality_contracts[mode]["sha256"],
            )
            _persist_with_log(
                bolivia_exports[run_name],
                (
                    args.persistent_output_dir
                    / slug
                    / f"seed_{seed}"
                    / "probabilities"
                    / "bolivia_holdout"
                    if args.persistent_output_dir
                    else None
                ),
                label=f"{run_name}-bolivia-holdout-predictions",
                mode=mode,
                seed=seed,
            )
            combined_export = combine_sen1_evaluation_exports(
                test_exports[run_name],
                bolivia_exports[run_name],
                run_dir / "probabilities" / "combined_held_out",
            )
            if not calibration_valid:
                descriptive_root = run_dir / "descriptive_only_outputs"
                descriptive_paths = {
                    "validation": write_sen1_descriptive_probability_report(
                        validation_exports[run_name],
                        descriptive_root / "validation.json",
                        model_name=run_name,
                        split_role="validation_calibration_population",
                        metadata_csv=args.metadata_csv,
                    ),
                    "standard_test": write_sen1_descriptive_probability_report(
                        test_exports[run_name],
                        descriptive_root / "standard_test.json",
                        model_name=run_name,
                        split_role="standard_test",
                        metadata_csv=args.metadata_csv,
                    ),
                    "bolivia_holdout": write_sen1_descriptive_probability_report(
                        bolivia_exports[run_name],
                        descriptive_root / "bolivia_holdout.json",
                        model_name=run_name,
                        split_role="bolivia_holdout",
                        metadata_csv=args.metadata_csv,
                    ),
                    "combined_held_out": write_sen1_descriptive_probability_report(
                        combined_export,
                        descriptive_root / "combined_held_out.json",
                        model_name=run_name,
                        split_role="combined_held_out",
                        metadata_csv=args.metadata_csv,
                    ),
                }
                split_report = descriptive_root / "descriptive_split_report.json"
                split_report.write_text(
                    json.dumps(
                        {
                            "schema": (
                                "geobwer.sen1floods11."
                                "descriptive_split_panel.v1"
                            ),
                            "status": "descriptive_only",
                            "formal_evidence": False,
                            "model": run_name,
                            "calibration_status": "calibration_invalid",
                            "calibration_failure_report_sha256": file_sha256(
                                calibration_failure_path
                            ),
                            "views": {
                                name: json.loads(path.read_text(encoding="utf-8"))
                                for name, path in descriptive_paths.items()
                            },
                            "inferential_geobwer_run": False,
                            "bootstrap_run": False,
                            "model_panel_inference_run": False,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                run_artifact_paths = {
                    "checkpoint": checkpoints[run_name],
                    "fit_protocol": run_dir / "fit_protocol.json",
                    "fit_completion": run_dir / "fit_complete.json",
                    "validation_prediction_contract": (
                        validation_exports[run_name]
                        / "prediction_completion_contract.json"
                    ),
                    "test_prediction_contract": (
                        test_exports[run_name]
                        / "prediction_completion_contract.json"
                    ),
                    "bolivia_prediction_contract": (
                        bolivia_exports[run_name]
                        / "prediction_completion_contract.json"
                    ),
                    "descriptive_split_report": split_report,
                    **{
                        f"descriptive_{name}": path
                        for name, path in descriptive_paths.items()
                    },
                }
                completed_run_artifacts[run_name] = {
                    key: _artifact_record(Path(path), args.output_dir)
                    for key, path in run_artifact_paths.items()
                    if path is not None
                }
                _persist_with_log(
                    run_dir,
                    (
                        args.persistent_output_dir / slug / f"seed_{seed}"
                        if args.persistent_output_dir
                        else None
                    ),
                    label=f"{run_name}-descriptive-only",
                    mode=mode,
                    seed=seed,
                )
                continue
            standard_bundle = finalize_sen1floods11_segmentation(
                test_exports[run_name],
                run_dir / "formal_outputs" / "standard_test",
                model_name=run_name,
                checkpoint_path=checkpoints[run_name],
                pretraining_checkpoint_path=args.checkpoint,
                pretraining_checkpoint_sha256=backbone_checkpoint_sha256,
                protocol_path=args.protocol,
                block_calibration_path=calibration_path,
                metadata_csv=args.metadata_csv,
                split="standard_test",
                sensor_mode=mode,
                terratorch_version=terratorch_version,
                model_selection_lineage={
                    "model_selection": "official_train_inner_event_disjoint",
                    "model_selection_fit_split": str(inner_splits[seed]["fit"]),
                    "model_selection_holdout_split": str(
                        inner_splits[seed]["selection"]
                    ),
                    "outer_validation_used_for_model_selection": False,
                    "seed": seed,
                    "input_quality_binding": str(
                        test_quality_binding
                    ),
                    "input_quality_binding_sha256": file_sha256(
                        test_quality_binding
                    ),
                },
                evaluation_split_role="standard_test",
            )
            bolivia_bundle = finalize_sen1floods11_segmentation(
                bolivia_exports[run_name],
                run_dir / "formal_outputs" / "bolivia_holdout",
                model_name=run_name,
                checkpoint_path=checkpoints[run_name],
                pretraining_checkpoint_path=args.checkpoint,
                pretraining_checkpoint_sha256=backbone_checkpoint_sha256,
                protocol_path=args.protocol,
                block_calibration_path=calibration_path,
                metadata_csv=args.metadata_csv,
                split="bolivia_holdout",
                sensor_mode=mode,
                terratorch_version=terratorch_version,
                model_selection_lineage={
                    "model_selection": "official_train_inner_event_disjoint",
                    "outer_validation_used_for_model_selection": False,
                    "bolivia_holdout_used_for_model_selection": False,
                    "seed": seed,
                    "input_quality_binding": str(
                        bolivia_quality_binding
                    ),
                    "input_quality_binding_sha256": file_sha256(
                        bolivia_quality_binding
                    ),
                },
                evaluation_split_role="bolivia_holdout",
            )
            bundle = finalize_sen1floods11_segmentation(
                combined_export,
                run_dir / "formal_outputs" / "combined_held_out",
                model_name=run_name,
                checkpoint_path=checkpoints[run_name],
                pretraining_checkpoint_path=args.checkpoint,
                pretraining_checkpoint_sha256=backbone_checkpoint_sha256,
                protocol_path=args.protocol,
                block_calibration_path=calibration_path,
                metadata_csv=args.metadata_csv,
                split="combined_held_out",
                sensor_mode=mode,
                terratorch_version=terratorch_version,
                model_selection_lineage={
                    "model_selection": "official_train_inner_event_disjoint",
                    "outer_validation_used_for_model_selection": False,
                    "bolivia_holdout_used_for_model_selection": False,
                    "seed": seed,
                    "input_quality_contract": str(
                        input_quality_contracts[mode]["path"]
                    ),
                    "input_quality_contract_sha256": str(
                        input_quality_contracts[mode]["sha256"]
                    ),
                    "imputation_policy": SEN1_IMPUTATION_POLICY,
                    "standard_test_formal_manifest": str(
                        standard_bundle.manifest
                    ),
                    "bolivia_holdout_formal_manifest": str(
                        bolivia_bundle.manifest
                    ),
                },
                evaluation_split_role="combined_held_out",
            )
            write_sen1_evaluation_split_report(
                run_dir / "formal_outputs" / "evaluation_split_report.json",
                standard_test_bundle=standard_bundle,
                bolivia_holdout_bundle=bolivia_bundle,
                combined_held_out_bundle=bundle,
            )
            (
                calibration_rows,
                calibration_probabilities,
                calibration_targets,
                calibration_valid_masks,
            ) = load_sen1_probability_units(
                validation_exports[run_name], metadata_csv=args.metadata_csv
            )
            formal_manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
            resolved_protocol = BWERProtocol.from_mapping(formal_manifest["protocol"])
            panel_tables[run_name] = bundle.audit_table
            panel_protocol = resolved_protocol
            run_segmentation_uncertainty_suite(
                calibration_probabilities,
                calibration_targets,
                bundle.output_dir,
                run_dir / "uncertainty_extensions",
                protocol=resolved_protocol,
                group_columns=("event_id",),
                calibration_valid_masks=calibration_valid_masks,
                calibration_sample_ids=[
                    str(row["sample_id"]) for row in calibration_rows
                ],
                crc_alpha=args.crc_alpha,
                n_bootstrap=args.audit_bootstrap,
                seed=seed,
            )
            run_artifact_paths = {
                "checkpoint": checkpoints[run_name],
                "fit_protocol": run_dir / "fit_protocol.json",
                "fit_completion": run_dir / "fit_complete.json",
                "validation_prediction_contract": (
                    validation_exports[run_name]
                    / "prediction_completion_contract.json"
                ),
                "test_prediction_contract": (
                    test_exports[run_name]
                    / "prediction_completion_contract.json"
                ),
                "bolivia_prediction_contract": (
                    bolivia_exports[run_name]
                    / "prediction_completion_contract.json"
                ),
                "standard_test_formal_manifest": standard_bundle.manifest,
                "bolivia_holdout_formal_manifest": bolivia_bundle.manifest,
                "combined_held_out_formal_manifest": bundle.manifest,
                "evaluation_split_report": (
                    run_dir
                    / "formal_outputs"
                    / "evaluation_split_report.json"
                ),
                "operational_log": (
                    run_dir / "operational" / "performance.jsonl"
                ),
            }
            completed_run_artifacts[run_name] = {
                key: _artifact_record(Path(path), args.output_dir)
                for key, path in run_artifact_paths.items()
                if path is not None
            }
            _persist_with_log(
                run_dir,
                (
                    args.persistent_output_dir / slug / f"seed_{seed}"
                    if args.persistent_output_dir
                    else None
                ),
                label=f"{run_name}-formal",
                mode=mode,
                seed=seed,
            )
    if not calibration_valid:
        expected_run_names = {
            f"terramind_v1_base_{mode.lower().replace('+', '_plus_')}_seed_{seed}"
            for mode in MODES
            for seed in (42, 73, 101)
        }
        if set(completed_run_artifacts) != expected_run_names:
            raise RuntimeError(
                "Descriptive-only completion requires all nine TerraMind "
                "test/Bolivia exports; observed="
                f"{sorted(completed_run_artifacts)}."
            )
        completion_contract = (
            args.output_dir / "descriptive_only_completion_contract.json"
        )
        completion_payload = {
            "schema": (
                "geobwer.sen1floods11."
                "terramind_descriptive_only_panel.v1"
            ),
            "status": "descriptive_only_complete",
            "formal_evidence": False,
            "package_version": __version__,
            "code_commit": frozen_code_commit,
            "terratorch_version": terratorch_version,
            "calibration_panel_scope": CALIBRATION_PANEL_SCOPE,
            "calibration_failure_report": _artifact_record(
                calibration_failure_path, args.output_dir
            ),
            "calibration_invalid_contract": _artifact_record(
                args.output_dir / "calibration_invalid_contract.json",
                args.output_dir,
            ),
            "validation_only_calibration": True,
            "test_or_bolivia_used_for_calibration": False,
            "run_count": len(completed_run_artifacts),
            "runs": completed_run_artifacts,
            "resume_source_root": (
                str(args.resume_source_root)
                if args.resume_source_root is not None
                else None
            ),
            "inference_disabled": {
                "scale_dependent_geobwer": True,
                "bootstrap_significance": True,
                "model_panel": True,
            },
            "permitted_claim": (
                "descriptive split-level point summaries only"
            ),
        }
        if completion_contract.exists():
            previous = json.loads(
                completion_contract.read_text(encoding="utf-8")
            )
            if previous != completion_payload:
                raise RuntimeError(
                    "Existing descriptive-only completion contract drifted. "
                    "Use a new output directory."
                )
        else:
            completion_contract.write_text(
                json.dumps(completion_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        _persist_with_log(
            args.output_dir,
            args.persistent_output_dir,
            label="campaign-descriptive-only-complete",
        )
        print("SEN1_TERRAMIND_STATUS=descriptive_only_complete")
        return
    if set(modes) == set(MODES):
        if panel_protocol is None:
            raise RuntimeError("Internal error: the Sen1 model-panel protocol was not resolved.")
        run_geobwer_model_panel(
            panel_tables,
            args.output_dir / "model_panel",
            protocol=panel_protocol,
            group_column="event_id",
            cluster_column="spatial_block_id",
            comparison_pairs=tuple(
                (
                    f"terramind_v1_base_{left}_seed_{seed}",
                    f"terramind_v1_base_{right}_seed_{seed}",
                )
                for seed in seeds
                for left, right in (
                    ("s1", "s2"),
                    ("s1", "s1_plus_s2"),
                    ("s2", "s1_plus_s2"),
                )
            ),
            n_bootstrap=args.audit_bootstrap,
            seed=seeds[0],
        )
    expected_run_names = {
        f"terramind_v1_base_{mode.lower().replace('+', '_plus_')}_seed_{seed}"
        for mode in MODES
        for seed in (42, 73, 101)
    }
    if formal_run and set(completed_run_artifacts) != expected_run_names:
        raise RuntimeError(
            "Formal TerraMind completion requires exactly nine completed "
            f"model×seed runs; expected={sorted(expected_run_names)}, "
            f"observed={sorted(completed_run_artifacts)}."
        )
    model_panel_root = args.output_dir / "model_panel"
    panel_artifacts = {
        path.relative_to(model_panel_root).as_posix(): _artifact_record(
            path, args.output_dir
        )
        for path in sorted(model_panel_root.rglob("*"))
        if path.is_file()
    }
    completion_contract = args.output_dir / "campaign_completion_contract.json"
    completion_contract.write_text(
        json.dumps(
            {
                "schema": (
                    "geobwer.sen1floods11.terramind_formal_panel.v1"
                ),
                "status": "complete",
                "formal_evidence": True,
                "package_version": __version__,
                "code_commit": frozen_code_commit,
                "terratorch_version": terratorch_version,
                "science_contract": {
                    "modes": list(MODES),
                    "seeds": [42, 73, 101],
                    "batch_size": 8,
                    "max_epochs": 100,
                    "precision": "16-mixed",
                    "deterministic": True,
                    "early_stopping_enabled": False,
                    "spatial_calibration_population": "validation_only_89",
                    "standard_test_count": 90,
                    "bolivia_holdout_count": 15,
                    "combined_evaluation_count": 105,
                },
                "operational_contract": {
                    "num_workers": int(args.num_workers),
                    "persistent_workers": bool(args.persistent_workers),
                    "prefetch_factor": int(args.prefetch_factor),
                    "pin_memory": True,
                    "checkpoint_mirror_every_n_epochs": int(
                        args.checkpoint_mirror_every_n_epochs
                    ),
                    "live_training_filesystem": "/content",
                    "persistent_storage_role": (
                        "checkpoint_and_completed_artifact_mirror"
                    ),
                },
                "read_only_external_validation_inputs": {
                    "supervised_unet": audited_supervised_lineage,
                    "prithvi": prithvi_lineage,
                },
                "source_preflight": _artifact_record(
                    args.output_dir / "source_preflight.json",
                    args.output_dir,
                ),
                "common_spatial_block_calibration": _artifact_record(
                    calibration_path,
                    args.output_dir,
                ),
                "run_count": len(completed_run_artifacts),
                "runs": completed_run_artifacts,
                "model_panel_artifacts": panel_artifacts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _persist_with_log(
        args.output_dir,
        args.persistent_output_dir,
        label="campaign-complete",
    )
    print(f"[terramind:campaign] complete: {args.output_dir}")
    print("SEN1_TERRAMIND_FORMAL_CAMPAIGN=COMPLETE")


if __name__ == "__main__":
    main()
