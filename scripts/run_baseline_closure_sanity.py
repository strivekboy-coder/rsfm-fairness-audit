from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from rsfm_fairness_audit.band_profiles import get_band_profile


FINAL_DATASET_ZIP_NAME = "fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged.zip"
FINAL_RESNET_ZIP_NAME = "resnet50_30k_location_disjoint_patched_metadata.zip"
FINAL_DOFA_ZIP_NAME = "dofa_scaled10000_30k_location_disjoint.zip"
FINAL_COMPARISON_ZIP_NAME = "comparison_resnet50_vs_dofa_scaled10000.zip"
FINAL_BUNDLE_ZIP_NAME = "fmow_step3_final_bundle_30k_location_disjoint.zip"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _status(name: str, status: str, evidence: str, note: str) -> dict[str, str]:
    return {"check": name, "status": status, "evidence": evidence, "note": note}


def _check_zip(path: Path | None, expected_name: str, label: str, required_members: tuple[str, ...] = ()) -> dict[str, str]:
    if path is None:
        return _status(label, "awaiting_colab_run", "", f"Pass --{label.replace('_', '-')} to verify {expected_name}.")
    if path.name != expected_name:
        return _status(label, "fail", str(path), f"Expected final artifact name {expected_name}, got {path.name}.")
    if not path.exists():
        return _status(label, "fail", str(path), "Path does not exist.")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            missing = [member for member in required_members if not any(name.endswith(member) for name in names)]
    except zipfile.BadZipFile:
        return _status(label, "fail", str(path), "Not a readable zip file.")
    if missing:
        return _status(label, "fail", str(path), f"Missing expected members ending with: {missing}")
    return _status(label, "pass", str(path), f"Verified final zip name and readability; members={len(names)}")


def _location_leakage_check(manifest: Path | None) -> dict[str, str]:
    if manifest is None or not manifest.exists():
        return _status(
            "location_disjoint_leakage",
            "recorded_done",
            "docs/experiments/scientific_findings.md; docs/datasets/fmow_sentinel.md",
            "Formal record states category + location_id train/val overlap is zero. Provide --manifest to recompute from artifact.",
        )
    rows = _read_csv(manifest)
    train = {
        (row.get("category", ""), row.get("location_id", ""))
        for row in rows
        if row.get("split") == "train"
    }
    val = {
        (row.get("category", ""), row.get("location_id", ""))
        for row in rows
        if row.get("split") == "val"
    }
    overlap = train & val
    return _status(
        "location_disjoint_leakage",
        "pass" if not overlap else "fail",
        str(manifest),
        f"train_groups={len(train)} val_groups={len(val)} overlap={len(overlap)}",
    )


def _class_mapping_check(run_dir: Path | None) -> dict[str, str]:
    if run_dir is None:
        return _status(
            "class_mapping",
            "recorded_done",
            "docs/experiments/fmow_step3_scientific_findings.md",
            "Formal record states ResNet metadata was patched and DOFA run metadata records class mapping. Provide --run-dir to recompute.",
        )
    metadata_path = run_dir / "run_metadata.json"
    audit_path = run_dir / "audit_table.csv"
    if not metadata_path.exists() or not audit_path.exists():
        return _status("class_mapping", "awaiting_colab_run", str(run_dir), "run_metadata.json or audit_table.csv is missing.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = _read_csv(audit_path)
    labels = {row.get("label") or row.get("category") or row.get("class_label") for row in rows}
    labels.discard("")
    mapping = metadata.get("class_mapping", {})
    ok = isinstance(mapping, dict) and len(mapping) >= len(labels) and len(labels) > 0
    return _status(
        "class_mapping",
        "pass" if ok else "fail",
        f"{metadata_path}; {audit_path}",
        f"audit_labels={len(labels)} class_mapping_entries={len(mapping) if isinstance(mapping, dict) else 'not_dict'}",
    )


def _embedding_cache_check(cache_dir: Path | None) -> list[dict[str, str]]:
    if cache_dir is None or not cache_dir.exists():
        return [
            _status(
                "dofa_embedding_cache_input_scale",
                "code_done",
                "src/rsfm_fairness_audit/fmow_sentinel_classification.py",
                "The DOFA embedding cache key includes input_scale. Provide --embedding-cache-dir to inspect completed cache metadata.",
            ),
            _status(
                "dofa_embedding_numeric",
                "recorded_done",
                "docs/experiments/fmow_step3_scientific_findings.md",
                "Formal record states scaled DOFA embeddings had no NaN/Inf and non-collapsed variance. Provide --embedding-cache-dir to recompute.",
            ),
        ]
    metadata_files = sorted(cache_dir.glob("dofa_*.json"))
    npz_files = sorted(cache_dir.glob("dofa_*.npz"))
    scales = []
    for path in metadata_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scales.append(payload.get("input_scale"))
        except json.JSONDecodeError:
            scales.append("invalid_json")
    scale_ok = bool(scales) and all(str(value) in {"10000", "10000.0"} for value in scales)

    numeric_notes: list[str] = []
    numeric_ok = bool(npz_files)
    for path in npz_files[:4]:
        data = np.load(path, allow_pickle=False)
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        finite = bool(np.isfinite(emb).all())
        std = float(np.std(emb)) if emb.size else float("nan")
        numeric_ok = numeric_ok and finite and not math.isclose(std, 0.0, abs_tol=1e-12)
        numeric_notes.append(f"{path.name}:shape={tuple(emb.shape)} finite={finite} std={std:.6g}")

    return [
        _status(
            "dofa_embedding_cache_input_scale",
            "pass" if scale_ok else "fail",
            str(cache_dir),
            f"metadata_files={len(metadata_files)} input_scales={scales}",
        ),
        _status(
            "dofa_embedding_numeric",
            "pass" if numeric_ok else "fail",
            str(cache_dir),
            "; ".join(numeric_notes) if numeric_notes else "No DOFA npz embedding files found.",
        ),
    ]


def _band_profile_check() -> dict[str, str]:
    profile = get_band_profile("sentinel2_13band_fmow")
    names = profile.get("band_names", [])
    wavelengths = profile.get("wavelength_list", [])
    ok = len(names) == len(wavelengths) == 13
    return _status(
        "fmow_band_wavelength_profile",
        "pass" if ok else "fail",
        "src/rsfm_fairness_audit/band_profiles.py",
        f"bands={','.join(names)} wavelengths={','.join(str(v) for v in wavelengths)}",
    )


def _raster_stats_check(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return _status(
            "fmow_per_band_statistics",
            "partial",
            "",
            "Workflow supports band_statistics_sample.csv, but no final stats file was supplied to this local check.",
        )
    rows = _read_csv(path)
    return _status(
        "fmow_per_band_statistics",
        "pass" if rows else "fail",
        str(path),
        f"rows={len(rows)}",
    )


def _formal_open_checks() -> list[dict[str, str]]:
    return [
        _status(
            "dofa_pooling_ablation",
            "awaiting_colab_run",
            "docs/experiments/fmow_step3_scientific_findings.md",
            "Current completed run uses flattened forward_features. CLS/mean pooling are not exposed as formal completed outputs.",
        ),
        _status(
            "tiny_overfit",
            "awaiting_colab_run",
            "",
            "Diagnostic only. No local data run was performed and no formal model result should depend on it.",
        ),
        _status(
            "random_split_sanity",
            "awaiting_colab_run",
            "",
            "Diagnostic only. Do not present random-split fMoW as a main result.",
        ),
    ]


def _write_report(checks: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_sanity_checks.json").write_text(json.dumps({"checks": checks}, indent=2), encoding="utf-8")
    lines = [
        "# Baseline Closure Sanity Checks",
        "",
        "This file is generated by `scripts/run_baseline_closure_sanity.py`. It does not train models, run inference, or modify BWER.",
        "",
        "| check | status | evidence | note |",
        "| --- | --- | --- | --- |",
    ]
    for row in checks:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['evidence']} | {row['note'].replace('|', '/')} |"
        )
    lines.append("")
    lines.append("Use this as a closure-readiness record, not as a scientific result table.")
    (output_dir / "baseline_sanity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate lightweight baseline-closure sanity records.")
    parser.add_argument("--prepared-dataset-zip", type=Path, help="Final prepared fMoW dataset zip.")
    parser.add_argument("--resnet-artifact-zip", type=Path, help="Final patched ResNet artifact zip.")
    parser.add_argument("--dofa-artifact-zip", type=Path, help="Final scaled DOFA artifact zip.")
    parser.add_argument("--comparison-artifact-zip", type=Path, help="Final ResNet-vs-DOFA comparison artifact zip.")
    parser.add_argument("--final-bundle-zip", type=Path, help="Final fMoW Step 3 bundle zip.")
    parser.add_argument("--manifest", type=Path, help="Optional final fMoW clean subset manifest.")
    parser.add_argument("--run-dir", type=Path, help="Optional completed fMoW run directory with audit_table.csv and run_metadata.json.")
    parser.add_argument("--embedding-cache-dir", type=Path, help="Optional completed DOFA embedding_cache directory.")
    parser.add_argument("--band-statistics", type=Path, help="Optional final band_statistics_sample.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results/baseline_closure_sanity"))
    args = parser.parse_args()

    checks: list[dict[str, str]] = [
        _check_zip(args.prepared_dataset_zip, FINAL_DATASET_ZIP_NAME, "prepared_dataset_zip", ("final_clean_subset_manifest",)),
        _check_zip(args.resnet_artifact_zip, FINAL_RESNET_ZIP_NAME, "resnet_artifact_zip", ("run_metadata.json", "audit_table.csv")),
        _check_zip(args.dofa_artifact_zip, FINAL_DOFA_ZIP_NAME, "dofa_artifact_zip", ("run_metadata.json", "audit_table.csv")),
        _check_zip(args.comparison_artifact_zip, FINAL_COMPARISON_ZIP_NAME, "comparison_artifact_zip", ("comparison_summary.csv",)),
        _check_zip(args.final_bundle_zip, FINAL_BUNDLE_ZIP_NAME, "final_bundle_zip", (FINAL_RESNET_ZIP_NAME, FINAL_DOFA_ZIP_NAME, FINAL_COMPARISON_ZIP_NAME)),
        _band_profile_check(),
        _location_leakage_check(args.manifest),
        _class_mapping_check(args.run_dir),
        _raster_stats_check(args.band_statistics),
    ]
    checks.extend(_embedding_cache_check(args.embedding_cache_dir))
    checks.extend(_formal_open_checks())
    _write_report(checks, args.output_dir)
    print(f"Wrote baseline sanity report: {args.output_dir / 'baseline_sanity_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
