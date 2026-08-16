"""Prepare fixed official covariates and rerun the CPU-only association stage.

This runner does not rebuild the atlas, train a model, or start experiments #8/#9.
Earth Engine performs public-raster point sampling; all joins, QA, statistics,
bootstraps, and figures run on CPU.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    print(f"[selected] {label}: {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/content/rsfm-fairness-audit"))
    parser.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/rsfm_fairness_audit"))
    parser.add_argument("--ee-project", default="rsfm-fairness-audit")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()
    repo = args.repo.resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    output_root = args.project_root / "outputs" / "geobwer_final_v3"
    atlas = output_root / "geographic_risk_atlas_v2_1"
    alpha_samples = _require(
        output_root / "alphaearth_geobwer_spatial_v2" / "formal_outputs" / "formal_audit_table.csv",
        "AlphaEarth canonical formal audit",
    )
    _require(atlas / "alphaearth_spatial_unit_risk.csv", "AlphaEarth atlas risk")
    fmow_risks = sorted(atlas.glob("fmow_*_spatial_unit_risk.csv"))
    if not fmow_risks:
        raise FileNotFoundError(f"Missing fMoW atlas risk tables: {atlas}")
    for path in fmow_risks:
        print(f"[selected] fMoW atlas risk: {path}", flush=True)

    covariates = args.project_root / "covariates" / "geographic_risk_v1_1"
    cache = args.project_root / "cache" / "geographic_risk_covariates_v1_1"
    association = output_root / "geographic_risk_association_v1_3"
    prepare = [
        sys.executable, str(repo / "scripts/analysis/prepare_geographic_risk_covariates.py"),
        "--atlas-dir", str(atlas), "--alphaearth-sample-csv", str(alpha_samples),
        "--output-dir", str(covariates), "--cache-dir", str(cache),
        "--ee-project", args.ee_project, "--batch-size", str(args.batch_size),
    ]
    print("[prepare]", " ".join(map(str, prepare)), flush=True)
    subprocess.run(prepare, cwd=repo, env=env, check=True)

    alpha_covariates = _require(covariates / "alphaearth_covariates.csv", "AlphaEarth canonical covariates")
    fmow_covariates = _require(covariates / "fmow_covariates.csv", "fMoW canonical covariates")
    command = [
        sys.executable, str(repo / "scripts/analysis/build_geographic_risk_association.py"),
        "--atlas-dir", str(atlas), "--output-dir", str(association),
        "--alphaearth-sample-csv", str(alpha_samples),
        "--alphaearth-external-csv", str(alpha_covariates),
        "--n-boot", str(args.n_boot),
    ]
    for risk_path in fmow_risks:
        model = risk_path.name.removeprefix("fmow_").removesuffix("_spatial_unit_risk.csv")
        command += ["--fmow-external-csv", f"{model}={fmow_covariates}"]
    print("[association]", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=repo, env=env, check=True)
    print(f"Covariates complete: {covariates}")
    print(f"Association complete: {association}")
    print("GPU required: no")
    print("Experiments #8/#9: not started")


if __name__ == "__main__":
    main()
