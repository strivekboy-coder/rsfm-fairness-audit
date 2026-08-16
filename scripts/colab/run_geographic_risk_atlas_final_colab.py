"""CPU-only final atlas + preregistered association postprocess for Colab.

Run after mounting Drive and pulling the repository. No model inference,
training, or large external download is performed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = (101, 202, 303)


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    print(f"[selected] {label}: {path}", flush=True)
    return path


def _optional(path: Path, label: str) -> Path | None:
    if path.is_file():
        print(f"[selected optional] {label}: {path}", flush=True)
        return path
    print(f"[unavailable optional] {label}: {path}", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/content/rsfm-fairness-audit"))
    parser.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/rsfm_fairness_audit"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = args.project_root / "outputs" / "geobwer_final_v3"
    alpha_root = output_root / "alphaearth_geobwer_spatial_v2"
    alpha_csv = _require(alpha_root / "formal_outputs" / "formal_audit_table.csv", "AlphaEarth sample audit")
    dofa = _require(output_root / "fmow_dofav2_geo_clean_v1" / "formal_outputs" / "formal_audit_table.csv", "fMoW DOFAv2")
    resnet_root = output_root / "fmow_resnet50_common9_v1"
    resnet = [
        _require(resnet_root / f"seed_{seed}" / "formal_outputs" / "formal_audit_table.csv", f"fMoW ResNet50 seed {seed}")
        for seed in SEEDS
    ]
    reben = output_root / "reben_terramind_paired_shift_v1"
    if not reben.is_dir():
        raise FileNotFoundError(f"Missing reBEN paired root: {reben}")
    atlas_dir = output_root / "geographic_risk_atlas_v2"
    association_dir = output_root / "geographic_risk_association_v1"
    atlas_command = [
        sys.executable, str(repo / "scripts/analysis/build_geographic_risk_atlas.py"),
        "--alphaearth-root", str(alpha_root),
        "--fmow-csv", f"DOFAv2={dofa}", "--fmow-seed-count", "DOFAv2=1",
    ]
    for path in resnet:
        atlas_command += ["--fmow-csv", f"ResNet50={path}"]
    atlas_command += [
        "--fmow-seed-count", "ResNet50=3", "--reben-paired-dir", str(reben),
        "--output-dir", str(atlas_dir),
    ]
    print("[atlas]", " ".join(map(str, atlas_command)), flush=True)
    subprocess.run(atlas_command, cwd=repo, check=True)

    covariate_root = args.project_root / "covariates" / "geographic_risk_v1"
    alpha_external = _optional(covariate_root / "alphaearth_covariates.csv", "AlphaEarth GHSL/population/nightlights")
    fmow_external = _optional(covariate_root / "fmow_covariates.csv", "fMoW GHSL/population/nightlights")
    association_command = [
        sys.executable, str(repo / "scripts/analysis/build_geographic_risk_association.py"),
        "--atlas-dir", str(atlas_dir), "--output-dir", str(association_dir),
        "--alphaearth-sample-csv", str(alpha_csv),
        "--fmow-sample-csv", f"DOFAv2={dofa}",
        "--fmow-sample-csv", f"ResNet50={resnet[0]}",
    ]
    if alpha_external:
        association_command += ["--alphaearth-external-csv", str(alpha_external)]
    if fmow_external:
        association_command += ["--fmow-external-csv", f"DOFAv2={fmow_external}",
                                "--fmow-external-csv", f"ResNet50={fmow_external}"]
    print("[association]", " ".join(map(str, association_command)), flush=True)
    subprocess.run(association_command, cwd=repo, check=True)
    print(f"Atlas complete: {atlas_dir}")
    print(f"Association complete: {association_dir}")
    print("GPU required: no")


if __name__ == "__main__":
    main()
