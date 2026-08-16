"""CPU-only final atlas + preregistered association postprocess for Colab.

Run after mounting Drive and pulling the repository. No model inference,
training, or large external download is performed.
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


def _optional(path: Path, label: str) -> Path | None:
    if path.is_file():
        print(f"[selected optional] {label}: {path}", flush=True)
        return path
    print(f"[unavailable optional] {label}: {path}", flush=True)
    return None


def _require_result_root(candidates: list[Path], label: str, filename: str) -> Path:
    for root in candidates:
        if (root / filename).is_file():
            print(f"[selected] {label}: {root}", flush=True)
            return root
    raise FileNotFoundError(
        f"Missing {label} ({filename}); checked: {', '.join(map(str, candidates))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/content/rsfm-fairness-audit"))
    parser.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/rsfm_fairness_audit"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from rsfm_fairness_audit.geographic_risk_atlas import (
        discover_canonical_fmow_seed_tables,
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    output_root = args.project_root / "outputs" / "geobwer_final_v3"
    alpha_root = output_root / "alphaearth_geobwer_spatial_v2"
    alpha_csv = _require(alpha_root / "formal_outputs" / "formal_audit_table.csv", "AlphaEarth sample audit")
    dofa = _require(output_root / "fmow_dofav2_geo_clean_v1" / "formal_outputs" / "formal_audit_table.csv", "fMoW DOFAv2")
    resnet_root = output_root / "fmow_resnet50_common9_v1"
    resnet, resnet_discovery = discover_canonical_fmow_seed_tables(resnet_root)
    print(
        "[discovered] fMoW ResNet50 canonical seeds: "
        + ", ".join(map(str, resnet_discovery["seeds"])),
        flush=True,
    )
    print(f"[scientific role] {resnet_discovery['scientific_role']}", flush=True)
    for path in resnet:
        print(f"[selected] fMoW ResNet50 canonical formal table: {path}", flush=True)
    for rejected in resnet_discovery["rejected_seed_directories"]:
        print(f"[rejected incomplete/noncanonical seed] {rejected}", flush=True)
    reben = output_root / "reben_terramind_paired_shift_v1"
    if not reben.is_dir():
        raise FileNotFoundError(f"Missing reBEN paired root: {reben}")
    croma = _require_result_root([
        args.project_root / "outputs" / "reben_croma_paired_shift_v1",
        output_root / "reben_croma_paired_shift_v1",
    ], "reBEN CROMA paired result", "paired_shift_country_deltas.csv")
    _require_result_root([reben], "reBEN TerraMind paired result", "paired_shift_country_deltas.csv")
    atlas_dir = output_root / "geographic_risk_atlas_v2_1"
    association_dir = output_root / "geographic_risk_association_v1_1"
    atlas_command = [
        sys.executable, str(repo / "scripts/analysis/build_geographic_risk_atlas.py"),
        "--alphaearth-root", str(alpha_root),
        "--fmow-csv", f"DOFAv2={dofa}", "--fmow-seed-count", "DOFAv2=1",
    ]
    for path in resnet:
        atlas_command += ["--fmow-csv", f"ResNet50={path}"]
    atlas_command += [
        "--fmow-seed-count", f"ResNet50={len(resnet)}", "--reben-paired-dir", str(reben),
        "--reben-model-paired-dir", f"TerraMind={reben}",
        "--reben-model-paired-dir", f"CROMA={croma}",
        "--output-dir", str(atlas_dir),
    ]
    print("[atlas]", " ".join(map(str, atlas_command)), flush=True)
    subprocess.run(atlas_command, cwd=repo, env=env, check=True)

    covariate_root = args.project_root / "covariates" / "geographic_risk_v1"
    alpha_external = _optional(covariate_root / "alphaearth_covariates.csv", "AlphaEarth GHSL/population/nightlights")
    fmow_external = _optional(covariate_root / "fmow_covariates.csv", "fMoW GHSL/population/nightlights")
    association_command = [
        sys.executable, str(repo / "scripts/analysis/build_geographic_risk_association.py"),
        "--atlas-dir", str(atlas_dir), "--output-dir", str(association_dir),
        "--alphaearth-sample-csv", str(alpha_csv),
        "--fmow-sample-csv", f"DOFAv2={dofa}",
        "--fmow-sample-csv", f"ResNet50={next((path for path in resnet if 'seed_101' in path.parts), resnet[0])}",
    ]
    if alpha_external:
        association_command += ["--alphaearth-external-csv", str(alpha_external)]
    if fmow_external:
        association_command += ["--fmow-external-csv", f"DOFAv2={fmow_external}",
                                "--fmow-external-csv", f"ResNet50={fmow_external}"]
    print("[association]", " ".join(map(str, association_command)), flush=True)
    subprocess.run(association_command, cwd=repo, env=env, check=True)
    print(f"Atlas complete: {atlas_dir}")
    print(f"Association complete: {association_dir}")
    print("GPU required: no")


if __name__ == "__main__":
    main()
