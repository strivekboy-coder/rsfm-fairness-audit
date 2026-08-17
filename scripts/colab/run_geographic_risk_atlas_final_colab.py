"""CPU-only final atlas + preregistered association postprocess for Colab.

Run after mounting Drive and pulling the repository. No model inference,
training, or large external download is performed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    print(f"[selected] {label}: {path}", flush=True)
    return path


def _require_result_root(candidates: list[Path], label: str, filename: str) -> Path:
    for root in candidates:
        if (root / filename).is_file():
            print(f"[selected] {label}: {root}", flush=True)
            return root
    raise FileNotFoundError(
        f"Missing {label} ({filename}); checked: {', '.join(map(str, candidates))}"
    )


def _mark_invalid_predecessors(paths: list[Path], replacement: dict[str, str]) -> list[str]:
    marked = []
    payload = {
        "status": "invalid_for_fmow_spatial_science",
        "scope": "fMoW spatial atlas, covariate alignment, maps, and associations only",
        "reason": (
            "location_id and category|location_id merge distinct original split sequences; "
            "canonical lat/lon were produced by the same contaminated aggregation"
        ),
        "invalid_unit_contracts": ["location_id", "category|location_id", "canonical_lat_lon"],
        "replacement_unit_contract": "split_original|category|location_id equivalent to raw archive parent",
        "alphaearth_and_reben_status": "unaffected_by_this_invalidation",
        "replacement": replacement,
    }
    for path in paths:
        if not path.is_dir():
            continue
        marker = path / "INVALID_FMOW_GEOGRAPHIC_IDENTITY.json"
        marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        marked.append(str(marker))
        print(f"[invalidated predecessor] {marker}", flush=True)
    return marked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/content/rsfm-fairness-audit"))
    parser.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/rsfm_fairness_audit"))
    parser.add_argument("--ee-project", default="rsfm-fairness-audit")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--n-boot", type=int, default=500)
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
    dofa_root = output_root / "fmow_dofav2_geo_clean_v1"
    dofa, dofa_discovery = discover_canonical_fmow_seed_tables(
        dofa_root / "probe_seeds", architecture=None, model_prefix="dofav2",
    )
    resnet_root = output_root / "fmow_resnet50_common9_v1"
    resnet, resnet_discovery = discover_canonical_fmow_seed_tables(resnet_root)
    for model, paths, discovery in (
        ("DOFAv2", dofa, dofa_discovery), ("ResNet50", resnet, resnet_discovery),
    ):
        print(
            f"[discovered] fMoW {model} canonical seeds: "
            + ", ".join(map(str, discovery["seeds"])), flush=True,
        )
        print(f"[scientific role] {model}: {discovery['scientific_role']}", flush=True)
        for path in paths:
            print(f"[selected] fMoW {model} canonical formal table: {path}", flush=True)
        for rejected in discovery["rejected_seed_directories"]:
            print(f"[rejected incomplete/noncanonical {model} seed] {rejected}", flush=True)
    reben = output_root / "reben_terramind_paired_shift_v1"
    if not reben.is_dir():
        raise FileNotFoundError(f"Missing reBEN paired root: {reben}")
    croma = _require_result_root([
        args.project_root / "outputs" / "reben_croma_paired_shift_v1",
        output_root / "reben_croma_paired_shift_v1",
    ], "reBEN CROMA paired result", "paired_shift_country_deltas.csv")
    _require_result_root([reben], "reBEN TerraMind paired result", "paired_shift_country_deltas.csv")
    atlas_dir = output_root / "geographic_risk_atlas_v4_original_sequence"
    covariate_dir = args.project_root / "covariates" / "geographic_risk_v3_original_sequence"
    cache_dir = args.project_root / "cache" / "geographic_risk_covariates_v3_original_sequence"
    association_dir = output_root / "geographic_risk_association_v3_original_sequence"
    atlas_command = [
        sys.executable, str(repo / "scripts/analysis/build_geographic_risk_atlas.py"),
        "--alphaearth-root", str(alpha_root),
    ]
    for path in dofa:
        atlas_command += ["--fmow-csv", f"DOFAv2={path}"]
    for path in resnet:
        atlas_command += ["--fmow-csv", f"ResNet50={path}"]
    atlas_command += [
        "--fmow-seed-count", f"DOFAv2={len(dofa)}",
        "--fmow-seed-count", f"ResNet50={len(resnet)}", "--reben-paired-dir", str(reben),
        "--reben-model-paired-dir", f"TerraMind={reben}",
        "--reben-model-paired-dir", f"CROMA={croma}",
        "--output-dir", str(atlas_dir),
    ]
    print("[atlas]", " ".join(map(str, atlas_command)), flush=True)
    subprocess.run(atlas_command, cwd=repo, env=env, check=True)

    prepare_command = [
        sys.executable, str(repo / "scripts/analysis/prepare_geographic_risk_covariates.py"),
        "--atlas-dir", str(atlas_dir), "--alphaearth-sample-csv", str(alpha_csv),
        "--output-dir", str(covariate_dir), "--cache-dir", str(cache_dir),
        "--ee-project", args.ee_project, "--batch-size", str(args.batch_size),
    ]
    print("[covariates]", " ".join(map(str, prepare_command)), flush=True)
    subprocess.run(prepare_command, cwd=repo, env=env, check=True)
    alpha_external = _require(covariate_dir / "alphaearth_covariates.csv", "AlphaEarth canonical covariates")
    fmow_external = _require(covariate_dir / "fmow_covariates.csv", "fMoW original-sequence canonical covariates")
    association_command = [
        sys.executable, str(repo / "scripts/analysis/build_geographic_risk_association.py"),
        "--atlas-dir", str(atlas_dir), "--output-dir", str(association_dir),
        "--alphaearth-sample-csv", str(alpha_csv),
        "--fmow-sample-csv", f"DOFAv2={dofa[0]}",
        "--fmow-sample-csv", f"ResNet50={next((path for path in resnet if 'seed_101' in path.parts), resnet[0])}",
        "--n-boot", str(args.n_boot),
    ]
    if alpha_external:
        association_command += ["--alphaearth-external-csv", str(alpha_external)]
    if fmow_external:
        association_command += ["--fmow-external-csv", f"DOFAv2={fmow_external}",
                                "--fmow-external-csv", f"ResNet50={fmow_external}"]
    print("[association]", " ".join(map(str, association_command)), flush=True)
    subprocess.run(association_command, cwd=repo, env=env, check=True)
    replacement = {
        "atlas": str(atlas_dir), "covariates": str(covariate_dir),
        "association": str(association_dir),
    }
    predecessors = sorted({
        *(
            path for path in output_root.glob("geographic_risk_atlas_v*")
            if path != atlas_dir
        ),
        *(
            path for path in output_root.glob("geographic_risk_association_v*")
            if path != association_dir
        ),
        *(
            path for path in (args.project_root / "covariates").glob("geographic_risk_v*")
            if path != covariate_dir
        ),
        *(
            path for path in (args.project_root / "cache").glob("geographic_risk_covariates_v*")
            if path != cache_dir
        ),
    }, key=str)
    invalidated = _mark_invalid_predecessors(predecessors, replacement)
    (atlas_dir / "invalidated_predecessors.json").write_text(
        json.dumps({"status": "complete", "markers": invalidated, **replacement}, indent=2),
        encoding="utf-8",
    )
    print(f"Atlas complete: {atlas_dir}")
    print(f"Covariates complete: {covariate_dir}")
    print(f"Association complete: {association_dir}")
    print("GPU required: no")


if __name__ == "__main__":
    main()
