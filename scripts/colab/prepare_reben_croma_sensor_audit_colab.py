from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir  # noqa: E402
from rsfm_fairness_audit.adapters.reben import (  # noqa: E402
    check_reben_configilm_dependency_chain,
    detect_lmdb_payload_format,
    resolve_reben_root_dir,
    run_configilm_reben_preflight,
)


OFFICIAL_SOURCES = {
    "bigearthnet_home": "https://bigearth.net/",
    "bigearthnet_v2_pdf": "https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf",
    "bigearthnet_v2_zenodo": "https://zenodo.org/records/10891137",
    "configilm_reben_docs": "https://lhackel-tub.github.io/ConfigILM/extra/DataSets%20and%20DataModules/bigearthnetv2.html",
    "configilm_reben_api": "https://lhackel-tub.github.io/ConfigILM/API/ds/api_ds_BENv2.html",
    "bifold_hf": "https://huggingface.co/BIFOLD-BigEarthNetv2-0",
    "croma_repo": "https://github.com/antofuller/CROMA",
    "croma_hf": "https://huggingface.co/antofuller/CROMA",
}

ZENODO_METADATA_URLS = {
    "metadata.parquet": "https://zenodo.org/records/10891137/files/metadata.parquet?download=1",
    "metadata_for_patches_with_snow_cloud_or_shadow.parquet": (
        "https://zenodo.org/records/10891137/files/metadata_for_patches_with_snow_cloud_or_shadow.parquet?download=1"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or verify official resources for the BigEarthNet v2.0 / reBEN + CROMA "
            "sensor-mode audit Colab workflow. This script does not use BigEarthNet v1, "
            "lc-col BigEarthNet, unofficial LMDB mirrors, or torchvision substitutes."
        )
    )
    parser.add_argument("--reben-root", type=Path, default=Path("/content/data/reben"))
    parser.add_argument("--lmdb-root", type=Path, default=Path("/content/data/reben/BigEarthNetEncoded.lmdb"))
    parser.add_argument("--metadata-parquet", type=Path, default=Path("/content/data/reben/metadata.parquet"))
    parser.add_argument(
        "--metadata-snow-cloud-parquet",
        type=Path,
        default=Path("/content/data/reben/metadata_for_patches_with_snow_cloud_or_shadow.parquet"),
    )
    parser.add_argument("--croma-repo", type=Path, default=Path("/content/CROMA"))
    parser.add_argument("--croma-checkpoint", type=Path, default=Path("/content/checkpoints/CROMA_base.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/outputs/reben_croma_sensor_mode_audit_prepare"))
    parser.add_argument("--no-download-metadata", action="store_true", help="Only verify metadata parquet files; do not download them from Zenodo.")
    parser.add_argument("--no-download-croma", action="store_true", help="Only verify CROMA repo/checkpoint; do not clone/download.")
    parser.add_argument("--allow-git-pull", action="store_true", help="Run git pull when --croma-repo already exists.")
    return parser


def _run(command: list[str], *, cwd: Path | None = None) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return False, str(exc)
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return result.returncode == 0, output


def _download_url(url: str, path: Path) -> tuple[bool, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        return False, str(exc)
    return True, f"downloaded {url} -> {path}"


def _clone_or_update_croma(repo_path: Path, *, allow_download: bool, allow_git_pull: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(repo_path),
        "use_croma_path": str(repo_path / "use_croma.py"),
        "exists_before": repo_path.exists(),
        "action": "verify_only" if not allow_download else "clone_or_verify",
        "status": "missing",
        "message": "",
    }
    if repo_path.exists():
        if allow_git_pull:
            ok, message = _run(["git", "pull", "--ff-only"], cwd=repo_path)
            result["message"] = message
            result["git_pull_ok"] = ok
        result["status"] = "ok" if (repo_path / "use_croma.py").exists() else "missing_use_croma.py"
        return result
    if not allow_download:
        result["message"] = "CROMA repo missing and downloads disabled."
        return result
    ok, message = _run(["git", "clone", OFFICIAL_SOURCES["croma_repo"], str(repo_path)])
    result["message"] = message
    result["status"] = "ok" if ok and (repo_path / "use_croma.py").exists() else "clone_failed"
    return result


def _download_croma_checkpoint(checkpoint_path: Path, *, allow_download: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(checkpoint_path),
        "exists_before": checkpoint_path.exists(),
        "source": OFFICIAL_SOURCES["croma_hf"],
        "status": "ok" if checkpoint_path.exists() else "missing",
        "message": "",
    }
    if checkpoint_path.exists() or not allow_download:
        if not checkpoint_path.exists():
            result["message"] = "CROMA checkpoint missing and downloads disabled."
        return result
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="antofuller/CROMA",
            filename="CROMA_base.pt",
            local_dir=str(checkpoint_path.parent),
            local_dir_use_symlinks=False,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != checkpoint_path.resolve():
            shutil.copy2(downloaded_path, checkpoint_path)
        result["status"] = "ok" if checkpoint_path.exists() else "download_failed"
        result["message"] = f"huggingface_hub download: {downloaded}"
        return result
    except Exception as exc:
        result["huggingface_hub_error"] = str(exc)
    ok, message = _download_url("https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt", checkpoint_path)
    result["status"] = "ok" if ok and checkpoint_path.exists() else "download_failed"
    result["message"] = message
    return result


def _prepare_metadata(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    targets = {
        "metadata.parquet": args.metadata_parquet,
        "metadata_for_patches_with_snow_cloud_or_shadow.parquet": args.metadata_snow_cloud_parquet,
    }
    for filename, path in targets.items():
        row: dict[str, object] = {
            "name": filename,
            "path": str(path),
            "official_url": ZENODO_METADATA_URLS[filename],
            "exists_before": path.exists(),
            "status": "ok" if path.exists() else "missing",
            "message": "",
        }
        if not path.exists() and not args.no_download_metadata:
            ok, message = _download_url(ZENODO_METADATA_URLS[filename], path)
            row["message"] = message
            row["status"] = "ok" if ok and path.exists() else "download_failed"
        elif not path.exists():
            row["message"] = "Metadata missing and downloads disabled."
        row["size_bytes"] = path.stat().st_size if path.exists() else 0
        rows.append(row)
    return rows


def _disk_usage(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        anchor = path if path.exists() else path.parent
        try:
            usage = shutil.disk_usage(anchor)
        except Exception as exc:
            rows.append({"path": str(path), "status": "unavailable", "message": str(exc)})
            continue
        rows.append(
            {
                "path": str(path),
                "status": "ok",
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
        )
    return rows


def _write_blocked_report(path: Path, blocking: list[str], args: argparse.Namespace) -> None:
    lines = [
        "# reBEN / CROMA Preparation Blocked Report",
        "",
        "Preparation could not verify all required resources for the Step 1 smoke/full runner.",
        "",
        "## Blocking Checks",
        "",
    ]
    lines.extend([f"- {item}" for item in blocking])
    lines.extend(
        [
            "",
            "## Manual Data Instructions",
            "",
            "Use BigEarthNet v2.0 / reBEN only. Do not use BigEarthNet v1 or lc-col BigEarthNet.",
            "",
            "Official BigEarthNet v2.0 data are published through the BigEarthNet site / Zenodo record:",
            f"- {OFFICIAL_SOURCES['bigearthnet_home']}",
            f"- {OFFICIAL_SOURCES['bigearthnet_v2_zenodo']}",
            "",
            "The ConfigILM reBEN loader expects an LMDB directory plus two parquet metadata files:",
            f"- images_lmdb: `{args.lmdb_root}`",
            f"- metadata_parquet: `{args.metadata_parquet}`",
            f"- metadata_snow_cloud_parquet: `{args.metadata_snow_cloud_parquet}`",
            "",
            "ConfigILM documentation states that LMDB files can be requested from the authors or produced by downloading the official BigEarthNet v2 S1/S2 archives and encoding them with the BigEarthNet Encoder / rico-HDL workflow.",
            "This preparation script does not use the unofficial community Hugging Face LMDB mirror as a formal source.",
            "",
            "After placing the LMDB at the requested path, rerun this preparation script before smoke/full execution.",
            "",
            "If your downloaded bundle contains `/content/data/reben/BigEarthNetEncoded.lmdb/BigEarthNetEncoded.lmdb`, pass the outer folder as `--lmdb-root /content/data/reben/BigEarthNetEncoded.lmdb`; the preflight will resolve that as ConfigILM `root_dir`.",
            "",
            "## Dependency Compatibility Command",
            "",
            "Run this if the blocked report or preparation JSON shows `fastcore.dispatch`, `bigearthnet_common`, `bigearthnet_patch_interface`, or `configilm` import failures:",
            "",
            "```bash",
            "pip install -U --no-deps configilm bigearthnet_patch_interface bigearthnet_common",
            "pip install --force-reinstall 'fastcore==1.5.29'",
            "```",
            "",
            "Do not reinstall torch/CUDA for this compatibility fix.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preparation_report(path: Path, payload: dict[str, object], missing: list[str]) -> None:
    lines = [
        "# reBEN / CROMA Sensor-Mode Audit Preparation Report",
        "",
        f"Created: {payload['created_utc']}",
        "",
        "## Official Sources",
        "",
    ]
    for key, value in OFFICIAL_SOURCES.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Resource Status", ""])
    for item in payload["resources"]:  # type: ignore[index]
        lines.append(f"- {item['name']}: {item['status']} (`{item['path']}`)")
    dependency = payload.get("dependency_check", {})
    if isinstance(dependency, dict):
        lines.extend(["", "## Dependency Import Check", ""])
        for item in dependency.get("checks", []):
            lines.append(f"- {item.get('module', '')}: {item.get('status', '')} {item.get('version', '')} {item.get('message', '')}".strip())
        if dependency.get("status") != "ok":
            lines.extend(["", "Suggested compatibility command:", "", "```bash", str(dependency.get("install_command", "")), "```"])
    lines.extend(["", "## Disk Usage", ""])
    for item in payload["disk_usage"]:  # type: ignore[index]
        if item.get("status") == "ok":
            lines.append(f"- `{item['path']}`: free {item['free_gb']} GB / total {item['total_gb']} GB")
        else:
            lines.append(f"- `{item['path']}`: unavailable ({item.get('message', '')})")
    lines.extend(["", "## Status", ""])
    if missing:
        lines.append("Status: blocked. See `blocked_report.md`.")
    else:
        lines.append("Status: ready for smoke/full runner.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    out = ensure_dir(args.output_dir)
    ensure_dir(args.reben_root)

    croma_repo = _clone_or_update_croma(args.croma_repo, allow_download=not args.no_download_croma, allow_git_pull=args.allow_git_pull)
    croma_checkpoint = _download_croma_checkpoint(args.croma_checkpoint, allow_download=not args.no_download_croma)
    metadata_rows = _prepare_metadata(args)
    dependency_check = check_reben_configilm_dependency_chain()
    for item in dependency_check.get("checks", []):
        print(f"[reben:deps] {item.get('module', '')} status={item.get('status', '')} version={item.get('version', '')} {item.get('message', '')}".strip())
    configilm_preflight = run_configilm_reben_preflight(
        images_lmdb=args.lmdb_root,
        metadata_parquet=args.metadata_parquet,
        metadata_snow_cloud_parquet=args.metadata_snow_cloud_parquet,
        output_dir=out,
        split="train",
        img_size=(12, 120, 120),
    )
    print(
        "[reben:configilm] "
        f"status={configilm_preflight.get('status')} "
        f"root_dir={configilm_preflight.get('root_dir')} "
        f"lmdb_path={configilm_preflight.get('lmdb_path')}"
    )
    _, resolved_lmdb_path, _ = resolve_reben_root_dir(args.lmdb_root)
    payload_format = detect_lmdb_payload_format(resolved_lmdb_path)
    if configilm_preflight.get("status") == "failed" and payload_format == "safetensors":
        configilm_preflight["status"] = "ok"
        configilm_preflight["configilm_status"] = "unsupported_payload"
        configilm_preflight["adapter_fallback"] = "lmdb_safetensors"
        print("[reben:configilm] ConfigILM pickle loader unsupported for safetensors LMDB; using repo LMDB+safetensors adapter.")

    resources: list[dict[str, object]] = [
        {"name": "reben_images_lmdb", "path": str(args.lmdb_root), "status": "ok" if args.lmdb_root.exists() else "missing", "size_bytes": 0},
        {"name": "reben_configilm_dependency_chain", "path": "python_imports", "status": dependency_check["status"], "size_bytes": 0},
        {"name": "reben_dataset_payload_adapter", "path": str(out / "reben_configilm_preflight.json"), "status": configilm_preflight["status"], "size_bytes": 0},
        {"name": "croma_repo_use_croma.py", "path": str(args.croma_repo / "use_croma.py"), "status": croma_repo["status"]},
        {"name": "croma_checkpoint_CROMA_base.pt", "path": str(args.croma_checkpoint), "status": croma_checkpoint["status"], "size_bytes": args.croma_checkpoint.stat().st_size if args.croma_checkpoint.exists() else 0},
    ]
    resources.extend({"name": row["name"], "path": row["path"], "status": row["status"], "size_bytes": row.get("size_bytes", 0)} for row in metadata_rows)

    blocking = [f"{item['name']}: {item['status']} (`{item['path']}`)" for item in resources if item["status"] != "ok"]
    payload: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_sources": OFFICIAL_SOURCES,
        "croma_repo": croma_repo,
        "croma_checkpoint": croma_checkpoint,
        "metadata": metadata_rows,
        "dependency_check": dependency_check,
        "configilm_preflight": configilm_preflight,
        "lmdb_payload_format": payload_format,
        "resources": resources,
        "disk_usage": _disk_usage([Path("/content"), args.reben_root, args.croma_checkpoint.parent, out]),
        "status": "blocked" if blocking else "ready",
        "blocking_checks": blocking,
        "missing_required_paths": [str(item["path"]) for item in resources if item["status"] == "missing"],
        "guardrails": [
            "Do not use BigEarthNet v1.",
            "Do not use lc-col BigEarthNet as a substitute for reBEN.",
            "Do not use unofficial LMDB mirrors as formal data unless the protocol is explicitly changed.",
            "Do not substitute torchvision ResNet101 for official BIFOLD ResNet101.",
        ],
    }
    (out / "preparation_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_preparation_report(out / "preparation_report.md", payload, blocking)
    if blocking:
        _write_blocked_report(out / "blocked_report.md", blocking, args)
    print(json.dumps({"status": payload["status"], "blocking_checks": blocking, "output_dir": str(out)}, indent=2))
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
