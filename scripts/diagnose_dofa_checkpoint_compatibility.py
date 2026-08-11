from __future__ import annotations

import argparse
from functools import partial
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


PREFIXES = ("module.", "_orig_mod.", "model.", "backbone.")
STATE_CONTAINERS = ("state_dict", "model_state_dict", "model", "backbone")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _numel(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    shape = _shape(value)
    if shape is None:
        return 0
    result = 1
    for item in shape:
        result *= item
    return result


def _normalise_key(raw_key: Any) -> str:
    key = str(raw_key)
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, Any], str]:
    state = checkpoint
    container = "<root>"
    if isinstance(state, Mapping):
        for key in STATE_CONTAINERS:
            candidate = state.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                container = key
                break
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a recognizable state dictionary.")
    return {_normalise_key(key): value for key, value in state.items()}, container


def compare_state_dicts(
    model_state: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
) -> dict[str, Any]:
    matched: list[str] = []
    missing_from_checkpoint: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for key, model_value in model_state.items():
        if key not in checkpoint_state:
            missing_from_checkpoint.append(key)
            continue
        checkpoint_value = checkpoint_state[key]
        if _shape(model_value) != _shape(checkpoint_value):
            shape_mismatches.append(
                {
                    "key": key,
                    "model_shape": _shape(model_value),
                    "checkpoint_shape": _shape(checkpoint_value),
                    "model_numel": _numel(model_value),
                    "checkpoint_numel": _numel(checkpoint_value),
                }
            )
            continue
        matched.append(key)

    missing_from_model = sorted(set(checkpoint_state) - set(model_state))
    model_total_numel = sum(_numel(value) for value in model_state.values())
    matched_numel = sum(_numel(model_state[key]) for key in matched)
    absent_model_numel = sum(_numel(model_state[key]) for key in missing_from_checkpoint)
    shape_mismatch_model_numel = sum(item["model_numel"] for item in shape_mismatches)
    extra_checkpoint_numel = sum(_numel(checkpoint_state[key]) for key in missing_from_model)

    return {
        "counts": {
            "model_keys": len(model_state),
            "checkpoint_keys": len(checkpoint_state),
            "matched_keys": len(matched),
            "model_keys_missing_from_checkpoint": len(missing_from_checkpoint),
            "checkpoint_keys_missing_from_model": len(missing_from_model),
            "same_name_shape_mismatches": len(shape_mismatches),
        },
        "parameters": {
            "model_total_numel": model_total_numel,
            "matched_model_numel": matched_numel,
            "model_numel_missing_from_checkpoint": absent_model_numel,
            "shape_mismatch_model_numel": shape_mismatch_model_numel,
            "checkpoint_numel_missing_from_model": extra_checkpoint_numel,
            "coverage": matched_numel / max(model_total_numel, 1),
        },
        "matched_keys": sorted(matched),
        "model_keys_missing_from_checkpoint": sorted(missing_from_checkpoint),
        "checkpoint_keys_missing_from_model": missing_from_model,
        "same_name_shape_mismatches": sorted(shape_mismatches, key=lambda item: item["key"]),
    }


def _top_level_summary(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {"type": type(checkpoint).__name__, "mapping": False}
    entries: dict[str, Any] = {}
    for key, value in checkpoint.items():
        entry: dict[str, Any] = {"type": type(value).__name__}
        if isinstance(value, Mapping):
            entry["mapping_keys"] = len(value)
        shape = _shape(value)
        if shape is not None:
            entry["shape"] = shape
        entries[str(key)] = entry
    return {"type": type(checkpoint).__name__, "mapping": True, "entries": entries}


def _git_revision(repo_path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower() if result.returncode == 0 else None


def _build_model(dofa_module: Any, constructor: str) -> Any:
    if constructor == "dofa_v1_patch16":
        return dofa_module.vit_base_patch16()
    if constructor != "dofav2_base_patch14":
        raise ValueError(f"Unknown constructor: {constructor}")

    import torch
    import torch.nn as nn

    class LayerScale(nn.Module):
        def __init__(self, dim: int, init_values: float = 1e-5) -> None:
            super().__init__()
            self.gamma = nn.Parameter(init_values * torch.ones(dim))

        def forward(self, value: Any) -> Any:
            return value * self.gamma

    # DOFAv2's author-maintained TerraTorch implementation defines the base
    # release as patch-14, 224 px, ViT-B/14, with LayerScale initialized at
    # 1e-5.  The official checkpoint is a root-level state dict, so constructing
    # the equivalent flat OFAViT keeps its published key namespace intact.
    model = dofa_module.OFAViT(
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=0,
        global_pool=False,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    for block in model.blocks:
        block.ls1 = LayerScale(768)
        block.ls2 = LayerScale(768)
    return model


def diagnose(repo_path: Path, checkpoint_path: Path, constructor: str) -> dict[str, Any]:
    import torch

    try:
        import timm
    except ImportError:
        timm = None

    repo_path = repo_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    sys.path.insert(0, str(repo_path))
    try:
        dofa_module = importlib.import_module("dofa_v1")
        model = _build_model(dofa_module, constructor)
    finally:
        try:
            sys.path.remove(str(repo_path))
        except ValueError:
            pass

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_state, container = extract_state_dict(checkpoint)
    comparison = compare_state_dicts(model.state_dict(), checkpoint_state)
    return {
        "schema": "rsfm.dofa_checkpoint_compatibility.v1",
        "environment": {
            "python": sys.version,
            "torch": getattr(torch, "__version__", "unknown"),
            "timm": getattr(timm, "__version__", "unavailable"),
        },
        "source": {
            "repo_path": str(repo_path),
            "repo_revision": _git_revision(repo_path),
            "constructor": constructor,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path),
            "top_level": _top_level_summary(checkpoint),
            "selected_state_container": container,
        },
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report exact DOFA model/checkpoint key and parameter compatibility without running inference."
    )
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--constructor",
        choices=("dofa_v1_patch16", "dofav2_base_patch14"),
        default="dofa_v1_patch16",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = diagnose(args.repo_path, args.checkpoint, args.constructor)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
