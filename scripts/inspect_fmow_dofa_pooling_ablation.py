from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fmow_sanity_common import write_csv, write_json
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.* in tests
    from scripts.fmow_sanity_common import write_csv, write_json


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_cache_npz(run_dir: Path, split: str) -> Path | None:
    metadata = _load_json(run_dir / "run_metadata.json")
    key = f"{split}_embedding_cache_path"
    if metadata.get(key):
        path = Path(str(metadata[key]))
        if path.exists():
            return path
    cache_root = run_dir.parent / "embedding_cache"
    candidates = sorted(cache_root.glob(f"dofa_{split}_*.npz")) if cache_root.exists() else []
    if not candidates:
        candidates = sorted(run_dir.rglob(f"dofa_{split}_*.npz"))
    return candidates[0] if candidates else None


def _array_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "path": str(path or ""),
            "exists": False,
            "shape": "",
            "mean": "",
            "variance": "",
            "sha256": "",
            "sample_ids_sha256": "",
        }
    data = np.load(path, allow_pickle=False)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    sample_ids = np.asarray(data["sample_ids"]).astype(str) if "sample_ids" in data else np.asarray([], dtype=str)
    return {
        "path": str(path),
        "exists": True,
        "shape": "x".join(str(v) for v in embeddings.shape),
        "mean": float(np.mean(embeddings)) if embeddings.size else "",
        "variance": float(np.var(embeddings)) if embeddings.size else "",
        "sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids.tolist()).encode("utf-8")).hexdigest() if sample_ids.size else "",
    }


def _compare_arrays(flat_path: Path | None, mean_path: Path | None) -> dict[str, Any]:
    if flat_path is None or mean_path is None or not flat_path.exists() or not mean_path.exists():
        return {"same_shape": False, "max_abs_diff": "", "identical": False, "note": "one or both cache files missing"}
    flat = np.asarray(np.load(flat_path, allow_pickle=False)["embeddings"], dtype=np.float32)
    mean = np.asarray(np.load(mean_path, allow_pickle=False)["embeddings"], dtype=np.float32)
    if flat.shape != mean.shape:
        return {"same_shape": False, "max_abs_diff": "", "identical": False, "note": f"shape mismatch: {flat.shape} vs {mean.shape}"}
    max_abs_diff = float(np.max(np.abs(flat - mean))) if flat.size else 0.0
    return {
        "same_shape": True,
        "max_abs_diff": max_abs_diff,
        "identical": bool(max_abs_diff == 0.0),
        "note": "",
    }


def _metadata_summary(run_dir: Path) -> dict[str, Any]:
    metadata = _load_json(run_dir / "run_metadata.json")
    debug = _load_json(run_dir / "model_debug.json")
    return {
        "embedding_pooling": metadata.get("embedding_pooling", debug.get("embedding_pooling", "")),
        "embedding_dim": metadata.get("embedding_dim", debug.get("embedding_dim", "")),
        "train_embedding_cache_key": metadata.get("train_embedding_cache_key", ""),
        "eval_embedding_cache_key": metadata.get("eval_embedding_cache_key", ""),
        "input_scale": metadata.get("input_scale", debug.get("input_scale", "")),
        "wavelength_list_length": len(metadata.get("wavelength_list", debug.get("wavelength_list", [])) or []),
    }


def inspect_pooling(output_dir: Path, flatten_dir: Path, mean_tokens_dir: Path) -> dict[str, Any]:
    flatten_meta = _metadata_summary(flatten_dir)
    mean_meta = _metadata_summary(mean_tokens_dir)
    rows: list[dict[str, Any]] = []
    pair_summaries: dict[str, Any] = {}
    for split in ("train", "eval"):
        flat_path = _resolve_cache_npz(flatten_dir, split)
        mean_path = _resolve_cache_npz(mean_tokens_dir, split)
        flat_stats = _array_stats(flat_path)
        mean_stats = _array_stats(mean_path)
        comparison = _compare_arrays(flat_path, mean_path)
        pair_summaries[split] = {
            "flatten": flat_stats,
            "mean_tokens": mean_stats,
            "comparison": comparison,
        }
        rows.append({"split": split, "pooling": "flatten", **flat_stats})
        rows.append({"split": split, "pooling": "mean_tokens", **mean_stats})
        rows.append(
            {
                "split": split,
                "pooling": "comparison",
                "path": "",
                "exists": "",
                "shape": f"{flat_stats.get('shape')} vs {mean_stats.get('shape')}",
                "mean": "",
                "variance": "",
                "sha256": "",
                "sample_ids_sha256": "",
                "same_shape": comparison["same_shape"],
                "max_abs_diff": comparison["max_abs_diff"],
                "identical": comparison["identical"],
                "note": comparison["note"],
            }
        )

    all_identical = all(pair_summaries[split]["comparison"]["identical"] for split in ("train", "eval"))
    cache_keys_differ = (
        flatten_meta.get("train_embedding_cache_key") != mean_meta.get("train_embedding_cache_key")
        or flatten_meta.get("eval_embedding_cache_key") != mean_meta.get("eval_embedding_cache_key")
    )
    pooling_labels_differ = flatten_meta.get("embedding_pooling") != mean_meta.get("embedding_pooling")
    shapes = {pair_summaries[split]["flatten"].get("shape") for split in ("train", "eval")}
    two_dimensional = all(len(str(shape).split("x")) == 2 for shape in shapes if shape)
    if all_identical and pooling_labels_differ and cache_keys_differ and two_dimensional:
        reason = (
            "The pooling parameter reached run metadata and cache keys, but cached embeddings are already 2D and identical. "
            "The current DOFA adapter/output path exposes a pooled representation rather than token/spatial features; "
            "there is no token dimension for mean_tokens to change."
        )
    elif all_identical and not pooling_labels_differ:
        reason = "Pooling labels are not distinct in run metadata; the pooling parameter likely did not reach the completed runs."
    elif all_identical:
        reason = "Flatten and mean_tokens embeddings are identical; inspect raw adapter output shape before interpreting pooling as an ablation."
    else:
        reason = "Flatten and mean_tokens embeddings differ; pooling changed the cached features."

    summary = {
        "flatten_dir": str(flatten_dir),
        "mean_tokens_dir": str(mean_tokens_dir),
        "flatten_metadata": flatten_meta,
        "mean_tokens_metadata": mean_meta,
        "all_identical": all_identical,
        "cache_keys_differ": cache_keys_differ,
        "pooling_labels_differ": pooling_labels_differ,
        "reason": reason,
        "splits": pair_summaries,
    }
    write_csv(output_dir / "pooling_embedding_diagnostics.csv", rows)
    write_json(output_dir / "pooling_embedding_diagnostics.json", summary)

    lines = [
        "# DOFA Pooling Ablation Sanity",
        "",
        "This is a protocol sanity finding, not a model-performance finding.",
        "",
        "- input_scale: `10000`",
        f"- flatten metadata pooling: `{flatten_meta.get('embedding_pooling', '')}`",
        f"- mean_tokens metadata pooling: `{mean_meta.get('embedding_pooling', '')}`",
        f"- cache keys differ: `{cache_keys_differ}`",
        f"- embeddings identical across train/eval: `{all_identical}`",
        f"- diagnosis: {reason}",
        "- CLS: unavailable in the current adapter/output contract; no CLS result is fabricated.",
        "",
        "Embedding diagnostics are written to `pooling_embedding_diagnostics.csv` and `pooling_embedding_diagnostics.json`.",
    ]
    (output_dir / "pooling_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect completed DOFA pooling ablation embedding caches without rerunning DOFA.")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/outputs/baseline_closure_sanity/dofa_pooling_ablation"))
    parser.add_argument("--flatten-dir", type=Path)
    parser.add_argument("--mean-tokens-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir
    flatten_dir = args.flatten_dir or output_dir / "flatten"
    mean_tokens_dir = args.mean_tokens_dir or output_dir / "mean_tokens"
    summary = inspect_pooling(output_dir, flatten_dir, mean_tokens_dir)
    print(f"Wrote pooling diagnostics: {output_dir / 'pooling_embedding_diagnostics.json'}")
    print(summary["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
