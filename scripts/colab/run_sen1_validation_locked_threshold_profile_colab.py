from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit import __version__
from rsfm_fairness_audit.bwer_core import compute_geobwer
from rsfm_fairness_audit.sen1floods11_formal import load_sen1_probability_units


SCHEMA = "geobwer.sen1.validation_locked_threshold_profile.v1"
DEFAULT_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.10, 0.901, 0.05))
SPLIT_COUNTS = {"validation": 89, "standard_test": 90, "bolivia_holdout": 15}
SPLIT_EVENTS = {"validation": 10, "standard_test": 10, "bolivia_holdout": 1}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_split(parts: Iterable[str]) -> str | None:
    lowered = {str(value).lower() for value in parts}
    if "validation" in lowered:
        return "validation"
    if "bolivia_holdout" in lowered:
        return "bolivia_holdout"
    if "test" in lowered or "standard_test" in lowered:
        return "standard_test"
    return None


def _mode(parts: Iterable[str]) -> str | None:
    lowered = [str(value).lower() for value in parts]
    for value in ("s1_plus_s2", "s2", "s1"):
        if value in lowered:
            return value
    return None


def discover_exports(
    *, unet_root: Path, terramind_root: Path, prithvi_root: Path,
) -> dict[str, dict[str, Any]]:
    """Discover the frozen 19-model/57-split panel without touching source files."""

    registry: dict[str, dict[str, Any]] = {}
    sources = (
        ("supervised_resnet34_unet", "resnet34_unet", unet_root),
        ("terramind_v1_base", "terramind_v1_base", terramind_root),
        ("prithvi_eo_v2_300_tl", "prithvi_eo_v2_300_tl", prithvi_root),
    )
    for family, model_prefix, root in sources:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing frozen Sen1 source root: {root}")
        # Writer manifest names differ across the frozen families
        # (for example rank_0 for U-Net and rank_000 for TerraMind).  The
        # index_parts directory is the stable probability-export contract.
        exports = sorted(
            path.parent
            for path in root.rglob("index_parts")
            if path.is_dir() and any(path.glob("*.jsonl"))
        )
        print(f"[sen1:threshold:discover] family={family} indexed_exports={len(exports)} root={root}")
        for export in exports:
            relative = export.relative_to(root)
            split = _canonical_split(relative.parts)
            if split is None or split not in SPLIT_COUNTS:
                continue
            if family == "prithvi_eo_v2_300_tl":
                mode, seed = "s2", None
            else:
                mode = _mode(relative.parts)
                seed_match = re.search(r"seed_(42|73|101)", relative.as_posix().lower())
                seed = int(seed_match.group(1)) if seed_match else None
                if mode is None or seed is None:
                    continue
            model = f"{model_prefix}_{mode}" + (f"_seed_{seed}" if seed is not None else "")
            entry = registry.setdefault(
                model,
                {"model": model, "family": family, "mode": mode, "seed": seed, "exports": {}},
            )
            if split in entry["exports"]:
                raise RuntimeError(f"Duplicate export for model={model} split={split}: {export}")
            entry["exports"][split] = export
    return registry


def _threshold_stats(
    *, rows: Sequence[Mapping[str, Any]], probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray], valid_masks: Sequence[np.ndarray],
    thresholds: Sequence[float],
) -> dict[float, dict[str, Any]]:
    identified = [
        index for index, valid in enumerate(valid_masks) if int(np.asarray(valid, dtype=bool).sum()) > 0
    ]
    if not identified:
        raise ValueError("Split contains no identified Sen1 chips.")
    ordered = tuple(sorted(float(value) for value in thresholds))
    bins = np.asarray([-np.inf, *ordered, np.inf], dtype=float)
    risks_by_threshold: dict[float, list[float]] = {value: [] for value in ordered}
    events_by_threshold: dict[float, dict[str, list[float]]] = {
        value: defaultdict(list) for value in ordered
    }
    pooled_by_threshold: dict[float, list[int]] = {value: [0, 0, 0] for value in ordered}
    # Histogram each chip once. This avoids threshold_count full-map comparisons
    # and keeps the 5-6 GB frozen panel practical on a Colab CPU runtime.
    for index in identified:
        valid = np.asarray(valid_masks[index], dtype=bool)
        truth = np.asarray(targets[index])[valid] == 1
        scores = np.asarray(probabilities[index], dtype=float)[valid]
        positive_hist = np.histogram(scores[truth], bins=bins)[0]
        negative_hist = np.histogram(scores[~truth], bins=bins)[0]
        positive_ge = np.cumsum(positive_hist[::-1])[::-1]
        negative_ge = np.cumsum(negative_hist[::-1])[::-1]
        positive_count = int(truth.sum())
        event = str(rows[index]["event_id"])
        for threshold_index, threshold in enumerate(ordered, start=1):
            tp = int(positive_ge[threshold_index])
            fp = int(negative_ge[threshold_index])
            fn = positive_count - tp
            union = tp + fp + fn
            risk = float(1.0 - (tp / union if union else 1.0))
            risks_by_threshold[threshold].append(risk)
            events_by_threshold[threshold][event].append(risk)
            pooled = pooled_by_threshold[threshold]
            pooled[0] += tp; pooled[1] += fp; pooled[2] += fn
    output: dict[float, dict[str, Any]] = {}
    for threshold in ordered:
        risks = risks_by_threshold[threshold]
        event_risks = events_by_threshold[threshold]
        pooled_tp, pooled_fp, pooled_fn = pooled_by_threshold[threshold]
        event_means = {event: float(np.mean(values)) for event, values in event_risks.items()}
        card = compute_geobwer(event_means, beta=0.10)
        pooled_union = pooled_tp + pooled_fp + pooled_fn
        output[float(threshold)] = {
            "identified_chip_count": len(identified),
            "all_ignore_chip_count": len(rows) - len(identified),
            "event_count": len(event_means),
            "mean_chip_risk": float(np.mean(risks)),
            "pooled_pixel_iou": float(pooled_tp / pooled_union if pooled_union else 1.0),
            "pooled_true_positive": pooled_tp,
            "pooled_false_positive": pooled_fp,
            "pooled_false_negative": pooled_fn,
            "event_mean_risk": card.mean_risk,
            "event_tail_risk": card.tail_risk,
            "event_geobwer": card.bwer,
            "tail_effective_events": card.allocation.tail_effective_groups,
            "max_tail_atom_share": card.allocation.max_tail_atom_share,
            "tail_regime": card.allocation.tail_regime,
            "event_risks": event_means,
        }
    return output


def _select_validation_threshold(stats: Mapping[float, Mapping[str, Any]]) -> float:
    # Validation-only selection: minimize chip-equal 1-IoU; deterministic tie-break toward 0.5.
    return min(
        stats,
        key=lambda threshold: (
            float(stats[threshold]["mean_chip_risk"]), abs(float(threshold) - 0.5), float(threshold)
        ),
    )


def run_profile(
    *, drive_root: Path, output_dir: Path, thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    expected_models: int = 19, expected_exports: int = 57,
    expected_counts: Mapping[str, int] = SPLIT_COUNTS,
    expected_events: Mapping[str, int] = SPLIT_EVENTS,
) -> dict[str, Path]:
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Threshold grid must be non-empty and strictly inside (0,1).")
    completion = output_dir / "completion_contract.json"
    if completion.exists():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("schema") == f"{SCHEMA}.completion":
            print(f"[sen1:threshold] already complete: {completion}")
            return {"completion": completion}
        raise RuntimeError(f"Existing non-complete output requires a new directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = (
        drive_root / "outputs" / "geobwer_final_v3" / "sen1_19model_descriptive_v2"
        / "official_446_metadata_binding.csv"
    )
    registry = discover_exports(
        unet_root=drive_root / "outputs" / "geobwer_final_v3" / "sen1_geobwer_v0428" / "supervised",
        terramind_root=drive_root / "outputs" / "geobwer_final_v3" / "sen1_geobwer_v0434" / "terramind_final",
        prithvi_root=drive_root / "outputs" / "geobwer_final_v3" / "sen1_geobwer_v0432" / "prithvi_final",
    )
    export_count = sum(len(entry["exports"]) for entry in registry.values())
    if len(registry) != int(expected_models) or export_count != int(expected_exports):
        inventory = {model: sorted(entry["exports"]) for model, entry in registry.items()}
        raise RuntimeError(
            f"Frozen panel discovery failed: models={len(registry)}/{expected_models}, "
            f"exports={export_count}/{expected_exports}, inventory={inventory}"
        )
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Missing official 446-chip metadata binding: {metadata_csv}")

    profile_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    model_cache_dir = output_dir / "model_cache"
    model_cache_dir.mkdir(exist_ok=True)
    for position, model in enumerate(sorted(registry), start=1):
        entry = registry[model]
        cache_path = model_cache_dir / f"{model}.json"
        if cache_path.exists():
            print(f"[sen1:threshold:model] {position}/{len(registry)} reuse={cache_path.name}")
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("schema") != f"{SCHEMA}.model_cache" or payload.get("thresholds") != list(thresholds):
                raise RuntimeError(f"Incompatible model cache; use a new output directory: {cache_path}")
            split_stats = {
                split: {float(key): value for key, value in values.items()}
                for split, values in payload["split_stats"].items()
            }
        else:
            split_stats: dict[str, dict[float, dict[str, Any]]] = {}
            for split in ("validation", "standard_test", "bolivia_holdout"):
                export = Path(entry["exports"][split])
                print(
                    f"[sen1:threshold:load] model={position}/{len(registry)} id={model} "
                    f"split={split} expected={expected_counts[split]} export={export}"
                )
                rows, probabilities, targets, valid_masks = load_sen1_probability_units(
                    export, metadata_csv=metadata_csv
                )
                if len(rows) != int(expected_counts[split]):
                    raise RuntimeError(f"{model}/{split} count={len(rows)} expected={expected_counts[split]}")
                observed_events = {str(row["event_id"]) for row in rows}
                if len(observed_events) != int(expected_events[split]):
                    raise RuntimeError(
                        f"{model}/{split} event_count={len(observed_events)} expected={expected_events[split]}"
                    )
                if split == "bolivia_holdout" and observed_events != {"Bolivia"}:
                    raise RuntimeError(f"Bolivia contract failed for {model}: {observed_events}")
                split_stats[split] = _threshold_stats(
                    rows=rows, probabilities=probabilities, targets=targets,
                    valid_masks=valid_masks, thresholds=thresholds,
                )
                del rows, probabilities, targets, valid_masks
                print(f"[sen1:threshold:load] complete model={model} split={split}")
            cache_payload = {
                "schema": f"{SCHEMA}.model_cache", "model": model,
                "thresholds": list(thresholds),
                "split_stats": {
                    split: {str(key): value for key, value in values.items()}
                    for split, values in split_stats.items()
                },
            }
            cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        selected = _select_validation_threshold(split_stats["validation"])
        selection_rows.append({
            "model": model, "family": entry["family"], "mode": entry["mode"], "seed": entry["seed"],
            "selected_threshold": selected, "selection_split": "validation89_only",
            "selection_objective": "minimum_chip_equal_mean_one_minus_flood_iou",
            "test_or_bolivia_used_for_selection": False,
        })
        combined: dict[float, dict[str, Any]] = {}
        for threshold in thresholds:
            standard = split_stats["standard_test"][threshold]
            bolivia = split_stats["bolivia_holdout"][threshold]
            event_risks = dict(standard["event_risks"])
            event_risks.update(bolivia["event_risks"])
            card = compute_geobwer(event_risks, beta=0.10)
            pooled_tp = int(standard["pooled_true_positive"]) + int(bolivia["pooled_true_positive"])
            pooled_fp = int(standard["pooled_false_positive"]) + int(bolivia["pooled_false_positive"])
            pooled_fn = int(standard["pooled_false_negative"]) + int(bolivia["pooled_false_negative"])
            pooled_union = pooled_tp + pooled_fp + pooled_fn
            combined[threshold] = {
                "identified_chip_count": standard["identified_chip_count"] + bolivia["identified_chip_count"],
                "all_ignore_chip_count": standard["all_ignore_chip_count"] + bolivia["all_ignore_chip_count"],
                "event_count": len(event_risks),
                "mean_chip_risk": (
                    standard["mean_chip_risk"] * standard["identified_chip_count"]
                    + bolivia["mean_chip_risk"] * bolivia["identified_chip_count"]
                ) / (standard["identified_chip_count"] + bolivia["identified_chip_count"]),
                "pooled_pixel_iou": float(pooled_tp / pooled_union if pooled_union else 1.0),
                "pooled_true_positive": pooled_tp,
                "pooled_false_positive": pooled_fp,
                "pooled_false_negative": pooled_fn,
                "event_mean_risk": card.mean_risk, "event_tail_risk": card.tail_risk,
                "event_geobwer": card.bwer,
                "tail_effective_events": card.allocation.tail_effective_groups,
                "max_tail_atom_share": card.allocation.max_tail_atom_share,
                "tail_regime": card.allocation.tail_regime,
            }
        all_splits = {**split_stats, "combined_held_out": combined}
        for split, values in all_splits.items():
            for threshold, stats in values.items():
                profile_rows.append({
                    "model": model, "family": entry["family"], "mode": entry["mode"], "seed": entry["seed"],
                    "split": split, "threshold": threshold,
                    "validation_selected_threshold": selected,
                    "is_validation_selected_operating_point": bool(abs(threshold - selected) < 1e-12),
                    **{key: value for key, value in stats.items() if key != "event_risks"},
                    "beta": 0.10, "evidence_status": "descriptive_only",
                    "spatial_inference_valid": False,
                })

    profile_path = output_dir / "validation_locked_threshold_profile.csv"
    selection_path = output_dir / "validation_threshold_selection.csv"
    _write_csv(profile_path, profile_rows)
    _write_csv(selection_path, selection_rows)
    ranking_rows: list[dict[str, Any]] = []
    for split in ("standard_test", "combined_held_out"):
        for threshold in thresholds:
            selected_rows = [
                row for row in profile_rows if row["split"] == split and row["threshold"] == threshold
            ]
            selected_rows.sort(key=lambda row: (float(row["event_geobwer"]), str(row["model"])))
            for rank, row in enumerate(selected_rows, start=1):
                ranking_rows.append({
                    "split": split, "threshold": threshold, "rank": rank, "model": row["model"],
                    "family": row["family"], "mode": row["mode"], "seed": row["seed"],
                    "event_geobwer": row["event_geobwer"],
                    "is_validation_selected_operating_point": row["is_validation_selected_operating_point"],
                })
    ranking_path = output_dir / "threshold_ranking_stability.csv"
    _write_csv(ranking_path, ranking_rows)
    manifest_path = output_dir / "postprocess_manifest.json"
    manifest = {
        "schema": SCHEMA, "status": "descriptive_complete", "package_version": __version__,
        "model_count": len(registry), "split_export_count": export_count,
        "threshold_grid": list(thresholds), "threshold_grid_source": "preregistered_finite_grid",
        "threshold_selection_data": "validation89_only", "test_or_bolivia_used_for_selection": False,
        "selection_objective": "minimum_chip_equal_mean_one_minus_flood_iou",
        "split_contract": {"counts": dict(expected_counts), "events": dict(expected_events)},
        "combined_contract": {"chip_count": 105, "event_count": 11},
        "model_training_or_inference": False, "source_artifacts_modified": False,
        "spatial_inference_valid": False,
        "interpretation": "Operating-point sensitivity for descriptive Event-GeoBWER; no formal spatial LCB.",
        "metadata_csv": str(metadata_csv), "metadata_sha256": _sha256(metadata_csv),
        "source_exports": {
            model: {split: str(path) for split, path in entry["exports"].items()}
            for model, entry in registry.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = [profile_path, selection_path, ranking_path, manifest_path]
    completion.write_text(json.dumps({
        "schema": f"{SCHEMA}.completion", "status": "complete", "package_version": __version__,
        "model_training_or_inference": False,
        "artifacts": [{"path": path.name, "sha256": _sha256(path)} for path in artifacts],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sen1:threshold] SEN1_THRESHOLD_PROFILE_COMPLETE output={output_dir}")
    return {
        "profile": profile_path, "selection": selection_path, "ranking": ranking_path,
        "manifest": manifest_path, "completion": completion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only Sen1 validation-locked threshold profile")
    parser.add_argument(
        "--drive-root", type=Path,
        default=Path("/content/drive/MyDrive/rsfm_fairness_audit"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    args = parser.parse_args()
    output = args.output_dir or (
        args.drive_root / "outputs" / "geobwer_final_v3" / "geobwer_evidence_rebuild_v060"
        / "sen1_validation_locked_threshold_v12"
    )
    paths = run_profile(
        drive_root=args.drive_root, output_dir=output,
        thresholds=tuple(float(value) for value in args.thresholds.split(",") if value.strip()),
    )
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
