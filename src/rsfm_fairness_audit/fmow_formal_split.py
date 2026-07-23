from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsfm_fairness_audit.formal_outputs import file_sha256
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


class FmowFormalSplitError(RuntimeError):
    """Raised when a leakage-free three-way fMoW split cannot be identified."""


def _text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    return "" if value.lower() in {"", "nan", "none", "null"} else value


def fmow_site_id(row: Mapping[str, Any]) -> str:
    """Return the category-scoped location key used by the fMoW protocol."""

    category = _text(row, "category") or _text(row, "class_label")
    location = _text(row, "location_id")
    if not category or not location:
        raise FmowFormalSplitError("Every fMoW row requires category and location_id to construct site_id.")
    return f"{category}|{location}"


def _stable_order(values: Sequence[str], *, seed: int, namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest(),
    )


def _balanced_site_subset(
    sites: Sequence[str],
    site_sizes: Mapping[str, int],
    *,
    fraction: float,
    seed: int,
    namespace: str,
) -> set[str]:
    """Choose a deterministic non-empty proper subset closest to the row target."""

    ordered = _stable_order(tuple(sites), seed=seed, namespace=namespace)
    if len(ordered) < 2:
        raise FmowFormalSplitError(
            f"Class {namespace!r} has only {len(ordered)} holdout site(s); calibration/test class support is not identifiable."
        )
    states: dict[int, tuple[int, ...]] = {0: ()}
    for index, site in enumerate(ordered):
        size = int(site_sizes[site])
        if size <= 0:
            raise FmowFormalSplitError(f"Invalid non-positive site size for {site!r}.")
        additions: dict[int, tuple[int, ...]] = {}
        for total, chosen in states.items():
            candidate_total = total + size
            candidate = chosen + (index,)
            incumbent = states.get(candidate_total) or additions.get(candidate_total)
            if incumbent is None or candidate < incumbent:
                additions[candidate_total] = candidate
        for total, chosen in additions.items():
            states.setdefault(total, chosen)
    target_rows = fraction * sum(int(site_sizes[site]) for site in ordered)
    target_sites = fraction * len(ordered)
    candidates = [
        (total, chosen)
        for total, chosen in states.items()
        if 0 < len(chosen) < len(ordered)
    ]
    if not candidates:
        raise FmowFormalSplitError(f"Could not form two non-empty site partitions for class {namespace!r}.")
    _, selected = min(
        candidates,
        key=lambda item: (
            abs(item[0] - target_rows),
            abs(len(item[1]) - target_sites),
            item[1],
        ),
    )
    return {ordered[index] for index in selected}


@dataclass(frozen=True)
class FmowFormalSplit:
    rows: tuple[dict[str, Any], ...]
    contract: dict[str, Any]
    support_rows: tuple[dict[str, Any], ...]


def build_fmow_formal_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_source_split: str = "train",
    holdout_source_split: str = "val",
    calibration_fraction: float = 0.50,
    seed: int = 42,
    source_split_column: str = "split",
) -> FmowFormalSplit:
    """Keep training fixed and split the old holdout by category-scoped site."""

    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be in (0,1).")
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise FmowFormalSplitError("The source manifest is empty.")
    sample_ids = [_text(row, "sample_id") or _text(row, "image_id") for row in materialized]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise FmowFormalSplitError("Source sample_id/image_id values must be non-empty and unique.")
    source_splits = [_text(row, source_split_column) for row in materialized]
    unexpected = sorted(set(source_splits) - {train_source_split, holdout_source_split})
    if unexpected:
        raise FmowFormalSplitError(
            f"Source manifest contains unexpected {source_split_column} values {unexpected}; use a new explicit split contract."
        )

    train_indices = [index for index, value in enumerate(source_splits) if value == train_source_split]
    holdout_indices = [index for index, value in enumerate(source_splits) if value == holdout_source_split]
    if not train_indices or not holdout_indices:
        raise FmowFormalSplitError("Both source train and holdout rows are required.")

    sites_by_index = [fmow_site_id(row) for row in materialized]
    train_sites = {sites_by_index[index] for index in train_indices}
    holdout_sites = {sites_by_index[index] for index in holdout_indices}
    overlap = train_sites & holdout_sites
    if overlap:
        raise FmowFormalSplitError(
            f"Source train/holdout leakage exists for {len(overlap)} category-scoped site IDs."
        )

    site_rows: dict[str, list[int]] = defaultdict(list)
    category_sites: dict[str, set[str]] = defaultdict(set)
    for index in holdout_indices:
        site = sites_by_index[index]
        category = _text(materialized[index], "category") or _text(materialized[index], "class_label")
        site_rows[site].append(index)
        category_sites[category].add(site)
    site_sizes = {site: len(indexes) for site, indexes in site_rows.items()}
    calibration_sites: set[str] = set()
    for category in sorted(category_sites):
        calibration_sites.update(
            _balanced_site_subset(
                tuple(category_sites[category]),
                site_sizes,
                fraction=float(calibration_fraction),
                seed=seed,
                namespace=category,
            )
        )
    test_sites = holdout_sites - calibration_sites
    if not calibration_sites or not test_sites or calibration_sites & test_sites:
        raise FmowFormalSplitError("Internal calibration/test site partition failure.")

    output: list[dict[str, Any]] = []
    train_index_set = set(train_indices)
    for index, source in enumerate(materialized):
        row = dict(source)
        row["source_split"] = source_splits[index]
        row["site_id"] = sites_by_index[index]
        if index in train_index_set:
            row["split"] = "train"
        elif sites_by_index[index] in calibration_sites:
            row["split"] = "calibration"
        else:
            row["split"] = "test"
        output.append(row)

    support: list[dict[str, Any]] = []
    for split in ("train", "calibration", "test"):
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in output:
            if row["split"] == split:
                by_category[_text(row, "category") or _text(row, "class_label")].append(row)
        for category in sorted(by_category):
            subset = by_category[category]
            support.append(
                {
                    "split": split,
                    "category": category,
                    "row_count": len(subset),
                    "site_count": len({row["site_id"] for row in subset}),
                    "country_count": len({_text(row, "country") for row in subset if _text(row, "country")}),
                }
            )
    holdout_categories = set(category_sites)
    for split in ("calibration", "test"):
        observed = {row["category"] for row in support if row["split"] == split}
        if observed != holdout_categories:
            raise FmowFormalSplitError(f"{split} does not retain every holdout category.")
    contract = {
        "schema": "geobwer.fmow.formal_split.v1",
        "source_split_column": source_split_column,
        "train_source_split": train_source_split,
        "holdout_source_split": holdout_source_split,
        "formal_splits": ["train", "calibration", "test"],
        "group_key": "site_id=category|location_id",
        "stratification": "exact_row_balance_within_category_over_indivisible_sites",
        "calibration_fraction_target": float(calibration_fraction),
        "seed": int(seed),
        "row_counts": {split: sum(row["split"] == split for row in output) for split in ("train", "calibration", "test")},
        "site_counts": {split: len({row["site_id"] for row in output if row["split"] == split}) for split in ("train", "calibration", "test")},
        "site_overlap_train_calibration": len(train_sites & calibration_sites),
        "site_overlap_train_test": len(train_sites & test_sites),
        "site_overlap_calibration_test": len(calibration_sites & test_sites),
        "class_count_calibration": len({row["category"] for row in support if row["split"] == "calibration"}),
        "class_count_test": len({row["category"] for row in support if row["split"] == "test"}),
    }
    return FmowFormalSplit(rows=tuple(output), contract=contract, support_rows=tuple(support))


def write_fmow_formal_split(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    train_source_split: str = "train",
    holdout_source_split: str = "val",
    calibration_fraction: float = 0.50,
    seed: int = 42,
) -> dict[str, Path]:
    source = Path(source_manifest)
    output = ensure_dir(output_dir)
    result = build_fmow_formal_split(
        read_csv_rows(source),
        train_source_split=train_source_split,
        holdout_source_split=holdout_source_split,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    manifest = output / "fmow_formal_manifest_train_calibration_test.csv"
    support = output / "fmow_formal_split_support.csv"
    contract = output / "fmow_formal_split_contract.json"
    write_csv(manifest, result.rows)
    write_csv(support, result.support_rows)
    payload = {
        **result.contract,
        "source_manifest_sha256": file_sha256(source),
        "formal_manifest_sha256": file_sha256(manifest),
    }
    contract.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest, "support": support, "contract": contract}


__all__ = [
    "FmowFormalSplit",
    "FmowFormalSplitError",
    "build_fmow_formal_split",
    "fmow_site_id",
    "write_fmow_formal_split",
]
