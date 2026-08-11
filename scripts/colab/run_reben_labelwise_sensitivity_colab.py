from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rsfm_fairness_audit.evidence_rebuild_v060 import (  # noqa: E402
    run_reben_labelwise_sensitivity,
    seal_evidence_output,
)


BUNDLE_PATTERN = re.compile(r"(.+)__(s1|s2|s1_plus_s2)__seed_(42|73|101)\.npz$")


def _run_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if not {"run_id", "geobwer"}.issubset(fields):
            return set()
        return {str(row.get("run_id", "")) for row in reader if row.get("run_id")}


def discover_inputs(search_root: Path) -> tuple[Path, Path]:
    if not search_root.is_dir():
        raise FileNotFoundError(f"Missing reBEN search root: {search_root}")
    print(f"[reben:labelwise:discover] scanning probability bundles under {search_root}", flush=True)
    by_parent: dict[Path, list[Path]] = defaultdict(list)
    for path in search_root.rglob("*.npz"):
        if BUNDLE_PATTERN.fullmatch(path.name):
            by_parent[path.parent].append(path)
    bundle_dirs = sorted(parent for parent, paths in by_parent.items() if len(paths) == 27)
    if len(bundle_dirs) != 1:
        inventory = {str(parent): len(paths) for parent, paths in sorted(by_parent.items())}
        raise RuntimeError(
            f"Expected one directory with exactly 27 frozen reBEN bundles; "
            f"found={list(map(str, bundle_dirs))}, inventory={inventory}. "
            "Pass --probability-dir explicitly if multiple frozen panels exist."
        )
    probability_dir = bundle_dirs[0]
    expected_ids = {path.stem for path in by_parent[probability_dir]}
    print(f"[reben:labelwise:discover] probability_dir={probability_dir}", flush=True)
    metric_candidates: list[Path] = []
    print(f"[reben:labelwise:discover] scanning unified metric CSVs under {search_root}", flush=True)
    for path in search_root.rglob("*.csv"):
        if "metric" not in path.name.lower() and "summary" not in path.name.lower():
            continue
        try:
            if expected_ids == _run_ids(path):
                metric_candidates.append(path)
        except (OSError, UnicodeError, csv.Error):
            continue
    metric_candidates = sorted(set(metric_candidates))
    if len(metric_candidates) != 1:
        raise RuntimeError(
            f"Expected one unified metric CSV covering the 27 bundle run IDs; "
            f"found={list(map(str, metric_candidates))}. "
            "Pass --unified-metrics explicitly if multiple summaries exist."
        )
    print(f"[reben:labelwise:discover] unified_metrics={metric_candidates[0]}", flush=True)
    return probability_dir, metric_candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only reBEN validation-locked labelwise sensitivity over 27 frozen runs"
    )
    parser.add_argument(
        "--drive-root", type=Path,
        default=Path("/content/drive/MyDrive/rsfm_fairness_audit"),
    )
    parser.add_argument("--probability-dir", type=Path)
    parser.add_argument("--unified-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    search_root = args.drive_root / "outputs" / "geobwer_final_v3"
    probability_dir = args.probability_dir
    unified_metrics = args.unified_metrics
    if probability_dir is None or unified_metrics is None:
        discovered_probability, discovered_metrics = discover_inputs(search_root)
        probability_dir = probability_dir or discovered_probability
        unified_metrics = unified_metrics or discovered_metrics
    output_dir = args.output_dir or (
        search_root / "geobwer_evidence_rebuild_v060" / "reben_labelwise_sensitivity_v13"
    )
    paths = run_reben_labelwise_sensitivity(
        probability_dir=probability_dir,
        unified_metrics=unified_metrics,
        output_dir=output_dir,
    )
    paths["completion"] = seal_evidence_output(output_dir)
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"REBEN_LABELWISE_SENSITIVITY_COMPLETE output={output_dir}")


if __name__ == "__main__":
    main()
