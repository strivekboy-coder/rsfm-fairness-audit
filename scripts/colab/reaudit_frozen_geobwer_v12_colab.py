from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rsfm_fairness_audit.bwer_protocol import BWERProtocol
from rsfm_fairness_audit.evidence_registry import load_canonical_evidence_registry
from rsfm_fairness_audit.frozen_evidence_reaudit import certification_protocol, reaudit_frozen_table


AXES = {
    "alphaearth": (
        "country_iso3", "region", "worldcover_class_name", "country_class",
        "region_class", "income_group", "urban_rural_or_built_proxy",
    ),
    "fmow_sentinel": (
        "country", "region", "un_region", "continent", "class", "class_name",
        "class_label", "country_class", "region_class", "region_superclass", "country_superclass",
    ),
    "reben": ("country", "region", "mode", "label", "class_name"),
}


def _header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return tuple(next(csv.reader(stream)))


def _nearest_protocol(table: Path, root: Path) -> BWERProtocol:
    current = table.parent
    while current == root or root in current.parents:
        candidates = (
            current / "geobwer_protocol.json",
            current / "geobwer_raw" / "geobwer_protocol.json",
        )
        for candidate in candidates:
            if candidate.exists():
                return BWERProtocol.from_mapping(json.loads(candidate.read_text(encoding="utf-8")))
        if current == root:
            break
        current = current.parent
    return BWERProtocol()


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only frozen evidence re-audit with certification 1.2")
    parser.add_argument("--task", choices=tuple(AXES), required=True)
    parser.add_argument("--registry-asset-id", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--registry",
        default=str(PROJECT_ROOT / "configs" / "analysis" / "canonical_evidence_registry_v1.yaml"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Source and re-audit roots must be disjoint.")
    tables = sorted(source_root.rglob("formal_audit_table.csv"))
    if not tables:
        raise FileNotFoundError(f"No formal_audit_table.csv beneath {source_root}")
    registry = load_canonical_evidence_registry(args.registry)
    inventory: list[dict[str, object]] = []
    completed: list[str] = []
    for index, table in enumerate(tables):
        header = _header(table)
        axes = tuple(axis for axis in AXES[args.task] if axis in header)
        cluster = next(
            (name for name in ("spatial_block_id", "source_tile_id", "site_id", "cluster_id") if name in header),
            "",
        )
        inventory.append({"table": str(table), "axes": list(axes), "cluster_column": cluster})
        if args.inventory_only or not axes or not cluster:
            continue
        source_protocol = _nearest_protocol(table, source_root)
        # AlphaEarth must not inherit the reference cluster calibration: its
        # spatial block scale requires a separate validation-only gate.
        calibration_signature = (
            "" if args.task == "alphaearth"
            else "bf3f976328dc202b00e38fa2436af155854991b0eee8a2a963621f928bbf19f9"
        )
        protocol = certification_protocol(
            source_protocol,
            task=args.task,
            calibration_signature=calibration_signature,
            min_clusters_for_inference=75,
        )
        destination = output_root / f"table_{index:04d}_{table.parent.name}"
        reaudit_frozen_table(
            source_table=table,
            output_dir=destination,
            protocol=protocol,
            group_columns=axes,
            cluster_column=cluster,
            registry=registry,
            registry_asset_id=args.registry_asset_id,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        completed.append(str(destination))
    if args.inventory_only:
        print(json.dumps(inventory, indent=2))
        print("GEOBWER_V12_FROZEN_INVENTORY=PASS")
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "batch_manifest.json").write_text(
        json.dumps(
            {
                "schema": "geobwer.frozen_reaudit_batch.v1",
                "task": args.task,
                "source_root": str(source_root),
                "completed": completed,
                "inventory": inventory,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"GEOBWER_V12_FROZEN_REAUDIT={args.task}:PASS")
    print(f"OUTPUT_ROOT={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
