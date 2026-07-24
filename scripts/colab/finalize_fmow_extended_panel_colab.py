from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.bwer_protocol import BWERProtocol  # noqa: E402
from rsfm_fairness_audit.geobwer_panel import run_geobwer_model_panel  # noqa: E402
from rsfm_fairness_audit.persistent_cache import hydrate_output, persist_output  # noqa: E402


def _seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(result) < 3 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Formal panel requires at least three unique seeds.")
    return result


def _manifest(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Required formal manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Manifest is not a JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the pre-registered same-seed DOFAv2/ResNet-50 fMoW GeoBWER panel."
    )
    parser.add_argument("--dofav2-root", type=Path, required=True)
    parser.add_argument("--resnet50-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-output-dir", type=Path)
    parser.add_argument("--seeds", type=_seeds, default=(42, 73, 101))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hydrate_output(args.output_dir, args.persistent_output_dir)
    tables: dict[str, Path] = {}
    protocol: BWERProtocol | None = None
    pairs: list[tuple[str, str]] = []
    for seed in args.seeds:
        dofa_name = f"dofav2_vit_base_e150_seed_{seed}"
        baseline_name = f"resnet50_common9_seed_{seed}"
        dofa_formal = args.dofav2_root / "probe_seeds" / f"seed_{seed}" / "formal_outputs"
        baseline_formal = args.resnet50_root / f"seed_{seed}" / "formal_outputs"
        for name, formal in ((dofa_name, dofa_formal), (baseline_name, baseline_formal)):
            manifest = _manifest(formal / "formal_output_manifest.json")
            observed_name = str(manifest.get("model_lineage", {}).get("model", ""))
            if observed_name and observed_name != name:
                raise RuntimeError(
                    f"Model identity drift for {formal}: expected={name}, observed={observed_name}."
                )
            table = formal / "formal_audit_table.csv"
            if not table.is_file():
                raise RuntimeError(f"Missing formal table: {table}")
            tables[name] = table
            current = BWERProtocol.from_mapping(manifest["protocol"])
            if protocol is not None and current.signature != protocol.signature:
                raise RuntimeError("fMoW models do not share one GeoBWER protocol hash.")
            protocol = current
        pairs.append((dofa_name, baseline_name))
    if protocol is None:
        raise RuntimeError("No formal fMoW protocol could be resolved.")
    panel = run_geobwer_model_panel(
        tables,
        args.output_dir / "model_panel",
        protocol=protocol,
        group_column="country",
        cluster_column="site_id",
        comparison_pairs=tuple(pairs),
        n_bootstrap=args.n_bootstrap,
        seed=args.seeds[0],
    )
    design = args.output_dir / "comparison_design.json"
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_text(
        json.dumps(
            {
                "schema": "geobwer.fmow.extended_comparison_design.v1",
                "primary_same_protocol_same_seed_pairs": pairs,
                "common_input_bands": "sentinel2_9_legacy",
                "split_protocol": "category_scoped_location_disjoint_train_calibration_test",
                "seed_ensemble_role": "secondary_deployment_ensemble_not_primary_architecture_test",
                "all_trained_models_export_complete_probabilities": True,
                "model_panel_summary": str(panel.model_summary),
                "paired_comparisons": str(panel.paired_comparisons),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    persist_output(args.output_dir, args.persistent_output_dir, label="fmow-extended-panel-complete")
    print(f"[fmow:extended-panel] complete: {design}")


if __name__ == "__main__":
    main()
