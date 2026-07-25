from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.fmow_geography_contract import (  # noqa: E402
    DEFAULT_CODE_POLICY,
    FmowGeographyContractError,
    build_fmow_geography_contract,
    write_fmow_geography_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the fMoW country/region assignment and mapping provenance."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--mapping-artifact", type=Path, required=True)
    parser.add_argument("--mapping-source-name", required=True)
    parser.add_argument("--mapping-source-version", required=True)
    parser.add_argument("--mapping-source-url", required=True)
    parser.add_argument("--code-policy", type=Path, default=DEFAULT_CODE_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    contract = build_fmow_geography_contract(
        args.metadata_csv,
        args.mapping_artifact,
        mapping_source_name=args.mapping_source_name,
        mapping_source_version=args.mapping_source_version,
        mapping_source_url=args.mapping_source_url,
        code_policy=args.code_policy,
    )
    output = write_fmow_geography_contract(args.output, contract)
    print(f"[fmow:geography-contract] written: {output}")
    print(f"[fmow:geography-contract] hash: {contract['contract_hash']}")
    print(
        "[fmow:geography-contract] formal_compatible: "
        f"{contract['formal_compatible']}"
    )
    if not contract["formal_compatible"]:
        raise FmowGeographyContractError(
            "The contract records unresolved geography and cannot bind a formal run."
        )


if __name__ == "__main__":
    main()
