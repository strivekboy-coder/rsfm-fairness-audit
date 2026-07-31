from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rsfm_fairness_audit.sen1_prithvi_artifact_audit import (  # noqa: E402
    audit_sen1_prithvi_v0429_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only external audit of frozen Sen1 Prithvi v0.4.29 outputs."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--core-metadata", type=Path, required=True)
    parser.add_argument("--bolivia-metadata", type=Path, required=True)
    parser.add_argument("--unet-campaign", type=Path, required=True)
    parser.add_argument("--unet-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    audit_sen1_prithvi_v0429_artifacts(
        args.source_root,
        core_metadata=args.core_metadata,
        bolivia_metadata=args.bolivia_metadata,
        unet_campaign=args.unet_campaign,
        unet_audit=args.unet_audit,
        output_json=args.output_json,
    )
    print("SEN1_V0429_PRITHVI_ARTIFACT_AUDIT=PASS")


if __name__ == "__main__":
    main()
