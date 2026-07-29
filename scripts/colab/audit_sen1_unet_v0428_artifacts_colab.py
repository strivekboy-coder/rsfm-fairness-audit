from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rsfm_fairness_audit.sen1_unet_artifact_audit import (  # noqa: E402
    audit_sen1_unet_v0428_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only audit of the frozen Sen1 v0.4.28 U-Net panel."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    audit_sen1_unet_v0428_artifacts(
        args.source_root,
        output_json=args.output_json,
        repository_root=PROJECT_ROOT,
    )
    print("SEN1_V0428_UNET_ARTIFACT_AUDIT=PASS")


if __name__ == "__main__":
    main()
