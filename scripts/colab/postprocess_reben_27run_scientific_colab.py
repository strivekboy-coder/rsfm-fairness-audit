from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.reben_scientific_postprocess import (  # noqa: E402
    run_reben_scientific_postprocess,
)


def _csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only CPU scientific postprocess for the frozen reBEN 27-run panel."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path)
    source.add_argument("--review-zip", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path("/content/reben_27run_scientific_work"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--betas", type=_csv_floats, default=(0.10, 0.20, 0.30))
    parser.add_argument("--min-units", type=int, default=20)
    parser.add_argument("--cluster-thresholds", type=_csv_ints, default=(2, 3, 5))
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = args.source_root
    if args.review_zip is not None:
        args.work_root.mkdir(parents=True, exist_ok=True)
        local_zip = args.work_root / args.review_zip.name
        if not local_zip.is_file() or local_zip.stat().st_size != args.review_zip.stat().st_size:
            print(f"[reben:postprocess] copying review package to local scratch: {local_zip}", flush=True)
            shutil.copy2(args.review_zip, local_zip)
        extract_root = args.work_root / "extracted"
        marker = extract_root / ".extract_complete"
        if not marker.is_file():
            print(f"[reben:postprocess] extracting review package: {extract_root}", flush=True)
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local_zip) as archive:
                archive.extractall(extract_root)
            marker.write_text(str(local_zip.stat().st_size), encoding="utf-8")
        source_root = extract_root
    assert source_root is not None
    print("[reben:postprocess] starting read-only CPU analysis", flush=True)
    artifacts = run_reben_scientific_postprocess(
        source_root,
        args.output_dir,
        beta=args.beta,
        betas=args.betas,
        min_units=args.min_units,
        cluster_thresholds=args.cluster_thresholds,
        confidence_level=args.confidence_level,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print("[reben:postprocess] complete", flush=True)
    for name, path in artifacts.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
