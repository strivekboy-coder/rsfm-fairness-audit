from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.reben_data_benchmark import (  # noqa: E402
    benchmark_reben_loader_workers,
)
from rsfm_fairness_audit.reben_resnet50_campaign import (  # noqa: E402
    MODE_CHANNELS,
    RebenResNet50Config,
    _dataset,
    _mode_slug,
)


def _workers(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or result[0] != 0:
        raise argparse.ArgumentTypeError("Worker list must start with 0.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded, forward-only reBEN LMDB/DataLoader benchmark."
    )
    parser.add_argument("--lmdb-root", type=Path, required=True)
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--metadata-snow-cloud-parquet", type=Path)
    parser.add_argument("--train-contract-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", action="append", choices=("S2", "S1+S2")
    )
    parser.add_argument("--workers", type=_workers, default=(0, 2, 4, 8))
    parser.add_argument("--batches", type=int, default=150)
    parser.add_argument("--checksum-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser


def _benchmark_contract(path: Path, mode: str) -> tuple[dict, str]:
    if path.is_file():
        contract = json.loads(path.read_text(encoding="utf-8"))
        if (
            contract.get("sensor_mode") == mode
            and len(contract.get("mean", [])) == MODE_CHANNELS[mode]
            and len(contract.get("std", [])) == MODE_CHANNELS[mode]
        ):
            return contract, "formal_train_contract"
    # A missing fusion normalization contract must not trigger a 155 GB
    # preflight during a bounded I/O benchmark. Neutral normalization performs
    # the identical decoding/collation/transfer work and is never persisted as
    # a scientific contract.
    return {
        "mean": np.zeros(MODE_CHANNELS[mode], dtype=float).tolist(),
        "std": np.ones(MODE_CHANNELS[mode], dtype=float).tolist(),
    }, "identity_normalization_benchmark_only"


def main() -> None:
    args = build_parser().parse_args()
    modes = tuple(args.mode or ("S2", "S1+S2"))
    reports = []
    for mode in modes:
        config = RebenResNet50Config(
            lmdb_root=args.lmdb_root,
            metadata_parquet=args.metadata_parquet,
            metadata_snow_cloud_parquet=args.metadata_snow_cloud_parquet,
            output_dir=args.output.parent / "_benchmark_only",
            sensor_modes=(mode,),
            seeds=(42, 73, 101),
            batch_size=args.batch_size,
            num_workers=0,
            prefetch_factor=args.prefetch_factor,
            device=args.device,
        )
        adapter = _dataset(config, "train", mode)
        contract, source = _benchmark_contract(
            args.train_contract_dir / f"{_mode_slug(mode)}.json",
            mode,
        )
        report = benchmark_reben_loader_workers(
            adapter,
            contract,
            config,
            mode=mode,
            worker_counts=args.workers,
            max_batches=args.batches,
            checksum_batches=args.checksum_batches,
            reference_checkpoint=(
                args.output.parent
                / f"{args.output.stem}_{_mode_slug(mode)}_reference.pt"
            ),
        )
        report["normalization_source"] = source
        reports.append(report)
        print(
            f"[reben:loader-benchmark] mode={mode} "
            f"recommended_num_workers={report['recommended_num_workers']}",
            flush=True,
        )
        for row in report["results"]:
            print(
                f"  workers={row['num_workers']} "
                f"samples_per_second={row['samples_per_second']:.3f} "
                f"data_wait_seconds={row['data_wait_seconds']:.3f} "
                f"step_seconds={row['step_seconds']:.3f}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "geobwer.reben.loader_benchmark_suite.v1",
                "formal_evidence": False,
                "reports": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[reben:loader-benchmark] report={args.output}", flush=True)


if __name__ == "__main__":
    main()
