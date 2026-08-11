from __future__ import annotations

import csv
import shutil
import uuid
from pathlib import Path

from scripts.colab.run_reben_labelwise_sensitivity_colab import discover_inputs


def test_discovers_one_exact_27_run_panel_and_matching_metrics() -> None:
    root = Path("work") / f"reben_labelwise_discovery_{uuid.uuid4().hex}"
    bundles = root / "frozen_probability_bundles"
    bundles.mkdir(parents=True)
    run_ids: list[str] = []
    for family in ("croma", "resnet50", "terramind"):
        for mode in ("s1", "s2", "s1_plus_s2"):
            for seed in (42, 73, 101):
                run_id = f"{family}__{mode}__seed_{seed}"
                run_ids.append(run_id)
                (bundles / f"{run_id}.npz").write_bytes(b"fixture")
    metrics = root / "unified_27run_metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_id", "geobwer"])
        writer.writeheader()
        writer.writerows({"run_id": run_id, "geobwer": 0.1} for run_id in run_ids)
    probability_dir, unified_metrics = discover_inputs(root)
    assert probability_dir == bundles
    assert unified_metrics == metrics
    shutil.rmtree(root)
