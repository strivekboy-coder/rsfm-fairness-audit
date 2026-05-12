from __future__ import annotations

from pathlib import Path
from typing import Sequence


def write_static_report(output_dir: str | Path, summary_rows: Sequence[dict], gap_rows: Sequence[dict]) -> Path:
    output = Path(output_dir)
    lines = [
        "# Dummy RSFM Fairness Audit Report",
        "",
        "This smoke report uses synthetic EO samples with intentional region, sensor, class, and region-class imbalance.",
        "",
        "## Summary",
        "",
        "| Gap | Average | Worst | Best-Worst Gap | Worst Group |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['gap_name']} | {row['average_performance']:.3f} | "
            f"{row['worst_region_performance']:.3f} | {row['best_worst_gap']:.3f} | {row['worst_group']} |"
        )
    lines.extend(["", "## Raw vs Balanced", ""])
    lines.append("| Slice | Raw Gap | Balanced Gap | Residual Gap |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in gap_rows:
        lines.append(
            f"| {row['slice_name']} | {row['raw_fairness_gap']:.3f} | "
            f"{row['balanced_fairness_gap']:.3f} | {row['residual_gap_after_balancing']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Average vs worst](average_vs_worst.png)",
            "",
            "![Sensor fairness heatmap](sensor_fairness_heatmap.png)",
            "",
            "![Representation shift](representation_shift.png)",
        ]
    )
    path = output / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
