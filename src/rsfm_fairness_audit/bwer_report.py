from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


def write_bwer_report(
    path: str | Path,
    summary_rows: Sequence[Mapping[str, object]],
    slice_rows: Sequence[Mapping[str, object]],
    warnings: Sequence[str],
    audit_level: str = "pilot",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = summary_rows[0].get("dataset", "") if summary_rows else ""
    model = summary_rows[0].get("model", "") if summary_rows else ""
    task = summary_rows[0].get("task", "") if summary_rows else ""
    lines = [
        "# BWER Slice Fairness Audit",
        "",
        f"- Audit level: `{audit_level}`",
        f"- Dataset: `{dataset}`",
        f"- Model: `{model}`",
        f"- Task: `{task}`",
        "",
        "BWER is a support-aware, composition-standardised, CVaR-style tail-risk statistic for deployment-relevant remote sensing slices. It is evidence of slice-level deployment risk, not causal proof of model bias.",
        "",
        "## BWER Summary",
        "",
        "| slice | balance | BWER | mean risk | tail risk | max BWER | worst slice | CI |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        ci = ""
        if row.get("ci_low") not in ("", None):
            ci = f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]"
        lines.append(
            "| {slice_variable} | {balance_variable} | {bwer:.4f} | {mean_risk:.4f} | {tail_risk:.4f} | {max_bwer:.4f} | {worst_slice} | {ci} |".format(
                slice_variable=row.get("slice_variable", ""),
                balance_variable=row.get("balance_variable", "") or "raw",
                bwer=float(row.get("bwer", float("nan"))),
                mean_risk=float(row.get("mean_risk", float("nan"))),
                tail_risk=float(row.get("tail_risk", float("nan"))),
                max_bwer=float(row.get("max_bwer", float("nan"))),
                worst_slice=row.get("worst_slice", ""),
                ci=ci,
            )
        )
    valid_count = sum(str(row.get("is_valid_slice")).lower() in {"true", "1"} for row in slice_rows)
    invalid_count = len(slice_rows) - valid_count
    lines.extend(
        [
            "",
            "## Slice Support",
            "",
            f"- Valid slice rows: {valid_count}",
            f"- Invalid slice rows retained for debugging: {invalid_count}",
            "",
            "For Sen1Floods11, chip-level classification is a sanity audit. Native pixel-level segmentation is the paper-grade disaster/event fairness path, and `event_id` should be interpreted as an operational disaster-event slice rather than a causal country fairness attribute.",
            "",
            "## Worst Tail Slices",
            "",
            "| slice | value | balance | n | risk |",
            "|---|---|---|---:|---:|",
        ]
    )
    tail_rows = [row for row in slice_rows if str(row.get("is_tail_slice")).lower() in {"true", "1"}]
    for row in tail_rows[:25]:
        lines.append(
            f"| {row.get('slice_variable', '')} | {row.get('slice_value', '')} | {row.get('balance_variable', '') or 'raw'} | {row.get('n_units', '')} | {float(row.get('balanced_risk', float('nan'))):.4f} |"
        )
    lines.extend(["", "## Warnings And Limitations", ""])
    if warnings:
        for warning in sorted(set(str(warning) for warning in warnings)):
            lines.append(f"- {warning}")
    else:
        lines.append("- No BWER warnings were emitted.")
    lines.extend(
        [
            "",
            "## Files Produced",
            "",
            "- `audit_table.csv`",
            "- `bwer_summary.csv`",
            "- `bwer_by_slice.csv`",
            "- `support_diagnostics.csv`",
            "- `bootstrap_ci.csv`",
            "- `warnings.json`",
            "- `figures/average_vs_bwer.png`",
            "- `figures/raw_vs_balanced_bwer.png`",
            "- `figures/worst_tail_slices.png`",
            "- `figures/slice_risk_heatmap.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_warnings_json(path: str | Path, warnings: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"warnings": sorted(set(str(warning) for warning in warnings))}, indent=2), encoding="utf-8")
