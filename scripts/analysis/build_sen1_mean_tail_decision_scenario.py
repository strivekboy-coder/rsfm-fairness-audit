"""Build the frozen Sen1Floods11 mean-vs-tail decision illustration.

This is a retrospective derivative of the existing 19-run slice atlas. It does
not select a new operating point, rerun a model, or modify canonical outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/optimization_1_7_v1/full_slice_distribution.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_supplementary_analyses_v1/sen1_decision_scenario"),
    )
    parser.add_argument(
        "--omnibus",
        type=Path,
        default=Path("outputs/thesis_draft_v0_1/derived/sen1_omnibus_summary.csv"),
        help="Authoritative configuration-level M/T/D summary used in the thesis.",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from rsfm_fairness_audit.paper_supplementary import (
        build_sen1_decision_scenario,
        write_csv,
        write_json,
    )

    source = args.source if args.source.is_absolute() else repo / args.source
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    if not source.is_file():
        raise FileNotFoundError(source)
    detail, summary = build_sen1_decision_scenario(read_rows(source))
    omnibus = args.omnibus if args.omnibus.is_absolute() else repo / args.omnibus
    if not omnibus.is_file():
        raise FileNotFoundError(omnibus)
    authoritative = read_rows(omnibus)
    mode_map = {"S1+S2": "s1_plus_s2", "S2": "s2"}
    for item in summary:
        match = [
            row for row in authoritative
            if row.get("model") == "supervised_resnet34_unet"
            and row.get("mode") == mode_map[str(item["mode"])]
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one authoritative omnibus row for {item['mode']}; found {len(match)}")
        for metric in ("M", "T", "D"):
            item[metric] = float(match[0][f"{metric}_mean"])
            item[f"{metric}_seed_sd"] = float(match[0][f"{metric}_sd"])
        item["metric_source"] = str(omnibus.resolve())
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "sen1_event_mean_vs_tail_choice.csv", detail)
    write_csv(output / "sen1_mean_vs_tail_choice_summary.csv", summary)

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    ordered = sorted(detail, key=lambda row: float(row["risk_delta_mean_minus_tail_choice"]), reverse=True)
    y = np.arange(len(ordered))
    fusion = np.asarray([float(row["mean_selected_risk"]) for row in ordered])
    s2 = np.asarray([float(row["tail_selected_risk"]) for row in ordered])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.7), gridspec_kw={"width_ratios": [1.55, 1]}, constrained_layout=True)
    ax = axes[0]
    for index in range(len(ordered)):
        color = "#D55E00" if fusion[index] > s2[index] else "#0072B2"
        ax.plot([s2[index], fusion[index]], [index, index], color="#B8C1CA", lw=1.1, zorder=1)
        ax.scatter(s2[index], index, facecolors="white", edgecolors="#0072B2", marker="o", s=30, lw=1.1, zorder=2)
        ax.scatter(fusion[index], index, color=color, marker="s", s=28, zorder=3)
    ax.set_yticks(y, [str(row["event"]) for row in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Event risk (1 - IoU; lower is better)")
    ax.grid(axis="x", color="#E2E7EB", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.set_title("Observed event-wise trade-off", loc="left", fontweight="bold")
    ax.scatter([], [], facecolors="white", edgecolors="#0072B2", marker="o", label="S2 (lower T)")
    ax.scatter([], [], color="#D55E00", marker="s", label="Fusion when risk is higher")
    ax.scatter([], [], color="#0072B2", marker="s", label="Fusion when risk is lower")
    ax.legend(frameon=False, loc="lower right", fontsize=7.2)

    ax = axes[1]
    modes = [row["mode"] for row in summary]
    values = np.asarray([[float(row[key]) for key in ("M", "T", "D")] for row in summary])
    x = np.arange(3)
    ax.plot(x, values[0], color="#D55E00", marker="s", lw=1.4, label=f"{modes[0]} (mean-selected)")
    ax.plot(x, values[1], color="#0072B2", marker="o", ls="--", lw=1.4, label=f"{modes[1]} (tail-selected)")
    ax.set_xticks(x, ["M", r"$T_{0.10}$", r"$D_{0.10}$"])
    ax.set_ylabel("Event risk")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E2E7EB", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-.20, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.set_title("Selection objective changes the choice", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"sen1_mean_tail_decision_scenario.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_json(output / "manifest.json", {
        "schema": "geobwer.paper_supplement.sen1_decision.v1",
        "status": "complete",
        "scientific_role": "retrospective operational illustration",
        "source": str(source.resolve()),
        "authoritative_mtd_source": str(omnibus.resolve()),
        "frozen_outputs_modified": False,
        "model_rerun": False,
        "comparison": "U-Net S1+S2 mean-selected versus U-Net S2 tail-selected",
        "warning": "Observed held-out comparison; not a prospective utility-policy trial.",
    })
    print(json.dumps({"output_dir": str(output.resolve()), "events": len(detail), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
