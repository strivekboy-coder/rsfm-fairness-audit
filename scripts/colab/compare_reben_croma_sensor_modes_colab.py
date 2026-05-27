from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv  # noqa: E402


DEFAULT_RUNS = {
    "S1": Path("/content/outputs/reben_croma_sensor_mode_audit_croma_s1_full"),
    "S2": Path("/content/outputs/reben_croma_sensor_mode_audit_croma_s2_full"),
    "S1+S2": Path("/content/outputs/reben_croma_sensor_mode_audit_croma_s1_plus_s2_full"),
}

DEFAULT_OUTPUT = Path("outputs/reben_croma_sensor_mode_audit_croma_comparison")
TARGET_COVERAGES = {0.7, 0.8, 0.9}


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("risk_name", "")),
        str(row.get("slice_variable", "")),
        str(row.get("balance_variable", "")),
    )


def _run_name_for_mode(mode: str) -> str:
    return f"croma_{mode.lower().replace('+', '_plus_')}"


def _read_required_csv(path: Path, description: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"{description} is empty: {path}")
    return rows


def _read_run(mode: str, run_dir: Path) -> dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory for {mode} does not exist: {run_dir}")
    run_name = _run_name_for_mode(mode)
    aggregate_path = run_dir / "aggregate_metrics.csv"
    if not aggregate_path.exists():
        aggregate_path = run_dir / f"aggregate_metrics_{run_name}.csv"
    bwer_path = run_dir / "bwer_summary.csv"
    selective_path = run_dir / "selective_risk_summary.csv"
    rows = {
        "mode": mode,
        "run_name": run_name,
        "run_dir": run_dir,
        "aggregate": _read_required_csv(aggregate_path, f"{mode} aggregate metrics"),
        "bwer": _read_required_csv(bwer_path, f"{mode} BWER summary"),
        "selective": _read_required_csv(selective_path, f"{mode} selective risk summary"),
    }
    return rows


def _annotate(rows: Sequence[Mapping[str, Any]], mode: str, run_name: str, source_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "sensor_mode": mode,
            "run_name": run_name,
            "source_dir": str(source_dir),
            **dict(row),
        }
        for row in rows
    ]


def _aggregate_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for row in run["aggregate"]:
            item = {
                "sensor_mode": run["mode"],
                "run_name": run["run_name"],
                "source_dir": str(run["run_dir"]),
                **dict(row),
            }
            macro_ap = _float(item.get("macro_ap"))
            micro_ap = _float(item.get("micro_ap"))
            item["aggregate_score_name"] = "macro_ap" if not math.isnan(macro_ap) else "micro_ap"
            item["aggregate_score"] = macro_ap if not math.isnan(macro_ap) else micro_ap
            item["aggregate_risk"] = 1.0 - item["aggregate_score"] if not math.isnan(float(item["aggregate_score"])) else ""
            rows.append(item)
    return rows


def _filter_bwer(rows: Sequence[Mapping[str, Any]], risk_name: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("risk_name", "")) == risk_name]


def _standardised_country_class(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("slice_variable", "")) == "country"
        and str(row.get("balance_variable", "")) == "class_label"
    ]


def _selective_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        for row in run["selective"]:
            coverage = _float(row.get("coverage_target"))
            if any(abs(coverage - target) < 1e-9 for target in TARGET_COVERAGES):
                output.append(
                    {
                        "sensor_mode": run["mode"],
                        "run_name": run["run_name"],
                        "source_dir": str(run["run_dir"]),
                        **dict(row),
                    }
                )
    return output


def _tail_rows(bwer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in bwer_rows:
        output.append(
            {
                "sensor_mode": row.get("sensor_mode", ""),
                "run_name": row.get("run_name", ""),
                "risk_name": row.get("risk_name", ""),
                "slice_variable": row.get("slice_variable", ""),
                "balance_variable": row.get("balance_variable", ""),
                "bwer": row.get("bwer", ""),
                "mean_risk": row.get("mean_risk", ""),
                "tail_risk": row.get("tail_risk", ""),
                "tail_slices": row.get("tail_slices", ""),
                "worst_slice": row.get("worst_slice", ""),
                "worst_slice_risk": row.get("worst_slice_risk", ""),
                "best_slice": row.get("best_slice", ""),
                "best_slice_risk": row.get("best_slice_risk", ""),
                "n_slices_valid": row.get("n_slices_valid", ""),
            }
        )
    return output


def _residual_tail_ratio(bwer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in bwer_rows:
        mean_risk = _float(row.get("mean_risk"))
        worst_risk = _float(row.get("worst_slice_risk"))
        ratio = worst_risk / mean_risk if mean_risk and not math.isnan(mean_risk) and not math.isnan(worst_risk) else float("nan")
        output.append(
            {
                "sensor_mode": row.get("sensor_mode", ""),
                "run_name": row.get("run_name", ""),
                "risk_name": row.get("risk_name", ""),
                "slice_variable": row.get("slice_variable", ""),
                "balance_variable": row.get("balance_variable", ""),
                "mean_risk": row.get("mean_risk", ""),
                "worst_slice": row.get("worst_slice", ""),
                "worst_slice_risk": row.get("worst_slice_risk", ""),
                "residual_tail_risk_ratio": ratio,
            }
        )
    return output


def _risk_reduction_gap(bwer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_mode_key = {(str(row.get("sensor_mode", "")), _key(row)): row for row in bwer_rows}
    output: list[dict[str, Any]] = []
    for baseline in ("S1", "S2"):
        for (mode, key), row in sorted(by_mode_key.items()):
            if mode != baseline:
                continue
            reference = by_mode_key.get(("S1+S2", key))
            if reference is None:
                continue
            output.append(
                {
                    "baseline_mode": baseline,
                    "reference_mode": "S1+S2",
                    "risk_name": key[0],
                    "slice_variable": key[1],
                    "balance_variable": key[2],
                    "baseline_mean_risk": row.get("mean_risk", ""),
                    "reference_mean_risk": reference.get("mean_risk", ""),
                    "mean_risk_reduction": _float(row.get("mean_risk")) - _float(reference.get("mean_risk")),
                    "baseline_tail_risk": row.get("tail_risk", ""),
                    "reference_tail_risk": reference.get("tail_risk", ""),
                    "tail_risk_reduction": _float(row.get("tail_risk")) - _float(reference.get("tail_risk")),
                    "baseline_bwer": row.get("bwer", ""),
                    "reference_bwer": reference.get("bwer", ""),
                    "bwer_reduction": _float(row.get("bwer")) - _float(reference.get("bwer")),
                    "baseline_worst_slice": row.get("worst_slice", ""),
                    "reference_worst_slice": reference.get("worst_slice", ""),
                    "baseline_worst_slice_risk": row.get("worst_slice_risk", ""),
                    "reference_worst_slice_risk": reference.get("worst_slice_risk", ""),
                    "worst_slice_risk_reduction": _float(row.get("worst_slice_risk")) - _float(reference.get("worst_slice_risk")),
                }
            )
    return output


def _aggregate_best_mode(aggregate_rows: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    valid = [(str(row.get("sensor_mode", "")), _float(row.get("aggregate_score"))) for row in aggregate_rows]
    valid = [(mode, value) for mode, value in valid if mode and not math.isnan(value)]
    if not valid:
        return "", float("nan")
    return max(valid, key=lambda item: item[1])


def _aggregate_best_vs_bwer_best(
    aggregate_rows: Sequence[Mapping[str, Any]],
    bwer_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    aggregate_mode, aggregate_score = _aggregate_best_mode(aggregate_rows)
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in bwer_rows:
        grouped.setdefault(_key(row), []).append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        valid = [(str(row.get("sensor_mode", "")), _float(row.get("bwer"))) for row in rows]
        valid = [(mode, value) for mode, value in valid if mode and not math.isnan(value)]
        if not valid:
            continue
        bwer_mode, bwer_value = min(valid, key=lambda item: item[1])
        output.append(
            {
                "aggregate_score_name": "macro_ap",
                "aggregate_best_mode": aggregate_mode,
                "aggregate_best_score": aggregate_score,
                "risk_name": key[0],
                "slice_variable": key[1],
                "balance_variable": key[2],
                "bwer_best_mode": bwer_mode,
                "bwer_best_value": bwer_value,
                "aggregate_best_equals_bwer_best": str(aggregate_mode == bwer_mode),
            }
        )
    return output


def _best_summary(rows: Sequence[Mapping[str, Any]], risk_name: str, slice_variable: str, balance_variable: str = "") -> str:
    candidates = [
        row
        for row in rows
        if str(row.get("risk_name", "")) == risk_name
        and str(row.get("slice_variable", "")) == slice_variable
        and str(row.get("balance_variable", "")) == balance_variable
    ]
    valid = [(str(row.get("sensor_mode", "")), _float(row.get("bwer"))) for row in candidates]
    valid = [(mode, value) for mode, value in valid if mode and not math.isnan(value)]
    if not valid:
        return "unavailable"
    mode, value = min(valid, key=lambda item: item[1])
    return f"{mode} (BWER={value:.6g})"


def _write_findings(
    path: Path,
    *,
    aggregate_rows: Sequence[Mapping[str, Any]],
    bwer_rows: Sequence[Mapping[str, Any]],
    aggregate_vs_bwer: Sequence[Mapping[str, Any]],
) -> None:
    aggregate_mode, aggregate_score = _aggregate_best_mode(aggregate_rows)
    mismatch_count = sum(1 for row in aggregate_vs_bwer if str(row.get("aggregate_best_equals_bwer_best")) == "False")
    lines = [
        "# reBEN / CROMA Sensor-Mode Comparison",
        "",
        "Generated by `scripts/colab/compare_reben_croma_sensor_modes_colab.py`.",
        "",
        "## Scope",
        "",
        "This is a post-hoc comparison of completed CROMA full-run outputs for S1, S2, and S1+S2. It does not modify raw outputs, recompute embeddings, change BWER definitions, or tune metrics.",
        "",
        "BCE-BWER and binary-error BWER are reported separately. BCE risk is the primary probability-aware multi-label risk primitive; thresholded binary error is a secondary diagnostic primitive.",
        "",
        "## Protocol Caveats",
        "",
        "- The BIFOLD supervised ResNet101 reference remains blocked because the official `reben_publication.BigEarthNetv2_0_ImageClassifier` code is unavailable in the current Colab environment. The analyzer therefore compares CROMA sensor modes only.",
        "- The currently used HF LMDB is an unofficial preconverted safetensors LMDB. It is useful for running the audit, but it should be treated as protocol-risk relative to an official ConfigILM pickle LMDB or directly reproduced BigEarthNet Encoder export.",
        "- Sensor mode is a cross-run condition, not a sample-level geography or fairness slice.",
        "",
        "## Cautious Observations",
        "",
        f"- Aggregate-best mode by macro-AP: `{aggregate_mode}` with score `{aggregate_score:.6g}`." if aggregate_mode else "- Aggregate-best mode by macro-AP: unavailable.",
        f"- Lowest BCE country Raw-BWER: {_best_summary(bwer_rows, 'risk_bce', 'country')}.",
        f"- Lowest BCE country | class_label standardised BWER: {_best_summary(bwer_rows, 'risk_bce', 'country', 'class_label')}.",
        f"- Lowest binary-error country Raw-BWER: {_best_summary(bwer_rows, 'risk_binary_error', 'country')}.",
        f"- Aggregate-best and BWER-best differ in {mismatch_count} comparable slice/risk rows.",
        "",
        "These rows can support a cautious sensor-mode risk comparison only after reviewing support diagnostics and the protocol-risk notes above. They do not establish causal fairness claims and should not be presented as a BIFOLD-vs-CROMA model comparison.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_sensor_modes(run_dirs: Mapping[str, Path], output_dir: Path) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    runs = [_read_run(mode, path) for mode, path in run_dirs.items()]
    aggregate = _aggregate_rows(runs)
    bwer = []
    for run in runs:
        bwer.extend(_annotate(run["bwer"], run["mode"], run["run_name"], run["run_dir"]))

    bce = _filter_bwer(bwer, "risk_bce")
    binary = _filter_bwer(bwer, "risk_binary_error")
    standardised = _standardised_country_class(bwer)
    selective = _selective_rows(runs)
    tails = _tail_rows(bwer)
    residual = _residual_tail_ratio(bwer)
    reduction = _risk_reduction_gap(bwer)
    aggregate_vs_bwer = _aggregate_best_vs_bwer_best(aggregate, bwer)

    artifacts = {
        "aggregate_sensor_mode_comparison": output / "aggregate_sensor_mode_comparison.csv",
        "bce_bwer_sensor_mode_comparison": output / "bce_bwer_sensor_mode_comparison.csv",
        "binary_error_bwer_sensor_mode_comparison": output / "binary_error_bwer_sensor_mode_comparison.csv",
        "standardised_country_class_bwer_comparison": output / "standardised_country_class_bwer_comparison.csv",
        "selective_risk_comparison": output / "selective_risk_comparison.csv",
        "worst_tail_slices_by_mode": output / "worst_tail_slices_by_mode.csv",
        "residual_tail_risk_ratio": output / "residual_tail_risk_ratio.csv",
        "risk_reduction_gap": output / "risk_reduction_gap.csv",
        "aggregate_best_vs_bwer_best": output / "aggregate_best_vs_bwer_best.csv",
        "scientific_findings_reben_croma": output / "scientific_findings_reben_croma.md",
    }
    write_csv(artifacts["aggregate_sensor_mode_comparison"], aggregate)
    write_csv(artifacts["bce_bwer_sensor_mode_comparison"], bce)
    write_csv(artifacts["binary_error_bwer_sensor_mode_comparison"], binary)
    write_csv(artifacts["standardised_country_class_bwer_comparison"], standardised)
    write_csv(artifacts["selective_risk_comparison"], selective)
    write_csv(artifacts["worst_tail_slices_by_mode"], tails)
    write_csv(artifacts["residual_tail_risk_ratio"], residual)
    write_csv(artifacts["risk_reduction_gap"], reduction)
    write_csv(artifacts["aggregate_best_vs_bwer_best"], aggregate_vs_bwer)
    _write_findings(
        artifacts["scientific_findings_reben_croma"],
        aggregate_rows=aggregate,
        bwer_rows=bwer,
        aggregate_vs_bwer=aggregate_vs_bwer,
    )
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare completed CROMA reBEN sensor-mode full-run outputs.")
    parser.add_argument("--s1-dir", type=Path, default=DEFAULT_RUNS["S1"])
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_RUNS["S2"])
    parser.add_argument("--s1-plus-s2-dir", type=Path, default=DEFAULT_RUNS["S1+S2"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = compare_sensor_modes(
        {
            "S1": args.s1_dir,
            "S2": args.s2_dir,
            "S1+S2": args.s1_plus_s2_dir,
        },
        args.output_dir,
    )
    print(f"[reben:croma:compare] output_dir={args.output_dir}")
    for name, path in artifacts.items():
        print(f"[reben:croma:compare] {name}: {path}")


if __name__ == "__main__":
    main()
