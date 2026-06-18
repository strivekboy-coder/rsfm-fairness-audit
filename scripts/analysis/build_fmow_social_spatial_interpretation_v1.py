from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from scripts.analysis.build_fmow_conformal_selective_audit import (
    _float,
    _is_missing,
    _mean,
    _normalise_rows,
    build_grouped_calibration_split,
    confidence_threshold_for_coverage,
    discover_fmow_audit_tables,
    retained_by_threshold,
)


DEFAULT_FMOW_SELECTIVE = Path("outputs/fmow_conformal_selective_audit_v1")
DEFAULT_UNIFIED_V2 = Path("outputs/unified_audit_matrix_v2")
DEFAULT_OUTPUT = Path("outputs/fmow_social_spatial_interpretation_v1")


def _selector_label(selector: str, coverage: Any) -> str:
    if selector == "baseline_all_test":
        return "baseline"
    if selector == "confidence_topk_test":
        return f"topk_{coverage}"
    if selector == "calibrated_confidence_threshold_diagnostic":
        return f"calibrated_threshold_{coverage}"
    return f"{selector}_{coverage}"


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    selector: str,
    coverage: float,
    slice_variable: str,
    min_support: int,
    overall_mean_risk: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = row.get(slice_variable)
        if not _is_missing(value):
            grouped.setdefault(str(value), []).append(row)
    output = []
    for value, items in sorted(grouped.items()):
        risks = [_float(row.get("_risk")) for row in items]
        support = len(items)
        countries = sorted({str(row.get("country")) for row in items if not _is_missing(row.get("country"))})
        output.append(
            {
                "run_id": run_id,
                "selector": selector,
                "scenario": _selector_label(selector, coverage),
                "coverage_target": coverage,
                "slice_variable": slice_variable,
                "slice_value": value,
                "country": value if slice_variable == "country" else "",
                "region": value if slice_variable == "region" else "",
                "support_count": support,
                "location_count": len({str(row.get("location_id")) for row in items if not _is_missing(row.get("location_id"))}),
                "class_count": len({str(row.get("class_label")) for row in items if not _is_missing(row.get("class_label"))}),
                "country_count": len(countries),
                "countries": ";".join(countries[:12]),
                "mean_risk": _mean(risks),
                "risk_excess_vs_overall": _mean(risks) - overall_mean_risk,
                "support_ok": support >= min_support,
                "support_filter": f"support_count >= {min_support}",
            }
        )
    return output


def _tail_lookup(bwer_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], set[str]]:
    lookup: dict[tuple[str, str, str, str], set[str]] = {}
    for row in bwer_rows:
        if row.get("analysis_type") != "raw":
            continue
        key = (str(row.get("run_id")), str(row.get("selector")), str(row.get("coverage_target")), str(row.get("slice_variable")))
        lookup[key] = {item for item in str(row.get("tail_slices", "")).split(";") if item}
    return lookup


def _annotate_tails(rows: list[dict[str, Any]], bwer_rows: Sequence[Mapping[str, Any]]) -> None:
    lookup = _tail_lookup(bwer_rows)
    for row in rows:
        key = (str(row.get("run_id")), str(row.get("selector")), str(row.get("coverage_target")), str(row.get("slice_variable")))
        row["is_bwer_tail_slice"] = row.get("slice_value") in lookup.get(key, set())


def _read_indicator_rows(indicator_csv: Path | None) -> tuple[dict[str, dict[str, str]], str]:
    if indicator_csv is None:
        return {}, "unavailable_no_indicator_csv"
    if not indicator_csv.exists():
        return {}, f"unavailable_missing_indicator_csv:{indicator_csv}"
    rows = read_csv_rows(indicator_csv)
    output = {}
    for row in rows:
        key = row.get("iso3") or row.get("country_code") or row.get("country") or row.get("Country Code")
        if key:
            output[str(key).upper()] = row
    return output, "available_user_supplied_indicator_csv"


def build_indicator_join(country_rows: Sequence[Mapping[str, Any]], indicator_csv: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    indicators, status = _read_indicator_rows(indicator_csv)
    output = []
    for row in country_rows:
        if row.get("slice_variable") != "country":
            continue
        country = str(row.get("country") or row.get("slice_value") or "").upper()
        indicator = indicators.get(country, {})
        output.append(
            {
                **dict(row),
                "indicator_status": status if indicator else ("unavailable_no_country_indicator_match" if indicators else status),
                "gdp_per_capita": indicator.get("gdp_per_capita", indicator.get("NY.GDP.PCAP.CD", "")),
                "population_density": indicator.get("population_density", indicator.get("EN.POP.DNST", "")),
                "income_group": indicator.get("income_group", indicator.get("IncomeGroup", "")),
                "urban_population_share": indicator.get("urban_population_share", indicator.get("SP.URB.TOTL.IN.ZS", "")),
            }
        )
    associations = _indicator_associations(output)
    return output, associations, status


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _indicator_associations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indicators = ["gdp_per_capita", "population_density", "urban_population_share"]
    output = []
    for indicator in indicators:
        for scenario in sorted({str(row.get("scenario")) for row in rows}):
            paired = [(_float(row.get(indicator)), _float(row.get("mean_risk"))) for row in rows if row.get("scenario") == scenario and bool(row.get("support_ok"))]
            clean = [(x, y) for x, y in paired if not math.isnan(x) and not math.isnan(y)]
            output.append(
                {
                    "indicator": indicator,
                    "scenario": scenario,
                    "n_countries": len(clean),
                    "pearson_r": _pearson([x for x, _ in clean], [y for _, y in clean]),
                    "association_status": "available" if len(clean) >= 3 else "unavailable_insufficient_indicator_matches",
                    "claim_scope": "exploratory association only; not causal",
                }
            )
    if not output:
        output.append({"indicator": "", "scenario": "", "n_countries": 0, "pearson_r": "", "association_status": "unavailable_no_indicator_data", "claim_scope": "exploratory association only; not causal"})
    return output


def support_filter_rows(rows: Sequence[Mapping[str, Any]], min_support: int) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if _float(row.get("support_count"), 0.0) >= min_support]


def _make_country_plot(rows: Sequence[Mapping[str, Any]], path_base: Path, *, title: str, metric: str = "mean_risk") -> None:
    ensure_dir(path_base.parent)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        path_base.with_suffix(".png").write_text("matplotlib unavailable\n", encoding="utf-8")
        path_base.with_suffix(".pdf").write_text("matplotlib unavailable\n", encoding="utf-8")
        return
    supported = sorted([row for row in rows if bool(row.get("support_ok"))], key=lambda row: -_float(row.get(metric)))[:30]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    values = [_float(row.get(metric)) for row in supported]
    bars = ax.bar(range(len(supported)), values, color=plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(supported)))))
    ax.set_xticks(range(len(supported)))
    ax.set_xticklabels([str(row.get("country") or row.get("slice_value")) for row in supported], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, row in zip(bars[:8], supported[:8]):
        if bool(row.get("is_bwer_tail_slice")):
            bar.set_edgecolor("#111111")
            bar.set_linewidth(1.2)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _make_scatter(rows: Sequence[Mapping[str, Any]], path_base: Path, *, x_key: str, y_key: str, title: str) -> None:
    ensure_dir(path_base.parent)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        path_base.with_suffix(".png").write_text("matplotlib unavailable\n", encoding="utf-8")
        path_base.with_suffix(".pdf").write_text("matplotlib unavailable\n", encoding="utf-8")
        return
    supported = [row for row in rows if bool(row.get("support_ok"))]
    x = [_float(row.get(x_key)) for row in supported]
    y = [_float(row.get(y_key)) for row in supported]
    labels = [str(row.get("country") or row.get("slice_value")) for row in supported]
    clean = [(a, b, label) for a, b, label in zip(x, y, labels) if not math.isnan(a) and not math.isnan(b)]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    if clean:
        ax.scatter([a for a, _, _ in clean], [b for _, b, _ in clean], s=28, alpha=0.75, color="#2F5DA8")
        for a, b, label in sorted(clean, key=lambda item: -item[1])[:8]:
            ax.text(a, b, label, fontsize=7)
    else:
        ax.text(0.5, 0.5, "Indicator data unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(y_key.replace("_", " "))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _make_region_heatmap(rows: Sequence[Mapping[str, Any]], path_base: Path) -> None:
    ensure_dir(path_base.parent)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        path_base.with_suffix(".png").write_text("matplotlib unavailable\n", encoding="utf-8")
        path_base.with_suffix(".pdf").write_text("matplotlib unavailable\n", encoding="utf-8")
        return
    supported = [row for row in rows if bool(row.get("support_ok"))]
    regions = sorted({str(row.get("region") or row.get("slice_value")) for row in supported})
    scenarios = sorted({str(row.get("scenario")) for row in supported})
    matrix = np.full((len(regions), len(scenarios)), np.nan)
    for i, region in enumerate(regions):
        for j, scenario in enumerate(scenarios):
            vals = [_float(row.get("mean_risk")) for row in supported if str(row.get("region") or row.get("slice_value")) == region and str(row.get("scenario")) == scenario]
            matrix[i, j] = _mean(vals)
    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 0.7), max(4, len(regions) * 0.22)))
    image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=np.nanmin(matrix) if np.isfinite(matrix).any() else 0, vmax=np.nanmax(matrix) if np.isfinite(matrix).any() else 1)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=7)
    ax.set_title("fMoW region mean risk by audit scenario")
    fig.colorbar(image, ax=ax, label="mean risk")
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _make_bwer_summary_plot(rank_rows: Sequence[Mapping[str, Any]], path_base: Path) -> None:
    ensure_dir(path_base.parent)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        path_base.with_suffix(".png").write_text("matplotlib unavailable\n", encoding="utf-8")
        path_base.with_suffix(".pdf").write_text("matplotlib unavailable\n", encoding="utf-8")
        return
    rows = [row for row in rank_rows if row.get("analysis_type") == "raw"]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    labels = [str(row.get("selector")).replace("confidence_topk_test", "top-k").replace("calibrated_confidence_threshold_diagnostic", "cal").replace("baseline_all_test", "base") + "\n" + str(row.get("coverage_target")) for row in rows]
    ax.bar(range(len(rows)), [_float(row.get("bwer_best")) for row in rows], color="#2E8B70")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Best country Raw-BWER")
    ax.set_title("Aggregate-best vs BWER-best divergence persists")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _write_figures(output: Path, country_rows: Sequence[Mapping[str, Any]], region_rows: Sequence[Mapping[str, Any]], indicator_rows: Sequence[Mapping[str, Any]], rank_rows: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    figures = ensure_dir(output / "figures")
    paths: dict[str, Path] = {}
    baseline = [row for row in country_rows if row.get("selector") == "baseline_all_test"]
    selective = [row for row in country_rows if row.get("selector") == "confidence_topk_test" and str(row.get("coverage_target")) == "0.8"]
    _make_country_plot(baseline, figures / "country_bwer_choropleth", title="fMoW country risk, baseline test split")
    _make_country_plot(selective, figures / "country_selective_bwer_choropleth", title="fMoW country risk, top-k selective 80%")
    _make_scatter(baseline, figures / "bwer_vs_support_scatter", x_key="support_count", y_key="mean_risk", title="Country risk vs support")
    _make_scatter(indicator_rows, figures / "bwer_vs_income_or_gdp_scatter", x_key="gdp_per_capita", y_key="mean_risk", title="Country risk vs GDP per capita")
    _make_region_heatmap(region_rows, figures / "region_bwer_heatmap")
    _make_bwer_summary_plot(rank_rows, figures / "aggregate_vs_selective_bwer_fmow_summary")
    for stem in [
        "country_bwer_choropleth",
        "country_selective_bwer_choropleth",
        "bwer_vs_support_scatter",
        "bwer_vs_income_or_gdp_scatter",
        "region_bwer_heatmap",
        "aggregate_vs_selective_bwer_fmow_summary",
    ]:
        paths[f"{stem}_png"] = figures / f"{stem}.png"
        paths[f"{stem}_pdf"] = figures / f"{stem}.pdf"
    return paths


def build_fmow_social_spatial_interpretation(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    fmow_selective_dir: Path = DEFAULT_FMOW_SELECTIVE,
    unified_v2_dir: Path = DEFAULT_UNIFIED_V2,
    indicator_csv: Path | None = None,
    min_support: int = 20,
    seed: int = 42,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    tables = discover_fmow_audit_tables()
    bwer_rows = read_csv_rows(fmow_selective_dir / "fmow_selective_bwer_summary.csv") + read_csv_rows(fmow_selective_dir / "fmow_conformal_bwer_summary.csv")
    selective_summary = read_csv_rows(fmow_selective_dir / "fmow_selective_risk_summary.csv")
    rank_rows = read_csv_rows(fmow_selective_dir / "rank_divergence_under_selective_audit.csv")
    by_summary = {(row["run_id"], row["selector"], row["coverage_target"]): _float(row.get("mean_risk")) for row in selective_summary}

    country_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for run_id, table in sorted(tables.items()):
        rows = _normalise_rows(read_csv_rows(table), run_id)
        split_rows, _report = build_grouped_calibration_split(rows, seed=seed, calibration_fraction=0.5)
        calibration = [row for row in split_rows if row.get("calibration_split") == "calibration"]
        test = [row for row in split_rows if row.get("calibration_split") == "test"]
        scenarios: list[tuple[str, float, list[Mapping[str, Any]]]] = [("baseline_all_test", 1.0, test)]
        for coverage in (0.7, 0.8, 0.9):
            scenarios.append(("confidence_topk_test", coverage, retained_by_threshold(test, confidence_threshold_for_coverage(test, coverage))))
            scenarios.append(("calibrated_confidence_threshold_diagnostic", coverage, retained_by_threshold(test, confidence_threshold_for_coverage(calibration, coverage))))
        for selector, coverage, scenario_rows in scenarios:
            overall = by_summary.get((run_id, selector, str(coverage)), _mean([_float(row.get("_risk")) for row in scenario_rows]))
            country_rows.extend(_group_rows(scenario_rows, run_id=run_id, selector=selector, coverage=coverage, slice_variable="country", min_support=min_support, overall_mean_risk=overall))
            region_rows.extend(_group_rows(scenario_rows, run_id=run_id, selector=selector, coverage=coverage, slice_variable="region", min_support=min_support, overall_mean_risk=overall))
        support_by_country: dict[str, list[Mapping[str, Any]]] = {}
        for row in test:
            if not _is_missing(row.get("country")):
                support_by_country.setdefault(str(row.get("country")), []).append(row)
        for country, items in sorted(support_by_country.items()):
            support_rows.append(
                {
                    "run_id": run_id,
                    "country": country,
                    "support_count": len(items),
                    "location_count": len({str(row.get("location_id")) for row in items if not _is_missing(row.get("location_id"))}),
                    "class_count": len({str(row.get("class_label")) for row in items if not _is_missing(row.get("class_label"))}),
                    "region": next((str(row.get("region")) for row in items if not _is_missing(row.get("region"))), ""),
                    "support_ok": len(items) >= min_support,
                    "support_filter": f"support_count >= {min_support}",
                }
            )
    _annotate_tails(country_rows, bwer_rows)
    _annotate_tails(region_rows, bwer_rows)
    country_rows = sorted(country_rows, key=lambda row: (str(row.get("run_id")), str(row.get("selector")), _float(row.get("coverage_target")), str(row.get("country"))))
    region_rows = sorted(region_rows, key=lambda row: (str(row.get("run_id")), str(row.get("selector")), _float(row.get("coverage_target")), str(row.get("region"))))

    indicator_join, associations, indicator_status = build_indicator_join(country_rows, indicator_csv)
    caveats = [
        {"category": "causal_scope", "caveat": "Associations are exploratory deployment-risk interpretation, not causal fairness claims."},
        {"category": "support_filter", "caveat": f"Country/region interpretation uses support_count >= {min_support}; low-support countries are retained in CSV but not treated as worst-slice evidence."},
        {"category": "indicator_data", "caveat": indicator_status},
        {"category": "map_fallback", "caveat": "Figures named choropleth use a country-code ranked heatmap fallback unless geospatial polygons are added later."},
        {"category": "conformal_scope", "caveat": "Calibrated-threshold rows remain diagnostic; full conformal prediction is not claimed without true-class probabilities or full probability vectors."},
    ]

    artifacts = {
        "fmow_country_risk_summary": output / "fmow_country_risk_summary.csv",
        "fmow_region_risk_summary": output / "fmow_region_risk_summary.csv",
        "fmow_spatial_support_summary": output / "fmow_spatial_support_summary.csv",
        "fmow_social_indicator_join": output / "fmow_social_indicator_join.csv",
        "fmow_risk_indicator_association": output / "fmow_risk_indicator_association.csv",
        "fmow_social_spatial_caveats": output / "fmow_social_spatial_caveats.csv",
        "fmow_social_spatial_report": output / "fmow_social_spatial_report.md",
    }
    write_csv(artifacts["fmow_country_risk_summary"], country_rows)
    write_csv(artifacts["fmow_region_risk_summary"], region_rows)
    write_csv(artifacts["fmow_spatial_support_summary"], support_rows)
    write_csv(artifacts["fmow_social_indicator_join"], indicator_join)
    write_csv(artifacts["fmow_risk_indicator_association"], associations)
    write_csv(artifacts["fmow_social_spatial_caveats"], caveats)
    fig_paths = _write_figures(output, country_rows, region_rows, indicator_join, rank_rows)
    artifacts.update({f"figure_{key}": value for key, value in fig_paths.items()})

    divergent = [row for row in rank_rows if str(row.get("rank_diverges")).lower() == "true"]
    supported_country_rows = len([row for row in country_rows if row.get("selector") == "baseline_all_test" and bool(row.get("support_ok"))])
    supported_unique_countries = len({row["country"] for row in country_rows if row.get("selector") == "baseline_all_test" and bool(row.get("support_ok"))})
    artifacts["fmow_social_spatial_report"].write_text(
        "# fMoW social-spatial interpretation v1\n\n"
        "This is a post-hoc interpretation layer over saved fMoW Step3 and selective-BWER outputs. No training or inference was run.\n\n"
        "## Findings\n\n"
        f"- Supported baseline run-country rows after filtering: {supported_country_rows}; unique countries: {supported_unique_countries}.\n"
        f"- Aggregate-vs-BWER divergence rows inherited from selective audit: {len(divergent)} / {len(rank_rows)}.\n"
        "- Country and region tables report support counts, retained coverage, mean risk, risk excess, and BWER-tail membership.\n"
        f"- Socio-economic indicator status: {indicator_status}.\n"
        "- No causal, global fairness, or full conformal prediction claim is made.\n\n"
        "## Interpretation boundary\n\n"
        "Use these outputs to describe exploratory spatial structure in deployment tail risk. Do not interpret country risk as country-level bias or causal disadvantage.\n",
        encoding="utf-8",
    )

    unified_summary = {
        "fmow_social_spatial_summary": unified_v2_dir / "fmow_social_spatial_interpretation_summary.csv",
        "paper_ready_report": unified_v2_dir / "paper_ready_fmow_selective_audit_report.md",
    }
    ensure_dir(unified_v2_dir)
    write_csv(
        unified_summary["fmow_social_spatial_summary"],
        [
            {
                "experiment_id": "fmow_sentinel_step3_social_spatial_v1",
                "supported_baseline_run_country_rows": supported_country_rows,
                "supported_baseline_unique_countries": supported_unique_countries,
                "rank_divergence_rows": len(rank_rows),
                "rank_divergence_true_rows": len(divergent),
                "indicator_status": indicator_status,
                "claim_scope": "exploratory post-hoc deployment-risk interpretation",
            }
        ],
    )
    previous = unified_summary["paper_ready_report"].read_text(encoding="utf-8") if unified_summary["paper_ready_report"].exists() else "# Paper-ready fMoW selective audit report\n"
    social_section = (
        "## Social-spatial interpretation v1\n\n"
        f"- Supported baseline run-country rows: {supported_country_rows}; unique countries: {supported_unique_countries}.\n"
        f"- Rank divergence persists in {len(divergent)} of {len(rank_rows)} selective/conformal-diagnostic rows.\n"
        f"- Indicator join status: {indicator_status}.\n"
        "- Interpretation remains exploratory and non-causal; support filtering is required before discussing worst countries.\n"
    )
    marker = "## Social-spatial interpretation v1"
    if marker in previous:
        previous = previous.split(marker, 1)[0].rstrip() + "\n\n" + social_section
    else:
        previous = previous.rstrip() + "\n\n" + social_section
    unified_summary["paper_ready_report"].write_text(previous, encoding="utf-8")
    artifacts.update({f"unified_v2_{key}": value for key, value in unified_summary.items()})
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fMoW social-spatial interpretation v1 assets.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fmow-selective-dir", type=Path, default=DEFAULT_FMOW_SELECTIVE)
    parser.add_argument("--unified-v2-dir", type=Path, default=DEFAULT_UNIFIED_V2)
    parser.add_argument("--indicator-csv", type=Path)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    artifacts = build_fmow_social_spatial_interpretation(
        output_dir=args.out,
        fmow_selective_dir=args.fmow_selective_dir,
        unified_v2_dir=args.unified_v2_dir,
        indicator_csv=args.indicator_csv,
        min_support=args.min_support,
        seed=args.seed,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
