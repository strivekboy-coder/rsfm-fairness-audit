from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv
from scripts.analysis.build_fmow_conformal_selective_audit import _float, _mean


DEFAULT_INPUT = Path("outputs/fmow_social_spatial_interpretation_v1")
DEFAULT_OUTPUT = Path("outputs/fmow_social_spatial_interpretation_v1")
DEFAULT_UNIFIED_V2 = Path("outputs/unified_audit_matrix_v2")
WORLD_BANK_BASE = "https://api.worldbank.org/v2"
INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "population_density": "EN.POP.DNST",
    "urban_population_share": "SP.URB.TOTL.IN.ZS",
}


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).lower() in {"nan", "none", "null"}


def _get_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _latest_indicator_values(indicator_code: str) -> dict[str, dict[str, Any]]:
    url = f"{WORLD_BANK_BASE}/country/all/indicator/{indicator_code}?format=json&per_page=20000"
    payload = _get_json(url)
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        countryiso3code = str(row.get("countryiso3code") or "").upper()
        if not countryiso3code or len(countryiso3code) != 3 or row.get("value") is None:
            continue
        year = int(row.get("date") or 0)
        if countryiso3code not in latest or year > int(latest[countryiso3code].get("year", 0)):
            latest[countryiso3code] = {"value": row.get("value"), "year": year}
    return latest


def _country_metadata() -> dict[str, dict[str, Any]]:
    url = f"{WORLD_BANK_BASE}/country?format=json&per_page=400"
    payload = _get_json(url)
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    output = {}
    for row in rows:
        iso3 = str(row.get("id") or "").upper()
        if len(iso3) == 3:
            output[iso3] = {
                "country_name": row.get("name", ""),
                "region": (row.get("region") or {}).get("value", ""),
                "income_group": (row.get("incomeLevel") or {}).get("value", ""),
                "lending_type": (row.get("lendingType") or {}).get("value", ""),
            }
    return output


def fetch_world_bank_indicators() -> tuple[dict[str, dict[str, Any]], str]:
    try:
        metadata = _country_metadata()
        indicator_values = {name: _latest_indicator_values(code) for name, code in INDICATORS.items()}
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return {}, f"unavailable_world_bank_fetch_failed:{type(exc).__name__}:{exc}"
    countries = set(metadata)
    for values in indicator_values.values():
        countries.update(values)
    joined: dict[str, dict[str, Any]] = {}
    for iso3 in sorted(countries):
        row = {"iso3": iso3, **metadata.get(iso3, {})}
        for name, values in indicator_values.items():
            item = values.get(iso3, {})
            row[name] = item.get("value", "")
            row[f"{name}_year"] = item.get("year", "")
            row[f"{name}_indicator"] = INDICATORS[name]
        joined[iso3] = row
    return joined, "available_world_bank_api"


def load_or_fetch_indicators(indicator_csv: Path | None = None) -> tuple[dict[str, dict[str, Any]], str]:
    if indicator_csv is not None:
        if not indicator_csv.exists():
            return {}, f"unavailable_missing_indicator_csv:{indicator_csv}"
        rows = read_csv_rows(indicator_csv)
        output = {}
        for row in rows:
            iso3 = str(row.get("iso3") or row.get("country_code") or row.get("Country Code") or "").upper()
            if iso3:
                output[iso3] = row
        return output, "available_user_supplied_indicator_csv"
    return fetch_world_bank_indicators()


def join_indicators(country_rows: Sequence[Mapping[str, Any]], indicators: Mapping[str, Mapping[str, Any]], status: str) -> list[dict[str, Any]]:
    output = []
    for row in country_rows:
        iso3 = str(row.get("country") or row.get("slice_value") or "").upper()
        indicator = indicators.get(iso3, {})
        output.append(
            {
                **dict(row),
                "iso3": iso3,
                "indicator_status": "matched" if indicator else ("unmatched_iso3" if indicators else status),
                "country_name": indicator.get("country_name", indicator.get("name", "")),
                "world_bank_region": indicator.get("region", ""),
                "income_group": indicator.get("income_group", indicator.get("IncomeGroup", "")),
                "gdp_per_capita": indicator.get("gdp_per_capita", indicator.get("NY.GDP.PCAP.CD", "")),
                "gdp_per_capita_year": indicator.get("gdp_per_capita_year", ""),
                "population_density": indicator.get("population_density", indicator.get("EN.POP.DNST", "")),
                "population_density_year": indicator.get("population_density_year", ""),
                "urban_population_share": indicator.get("urban_population_share", indicator.get("SP.URB.TOTL.IN.ZS", "")),
                "urban_population_share_year": indicator.get("urban_population_share_year", ""),
            }
        )
    return output


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def association_rows(joined_rows: Sequence[Mapping[str, Any]], min_support: int = 20) -> list[dict[str, Any]]:
    output = []
    indicators = ["gdp_per_capita", "population_density", "urban_population_share"]
    for run_id in sorted({str(row.get("run_id")) for row in joined_rows}):
        for scenario in sorted({str(row.get("scenario")) for row in joined_rows if str(row.get("run_id")) == run_id}):
            rows = [
                row
                for row in joined_rows
                if str(row.get("run_id")) == run_id
                and str(row.get("scenario")) == scenario
                and _float(row.get("support_count"), 0.0) >= min_support
            ]
            for indicator in indicators:
                pairs = [(_float(row.get(indicator)), _float(row.get("mean_risk")), _float(row.get("support_count"))) for row in rows]
                clean = [(x, y, w) for x, y, w in pairs if not math.isnan(x) and not math.isnan(y)]
                output.append(
                    {
                        "run_id": run_id,
                        "scenario": scenario,
                        "indicator": indicator,
                        "n_countries": len(clean),
                        "pearson_r": _pearson([x for x, _, _ in clean], [y for _, y, _ in clean]),
                        "support_weighted_mean_risk": np.average([y for _, y, w in clean], weights=[w for _, _, w in clean]) if clean else "",
                        "support_filter": f"support_count >= {min_support}",
                        "association_status": "available" if len(clean) >= 3 else "unavailable_insufficient_matches",
                        "claim_scope": "exploratory deployment-risk association; not causal",
                    }
                )
    return output


def caveat_rows(indicator_status: str, real_map_status: str) -> list[dict[str, str]]:
    return [
        {"category": "indicator_source", "caveat": indicator_status},
        {"category": "support_filter", "caveat": "Country-level associations use support_count >= 20 before interpretation."},
        {"category": "causal_scope", "caveat": "Associations are exploratory deployment-risk interpretation only; no causal or global-fairness claim."},
        {"category": "map_status", "caveat": real_map_status},
        {"category": "conformal_scope", "caveat": "Calibrated threshold remains diagnostic, not full APS/RAPS conformal prediction."},
    ]


def _supported_baseline_rows(joined_rows: Sequence[Mapping[str, Any]], run_id: str = "resnet50_13band") -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in joined_rows
        if row.get("run_id") == run_id
        and row.get("scenario") == "baseline"
        and _float(row.get("support_count"), 0.0) >= 20
    ]
    return sorted(rows, key=lambda row: -_float(row.get("mean_risk")))


def _plot_indicator_scatter(rows: Sequence[Mapping[str, Any]], path_base: Path, indicator: str, title: str) -> None:
    ensure_dir(path_base.parent)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    clean = [(float(_float(row.get(indicator))), float(_float(row.get("mean_risk"))), str(row.get("iso3")), _float(row.get("support_count"))) for row in rows if not math.isnan(_float(row.get(indicator))) and not math.isnan(_float(row.get("mean_risk")))]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if clean:
        sizes = [max(18, min(120, math.sqrt(max(1.0, support)) * 8)) for _, _, _, support in clean]
        ax.scatter([x for x, _, _, _ in clean], [y for _, y, _, _ in clean], s=sizes, alpha=0.7, color="#2F5DA8", edgecolor="white", linewidth=0.4)
        for x, y, iso3, _support in sorted(clean, key=lambda item: -item[1])[:8]:
            ax.text(x, y, iso3, fontsize=7)
    else:
        ax.text(0.5, 0.5, "Indicator data unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(indicator.replace("_", " "))
    ax.set_ylabel("Country mean risk")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_support_adjusted(rows: Sequence[Mapping[str, Any]], path_base: Path) -> None:
    ensure_dir(path_base.parent)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    clean = [(math.log10(_float(row.get("gdp_per_capita"))), _float(row.get("risk_excess_vs_overall")), str(row.get("iso3")), _float(row.get("support_count"))) for row in rows if _float(row.get("gdp_per_capita")) > 0 and not math.isnan(_float(row.get("risk_excess_vs_overall")))]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if clean:
        sizes = [max(18, min(140, math.sqrt(max(1.0, support)) * 8)) for _, _, _, support in clean]
        ax.scatter([x for x, _, _, _ in clean], [y for _, y, _, _ in clean], s=sizes, alpha=0.7, color="#2E8B70", edgecolor="white", linewidth=0.4)
        for x, y, iso3, _support in sorted(clean, key=lambda item: -abs(item[1]))[:8]:
            ax.text(x, y, iso3, fontsize=7)
    else:
        ax.text(0.5, 0.5, "GDP data unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("log10 GDP per capita")
    ax.set_ylabel("Risk excess vs scenario mean")
    ax.set_title("Support-adjusted fMoW country risk vs GDP")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_ranked_barplot(rows: Sequence[Mapping[str, Any]], path_base: Path) -> None:
    ensure_dir(path_base.parent)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    top = sorted(rows, key=lambda row: -_float(row.get("mean_risk")))[:30]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(range(len(top)), [_float(row.get("mean_risk")) for row in top], color="#5A6C86")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([str(row.get("iso3")) for row in top], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Country mean risk")
    ax.set_title("fMoW country BWER-ranked barplot fallback")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _try_real_choropleth(rows: Sequence[Mapping[str, Any]], path_base: Path) -> str:
    try:
        import geopandas as gpd  # type: ignore
    except Exception:
        _plot_ranked_barplot(rows, path_base.with_name("country_bwer_ranked_barplot_v1_1"))
        return "unavailable_geopandas_not_available; wrote country_bwer_ranked_barplot_v1_1 instead"
    try:
        world_path = gpd.datasets.get_path("naturalearth_lowres")  # type: ignore[attr-defined]
        world = gpd.read_file(world_path)
        data = {str(row.get("iso3")): _float(row.get("mean_risk")) for row in rows}
        world["mean_risk"] = world["iso_a3"].map(data)
        ax = world.plot(column="mean_risk", legend=True, missing_kwds={"color": "#eeeeee"}, figsize=(10, 5), cmap="magma")
        ax.set_axis_off()
        ax.set_title("fMoW country mean risk, support-filtered")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(path_base.with_suffix(".png"), dpi=180)
        fig.savefig(path_base.with_suffix(".pdf"))
        return "available_naturalearth_geopandas"
    except Exception as exc:
        _plot_ranked_barplot(rows, path_base.with_name("country_bwer_ranked_barplot_v1_1"))
        return f"unavailable_polygon_render_failed:{type(exc).__name__}; wrote country_bwer_ranked_barplot_v1_1 instead"


def write_figures(joined_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> tuple[dict[str, Path], str]:
    figures = ensure_dir(output_dir / "figures")
    rows = _supported_baseline_rows(joined_rows)
    paths = {
        "bwer_vs_gdp_per_capita_scatter_png": figures / "bwer_vs_gdp_per_capita_scatter.png",
        "bwer_vs_gdp_per_capita_scatter_pdf": figures / "bwer_vs_gdp_per_capita_scatter.pdf",
        "bwer_vs_population_density_scatter_png": figures / "bwer_vs_population_density_scatter.png",
        "bwer_vs_population_density_scatter_pdf": figures / "bwer_vs_population_density_scatter.pdf",
        "bwer_vs_urban_population_share_scatter_png": figures / "bwer_vs_urban_population_share_scatter.png",
        "bwer_vs_urban_population_share_scatter_pdf": figures / "bwer_vs_urban_population_share_scatter.pdf",
        "support_adjusted_bwer_indicator_scatter_png": figures / "support_adjusted_bwer_indicator_scatter.png",
        "support_adjusted_bwer_indicator_scatter_pdf": figures / "support_adjusted_bwer_indicator_scatter.pdf",
    }
    _plot_indicator_scatter(rows, figures / "bwer_vs_gdp_per_capita_scatter", "gdp_per_capita", "fMoW country risk vs GDP per capita")
    _plot_indicator_scatter(rows, figures / "bwer_vs_population_density_scatter", "population_density", "fMoW country risk vs population density")
    _plot_indicator_scatter(rows, figures / "bwer_vs_urban_population_share_scatter", "urban_population_share", "fMoW country risk vs urban population share")
    _plot_support_adjusted(rows, figures / "support_adjusted_bwer_indicator_scatter")
    map_status = _try_real_choropleth(rows, figures / "country_bwer_choropleth_v1_1")
    if map_status.startswith("available"):
        paths["country_bwer_choropleth_png"] = figures / "country_bwer_choropleth_v1_1.png"
        paths["country_bwer_choropleth_pdf"] = figures / "country_bwer_choropleth_v1_1.pdf"
    else:
        paths["country_bwer_ranked_barplot_png"] = figures / "country_bwer_ranked_barplot_v1_1.png"
        paths["country_bwer_ranked_barplot_pdf"] = figures / "country_bwer_ranked_barplot_v1_1.pdf"
    return paths, map_status


def build_fmow_social_spatial_v1_1(
    *,
    input_dir: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    indicator_csv: Path | None = None,
    min_support: int = 20,
    unified_v2_dir: Path = DEFAULT_UNIFIED_V2,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    country_rows = read_csv_rows(input_dir / "fmow_country_risk_summary.csv")
    indicators, indicator_status = load_or_fetch_indicators(indicator_csv)
    joined = join_indicators(country_rows, indicators, indicator_status)
    associations = association_rows(joined, min_support=min_support)
    figure_paths, map_status = write_figures(joined, output)
    caveats = caveat_rows(indicator_status, map_status)

    artifacts = {
        "fmow_social_indicator_join_v1_1": output / "fmow_social_indicator_join_v1_1.csv",
        "fmow_risk_indicator_association_v1_1": output / "fmow_risk_indicator_association_v1_1.csv",
        "fmow_social_spatial_report_v1_1": output / "fmow_social_spatial_report_v1_1.md",
        "fmow_social_spatial_caveats_v1_1": output / "fmow_social_spatial_caveats_v1_1.csv",
    }
    write_csv(artifacts["fmow_social_indicator_join_v1_1"], joined)
    write_csv(artifacts["fmow_risk_indicator_association_v1_1"], associations)
    write_csv(artifacts["fmow_social_spatial_caveats_v1_1"], caveats)
    matched = sum(1 for row in joined if row.get("indicator_status") == "matched")
    available_assoc = [row for row in associations if row.get("association_status") == "available"]
    artifacts["fmow_social_spatial_report_v1_1"].write_text(
        "# fMoW social-spatial interpretation v1.1\n\n"
        "This is a post-hoc exploratory deployment-risk interpretation using saved fMoW outputs and World Bank country indicators. No training or inference was run.\n\n"
        "## Indicator join\n\n"
        f"- Indicator source status: {indicator_status}.\n"
        f"- Joined country-scenario rows with matched indicators: {matched} / {len(joined)}.\n"
        f"- Support filter for interpretation: support_count >= {min_support}.\n\n"
        "## Association scope\n\n"
        f"- Available association rows: {len(available_assoc)} / {len(associations)}.\n"
        "- Pearson correlations are exploratory associations between country-level indicators and fMoW country mean risk; they are not causal claims.\n"
        f"- Map status: {map_status}.\n",
        encoding="utf-8",
    )
    ensure_dir(unified_v2_dir)
    unified_summary = unified_v2_dir / "fmow_social_spatial_indicator_association_v1_1_summary.csv"
    write_csv(
        unified_summary,
        [
            {
                "experiment_id": "fmow_sentinel_step3_social_spatial_v1_1",
                "indicator_status": indicator_status,
                "matched_country_scenario_rows": matched,
                "total_country_scenario_rows": len(joined),
                "available_association_rows": len(available_assoc),
                "total_association_rows": len(associations),
                "map_status": map_status,
                "claim_scope": "exploratory deployment-risk association; not causal",
            }
        ],
    )
    paper_report = unified_v2_dir / "paper_ready_fmow_selective_audit_report.md"
    previous = paper_report.read_text(encoding="utf-8") if paper_report.exists() else "# Paper-ready fMoW selective audit report\n"
    section = (
        "## Social-spatial interpretation v1.1\n\n"
        f"- World Bank indicator status: {indicator_status}.\n"
        f"- Matched country-scenario rows: {matched} / {len(joined)}.\n"
        f"- Available exploratory association rows: {len(available_assoc)} / {len(associations)}.\n"
        f"- Map status: {map_status}.\n"
        "- Country-level associations use support_count >= 20 and must be framed as exploratory, non-causal deployment-risk interpretation.\n"
    )
    marker = "## Social-spatial interpretation v1.1"
    if marker in previous:
        previous = previous.split(marker, 1)[0].rstrip() + "\n\n" + section
    else:
        previous = previous.rstrip() + "\n\n" + section
    paper_report.write_text(previous, encoding="utf-8")
    artifacts["unified_v2_fmow_social_spatial_indicator_association_v1_1_summary"] = unified_summary
    artifacts["unified_v2_paper_ready_report"] = paper_report
    artifacts.update({f"figure_{key}": value for key, value in figure_paths.items()})
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fMoW social-spatial interpretation v1.1 with World Bank indicators.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--indicator-csv", type=Path)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--unified-v2-dir", type=Path, default=DEFAULT_UNIFIED_V2)
    args = parser.parse_args()
    artifacts = build_fmow_social_spatial_v1_1(input_dir=args.input_dir, output_dir=args.out, indicator_csv=args.indicator_csv, min_support=args.min_support, unified_v2_dir=args.unified_v2_dir)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
