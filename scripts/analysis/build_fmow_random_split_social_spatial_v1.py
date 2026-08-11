from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


DEFAULT_OUTPUT = Path("outputs/fmow_random_split_social_spatial_v1")
ASSET_CONFIG = PROJECT_ROOT / "configs" / "analysis" / "fmow_asset_sources.json"


def _canonical_fmow_from_config(path: Path = ASSET_CONFIG) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("canonical_fmow_dir")
    if not value:
        raise ValueError(f"Missing canonical_fmow_dir in {path}")
    resolved = Path(str(value))
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


CANONICAL_FMOW = _canonical_fmow_from_config()
DEFAULT_INDICATOR_JOIN = Path("outputs/fmow_social_spatial_interpretation_v1/fmow_social_indicator_join_v1_1.csv")
DEFAULT_UNIFIED_V3 = Path("outputs/unified_paper_package_v3")


RUN_FILES = {
    "resnet50_random_split_sanity": "random_split_resnet50_16epoch_bwer_bwer_by_slice.csv",
    "dofa_random_split_sanity": "dofa_random_split_sanity_bwer_bwer_by_slice.csv",
}


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    clean = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(clean) < 3:
        return float("nan")
    x_mean = sum(x for x, _ in clean) / len(clean)
    y_mean = sum(y for _, y in clean) / len(clean)
    num = sum((x - x_mean) * (y - y_mean) for x, y in clean)
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x, _ in clean))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for _, y in clean))
    return num / (den_x * den_y) if den_x and den_y else float("nan")


def _indicator_map(indicator_join: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv_rows(indicator_join) if indicator_join.exists() else []
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        iso3 = row.get("iso3") or row.get("country") or row.get("slice_value")
        if not iso3 or iso3 in output or row.get("indicator_status") != "matched":
            continue
        output[iso3] = {
            "country_name": row.get("country_name", ""),
            "world_bank_region": row.get("world_bank_region", ""),
            "income_group": row.get("income_group", ""),
            "gdp_per_capita": row.get("gdp_per_capita", ""),
            "gdp_per_capita_year": row.get("gdp_per_capita_year", ""),
            "population_density": row.get("population_density", ""),
            "population_density_year": row.get("population_density_year", ""),
            "urban_population_share": row.get("urban_population_share", ""),
            "urban_population_share_year": row.get("urban_population_share_year", ""),
        }
    return output


def _country_rows(canonical_dir: Path, min_support: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, filename in RUN_FILES.items():
        path = canonical_dir / filename
        if not path.exists():
            rows.append(
                {
                    "run_id": run_id,
                    "protocol": "random_split_sanity",
                    "slice_variable": "country",
                    "country": "",
                    "support_count": "",
                    "mean_risk": "",
                    "support_ok": False,
                    "data_status": f"missing:{path}",
                }
            )
            continue
        for row in read_csv_rows(path):
            if row.get("slice_variable") != "country":
                continue
            support = _float(row.get("sample_count") or row.get("n_units"), 0.0)
            rows.append(
                {
                    "run_id": run_id,
                    "protocol": "random_split_sanity",
                    "evidence_scope": "sanity/protocol contrast only; not deployment evidence",
                    "slice_variable": "country",
                    "country": row.get("slice_value", ""),
                    "iso3": row.get("slice_value", ""),
                    "support_count": int(support),
                    "mean_risk": _float(row.get("raw_risk")),
                    "balanced_risk": _float(row.get("balanced_risk")),
                    "raw_score": _float(row.get("raw_score")),
                    "support_ok": support >= min_support and str(row.get("is_valid_slice")).lower() == "true",
                    "support_filter": f"support_count >= {min_support}",
                    "is_tail_slice": row.get("is_tail_slice", ""),
                    "rank_by_risk": row.get("rank_by_risk", ""),
                    "data_status": "available",
                }
            )
    return rows


def _region_rows(canonical_dir: Path, min_support: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, filename in RUN_FILES.items():
        path = canonical_dir / filename
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            if row.get("slice_variable") != "region":
                continue
            support = _float(row.get("sample_count") or row.get("n_units"), 0.0)
            rows.append(
                {
                    "run_id": run_id,
                    "protocol": "random_split_sanity",
                    "region": row.get("slice_value", ""),
                    "support_count": int(support),
                    "mean_risk": _float(row.get("raw_risk")),
                    "support_ok": support >= min_support and str(row.get("is_valid_slice")).lower() == "true",
                    "support_filter": f"support_count >= {min_support}",
                    "evidence_scope": "sanity/protocol contrast only; not deployment evidence",
                }
            )
    return rows


def _join_indicators(country_rows: Sequence[Mapping[str, Any]], indicators: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    joined = []
    for row in country_rows:
        iso3 = str(row.get("iso3") or row.get("country") or "")
        indicator = indicators.get(iso3, {})
        joined.append(
            {
                **dict(row),
                "indicator_status": "matched" if indicator else "unmatched_iso3",
                "country_name": indicator.get("country_name", ""),
                "world_bank_region": indicator.get("world_bank_region", ""),
                "income_group": indicator.get("income_group", ""),
                "gdp_per_capita": indicator.get("gdp_per_capita", ""),
                "gdp_per_capita_year": indicator.get("gdp_per_capita_year", ""),
                "population_density": indicator.get("population_density", ""),
                "population_density_year": indicator.get("population_density_year", ""),
                "urban_population_share": indicator.get("urban_population_share", ""),
                "urban_population_share_year": indicator.get("urban_population_share_year", ""),
            }
        )
    return joined


def _associations(joined: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for run_id in sorted({str(row.get("run_id")) for row in joined if row.get("run_id")}):
        base = [row for row in joined if row.get("run_id") == run_id and bool(row.get("support_ok")) and row.get("indicator_status") == "matched"]
        for indicator in ("gdp_per_capita", "population_density", "urban_population_share"):
            pairs = [(_float(row.get(indicator)), _float(row.get("mean_risk")), _float(row.get("support_count"))) for row in base]
            clean = [(x, y, w) for x, y, w in pairs if not math.isnan(x) and not math.isnan(y)]
            n = len(clean)
            output.append(
                {
                    "run_id": run_id,
                    "protocol": "random_split_sanity",
                    "indicator": indicator,
                    "n_countries": n,
                    "pearson_r": _pearson([x for x, _, _ in clean], [y for _, y, _ in clean]) if n >= 3 else "",
                    "support_weighted_mean_risk": (sum(y * w for _, y, w in clean) / sum(w for _, _, w in clean)) if clean and sum(w for _, _, w in clean) else "",
                    "support_filter": next((str(row.get("support_filter")) for row in base), "support_count >= 20"),
                    "association_status": "available" if n >= 3 else "unavailable_insufficient_indicator_matches",
                    "claim_scope": "random split sanity/protocol contrast only; exploratory association; not causal and not deployment evidence",
                }
            )
    return output


def _write_figures(output: Path, joined: Sequence[Mapping[str, Any]], associations: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures = ensure_dir(output / "figures")
    paths: dict[str, Path] = {}

    def save(fig: Any, name: str) -> None:
        png = figures / f"{name}.png"
        pdf = figures / f"{name}.pdf"
        fig.tight_layout()
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
        paths[f"{name}_png"] = png
        paths[f"{name}_pdf"] = pdf

    for indicator, label in [
        ("gdp_per_capita", "GDP per capita"),
        ("population_density", "Population density"),
        ("urban_population_share", "Urban population share"),
    ]:
        fig, ax = plt.subplots(figsize=(6.3, 4.0))
        for run_id, color in [("resnet50_random_split_sanity", "#2F5DA8"), ("dofa_random_split_sanity", "#6B8E23")]:
            clean = [
                (_float(row.get(indicator)), _float(row.get("mean_risk")), _float(row.get("support_count")))
                for row in joined
                if row.get("run_id") == run_id and bool(row.get("support_ok")) and row.get("indicator_status") == "matched"
            ]
            clean = [(x, y, s) for x, y, s in clean if not math.isnan(x) and not math.isnan(y)]
            ax.scatter([x for x, _, _ in clean], [y for _, y, _ in clean], s=[max(18, min(120, s / 2)) for _, _, s in clean], alpha=0.65, label=run_id.replace("_random_split_sanity", ""), color=color)
        ax.set_xlabel(label)
        ax.set_ylabel("Country mean risk")
        ax.set_title(f"Random-split country risk vs {label}")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, f"random_split_bwer_vs_{indicator}_scatter")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    labels = [f"{row.get('run_id','').replace('_random_split_sanity','')}\n{row.get('indicator')}" for row in associations]
    vals = [_float(row.get("pearson_r"), 0.0) for row in associations]
    ax.bar(range(len(vals)), vals, color="#5E6C84")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title("Random-split social indicator associations")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "random_split_indicator_association_summary")
    return paths


def build_fmow_random_split_social_spatial(
    output_dir: Path = DEFAULT_OUTPUT,
    canonical_dir: Path = CANONICAL_FMOW,
    indicator_join: Path = DEFAULT_INDICATOR_JOIN,
    min_support: int = 20,
    unified_v3_dir: Path | None = DEFAULT_UNIFIED_V3,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    country = _country_rows(canonical_dir, min_support)
    region = _region_rows(canonical_dir, min_support)
    joined = _join_indicators(country, _indicator_map(indicator_join))
    associations = _associations(joined)
    caveats = [
        {"category": "protocol_scope", "caveat": "Random split is sanity/protocol contrast only and must not be used as deployment evidence."},
        {"category": "association_scope", "caveat": "Country-level associations are exploratory and non-causal."},
        {"category": "support_filter", "caveat": f"Country associations use support_count >= {min_support}."},
        {"category": "indicator_source", "caveat": f"Indicators reused from {indicator_join}; no network fetch required."},
    ]
    artifacts = {
        "fmow_random_split_country_risk_summary": output / "fmow_random_split_country_risk_summary.csv",
        "fmow_random_split_region_risk_summary": output / "fmow_random_split_region_risk_summary.csv",
        "fmow_random_split_social_indicator_join": output / "fmow_random_split_social_indicator_join.csv",
        "fmow_random_split_risk_indicator_association": output / "fmow_random_split_risk_indicator_association.csv",
        "fmow_random_split_social_spatial_caveats": output / "fmow_random_split_social_spatial_caveats.csv",
        "fmow_random_split_social_spatial_report": output / "fmow_random_split_social_spatial_report.md",
    }
    write_csv(artifacts["fmow_random_split_country_risk_summary"], country)
    write_csv(artifacts["fmow_random_split_region_risk_summary"], region)
    write_csv(artifacts["fmow_random_split_social_indicator_join"], joined)
    write_csv(artifacts["fmow_random_split_risk_indicator_association"], associations)
    write_csv(artifacts["fmow_random_split_social_spatial_caveats"], caveats)
    matched = sum(1 for row in joined if row.get("indicator_status") == "matched")
    max_abs = max((_float(row.get("pearson_r"), 0.0) for row in associations if row.get("association_status") == "available"), key=abs, default=0.0)
    artifacts["fmow_random_split_social_spatial_report"].write_text(
        "# fMoW random-split social-spatial interpretation v1\n\n"
        "This is a post-hoc sanity/protocol-contrast interpretation. No training or inference was run.\n\n"
        f"- Indicator matched country rows: {matched} / {len(joined)}.\n"
        f"- Support filter: support_count >= {min_support}.\n"
        f"- Largest absolute Pearson correlation among available random-split country associations: {max_abs:.3f}.\n"
        "- These associations are exploratory, non-causal, and not deployment evidence because the split is random.\n",
        encoding="utf-8",
    )
    artifacts.update({f"figure_{key}": value for key, value in _write_figures(output, joined, associations).items()})
    if unified_v3_dir is not None:
        ensure_dir(unified_v3_dir)
        unified_summary = unified_v3_dir / "fmow_random_split_social_spatial_summary_v3.csv"
        write_csv(
            unified_summary,
            [
                {
                    "experiment_id": "fmow_random_split_social_spatial_v1",
                    "protocol": "random_split_sanity",
                    "matched_country_rows": matched,
                    "association_rows": len(associations),
                    "largest_abs_pearson_r": max_abs,
                    "claim_scope": "sanity/protocol contrast only; not deployment evidence",
                    "source_dir": str(output),
                }
            ],
        )
        artifacts["unified_v3_fmow_random_split_social_spatial_summary"] = unified_summary
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fMoW random-split social-spatial protocol-contrast assets.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canonical-dir", type=Path, default=CANONICAL_FMOW)
    parser.add_argument("--indicator-join", type=Path, default=DEFAULT_INDICATOR_JOIN)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--unified-v3-dir", type=Path, default=DEFAULT_UNIFIED_V3)
    args = parser.parse_args()
    for name, path in build_fmow_random_split_social_spatial(args.out, args.canonical_dir, args.indicator_join, args.min_support, args.unified_v3_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
