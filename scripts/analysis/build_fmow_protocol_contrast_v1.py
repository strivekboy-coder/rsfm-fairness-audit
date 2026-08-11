from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


DEFAULT_OUTPUT = Path("outputs/fmow_protocol_contrast_v1")
ASSET_CONFIG = PROJECT_ROOT / "configs" / "analysis" / "fmow_asset_sources.json"


def _canonical_fmow_from_config(path: Path = ASSET_CONFIG) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("canonical_fmow_dir")
    if not value:
        raise ValueError(f"Missing canonical_fmow_dir in {path}")
    resolved = Path(str(value))
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


CANONICAL_FMOW = _canonical_fmow_from_config()


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_bwer_slices(text: Any) -> dict[str, float]:
    output = {}
    for item in str(text or "").split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        output[key] = _float(value)
    return output


def _registry_rows() -> list[dict[str, Any]]:
    return [
        {
            "run_id": "resnet50_13band",
            "model_family": "resnet",
            "protocol": "location_disjoint",
            "protocol_status": "valid_benchmark_formal_partial",
            "accuracy": 0.20002233638597275,
            "balanced_accuracy": 0.18405095743992678,
            "macro_f1": 0.1724502860076166,
            "top5_accuracy": 0.45175340629886085,
            "country_raw_bwer": 0.17361207464336303,
            "country_class_standardised_bwer": 0.14234836453364907,
            "source": "final_step3 comparison_summary.csv / Drive-real audit",
        },
        {
            "run_id": "dofa_scaled10000",
            "model_family": "dofa",
            "protocol": "location_disjoint",
            "protocol_status": "valid_benchmark_formal_partial",
            "accuracy": 0.17768595041322313,
            "balanced_accuracy": 0.1790738605980416,
            "macro_f1": 0.16865861362390494,
            "top5_accuracy": "",
            "country_raw_bwer": 0.16141538857738702,
            "country_class_standardised_bwer": 0.1269780950367737,
            "source": "final_step3 comparison_summary.csv / Drive-real audit",
        },
        {
            "run_id": "resnet50_random_split_sanity",
            "model_family": "resnet",
            "protocol": "random_split",
            "protocol_status": "sanity_protocol_contrast",
            "accuracy": 0.7118888888888889,
            "balanced_accuracy": 0.6672910350320261,
            "macro_f1": 0.678352105923704,
            "top5_accuracy": 0.8354444444444444,
            "country_raw_bwer": 0.2517475897959062,
            "country_class_standardised_bwer": 0.257785,
            "source": "random_split_resnet50_16epoch sanity outputs / registry",
        },
        {
            "run_id": "dofa_random_split_sanity",
            "model_family": "dofa",
            "protocol": "random_split",
            "protocol_status": "sanity_protocol_contrast",
            "accuracy": 0.38433333333333336,
            "balanced_accuracy": 0.3845599647290616,
            "macro_f1": 0.3798401872360772,
            "top5_accuracy": "",
            "country_raw_bwer": 0.205189,
            "country_class_standardised_bwer": 0.176990,
            "source": "dofa_random_split_sanity outputs / registry",
        },
    ]


def _enrich_from_small_files(rows: list[dict[str, Any]], canonical_dir: Path) -> None:
    path = canonical_dir / "random_split_resnet50_16epoch_random_split_vs_location_disjoint_summary.csv"
    if not path.exists():
        return
    for row in read_csv_rows(path):
        if row.get("protocol") == "random_split_sanity":
            target = next(item for item in rows if item["run_id"] == "resnet50_random_split_sanity")
        elif row.get("protocol") == "location_disjoint":
            target = next(item for item in rows if item["run_id"] == "resnet50_13band")
        else:
            continue
        target["accuracy"] = _float(row.get("accuracy"))
        target["balanced_accuracy"] = _float(row.get("balanced_accuracy"))
        target["macro_f1"] = _float(row.get("macro_f1"))
        target["top5_accuracy"] = _float(row.get("top5_accuracy"))
        bwer = _parse_bwer_slices(row.get("bwer_slices"))
        target["country_raw_bwer"] = bwer.get("country", target.get("country_raw_bwer"))


def _contrast_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family in ("resnet", "dofa"):
        random_row = next(row for row in summary if row["model_family"] == family and row["protocol"] == "random_split")
        location_row = next(row for row in summary if row["model_family"] == family and row["protocol"] == "location_disjoint")
        rows.append(
            {
                "model_family": family,
                "random_accuracy": random_row.get("accuracy"),
                "location_disjoint_accuracy": location_row.get("accuracy"),
                "accuracy_drop_random_to_location": _float(random_row.get("accuracy")) - _float(location_row.get("accuracy")),
                "random_country_raw_bwer": random_row.get("country_raw_bwer"),
                "location_country_raw_bwer": location_row.get("country_raw_bwer"),
                "bwer_delta_random_minus_location": _float(random_row.get("country_raw_bwer")) - _float(location_row.get("country_raw_bwer")),
                "interpretation": "random split is a sanity/protocol contrast only; location-disjoint is the formal deployment protocol",
            }
        )
    return rows


def _claims(summary: list[dict[str, Any]], contrast: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "claim": "Random-split accuracy is much higher than location-disjoint accuracy for fMoW.",
            "support": "supported_protocol_contrast",
            "evidence": "; ".join(f"{row['model_family']} drop={row['accuracy_drop_random_to_location']:.3f}" for row in contrast),
            "scope": "sanity/protocol contrast; not deployment evidence",
        },
        {
            "claim": "Location-disjoint fMoW remains the formal deployment geography protocol.",
            "support": "supported",
            "evidence": "final_step3 location-disjoint outputs and Drive-real audit contract",
            "scope": "formal fMoW deployment evidence",
        },
        {
            "claim": "Random split can give an overly optimistic view of aggregate performance.",
            "support": "supported",
            "evidence": "random-split accuracy exceeds location-disjoint accuracy for ResNet50 and DOFA",
            "scope": "protocol claim only",
        },
    ]


def _caveats() -> list[dict[str, Any]]:
    return [
        {"category": "protocol_scope", "caveat": "Random split is sanity/protocol contrast, not formal deployment evidence."},
        {"category": "deployment_protocol", "caveat": "Location-disjoint Step3 is the formal fMoW deployment geography audit."},
        {"category": "preprocessing", "caveat": "Formal DOFA uses scaled10000 preprocessing; unscaled/debug DOFA is excluded."},
        {"category": "selective_scope", "caveat": "Selective and calibrated-threshold diagnostics are location-disjoint only unless random split selective tables are explicitly generated."},
    ]


def _write_figures(output: Path, summary: list[dict[str, Any]]) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    figures = ensure_dir(output / "figures")
    paths = {
        "random_vs_location_accuracy_png": figures / "random_vs_location_accuracy.png",
        "random_vs_location_accuracy_pdf": figures / "random_vs_location_accuracy.pdf",
        "random_vs_location_bwer_png": figures / "random_vs_location_bwer.png",
        "random_vs_location_bwer_pdf": figures / "random_vs_location_bwer.pdf",
        "protocol_contrast_aggregate_vs_bwer_png": figures / "protocol_contrast_aggregate_vs_bwer.png",
        "protocol_contrast_aggregate_vs_bwer_pdf": figures / "protocol_contrast_aggregate_vs_bwer.pdf",
    }
    labels = ["ResNet50", "DOFA"]
    colors = {"random_split": "#8C6D31", "location_disjoint": "#2F5DA8"}
    x = np.arange(len(labels))
    width = 0.36
    for metric, title, ylabel, png, pdf in [
        ("accuracy", "Random split vs location-disjoint accuracy", "Accuracy", paths["random_vs_location_accuracy_png"], paths["random_vs_location_accuracy_pdf"]),
        ("country_raw_bwer", "Random split vs location-disjoint country Raw-BWER", "Country Raw-BWER", paths["random_vs_location_bwer_png"], paths["random_vs_location_bwer_pdf"]),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for offset, protocol in [(-width / 2, "random_split"), (width / 2, "location_disjoint")]:
            vals = [_float(next(row.get(metric) for row in summary if row["model_family"] == family and row["protocol"] == protocol)) for family in ("resnet", "dofa")]
            ax.bar(x + offset, vals, width=width, label=protocol.replace("_", " "), color=colors[protocol])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for protocol in ("random_split", "location_disjoint"):
        items = [row for row in summary if row["protocol"] == protocol]
        ax.scatter([_float(row["accuracy"]) for row in items], [_float(row["country_raw_bwer"]) for row in items], s=70, color=colors[protocol], label=protocol.replace("_", " "))
        for row in items:
            ax.text(_float(row["accuracy"]), _float(row["country_raw_bwer"]), row["model_family"], fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Country Raw-BWER")
    ax.set_title("Protocol contrast: aggregate vs geography tail risk")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(paths["protocol_contrast_aggregate_vs_bwer_png"], dpi=180)
    fig.savefig(paths["protocol_contrast_aggregate_vs_bwer_pdf"])
    plt.close(fig)
    return paths


def build_fmow_protocol_contrast(output_dir: Path = DEFAULT_OUTPUT, canonical_dir: Path = CANONICAL_FMOW) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    summary = _registry_rows()
    _enrich_from_small_files(summary, canonical_dir)
    contrast = _contrast_rows(summary)
    artifacts = {
        "fmow_random_vs_location_disjoint_summary": output / "fmow_random_vs_location_disjoint_summary.csv",
        "fmow_protocol_contrast_bwer_summary": output / "fmow_protocol_contrast_bwer_summary.csv",
        "fmow_protocol_contrast_claims": output / "fmow_protocol_contrast_claims.csv",
        "fmow_protocol_contrast_caveats": output / "fmow_protocol_contrast_caveats.csv",
        "fmow_protocol_contrast_report": output / "fmow_protocol_contrast_report.md",
    }
    write_csv(artifacts["fmow_random_vs_location_disjoint_summary"], contrast)
    write_csv(artifacts["fmow_protocol_contrast_bwer_summary"], summary)
    write_csv(artifacts["fmow_protocol_contrast_claims"], _claims(summary, contrast))
    write_csv(artifacts["fmow_protocol_contrast_caveats"], _caveats())
    artifacts["fmow_protocol_contrast_report"].write_text(
        "# fMoW random-split protocol contrast v1\n\n"
        "This is a post-hoc protocol contrast. No model training or inference was run.\n\n"
        "## Findings\n\n"
        + "\n".join(f"- {row['model_family']}: random accuracy {row['random_accuracy']:.3f} vs location-disjoint {row['location_disjoint_accuracy']:.3f}; accuracy drop {row['accuracy_drop_random_to_location']:.3f}." for row in contrast)
        + "\n\nRandom split is a sanity/protocol contrast only. Location-disjoint remains the formal deployment geography protocol.\n",
        encoding="utf-8",
    )
    artifacts.update({f"figure_{key}": value for key, value in _write_figures(output, summary).items()})
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canonical-dir", type=Path, default=CANONICAL_FMOW)
    args = parser.parse_args()
    for name, path in build_fmow_protocol_contrast(args.out, args.canonical_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
