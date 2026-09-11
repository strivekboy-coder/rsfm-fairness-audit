"""Build paired, multi-label reBEN shift-error structure from frozen audits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = ["sample_id", "class_index", "class_label", "label_true", "label_prediction", "country"]


def load_paired(id_path: Path, shifted_path: Path, *, label_count: int = 19, chunk_samples: int = 10000):
    chunksize = label_count * chunk_samples
    truths, id_preds, shifted_preds, countries = [], [], [], []
    labels: list[str] | None = None
    id_iter = pd.read_csv(id_path, usecols=REQUIRED, chunksize=chunksize)
    shifted_iter = pd.read_csv(shifted_path, usecols=REQUIRED, chunksize=chunksize)
    chunk_count = 0
    while True:
        left = next(id_iter, None)
        right = next(shifted_iter, None)
        if left is None or right is None:
            if left is not None or right is not None:
                raise ValueError("ID and shifted audit tables have different row counts.")
            break
        chunk_count += 1
        if len(left) != len(right) or len(left) % label_count:
            raise ValueError(f"Unaligned paired chunk {chunk_count}: {len(left)} vs {len(right)} rows.")
        left_key = left[["sample_id", "class_index"]].astype(str).to_numpy()
        right_key = right[["sample_id", "class_index"]].astype(str).to_numpy()
        if not np.array_equal(left_key, right_key):
            raise ValueError(f"Paired keys differ in chunk {chunk_count}; refusing positional merge.")
        n = len(left) // label_count
        sample_grid = left.sample_id.astype(str).to_numpy().reshape(n, label_count)
        if not np.all(sample_grid == sample_grid[:, :1]):
            raise ValueError(f"Rows are not contiguous sample-by-label blocks in chunk {chunk_count}.")
        class_grid = left.class_index.to_numpy().reshape(n, label_count)
        expected = np.arange(label_count)
        if not np.all(class_grid == expected):
            raise ValueError(f"Class order is not 0..{label_count - 1} in chunk {chunk_count}.")
        if labels is None:
            labels = left.class_label.astype(str).to_numpy()[:label_count].tolist()
        truths.append(left.label_true.to_numpy(dtype=np.int8).reshape(n, label_count))
        id_preds.append(left.label_prediction.to_numpy(dtype=np.int8).reshape(n, label_count))
        shifted_preds.append(right.label_prediction.to_numpy(dtype=np.int8).reshape(n, label_count))
        countries.extend(left.country.astype(str).to_numpy()[::label_count].tolist())
        print(f"[reBEN] paired chunk {chunk_count}: {n:,} samples", flush=True)
    if not truths or labels is None:
        raise ValueError("No paired label-audit rows were read.")
    return np.concatenate(truths), np.concatenate(id_preds), np.concatenate(shifted_preds), labels, np.asarray(countries)


def matrix_rows(matrix_id: np.ndarray, matrix_shift: np.ndarray, labels: list[str], fn_id: np.ndarray, fn_shift: np.ndarray):
    rows = []
    for i, target in enumerate(labels):
        for j, false_positive in enumerate(labels):
            if i == j:
                continue
            id_rate = matrix_id[i, j] / fn_id[i] if fn_id[i] else np.nan
            shift_rate = matrix_shift[i, j] / fn_shift[i] if fn_shift[i] else np.nan
            rows.append({
                "missed_target_label": target,
                "cooccurring_false_positive_label": false_positive,
                "id_count": int(matrix_id[i, j]), "shift_count": int(matrix_shift[i, j]),
                "id_conditional_rate": id_rate, "shift_conditional_rate": shift_rate,
                "delta_conditional_rate": shift_rate - id_rate if np.isfinite(id_rate) and np.isfinite(shift_rate) else np.nan,
                "interpretation": "co-error association; not an exclusive-class substitution",
            })
    return rows


def plot_summary(summary: pd.DataFrame, matrix: pd.DataFrame, output: Path, model: str) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
    ordered = summary.sort_values("delta_fnr_mean", ascending=True).reset_index(drop=True)
    labels = ordered.class_label.astype(str).tolist()
    coerror = matrix.groupby(["missed_target_label", "cooccurring_false_positive_label"], as_index=False).delta_conditional_rate.mean()
    pivot = coerror.pivot(index="missed_target_label", columns="cooccurring_false_positive_label", values="delta_conditional_rate").reindex(index=labels, columns=labels)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.45]}, constrained_layout=True)
    for ax, mean, sd, title, color in (
        (axes[0], "delta_fnr_mean", "delta_fnr_sd", "Change in false-negative rate", "#D55E00"),
        (axes[1], "delta_fpr_mean", "delta_fpr_sd", "Change in false-positive rate", "#0072B2"),
    ):
        y = np.arange(len(ordered))
        ax.axvline(0, color="#6F7780", lw=.8)
        ax.errorbar(ordered[mean], y, xerr=ordered[sd].fillna(0), fmt="o", color=color, ecolor=color, markersize=3, capsize=2, lw=.8)
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.set_xlabel("S1 shift minus S2 ID")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#E2E7EB", lw=.5)
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) if np.isfinite(pivot.to_numpy()).any() else 1.0
    image = axes[2].imshow(pivot, cmap="PuOr_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[2].set_xticks(np.arange(len(labels)), labels, rotation=90)
    axes[2].set_yticks(np.arange(len(labels)), labels)
    axes[2].set_xlabel("Co-occurring false-positive label")
    axes[2].set_ylabel("Missed target label")
    axes[2].set_title("Change in conditional co-error", loc="left", fontweight="bold", fontsize=8.5)
    colorbar = fig.colorbar(image, ax=axes[2], fraction=.045, pad=.03)
    colorbar.set_label("S1 shift minus S2 ID")
    for index, ax in enumerate(axes):
        ax.text(-.16, 1.04, chr(97 + index), transform=ax.transAxes, fontweight="bold", fontsize=10)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"reben_shift_error_structure_{model.lower()}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 73, 101])
    parser.add_argument("--label-count", type=int, default=19)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from rsfm_fairness_audit.paper_supplementary import summarize_paired_multilabel_arrays, write_csv, write_json

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    all_label_rows, all_country_rows, all_matrix_rows = [], [], []
    for seed in args.seeds:
        seed_dir = args.paired_root / f"seed_{seed}"
        id_path, shifted_path = seed_dir / "id_label_audit.csv", seed_dir / "ood_label_audit.csv"
        if not id_path.is_file() or not shifted_path.is_file():
            raise FileNotFoundError(f"Missing frozen paired audits below {seed_dir}")
        truth, id_pred, shifted_pred, labels, country = load_paired(id_path, shifted_path, label_count=args.label_count)
        label_rows, matrix_id, matrix_shift = summarize_paired_multilabel_arrays(truth, id_pred, shifted_pred, labels)
        fn_id = np.sum((truth == 1) & (id_pred == 0), axis=0)
        fn_shift = np.sum((truth == 1) & (shifted_pred == 0), axis=0)
        for row in label_rows:
            all_label_rows.append({"model": args.model, "seed": seed, "scope": "all", **row})
        for value in sorted(set(country)):
            mask = country == value
            country_rows, _, _ = summarize_paired_multilabel_arrays(truth[mask], id_pred[mask], shifted_pred[mask], labels)
            for row in country_rows:
                all_country_rows.append({"model": args.model, "seed": seed, "country": value, **row})
        for row in matrix_rows(matrix_id, matrix_shift, labels, fn_id, fn_shift):
            all_matrix_rows.append({"model": args.model, "seed": seed, **row})
    write_csv(output / "reben_label_error_transitions.csv", all_label_rows)
    write_csv(output / "reben_country_label_error_transitions.csv", all_country_rows)
    write_csv(output / "reben_conditional_coerror_matrix.csv", all_matrix_rows)
    frame = pd.DataFrame(all_label_rows)
    summary = frame.groupby(["model", "class_label"], as_index=False).agg(
        seed_count=("seed", "nunique"), delta_fnr_mean=("delta_fnr", "mean"), delta_fnr_sd=("delta_fnr", "std"),
        delta_fpr_mean=("delta_fpr", "mean"), delta_fpr_sd=("delta_fpr", "std"),
        new_error_rate_mean=("new_error_rate", "mean"), persistent_error_rate_mean=("persistent_error_rate", "mean"),
    )
    summary.to_csv(output / "reben_label_error_transition_summary.csv", index=False)
    plot_summary(summary, pd.DataFrame(all_matrix_rows), output, args.model)
    write_json(output / "manifest.json", {
        "schema": "geobwer.paper_supplement.reben_shift_error.v1", "status": "complete",
        "model": args.model, "seeds": args.seeds, "paired_root": str(args.paired_root),
        "same_sample_pairing_required": True, "exclusive_multiclass_confusion_claim": False,
        "frozen_outputs_modified": False, "model_rerun": False,
    })
    print(json.dumps({"output_dir": str(output), "label_rows": len(all_label_rows), "country_label_rows": len(all_country_rows)}, indent=2))


if __name__ == "__main__":
    main()
