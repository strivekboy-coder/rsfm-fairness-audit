from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(row: Mapping[str, str], data_root: Path | None) -> Path:
    candidates: list[Path] = []
    for key in ("image_path", "archive_path", "target_path", "extracted_image_path", "raster_path", "path", "extracted_path"):
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        path = Path(value)
        if path.is_absolute():
            if path.exists():
                return path
            continue
        if data_root is not None:
            candidates.append(data_root / path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    raise FileNotFoundError("row is missing extracted_path/image_path/raster_path/path")


def read_raster_shape(path: Path) -> tuple[int, int, int | str, str]:
    try:
        import rasterio  # type: ignore

        with rasterio.open(path) as src:
            return int(src.width), int(src.height), int(src.count), "rasterio"
    except Exception as rasterio_error:
        try:
            import tifffile  # type: ignore

            with tifffile.TiffFile(path) as tif:
                shape = tuple(int(value) for value in tif.series[0].shape)
            if len(shape) == 2:
                height, width = shape
                bands: int | str = 1
            elif len(shape) == 3 and shape[0] in {1, 3, 4, 9, 10, 12, 13}:
                bands, height, width = shape
            elif len(shape) == 3:
                height, width, bands = shape
            else:
                height = int(shape[-2])
                width = int(shape[-1])
                bands = "x".join(str(value) for value in shape[:-2])
            return int(width), int(height), bands, "tifffile"
        except Exception as tifffile_error:
            raise RuntimeError(f"Could not read raster shape with rasterio ({rasterio_error}) or tifffile ({tifffile_error})") from tifffile_error


def sample_id_for_row(row: Mapping[str, str], index: int) -> str:
    value = str(row.get("sample_id", "") or "").strip()
    if value:
        return value
    image_id = str(row.get("image_id", "") or "").strip()
    if image_id:
        return image_id
    return f"row_{index}"


def analyze_manifest(manifest: Path, data_root: Path | None, output_dir: Path, *, progress_every: int = 1000) -> dict[str, Path]:
    rows = read_csv(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    print(f"[stage] Reading patch sizes manifest={manifest} rows={len(rows)}")
    for index, row in enumerate(rows, start=1):
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(rows)):
            print(f"[progress] patch-size diagnostics {index}/{len(rows)}")
        sample_id = sample_id_for_row(row, index)
        try:
            path = resolve_image_path(row, data_root)
            width, height, band_count, reader = read_raster_shape(path)
            area = width * height
            aspect = width / height if height else math.nan
            status = "ok"
            warning = ""
        except Exception as exc:
            path = Path(str(row.get("extracted_path") or row.get("image_path") or ""))
            width = height = area = ""
            band_count = ""
            aspect = ""
            reader = ""
            status = "read_failed"
            warning = str(exc)
            warnings.append({"sample_id": sample_id, "category": row.get("category", ""), "warning": warning})

        per_sample.append(
            {
                "sample_id": sample_id,
                "image_id": row.get("image_id", ""),
                "image_path": row.get("image_path", ""),
                "extracted_path": row.get("extracted_path", row.get("extracted_image_path", "")),
                "resolved_path": str(path),
                "category": row.get("category", ""),
                "split": row.get("split", ""),
                "country": row.get("country", ""),
                "region": row.get("region", ""),
                "un_region": row.get("un_region", ""),
                "continent": row.get("continent", ""),
                "width": width,
                "height": height,
                "area_pixels": area,
                "aspect_ratio": aspect,
                "band_count": band_count,
                "reader": reader,
                "status": status,
                "warning": warning,
            }
        )

    ok_rows = [row for row in per_sample if row["status"] == "ok"]
    if rows and not ok_rows:
        artifacts = {
            "per_sample": output_dir / "patch_size_per_sample.csv",
            "warnings": output_dir / "warnings.json",
        }
        write_csv(artifacts["per_sample"], per_sample)
        artifacts["warnings"].write_text(json.dumps(warnings, indent=2, sort_keys=True), encoding="utf-8")
        raise RuntimeError(
            f"Patch-size diagnostics found 0 readable rasters from {len(rows)} manifest rows. "
            "Check --data-root and path columns; stale absolute extracted_path values are ignored when they do not exist."
        )
    by_category = summarize_groups(ok_rows, "category")
    by_split = summarize_groups(ok_rows, "split")

    artifacts = {
        "per_sample": output_dir / "patch_size_per_sample.csv",
        "by_category": output_dir / "patch_size_by_category.csv",
        "by_split": output_dir / "patch_size_by_split.csv",
        "warnings": output_dir / "warnings.json",
        "report": output_dir / "patch_size_diagnostic_report.md",
    }
    write_csv(artifacts["per_sample"], per_sample)
    write_csv(artifacts["by_category"], by_category)
    write_csv(artifacts["by_split"], by_split)
    artifacts["warnings"].write_text(json.dumps(warnings, indent=2, sort_keys=True), encoding="utf-8")
    figure_paths = write_figures(ok_rows, by_category, output_dir)
    artifacts.update(figure_paths)
    write_report(artifacts["report"], manifest, data_root, rows, ok_rows, by_category, by_split, warnings, figure_paths)
    return artifacts


def summarize_groups(rows: list[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "") or "missing")].append(row)
    out: list[dict[str, Any]] = []
    for name, group in sorted(groups.items(), key=lambda item: item[0]):
        widths = [float(row["width"]) for row in group]
        heights = [float(row["height"]) for row in group]
        areas = [float(row["area_pixels"]) for row in group]
        aspects = [float(row["aspect_ratio"]) for row in group]
        out.append(
            {
                key: name,
                "sample_count": len(group),
                "width_min": min(widths),
                "width_median": median(widths),
                "width_mean": mean(widths),
                "width_max": max(widths),
                "height_min": min(heights),
                "height_median": median(heights),
                "height_mean": mean(heights),
                "height_max": max(heights),
                "area_min": min(areas),
                "area_median": median(areas),
                "area_mean": mean(areas),
                "area_max": max(areas),
                "aspect_ratio_median": median(aspects),
            }
        )
    return out


def write_figures(
    ok_rows: list[Mapping[str, Any]], by_category: list[Mapping[str, Any]], output_dir: Path
) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    top_categories = sorted(by_category, key=lambda row: float(row["area_median"]), reverse=True)
    labels = [str(row["category"]) for row in top_categories]
    values = [float(row["area_median"]) for row in top_categories]
    if labels:
        height = max(6, min(18, len(labels) * 0.22))
        plt.figure(figsize=(10, height))
        plt.barh(labels[::-1], values[::-1])
        plt.xlabel("Median area (pixels)")
        plt.ylabel("Category")
        plt.tight_layout()
        paths["category_median_area"] = figures / "category_median_area.png"
        plt.savefig(paths["category_median_area"], dpi=160)
        plt.close()

    widths = [float(row["width"]) for row in ok_rows]
    heights = [float(row["height"]) for row in ok_rows]
    areas = [float(row["area_pixels"]) for row in ok_rows]
    if widths and heights:
        plt.figure(figsize=(8, 5))
        plt.scatter(widths, heights, s=8, alpha=0.25)
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.tight_layout()
        paths["width_height_distribution"] = figures / "width_height_distribution.png"
        plt.savefig(paths["width_height_distribution"], dpi=160)
        plt.close()
    if areas:
        plt.figure(figsize=(8, 5))
        plt.hist(areas, bins=60)
        plt.xlabel("Area (pixels)")
        plt.ylabel("Sample count")
        plt.tight_layout()
        paths["area_distribution"] = figures / "area_distribution.png"
        plt.savefig(paths["area_distribution"], dpi=160)
        plt.close()
    return paths


def write_report(
    path: Path,
    manifest: Path,
    data_root: Path | None,
    rows: list[Mapping[str, Any]],
    ok_rows: list[Mapping[str, Any]],
    by_category: list[Mapping[str, Any]],
    by_split: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
    figure_paths: Mapping[str, Path],
) -> None:
    areas = [float(row["area_pixels"]) for row in ok_rows]
    widths = [float(row["width"]) for row in ok_rows]
    heights = [float(row["height"]) for row in ok_rows]
    smallest = sorted(by_category, key=lambda row: float(row["area_median"]))[:10]
    largest = sorted(by_category, key=lambda row: float(row["area_median"]), reverse=True)[:10]
    lines = [
        "# fMoW-Sentinel Patch Size Diagnostics",
        "",
        "This is a dataset/protocol interpretability diagnostic. It is not a model experiment, does not train or run inference, and does not change the baseline closure results.",
        "",
        f"- manifest: `{manifest}`",
        f"- data_root: `{data_root or ''}`",
        f"- rows requested: {len(rows)}",
        f"- readable rasters: {len(ok_rows)}",
        f"- read failures: {len(warnings)}",
        "",
        "## Protocol Note",
        "",
        "fMoW-Sentinel has variable patch extent even under consistent Sentinel-2 spatial resolution. Width, height, and pixel area can differ strongly by category. Resizing normalizes the model input tensor shape but does not recover missing scene context for small original patches.",
        "",
        "This diagnostic can help interpret why large-object classes such as airports can be visually obvious in RGB previews, while smaller or less visually distinctive classes such as solar farms, hospitals, schools, or police stations may be difficult for human/GPT RGB preview inspection. It should not be framed as a fairness main finding.",
        "",
        "## Overall Shape Summary",
        "",
    ]
    if ok_rows:
        lines.extend(
            [
                f"- width min / median / max: {min(widths):.0f} / {median(widths):.0f} / {max(widths):.0f}",
                f"- height min / median / max: {min(heights):.0f} / {median(heights):.0f} / {max(heights):.0f}",
                f"- area min / median / max: {min(areas):.0f} / {median(areas):.0f} / {max(areas):.0f}",
                "",
            ]
        )
    lines.extend(["## Smallest Categories By Median Area", ""])
    for row in smallest:
        lines.append(f"- {row['category']}: median area {float(row['area_median']):.0f} pixels, n={row['sample_count']}")
    lines.extend(["", "## Largest Categories By Median Area", ""])
    for row in largest:
        lines.append(f"- {row['category']}: median area {float(row['area_median']):.0f} pixels, n={row['sample_count']}")
    lines.extend(["", "## Split Summary", ""])
    for row in by_split:
        lines.append(
            f"- {row['split']}: n={row['sample_count']}, median area={float(row['area_median']):.0f}, "
            f"median width={float(row['width_median']):.0f}, median height={float(row['height_median']):.0f}"
        )
    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for name, figure_path in figure_paths.items():
            lines.append(f"- {name}: `{figure_path}`")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This diagnostic does not modify formal ResNet/DOFA results.",
            "- This diagnostic does not enter BigEarthNet/reBEN/CROMA Step 1.",
            "- This diagnostic supports dataset/protocol interpretation only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze fMoW-Sentinel TIFF patch size variation from a clean subset manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    analyze_manifest(args.manifest, args.data_root, args.output_dir, progress_every=args.progress_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
