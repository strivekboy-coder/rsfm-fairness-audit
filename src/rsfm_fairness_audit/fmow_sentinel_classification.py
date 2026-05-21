from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.adapters.dofa import DOFAAdapter
from rsfm_fairness_audit.audit_pipeline import evaluate_bwer_from_file
from rsfm_fairness_audit.audit_table import build_audit_table_from_predictions, validate_audit_table, write_audit_table
from rsfm_fairness_audit.band_profiles import get_band_profile
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


GEOGRAPHY_COLUMNS = (
    "timestamp",
    "year",
    "month",
    "season",
    "location_id",
    "latitude",
    "longitude",
    "country",
    "continent",
    "un_region",
    "region",
    "latitude_band",
)


@dataclass(frozen=True)
class FmowClassificationConfig:
    metadata_csv: Path
    output_dir: Path
    data_root: Path | None = None
    model: str = "supervised_stats"
    model_config: Path | None = None
    train_split: str = "train"
    eval_split: str = "val"
    max_samples: int | None = None
    image_size: int = 96
    batch_size: int = 32
    seed: int = 42
    split_protocol: str = "official_split"
    eval_scope: str = "val"
    band_profile: str = "sentinel2_13band_fmow"
    allow_torch_hub_download: bool = False
    run_bwer: bool = False
    bwer_bootstrap: int = 0


@dataclass(frozen=True)
class FmowBwerConfig:
    input_dir: Path
    output_dir: Path
    audit_table: Path | None = None
    bootstrap: int = 0
    seed: int = 42


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == "" or str(value).lower() in {"nan", "none", "null"}


def _safe_float(value: Any) -> float | str:
    if _is_missing(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _derive_date_fields(row: dict[str, Any]) -> None:
    timestamp = str(row.get("timestamp", "") or "")
    if not timestamp:
        return
    if not row.get("year") and len(timestamp) >= 4:
        row["year"] = timestamp[:4]
    if not row.get("month") and len(timestamp) >= 7:
        try:
            month = int(timestamp[5:7])
        except ValueError:
            return
        row["month"] = str(month)
        if not row.get("season"):
            if month in {12, 1, 2}:
                row["season"] = "DJF"
            elif month in {3, 4, 5}:
                row["season"] = "MAM"
            elif month in {6, 7, 8}:
                row["season"] = "JJA"
            else:
                row["season"] = "SON"


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("sample_id", "image_id", "id"):
        if key in row and not _is_missing(row[key]):
            return str(row[key])
    return str(index)


def _resolve_image_path(row: Mapping[str, Any], data_root: Path | None) -> Path:
    value = row.get("image_path") or row.get("raster_path") or row.get("path")
    if _is_missing(value):
        raise FileNotFoundError("row is missing image_path/raster_path/path")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (data_root / path) if data_root else path


def _read_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        data = np.load(path)
        for key in ("image", "chip", "arr_0", "x"):
            if key in data:
                return np.asarray(data[key])
        return np.asarray(data[data.files[0]])
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as src:
                return src.read()
        except ImportError:
            try:
                import tifffile
            except ImportError as exc:
                raise RuntimeError("Reading TIFF rasters requires rasterio or tifffile.") from exc
            return np.asarray(tifffile.imread(path))
    raise ValueError(f"Unsupported fMoW-Sentinel raster extension: {path.suffix}")


def _to_channels_first(array: np.ndarray, expected_bands: int) -> np.ndarray:
    if array.ndim == 2:
        raise ValueError("Expected 13-band image, got a single 2D array.")
    if array.ndim != 3:
        raise ValueError(f"Expected image shape [bands,H,W] or [H,W,bands], got {array.shape}.")
    if array.shape[0] == expected_bands:
        out = array
    elif array.shape[-1] == expected_bands:
        out = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"Expected {expected_bands} bands, got shape {array.shape}.")
    out = out.astype(np.float32, copy=False)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _resize_nearest(chip: np.ndarray, size: int) -> np.ndarray:
    if chip.shape[-2:] == (size, size):
        return chip
    h, w = chip.shape[-2:]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid spatial shape: {chip.shape}.")
    y = np.linspace(0, h - 1, size).round().astype(int)
    x = np.linspace(0, w - 1, size).round().astype(int)
    return chip[:, y][:, :, x]


def load_fmow_sentinel_image(
    row: Mapping[str, Any],
    data_root: Path | None = None,
    image_size: int = 96,
    band_profile: str = "sentinel2_13band_fmow",
) -> np.ndarray:
    profile = get_band_profile(band_profile)
    expected_bands = int(profile["expected_bands"])
    path = _resolve_image_path(row, data_root)
    if not path.exists():
        raise FileNotFoundError(path)
    chip = _to_channels_first(_read_array(path), expected_bands)
    return _resize_nearest(chip, image_size)


def _image_features(chip: np.ndarray) -> np.ndarray:
    flat = chip.reshape(chip.shape[0], -1)
    return np.concatenate(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            flat.min(axis=1),
            flat.max(axis=1),
        ]
    ).astype(np.float32)


class _NearestCentroidClassifier:
    def __init__(self) -> None:
        self.classes: list[str] = []
        self.centroids: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> None:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for feature, label in zip(features, labels):
            grouped[str(label)].append(feature)
        self.classes = sorted(grouped)
        self.centroids = np.stack([np.mean(grouped[label], axis=0) for label in self.classes]).astype(np.float32)

    def predict(self, features: np.ndarray) -> list[str]:
        if self.centroids is None:
            raise RuntimeError("classifier has not been fitted")
        diff = features[:, None, :] - self.centroids[None, :, :]
        distances = np.sum(diff * diff, axis=2)
        return [self.classes[int(index)] for index in np.argmin(distances, axis=1)]


def _load_metadata(path: Path, max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in read_csv_rows(path)]
    for index, row in enumerate(rows):
        row["sample_id"] = _sample_id(row, index)
        row["category"] = row.get("category") or row.get("label") or row.get("class_label") or ""
        row["split"] = row.get("split") or "all"
        _derive_date_fields(row)
    if max_samples and max_samples > 0 and len(rows) > max_samples:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(rows), size=max_samples, replace=False).tolist())
        rows = [rows[index] for index in indices]
    return rows


def _split_rows(rows: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return list(rows)
    return [row for row in rows if str(row.get("split", "")) == split]


def _rows_to_features(rows: Sequence[dict[str, Any]], config: FmowClassificationConfig) -> tuple[np.ndarray, list[str], list[dict[str, Any]], list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    ok_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in rows:
        try:
            chip = load_fmow_sentinel_image(row, config.data_root, config.image_size, config.band_profile)
        except Exception as exc:
            warnings.append(f"Skipping sample_id={row.get('sample_id')}: {exc}")
            continue
        features.append(_image_features(chip))
        labels.append(str(row.get("category", "")))
        ok_rows.append(row)
    if not features:
        raise ValueError("No readable fMoW-Sentinel images found for feature extraction.")
    return np.stack(features).astype(np.float32), labels, ok_rows, warnings


def _dofa_embeddings(
    rows: Sequence[dict[str, Any]],
    config: FmowClassificationConfig,
    adapter: DOFAAdapter,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]], list[str]]:
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    ok_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    batch_samples: list[dict[str, Any]] = []
    batch_meta: list[dict[str, Any]] = []
    for row in rows:
        try:
            chip = load_fmow_sentinel_image(row, config.data_root, config.image_size, config.band_profile)
        except Exception as exc:
            warnings.append(f"Skipping sample_id={row.get('sample_id')}: {exc}")
            continue
        batch_samples.append({"image": chip})
        batch_meta.append(row)
        if len(batch_samples) >= config.batch_size:
            prepared = adapter.preprocess({"samples": batch_samples, "metadata": batch_meta})
            embeddings.append(adapter.extract_embeddings(prepared))
            labels.extend(str(item.get("category", "")) for item in batch_meta)
            ok_rows.extend(batch_meta)
            batch_samples, batch_meta = [], []
    if batch_samples:
        prepared = adapter.preprocess({"samples": batch_samples, "metadata": batch_meta})
        embeddings.append(adapter.extract_embeddings(prepared))
        labels.extend(str(item.get("category", "")) for item in batch_meta)
        ok_rows.extend(batch_meta)
    if not embeddings:
        raise ValueError("No readable fMoW-Sentinel images found for DOFA embedding extraction.")
    return np.concatenate(embeddings, axis=0).astype(np.float32), labels, ok_rows, warnings


def _model_metadata(config: FmowClassificationConfig) -> dict[str, str]:
    if config.model == "dofa":
        return {
            "model": "dofa_fmow_sentinel",
            "model_family": "dofa",
            "model_variant": "vit_base_dofa_frozen_probe",
            "adaptation_protocol": "frozen_probe",
            "training_budget": "frozen_encoder_nearest_centroid_probe",
        }
    return {
        "model": "supervised_stats_fmow_sentinel",
        "model_family": "supervised_stats",
        "model_variant": "band_stats_nearest_centroid",
        "adaptation_protocol": "supervised_baseline",
        "training_budget": "nearest_centroid_on_13band_image_statistics",
    }


def _prediction_rows(
    eval_rows: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    config: FmowClassificationConfig,
) -> list[dict[str, Any]]:
    meta = _model_metadata(config)
    rows: list[dict[str, Any]] = []
    for row, prediction in zip(eval_rows, predictions):
        label = str(row.get("category", ""))
        correct = float(prediction == label)
        out: dict[str, Any] = {
            "sample_id": row.get("sample_id", ""),
            "image_id": row.get("image_id", ""),
            "image_path": row.get("image_path", ""),
            "dataset": "fmow_sentinel",
            "task": "scene_classification",
            "split": row.get("split", config.eval_split),
            "label": label,
            "category": label,
            "class_label": label,
            "prediction": prediction,
            "correct": correct,
            "score": correct,
            "risk": 1.0 - correct,
            "model": meta["model"],
            "model_family": meta["model_family"],
            "model_variant": meta["model_variant"],
            "input_mode": "s2_13band_image_only",
            "adaptation_protocol": meta["adaptation_protocol"],
            "training_budget": meta["training_budget"],
            "split_protocol": config.split_protocol,
            "eval_scope": config.eval_scope or config.eval_split,
            "resolution": str(config.image_size),
            "band_profile": config.band_profile,
            "metadata_provenance": row.get("metadata_provenance", "location_level_geography_enrichment"),
        }
        for key in GEOGRAPHY_COLUMNS:
            out[key] = row.get(key, "")
        rows.append(out)
    return rows


def _write_report(output_dir: Path, predictions: Sequence[Mapping[str, Any]], warnings: Sequence[str], config: FmowClassificationConfig) -> Path:
    accuracy = np.mean([float(row["correct"]) for row in predictions]) if predictions else float("nan")
    counts = Counter(str(row.get("country", "")) for row in predictions if not _is_missing(row.get("country")))
    path = output_dir / "report.md"
    path.write_text(
        "\n".join(
            [
                "# fMoW-Sentinel Image-Only Classification Prototype",
                "",
                f"- model: `{_model_metadata(config)['model']}`",
                f"- model_variant: `{_model_metadata(config)['model_variant']}`",
                f"- adaptation_protocol: `{_model_metadata(config)['adaptation_protocol']}`",
                f"- split_protocol: `{config.split_protocol}`",
                f"- train_split: `{config.train_split}`",
                f"- eval_split: `{config.eval_split}`",
                f"- image_size: `{config.image_size}`",
                f"- band_profile: `{config.band_profile}`",
                f"- prediction rows: {len(predictions)}",
                f"- aggregate accuracy: {accuracy:.6f}" if not math.isnan(accuracy) else "- aggregate accuracy: n/a",
                f"- geography slices with non-missing country: {len(counts)}",
                "",
                "Geography metadata is used only for audit slicing, support diagnostics, and BWER reporting. "
                "Country, region, coordinates, timestamps, and location IDs are not model inputs.",
                "",
                "The current fMoW geography enrichment is location-level, not an image-level exact join; reports must preserve that provenance.",
                "",
                "## Warnings",
                *(f"- {warning}" for warning in warnings[:50]),
            ]
        ),
        encoding="utf-8",
    )
    return path


def run_fmow_sentinel_classification(config: FmowClassificationConfig) -> dict[str, Path]:
    output = ensure_dir(config.output_dir)
    rows = _load_metadata(config.metadata_csv, config.max_samples, config.seed)
    train_rows = _split_rows(rows, config.train_split)
    eval_rows = _split_rows(rows, config.eval_split)
    if not train_rows:
        raise ValueError(f"No training rows found for split={config.train_split!r}.")
    if not eval_rows:
        raise ValueError(f"No evaluation rows found for split={config.eval_split!r}.")

    if config.model == "dofa":
        if config.model_config is None:
            raise ValueError("--model-config is required for --model dofa.")
        adapter = DOFAAdapter.from_config_file(config.model_config)
        if config.allow_torch_hub_download:
            adapter.allow_torch_hub_download = True
        adapter.load_model()
        train_x, train_y, train_ok, warnings_train = _dofa_embeddings(train_rows, config, adapter)
        eval_x, _eval_y, eval_ok, warnings_eval = _dofa_embeddings(eval_rows, config, adapter)
    elif config.model == "supervised_stats":
        train_x, train_y, train_ok, warnings_train = _rows_to_features(train_rows, config)
        eval_x, _eval_y, eval_ok, warnings_eval = _rows_to_features(eval_rows, config)
    else:
        raise ValueError(f"Unsupported fMoW-Sentinel classification model: {config.model}")

    classifier = _NearestCentroidClassifier()
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(eval_x)
    prediction_rows = _prediction_rows(eval_ok, predictions, config)
    audit_rows = build_audit_table_from_predictions_from_rows(prediction_rows)

    artifacts = {
        "predictions": output / "predictions.csv",
        "audit_table": output / "audit_table.csv",
        "run_metadata": output / "run_metadata.json",
        "model_debug": output / "model_debug.json",
        "report": output / "report.md",
    }
    write_csv(artifacts["predictions"], prediction_rows)
    write_audit_table(artifacts["audit_table"], audit_rows)
    warnings = warnings_train + warnings_eval
    accuracy = np.mean([float(row["correct"]) for row in prediction_rows]) if prediction_rows else float("nan")
    metadata = {
        "dataset": "fmow_sentinel",
        "task": "scene_classification",
        "model": _model_metadata(config)["model"],
        "model_family": _model_metadata(config)["model_family"],
        "model_variant": _model_metadata(config)["model_variant"],
        "adaptation_protocol": _model_metadata(config)["adaptation_protocol"],
        "split_protocol": config.split_protocol,
        "train_split": config.train_split,
        "eval_split": config.eval_split,
        "eval_scope": config.eval_scope or config.eval_split,
        "image_size": config.image_size,
        "band_profile": config.band_profile,
        "input_mode": "s2_13band_image_only",
        "train_rows_requested": len(train_rows),
        "train_rows_readable": len(train_ok),
        "eval_rows_requested": len(eval_rows),
        "eval_rows_readable": len(eval_ok),
        "aggregate_accuracy": accuracy,
        "geography_metadata_usage": "audit_slicing_only_not_model_input",
        "geography_join_level": "location_level",
    }
    artifacts["run_metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    artifacts["model_debug"].write_text(
        json.dumps(
            {
                "model": metadata["model"],
                "classifier": "nearest_centroid",
                "feature_dimension": int(train_x.shape[1]),
                "classes": classifier.classes,
                "warnings": warnings[:100],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(output, prediction_rows, warnings, config)
    if config.run_bwer:
        artifacts.update(run_fmow_geography_bwer(FmowBwerConfig(input_dir=output, output_dir=output / "bwer", bootstrap=config.bwer_bootstrap, seed=config.seed)))
    return artifacts


def build_audit_table_from_predictions_from_rows(prediction_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    temp_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        out = dict(row)
        out["unit_id"] = out.get("sample_id", out.get("image_id", ""))
        out["y_true"] = out.get("label", "")
        out["y_pred"] = out.get("prediction", "")
        out["aggregation_level"] = "sample"
        temp_rows.append(out)
    validate_audit_table(temp_rows)
    return temp_rows


def run_fmow_geography_bwer(config: FmowBwerConfig) -> dict[str, Path]:
    audit_table = config.audit_table or config.input_dir / "audit_table.csv"
    if not audit_table.exists():
        predictions = config.input_dir / "predictions.csv"
        if not predictions.exists():
            raise FileNotFoundError(f"Expected audit_table.csv or predictions.csv in {config.input_dir}")
        rows = build_audit_table_from_predictions(predictions, dataset="fmow_sentinel", model="", task="scene_classification")
        audit_table = ensure_dir(config.output_dir) / "audit_table.csv"
        write_audit_table(audit_table, rows)
    artifacts = evaluate_bwer_from_file(
        audit_table,
        dataset="fmow_sentinel",
        model=_infer_model_from_audit_table(audit_table),
        task="scene_classification",
        output_dir=config.output_dir,
        bootstrap=config.bootstrap,
        seed=config.seed,
        score_column="correct",
        risk_column="risk",
        audit_level="pilot",
    )
    return {f"bwer_{key}": value for key, value in artifacts.items()}


def _infer_model_from_audit_table(path: Path) -> str:
    rows = read_csv_rows(path)
    return str(rows[0].get("model", "fmow_sentinel_model")) if rows else "fmow_sentinel_model"


def _read_bwer_summary(run_dir: Path) -> list[dict[str, str]]:
    for path in (run_dir / "bwer" / "bwer_summary.csv", run_dir / "bwer_summary.csv"):
        if path.exists():
            return read_csv_rows(path)
    return []


def _bwer_value(rows: Sequence[Mapping[str, Any]], slice_variable: str, balance_variable: str = "") -> dict[str, Any]:
    for row in rows:
        if str(row.get("slice_variable", "")) == slice_variable and str(row.get("balance_variable", "")) == balance_variable:
            return dict(row)
    return {}


def compare_fmow_runs(runs: Mapping[str, Path], output_dir: Path) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    summary_rows: list[dict[str, Any]] = []
    average_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    for run_name, run_dir in runs.items():
        audit_path = run_dir / "audit_table.csv"
        if not audit_path.exists():
            raise FileNotFoundError(f"Missing audit_table.csv for run {run_name}: {run_dir}")
        audit_rows = read_csv_rows(audit_path)
        bwer_rows = _read_bwer_summary(run_dir)
        correct = [float(row.get("correct", row.get("score", 0.0))) for row in audit_rows if not _is_missing(row.get("correct", row.get("score", "")))]
        accuracy = float(np.mean(correct)) if correct else float("nan")
        first = audit_rows[0] if audit_rows else {}
        raw_country = _bwer_value(bwer_rows, "country")
        std_country = _bwer_value(bwer_rows, "country", "class_label") or _bwer_value(bwer_rows, "country", "category")
        summary = {
            "run_name": run_name,
            "model": first.get("model", ""),
            "model_family": first.get("model_family", ""),
            "model_variant": first.get("model_variant", ""),
            "adaptation_protocol": first.get("adaptation_protocol", ""),
            "split_protocol": first.get("split_protocol", ""),
            "eval_scope": first.get("eval_scope", first.get("split", "")),
            "dataset": first.get("dataset", "fmow_sentinel"),
            "task": first.get("task", "scene_classification"),
            "resolution": first.get("resolution", ""),
            "aggregate_accuracy": accuracy,
            "aggregate_error": 1.0 - accuracy if not math.isnan(accuracy) else "",
            "raw_bwer_country": raw_country.get("bwer", ""),
            "standardised_bwer_country_class": std_country.get("bwer", ""),
            "worst_country_slice": raw_country.get("worst_slice", ""),
            "tail_country_slices": raw_country.get("tail_slices", ""),
            "notes": "Protocol-aware image-only comparison; geography metadata is audit-only and not model input.",
        }
        summary_rows.append(summary)
        average_rows.append(
            {
                "run_name": run_name,
                "average_score": accuracy,
                "raw_bwer": raw_country.get("bwer", ""),
                "standardised_bwer": std_country.get("bwer", ""),
                "model_label": summary["model_variant"] or summary["model"],
                "protocol_label": summary["adaptation_protocol"],
                "split_label": summary["split_protocol"],
            }
        )
        by_slice_path = run_dir / "bwer" / "bwer_by_slice.csv"
        if by_slice_path.exists():
            for row in read_csv_rows(by_slice_path):
                if str(row.get("slice_variable", "")) in {"country", "continent", "un_region", "region", "latitude_band", "season", "category"}:
                    item = dict(row)
                    item["run_name"] = run_name
                    slice_rows.append(item)
    artifacts = {
        "comparison_summary": output / "comparison_summary.csv",
        "average_vs_bwer": output / "average_vs_bwer.csv",
        "geography_slice_comparison": output / "geography_slice_comparison.csv",
        "comparison_report": output / "comparison_report.md",
    }
    write_csv(artifacts["comparison_summary"], summary_rows)
    write_csv(artifacts["average_vs_bwer"], average_rows)
    write_csv(artifacts["geography_slice_comparison"], slice_rows)
    _write_comparison_report(artifacts["comparison_report"], summary_rows)
    return artifacts


def _write_comparison_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# fMoW-Sentinel Geography BWER Run Comparison",
        "",
        "This is a protocol-aware comparison of completed image-only fMoW-Sentinel runs.",
        "Geography metadata is used only for audit slicing and reporting, not as model input.",
        "",
        "| run | accuracy | Raw-BWER(country) | Standardised-BWER(country | class) | protocol |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('run_name','')} | {row.get('aggregate_accuracy','')} | "
            f"{row.get('raw_bwer_country','')} | {row.get('standardised_bwer_country_class','')} | "
            f"{row.get('adaptation_protocol','')} / {row.get('split_protocol','')} |"
        )
    lines.extend(
        [
            "",
            "Report average accuracy together with Raw-BWER and class-standardised BWER. "
            "A higher aggregate score does not by itself imply lower geography tail risk.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def zip_output_dir(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.make_archive(str(destination.with_suffix("")), "zip", source.parent, source.name)
    return destination
