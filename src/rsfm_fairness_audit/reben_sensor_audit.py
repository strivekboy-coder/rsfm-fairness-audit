from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsfm_fairness_audit.bwer import BWERConfig, compute_bwer_family, create_interaction_slice
from rsfm_fairness_audit.io import ensure_dir, read_csv_rows, write_csv


SOURCE_VERIFICATION_URLS = [
    "https://bigearth.net/",
    "https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf",
    "https://huggingface.co/BIFOLD-BigEarthNetv2-0",
    "https://lhackel-tub.github.io/ConfigILM/extra/DataSets%20and%20DataModules/bigearthnetv2.html",
    "https://github.com/antofuller/croma",
]
SOURCE_VERIFICATION_SUMMARY = {
    "dataset": "BigEarthNet v2.0 / reBEN",
    "paired_patches": "549,488 paired Sentinel-1/Sentinel-2 patches",
    "labels": "19-label multi-label one-hot vector via ConfigILM/reBEN; class names follow the BigEarthNet v2.0 PDF Table 1 19-class nomenclature",
    "configilm_loader": "BENv2DataSet with images_lmdb, metadata_parquet, metadata_snow_cloud_parquet",
    "croma_outputs": "SAR_GAP, optical_GAP, joint_GAP",
    "bifold_resnet101": "official v0.2.0 S1/S2/all model cards require reben_publication BigEarthNetv2_0_ImageClassifier.from_pretrained",
}

REBEN_DATASET = "bigearthnet_v2_reben"
REBEN_TASK = "multilabel_scene_classification"
REBEN_LABEL_COUNT = 19
REBEN_19_CLASS_NAMES = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)
REBEN_PRIMARY_SLICES = ("class_label", "country")
REBEN_GEOGRAPHY_SLICES = ("country",)
REBEN_INTERACTION_SLICES = (("country", "class_label", "country_x_class"),)
REBEN_STANDARDISED = (("country", "class_label"),)
REBEN_ALPHA_VALUES = (0.1, 0.2, 0.3)
REBEN_SUPPORT_VALUES = (10, 20, 30)
REBEN_MISSING_POLICIES = ("renormalize", "overlap", "invalidate")
REBEN_CROMA_EMBEDDING_KEYS = {
    "S1": "SAR_GAP",
    "S2": "optical_GAP",
    "S1+S2": "joint_GAP",
}
REBEN_BIFOLD_RESNET101_IDS = {
    "S1": "BIFOLD-BigEarthNetv2-0/resnet101-s1-v0.2.0",
    "S2": "BIFOLD-BigEarthNetv2-0/resnet101-s2-v0.2.0",
    "S1+S2": "BIFOLD-BigEarthNetv2-0/resnet101-all-v0.2.0",
}
REBEN_REQUIRED_RUNS = (
    "croma_s1",
    "croma_s2",
    "croma_s1_plus_s2",
    "bifold_resnet101_s1",
    "bifold_resnet101_s2",
    "bifold_resnet101_s1_plus_s2",
)


@dataclass(frozen=True)
class RebenRunLabels:
    model_family: str
    model_variant: str
    sensor_mode: str
    adaptation_protocol: str
    split_protocol: str = "official_split"
    eval_scope: str = "validation"
    input_mode: str = "sensor_image_only"
    band_profile: str = "reben_official"


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    labels = np.asarray(y_true, dtype=float)
    probs = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))


def average_precision_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    cumulative_tp = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    precision_at_k = cumulative_tp / ranks
    return float(np.sum(precision_at_k * ranked) / positives)


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = (2.0 * tp) + fp + fn
    return 0.0 if denom <= 0 else float((2.0 * tp) / denom)


def select_thresholds_from_validation(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    grid: Sequence[float] | None = None,
) -> np.ndarray:
    """Select per-label thresholds from validation labels/probabilities only."""
    labels = np.asarray(y_true_val, dtype=int)
    probs = np.asarray(y_prob_val, dtype=float)
    if labels.shape != probs.shape or labels.ndim != 2:
        raise ValueError("Validation labels/probabilities must have matching shape [n_samples, n_labels].")
    thresholds = np.zeros(labels.shape[1], dtype=float)
    candidates = np.asarray(grid if grid is not None else np.linspace(0.05, 0.95, 19), dtype=float)
    for label_index in range(labels.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0
        truth = labels[:, label_index]
        scores = probs[:, label_index]
        for threshold in candidates:
            pred = scores >= threshold
            tp = float(np.sum(pred & (truth == 1)))
            fp = float(np.sum(pred & (truth == 0)))
            fn = float(np.sum((~pred) & (truth == 1)))
            f1 = _f1_from_counts(tp, fp, fn)
            if f1 > best_f1 or (math.isclose(f1, best_f1) and threshold < best_threshold):
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[label_index] = best_threshold
    return thresholds


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Sequence[float],
    class_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)
    threshold_arr = np.asarray(thresholds, dtype=float)
    if labels.shape != probs.shape or labels.ndim != 2:
        raise ValueError("Labels/probabilities must have matching shape [n_samples, n_labels].")
    if threshold_arr.shape != (labels.shape[1],):
        raise ValueError("thresholds must have one value per label.")
    names = list(class_names or default_reben_class_names()[: labels.shape[1]])
    pred = probs >= threshold_arr[None, :]
    ap_values = [average_precision_score_binary(labels[:, index], probs[:, index]) for index in range(labels.shape[1])]
    f1_values = []
    per_class = []
    for index, name in enumerate(names):
        truth = labels[:, index] == 1
        guess = pred[:, index]
        tp = float(np.sum(guess & truth))
        fp = float(np.sum(guess & ~truth))
        fn = float(np.sum(~guess & truth))
        f1 = _f1_from_counts(tp, fp, fn)
        f1_values.append(f1)
        per_class.append(
            {
                "class_index": index,
                "class_label": name,
                "average_precision": ap_values[index],
                "f1": f1,
                "threshold": float(threshold_arr[index]),
                "positive_support": int(np.sum(truth)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    micro_tp = float(np.sum(pred & (labels == 1)))
    micro_fp = float(np.sum(pred & (labels == 0)))
    micro_fn = float(np.sum((~pred) & (labels == 1)))
    flat_ap = average_precision_score_binary(labels.reshape(-1), probs.reshape(-1))
    summary = {
        "macro_ap": float(np.nanmean(ap_values)),
        "micro_ap": flat_ap,
        "macro_f1": float(np.nanmean(f1_values)),
        "micro_f1": _f1_from_counts(micro_tp, micro_fp, micro_fn),
        "mean_bce_risk": float(np.mean(binary_cross_entropy(labels, probs))),
        "n_samples": int(labels.shape[0]),
        "n_labels": int(labels.shape[1]),
        "threshold_policy": "validation_per_label_f1",
    }
    return summary, per_class


def train_linear_multilabel_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 42,
    device: str = "auto",
    log_prefix: str = "[reben:probe]",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train a lightweight BCE linear probe on frozen multi-label embeddings."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - exercised in Colab/runtime environments
        raise RuntimeError("PyTorch is required for the formal reBEN linear multi-label probe.") from exc

    x_train = np.asarray(train_embeddings, dtype=np.float32)
    y_train = np.asarray(train_labels, dtype=np.float32)
    x_eval = np.asarray(eval_embeddings, dtype=np.float32)
    if x_train.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays [n_samples, embedding_dim].")
    if y_train.ndim != 2:
        raise ValueError("reBEN train labels must be multi-label arrays [n_samples, n_labels].")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("train_embeddings and train_labels row counts differ.")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    x_train_norm = (x_train - mean) / std
    x_eval_norm = (x_eval - mean) / std
    model = nn.Linear(x_train.shape[1], y_train.shape[1]).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    indices = np.arange(x_train.shape[0])
    history: list[dict[str, Any]] = []
    print(
        f"{log_prefix} training linear probe "
        f"samples={x_train.shape[0]} labels={y_train.shape[1]} "
        f"embedding_dim={x_train.shape[1]} epochs={int(epochs)} device={torch_device}"
    )
    for epoch in range(1, int(epochs) + 1):
        rng.shuffle(indices)
        losses = []
        model.train()
        for start in range(0, len(indices), int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            xb = torch.as_tensor(x_train_norm[batch_idx], dtype=torch.float32, device=torch_device)
            yb = torch.as_tensor(y_train[batch_idx], dtype=torch.float32, device=torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_bce_loss": epoch_loss})
        print(f"{log_prefix} epoch={epoch}/{int(epochs)} train_bce_loss={epoch_loss:.6f}")
    model.eval()
    with torch.no_grad():
        logits_eval = model(torch.as_tensor(x_eval_norm, dtype=torch.float32, device=torch_device)).detach().cpu().numpy()
    print(f"{log_prefix} finished linear probe; eval_samples={x_eval.shape[0]}")
    metadata = {
        "probe": "linear_multilabel_bce",
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "device": str(torch_device),
        "embedding_dim": int(x_train.shape[1]),
        "n_labels": int(y_train.shape[1]),
        "embedding_standardization": "train_split_mean_std",
        "history": history,
    }
    return logits_eval.astype(np.float32), metadata


def write_thresholds(path: str | Path, thresholds: Sequence[float], class_names: Sequence[str]) -> None:
    rows = [
        {
            "class_index": index,
            "class_label": class_names[index],
            "threshold": float(value),
            "threshold_policy": "validation_per_label_f1",
        }
        for index, value in enumerate(thresholds)
    ]
    write_csv(path, rows)


def write_thresholds_json(path: str | Path, thresholds: Sequence[float], class_names: Sequence[str]) -> None:
    payload = {
        "threshold_policy": "validation_per_label_f1",
        "thresholds": [
            {"class_index": index, "class_label": class_names[index], "threshold": float(value)}
            for index, value in enumerate(thresholds)
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_outputs(
    output_dir: str | Path,
    *,
    run_name: str,
    sample_rows: Sequence[Mapping[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Sequence[float],
    class_names: Sequence[str],
    run_labels: RebenRunLabels,
    probe_metadata: Mapping[str, Any] | None = None,
    run_bwer: bool = True,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    print(f"[reben:outputs] computing metrics for {run_name}")
    summary, per_class = compute_multilabel_metrics(y_true, y_prob, thresholds, class_names)
    print(f"[reben:outputs] expanding predictions for {run_name}")
    predictions = expand_predictions_to_label_audit_rows(sample_rows, y_true, y_prob, thresholds, run_labels, class_names)
    artifacts = {
        "aggregate_metrics": output / f"aggregate_metrics_{run_name}.csv",
        "per_class_metrics": output / f"per_class_metrics_{run_name}.csv",
        "thresholds": output / f"thresholds_{run_name}.csv",
        "thresholds_json": output / f"thresholds_{run_name}.json",
        "predictions": output / f"predictions_{run_name}.csv",
        "run_metadata": output / f"run_metadata_{run_name}.json",
    }
    write_csv(artifacts["aggregate_metrics"], [{**summary, "run_name": run_name, "model": run_labels.model_variant, "sensor_mode": run_labels.sensor_mode}])
    write_csv(artifacts["per_class_metrics"], [{**row, "run_name": run_name} for row in per_class])
    write_thresholds(artifacts["thresholds"], thresholds, class_names)
    write_thresholds_json(artifacts["thresholds_json"], thresholds, class_names)
    print(f"[reben:outputs] writing predictions for {run_name}: rows={len(predictions)}")
    write_csv(artifacts["predictions"], predictions)
    metadata = {
        "run_name": run_name,
        "dataset": REBEN_DATASET,
        "task": REBEN_TASK,
        "label_count": int(np.asarray(y_true).shape[1]),
        "sample_count": int(np.asarray(y_true).shape[0]),
        "model_family": run_labels.model_family,
        "model_variant": run_labels.model_variant,
        "sensor_mode": run_labels.sensor_mode,
        "input_mode": run_labels.input_mode,
        "adaptation_protocol": run_labels.adaptation_protocol,
        "split_protocol": run_labels.split_protocol,
        "eval_scope": run_labels.eval_scope,
        "band_profile": run_labels.band_profile,
        "threshold_policy": "validation_per_label_f1",
        "probe_metadata": dict(probe_metadata or {}),
    }
    artifacts["run_metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if run_bwer:
        print(f"[reben:bwer] running post-hoc BWER for {run_name}")
        bwer_artifacts = run_reben_multilabel_bwer(
            predictions,
            output / "bwer" / run_name,
            model_name=run_name,
            split=run_labels.eval_scope,
            risk_column="risk_bce",
        )
        artifacts.update({f"bwer_{key}": value for key, value in bwer_artifacts.items()})
        secondary_bwer_artifacts = run_reben_multilabel_bwer(
            predictions,
            output / "bwer" / f"{run_name}_binary_error",
            model_name=f"{run_name}_binary_error",
            split=run_labels.eval_scope,
            risk_column="risk_binary_error",
        )
        artifacts.update({f"bwer_binary_error_{key}": value for key, value in secondary_bwer_artifacts.items()})
        selective_path = output / "bwer" / run_name / "selective_risk_summary.csv"
        write_csv(selective_path, compute_selective_risk(predictions))
        artifacts["selective_risk_summary"] = selective_path
        print(f"[reben:bwer] finished post-hoc BWER for {run_name}")
    print(f"[reben:outputs] finished {run_name}")
    return artifacts


def validate_bifold_resnet101_refs(ids_by_mode: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, expected in REBEN_BIFOLD_RESNET101_IDS.items():
        observed = str(ids_by_mode.get(mode, ""))
        is_local_path = bool(observed) and Path(observed).exists()
        is_official_hf_id = observed == expected
        status = "ok" if is_official_hf_id or is_local_path else "invalid"
        if is_official_hf_id:
            reason = "official_v0.2.0_resnet101_hf_id"
        elif is_local_path:
            reason = "local_path_provided_assumed_official_v0.2.0_export_user_must_preserve_provenance"
        else:
            reason = "expected_official_v0.2.0_id_or_existing_local_path_do_not_use_v0.1.1_or_substitute"
        rows.append(
            {
                "sensor_mode": mode,
                "expected_hf_id": expected,
                "observed_hf_id": observed,
                "status": status,
                "reason": reason,
            }
        )
    return rows


def validate_bifold_resnet101_ids(ids_by_mode: Mapping[str, str]) -> list[dict[str, Any]]:
    return validate_bifold_resnet101_refs(ids_by_mode)


def extract_croma_reben_embeddings(
    dataset: Any,
    adapter: Any,
    *,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extract frozen CROMA embeddings from a reBEN multi-label dataset adapter."""
    metadata_rows = dataset.load_metadata()
    adapter.load_model()
    print(f"[reben:croma] extracting embeddings samples={len(metadata_rows)} batch_size={int(batch_size)}")
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for start in range(0, len(metadata_rows), int(batch_size)):
        end = min(start + int(batch_size), len(metadata_rows))
        samples = [dataset.load_sample(index) for index in range(start, end)]
        prepared = adapter.preprocess(
            {
                "samples": samples,
                "metadata": [sample["metadata"] for sample in samples],
            }
        )
        embeddings.append(adapter.extract_embeddings(prepared))
        labels.extend([np.asarray(sample["metadata"]["label_vector"], dtype=np.int64) for sample in samples])
        metadata.extend([dict(sample["metadata"]) for sample in samples])
        print(f"[reben:croma] embeddings {end}/{len(metadata_rows)}")
    if not embeddings:
        raise ValueError("No reBEN samples available for CROMA embedding extraction.")
    print(f"[reben:croma] finished embeddings samples={len(metadata_rows)}")
    return np.vstack(embeddings).astype(np.float32), np.vstack(labels).astype(np.int64), metadata


class BifoldResNet101ConfigurationError(RuntimeError):
    """Raised when the official BIFOLD/reBEN ResNet101 path cannot be used."""


class BifoldResNet101Runner:
    """Thin wrapper around official reBEN publication model cards.

    The model cards explicitly require:

    `from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier`

    and then `BigEarthNetv2_0_ImageClassifier.from_pretrained(<hf_id_or_path>)`.
    This wrapper refuses to invent a torchvision substitute.
    """

    def __init__(self, model_id_or_path: str, *, device: str = "auto", model: Any | None = None) -> None:
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.model = model
        self._torch_device: Any | None = None

    def _resolve_device(self) -> Any:
        if self._torch_device is not None:
            return self._torch_device
        try:
            import torch
        except ImportError as exc:
            raise BifoldResNet101ConfigurationError("PyTorch is required for BIFOLD ResNet101 inference.") from exc
        name = "cuda" if self.device == "auto" and torch.cuda.is_available() else "cpu" if self.device == "auto" else self.device
        if name == "cuda" and not torch.cuda.is_available():
            raise BifoldResNet101ConfigurationError("BIFOLD device='cuda' requested but CUDA is unavailable.")
        self._torch_device = torch.device(name)
        return self._torch_device

    def load_model(self) -> None:
        if self.model is not None:
            if hasattr(self.model, "eval"):
                self.model.eval()
            return
        try:
            from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
        except ImportError as exc:
            raise BifoldResNet101ConfigurationError(
                "Official BIFOLD ResNet101 requires the reBEN publication code exposing "
                "`reben_publication.BigEarthNetv2_0_ImageClassifier`. Install that code in Colab; "
                "do not substitute torchvision ResNet101."
            ) from exc
        self.model = BigEarthNetv2_0_ImageClassifier.from_pretrained(self.model_id_or_path)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self._resolve_device())
        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load_model()
        try:
            import torch
        except ImportError as exc:
            raise BifoldResNet101ConfigurationError("PyTorch is required for BIFOLD ResNet101 inference.") from exc
        tensor = torch.as_tensor(images, dtype=torch.float32, device=self._resolve_device())
        with torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, Mapping):
            for key in ("logits", "prediction", "predictions", "scores"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (list, tuple)):
            output = output[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        logits = np.asarray(output, dtype=np.float32)
        if logits.ndim != 2 or logits.shape[1] != REBEN_LABEL_COUNT:
            raise BifoldResNet101ConfigurationError(f"Expected BIFOLD logits [N,19], got {logits.shape}.")
        return logits


def run_bifold_resnet101_reben_inference(
    *,
    eval_dataset: Any,
    model_runner: BifoldResNet101Runner,
    output_dir: str | Path,
    run_name: str,
    run_labels: RebenRunLabels,
    class_names: Sequence[str] | None = None,
    batch_size: int = 64,
    threshold_source: str = "validation_eval_scores",
    run_bwer: bool = True,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    metadata_rows = eval_dataset.load_metadata()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    model_runner.load_model()
    for start in range(0, len(metadata_rows), int(batch_size)):
        end = min(start + int(batch_size), len(metadata_rows))
        samples = [eval_dataset.load_sample(index) for index in range(start, end)]
        images = np.stack([sample["image"] for sample in samples]).astype(np.float32)
        logits.append(model_runner.predict_logits(images))
        labels.extend([np.asarray(sample["metadata"]["label_vector"], dtype=np.int64) for sample in samples])
        metadata.extend([dict(sample["metadata"]) for sample in samples])
        print(f"[reben:bifold] inference {end}/{len(metadata_rows)}")
    if not logits:
        raise ValueError("No reBEN samples available for BIFOLD inference.")
    y_logits = np.vstack(logits).astype(np.float32)
    y_prob = sigmoid(y_logits)
    y_true = np.vstack(labels).astype(np.int64)
    # The model cards report validation/test scores but do not expose saved
    # training thresholds in a stable public interface. For validation audits,
    # thresholds are selected on validation scores and must be frozen before any
    # future test-set evaluation.
    thresholds = select_thresholds_from_validation(y_true, y_prob)
    names = list(class_names or default_reben_class_names()[: y_true.shape[1]])
    cache_path = output / f"logits_{run_name}.npz"
    np.savez_compressed(
        cache_path,
        logits=y_logits,
        probabilities=y_prob,
        labels=y_true,
        sample_ids=np.asarray([str(row.get("sample_id", index)) for index, row in enumerate(metadata)]),
    )
    artifacts = write_run_outputs(
        output,
        run_name=run_name,
        sample_rows=metadata,
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds,
        class_names=names,
        run_labels=run_labels,
        probe_metadata={
            "official_model_id_or_path": model_runner.model_id_or_path,
            "logits_cache": str(cache_path),
            "threshold_source": threshold_source,
            "reben_loader": eval_dataset.loader_info() if hasattr(eval_dataset, "loader_info") else {},
        },
        run_bwer=run_bwer,
    )
    artifacts["logits"] = cache_path
    return artifacts


def run_croma_reben_frozen_probe(
    *,
    train_dataset: Any,
    eval_dataset: Any,
    croma_adapter: Any,
    output_dir: str | Path,
    run_name: str,
    run_labels: RebenRunLabels,
    class_names: Sequence[str] | None = None,
    batch_size: int = 64,
    probe_epochs: int = 100,
    probe_learning_rate: float = 1e-2,
    probe_weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
    run_bwer: bool = True,
) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    print(f"[reben:croma] {run_name}: extracting train embeddings")
    train_embeddings, train_labels, train_metadata = extract_croma_reben_embeddings(train_dataset, croma_adapter, batch_size=batch_size)
    print(f"[reben:croma] {run_name}: extracting eval embeddings")
    eval_embeddings, eval_labels, eval_metadata = extract_croma_reben_embeddings(eval_dataset, croma_adapter, batch_size=batch_size)
    print(f"[reben:croma] {run_name}: fitting frozen-embedding linear probe")
    logits_eval, probe_metadata = train_linear_multilabel_probe(
        train_embeddings,
        train_labels,
        eval_embeddings,
        epochs=probe_epochs,
        learning_rate=probe_learning_rate,
        weight_decay=probe_weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
        log_prefix=f"[reben:probe:{run_name}]",
    )
    # Validation-only threshold policy: for the formal validation eval path,
    # thresholds are selected on the same validation split and must be frozen
    # before any future final test-set evaluation.
    y_prob_eval = sigmoid(logits_eval)
    thresholds = select_thresholds_from_validation(eval_labels, y_prob_eval)
    names = list(class_names or default_reben_class_names()[: eval_labels.shape[1]])
    cache_path = output / f"embeddings_{run_name}.npz"
    print(f"[reben:croma] {run_name}: writing embedding cache {cache_path}")
    np.savez_compressed(
        cache_path,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        eval_embeddings=eval_embeddings,
        eval_labels=eval_labels,
        eval_logits=logits_eval,
        train_sample_ids=np.asarray([str(row.get("sample_id", index)) for index, row in enumerate(train_metadata)]),
        eval_sample_ids=np.asarray([str(row.get("sample_id", index)) for index, row in enumerate(eval_metadata)]),
    )
    print(f"[reben:croma] {run_name}: writing metrics, predictions, and BWER outputs")
    artifacts = write_run_outputs(
        output,
        run_name=run_name,
        sample_rows=eval_metadata,
        y_true=eval_labels,
        y_prob=y_prob_eval,
        thresholds=thresholds,
        class_names=names,
        run_labels=run_labels,
        probe_metadata={
            **probe_metadata,
            "embedding_cache": str(cache_path),
            "train_reben_loader": train_dataset.loader_info() if hasattr(train_dataset, "loader_info") else {},
            "eval_reben_loader": eval_dataset.loader_info() if hasattr(eval_dataset, "loader_info") else {},
        },
        run_bwer=run_bwer,
    )
    artifacts["embeddings"] = cache_path
    return artifacts


def _read_many(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if path.exists():
            rows.extend(read_csv_rows(path))
    return rows


def collect_reben_sensor_audit_outputs(output_dir: str | Path) -> dict[str, Path]:
    """Collect per-run Step 1 outputs into the required top-level package files."""
    output = ensure_dir(output_dir)
    print(f"[reben:collect] collecting per-run outputs in {output}")
    artifacts = {
        "aggregate_metrics": output / "aggregate_metrics.csv",
        "per_class_metrics": output / "per_class_metrics.csv",
        "bwer_summary": output / "bwer_summary.csv",
        "bwer_by_slice": output / "bwer_by_slice.csv",
        "support_sensitivity": output / "support_sensitivity.csv",
        "alpha_sensitivity": output / "alpha_sensitivity.csv",
        "missing_policy_sensitivity": output / "missing_policy_sensitivity.csv",
        "selective_risk_summary": output / "selective_risk_summary.csv",
        "sensor_mode_summary": output / "sensor_mode_summary.csv",
        "audit_report": output / "reports" / "audit_report.md",
        "aggregate_vs_bwer": output / "figures" / "aggregate_vs_bwer.png",
        "sensor_mode_comparison": output / "figures" / "sensor_mode_comparison.png",
        "country_bwer_summary": output / "figures" / "country_bwer_summary.png",
    }
    print("[reben:collect] writing aggregate_metrics.csv")
    write_csv(artifacts["aggregate_metrics"], _read_many(sorted(output.glob("aggregate_metrics_*.csv"))))
    print("[reben:collect] writing per_class_metrics.csv")
    write_csv(artifacts["per_class_metrics"], _read_many(sorted(output.glob("per_class_metrics_*.csv"))))
    bwer_dirs = sorted((output / "bwer").glob("*")) if (output / "bwer").exists() else []
    print(f"[reben:collect] collecting BWER outputs dirs={len(bwer_dirs)}")
    write_csv(artifacts["bwer_summary"], _read_many([path / "bwer_summary.csv" for path in bwer_dirs]))
    write_csv(artifacts["bwer_by_slice"], _read_many([path / "bwer_by_slice.csv" for path in bwer_dirs]))
    write_csv(artifacts["support_sensitivity"], _read_many([path / "support_sensitivity.csv" for path in bwer_dirs]))
    write_csv(artifacts["alpha_sensitivity"], _read_many([path / "alpha_sensitivity.csv" for path in bwer_dirs]))
    write_csv(artifacts["missing_policy_sensitivity"], _read_many([path / "missing_policy_sensitivity.csv" for path in bwer_dirs]))
    write_csv(artifacts["selective_risk_summary"], _read_many([path / "selective_risk_summary.csv" for path in bwer_dirs]))
    write_csv(artifacts["sensor_mode_summary"], _sensor_mode_cross_run_summary(read_csv_rows(artifacts["aggregate_metrics"]) if artifacts["aggregate_metrics"].exists() else []))
    print("[reben:collect] writing reports and figures")
    _write_reben_audit_report(artifacts["audit_report"], artifacts["aggregate_metrics"], artifacts["bwer_summary"])
    _plot_reben_figures(artifacts)
    print("[reben:collect] finished top-level collection")
    return artifacts


def _sensor_mode_cross_run_summary(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize sensor mode as a cross-run condition, not a sample slice."""
    rows = [dict(row) for row in aggregate_rows if row.get("sensor_mode")]
    output: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("model", row.get("model_variant", "model"))), []).append(row)
    for model, items in grouped.items():
        risks = []
        for row in items:
            macro_ap = _float_or_nan(row.get("macro_ap"))
            risk = float("nan") if math.isnan(macro_ap) else 1.0 - macro_ap
            risks.append(risk)
            output.append(
                {
                    "model": model,
                    "run_name": row.get("run_name", ""),
                    "sensor_mode": row.get("sensor_mode", ""),
                    "macro_ap": row.get("macro_ap", ""),
                    "macro_f1": row.get("macro_f1", ""),
                    "cross_run_mode_risk": risk,
                    "definition": "cross_run_sensor_mode_risk=1-macro_ap; sensor_mode is not a per-sample slice",
                }
            )
        valid = [risk for risk in risks if not math.isnan(risk)]
        if len(valid) >= 2:
            output.append(
                {
                    "model": model,
                    "run_name": "sensor_mode_cross_run_bwer",
                    "sensor_mode": "cross_run",
                    "macro_ap": "",
                    "macro_f1": "",
                    "cross_run_mode_risk": "",
                    "cross_run_mode_bwer": float(max(valid) - float(np.mean(valid))),
                    "definition": "max(1-macro_ap across sensor modes) - mean(1-macro_ap across sensor modes)",
                }
            )
    return output


def _write_reben_audit_report(path: Path, aggregate_path: Path, bwer_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = read_csv_rows(aggregate_path) if aggregate_path.exists() else []
    bwer = read_csv_rows(bwer_path) if bwer_path.exists() else []
    lines = [
        "# reBEN / CROMA Sensor-Mode Audit Report",
        "",
        "This report is generated from completed BigEarthNet v2.0 / reBEN sensor-mode runs.",
        "Sensor mode is treated as a cross-run experimental condition, not a per-sample slice.",
        "",
        "Primary risk primitive: label-wise BCE risk. Secondary diagnostic primitive: thresholded label-wise binary error.",
        "",
        f"Aggregate metric rows: {len(aggregate)}.",
        f"BWER summary rows: {len(bwer)}.",
    ]
    if aggregate:
        lines.extend(["", "## Aggregate Rows"])
        for row in aggregate:
            lines.append(
                f"- {row.get('run_name', '')}: sensor_mode={row.get('sensor_mode', '')}, "
                f"macro_ap={row.get('macro_ap', '')}, macro_f1={row.get('macro_f1', '')}."
            )
        lines.extend(
            [
                "",
                "## Sensor-Mode Cross-Run Definition",
                "",
                "Sensor mode is not used as a sample-level slice. Any sensor-mode BWER is defined only as a cross-run mode-level diagnostic over completed S1/S2/S1+S2 rows.",
            ]
        )
    if bwer:
        lines.extend(["", "## BWER Rows"])
        for row in bwer:
            if row.get("slice_variable") in {"country", "class_label"} and not row.get("balance_variable"):
                lines.append(
                    f"- {row.get('model', '')} BWER({row.get('slice_variable', '')})={row.get('bwer', '')}; "
                    f"worst={row.get('worst_slice', '')}; tail={row.get('tail_slices', '')}."
                )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Do not use single-label accuracy-style risk for this multi-label audit.",
            "- Country x class is diagnostic unless support thresholds are satisfied.",
            "- CROMA and BIFOLD ResNet101 comparisons are protocol-aware, not architecture-only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _plot_reben_figures(artifacts: Mapping[str, Path]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    aggregate = read_csv_rows(artifacts["aggregate_metrics"]) if artifacts["aggregate_metrics"].exists() else []
    bwer = read_csv_rows(artifacts["bwer_summary"]) if artifacts["bwer_summary"].exists() else []
    figures = ensure_dir(artifacts["aggregate_vs_bwer"].parent)
    country = [row for row in bwer if row.get("slice_variable") == "country" and not row.get("balance_variable")]
    bwer_by_model = {row.get("model"): _float_or_nan(row.get("bwer")) for row in country}
    if aggregate and country:
        xs = [_float_or_nan(row.get("macro_ap")) for row in aggregate]
        ys = [bwer_by_model.get(row.get("run_name"), float("nan")) for row in aggregate]
        labels = [row.get("run_name", "") for row in aggregate]
        plt.figure(figsize=(6, 4))
        plt.scatter(xs, ys)
        for x, y, label in zip(xs, ys, labels):
            if not math.isnan(x) and not math.isnan(y):
                plt.annotate(label, (x, y), fontsize=8)
        plt.xlabel("macro AP")
        plt.ylabel("Raw-BWER(country)")
        plt.tight_layout()
        plt.savefig(artifacts["aggregate_vs_bwer"], dpi=160)
        plt.close()
    if aggregate:
        labels = [row.get("run_name", "") for row in aggregate]
        values = [_float_or_nan(row.get("macro_ap")) for row in aggregate]
        plt.figure(figsize=(max(6, len(labels) * 1.2), 4))
        plt.bar(labels, values)
        plt.ylabel("macro AP")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(artifacts["sensor_mode_comparison"], dpi=160)
        plt.close()
    if country:
        labels = [row.get("model", "") for row in country]
        values = [_float_or_nan(row.get("bwer")) for row in country]
        plt.figure(figsize=(max(6, len(labels) * 1.2), 4))
        plt.bar(labels, values)
        plt.ylabel("Raw-BWER(country)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(artifacts["country_bwer_summary"], dpi=160)
        plt.close()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reben_sensor_audit_contract(output_dir: str | Path) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    required_top_level = [
        "source_verification_report.md",
        "dataset_preflight.json",
        "split_summary.csv",
        "class_support.csv",
        "country_support.csv",
        "aggregate_metrics.csv",
        "per_class_metrics.csv",
        "bwer_summary.csv",
        "bwer_by_slice.csv",
        "support_sensitivity.csv",
        "alpha_sensitivity.csv",
        "missing_policy_sensitivity.csv",
        "reports/audit_report.md",
        "reports/support_preflight.md",
        "reports/metric_primitives.md",
        "reports/protocol_risk.md",
        "reports/model_protocol.md",
        "figures/aggregate_vs_bwer.png",
        "figures/sensor_mode_comparison.png",
        "figures/country_bwer_summary.png",
    ]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for rel in required_top_level:
        path = output / rel
        present = path.exists() and path.stat().st_size > 0
        if not present:
            missing.append(rel)
        rows.append({"artifact": rel, "present": present, "size_bytes": path.stat().st_size if path.exists() else 0})
    for run_name in REBEN_REQUIRED_RUNS:
        for rel in [
            f"run_metadata_{run_name}.json",
            f"thresholds_{run_name}.json",
            f"predictions_{run_name}.csv",
        ]:
            path = output / rel
            present = path.exists() and path.stat().st_size > 0
            if not present:
                missing.append(rel)
            rows.append({"artifact": rel, "present": present, "size_bytes": path.stat().st_size if path.exists() else 0})
    report = [
        "# reBEN / CROMA Sensor-Mode Audit Contract Validation",
        "",
        f"Output directory: `{output}`",
        f"Required runs: {', '.join(REBEN_REQUIRED_RUNS)}",
        "",
        f"Missing/blocking artifacts: {len(missing)}",
    ]
    if missing:
        report.extend(["", "## Missing", ""])
        report.extend([f"- `{item}`" for item in missing])
    else:
        report.extend(["", "All required artifacts are present."])
    artifacts = {
        "contract_validation": output / "reben_contract_validation.csv",
        "contract_report": output / "reben_contract_validation.md",
        "archive_manifest": output / "archive_manifest.json",
    }
    write_csv(artifacts["contract_validation"], rows)
    artifacts["contract_report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest_files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.suffix.lower() not in {".npz", ".pt", ".pth", ".safetensors"}:
            manifest_files.append(
                {
                    "path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    artifacts["archive_manifest"].write_text(
        json.dumps(
            {
                "dataset": REBEN_DATASET,
                "task": REBEN_TASK,
                "required_runs": list(REBEN_REQUIRED_RUNS),
                "ready_for_interpretation": not missing,
                "missing_artifacts": missing,
                "files": manifest_files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifacts


def write_reben_blocked_report(output_dir: str | Path, reason: str, details: Mapping[str, Any] | None = None) -> Path:
    output = ensure_dir(output_dir)
    path = output / "blocked_report.md"
    lines = [
        "# reBEN / CROMA Sensor-Mode Audit Blocked Report",
        "",
        f"Reason: {reason}",
        "",
        "This runner refuses to substitute BigEarthNet v1, BEN-GE pilots, torchvision ResNet101, or single-label BWER primitives.",
    ]
    if details:
        lines.extend(["", "## Details", ""])
        for key, value in details.items():
            lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def expand_predictions_to_label_audit_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Sequence[float],
    run_labels: RebenRunLabels,
    class_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)
    threshold_arr = np.asarray(thresholds, dtype=float)
    if len(sample_rows) != labels.shape[0] or labels.shape != probs.shape:
        raise ValueError("sample_rows, labels, and probabilities have inconsistent shapes.")
    if labels.ndim != 2:
        raise ValueError("reBEN audit rows require multi-label [n_samples, n_labels] arrays.")
    names = list(class_names or default_reben_class_names()[: labels.shape[1]])
    bce = binary_cross_entropy(labels, probs)
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(sample_rows):
        base = {
            "sample_id": sample.get("sample_id") or sample.get("patch_id") or f"sample_{sample_index:06d}",
            "patch_id": sample.get("patch_id", sample.get("sample_id", "")),
            "dataset": REBEN_DATASET,
            "task": REBEN_TASK,
            "split": sample.get("split", run_labels.eval_scope),
            "country": sample.get("country", ""),
            "cloud_snow_shadow": sample.get("cloud_snow_shadow", sample.get("snow_cloud_shadow", "")),
            "sensor_mode": run_labels.sensor_mode,
            "model_family": run_labels.model_family,
            "model_variant": run_labels.model_variant,
            "input_mode": run_labels.input_mode,
            "adaptation_protocol": run_labels.adaptation_protocol,
            "split_protocol": run_labels.split_protocol,
            "eval_scope": run_labels.eval_scope,
            "band_profile": run_labels.band_profile,
        }
        for class_index, class_label in enumerate(names):
            probability = float(probs[sample_index, class_index])
            threshold = float(threshold_arr[class_index])
            predicted = int(probability >= threshold)
            truth = int(labels[sample_index, class_index])
            confidence = max(probability, 1.0 - probability)
            rows.append(
                {
                    **base,
                    "class_index": class_index,
                    "class_label": class_label,
                    "label_true": truth,
                    "label_probability": probability,
                    "label_prediction": predicted,
                    "threshold": threshold,
                    "correct": int(predicted == truth),
                    "risk_bce": float(bce[sample_index, class_index]),
                    "risk_binary_error": int(predicted != truth),
                    "confidence": confidence,
                }
            )
    return rows


def validate_multilabel_audit_rows(rows: Sequence[Mapping[str, Any]], risk_column: str = "risk_bce") -> None:
    if not rows:
        raise ValueError("reBEN multi-label audit table is empty.")
    required = {"sample_id", "class_label", "label_true", "label_probability", risk_column}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"reBEN multi-label BWER requires label-expanded rows; missing columns: {', '.join(missing)}")
    if "prediction" in rows[0] and "class_label" not in rows[0]:
        raise ValueError("Single-label prediction tables are invalid for reBEN multi-label BWER.")


def run_reben_multilabel_bwer(
    audit_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    model_name: str,
    split: str = "validation",
    risk_column: str = "risk_bce",
    alpha: float = 0.1,
    min_support: int = 20,
) -> dict[str, Path]:
    validate_multilabel_audit_rows(audit_rows, risk_column=risk_column)
    output = ensure_dir(output_dir)
    rows = [dict(row) for row in audit_rows]
    for left, right, name in REBEN_INTERACTION_SLICES:
        if left in rows[0] and right in rows[0]:
            rows = create_interaction_slice(rows, [left, right], name)
    base_config = BWERConfig(
        dataset=REBEN_DATASET,
        model=model_name,
        task=REBEN_TASK,
        split=split,
        tail_fraction=alpha,
        min_samples_per_slice=min_support,
        min_slices_required=2,
        risk_name=risk_column,
        score_name="correct",
        missing_balance_policy="renormalize",
    )
    slice_variables = [name for name in ["class_label", "country", "country_x_class", "cloud_snow_shadow"] if name in rows[0]]
    balance_variables = [None]
    summary, by_slice, support, _, warnings = compute_bwer_family(
        rows,
        base_config,
        slice_variables=slice_variables,
        balance_variables=balance_variables,
        score_column="correct",
        risk_column=risk_column,
    )
    std_summary, std_by_slice, std_support, _, std_warnings = compute_bwer_family(
        rows,
        base_config,
        slice_variables=[slice_name for slice_name, balance in REBEN_STANDARDISED if slice_name in rows[0] and balance in rows[0]],
        balance_variables=["class_label"] if "class_label" in rows[0] else [],
        score_column="correct",
        risk_column=risk_column,
    )
    summary.extend(std_summary)
    by_slice.extend(std_by_slice)
    support.extend(std_support)
    warnings.extend(std_warnings)
    artifacts = {
        "bwer_summary": output / "bwer_summary.csv",
        "bwer_by_slice": output / "bwer_by_slice.csv",
        "support_diagnostics": output / "support_diagnostics.csv",
        "support_sensitivity": output / "support_sensitivity.csv",
        "alpha_sensitivity": output / "alpha_sensitivity.csv",
        "missing_policy_sensitivity": output / "missing_policy_sensitivity.csv",
        "warnings": output / "warnings.json",
    }
    write_csv(artifacts["bwer_summary"], summary)
    write_csv(artifacts["bwer_by_slice"], by_slice)
    write_csv(artifacts["support_diagnostics"], support)
    write_csv(artifacts["support_sensitivity"], _support_sensitivity(rows, base_config, risk_column))
    write_csv(artifacts["alpha_sensitivity"], _alpha_sensitivity(rows, base_config, risk_column))
    write_csv(artifacts["missing_policy_sensitivity"], _missing_policy_sensitivity(rows, base_config, risk_column))
    artifacts["warnings"].write_text(json.dumps(sorted(set(warnings)), indent=2), encoding="utf-8")
    return artifacts


def _support_sensitivity(rows: Sequence[Mapping[str, Any]], config: BWERConfig, risk_column: str) -> list[dict[str, Any]]:
    output = []
    for support in REBEN_SUPPORT_VALUES:
        cfg = BWERConfig(**{**config.__dict__, "min_samples_per_slice": support})
        summary, _, _, _, _ = compute_bwer_family(rows, cfg, ["country"], [None], score_column="correct", risk_column=risk_column)
        for row in summary:
            output.append({**row, "sensitivity_type": "min_support", "sensitivity_value": support})
    return output


def _alpha_sensitivity(rows: Sequence[Mapping[str, Any]], config: BWERConfig, risk_column: str) -> list[dict[str, Any]]:
    output = []
    for alpha in REBEN_ALPHA_VALUES:
        cfg = BWERConfig(**{**config.__dict__, "tail_fraction": alpha})
        summary, _, _, _, _ = compute_bwer_family(rows, cfg, ["country", "class_label"], [None], score_column="correct", risk_column=risk_column)
        for row in summary:
            output.append({**row, "sensitivity_type": "alpha", "sensitivity_value": alpha})
    return output


def _missing_policy_sensitivity(rows: Sequence[Mapping[str, Any]], config: BWERConfig, risk_column: str) -> list[dict[str, Any]]:
    output = []
    if "country" not in rows[0] or "class_label" not in rows[0]:
        return output
    for policy in REBEN_MISSING_POLICIES:
        cfg = BWERConfig(**{**config.__dict__, "missing_balance_policy": policy})
        summary, _, _, _, _ = compute_bwer_family(rows, cfg, ["country"], ["class_label"], score_column="correct", risk_column=risk_column)
        for row in summary:
            output.append({**row, "sensitivity_type": "missing_policy", "sensitivity_value": policy})
    return output


def compute_selective_risk(
    audit_rows: Sequence[Mapping[str, Any]],
    coverages: Sequence[float] = (0.8, 0.7, 0.9),
    risk_column: str = "risk_bce",
    confidence_column: str = "confidence",
    slice_columns: Sequence[str] = ("country", "class_label"),
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in audit_rows if confidence_column in row and risk_column in row]
    if not rows:
        return []
    confidences = np.asarray([float(row[confidence_column]) for row in rows], dtype=float)
    risks = np.asarray([float(row[risk_column]) for row in rows], dtype=float)
    output: list[dict[str, Any]] = []
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            continue
        threshold = float(np.quantile(confidences, 1.0 - coverage))
        retained = confidences >= threshold
        output.append(
            {
                "coverage_target": coverage,
                "slice_variable": "all",
                "slice_value": "all",
                "confidence_threshold": threshold,
                "retained_count": int(np.sum(retained)),
                "total_count": int(len(rows)),
                "retained_coverage": float(np.mean(retained)),
                "abstention_rate": float(1.0 - np.mean(retained)),
                "mean_risk": float(np.mean(risks[retained])) if np.any(retained) else float("nan"),
            }
        )
        for column in slice_columns:
            if column not in rows[0]:
                continue
            for value in sorted({str(row.get(column)) for row in rows if row.get(column) not in {None, ""}}):
                idx = np.asarray([str(row.get(column)) == value for row in rows], dtype=bool)
                kept = idx & retained
                output.append(
                    {
                        "coverage_target": coverage,
                        "slice_variable": column,
                        "slice_value": value,
                        "confidence_threshold": threshold,
                        "retained_count": int(np.sum(kept)),
                        "total_count": int(np.sum(idx)),
                        "retained_coverage": float(np.sum(kept) / max(1, np.sum(idx))),
                        "abstention_rate": float(1.0 - (np.sum(kept) / max(1, np.sum(idx)))),
                        "mean_risk": float(np.mean(risks[kept])) if np.any(kept) else float("nan"),
                    }
                )
    return output


def read_label_expanded_predictions(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    validate_multilabel_audit_rows(rows)
    return rows


def _parse_label_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        data = json.loads(text.replace("'", '"'))
        if isinstance(data, list):
            return [str(item) for item in data]
    except Exception:
        pass
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def default_reben_class_names() -> list[str]:
    return list(REBEN_19_CLASS_NAMES)


def read_reben_metadata(path: str | Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Read reBEN metadata from parquet when pandas is available, or CSV for tests."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"reBEN metadata does not exist: {metadata_path}")
    if metadata_path.suffix.lower() == ".csv":
        rows = [dict(row) for row in read_csv_rows(metadata_path)]
    elif metadata_path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on Colab/runtime extras
            raise RuntimeError("Reading official reBEN metadata.parquet requires pandas + a parquet engine.") from exc
        frame = pd.read_parquet(metadata_path)
        if max_rows:
            frame = frame.head(int(max_rows))
        rows = frame.to_dict(orient="records")
    else:
        raise ValueError("reBEN metadata must be .parquet for official runs or .csv for lightweight tests.")
    return rows[: int(max_rows)] if max_rows else rows


def run_reben_dataset_preflight(
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    max_rows: int | None = None,
) -> dict[str, Path]:
    rows = read_reben_metadata(metadata_path, max_rows=max_rows)
    output = ensure_dir(output_dir)
    artifacts = {
        "dataset_preflight": output / "dataset_preflight.json",
        "split_summary": output / "split_summary.csv",
        "class_support": output / "class_support.csv",
        "country_support": output / "country_support.csv",
    }
    split_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    country_counts: dict[str, int] = {}
    missing_country = 0
    missing_labels = 0
    for row in rows:
        split = str(row.get("split", "missing") or "missing")
        split_counts[split] = split_counts.get(split, 0) + 1
        country = str(row.get("country", "") or "")
        if country:
            country_counts[country] = country_counts.get(country, 0) + 1
        else:
            missing_country += 1
        labels = _parse_label_list(row.get("labels", row.get("label", row.get("class_labels"))))
        if not labels:
            missing_labels += 1
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
    write_csv(
        artifacts["split_summary"],
        [{"split": split, "sample_count": count} for split, count in sorted(split_counts.items())],
    )
    write_csv(
        artifacts["class_support"],
        [{"class_label": label, "positive_support": count} for label, count in sorted(class_counts.items())],
    )
    write_csv(
        artifacts["country_support"],
        [{"country": country, "sample_count": count} for country, count in sorted(country_counts.items())],
    )
    preflight = {
        "dataset": REBEN_DATASET,
        "metadata_path": str(metadata_path),
        "rows_read": len(rows),
        "label_count_observed": len(class_counts),
        "expected_label_count": REBEN_LABEL_COUNT,
        "country_count_observed": len(country_counts),
        "missing_country_rows": missing_country,
        "missing_label_rows": missing_labels,
        "split_values": sorted(split_counts),
        "status": "ok" if class_counts and len(class_counts) == REBEN_LABEL_COUNT else "check_label_count",
    }
    artifacts["dataset_preflight"].write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    return artifacts


def write_reben_source_verification_report(output_dir: str | Path) -> Path:
    output = ensure_dir(output_dir)
    path = output / "source_verification_report.md"
    lines = [
        "# reBEN / CROMA Sensor-Mode Audit Source Verification",
        "",
        "Status: verified enough to proceed with a protocol-risk-aware implementation.",
        "",
        "This report is generated for the current run output directory so the evidence package is self-contained.",
        "",
        "## Sources Checked",
        "",
    ]
    lines.extend([f"- {url}" for url in SOURCE_VERIFICATION_URLS])
    lines.extend(
        [
            "",
            "## Verified Protocol Facts",
            "",
        ]
    )
    for key, value in SOURCE_VERIFICATION_SUMMARY.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Use BigEarthNet v2.0 / reBEN, not BigEarthNet v1 or old pilots.",
            "- Use multi-label metrics and label-expanded BWER primitives.",
            "- Use official CROMA implementation/checkpoint for CROMA rows.",
            "- Use official BIFOLD ResNet101 v0.2.0 refs through reBEN publication code; do not substitute torchvision ResNet101.",
            "- Treat sensor_mode as cross-run experimental mode, not a per-sample slice.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def croma_embedding_from_output(outputs: Mapping[str, Any], sensor_mode: str) -> np.ndarray:
    if sensor_mode not in REBEN_CROMA_EMBEDDING_KEYS:
        raise ValueError("sensor_mode must be one of S1, S2, S1+S2.")
    key = REBEN_CROMA_EMBEDDING_KEYS[sensor_mode]
    if key not in outputs:
        raise ValueError(f"CROMA output missing {key}; available keys: {sorted(outputs)}")
    values = outputs[key]
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"CROMA {key} output must have a batch dimension and feature dimension, got {array.shape}.")
    return array.reshape(array.shape[0], -1)
