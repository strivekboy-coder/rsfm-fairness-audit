from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rsfm_fairness_audit.adapters.terramind import (  # noqa: E402
    INPUT_PROFILES,
    S1_MEAN,
    S1_STD,
)
from rsfm_fairness_audit.formal_outputs import file_sha256  # noqa: E402
from rsfm_fairness_audit.sen1_input_quality import (  # noqa: E402
    SEN1_IMPUTATION_POLICY,
    normalize_named_modalities,
)
from rsfm_fairness_audit.sen1_supervised_campaign import (  # noqa: E402
    Sen1SupervisedConfig,
    _mask,
    _mode_array,
    _normalize_input,
    compute_train_normalization,
)
from rsfm_fairness_audit.terramind_sen1_config import (  # noqa: E402
    read_sen1floods11_split_prefixes,
)


EXPECTED_SAMPLE_ID = "Paraguay_34417"
EXPECTED_S1_SHA256 = (
    "1e495c046f50b200a40839161ea97aedcc59344a0e2e1a285c33f37b3e0b7e5a"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "No-training exact-source gate for the official Paraguay_34417 "
            "complete-S1-missing contract in U-Net and TerraMind preprocessing."
        )
    )
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--s2-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--bolivia-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", default=EXPECTED_SAMPLE_ID)
    parser.add_argument("--expected-s1-sha256", default=EXPECTED_S1_SHA256)
    return parser


def _assert_zero(value: np.ndarray, label: str) -> None:
    if not np.array_equal(value, np.zeros_like(value)):
        raise RuntimeError(f"{label} is not exactly normalized zero.")


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_id != EXPECTED_SAMPLE_ID:
        raise RuntimeError(
            f"The frozen exact-source gate requires {EXPECTED_SAMPLE_ID}."
        )
    s1_path = args.s1_root / f"{args.sample_id}_S1Hand.tif"
    if file_sha256(s1_path) != args.expected_s1_sha256:
        raise RuntimeError(
            "Official Paraguay S1 SHA-256 changed; do not apply the frozen "
            "evaluation-only policy to unverified bytes."
        )
    config = Sen1SupervisedConfig(
        s1_root=args.s1_root,
        s2_root=args.s2_root,
        label_root=args.label_root,
        train_split=args.train_split,
        validation_split=args.val_split,
        test_split=args.test_split,
        bolivia_split=args.bolivia_split,
        output_dir=args.output_dir,
        sensor_modes=("S1",),
        seeds=(42,),
        diagnostic_max_samples=1,
        device="cpu",
    )
    train_prefixes = read_sen1floods11_split_prefixes(args.train_split)
    if len(train_prefixes) != 252:
        raise RuntimeError(
            f"Expected the full official train252 normalization population, got {len(train_prefixes)}."
        )
    s1_normalization = compute_train_normalization(config, train_prefixes, "S1")
    fusion_normalization = compute_train_normalization(
        config, train_prefixes, "S1+S2"
    )
    raw_s1 = _mode_array(config, args.sample_id, "S1")
    raw_s2 = _mode_array(config, args.sample_id, "S2")
    raw_fusion = _mode_array(config, args.sample_id, "S1+S2")
    target = _mask(config, args.sample_id)
    if int(np.isin(target, [0, 1]).sum()) <= 0:
        raise RuntimeError(
            "Paraguay_34417 is not an all-ignore chip; its label risk must remain identifiable."
        )

    unet_s1, unet_s1_quality = _normalize_input(
        raw_s1,
        mean=np.asarray(s1_normalization["mean"]),
        std=np.asarray(s1_normalization["std"]),
        prefix=args.sample_id,
        mode="S1",
        split_role="standard_test",
    )
    unet_fusion, unet_fusion_quality = _normalize_input(
        raw_fusion,
        mean=np.asarray(fusion_normalization["mean"]),
        std=np.asarray(fusion_normalization["std"]),
        prefix=args.sample_id,
        mode="S1+S2",
        split_role="standard_test",
    )
    _assert_zero(unet_s1, "U-Net S1")
    _assert_zero(unet_fusion[:2], "U-Net fusion S1")
    expected_unet_s2 = (
        raw_s2
        - np.asarray(fusion_normalization["mean"], dtype=np.float32)[2:, None, None]
    ) / np.asarray(fusion_normalization["std"], dtype=np.float32)[2:, None, None]
    np.testing.assert_array_equal(unet_fusion[2:], expected_unet_s2)

    profile = INPUT_PROFILES["sen1floods11_l1c"]
    tm_s1, tm_s1_quality = normalize_named_modalities(
        {"S1GRD": raw_s1},
        means={"S1GRD": S1_MEAN},
        stds={"S1GRD": S1_STD},
        prefix=args.sample_id,
        split_role="standard_test",
    )
    tm_fusion, tm_fusion_quality = normalize_named_modalities(
        {"S2L1C": raw_s2, "S1GRD": raw_s1},
        means={"S2L1C": profile.s2_mean, "S1GRD": S1_MEAN},
        stds={"S2L1C": profile.s2_std, "S1GRD": S1_STD},
        prefix=args.sample_id,
        split_role="standard_test",
    )
    tm_s2, _ = normalize_named_modalities(
        {"S2L1C": raw_s2},
        means={"S2L1C": profile.s2_mean},
        stds={"S2L1C": profile.s2_std},
        prefix=args.sample_id,
        split_role="standard_test",
    )
    _assert_zero(tm_s1["S1GRD"], "TerraMind S1")
    _assert_zero(tm_fusion["S1GRD"], "TerraMind fusion S1")
    np.testing.assert_array_equal(
        tm_fusion["S2L1C"],
        tm_s2["S2L1C"],
    )
    for name, value in {
        "unet_s1": unet_s1,
        "unet_fusion": unet_fusion,
        "terramind_s1": tm_s1["S1GRD"],
        "terramind_fusion_s1": tm_fusion["S1GRD"],
        "terramind_fusion_s2": tm_fusion["S2L1C"],
    }.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{name} contains NaN/Inf after preprocessing.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "geobwer.sen1floods11.complete_modality_runtime_gate.v1",
        "status": "pass",
        "training_executed": False,
        "sample_id": args.sample_id,
        "split_role": "standard_test",
        "s1_path": str(s1_path),
        "s1_sha256": file_sha256(s1_path),
        "imputation_policy": SEN1_IMPUTATION_POLICY,
        "normalization_source": {
            "unet": "complete_official_train252",
            "terramind": "frozen_terramind_pretraining_statistics",
        },
        "label_valid_pixel_count": int(np.isin(target, [0, 1]).sum()),
        "unet": {
            "s1_all_zero": True,
            "fusion_s1_all_zero": True,
            "fusion_s2_matches_s2_normalization": True,
            "s1_quality": unet_s1_quality,
            "fusion_quality": unet_fusion_quality,
        },
        "terramind": {
            "s1_all_zero": True,
            "fusion_s1_all_zero": True,
            "fusion_s2_matches_s2_only": True,
            "s1_quality": tm_s1_quality,
            "fusion_quality": tm_fusion_quality,
        },
    }
    path = args.output_dir / "complete_modality_runtime_gate.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("SEN1_COMPLETE_MODALITY_RUNTIME_GATE=PASS")
    print(f"REPORT={path}")


if __name__ == "__main__":
    main()
