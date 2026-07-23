from __future__ import annotations

import zipfile
from pathlib import Path

from rsfm_fairness_audit.drive_artifact_preflight import decide_execution, inspect_drive_artifacts


FIXTURE_ROOT = Path("work/test_artifacts/drive_artifact_preflight")


def test_drive_preflight_reads_zip_headers_without_extraction() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    archive = FIXTURE_ROOT / "fmow_outputs.zip"
    header = "sample_id,location_id,split," + ",".join(f"prob_{index}" for index in range(62)) + "\n"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("fmow_predictions.csv", header + "s1,l1,test," + ",".join(["0"] * 62) + "\n")
        handle.writestr("arrays/probabilities.npz", b"not-loaded-by-preflight")
    members, headers, warnings = inspect_drive_artifacts([archive])
    assert not warnings
    assert any(record.member.endswith("probabilities.npz") for record in members)
    [fmow_header] = [record for record in headers if "fmow_predictions" in record.source]
    assert fmow_header.full_probability_columns == 62
    decisions = decide_execution(members, headers)
    fmow = next(record for record in decisions if record.task == "fMoW-Sentinel/DOFAv2")
    assert fmow.decision == "reuse_complete_predictions"


def test_alphaearth_decision_requires_coordinates() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    table = FIXTURE_ROOT / "alphaearth_all_split_predictions.csv"
    table.write_text("sample_id,split,latitude,longitude,prob_a,prob_b\ns,test,0,0,0.5,0.5\n", encoding="utf-8")
    members, headers, warnings = inspect_drive_artifacts([table])
    assert not members
    assert not warnings
    alpha = next(record for record in decide_execution(members, headers) if record.task == "AlphaEarth/WorldCover")
    assert alpha.decision == "postprocess_existing_probabilities"
