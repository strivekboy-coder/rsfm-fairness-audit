from __future__ import annotations

import pytest

from rsfm_fairness_audit.fmow_formal_split import (
    FmowFormalSplitError,
    build_fmow_formal_split,
    fmow_site_id,
)


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for category in ("airport", "port"):
        for split, sites in (("train", ("0", "1")), ("val", ("2", "3", "4", "5"))):
            for site in sites:
                for image in range(1 + (int(site) % 2)):
                    rows.append(
                        {
                            "sample_id": f"{category}-{site}-{image}",
                            "category": category,
                            "location_id": site,
                            "country": "DEU" if site in {"0", "2", "4"} else "FRA",
                            "split": split,
                        }
                    )
    return rows


def test_fmow_site_id_is_category_scoped() -> None:
    assert fmow_site_id({"category": "airport", "location_id": "7"}) == "airport|7"
    assert fmow_site_id({"category": "port", "location_id": "7"}) == "port|7"


def test_formal_split_is_deterministic_disjoint_and_class_complete() -> None:
    first = build_fmow_formal_split(_rows(), seed=9)
    second = build_fmow_formal_split(_rows(), seed=9)
    assert [row["split"] for row in first.rows] == [row["split"] for row in second.rows]
    sites = {
        split: {row["site_id"] for row in first.rows if row["split"] == split}
        for split in ("train", "calibration", "test")
    }
    assert sites["train"].isdisjoint(sites["calibration"] | sites["test"])
    assert sites["calibration"].isdisjoint(sites["test"])
    for split in ("calibration", "test"):
        assert {row["category"] for row in first.rows if row["split"] == split} == {"airport", "port"}
    assert first.contract["site_overlap_calibration_test"] == 0


def test_formal_split_rejects_class_with_one_holdout_site() -> None:
    rows = [
        {"sample_id": "a", "category": "airport", "location_id": "1", "split": "train"},
        {"sample_id": "b", "category": "airport", "location_id": "2", "split": "val"},
    ]
    with pytest.raises(FmowFormalSplitError, match="only 1 holdout site"):
        build_fmow_formal_split(rows)
