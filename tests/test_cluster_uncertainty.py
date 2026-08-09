from __future__ import annotations

import numpy as np
import pytest

from rsfm_fairness_audit.cluster_uncertainty import (
    cluster_hoeffding_crc_lac,
    cluster_max_lac,
)


def _bundle(n: int = 40):
    p = np.tile(np.asarray([[0.8, 0.2], [0.3, 0.7]]), (n // 2, 1))
    y = np.tile(np.asarray([0, 1]), n // 2)
    return p, y


def test_cluster_max_lac_uses_disjoint_cluster_units() -> None:
    cp, cy = _bundle(); tp, ty = _bundle()
    result = cluster_max_lac(
        cp, cy, [f"c{i // 2}" for i in range(40)],
        tp, ty, [f"t{i // 2}" for i in range(40)], alpha=0.1,
    )
    assert result.calibration_cluster_count == 20
    assert result.evidence_status == "formal_confirmed"
    assert result.marginal_coverage == 1.0


def test_cluster_methods_reject_calibration_test_cluster_overlap() -> None:
    p, y = _bundle()
    with pytest.raises(ValueError, match="disjoint"):
        cluster_max_lac(p, y, [f"c{i}" for i in range(40)], p, y, [f"c{i}" for i in range(40)])


def test_cluster_crc_reports_nonidentification_instead_of_claiming_control() -> None:
    cp, cy = _bundle(); tp, ty = _bundle()
    result = cluster_hoeffding_crc_lac(
        cp, cy, [f"c{i // 2}" for i in range(40)],
        tp, ty, [f"t{i // 2}" for i in range(40)], alpha=0.01,
    )
    assert result.evidence_status == "not_identified"
    assert result.threshold == 1.0
    assert result.mean_set_size == 2.0


def test_failed_spatial_design_is_descriptive_only() -> None:
    cp, cy = _bundle(); tp, ty = _bundle()
    result = cluster_max_lac(
        cp, cy, [f"c{i // 2}" for i in range(40)],
        tp, ty, [f"t{i // 2}" for i in range(40)], cluster_design_valid=False,
    )
    assert result.evidence_status == "descriptive_only"
