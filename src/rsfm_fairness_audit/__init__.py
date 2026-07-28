"""Fairness auditing tools for Remote Sensing Foundation Models."""

from rsfm_fairness_audit.bwer_protocol import BWERProtocol, Protocol, Validity
from rsfm_fairness_audit.geobwer import audit, audit_rows, compare, confirm
from rsfm_fairness_audit.geobwer_panel import run_geobwer_model_panel
from rsfm_fairness_audit.geobwer_extensions import (
    run_multiclass_spatial_upgrade,
    run_multiclass_uncertainty_suite,
    run_multilabel_uncertainty_suite,
    run_segmentation_uncertainty_suite,
)
from rsfm_fairness_audit.spatial_conformal import SpatialConformalConfig

__all__ = [
    "__version__",
    "BWERProtocol",
    "Protocol",
    "Validity",
    "audit",
    "audit_rows",
    "compare",
    "confirm",
    "run_geobwer_model_panel",
    "run_multiclass_spatial_upgrade",
    "run_multiclass_uncertainty_suite",
    "run_multilabel_uncertainty_suite",
    "run_segmentation_uncertainty_suite",
    "SpatialConformalConfig",
]

__version__ = "0.4.19"
