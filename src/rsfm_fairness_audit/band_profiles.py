from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


BAND_PROFILES: dict[str, dict[str, Any]] = {
    "sentinel2_12_lccol": {
        "expected_bands": 12,
        "band_names": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
        "wavelength_list": [0.443, 0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 0.945, 1.61, 2.19],
        # lc-col exports TorchGeo-style Sentinel-2 chips. Use identity normalization
        # until a run-specific empirical 12-band normalization is documented.
        "normalization_mean": [0.0] * 12,
        "normalization_std": [1.0] * 12,
    },
    "sentinel2_10_bigearthnet_v2": {
        "expected_bands": 10,
        "band_names": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
        "wavelength_list": [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19],
        "normalization_mean": [0.0] * 10,
        "normalization_std": [1.0] * 10,
    },
    "sentinel2_13band_fmow": {
        "expected_bands": 13,
        "band_names": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
        "wavelength_list": [0.443, 0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 0.945, 1.373, 1.61, 2.19],
        # fMoW-Sentinel Step 3 starts with robust image-only prototype runs.
        # Use identity normalization until a run-specific empirical 13-band
        # normalization is computed and documented.
        "normalization_mean": [0.0] * 13,
        "normalization_std": [1.0] * 13,
    },
    "sentinel2_9_legacy": {
        "expected_bands": 9,
        "band_names": ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"],
        "wavelength_list": [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19],
        "normalization_mean": [
            114.1099739,
            114.81779093,
            126.63977424,
            84.33539309,
            97.84789168,
            103.94461911,
            101.435633,
            72.32804172,
            56.66528851,
        ],
        "normalization_std": [
            77.84352553,
            69.96844919,
            67.42465279,
            64.57022983,
            61.72545487,
            61.34187099,
            60.29744676,
            47.88519516,
            42.55886798,
        ],
    },
}


class BandProfileError(ValueError):
    """Raised when a band profile is missing or internally inconsistent."""


def get_band_profile(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    if name not in BAND_PROFILES:
        raise BandProfileError(f"Unknown band_profile={name!r}. Available profiles: {sorted(BAND_PROFILES)}")
    profile = deepcopy(BAND_PROFILES[name])
    validate_band_profile(name, profile)
    return profile


def validate_band_profile(name: str, profile: Mapping[str, Any]) -> None:
    expected = int(profile["expected_bands"])
    for key in ["band_names", "wavelength_list", "normalization_mean", "normalization_std"]:
        values = profile.get(key)
        if not isinstance(values, list) or len(values) != expected:
            raise BandProfileError(
                f"band_profile={name!r} has expected_bands={expected} but {key} length is "
                f"{len(values) if isinstance(values, list) else 'missing'}."
            )
