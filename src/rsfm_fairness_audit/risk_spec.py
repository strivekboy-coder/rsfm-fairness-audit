from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


RISK_SPEC_VERSION = "geobwer_risk_spec_v1"


@dataclass(frozen=True)
class RiskSpec:
    """Versioned definition of the loss that enters a GeoBWER audit.

    GeoBWER is a deviation functional over *risks*.  Formal certification is
    therefore only meaningful after the direction, bounds, aggregation unit,
    reference target, and any threshold/ignore policy have been frozen.
    """

    name: str = "risk"
    direction: str = "higher_is_worse"
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    unit: str = "independent_unit"
    aggregation: str = "mean_within_slice"
    reference: str = "task_reference"
    ignore_policy: str = "none"
    threshold_source: str = "not_applicable"
    task_adapter: str = "generic"
    class_mapping_hash: str = ""
    version: str = RISK_SPEC_VERSION
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("RiskSpec.name must be non-empty.")
        if self.direction != "higher_is_worse":
            raise ValueError(
                "Formal GeoBWER requires a loss/risk with direction=higher_is_worse; "
                "convert utility scores to a bounded loss before auditing."
            )
        low, high = float(self.lower_bound), float(self.upper_bound)
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("RiskSpec bounds must be finite and strictly ordered.")
        if not str(self.unit).strip() or not str(self.aggregation).strip():
            raise ValueError("RiskSpec unit and aggregation must be non-empty.")
        if self.version != RISK_SPEC_VERSION:
            raise ValueError(f"Unsupported RiskSpec version={self.version!r}.")

    @property
    def bounds(self) -> tuple[float, float]:
        return float(self.lower_bound), float(self.upper_bound)

    def validate_values(self, values: Any, *, tolerance: float = 1e-12) -> None:
        low, high = self.bounds
        for index, raw in enumerate(values):
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"RiskSpec {self.name}: non-finite risk at index={index}.")
            if value < low - tolerance or value > high + tolerance:
                raise ValueError(
                    f"RiskSpec {self.name}: risk={value} at index={index} is outside [{low}, {high}]."
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = {key: value for key, value in self.metadata}
        return payload

    @property
    def signature(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RiskSpec":
        data = dict(values)
        metadata = data.get("metadata", ())
        if isinstance(metadata, Mapping):
            data["metadata"] = tuple(sorted((str(key), str(value)) for key, value in metadata.items()))
        return cls(**data)
