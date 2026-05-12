from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def balanced_indices(
    metadata: Sequence[dict],
    keys: Sequence[str] = ("region", "label"),
    max_per_group: int | None = None,
    seed: int = 11,
) -> np.ndarray:
    """Return indices balanced by the requested metadata keys."""

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        group = tuple(str(row[key]) for key in keys)
        groups[group].append(index)

    if not groups:
        return np.asarray([], dtype=np.int64)

    target = min(len(indices) for indices in groups.values())
    if max_per_group is not None:
        target = min(target, max_per_group)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for indices in groups.values():
        selected.extend(rng.choice(indices, size=target, replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)
