"""Axis helpers shared by every geometry routine.

The classic backend hard-coded ``z`` in some places and ``y`` in others.
Everything here goes through :func:`axis_index` / :func:`horizontal_axes`
so a model can declare its up axis once and have it respected end to end.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

_AXES = {"x": 0, "y": 1, "z": 2}


def axis_index(up_axis: str) -> int:
    """Index of the vertical axis. Defaults to ``z`` for unknown input."""
    return _AXES.get((up_axis or "z").lower(), 2)


def horizontal_axes(up_axis: str) -> Tuple[int, int]:
    """The two axis indices that form the horizontal plane, in ascending order."""
    up = axis_index(up_axis)
    return tuple(i for i in (0, 1, 2) if i != up)  # type: ignore[return-value]


def bounds_of(points: Sequence[Sequence[float]]) -> List[List[float]]:
    """Axis-aligned bounds as ``[[minx, miny, minz], [maxx, maxy, maxz]]``."""
    if not points:
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for p in points:
        for a in range(3):
            value = p[a]
            if value < lo[a]:
                lo[a] = value
            if value > hi[a]:
                hi[a] = value
    return [lo, hi]
