"""Walls and doors: obstacle extraction and edge blocking.

The classic backend tested every candidate edge against every wall, inside the
Dijkstra relaxation loop. That is O(E * W) work repeated on every single path
query. Here walls are rasterised into a uniform 2D bin index once, and edge
blocking is resolved once at graph-build time into a boolean mask that path
queries simply read.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .common import axis_index, horizontal_axes


class WallField:
    """A queryable set of 2D wall centrelines with vertical extents."""

    __slots__ = ("segments", "up_axis", "_bins", "_cell", "_count")

    def __init__(self, segments: Sequence[dict], up_axis: str = "z", cell: float = 2.0):
        self.segments = list(segments)
        self.up_axis = up_axis
        self._cell = max(float(cell), 0.25)
        self._bins: Dict[Tuple[int, int], List[int]] = {}
        self._count = len(self.segments)
        self._index()

    def __len__(self) -> int:
        return self._count

    def _index(self) -> None:
        inv = 1.0 / self._cell
        for i, wall in enumerate(self.segments):
            (x0, y0), (x1, y1) = wall["segment"]
            bx0, bx1 = sorted((int(np.floor(x0 * inv)), int(np.floor(x1 * inv))))
            by0, by1 = sorted((int(np.floor(y0 * inv)), int(np.floor(y1 * inv))))
            for bx in range(bx0, bx1 + 1):
                for by in range(by0, by1 + 1):
                    self._bins.setdefault((bx, by), []).append(i)

    def _candidates(self, p: Tuple[float, float], q: Tuple[float, float]) -> Iterable[int]:
        inv = 1.0 / self._cell
        bx0, bx1 = sorted((int(np.floor(p[0] * inv)), int(np.floor(q[0] * inv))))
        by0, by1 = sorted((int(np.floor(p[1] * inv)), int(np.floor(q[1] * inv))))
        seen: Set[int] = set()
        for bx in range(bx0, bx1 + 1):
            for by in range(by0, by1 + 1):
                for i in self._bins.get((bx, by), ()):
                    if i not in seen:
                        seen.add(i)
                        yield i

    def blocks(
        self,
        p: Sequence[float],
        q: Sequence[float],
    ) -> bool:
        """True when the 3D segment ``p -> q`` crosses a wall."""
        if not self._count:
            return False
        up = axis_index(self.up_axis)
        h0, h1 = horizontal_axes(self.up_axis)
        a = (float(p[h0]), float(p[h1]))
        b = (float(q[h0]), float(q[h1]))
        lo = min(p[up], q[up])
        hi = max(p[up], q[up])
        for i in self._candidates(a, b):
            wall = self.segments[i]
            if hi < wall["up_min"] or lo > wall["up_max"]:
                continue
            if segments_intersect_2d(a, b, wall["segment"][0], wall["segment"][1]):
                return True
        return False


def segments_intersect_2d(p1, p2, p3, p4) -> bool:
    """Proper-intersection test for two 2D segments (touching does not count)."""

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    # Strict opposite signs on both sides. A zero means an endpoint lies on the
    # other segment, which is a touch rather than a crossing - treating that as
    # blocked would wall off any path node that merely grazes a wall line.
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
        and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )


def wall_segments_from_geometry(
    wall_geometries: Iterable, up_axis: str = "z"
) -> List[dict]:
    """Reduce each wall solid to a 2D centreline plus its vertical extent."""
    up = axis_index(up_axis)
    h0, h1 = horizontal_axes(up_axis)
    out: List[dict] = []
    for geom in wall_geometries or ():
        verts = _vertex_array(geom)
        if verts is None or len(verts) < 3:
            continue
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        span0 = float(hi[h0] - lo[h0])
        span1 = float(hi[h1] - lo[h1])
        mid0 = float((lo[h0] + hi[h0]) * 0.5)
        mid1 = float((lo[h1] + hi[h1]) * 0.5)
        if span0 >= span1:
            segment = ((float(lo[h0]), mid1), (float(hi[h0]), mid1))
            thickness = span1
        else:
            segment = ((mid0, float(lo[h1])), (mid0, float(hi[h1])))
            thickness = span0
        out.append(
            {
                "segment": segment,
                "thickness": thickness,
                "up_min": float(lo[up]),
                "up_max": float(hi[up]),
            }
        )
    return out


def door_positions_from_geometry(
    door_geometries: Iterable, up_axis: str = "z"
) -> np.ndarray:
    """One waypoint per door, at the floor-level centre of the opening.

    Returned positions are raw: the caller applies agent height uniformly so
    doors, floors and stairs all share the same vertical offset.
    """
    up = axis_index(up_axis)
    out: List[List[float]] = []
    for geom in door_geometries or ():
        verts = _vertex_array(geom)
        if verts is None or len(verts) < 3:
            continue
        centre = verts.mean(axis=0)
        coord = [float(centre[0]), float(centre[1]), float(centre[2])]
        coord[up] = float(verts[:, up].min())
        out.append(coord)
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 3), dtype=np.float32)


def blocked_grid_cells(
    wall_geometries: Iterable,
    cell_sizes: Sequence[float],
    up_axis: str = "z",
) -> Set[Tuple[int, int, int]]:
    """Rasterise wall bounding boxes onto the snap grid."""
    inv = [1.0 / c if c > 0 else 1.0 for c in cell_sizes]
    blocked: Set[Tuple[int, int, int]] = set()
    for geom in wall_geometries or ():
        verts = _vertex_array(geom)
        if verts is None or len(verts) < 3:
            continue
        lo = np.round(verts.min(axis=0) * inv).astype(np.int64)
        hi = np.round(verts.max(axis=0) * inv).astype(np.int64)
        # Guard against a degenerate wall exploding the cell count.
        if np.prod(hi - lo + 1) > 200_000:
            continue
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    blocked.add((ix, iy, iz))
    return blocked


def edge_block_mask(
    points: np.ndarray,
    edges: np.ndarray,
    walls: Optional[WallField],
    exempt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean mask, one entry per edge, True where a wall blocks it.

    Computed once at graph-build time. Edges touching an exempt node (doors)
    are never blocked, since a door is precisely a hole in a wall.
    """
    count = len(edges)
    mask = np.zeros(count, dtype=bool)
    if not walls or not len(walls) or count == 0:
        return mask
    exempt_set = set(exempt.tolist()) if exempt is not None and len(exempt) else set()
    for i in range(count):
        a, b = int(edges[i, 0]), int(edges[i, 1])
        if a in exempt_set or b in exempt_set:
            continue
        if walls.blocks(points[a], points[b]):
            mask[i] = True
    return mask


def _vertex_array(geom) -> Optional[np.ndarray]:
    """Accept either a pydantic geometry model or a plain dict/sequence."""
    verts = getattr(geom, "vertices", None)
    if verts is None and isinstance(geom, dict):
        verts = geom.get("vertices")
    if verts is None:
        return None
    arr = np.asarray(verts, dtype=np.float64)
    if arr.size < 9:
        return None
    return arr[: (arr.size // 3) * 3].reshape(-1, 3)
