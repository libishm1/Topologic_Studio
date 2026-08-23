"""Navigation-graph construction from a sampled point cloud.

Classic built adjacency with a hand-rolled spatial hash and a 27-cell Python
loop per point. That is O(N * bucket) interpreted work and was the slowest
step of ``/ifc-egress-graph``. Here the same neighbourhood query is a single
``cKDTree.query_pairs`` call, and every filter afterwards is a numpy mask over
the resulting pair array.

Node kinds are carried in a parallel int array rather than the classic
"first K entries are stairs" convention, which was easy to get wrong once
door waypoints were appended to the same list.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .common import axis_index, horizontal_axes

KIND_FLOOR = 0
KIND_STAIR = 1
KIND_DOOR = 2

KIND_NAMES = {KIND_FLOOR: "floor", KIND_STAIR: "stair", KIND_DOOR: "door"}

_NO_EDGES = np.zeros((0, 2), dtype=np.int32)


#: Neighbour cap per node. A pure radius query couples node density to edge
#: count quadratically: 0.5 m sampling inside a 2.25 m radius gives ~56 edges
#: per node, which is what made Classic's graphs enormous without making the
#: routes any better. Capping keeps the fine sampling (so doorways survive)
#: while holding the graph sparse.
DEFAULT_MAX_DEGREE = 12


def build_adjacency(
    points: np.ndarray,
    kinds: np.ndarray,
    max_edge_floor: float,
    max_edge_stair: float,
    up_axis: str = "z",
    rectilinear: bool = False,
    stair_link_radius: float = 2.0,
    door_link_radius: float = 3.0,
    max_degree: int = DEFAULT_MAX_DEGREE,
) -> np.ndarray:
    """Return an ``(M, 2)`` int32 array of undirected edges with ``i < j``."""
    if points is None or len(points) < 2:
        return _NO_EDGES.copy()

    max_r = max(float(max_edge_floor), float(max_edge_stair))
    if max_r <= 0:
        return _NO_EDGES.copy()

    tree = cKDTree(points)
    pairs = _candidate_pairs(tree, points, max_r, max_degree)

    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        delta = points[b] - points[a]
        dist = np.linalg.norm(delta, axis=1)

        ka, kb = kinds[a], kinds[b]
        stair_stair = (ka == KIND_STAIR) & (kb == KIND_STAIR)
        floor_floor = (ka != KIND_STAIR) & (kb != KIND_STAIR)

        keep = (stair_stair & (dist <= max_edge_stair)) | (
            floor_floor & (dist <= max_edge_floor)
        )

        if rectilinear:
            h0, h1 = horizontal_axes(up_axis)
            d0 = np.abs(delta[:, h0])
            d1 = np.abs(delta[:, h1])
            major = np.maximum(d0, d1)
            minor = np.minimum(d0, d1)
            diagonal = (major > 0.01) & (minor > 0.25 * major)
            keep &= ~(floor_floor & diagonal)

        edges = pairs[keep].astype(np.int32, copy=False)
    else:
        edges = _NO_EDGES.copy()

    # Stair/floor links are made deliberately at landings rather than by raw
    # proximity, so a flight never short-circuits sideways into a slab.
    extra = [
        part
        for part in (
            _link_stair_landings(points, kinds, up_axis, stair_link_radius),
            _link_doors(points, kinds, up_axis, door_link_radius),
        )
        if len(part)
    ]
    if extra:
        edges = np.concatenate([edges] + extra)

    return dedupe_edges(edges)


def _candidate_pairs(
    tree: cKDTree, points: np.ndarray, radius: float, max_degree: int
) -> np.ndarray:
    """Neighbour pairs within ``radius``, capped at ``max_degree`` per node.

    Falls back to an uncapped radius query when no cap is requested. The cap is
    one-sided (a may keep b without b keeping a); the union after deduping is
    what forms the graph, which slightly exceeds the cap but never drops a link
    that either endpoint considered close.
    """
    if max_degree is None or max_degree <= 0:
        return tree.query_pairs(radius, output_type="ndarray")

    k = min(max_degree + 1, len(points))  # +1 because the nearest hit is self
    distances, indices = tree.query(points, k=k, distance_upper_bound=radius)
    if distances.ndim == 1:  # k == 1, only self
        return np.zeros((0, 2), dtype=np.int64)

    rows = np.repeat(np.arange(len(points)), k)
    cols = indices.reshape(-1)
    dist = distances.reshape(-1)

    # scipy marks "no neighbour" with an out-of-range index and inf distance.
    valid = np.isfinite(dist) & (cols < len(points)) & (rows != cols)
    return np.stack([rows[valid], cols[valid]], axis=1)


def dedupe_edges(edges: np.ndarray) -> np.ndarray:
    """Normalise to ``i < j`` ordering, drop self-loops and duplicates."""
    if edges is None or len(edges) == 0:
        return _NO_EDGES.copy()
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    stacked = np.stack([lo, hi], axis=1)
    stacked = stacked[lo != hi]
    if len(stacked) == 0:
        return _NO_EDGES.copy()
    return np.unique(stacked, axis=0).astype(np.int32, copy=False)


def _link_stair_landings(
    points: np.ndarray, kinds: np.ndarray, up_axis: str, radius: float
) -> np.ndarray:
    """Connect the top and bottom of each stair run to nearby floor points.

    Without this the stair point cloud is an island: proximity alone will not
    bridge to a slab because mixed-kind pairs are rejected above.
    """
    stair_idx = np.flatnonzero(kinds == KIND_STAIR)
    floor_idx = np.flatnonzero(kinds != KIND_STAIR)
    if len(stair_idx) == 0 or len(floor_idx) == 0:
        return _NO_EDGES.copy()

    up = axis_index(up_axis)
    order = np.argsort(points[stair_idx, up])
    # The lowest and highest ~10% of a flight are treated as its landings.
    count = max(2, len(stair_idx) // 10)
    endpoints = stair_idx[np.unique(np.concatenate([order[:count], order[-count:]]))]

    floor_tree = cKDTree(points[floor_idx])
    out: List[Tuple[int, int]] = []
    for si in endpoints:
        neighbours = floor_tree.query_ball_point(points[si], radius)
        if not neighbours:
            continue
        cand = floor_idx[np.asarray(neighbours, dtype=np.int64)]
        # Same storey only: a landing must not tie to the slab above it.
        cand = cand[np.abs(points[cand, up] - points[si, up]) <= 1.0]
        if len(cand) == 0:
            continue
        nearest = cand[np.argsort(np.linalg.norm(points[cand] - points[si], axis=1))[:3]]
        out.extend((int(si), int(fi)) for fi in nearest)

    return np.asarray(out, dtype=np.int32) if out else _NO_EDGES.copy()


def _link_doors(
    points: np.ndarray, kinds: np.ndarray, up_axis: str, radius: float
) -> np.ndarray:
    """Wire each door waypoint into the walkable points around it."""
    door_idx = np.flatnonzero(kinds == KIND_DOOR)
    walk_idx = np.flatnonzero(kinds != KIND_DOOR)
    if len(door_idx) == 0 or len(walk_idx) == 0:
        return _NO_EDGES.copy()

    up = axis_index(up_axis)
    walk_tree = cKDTree(points[walk_idx])
    out: List[Tuple[int, int]] = []
    for di in door_idx:
        neighbours = walk_tree.query_ball_point(points[di], radius)
        if not neighbours:
            continue
        cand = walk_idx[np.asarray(neighbours, dtype=np.int64)]
        cand = cand[np.abs(points[cand, up] - points[di, up]) <= 1.5]
        if len(cand) == 0:
            continue
        nearest = cand[np.argsort(np.linalg.norm(points[cand] - points[di], axis=1))[:6]]
        out.extend((int(di), int(wi)) for wi in nearest)

    return np.asarray(out, dtype=np.int32) if out else _NO_EDGES.copy()


def snap_to_grid(
    points: np.ndarray,
    kinds: np.ndarray,
    cell_size: float,
    up_axis: str = "z",
    vertical_cell_size: Optional[float] = None,
    max_gap: int = 5,
    blocked_cells: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectilinear mode: snap to a grid, bridge gaps, connect 6-neighbours.

    Returns ``(points, kinds, edges)`` for the snapped graph. Node indices are
    re-issued, so callers must not carry old indices across this call.
    """
    if points is None or len(points) == 0 or cell_size <= 0:
        return points, kinds, _NO_EDGES.copy()

    up = axis_index(up_axis)
    sizes = np.full(3, float(cell_size))
    sizes[up] = float(vertical_cell_size or cell_size)
    inv = 1.0 / sizes

    cells = np.round(points * inv).astype(np.int64)
    unique_cells, first_index = np.unique(cells, axis=0, return_index=True)

    kind_of: Dict[tuple, int] = {
        tuple(cell): int(kinds[i]) for cell, i in zip(unique_cells, first_index)
    }
    occupied = set(kind_of)
    if blocked_cells:
        # A door cell stays walkable even where a wall rasterised over it.
        doors = {c for c, k in kind_of.items() if k == KIND_DOOR}
        occupied -= blocked_cells - doors
        for cell in list(kind_of):
            if cell not in occupied:
                kind_of.pop(cell, None)

    filled = set(occupied)
    for axis in range(3):
        columns: Dict[tuple, List[int]] = {}
        for cell in occupied:
            key = tuple(cell[i] for i in range(3) if i != axis)
            columns.setdefault(key, []).append(cell[axis])
        for key, positions in columns.items():
            positions.sort()
            for lo, hi in zip(positions, positions[1:]):
                if 1 < hi - lo <= max_gap:
                    for g in range(lo + 1, hi):
                        candidate = list(key)
                        candidate.insert(axis, g)
                        candidate_t = tuple(candidate)
                        if blocked_cells and candidate_t in blocked_cells:
                            continue  # never bridge through a wall
                        filled.add(candidate_t)

    ordered = sorted(filled)
    cell_to_index = {cell: i for i, cell in enumerate(ordered)}
    out_points = np.asarray(
        [[c[0] * sizes[0], c[1] * sizes[1], c[2] * sizes[2]] for c in ordered],
        dtype=np.float32,
    )
    out_kinds = np.asarray([kind_of.get(c, KIND_FLOOR) for c in ordered], dtype=np.int8)

    edges: List[Tuple[int, int]] = []
    for cell, i in cell_to_index.items():
        for delta in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            neighbour = (cell[0] + delta[0], cell[1] + delta[1], cell[2] + delta[2])
            j = cell_to_index.get(neighbour)
            if j is not None:
                edges.append((i, j))

    edge_array = np.asarray(edges, dtype=np.int32) if edges else _NO_EDGES.copy()
    return out_points, out_kinds, edge_array
