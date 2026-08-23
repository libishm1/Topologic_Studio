"""Turns geometry (or a pre-sampled point cloud) into a stored NavGraph.

This is the seam the Next architecture is built around. The browser already
parsed the IFC into fragments to draw it; asking it to also sample walkable
points costs almost nothing extra there and removes a full triangle-soup
upload plus a Python resampling pass from the critical path.

Both entry points converge on :func:`build_from_points`, so the two request
shapes cannot drift apart in behaviour.
"""
from __future__ import annotations

import base64
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import settings
from ..geometry import adjacency as adj
from ..geometry import obstacles
from ..geometry.common import axis_index
from ..geometry.sampling import decimate, sample_walkable_points
from ..models import GraphOptions, GraphStats, IfcEgressRequest, PointCloudRequest
from ..perf import Timer
from ..store import NavGraph, store


class BuildError(ValueError):
    """Raised when the supplied geometry cannot produce a usable graph."""


def build_from_points(
    floor_points: np.ndarray,
    stair_points: np.ndarray,
    door_points: np.ndarray,
    wall_segments: Sequence[dict],
    options: GraphOptions,
    timer: Optional[Timer] = None,
    mode: str = "ifc",
) -> Tuple[NavGraph, str]:
    """Assemble, index and store a navigation graph."""
    parts: List[np.ndarray] = []
    kind_parts: List[np.ndarray] = []
    for cloud, kind in (
        (stair_points, adj.KIND_STAIR),
        (floor_points, adj.KIND_FLOOR),
        (door_points, adj.KIND_DOOR),
    ):
        if cloud is None or len(cloud) == 0:
            continue
        cloud = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
        parts.append(cloud)
        kind_parts.append(np.full(len(cloud), kind, dtype=np.int8))

    if not parts:
        raise BuildError("No walkable points were produced from the supplied geometry.")

    points = np.concatenate(parts)
    kinds = np.concatenate(kind_parts)

    # Agent height is applied along the model's own up axis. Classic added it
    # to index 2 unconditionally, which silently pushed points sideways on a
    # y-up model - the default the frontend actually used.
    up = axis_index(options.up_axis)
    if options.agent_height:
        points = points.copy()
        points[:, up] += float(options.agent_height)

    if len(points) > options.max_points:
        # Keep a spatially even subset rather than the first N, which would
        # otherwise favour whichever storey happened to be sampled first.
        keep = np.linspace(0, len(points) - 1, options.max_points).astype(np.int64)
        points, kinds = points[keep], kinds[keep]

    blocked_cells = None
    if options.grid_snap:
        cell = options.grid_cell_size or options.max_edge_floor
        sizes = [cell, cell, cell]
        sizes[up] = min(cell, 0.15)  # a stair tread needs its own vertical cell
        if options.use_walls and wall_segments:
            blocked_cells = _blocked_cells_from_segments(wall_segments, sizes, options.up_axis)
        with _phase(timer, "snap"):
            points, kinds, edges = adj.snap_to_grid(
                points,
                kinds,
                cell_size=cell,
                up_axis=options.up_axis,
                vertical_cell_size=sizes[up],
                max_gap=max(2, int(np.ceil(options.max_edge_floor / cell)) + 1),
                blocked_cells=blocked_cells,
            )
    else:
        if options.decimate > 0:
            with _phase(timer, "decimate"):
                keep_idx = _decimate_index(points, options.decimate)
                points, kinds = points[keep_idx], kinds[keep_idx]
        with _phase(timer, "adjacency"):
            edges = adj.build_adjacency(
                points,
                kinds,
                max_edge_floor=options.max_edge_floor,
                max_edge_stair=options.max_edge_stair,
                up_axis=options.up_axis,
                rectilinear=options.rectilinear,
                max_degree=options.max_degree,
            )

    if len(points) > settings.max_graph_nodes:
        raise BuildError(
            f"Graph would have {len(points)} nodes, above the {settings.max_graph_nodes} limit. "
            "Increase the sampling spacing or enable decimation."
        )

    walls = None
    blocked = None
    if options.use_walls and wall_segments and not options.grid_snap:
        with _phase(timer, "walls"):
            walls = obstacles.WallField(wall_segments, up_axis=options.up_axis)
            blocked = obstacles.edge_block_mask(
                points, edges, walls, exempt=np.flatnonzero(kinds == adj.KIND_DOOR)
            )

    graph = NavGraph(
        points=np.ascontiguousarray(points, dtype=np.float32),
        edges=np.ascontiguousarray(edges, dtype=np.int32),
        kinds=np.ascontiguousarray(kinds, dtype=np.int8),
        up_axis=options.up_axis,
        walls=walls,
        blocked=blocked,
        mode=mode,
    )
    graph_id = store.put(graph)
    return graph, graph_id


def build_from_point_cloud(
    req: PointCloudRequest, timer: Optional[Timer] = None
) -> Tuple[NavGraph, str]:
    """Preferred path: the browser already sampled the walkable surfaces."""
    segments = [
        {
            "segment": (tuple(w.segment[0][:2]), tuple(w.segment[1][:2])),
            "thickness": w.thickness,
            "up_min": w.up_min,
            "up_max": w.up_max,
        }
        for w in req.walls
        if len(w.segment) >= 2 and len(w.segment[0]) >= 2 and len(w.segment[1]) >= 2
    ]
    return build_from_points(
        floor_points=_as_points(req.floor_points),
        stair_points=_as_points(req.stair_points),
        door_points=_as_points(req.door_points),
        wall_segments=segments,
        options=req.options,
        timer=timer,
    )


def build_from_ifc_geometry(
    req: IfcEgressRequest, timer: Optional[Timer] = None
) -> Tuple[NavGraph, str]:
    """Legacy path: sample triangle soup server-side, as Classic did."""
    floors = req.floors or []
    stairs = req.stairs or []
    if not floors and not stairs:
        raise BuildError("No IFC floor or stair geometry provided.")

    up_axis = req.up_axis if req.up_axis in ("x", "y", "z") else "z"
    up = axis_index(up_axis)

    base_spacing = max(req.base_spacing, 0.25)
    stair_spacing = max(base_spacing * 0.3, 0.15)

    # Roof slabs are geometrically walkable but are not floors. Anything within
    # a metre of the model's ceiling is excluded, matching Classic.
    heights = [
        float(np.asarray(g.vertices, dtype=np.float64)[up::3].max())
        for g in (floors + stairs)
        if g.vertices and len(g.vertices) >= 3
    ]
    exclude_above = (max(heights) - 1.0) if heights else None

    with _phase(timer, "sample"):
        stair_points = _sample_group(
            stairs, stair_spacing, 45.0, up_axis, req.max_points, exclude_above
        )
        floor_points = _sample_group(
            floors, base_spacing, 10.0, up_axis, req.max_points, exclude_above
        )

    with _phase(timer, "obstacles"):
        door_points = obstacles.door_positions_from_geometry(req.doors, up_axis)
        wall_segments = (
            obstacles.wall_segments_from_geometry(req.walls, up_axis)
            if req.use_walls and req.walls
            else []
        )

    options = GraphOptions(
        up_axis=up_axis,
        agent_height=req.agent_height,
        max_edge_floor=req.max_edge_floor if req.max_edge_floor is not None else base_spacing * 1.5,
        max_edge_stair=req.max_edge_stair if req.max_edge_stair is not None else 0.4,
        use_walls=req.use_walls,
        rectilinear=req.rectilinear,
        grid_snap=req.grid_snap,
        grid_cell_size=req.grid_cell_size,
        # Triangle sampling duplicates points wherever triangles meet. Voxel
        # decimation at a third of the edge budget removes that without
        # thinning the walkable surface.
        decimate=max(0.05, base_spacing / 3.0),
        max_points=min(req.max_points, settings.max_graph_nodes),
    )

    return build_from_points(
        floor_points=floor_points,
        stair_points=stair_points,
        door_points=door_points,
        wall_segments=wall_segments,
        options=options,
        timer=timer,
    )


def graph_stats(graph: NavGraph) -> GraphStats:
    from ..graphs.pathfinding import largest_component

    component = largest_component(graph)
    return GraphStats(
        nodes=graph.node_count,
        edges=graph.edge_count,
        floor_nodes=int((graph.kinds == adj.KIND_FLOOR).sum()),
        stair_nodes=int((graph.kinds == adj.KIND_STAIR).sum()),
        door_nodes=int((graph.kinds == adj.KIND_DOOR).sum()),
        wall_segments=len(graph.walls) if graph.walls else 0,
        blocked_edges=int(graph.blocked.sum()) if graph.blocked is not None else 0,
        components=_component_count(graph),
        largest_component=int(component.sum()),
    )


def encode_graph(graph: NavGraph) -> Dict[str, str]:
    """Binary-encode node and edge arrays.

    A 20k-node graph is ~2 MB as JSON numbers and ~320 KB as base64 float32,
    and the browser decodes it straight into a typed array with no per-number
    parse. This is the difference between a snappy graph overlay and a
    half-second stall.
    """
    return {
        "nodes_b64": _b64(graph.points.astype(np.float32, copy=False)),
        "edges_b64": _b64(graph.edges.astype(np.uint32, copy=False)),
        "kinds_b64": _b64(graph.kinds.astype(np.uint8, copy=False)),
    }


def legacy_edge_payload(graph: NavGraph, limit: int = 60000) -> Tuple[list, list]:
    """Classic ``edges`` / ``edge_ids`` arrays, for the old frontend."""
    count = min(graph.edge_count, limit)
    edge_list = []
    edge_ids = []
    for i in range(count):
        a, b = int(graph.edges[i, 0]), int(graph.edges[i, 1])
        edge_list.append([graph.points[a].tolist(), graph.points[b].tolist()])
        edge_ids.append([f"ifc_{a}", f"ifc_{b}"])
    return edge_list, edge_ids


# --------------------------------------------------------------------------
# internals


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _as_points(values) -> np.ndarray:
    if not values:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    arr = arr[:, :3]
    return arr[np.isfinite(arr).all(axis=1)]


def _sample_group(
    geometries,
    spacing: float,
    max_slope: float,
    up_axis: str,
    budget: int,
    exclude_above: Optional[float],
) -> np.ndarray:
    if not geometries:
        return np.zeros((0, 3), dtype=np.float32)
    per_item = max(64, budget // max(1, len(geometries)))
    chunks = []
    total = 0
    for geom in geometries:
        if total >= budget:
            break
        pts = sample_walkable_points(
            geom.vertices,
            geom.indices,
            geom.normals,
            spacing=spacing,
            max_slope_deg=max_slope,
            up_axis=up_axis,
            max_points=min(per_item, budget - total),
            exclude_above=exclude_above,
        )
        if len(pts):
            chunks.append(pts)
            total += len(pts)
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks)


def _decimate_index(points: np.ndarray, cell: float) -> np.ndarray:
    keys = np.round(points / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return first


def _blocked_cells_from_segments(segments, sizes, up_axis):
    """Rasterise wall centrelines onto the snap grid."""
    up = axis_index(up_axis)
    horizontal = [i for i in range(3) if i != up]
    inv = [1.0 / s if s > 0 else 1.0 for s in sizes]
    blocked = set()
    for wall in segments:
        (x0, y0), (x1, y1) = wall["segment"]
        steps = int(max(abs(x1 - x0) * inv[horizontal[0]], abs(y1 - y0) * inv[horizontal[1]])) + 1
        steps = min(steps, 4000)
        for t in np.linspace(0.0, 1.0, steps + 1):
            hx = x0 + (x1 - x0) * t
            hy = y0 + (y1 - y0) * t
            lo = int(round(wall["up_min"] * inv[up]))
            hi = int(round(wall["up_max"] * inv[up]))
            if hi - lo > 4000:
                hi = lo + 4000
            for iz in range(lo, hi + 1):
                cell = [0, 0, 0]
                cell[horizontal[0]] = int(round(hx * inv[horizontal[0]]))
                cell[horizontal[1]] = int(round(hy * inv[horizontal[1]]))
                cell[up] = iz
                blocked.add(tuple(cell))
    return blocked


def _component_count(graph: NavGraph) -> int:
    n = graph.node_count
    if n == 0:
        return 0
    indptr, neighbours, _ = graph.csr()
    seen = np.zeros(n, dtype=bool)
    count = 0
    for seed in range(n):
        if seen[seed]:
            continue
        count += 1
        seen[seed] = True
        frontier = [seed]
        while frontier:
            nxt = []
            for node in frontier:
                for k in range(indptr[node], indptr[node + 1]):
                    nbr = int(neighbours[k])
                    if not seen[nbr]:
                        seen[nbr] = True
                        nxt.append(nbr)
            frontier = nxt
    return count


class _NullPhase:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _phase(timer: Optional[Timer], name: str):
    return timer.phase(name) if timer is not None else _NullPhase()
