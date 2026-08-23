"""Helpers the legacy topology contract depends on.

Lifted from Classic's ``main.py`` so ``legacy/contract.py`` stays a faithful
copy. The wire/cell graphs these produce are adapted onto the new
:class:`~app.store.NavGraph` by :func:`register_legacy_graph`, so the fire and
RL endpoints work identically for the JSON-contract workflow.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..store import NavGraph, store

def _vertex_coord_map(vertices):
    coords = {}
    for v in vertices or []:
        uid = v.get("uid") or v.get("uuid")
        coord = v.get("coordinates") or v.get("Coordinates")
        if uid and coord and len(coord) >= 3:
            coords[uid] = coord[:3]
    return coords

def _build_adjacency(edges, vertex_ids, allowed_kinds=None):
    adj = {uid: set() for uid in vertex_ids}
    for e in edges or []:
        if allowed_kinds is not None:
            kind = (e.get("dictionary") or {}).get("edgeKind")
            if kind not in allowed_kinds:
                continue
        verts = e.get("vertices") or []
        if len(verts) < 2:
            continue
        v0, v1 = verts[0], verts[1]
        if v0 in adj and v1 in adj:
            adj[v0].add(v1)
            adj[v1].add(v0)
    return {k: list(v) for k, v in adj.items()}

def _nearest_node_id(coords_map, point, allowed=None):
    if not coords_map or not point or len(point) < 3:
        return None
    px, py, pz = point[0], point[1], point[2]
    best = None
    best_d = 1e18
    allowed_set = set(allowed) if allowed is not None else None

    # First pass: prefer nodes on the same floor (within 1.5m Y-distance)
    # Y is the vertical axis - this ensures we pick the right floor in multi-story buildings
    for uid, coord in coords_map.items():
        if allowed_set is not None and uid not in allowed_set:
            continue
        x, y, z = coord
        y_diff = abs(y - py)
        if y_diff <= 1.5:  # Same floor level (generous tolerance for IFC precision)
            xz_d = (x - px) ** 2 + (z - pz) ** 2
            if xz_d < best_d:
                best = uid
                best_d = xz_d

    # If we found a node on the same floor, return it
    if best is not None:
        return best

    # Second pass: if no nodes on same floor, find nearest in 3D space as fallback
    for uid, coord in coords_map.items():
        if allowed_set is not None and uid not in allowed_set:
            continue
        x, y, z = coord
        d = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
        if d < best_d:
            best = uid
            best_d = d

    return best

def _resolve_start_id(graph, start_id=None, start_point=None):
    adj = graph.get("adjacency") or {}
    if start_id and start_id in adj:
        return start_id
    if start_point:
        coords = graph.get("coords", {})
        return _nearest_node_id(coords, start_point, allowed=adj.keys())
    return None


def _default_start_id(graph):
    adj = graph.get("adjacency") or {}
    if not adj:
        return None
    coords = graph.get("coords") or {}
    if coords:
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        zs = [c[2] for c in coords.values()]
        center = [
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            (min(zs) + max(zs)) / 2,
        ]
        return _nearest_node_id(coords, center, allowed=adj.keys())
    return next(iter(adj.keys()), None)

def _estimate_step_size(coords, adjacency=None):
    if coords and adjacency:
        min_dist = None
        for node, nbrs in adjacency.items():
            if node not in coords:
                continue
            x0, y0, z0 = coords[node]
            for nbr in nbrs:
                if nbr not in coords:
                    continue
                x1, y1, z1 = coords[nbr]
                d = ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
                if d <= 0:
                    continue
                if min_dist is None or d < min_dist:
                    min_dist = d
        if min_dist:
            return min_dist
    return 1.0



def register_legacy_graph(
    mode: str,
    coords: Dict[str, List[float]],
    adjacency: Dict[str, List[str]],
    bboxes: Optional[List[dict]] = None,
    up_axis: str = "z",
) -> Optional[str]:
    """Adapt a string-keyed legacy graph onto NavGraph and store it."""
    if not coords:
        return None
    ids = list(coords.keys())
    index = {uid: i for i, uid in enumerate(ids)}
    points = np.asarray([coords[uid][:3] for uid in ids], dtype=np.float32)

    pairs = set()
    for uid, neighbours in (adjacency or {}).items():
        a = index.get(uid)
        if a is None:
            continue
        for other in neighbours or ():
            b = index.get(other)
            if b is None or b == a:
                continue
            pairs.add((a, b) if a < b else (b, a))

    edges = (
        np.asarray(sorted(pairs), dtype=np.int32)
        if pairs
        else np.zeros((0, 2), dtype=np.int32)
    )
    graph = NavGraph(
        points=points,
        edges=edges,
        kinds=np.zeros(len(points), dtype=np.int8),
        up_axis=up_axis,
        mode=mode,
        meta={"legacy_ids": ids, "bboxes": bboxes or []},
    )
    return store.put(graph)


def _build_cell_grid(bounds, cell_size=1.0, max_cells=2000):
    minx = bounds.get("minx")
    maxx = bounds.get("maxx")
    miny = bounds.get("miny")
    maxy = bounds.get("maxy")
    minz = bounds.get("minz")
    maxz = bounds.get("maxz")
    if minx is None or maxx is None or not math.isfinite(minx) or not math.isfinite(maxx):
        return [], {}, None
    span_x = maxx - minx
    span_y = maxy - miny
    span_z = maxz - minz
    if span_x <= 0 or span_y <= 0 or span_z <= 0:
        return [], {}, None
    nx = max(1, int(span_x / cell_size))
    ny = max(1, int(span_y / cell_size))
    nz = max(1, int(span_z / cell_size))
    total = nx * ny * nz
    if total > max_cells:
        scale = (max_cells / total) ** (1.0 / 3.0)
        nx = max(1, int(nx * scale))
        ny = max(1, int(ny * scale))
        nz = max(1, int(nz * scale))
    dx = span_x / nx
    dy = span_y / ny
    dz = span_z / nz
    step = min(dx, dy, dz)

    nodes = []
    node_index = {}
    for k in range(nz):
        z = minz + (k + 0.5) * dz
        for j in range(ny):
            y = miny + (j + 0.5) * dy
            for i in range(nx):
                x = minx + (i + 0.5) * dx
                uid = f"cell_{i}_{j}_{k}"
                nodes.append({
                    "id": uid,
                    "center": [x, y, z],
                    "minx": x - dx * 0.5,
                    "maxx": x + dx * 0.5,
                    "miny": y - dy * 0.5,
                    "maxy": y + dy * 0.5,
                    "z_min": z - dz * 0.5,
                    "z_max": z + dz * 0.5,
                })
                node_index[(i, j, k)] = uid

    adjacency = {n["id"]: [] for n in nodes}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                uid = node_index[(i, j, k)]
                for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        adjacency[uid].append(node_index[(ni, nj, nk)])

    return nodes, adjacency, step
