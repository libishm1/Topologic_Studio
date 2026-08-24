"""Walkable-surface sampling.

The classic implementation looped over every triangle in Python and built a
list of ``[x, y, z]`` lists.  On a mid-size model that is millions of
interpreter iterations.  This version does the same maths with numpy, in a
handful of vectorised passes, and returns a single ``(N, 3)`` float32 array.

The sampling rule is unchanged so results stay comparable with Classic:

* reject triangles whose normal tilts more than ``max_slope_deg`` from up,
* reject triangles whose vertices all sit above ``exclude_above`` (roofs),
* lay a barycentric lattice over each surviving triangle with a resolution
  driven by ``spacing``.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .common import axis_index, horizontal_axes

_EMPTY = np.zeros((0, 3), dtype=np.float32)


def sample_walkable_points(
    vertices: Sequence[float],
    indices: Sequence[int],
    normals: Optional[Sequence[float]] = None,
    spacing: float = 0.5,
    max_slope_deg: float = 10.0,
    up_axis: str = "z",
    max_points: int = 20000,
    exclude_above: Optional[float] = None,
    require_upward: bool = True,
) -> np.ndarray:
    """Return an ``(N, 3)`` float32 array of walkable sample points.

    ``require_upward`` keeps only surfaces whose normal points up. You stand on
    the top of a slab, not its underside, and not on a ceiling; accepting either
    orientation produces a second point sheet beneath every storey which the
    neighbour search then braces into a space-frame truss. Set it False only
    when a model's face winding cannot be trusted.
    """
    if vertices is None or indices is None:
        return _EMPTY
    verts = np.asarray(vertices, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    if verts.size < 9 or idx.size < 3:
        return _EMPTY

    verts = verts[: (verts.size // 3) * 3].reshape(-1, 3)
    tri_count = idx.size // 3
    if tri_count == 0:
        return _EMPTY
    tris = idx[: tri_count * 3].reshape(-1, 3)

    # Drop triangles that reference vertices outside the buffer instead of
    # silently indexing garbage.
    valid = (tris >= 0).all(axis=1) & (tris < len(verts)).all(axis=1)
    tris = tris[valid]
    if len(tris) == 0:
        return _EMPTY

    v1 = verts[tris[:, 0]]
    v2 = verts[tris[:, 1]]
    v3 = verts[tris[:, 2]]

    up = axis_index(up_axis)

    if exclude_above is not None:
        keep = ~(
            (v1[:, up] >= exclude_above)
            & (v2[:, up] >= exclude_above)
            & (v3[:, up] >= exclude_above)
        )
        v1, v2, v3, tris = v1[keep], v2[keep], v3[keep], tris[keep]
        if len(tris) == 0:
            return _EMPTY

    edge_a = v2 - v1
    edge_b = v3 - v1
    cross = np.cross(edge_a, edge_b)
    twice_area = np.linalg.norm(cross, axis=1)

    non_degenerate = twice_area > 2e-6
    v1, v2, v3 = v1[non_degenerate], v2[non_degenerate], v3[non_degenerate]
    cross = cross[non_degenerate]
    twice_area = twice_area[non_degenerate]
    if len(v1) == 0:
        return _EMPTY

    # Prefer the supplied vertex normals (averaged over the triangle) and fall
    # back to the geometric face normal.
    face_normal_up = cross[:, up] / twice_area
    if normals is not None and len(normals) >= len(verts) * 3:
        norm = np.asarray(normals, dtype=np.float64)
        norm = norm[: (norm.size // 3) * 3].reshape(-1, 3)
        if norm.shape[0] >= verts.shape[0]:
            if np.issubdtype(np.asarray(normals).dtype, np.integer):
                norm = norm / 32767.0
            tris_kept = tris[non_degenerate] if len(tris) == len(non_degenerate) else tris
            try:
                avg = (
                    norm[tris_kept[:, 0]] + norm[tris_kept[:, 1]] + norm[tris_kept[:, 2]]
                ) / 3.0
                lengths = np.linalg.norm(avg, axis=1)
                usable = lengths > 1e-9
                supplied_up = np.where(usable, avg[:, up] / np.where(usable, lengths, 1.0), 0.0)
                if supplied_up.shape == face_normal_up.shape:
                    face_normal_up = supplied_up
            except (IndexError, ValueError):
                pass

    min_up = math.cos(math.radians(max(0.0, min(89.0, max_slope_deg))))
    walkable = (
        face_normal_up >= min_up if require_upward else np.abs(face_normal_up) >= min_up
    )
    v1, v2, v3 = v1[walkable], v2[walkable], v3[walkable]
    twice_area = twice_area[walkable]
    if len(v1) == 0:
        return _EMPTY

    areas = 0.5 * twice_area
    spacing = max(float(spacing), 1e-3)
    # Lattice resolution per triangle: roughly one sample every `spacing`
    # metres along the triangle's characteristic length.
    subdiv = np.ceil(np.sqrt(areas) / spacing).astype(np.int64)
    np.clip(subdiv, 1, 64, out=subdiv)

    # Big triangles dominate the budget; sample them first so a truncated run
    # still covers the floor plate rather than one corner of it.
    order = np.argsort(-areas)
    v1, v2, v3, subdiv = v1[order], v2[order], v3[order], subdiv[order]

    chunks = []
    produced = 0
    # Group triangles by lattice resolution so each group is one vectorised op.
    for n in np.unique(subdiv):
        if produced >= max_points:
            break
        mask = subdiv == n
        group1, group2, group3 = v1[mask], v2[mask], v3[mask]
        lattice = _barycentric_lattice(int(n))
        pts = (
            group1[:, None, :] * lattice[None, :, 0, None]
            + group2[:, None, :] * lattice[None, :, 1, None]
            + group3[:, None, :] * lattice[None, :, 2, None]
        ).reshape(-1, 3)
        remaining = max_points - produced
        if len(pts) > remaining:
            pts = pts[:remaining]
        chunks.append(pts)
        produced += len(pts)

    if not chunks:
        return _EMPTY
    return np.concatenate(chunks).astype(np.float32, copy=False)


_LATTICE_CACHE: dict[int, np.ndarray] = {}


def _barycentric_lattice(n: int) -> np.ndarray:
    """Barycentric weights for an ``n``-subdivision triangle lattice."""
    cached = _LATTICE_CACHE.get(n)
    if cached is not None:
        return cached
    rows = []
    for u in range(n + 1):
        for v in range(n + 1 - u):
            alpha = u / n
            beta = v / n
            rows.append((alpha, beta, 1.0 - alpha - beta))
    lattice = np.asarray(rows, dtype=np.float64)
    _LATTICE_CACHE[n] = lattice
    return lattice


def decimate(points: np.ndarray, cell: float) -> np.ndarray:
    """Keep at most one point per ``cell``-sized voxel.

    Sampling triangle-by-triangle produces heavy duplication where triangles
    meet.  Voxel decimation removes it in one pass and keeps the graph small,
    which is what actually makes pathfinding fast.
    """
    if points is None or len(points) == 0 or cell <= 0:
        return points if points is not None else _EMPTY
    keys = np.round(points / cell).astype(np.int64)
    # np.unique on a structured view is the fastest reliable way to dedupe rows.
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return points[first]


def collapse_columns(
    points: np.ndarray,
    cell: float,
    up_axis: str = "z",
    gap: float = 0.9,
) -> np.ndarray:
    """Reduce each vertical column of points to its walking surface.

    A slab is a solid, so sampling accepts both its top face and its underside:
    a downward-facing normal is just as horizontal as an upward one, and IFC
    winding is not dependable enough to separate them. The result is two
    parallel sheets a slab-thickness apart which the neighbour search then
    braces into a truss. Ceilings (IFCCOVERING) add more phantom sheets.

    Points are bucketed by horizontal cell, sorted by height, and split wherever
    the vertical gap exceeds ``gap``. Each run keeps only its highest point, so a
    slab's two faces collapse onto the top one while separate storeys survive.
    """
    if points is None or len(points) == 0:
        return points if points is not None else _EMPTY

    up = axis_index(up_axis)
    h0, h1 = horizontal_axes(up_axis)
    inv = 1.0 / max(float(cell), 1e-3)

    keys = np.stack(
        [
            np.round(points[:, h0] * inv).astype(np.int64),
            np.round(points[:, h1] * inv).astype(np.int64),
        ],
        axis=1,
    )
    # Sort by (column, height) so each column is a contiguous ascending run.
    order = np.lexsort((points[:, up], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    sorted_up = points[order, up]

    new_column = np.empty(len(order), dtype=bool)
    new_column[0] = True
    new_column[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)

    jump = np.empty(len(order), dtype=bool)
    jump[0] = True
    jump[1:] = (sorted_up[1:] - sorted_up[:-1]) > gap

    starts_run = new_column | jump
    # The last row of every run is its highest point, i.e. the row just before
    # the next run starts.
    ends_run = np.empty(len(order), dtype=bool)
    ends_run[:-1] = starts_run[1:]
    ends_run[-1] = True

    return points[order[ends_run]]
