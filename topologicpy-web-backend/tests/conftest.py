from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geometry.adjacency import KIND_DOOR, KIND_FLOOR, KIND_STAIR  # noqa: E402
from app.store import NavGraph  # noqa: E402


@pytest.fixture
def grid_graph():
    """A 12x12 single-storey grid with unit spacing, z up."""
    n = 12
    points = []
    index = {}
    for i in range(n):
        for j in range(n):
            index[(i, j)] = len(points)
            points.append([float(i), float(j), 0.0])
    edges = []
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                other = (i + di, j + dj)
                if other in index:
                    edges.append([index[(i, j)], index[other]])
    return NavGraph(
        points=np.asarray(points, dtype=np.float32),
        edges=np.asarray(edges, dtype=np.int32),
        kinds=np.zeros(len(points), dtype=np.int8),
        up_axis="z",
        meta={"index": index, "n": n},
    )


@pytest.fixture
def two_storey_geometry():
    """Two 6x6 m slabs plus a stair between them, as triangle soup.

    Mirrors the shape of what the frontend extracts from an IFC model, so the
    legacy sampling path is exercised on something structurally realistic.
    """

    def slab(z, x0=0.0, y0=0.0, w=6.0, d=6.0):
        verts = [
            x0, y0, z,
            x0 + w, y0, z,
            x0 + w, y0 + d, z,
            x0, y0 + d, z,
        ]
        return {"expressID": int(z * 100), "vertices": verts, "indices": [0, 1, 2, 0, 2, 3]}

    # A stair as a run of small horizontal treads climbing in z.
    stair_verts = []
    stair_indices = []
    treads = 16
    for t in range(treads):
        y = 6.0 + t * 0.25
        z = t * 0.1875
        base = len(stair_verts) // 3
        stair_verts += [
            2.0, y, z,
            3.2, y, z,
            3.2, y + 0.25, z,
            2.0, y + 0.25, z,
        ]
        stair_indices += [base, base + 1, base + 2, base, base + 2, base + 3]

    door = {
        "expressID": 900,
        "vertices": [
            2.6, 0.0, 0.0,
            3.6, 0.0, 0.0,
            3.6, 0.0, 2.1,
            2.6, 0.0, 2.1,
        ],
        "indices": [0, 1, 2, 0, 2, 3],
    }

    wall = {
        "expressID": 800,
        "vertices": [
            0.0, 3.0, 0.0,
            6.0, 3.0, 0.0,
            6.0, 3.1, 0.0,
            0.0, 3.1, 0.0,
            0.0, 3.0, 2.6,
            6.0, 3.0, 2.6,
            6.0, 3.1, 2.6,
            0.0, 3.1, 2.6,
        ],
        "indices": [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
    }

    return {
        "floors": [slab(0.0), slab(3.0)],
        "stairs": [{"expressID": 500, "vertices": stair_verts, "indices": stair_indices}],
        "doors": [door],
        "walls": [wall],
    }


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
