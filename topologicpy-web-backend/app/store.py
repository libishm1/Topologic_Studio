"""Navigation graph model and the server-side graph store.

Classic kept a single ``LAST_GRAPHS = {"wire": None, "cell": None, "ifc": None}``
dict at module scope. Two browser tabs, or two users, would silently overwrite
each other's graph, and every endpoint depended on that implicit mutable state.

Here a build returns an opaque ``graph_id`` that the client passes back. The
store is an LRU with a TTL so a long-running server cannot grow without bound.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import settings
from .geometry.adjacency import KIND_DOOR
from .geometry.common import axis_index
from .geometry.obstacles import WallField


@dataclass
class NavGraph:
    """An immutable-by-convention navigation graph.

    Coordinates live in one ``(N, 3)`` float32 array and edges in one
    ``(M, 2)`` int32 array. Node identity is the row index; the string form
    ``ifc_<index>`` exists only at the API boundary for backward compatibility.
    """

    points: np.ndarray
    edges: np.ndarray
    kinds: np.ndarray
    up_axis: str = "z"
    walls: Optional[WallField] = None
    blocked: Optional[np.ndarray] = None
    mode: str = "ifc"
    meta: Dict = field(default_factory=dict)

    # Lazily built structures.
    _csr: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default=None, repr=False, compare=False
    )
    _tree: Optional[object] = field(default=None, repr=False, compare=False)
    _topologic: Optional[object] = field(default=None, repr=False, compare=False)

    @property
    def node_count(self) -> int:
        return int(len(self.points))

    @property
    def edge_count(self) -> int:
        return int(len(self.edges))

    @property
    def door_nodes(self) -> np.ndarray:
        return np.flatnonzero(self.kinds == KIND_DOOR)

    @property
    def edge_lengths(self) -> np.ndarray:
        if self.edge_count == 0:
            return np.zeros(0, dtype=np.float32)
        delta = self.points[self.edges[:, 1]] - self.points[self.edges[:, 0]]
        return np.linalg.norm(delta, axis=1).astype(np.float32, copy=False)

    def csr(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adjacency in CSR form: ``(indptr, neighbours, edge_ids)``.

        Both directions of every undirected edge are present. ``edge_ids``
        maps each directed entry back to its row in ``self.edges`` so a path
        query can consult per-edge data (blocked flags, hazard costs) without
        another lookup.
        """
        if self._csr is not None:
            return self._csr
        n = self.node_count
        if self.edge_count == 0:
            empty = (
                np.zeros(n + 1, dtype=np.int64),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
            )
            object.__setattr__(self, "_csr", empty)
            return empty

        src = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        dst = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        eid = np.tile(np.arange(self.edge_count, dtype=np.int32), 2)

        order = np.argsort(src, kind="stable")
        src, dst, eid = src[order], dst[order], eid[order]

        counts = np.bincount(src, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])

        result = (indptr, dst.astype(np.int32, copy=False), eid)
        object.__setattr__(self, "_csr", result)
        return result

    def tree(self):
        """KD-tree over node positions, for nearest-node snapping."""
        if self._tree is None:
            from scipy.spatial import cKDTree

            object.__setattr__(self, "_tree", cKDTree(self.points))
        return self._tree

    def nearest(self, point, max_distance: Optional[float] = None) -> Optional[int]:
        """Index of the node closest to ``point``, or None."""
        if point is None or len(point) < 3 or self.node_count == 0:
            return None
        query = np.asarray(point[:3], dtype=np.float64)
        if not np.all(np.isfinite(query)):
            return None
        distance, index = self.tree().query(query, k=1)
        if max_distance is not None and distance > max_distance:
            return None
        return int(index)

    def step_size(self) -> float:
        """Representative edge length, used to pace radial fire spread."""
        lengths = self.edge_lengths
        if len(lengths) == 0:
            return 1.0
        value = float(np.median(lengths))
        return value if math.isfinite(value) and value > 1e-6 else 1.0

    def bounds(self) -> List[List[float]]:
        if self.node_count == 0:
            return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        return [self.points.min(axis=0).tolist(), self.points.max(axis=0).tolist()]

    def coord_map(self) -> Dict[str, List[float]]:
        """Legacy ``{"ifc_0": [x, y, z]}`` view for the classic API shape."""
        return {f"ifc_{i}": self.points[i].tolist() for i in range(self.node_count)}

    def adjacency_map(self) -> Dict[str, List[str]]:
        """Legacy ``{"ifc_0": ["ifc_1", ...]}`` view."""
        indptr, neighbours, _ = self.csr()
        out: Dict[str, List[str]] = {}
        for i in range(self.node_count):
            lo, hi = indptr[i], indptr[i + 1]
            out[f"ifc_{i}"] = [f"ifc_{int(j)}" for j in neighbours[lo:hi]]
        return out

    def up_index(self) -> int:
        return axis_index(self.up_axis)


class GraphStore:
    """Thread-safe LRU + TTL store of built graphs."""

    def __init__(self, capacity: Optional[int] = None, ttl: Optional[float] = None):
        self._capacity = capacity or settings.graph_cache_size
        self._ttl = ttl if ttl is not None else settings.graph_ttl_seconds
        self._items: "OrderedDict[str, Tuple[float, NavGraph]]" = OrderedDict()
        self._lock = threading.Lock()
        #: Most recent graph per mode, so clients that predate graph ids still work.
        self._latest: Dict[str, str] = {}

    def put(self, graph: NavGraph, graph_id: Optional[str] = None) -> str:
        gid = graph_id or uuid.uuid4().hex[:16]
        with self._lock:
            self._items[gid] = (time.time(), graph)
            self._items.move_to_end(gid)
            self._latest[graph.mode] = gid
            self._evict_locked()
        return gid

    def get(self, graph_id: Optional[str], mode: Optional[str] = None) -> Optional[NavGraph]:
        with self._lock:
            self._evict_locked()
            gid = graph_id
            if not gid and mode:
                gid = self._latest.get(mode)
            if not gid:
                return None
            entry = self._items.get(gid)
            if entry is None:
                return None
            self._items[gid] = (time.time(), entry[1])
            self._items.move_to_end(gid)
            return entry[1]

    def drop(self, graph_id: str) -> bool:
        with self._lock:
            existed = self._items.pop(graph_id, None) is not None
            for mode, gid in list(self._latest.items()):
                if gid == graph_id:
                    self._latest.pop(mode, None)
            return existed

    def stats(self) -> Dict:
        with self._lock:
            self._evict_locked()
            return {
                "count": len(self._items),
                "capacity": self._capacity,
                "ttl_seconds": self._ttl,
                "latest": dict(self._latest),
                "graphs": [
                    {
                        "graph_id": gid,
                        "mode": graph.mode,
                        "nodes": graph.node_count,
                        "edges": graph.edge_count,
                        "age_seconds": round(time.time() - touched, 1),
                    }
                    for gid, (touched, graph) in self._items.items()
                ],
            }

    def _evict_locked(self) -> None:
        if self._ttl > 0:
            cutoff = time.time() - self._ttl
            for gid in [g for g, (t, _) in self._items.items() if t < cutoff]:
                self._items.pop(gid, None)
                for mode, latest in list(self._latest.items()):
                    if latest == gid:
                        self._latest.pop(mode, None)
        while len(self._items) > self._capacity:
            gid, _ = self._items.popitem(last=False)
            for mode, latest in list(self._latest.items()):
                if latest == gid:
                    self._latest.pop(mode, None)


store = GraphStore()
