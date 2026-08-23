"""Shortest-path engines.

Two engines, one contract:

``fast``
    A* over the CSR adjacency with a Euclidean heuristic. Admissible because
    every edge weight is at least its Euclidean length, so the result is a
    true optimum, not an approximation. Milliseconds on graphs of this size.

``topologicpy``
    Delegates to ``Graph.ShortestPath``. Kept because it is the semantics the
    research side of this project reasons about, and because 0.9 gained
    ``edgeCostFunc`` / ``edgeFilter``, which let hazard reweighting reuse a
    cached graph instead of rebuilding one per recompute.

Wall blocking is a precomputed per-edge boolean on the graph, so neither
engine pays for geometry tests during the search.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..store import NavGraph

ENGINE_FAST = "fast"
ENGINE_TOPOLOGICPY = "topologicpy"
ENGINES = (ENGINE_FAST, ENGINE_TOPOLOGICPY)


@dataclass
class PathResult:
    node_ids: List[int]
    points: List[List[float]]
    cost: float
    engine: str
    found: bool
    fallback_from: Optional[str] = None
    note: Optional[str] = None

    @property
    def length(self) -> float:
        total = 0.0
        for a, b in zip(self.points, self.points[1:]):
            total += math.dist(a, b)
        return total


def hazard_weights(
    graph: NavGraph,
    temperatures: Optional[np.ndarray],
    alpha: float,
    ambient: float = 20.0,
    span: float = 100.0,
) -> np.ndarray:
    """Per-edge traversal cost: ``length * (1 + alpha * normalised_hazard)``.

    Hazard is the mean endpoint temperature mapped onto 0..1 across
    ``ambient .. ambient + span`` degrees, matching the classic formulation so
    numbers stay comparable between the two lines.
    """
    lengths = graph.edge_lengths.astype(np.float64)
    if temperatures is None or alpha <= 0 or graph.edge_count == 0:
        return lengths
    avg = 0.5 * (temperatures[graph.edges[:, 0]] + temperatures[graph.edges[:, 1]])
    normalised = np.clip((avg - ambient) / span, 0.0, None)
    return lengths * (1.0 + alpha * normalised)


def blocked_mask(
    graph: NavGraph,
    use_walls: bool,
    temperatures: Optional[np.ndarray] = None,
    lethality_threshold: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Combine wall blocking and lethal-temperature blocking into one mask."""
    mask = None
    if use_walls and graph.blocked is not None and graph.blocked.any():
        mask = graph.blocked.copy()
    if (
        lethality_threshold is not None
        and temperatures is not None
        and graph.edge_count
    ):
        avg = 0.5 * (temperatures[graph.edges[:, 0]] + temperatures[graph.edges[:, 1]])
        lethal = avg > lethality_threshold
        mask = lethal if mask is None else (mask | lethal)
    return mask


def shortest_path(
    graph: NavGraph,
    start: int,
    end: int,
    engine: str = ENGINE_FAST,
    weights: Optional[np.ndarray] = None,
    blocked: Optional[np.ndarray] = None,
    allow_fallback: bool = True,
) -> PathResult:
    """Find a path, falling back to the fast engine if TopologicPy cannot."""
    engine = engine if engine in ENGINES else ENGINE_FAST

    if engine == ENGINE_TOPOLOGICPY:
        from .topologic_engine import topologic_shortest_path

        result = topologic_shortest_path(graph, start, end, weights, blocked)
        if result.found or not allow_fallback:
            return result
        fast = astar(graph, start, end, weights, blocked)
        fast.fallback_from = ENGINE_TOPOLOGICPY
        fast.note = result.note or "TopologicPy engine returned no path."
        return fast

    return astar(graph, start, end, weights, blocked)


def astar(
    graph: NavGraph,
    start: int,
    end: int,
    weights: Optional[np.ndarray] = None,
    blocked: Optional[np.ndarray] = None,
) -> PathResult:
    """A* with a Euclidean heuristic over the CSR adjacency."""
    n = graph.node_count
    if not (0 <= start < n) or not (0 <= end < n):
        return PathResult([], [], math.inf, ENGINE_FAST, False, note="Endpoint out of range.")
    if start == end:
        return PathResult([start], [graph.points[start].tolist()], 0.0, ENGINE_FAST, True)

    indptr, neighbours, edge_ids = graph.csr()
    if weights is None:
        weights = graph.edge_lengths.astype(np.float64)

    points = graph.points.astype(np.float64, copy=False)
    goal = points[end]

    # The heuristic must never exceed the true remaining cost. Hazard weighting
    # only ever scales a length up, so straight-line distance stays admissible.
    def heuristic(i: int) -> float:
        d = points[i] - goal
        return math.sqrt(float(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))

    dist = np.full(n, math.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    closed = np.zeros(n, dtype=bool)
    dist[start] = 0.0

    heap: List[tuple] = [(heuristic(start), start)]
    while heap:
        _, node = heapq.heappop(heap)
        if closed[node]:
            continue
        if node == end:
            break
        closed[node] = True
        base = dist[node]
        for k in range(indptr[node], indptr[node + 1]):
            nbr = int(neighbours[k])
            if closed[nbr]:
                continue
            eid = int(edge_ids[k])
            if blocked is not None and blocked[eid]:
                continue
            candidate = base + float(weights[eid])
            if candidate < dist[nbr]:
                dist[nbr] = candidate
                prev[nbr] = node
                heapq.heappush(heap, (candidate + heuristic(nbr), nbr))

    if not math.isfinite(dist[end]):
        return PathResult([], [], math.inf, ENGINE_FAST, False, note="No route between the two points.")

    path: List[int] = [end]
    cursor = end
    while cursor != start:
        cursor = int(prev[cursor])
        if cursor < 0:
            return PathResult([], [], math.inf, ENGINE_FAST, False, note="Path reconstruction failed.")
        path.append(cursor)
    path.reverse()

    return PathResult(
        node_ids=path,
        points=[graph.points[i].tolist() for i in path],
        cost=float(dist[end]),
        engine=ENGINE_FAST,
        found=True,
    )


def reachable_from(
    graph: NavGraph, start: int, blocked: Optional[np.ndarray] = None
) -> np.ndarray:
    """Boolean mask of nodes reachable from ``start`` (BFS over CSR)."""
    n = graph.node_count
    seen = np.zeros(n, dtype=bool)
    if not (0 <= start < n):
        return seen
    indptr, neighbours, edge_ids = graph.csr()
    seen[start] = True
    frontier = [start]
    while frontier:
        nxt: List[int] = []
        for node in frontier:
            for k in range(indptr[node], indptr[node + 1]):
                if blocked is not None and blocked[int(edge_ids[k])]:
                    continue
                nbr = int(neighbours[k])
                if not seen[nbr]:
                    seen[nbr] = True
                    nxt.append(nbr)
        frontier = nxt
    return seen


def largest_component(graph: NavGraph) -> np.ndarray:
    """Boolean mask of the biggest connected component.

    Sampling noise routinely leaves a handful of orphan points floating off
    the model. Snapping a pick to one of those is the classic "no path found"
    complaint, so endpoint resolution restricts itself to this component.
    """
    n = graph.node_count
    best = np.zeros(n, dtype=bool)
    if n == 0:
        return best
    unvisited = np.ones(n, dtype=bool)
    indptr, neighbours, _ = graph.csr()
    while unvisited.any():
        seed = int(np.argmax(unvisited))
        component = np.zeros(n, dtype=bool)
        component[seed] = True
        frontier = [seed]
        while frontier:
            nxt: List[int] = []
            for node in frontier:
                for k in range(indptr[node], indptr[node + 1]):
                    nbr = int(neighbours[k])
                    if not component[nbr]:
                        component[nbr] = True
                        nxt.append(nbr)
            frontier = nxt
        unvisited &= ~component
        if component.sum() > best.sum():
            best = component
    return best


def resolve_endpoint(
    graph: NavGraph,
    point: Optional[Sequence[float]],
    node_id: Optional[str] = None,
    restrict: Optional[np.ndarray] = None,
    max_distance: Optional[float] = None,
) -> Optional[int]:
    """Turn a picked 3D point (or a legacy ``ifc_N`` id) into a node index."""
    if node_id:
        index = _parse_node_id(node_id)
        if index is not None and 0 <= index < graph.node_count:
            return index
    if point is None or len(point) < 3:
        return None
    query = np.asarray(point[:3], dtype=np.float64)
    if not np.all(np.isfinite(query)):
        return None

    if restrict is not None and restrict.any() and not restrict.all():
        candidates = np.flatnonzero(restrict)
        d = np.linalg.norm(graph.points[candidates] - query, axis=1)
        best = int(candidates[int(np.argmin(d))])
        if max_distance is not None and float(d.min()) > max_distance:
            return None
        return best

    return graph.nearest(query, max_distance=max_distance)


def _parse_node_id(node_id: str) -> Optional[int]:
    if not isinstance(node_id, str):
        return None
    raw = node_id[4:] if node_id.startswith("ifc_") else node_id
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
