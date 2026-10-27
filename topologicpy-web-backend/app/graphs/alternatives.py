"""Alternative egress routes.

Two methods, one result shape:

``yen``
    The k shortest loopless paths, in increasing cost (Yen 1971). This is what
    ``TGraph.KShortestPaths`` computes; TopologicPy ships it from 0.9.87, and
    Studio pins 0.9.86 (see requirements.txt), so the built-in implementation
    below serves both engines until the pin moves. When the installed TGraph
    has ``KShortestPaths`` and the TopologicPy engine is selected, it is used.

    On a navigation lattice Yen answers a different question from the one an
    egress designer asks. Many paths tie, and the second-best path is the best
    one with a node or two stepped around. On the Duplex (1,765 nodes, upper
    floor to a ground-floor door) the second to fifth routes were all 19.17 m,
    exactly the first route's length, and lay entirely within 1.5 m of it. Kept
    because it is TopologicPy's semantics and the exact answer to "the next
    cheapest path".

``distinct``
    Routes that use different space - the "is there a second way out?"
    question. A penalty method (Akgün, Erkut and Batta 2000): after each route,
    every edge within ``separation`` metres of it costs ``1 + penalty`` times
    more, and the search runs again. A candidate is kept when at most
    ``max_overlap`` of its length runs within ``separation`` of a route already
    kept, and it costs at most ``max_stretch`` times the shortest. Fewer than
    ``k`` routes back is itself a finding: the building offers no other route
    within that stretch. Same Duplex query: 19.17 m and 27.40 m, the second
    sharing 12% of its length with the first, and no third route within 1.5x;
    53 ms, against 170 ms for three Yen routes.

Overlap is measured spatially, not by shared edges. On a lattice two routes
can share no edge yet run half a metre apart down the same corridor, and a
fire that blocks one blocks both.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..store import NavGraph
from .pathfinding import ENGINE_FAST, ENGINE_TOPOLOGICPY, PathResult, astar, shortest_path

METHODS = ("distinct", "yen")

#: Metres within which two routes count as sharing space. Corridors run about
#: 1.2-2 m wide and the sampler keeps nodes off the walls, so two routes down
#: one corridor stay within this; at 1 m they could count as "different" while
#: a fire in that corridor blocks both.
SEPARATION = 1.5


@dataclass
class Alternative:
    path: PathResult
    #: Cost under the request's own weights (hazard included), never the
    #: penalised weights the distinct search uses internally.
    cost: float
    #: ``cost`` over the best route's cost.
    stretch: float
    #: Share of this route's length within ``separation`` of an earlier route.
    overlap: float


def alternative_routes(
    graph: NavGraph,
    start: int,
    end: int,
    k: int = 3,
    method: str = "distinct",
    engine: str = ENGINE_FAST,
    weights: Optional[np.ndarray] = None,
    blocked: Optional[np.ndarray] = None,
    max_overlap: float = 0.5,
    max_stretch: float = 1.5,
    separation: float = SEPARATION,
    penalty: float = 1.0,
    time_budget: float = 5.0,
) -> Tuple[List[Alternative], List[str]]:
    """Up to ``k`` routes from ``start`` to ``end``, best first, plus notes."""
    base = _base_weights(graph, weights)
    if method == "yen":
        routes, notes = _yen(graph, start, end, k, engine, base, blocked, time_budget)
    else:
        routes, notes = _distinct(
            graph, start, end, k, engine, base, blocked,
            max_overlap, max_stretch, separation, penalty, time_budget,
        )
        # Found in the order the penalties uncovered them, not by cost.
        routes = routes[:1] + sorted(routes[1:], key=lambda r: r[1])
    if not routes:
        return [], notes

    best = routes[0][1]
    out: List[Alternative] = []
    kept: List[np.ndarray] = []
    for path, cost in routes:
        points = np.asarray(path.points, dtype=np.float64)
        overlap = max((_shared_share(points, other, separation) for other in kept), default=0.0)
        stretch = cost / best if best > 0 else 1.0
        out.append(Alternative(path, float(cost), float(stretch), float(overlap)))
        kept.append(_samples(points, separation / 4.0))

    if method == "distinct" and len(out) < k:
        notes.append(
            f"Found {len(out)} of {k} routes that use substantially different space "
            f"(at most {round(max_overlap * 100)}% within {separation:g} m of a route found before) "
            f"within {max_stretch:g}x the shortest cost. The search stopped there; it does not prove "
            f"that no other such route exists. Yen lists the next cheapest routes exactly."
        )
    return out, notes


# --------------------------------------------------------------------------
# distinct: penalty method


def _distinct(
    graph, start, end, k, engine, base, blocked,
    max_overlap, max_stretch, separation, penalty, time_budget,
):
    notes: List[str] = []
    first = shortest_path(graph, start, end, engine=engine, weights=base, blocked=blocked)
    if not first.found:
        return [], [first.note or "No route between the two points."]

    best_cost = _true_cost(graph, first.node_ids, base)
    routes = [(first, best_cost)]
    kept = [_samples(np.asarray(first.points, dtype=np.float64), separation / 4.0)]

    midpoints = 0.5 * (
        graph.points[graph.edges[:, 0]].astype(np.float64)
        + graph.points[graph.edges[:, 1]].astype(np.float64)
    )
    mid_tree = cKDTree(midpoints)
    penalised = base.copy()
    last = first
    deadline = time.perf_counter() + time_budget

    # Every route found, kept or not, pushes the next search away from it, so
    # a rejected near-duplicate still makes progress.
    for _ in range(4 * k):
        if len(routes) >= k:
            break
        if time.perf_counter() > deadline:
            notes.append(f"Stopped after {time_budget:g} s with {len(routes)} routes.")
            break
        samples = _samples(np.asarray(last.points, dtype=np.float64), separation / 2.0)
        near = mid_tree.query_ball_point(samples, r=separation)
        hit = np.unique(np.fromiter(itertools.chain.from_iterable(near), dtype=np.int64))
        if len(hit) == 0:
            break
        penalised[hit] *= 1.0 + penalty

        candidate = shortest_path(
            graph, start, end, engine=engine, weights=penalised, blocked=blocked
        )
        if not candidate.found:
            break
        cost = _true_cost(graph, candidate.node_ids, base)
        if cost > max_stretch * best_cost + 1e-9:
            # Penalties only ever rise, so later routes are pushed further out
            # still. Past the stretch limit there is nothing left to find.
            break
        last = candidate
        points = np.asarray(candidate.points, dtype=np.float64)
        if max(_shared_share(points, other, separation) for other in kept) <= max_overlap:
            routes.append((candidate, cost))
            kept.append(_samples(points, separation / 4.0))

    return routes, notes


# --------------------------------------------------------------------------
# yen: k shortest loopless paths


def _yen(graph, start, end, k, engine, base, blocked, time_budget):
    if engine == ENGINE_TOPOLOGICPY:
        from .topologic_engine import has_k_shortest_paths, topologic_k_shortest_paths

        if has_k_shortest_paths():
            found = topologic_k_shortest_paths(graph, start, end, k, base, blocked, time_budget)
            if found is not None:
                return [(p, _true_cost(graph, p.node_ids, base)) for p in found], []
            notes = ["TGraph.KShortestPaths failed; the built-in Yen search ran instead."]
        else:
            notes = [
                "TGraph.KShortestPaths ships with topologicpy 0.9.87; this server has "
                "an earlier version, so the built-in Yen search ran instead."
            ]
    else:
        notes = []

    mask = blocked.copy() if blocked is not None else np.zeros(graph.edge_count, dtype=bool)
    first = astar(graph, start, end, base, mask)
    if not first.found:
        return [], [first.note or "No route between the two points."]

    indptr, _, edge_ids = graph.csr()
    accepted: List[Tuple[List[int], float]] = [(first.node_ids, first.cost)]
    candidates: list = []
    seen = {tuple(first.node_ids)}
    counter = itertools.count()
    deadline = time.perf_counter() + time_budget

    while len(accepted) < k:
        previous, _ = accepted[-1]
        previous_edges = _path_edges(graph, previous, base)
        root_cost = 0.0
        timed_out = False
        for i in range(len(previous) - 1):
            if time.perf_counter() > deadline:
                timed_out = True
                break
            root = previous[: i + 1]
            spur_mask = mask.copy()
            # Leave the root by an edge no accepted path with this root used.
            for nodes, _ in accepted:
                if len(nodes) > i + 1 and nodes[: i + 1] == root:
                    spur_mask[_edges_between(graph, nodes[i], nodes[i + 1])] = True
            # Loopless: the spur may not revisit the root.
            for node in root[:-1]:
                spur_mask[edge_ids[indptr[node]: indptr[node + 1]]] = True

            spur = astar(graph, previous[i], end, base, spur_mask)
            if spur.found:
                total = root[:-1] + spur.node_ids
                key = tuple(total)
                if key not in seen:
                    seen.add(key)
                    heapq.heappush(candidates, (root_cost + spur.cost, next(counter), total))
            root_cost += float(base[previous_edges[i]])
        if timed_out:
            notes.append(f"Stopped after {time_budget:g} s with {len(accepted)} routes.")
            break
        if not candidates:
            break
        cost, _, nodes = heapq.heappop(candidates)
        accepted.append((nodes, cost))

    routes = []
    for nodes, cost in accepted:
        result = PathResult(
            node_ids=list(nodes),
            points=[graph.points[i].tolist() for i in nodes],
            cost=float(cost),
            engine=ENGINE_FAST,
            found=True,
        )
        routes.append((result, float(cost)))
    return routes, notes


# --------------------------------------------------------------------------
# helpers


def _base_weights(graph: NavGraph, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None or len(weights) != graph.edge_count:
        return graph.edge_lengths.astype(np.float64)
    return np.asarray(weights, dtype=np.float64)


def _edges_between(graph: NavGraph, u: int, v: int) -> np.ndarray:
    indptr, neighbours, edge_ids = graph.csr()
    lo, hi = indptr[u], indptr[u + 1]
    return edge_ids[lo:hi][neighbours[lo:hi] == v]


def _path_edges(graph: NavGraph, nodes: List[int], weights: np.ndarray) -> np.ndarray:
    """The cheapest edge joining each consecutive pair of ``nodes``."""
    out = np.empty(max(len(nodes) - 1, 0), dtype=np.int64)
    for i, (u, v) in enumerate(zip(nodes, nodes[1:])):
        between = _edges_between(graph, u, v)
        out[i] = between[int(np.argmin(weights[between]))]
    return out


def _true_cost(graph: NavGraph, nodes: List[int], weights: np.ndarray) -> float:
    if len(nodes) < 2:
        return 0.0
    return float(weights[_path_edges(graph, nodes, weights)].sum())


def _samples(points: np.ndarray, step: float) -> np.ndarray:
    """Points along a polyline, no two consecutive ones more than ``step`` apart."""
    if len(points) < 2:
        return points.reshape(-1, 3)
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


def _shared_share(points: np.ndarray, other_samples: np.ndarray, separation: float) -> float:
    """Share of the polyline ``points`` lying within ``separation`` of another route."""
    if len(points) < 2 or len(other_samples) == 0:
        return 0.0
    a, b = points[:-1], points[1:]
    lengths = np.linalg.norm(b - a, axis=1)
    total = float(lengths.sum())
    if total <= 0:
        return 1.0
    distance, _ = cKDTree(other_samples).query(0.5 * (a + b))
    return float(lengths[distance <= separation].sum() / total)
