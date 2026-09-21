"""TopologicPy-backed pathfinding, on the modern ``TGraph`` class.

TopologicPy ships two graph classes. The legacy ``Graph`` is object-based: a
vertex is a geometric Vertex, an edge an Edge, and identity has to be recovered
through dictionaries or rounded coordinates. ``TGraph`` (0.9.7x) is
index-based, and that changes the arithmetic completely.

Measured on a 2,601-node / 5,100-edge lattice, topologicpy 0.9.71:

===========================================  =======  =======
approach                                     build    query
===========================================  =======  =======
legacy ``Graph.ByMeshData`` + ShortestPath   2.12 s   4.95 s
``TGraph.ByMeshData`` + ShortestPath         118 ms   9.3 ms
built-in A* (for reference)                  -        5.8 ms
===========================================  =======  =======

So the August conclusion - "TopologicPy is ~100x too slow to route with" - was
a fact about the legacy class, not about TopologicPy. On TGraph it is within
about 1.7x of the hand-written A*.

``ShortestPath`` returns node indices directly, and those indices are the rows
of :class:`~app.store.NavGraph`, so the coordinate-rounding reverse map the old
implementation needed is gone.

``ByMeshData`` is used rather than the faster ``ByEdgeIndexPairs`` (6 ms) for a
correctness reason: ``ByEdgeIndexPairs`` builds a purely topological graph and
its ``edgeDictionaries`` are *not* read by ``ShortestPath``, so
``edgeKey="Length"`` silently degenerates into counting hops. That is easy to
miss on a unit-spaced test grid, where hop count and distance agree.
``ByMeshData`` carries the coordinates, so ``Length`` is the real edge length.

Hazard weighting and wall blocking go through ``edgeCostFunc`` / ``edgeFilter``.
Each receives a dict carrying ``index``, the edge's row in ``NavGraph.edges``,
so per-edge data is looked up exactly and one cached graph serves every query.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from ..store import NavGraph

logger = logging.getLogger(__name__)

LENGTH_KEY = "Length"


class TopologicUnavailable(RuntimeError):
    """Raised when topologicpy (or its native core) cannot be used."""


def _tgraph():
    """Import TGraph lazily so the API still boots without topologicpy.

    topologicpy 0.9 split the native backend into a separate ``topologic_core``
    distribution. Importing topologicpy alone succeeds and then every geometry
    call fails deep inside ``Core.Backend()``. Surfacing that here as one clear
    error beats a stack trace per request.
    """
    try:
        from topologicpy.TGraph import TGraph
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TopologicUnavailable(str(exc)) from exc
    return TGraph


def available() -> bool:
    try:
        TGraph = _tgraph()
        # Touch the native backend: a two-node graph is enough to prove it works.
        return TGraph.ByMeshData([[0, 0, 0], [1, 0, 0]], [[0, 1]]) is not None
    except Exception:  # pragma: no cover - environment dependent
        return False


def version_info() -> dict:
    info = {
        "topologicpy": None,
        "topologic_core": None,
        "graph_class": "TGraph",
        "usable": False,
        "error": None,
    }
    try:
        import topologicpy

        info["topologicpy"] = getattr(topologicpy, "__version__", "unknown")
    except Exception as exc:
        info["error"] = f"topologicpy import failed: {exc}"
        return info
    try:
        import topologic_core

        info["topologic_core"] = getattr(topologic_core, "__version__", "installed")
    except Exception as exc:
        info["error"] = f"topologic_core import failed: {exc}"
        return info
    try:
        info["usable"] = available()
        if not info["usable"]:
            info["error"] = "topologicpy imported but TGraph could not be built."
    except Exception as exc:  # pragma: no cover
        info["error"] = str(exc)
    return info


def build_topologic_graph(graph: NavGraph):
    """Build and memoise the TGraph for ``graph``.

    One cached graph serves every query: weighting and blocking are applied per
    call through callbacks rather than by rebuilding.
    """
    cached = getattr(graph, "_topologic", None)
    if cached is not None:
        return cached

    if graph.node_count == 0 or graph.edge_count == 0:
        raise TopologicUnavailable("Graph has no vertices or edges.")

    TGraph = _tgraph()
    built = TGraph.ByMeshData(
        graph.points.astype(float).tolist(),
        graph.edges.astype(int).tolist(),
    )
    if built is None:
        raise TopologicUnavailable("TGraph.ByMeshData returned None.")

    object.__setattr__(graph, "_topologic", built)
    return built


def topologic_shortest_path(
    graph: NavGraph,
    start: int,
    end: int,
    weights: Optional[np.ndarray] = None,
    blocked: Optional[np.ndarray] = None,
):
    """Shortest path via ``TGraph.ShortestPath``, returned as a PathResult."""
    from .pathfinding import ENGINE_TOPOLOGICPY, PathResult

    def fail(note: str) -> PathResult:
        return PathResult([], [], math.inf, ENGINE_TOPOLOGICPY, False, note=note)

    if not (0 <= start < graph.node_count) or not (0 <= end < graph.node_count):
        return fail("Endpoint out of range.")
    if start == end:
        return PathResult(
            [start], [graph.points[start].tolist()], 0.0, ENGINE_TOPOLOGICPY, True
        )

    try:
        TGraph = _tgraph()
        topo = build_topologic_graph(graph)
    except TopologicUnavailable as exc:
        return fail(str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("TGraph build failed: %s", exc)
        return fail(f"TGraph build failed: {exc}")

    # Both callbacks receive a dict whose "index" is the edge's row in
    # NavGraph.edges, so per-edge lookup is exact.
    cost_func = None
    if weights is not None and len(weights) == graph.edge_count:

        def cost_func(edge, _w=weights):
            index = edge.get("index")
            if isinstance(index, (int, np.integer)) and 0 <= index < len(_w):
                return float(_w[index])
            return 1.0

    filter_func = None
    if blocked is not None and blocked.any():

        def filter_func(edge, _b=blocked):
            index = edge.get("index")
            if isinstance(index, (int, np.integer)) and 0 <= index < len(_b):
                return not bool(_b[index])
            return True

    try:
        result = TGraph.ShortestPath(
            topo,
            int(start),
            int(end),
            edgeKey=LENGTH_KEY,
            edgeCostFunc=cost_func,
            edgeFilter=filter_func,
            returnCost=True,
            silent=True,
        )
    except Exception as exc:
        logger.warning("TGraph.ShortestPath failed: %s", exc)
        return fail(f"TGraph.ShortestPath failed: {exc}")

    node_ids, cost = _unpack(result)
    valid = [i for i in node_ids if 0 <= i < graph.node_count]
    if len(valid) < 2:
        return fail("TopologicPy found no path.")

    points = [graph.points[i].tolist() for i in valid]
    if cost is None or not math.isfinite(cost):
        cost = sum(math.dist(a, b) for a, b in zip(points, points[1:]))

    return PathResult(
        node_ids=valid,
        points=points,
        cost=float(cost),
        engine=ENGINE_TOPOLOGICPY,
        found=True,
    )


def _unpack(result) -> Tuple[List[int], Optional[float]]:
    """Normalise ShortestPath's return value.

    With ``returnCost=True`` it yields a tuple whose first element is the list
    of node indices and whose last numeric element is the cost; without it, a
    bare list of indices. Both shapes are accepted so a version bump cannot
    silently break routing.
    """
    if result is None:
        return [], None

    if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)):
        indices = [int(i) for i in result[0] if isinstance(i, (int, np.integer))]
        cost = None
        for part in reversed(result[1:]):
            if isinstance(part, (int, float)) and not isinstance(part, bool):
                cost = float(part)
                break
        return indices, cost

    if isinstance(result, (list, tuple)):
        return [int(i) for i in result if isinstance(i, (int, np.integer))], None

    return [], None
