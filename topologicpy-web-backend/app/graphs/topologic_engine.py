"""TopologicPy-backed graph construction and pathfinding.

Two things changed between 0.8.93 and 0.9.64 that matter here, both measured
on this machine with a 1600-node / 3120-edge grid:

* ``Graph.ByMeshData(..., ontology=False)`` builds in ~1.0s where the classic
  ``Graph.ByVerticesEdges`` path took ~4.3s (and ~12.0s with the default
  ``ontology=True``). Building per-Vertex/per-Edge objects in Python was the
  bottleneck, not the graph itself.
* ``Graph.ShortestPath`` gained ``edgeCostFunc`` and ``edgeFilter``. Hazard
  reweighting no longer needs a rebuild: the classic dynamic-reroute loop
  reconstructed the whole graph every recompute.

So the graph is built once, cached on the :class:`NavGraph`, and reweighted
per query. ``node_id`` travels in each vertex dictionary, which removes the
classic coordinate-rounding reverse map entirely.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np

from ..store import NavGraph

logger = logging.getLogger(__name__)

NODE_KEY = "node_id"
EDGE_KEY = "edge_id"
COST_KEY = "cost"
LENGTH_KEY = "Length"


class TopologicUnavailable(RuntimeError):
    """Raised when topologicpy (or its native core) cannot be imported."""


def _imports():
    """Import topologicpy lazily so the API still boots without it.

    topologicpy 0.9 split the native backend into a separate ``topologic_core``
    distribution. Importing topologicpy alone succeeds, then every geometry
    call fails deep inside ``Core.Backend()``. Surfacing that here as one clear
    error beats a stack trace per request.
    """
    try:
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TopologicUnavailable(str(exc)) from exc
    return Dictionary, Graph, Topology, Vertex, Wire


def available() -> bool:
    try:
        Dictionary, _, _, _, _ = _imports()
        # Touch the native backend so a missing topologic_core is caught here.
        Dictionary.ByKeysValues(["probe"], [1])
        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


def version_info() -> dict:
    info = {"topologicpy": None, "topologic_core": None, "usable": False, "error": None}
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
            info["error"] = "topologicpy imported but the native backend is not usable."
    except Exception as exc:  # pragma: no cover
        info["error"] = str(exc)
    return info


def build_topologic_graph(graph: NavGraph):
    """Build (and memoise) the TopologicPy Graph for ``graph``.

    Edge dictionaries carry the unweighted length under both ``cost`` and
    ``Length``. Hazard weighting is applied per query through ``edgeCostFunc``
    rather than baked in, so one cached graph serves every alpha.
    """
    cached = getattr(graph, "_topologic", None)
    if cached is not None:
        return cached

    Dictionary, Graph, _, _, _ = _imports()

    if graph.node_count == 0 or graph.edge_count == 0:
        raise TopologicUnavailable("Graph has no vertices or edges.")

    coords = graph.points.astype(float).tolist()
    edge_pairs = graph.edges.astype(int).tolist()
    lengths = graph.edge_lengths.astype(float)

    vertex_dicts = [
        Dictionary.ByKeysValues([NODE_KEY], [i]) for i in range(graph.node_count)
    ]
    # The edge index rides along in the dictionary. edgeCostFunc/edgeFilter
    # receive an edge object, not an index, and this is what lets them look up
    # per-edge hazard data exactly instead of matching on rounded geometry.
    edge_dicts = [
        Dictionary.ByKeysValues(
            [COST_KEY, LENGTH_KEY, EDGE_KEY], [float(l), float(l), int(i)]
        )
        for i, l in enumerate(lengths)
    ]

    built = Graph.ByMeshData(
        coords,
        edge_pairs,
        vertexDictionaries=vertex_dicts,
        edgeDictionaries=edge_dicts,
        ontology=False,
    )
    if built is None:
        raise TopologicUnavailable("Graph.ByMeshData returned None.")

    object.__setattr__(graph, "_topologic", built)
    return built


def _vertex_lookup(graph: NavGraph, topo_graph):
    """Map node index -> TopologicPy vertex, memoised on the NavGraph."""
    cached = graph.meta.get("_topologic_vertices")
    if cached is not None:
        return cached
    Dictionary, Graph, Topology, _, _ = _imports()
    lookup = {}
    for v in Graph.Vertices(topo_graph):
        d = Topology.Dictionary(v)
        if d is None:
            continue
        node_id = Dictionary.ValueAtKey(d, NODE_KEY)
        if node_id is not None:
            lookup[int(node_id)] = v
    graph.meta["_topologic_vertices"] = lookup
    return lookup


def topologic_shortest_path(
    graph: NavGraph,
    start: int,
    end: int,
    weights: Optional[np.ndarray] = None,
    blocked: Optional[np.ndarray] = None,
):
    """Shortest path via ``Graph.ShortestPath``, returning a PathResult."""
    from .pathfinding import ENGINE_TOPOLOGICPY, PathResult

    def fail(note: str) -> PathResult:
        return PathResult([], [], math.inf, ENGINE_TOPOLOGICPY, False, note=note)

    try:
        Dictionary, Graph, Topology, Vertex, Wire = _imports()
    except TopologicUnavailable as exc:
        return fail(f"TopologicPy unavailable: {exc}")

    try:
        topo_graph = build_topologic_graph(graph)
        lookup = _vertex_lookup(graph, topo_graph)
    except Exception as exc:
        logger.warning("TopologicPy graph build failed: %s", exc)
        return fail(f"TopologicPy graph build failed: {exc}")

    start_vertex = lookup.get(int(start))
    end_vertex = lookup.get(int(end))
    if start_vertex is None or end_vertex is None:
        return fail("Endpoint not present in the TopologicPy graph.")

    # Reweighting and blocking run as callbacks over the cached graph, keyed by
    # the edge index stored in each edge dictionary. Classic rebuilt the entire
    # TopologicPy graph for every hazard recompute; this does not.
    def _edge_index(edge) -> Optional[int]:
        d = Topology.Dictionary(edge)
        if d is None:
            return None
        value = Dictionary.ValueAtKey(d, EDGE_KEY)
        return int(value) if value is not None else None

    cost_func = None
    filter_func = None

    if weights is not None and len(weights) == graph.edge_count:
        lengths = graph.edge_lengths.astype(np.float64)

        def cost_func(edge):
            index = _edge_index(edge)
            if index is None or not (0 <= index < len(weights)):
                return 1.0
            return float(weights[index])

        # A* stays admissible only while every cost is at least the straight
        # line between its endpoints. Hazard weighting only scales lengths up,
        # but a caller could pass anything, so verify rather than assume.
        use_astar = bool(np.all(weights >= lengths - 1e-9))
    else:
        use_astar = True

    if blocked is not None and blocked.any():
        blocked_set = {int(i) for i in np.flatnonzero(blocked)}

        def filter_func(edge):
            index = _edge_index(edge)
            if index is None:
                return True
            return index not in blocked_set

    try:
        wire = Graph.ShortestPath(
            topo_graph,
            start_vertex,
            end_vertex,
            edgeKey=COST_KEY,
            edgeCostFunc=cost_func,
            edgeFilter=filter_func,
            useAStar=use_astar,
            silent=True,
        )
    except Exception as exc:
        logger.warning("Graph.ShortestPath failed: %s", exc)
        return fail(f"Graph.ShortestPath failed: {exc}")

    if wire is None:
        return fail("TopologicPy found no path.")

    try:
        vertices = Wire.Vertices(wire)
    except Exception as exc:
        return fail(f"Could not read path vertices: {exc}")

    node_ids: List[int] = []
    for v in vertices or []:
        d = Topology.Dictionary(v)
        node_id = Dictionary.ValueAtKey(d, NODE_KEY) if d else None
        if node_id is None:
            # A vertex without our dictionary means the wire was rebuilt; fall
            # back to a nearest-node match rather than dropping the path.
            coords = Vertex.Coordinates(v)
            node_id = graph.nearest(coords)
        if node_id is not None:
            node_ids.append(int(node_id))

    node_ids = _dedupe_consecutive(node_ids)
    if len(node_ids) < 2:
        return fail("TopologicPy returned a degenerate path.")

    points = [graph.points[i].tolist() for i in node_ids]
    if weights is not None and len(weights) == graph.edge_count:
        cost = _path_cost(graph, node_ids, weights)
    else:
        cost = sum(math.dist(a, b) for a, b in zip(points, points[1:]))

    return PathResult(
        node_ids=node_ids,
        points=points,
        cost=float(cost),
        engine=ENGINE_TOPOLOGICPY,
        found=True,
    )


def _path_cost(graph: NavGraph, node_ids: List[int], weights: np.ndarray) -> float:
    index = graph.meta.get("_edge_index_map")
    if index is None:
        index = {}
        for eid in range(graph.edge_count):
            a, b = int(graph.edges[eid, 0]), int(graph.edges[eid, 1])
            index[(a, b)] = eid
            index[(b, a)] = eid
        graph.meta["_edge_index_map"] = index
    total = 0.0
    for a, b in zip(node_ids, node_ids[1:]):
        eid = index.get((a, b))
        if eid is None:
            total += float(np.linalg.norm(graph.points[b] - graph.points[a]))
        else:
            total += float(weights[eid])
    return total


def _dedupe_consecutive(values: List[int]) -> List[int]:
    out: List[int] = []
    for v in values:
        if not out or out[-1] != v:
            out.append(v)
    return out
