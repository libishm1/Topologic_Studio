"""Cross-check a Studio graph against TopologicPy's own navigation graph.

TopologicPy 0.9.86 added ``Navigation``. It reads the IFC with IfcOpenShell,
keeps the upward-facing walkable surfaces, cuts obstacles at walking height
out of them, and lays a grid over what is left, with stairs, ramps and lifts as
explicit connectors. Studio reaches the same place by another road: the
browser samples the fragments and the server links the samples. The two
graphs are independent estimates of the same walkable space, so where they
disagree, one of them is wrong about the building.

Frames. Navigation works in IFC world coordinates, metres, Z up. The viewer
loads with web-ifc's ``COORDINATE_TO_ORIGIN`` and Y up, so its points are the
IFC points turned by ``(x, y, z) -> (x, z, -y)`` and then moved by the model's
coordination matrix, which That Open reports (``model.getCoordinationMatrix``).
Studio then lifts every node by the agent height. With the matrix the frame
change is exact: on the Duplex it is a pure translation (-2.514, -3.262,
-11.635), and with the 0.75 m lift added the two graphs sit within a median
0.21 m of each other at 0.5 m spacing.

Without the matrix (a caller that cannot supply it) the offset is fitted by a
translation-only trimmed ICP. That is a fallback, not an equal: two lattices
match almost as well shifted by part of a grid step, and the Duplex fit
slid 0.15 m along x, which cut Studio's measured coverage from 91% to 87%. A
storey only one graph has is worse - the fit can put one graph's upper floor
on the other's ground floor. The response says which alignment was used.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.adjacency import KIND_DOOR, KIND_FLOOR, KIND_STAIR
from ..store import NavGraph

#: web-ifc's turn from IFC's Z-up frame into three.js's Y-up one.
TURNS = {
    "y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "z": np.eye(3),
}
KIND_NAMES = {KIND_FLOOR: "floor", KIND_STAIR: "stair", KIND_DOOR: "door"}

#: Share of nearest-neighbour pairs the fit trusts at each step.
TRIM = 0.7
#: A route endpoint further than this from Navigation's walkable space is not
#: snapped onto it: the comparison would be between two different routes.
MAX_SNAP = 2.0


class NavigationUnavailable(RuntimeError):
    """topologicpy is missing, or too old to have ``Navigation``."""


def available() -> bool:
    try:
        from topologicpy.Navigation import Navigation  # noqa: F401
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------
# building Navigation's graph


@dataclass
class NavigationGraph:
    """Navigation's graph as arrays, in the IFC frame."""

    points: np.ndarray
    edges: np.ndarray
    kinds: List[str]
    storey_of: List[str]
    storeys: List[Dict]
    tgraph: object
    parse_ms: float
    graph_ms: float


_lock = threading.Lock()
_models: "OrderedDict[str, tuple]" = OrderedDict()
_graphs: "OrderedDict[tuple, NavigationGraph]" = OrderedDict()
_CACHE = 3


def navigation_graph(data: bytes, spacing: float) -> tuple:
    """``(NavigationGraph, cached)`` for an IFC file's bytes.

    Parsing is the slow half (3.1 s on the Duplex, against 1.7-4.1 s for the
    grid), so the parsed model is kept per file and only the grid is rebuilt
    when the spacing changes. One build at a time: TopologicPy's native core
    is not known to be thread-safe.
    """
    try:
        from topologicpy.Navigation import Navigation
        from topologicpy.TGraph import TGraph
    except Exception as exc:
        raise NavigationUnavailable(
            f"topologicpy.Navigation is not available ({exc}). It needs topologicpy 0.9.86 or later."
        ) from exc

    digest = hashlib.sha256(data).hexdigest()
    key = (digest, round(float(spacing), 4))
    with _lock:
        if key in _graphs:
            _graphs.move_to_end(key)
            return _graphs[key], True

        parse_ms = 0.0
        if digest in _models:
            _models.move_to_end(digest)
            model = _models[digest]
        else:
            t0 = time.perf_counter()
            model = _parse(Navigation, data)
            parse_ms = (time.perf_counter() - t0) * 1000.0
            _models[digest] = model
            while len(_models) > _CACHE:
                _models.popitem(last=False)

        t0 = time.perf_counter()
        tgraph = Navigation.Graph(model, spacing=float(spacing), graphType="tgraph")
        vertices = TGraph.Vertices(tgraph)
        edges = TGraph.Edges(tgraph)
        graph_ms = (time.perf_counter() - t0) * 1000.0

        row = {v["index"]: i for i, v in enumerate(vertices)}
        built = NavigationGraph(
            points=np.array(
                [[v["dictionary"][axis] for axis in "xyz"] for v in vertices], dtype=np.float64
            ).reshape(-1, 3),
            edges=np.array(
                [[row[e["src"]], row[e["dst"]]] for e in edges if e["src"] in row and e["dst"] in row],
                dtype=np.int64,
            ).reshape(-1, 2),
            kinds=[str(v["dictionary"].get("kind", "floor")) for v in vertices],
            storey_of=[str(v["dictionary"].get("storey", "")) for v in vertices],
            storeys=[
                {"id": str(s["id"]), "name": str(s.get("name") or s["id"]), "elevation": float(s["elevation"])}
                for s in model.storeys
            ],
            tgraph=tgraph,
            parse_ms=parse_ms,
            graph_ms=graph_ms,
        )
        _graphs[key] = built
        while len(_graphs) > _CACHE:
            _graphs.popitem(last=False)
        return built, False


def _parse(Navigation, data: bytes):
    """``Navigation.ByIFCPath`` wants a path, so the upload goes to a temp file."""
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
    try:
        handle.write(data)
        handle.close()
        return Navigation.ByIFCPath(handle.name)
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass


# --------------------------------------------------------------------------
# lining the two graphs up


@dataclass
class Alignment:
    turn: np.ndarray
    offset: np.ndarray
    median: float
    p90: float
    #: "coordination" (exact, from the viewer) or "fitted" (ICP fallback).
    method: str = "fitted"

    def apply(self, points: np.ndarray) -> np.ndarray:
        return points @ self.turn.T + self.offset


def from_coordination(
    matrix: Sequence[float],
    ifc_points: np.ndarray,
    studio_points: np.ndarray,
    up_axis: str,
    lift: float = 0.0,
) -> Alignment:
    """The viewer's own frame change: web-ifc's turn, the coordination matrix
    (16 numbers, column-major as three.js stores them), then Studio's lift."""
    if up_axis not in TURNS:
        raise ValueError(f"Cannot compare a graph whose up axis is {up_axis!r}.")
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (16,) or not np.all(np.isfinite(values)):
        raise ValueError("The coordination matrix must be 16 finite numbers.")
    m = values.reshape(4, 4).T
    offset = m[:3, 3].copy()
    offset["xyz".index(up_axis)] += lift
    fit = Alignment(m[:3, :3] @ TURNS[up_axis], offset, 0.0, 0.0, "coordination")
    distance, _ = cKDTree(studio_points.astype(np.float64)).query(fit.apply(ifc_points))
    fit.median = float(np.median(distance))
    fit.p90 = float(np.quantile(distance, 0.9))
    return fit


def align(ifc_points: np.ndarray, studio_points: np.ndarray, up_axis: str) -> Alignment:
    """Fit the offset that carries Navigation's points onto Studio's.

    Fallback for callers without the coordination matrix; see the module
    docstring for how it goes wrong. Three starts (matching minima, centroids,
    maxima) and the lowest median gap wins.
    """
    if up_axis not in TURNS:
        raise ValueError(f"Cannot compare a graph whose up axis is {up_axis!r}.")
    turn = TURNS[up_axis]
    turned = ifc_points @ turn.T
    target = studio_points.astype(np.float64)
    tree = cKDTree(target)

    best = None
    for offset in (
        target.min(axis=0) - turned.min(axis=0),
        target.mean(axis=0) - turned.mean(axis=0),
        target.max(axis=0) - turned.max(axis=0),
    ):
        offset = offset.copy()
        for _ in range(60):
            distance, index = tree.query(turned + offset)
            keep = distance <= np.quantile(distance, TRIM)
            step = np.median(target[index[keep]] - (turned[keep] + offset), axis=0)
            offset += step
            if float(np.linalg.norm(step)) < 1e-5:
                break
        distance, _ = tree.query(turned + offset)
        fit = (float(np.median(distance)), float(np.quantile(distance, 0.9)), offset)
        if best is None or fit[0] < best[0]:
            best = fit
    return Alignment(turn, best[2], best[0], best[1])


def covered(points: np.ndarray, others: np.ndarray, radius: float) -> np.ndarray:
    """Which of ``points`` have one of ``others`` within ``radius``."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if len(others) == 0:
        return np.zeros(len(points), dtype=bool)
    distance, _ = cKDTree(others).query(points, distance_upper_bound=radius)
    return distance <= radius


# --------------------------------------------------------------------------
# the comparison


def compare(
    graph: NavGraph,
    nav: NavigationGraph,
    spacing: float,
    studio_route: Optional[Sequence[Sequence[float]]] = None,
    start_point: Optional[Sequence[float]] = None,
    end_point: Optional[Sequence[float]] = None,
    coordination: Optional[Sequence[float]] = None,
) -> Dict:
    """Coverage both ways, per storey and per node kind, plus one route each."""
    notes: List[str] = []
    t0 = time.perf_counter()
    if coordination is not None:
        lift = float(graph.meta.get("agent_height", 0.0))
        fit = from_coordination(coordination, nav.points, graph.points, graph.up_axis, lift)
    else:
        fit = align(nav.points, graph.points, graph.up_axis)
        notes.append(
            "No coordination matrix was sent, so the two graphs were lined up by "
            "fitting. A fit can slide by part of a grid step, or by a storey when "
            "only one graph has it; treat the coverage figures as approximate."
        )
    view = fit.apply(nav.points)
    align_ms = (time.perf_counter() - t0) * 1000.0

    radius = float(spacing)
    nav_hit = covered(view, graph.points, radius)
    studio_hit = covered(graph.points.astype(np.float64), view, radius)
    if fit.median > spacing:
        notes.append(
            f"The two graphs did not line up (median gap {fit.median:.2f} m at "
            f"{spacing:g} m spacing), so the coverage figures are not meaningful. "
            "Check that this IFC is the loaded model and the up axis is right."
        )

    by_kind = {}
    for code, name in KIND_NAMES.items():
        mask = graph.kinds == code
        if mask.any():
            by_kind[name] = round(float(studio_hit[mask].mean()), 4)

    storeys = _per_storey(graph, nav, fit, nav_hit, studio_hit)

    route = None
    if start_point is not None and end_point is not None:
        t1 = time.perf_counter()
        route = _route(nav, view, studio_route, start_point, end_point)
        route["ms"] = round((time.perf_counter() - t1) * 1000.0, 2)

    count, _labels, _largest = _components(len(nav.points), nav.edges)
    nav_kinds: Dict[str, int] = {}
    for kind in nav.kinds:
        nav_kinds[kind] = nav_kinds.get(kind, 0) + 1

    return {
        "radius": radius,
        "navigation_nodes": int(len(nav.points)),
        "navigation_edges": int(len(nav.edges)),
        "navigation_components": int(count),
        "navigation_kinds": nav_kinds,
        "studio_nodes": graph.node_count,
        "studio_edges": graph.edge_count,
        "alignment": fit.method,
        "offset": [round(float(v), 4) for v in fit.offset],
        "residual_median": round(fit.median, 4),
        "residual_p90": round(fit.p90, 4),
        "navigation_covered": round(float(nav_hit.mean()) if len(nav_hit) else 0.0, 4),
        "studio_covered": round(float(studio_hit.mean()) if len(studio_hit) else 0.0, 4),
        "studio_covered_by_kind": by_kind,
        "storeys": storeys,
        "route": route,
        "view_points": view,
        "timings": {
            "parse_ms": round(nav.parse_ms, 1),
            "graph_ms": round(nav.graph_ms, 1),
            "align_ms": round(align_ms, 1),
        },
        "notes": notes,
    }


def _per_storey(graph, nav, fit, nav_hit, studio_hit) -> List[Dict]:
    """Coverage per IFC storey.

    Navigation tags every node with its storey. Studio nodes are assigned to
    the highest storey whose (fitted) floor lies at or below them, with half a
    metre of slack; stair nodes land on the storey they climb from.
    """
    up = graph.up_index()
    heights = graph.points[:, up].astype(np.float64)
    levels = sorted(nav.storeys, key=lambda s: s["elevation"])
    floor_at = [float(fit.apply(np.array([[0.0, 0.0, s["elevation"]]]))[0, up]) for s in levels]
    # Navigation's nodes sit on the slab; Studio's are lifted by the agent
    # height, which the fitted offset carries. Map through it.
    assigned = np.full(len(heights), 0, dtype=np.int64)
    for i, level in enumerate(floor_at):
        assigned[heights >= level - 0.5] = i
    nav_storey = np.array(nav.storey_of)

    out = []
    for i, level in enumerate(levels):
        nav_mask = nav_storey == level["id"]
        studio_mask = assigned == i
        if not nav_mask.any() and not studio_mask.any():
            continue
        out.append(
            {
                "id": level["id"],
                "name": level["name"],
                "elevation": round(level["elevation"], 3),
                "navigation_nodes": int(nav_mask.sum()),
                "studio_nodes": int(studio_mask.sum()),
                "navigation_covered": round(float(nav_hit[nav_mask].mean()), 4) if nav_mask.any() else None,
                "studio_covered": round(float(studio_hit[studio_mask].mean()), 4) if studio_mask.any() else None,
            }
        )
    return out


def _route(nav, view, studio_route, start_point, end_point) -> Dict:
    from .topologic_engine import _tgraph, _unpack

    result: Dict = {
        "found": False,
        "studio_length": _polyline_length(studio_route) if studio_route else None,
        "navigation_length": None,
        "ratio": None,
        "points": [],
        "note": None,
    }
    if len(view) == 0:
        result["note"] = "Navigation produced no walkable nodes."
        return result

    # Snap to the biggest component within reach rather than to the nearest
    # node: a one-node fragment under a wall is often nearest. Snapping only
    # into the overall largest component is wrong the other way - when the
    # storeys are not connected it reports "no walkable space" for a floor
    # that has plenty, and hides the real finding.
    _, labels, _ = _components(len(view), nav.edges)
    sizes = np.bincount(labels)
    tree = cKDTree(view)
    snapped = []
    for label, point in (("start", start_point), ("exit", end_point)):
        query = np.asarray(point[:3], dtype=np.float64)
        near = tree.query_ball_point(query, r=MAX_SNAP)
        if not near:
            distance, _ = tree.query(query)
            result["note"] = (
                f"Navigation has no walkable space within {MAX_SNAP:g} m of the {label} "
                f"(nearest {distance:.1f} m), so there is no route to compare."
            )
            return result
        near = np.asarray(near)
        gaps = np.linalg.norm(view[near] - query, axis=1)
        snapped.append(int(near[np.lexsort((gaps, -sizes[labels[near]]))[0]]))
    if labels[snapped[0]] != labels[snapped[1]]:
        result["note"] = (
            "In Navigation's graph the start and exit are not connected "
            f"(it has {len(sizes)} separate parts at this spacing), so it has no route."
        )
        return result

    try:
        TGraph = _tgraph()
        raw = TGraph.ShortestPath(
            nav.tgraph, snapped[0], snapped[1], edgeKey="Length", returnCost=True, silent=True
        )
    except Exception as exc:
        result["note"] = f"TGraph.ShortestPath failed on Navigation's graph: {exc}"
        return result

    nodes, _cost = _unpack(raw)
    if len(nodes) < 2:
        result["note"] = "Navigation's graph has no route between the two points."
        return result
    points = view[nodes]
    result["found"] = True
    result["points"] = points.round(4).tolist()
    result["navigation_length"] = round(_polyline_length(points), 4)
    if result["studio_length"]:
        result["ratio"] = round(result["navigation_length"] / result["studio_length"], 4)
    return result


def _components(n: int, edges: np.ndarray):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if n == 0:
        return 0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    matrix = coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n, n)
    ) if len(edges) else coo_matrix((n, n), dtype=np.int8)
    count, labels = connected_components(matrix, directed=False)
    largest = labels == int(np.argmax(np.bincount(labels)))
    return int(count), labels, largest


def _polyline_length(points) -> float:
    array = np.asarray(points, dtype=np.float64)
    if len(array) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())
