"""IFC navigation-graph and pathfinding endpoints.

``/api/ifc/*`` is the Next contract. The unprefixed ``/ifc-egress-*`` routes
reproduce the Classic contract on top of the same machinery so the old
frontend keeps working against this backend, which is what makes an A/B
comparison on one machine possible.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..graphs import pathfinding
from ..models import (
    ComparePathResponse,
    GraphResponse,
    IfcEgressPathRequest,
    IfcEgressRequest,
    PathRequest,
    PathResponse,
    PointCloudRequest,
)
from ..perf import Timer
from ..services import graph_builder
from ..store import NavGraph, store

router = APIRouter()


# --------------------------------------------------------------------------
# graph construction


@router.post("/api/ifc/graph", response_model=GraphResponse, tags=["ifc"])
def build_graph_from_points(req: PointCloudRequest) -> GraphResponse:
    """Build a navigation graph from browser-sampled walkable points."""
    timer = Timer("api/ifc/graph")
    try:
        graph, graph_id = graph_builder.build_from_point_cloud(req, timer)
    except graph_builder.BuildError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = _graph_response(graph, graph_id, timer)
    timer.emit(nodes=graph.node_count, edges=graph.edge_count)
    return response


@router.post("/ifc-egress-graph", tags=["legacy"])
def legacy_egress_graph(req: IfcEgressRequest) -> Dict:
    """Classic contract: raw triangle soup in, edge list out."""
    timer = Timer("ifc-egress-graph")
    try:
        graph, graph_id = graph_builder.build_from_ifc_geometry(req, timer)
    except graph_builder.BuildError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stats = graph_builder.graph_stats(graph)
    with timer.phase("serialize"):
        edges, edge_ids = graph_builder.legacy_edge_payload(graph)
        coords = graph.coord_map()
    timer.emit(nodes=graph.node_count, edges=graph.edge_count)

    return {
        "mode": "ifc",
        "graph_id": graph_id,
        "stats": stats.model_dump(),
        "edges": edges,
        "edge_ids": edge_ids,
        "coords": coords,
    }


@router.get("/api/ifc/graph/{graph_id}", response_model=GraphResponse, tags=["ifc"])
def get_graph(graph_id: str) -> GraphResponse:
    graph = _require_graph(graph_id)
    return _graph_response(graph, graph_id, None)


@router.delete("/api/ifc/graph/{graph_id}", tags=["ifc"])
def delete_graph(graph_id: str) -> Dict:
    return {"deleted": store.drop(graph_id)}


@router.get("/api/graphs", tags=["ifc"])
def list_graphs() -> Dict:
    return store.stats()


# --------------------------------------------------------------------------
# pathfinding


@router.post("/api/ifc/path", response_model=PathResponse, tags=["ifc"])
def compute_path(req: PathRequest) -> PathResponse:
    graph = _require_graph(req.graph_id, req.mode)
    timer = Timer("api/ifc/path")

    start, end = _resolve_endpoints(graph, req)
    temperatures = _temperature_array(graph, req.temperatures)
    weights = pathfinding.hazard_weights(graph, temperatures, req.alpha)
    blocked = pathfinding.blocked_mask(
        graph, req.use_walls, temperatures, req.lethality_threshold
    )

    with timer.phase("search"):
        result = pathfinding.shortest_path(
            graph, start, end, engine=req.engine, weights=weights, blocked=blocked
        )

    if not result.found and blocked is not None and blocked.any():
        # A lethality threshold that isolates the exit is a common own-goal.
        # Retry unblocked rather than reporting a flat "no path".
        with timer.phase("retry_unblocked"):
            retry = pathfinding.shortest_path(
                graph, start, end, engine=req.engine, weights=weights, blocked=None
            )
        if retry.found:
            retry.note = (
                "No route avoids every blocked edge; this path ignores the "
                "lethality threshold and/or wall blocking."
            )
            result = retry

    timer.emit(found=result.found, engine=result.engine, points=len(result.points))
    return _path_response(result, req.graph_id or store.stats().get("latest", {}).get(req.mode))


@router.post("/api/ifc/path/compare", response_model=ComparePathResponse, tags=["ifc"])
def compare_engines(req: PathRequest) -> ComparePathResponse:
    """Run both engines on one graph and report whether they agree.

    This is the parity check the roadmap asks for before retiring either
    engine, exposed as an endpoint so it can be run against real models
    rather than only in unit tests.
    """
    graph = _require_graph(req.graph_id, req.mode)
    start, end = _resolve_endpoints(graph, req)
    temperatures = _temperature_array(graph, req.temperatures)
    weights = pathfinding.hazard_weights(graph, temperatures, req.alpha)
    blocked = pathfinding.blocked_mask(
        graph, req.use_walls, temperatures, req.lethality_threshold
    )

    t0 = time.perf_counter()
    fast = pathfinding.shortest_path(
        graph, start, end, engine=pathfinding.ENGINE_FAST, weights=weights, blocked=blocked
    )
    fast_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    topo = pathfinding.shortest_path(
        graph,
        start,
        end,
        engine=pathfinding.ENGINE_TOPOLOGICPY,
        weights=weights,
        blocked=blocked,
        allow_fallback=False,
    )
    topo_ms = (time.perf_counter() - t0) * 1000.0

    same = bool(fast.found and topo.found and fast.node_ids == topo.node_ids)
    delta = (
        abs(fast.cost - topo.cost)
        if fast.found and topo.found and np.isfinite(fast.cost) and np.isfinite(topo.cost)
        else float("inf")
    )

    return ComparePathResponse(
        graph_id=req.graph_id,
        fast=_path_response(fast, req.graph_id),
        topologicpy=_path_response(topo, req.graph_id),
        same_route=same,
        cost_delta=float(delta),
        fast_ms=round(fast_ms, 3),
        topologicpy_ms=round(topo_ms, 3),
    )


@router.post("/ifc-egress-path", tags=["legacy"])
def legacy_egress_path(
    req: IfcEgressPathRequest,
    engine: Optional[str] = Query(None, description="fast | topologicpy"),
) -> Dict:
    """Classic contract, now served by the shared engine layer."""
    graph = _require_graph(req.graph_id, "ifc")
    chosen = engine or req.engine or pathfinding.ENGINE_FAST

    component = pathfinding.largest_component(graph)
    start = pathfinding.resolve_endpoint(graph, req.start_point, restrict=component)
    end = pathfinding.resolve_endpoint(graph, req.end_point, restrict=component)
    if start is None or end is None:
        raise HTTPException(status_code=400, detail="Invalid start or end point.")

    result = pathfinding.shortest_path(
        graph,
        start,
        end,
        engine=chosen,
        blocked=pathfinding.blocked_mask(graph, True),
    )
    if not result.found:
        raise HTTPException(status_code=404, detail="No path found between start and end points.")
    return {"mode": "ifc", "points": result.points, "engine": result.engine}


# --------------------------------------------------------------------------
# helpers


def _require_graph(graph_id: Optional[str], mode: str = "ifc") -> NavGraph:
    graph = store.get(graph_id, mode)
    if graph is None:
        raise HTTPException(
            status_code=400,
            detail="No navigation graph available. Build one first (POST /api/ifc/graph).",
        )
    return graph


def _resolve_endpoints(graph: NavGraph, req: PathRequest):
    component = pathfinding.largest_component(graph)
    start = pathfinding.resolve_endpoint(graph, req.start_point, req.start_id, component)
    end = pathfinding.resolve_endpoint(graph, req.end_point, req.end_id, component)
    if start is None:
        raise HTTPException(status_code=400, detail="Could not resolve the start point.")
    if end is None:
        raise HTTPException(status_code=400, detail="Could not resolve the end point.")
    return start, end


def _temperature_array(
    graph: NavGraph, temperatures: Optional[Dict[str, float]]
) -> Optional[np.ndarray]:
    if not temperatures:
        return None
    field = np.full(graph.node_count, 20.0, dtype=np.float64)
    for key, value in temperatures.items():
        index = pathfinding._parse_node_id(key)
        if index is not None and 0 <= index < graph.node_count:
            field[index] = float(value)
    return field


def _graph_response(graph: NavGraph, graph_id: str, timer: Optional[Timer]) -> GraphResponse:
    encoded = graph_builder.encode_graph(graph)
    return GraphResponse(
        graph_id=graph_id,
        mode=graph.mode,
        stats=graph_builder.graph_stats(graph),
        up_axis=graph.up_axis,
        bounds=graph.bounds(),
        timings={k: round(v, 4) for k, v in (timer.phases if timer else {}).items()},
        **encoded,
    )


def _path_response(result, graph_id: Optional[str]) -> PathResponse:
    return PathResponse(
        graph_id=graph_id,
        found=result.found,
        points=result.points,
        node_ids=result.node_ids,
        cost=float(result.cost) if np.isfinite(result.cost) else 0.0,
        length=round(result.length, 4),
        engine=result.engine,
        fallback_from=result.fallback_from,
        note=result.note,
    )
