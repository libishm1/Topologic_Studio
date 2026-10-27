"""IFC navigation-graph and pathfinding endpoints.

``/api/ifc/*`` is the Next contract. The unprefixed ``/ifc-egress-*`` routes
reproduce the Classic contract on top of the same machinery so the old
frontend keeps working against this backend, which is what makes an A/B
comparison on one machine possible.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..graphs import alternatives, navigation_check, pathfinding
from ..models import (
    AlternativeRoute,
    AlternativesRequest,
    AlternativesResponse,
    ComparePathResponse,
    GraphImportRequest,
    GraphResponse,
    IfcEgressPathRequest,
    IfcEgressRequest,
    NavigationCheckResponse,
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


@router.post("/api/ifc/graph/import", response_model=GraphResponse, tags=["ifc"])
def import_graph(req: GraphImportRequest) -> GraphResponse:
    """Store a graph the browser built, so server-side features can use it.

    The browser keeps the arrays, so only the id and statistics come back.
    """
    timer = Timer("api/ifc/graph/import")
    try:
        graph, graph_id = graph_builder.import_graph(req, timer)
    except graph_builder.BuildError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    timer.emit(nodes=graph.node_count, edges=graph.edge_count)
    return GraphResponse(
        graph_id=graph_id,
        mode=graph.mode,
        stats=graph_builder.graph_stats(graph),
        up_axis=graph.up_axis,
        bounds=graph.bounds(),
        timings={k: round(v, 4) for k, v in timer.phases.items()},
    )


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

    if not pathfinding.same_part(graph, start, end):
        # Separate dwellings, or a room reached only through a door the model
        # does not contain. Searching would only confirm it more slowly.
        result = pathfinding.PathResult(
            [], [], float("inf"), req.engine, False,
            note="The start and exit are in separate parts of the building (for "
            "example two dwellings, or a room whose door is not in the model), "
            "so no route joins them.",
        )
        timer.emit(found=False, engine=req.engine, points=0)
        return _path_response(result, req.graph_id)

    with timer.phase("search"):
        result = pathfinding.shortest_path(
            graph, start, end, engine=req.engine, weights=weights, blocked=blocked
        )

    walls_only = pathfinding.blocked_mask(graph, req.use_walls)
    lethal = blocked is not None and (walls_only is None or (blocked != walls_only).any())
    if not result.found and lethal:
        # Relax only the lethality threshold, never the walls: on the lattice a
        # wall is exactly its blocked links, and dropping them routed through
        # walls. A route that needs lethal conditions is shown as the least-bad
        # option but not reported as found (review 2026-10-27, finding W4).
        with timer.phase("retry_without_threshold"):
            retry = pathfinding.shortest_path(
                graph, start, end, engine=req.engine, weights=weights, blocked=walls_only
            )
        if retry.found:
            retry.found = False
            retry.note = (
                "Every route passes through conditions above the lethality "
                "threshold. The least-hazard one is drawn, but it is not a safe route."
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


@router.post("/api/ifc/path/alternatives", response_model=AlternativesResponse, tags=["ifc"])
def alternative_paths(req: AlternativesRequest) -> AlternativesResponse:
    """Up to ``k`` routes, best first. ``distinct`` by default; see graphs/alternatives."""
    graph = _require_graph(req.graph_id, req.mode)
    timer = Timer("api/ifc/path/alternatives")
    start, end = _resolve_endpoints(graph, req)
    temperatures = _temperature_array(graph, req.temperatures)
    weights = pathfinding.hazard_weights(graph, temperatures, req.alpha)
    blocked = pathfinding.blocked_mask(
        graph, req.use_walls, temperatures, req.lethality_threshold
    )

    t0 = time.perf_counter()
    with timer.phase("search"):
        found, notes = alternatives.alternative_routes(
            graph,
            start,
            end,
            k=req.k,
            method=req.method,
            engine=req.engine,
            weights=weights,
            blocked=blocked,
            max_overlap=req.max_overlap,
            max_stretch=req.max_stretch,
            separation=req.separation,
        )
    ms = (time.perf_counter() - t0) * 1000.0

    routes: List[AlternativeRoute] = []
    for rank, alt in enumerate(found, start=1):
        base = _path_response(alt.path, req.graph_id)
        routes.append(
            AlternativeRoute(
                **base.model_dump(exclude={"cost"}),
                cost=round(alt.cost, 4),
                rank=rank,
                stretch=round(alt.stretch, 4),
                overlap=round(alt.overlap, 4),
            )
        )
    timer.emit(method=req.method, routes=len(routes), requested=req.k)
    return AlternativesResponse(
        graph_id=req.graph_id,
        method=req.method,
        engine=routes[0].engine if routes else req.engine,
        requested=req.k,
        routes=routes,
        ms=round(ms, 2),
        note=" ".join(notes) or None,
    )


@router.post("/api/ifc/navigation-check", response_model=NavigationCheckResponse, tags=["ifc"])
async def navigation_check_route(
    file: UploadFile = File(...),
    graph_id: Optional[str] = Form(None),
    spacing: float = Form(0.5, ge=0.25, le=5.0),
    start_point: Optional[str] = Form(None, description="JSON [x, y, z] in the viewer frame"),
    end_point: Optional[str] = Form(None, description="JSON [x, y, z] in the viewer frame"),
    use_walls: bool = Form(True),
    coordination: Optional[str] = Form(
        None, description="JSON, the model's 16-number coordination matrix (column-major)"
    ),
) -> NavigationCheckResponse:
    """Build TopologicPy's ``Navigation`` graph of the IFC and compare it with ours.

    The IFC has to travel: Studio's graph is built from points the browser
    sampled, so the server has never seen the file. See graphs/navigation_check.
    """
    graph = _require_graph(graph_id)
    if not (file.filename or "").lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Only .ifc files can be checked.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file provided.")
    if len(data) > settings.max_upload_mib * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"IFC larger than {settings.max_upload_mib} MiB."
        )
    start = _parse_point(start_point, "start_point")
    end = _parse_point(end_point, "end_point")
    matrix = None
    if coordination:
        try:
            matrix = [float(v) for v in json.loads(coordination)]
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail="coordination must be a JSON list of 16 numbers."
            ) from exc

    return await run_in_threadpool(
        _navigation_check, graph, graph_id, data, spacing, start, end, use_walls, matrix
    )


def _navigation_check(graph, graph_id, data, spacing, start, end, use_walls, matrix):
    timer = Timer("api/ifc/navigation-check")
    try:
        with timer.phase("navigation"):
            nav, cached = navigation_check.navigation_graph(data, spacing)
    except navigation_check.NavigationUnavailable as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"TopologicPy could not build a navigation graph: {exc}"
        ) from exc
    if len(nav.points) == 0:
        raise HTTPException(status_code=422, detail="TopologicPy found no walkable surfaces.")

    studio_route = None
    if start is not None and end is not None:
        component = pathfinding.routable_mask(graph)
        a = pathfinding.resolve_endpoint(graph, start, restrict=component)
        b = pathfinding.resolve_endpoint(graph, end, restrict=component)
        if a is not None and b is not None:
            result = pathfinding.shortest_path(
                graph, a, b, blocked=pathfinding.blocked_mask(graph, use_walls)
            )
            studio_route = result.points if result.found else None

    try:
        with timer.phase("compare"):
            report = navigation_check.compare(
                graph, nav, spacing, studio_route, start, end, coordination=matrix
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    view = report.pop("view_points")
    timer.emit(nodes=report["navigation_nodes"], cached=cached)
    return NavigationCheckResponse(
        graph_id=graph_id,
        spacing=spacing,
        cached=cached,
        nodes_b64=graph_builder._b64(view.astype(np.float32)),
        edges_b64=graph_builder._b64(nav.edges.astype(np.uint32)),
        **report,
    )


def _parse_point(raw: Optional[str], name: str) -> Optional[List[float]]:
    if raw is None or raw == "":
        return None
    try:
        value = json.loads(raw)
        point = [float(v) for v in value[:3]]
    except (ValueError, TypeError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a JSON [x, y, z].") from exc
    if len(point) != 3 or not all(np.isfinite(point)):
        raise HTTPException(status_code=400, detail=f"{name} must be a JSON [x, y, z].")
    return point


@router.post("/ifc-egress-path", tags=["legacy"])
def legacy_egress_path(
    req: IfcEgressPathRequest,
    engine: Optional[str] = Query(None, description="fast | topologicpy"),
) -> Dict:
    """Classic contract, now served by the shared engine layer."""
    graph = _require_graph(req.graph_id, "ifc")
    chosen = engine or req.engine or pathfinding.ENGINE_FAST

    component = pathfinding.routable_mask(graph)
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
    component = pathfinding.routable_mask(graph)
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
