"""Fire simulation, dynamic rerouting and reinforcement learning endpoints."""
from __future__ import annotations

import json
import time
from typing import Dict, Generator, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..graphs import fire as fire_model
from ..graphs import pathfinding
from ..graphs.rl import q_learning_path
from ..models import FireSimRequest, RLRequest, RLResponse
from ..store import NavGraph, store

router = APIRouter()


def _require_graph(graph_id: Optional[str], mode: str) -> NavGraph:
    graph = store.get(graph_id, mode)
    if graph is None:
        raise HTTPException(
            status_code=400, detail="No graph available. Build a navigation graph first."
        )
    return graph


def _seed(graph: NavGraph, req) -> int:
    component = pathfinding.largest_component(graph)
    start = pathfinding.resolve_endpoint(
        graph, getattr(req, "start_point", None), getattr(req, "start_id", None), component
    )
    if start is None:
        # Falling back to node 0 hid real errors in Classic; be explicit.
        raise HTTPException(status_code=400, detail="Could not resolve the fire start point.")
    return start


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/api/fire/timeline", tags=["simulation"])
def fire_timeline(req: FireSimRequest) -> Dict:
    """Precomputed ignition timeline, returned in one response."""
    graph = _require_graph(req.graph_id, req.mode)
    start = _seed(graph, req)
    blocked = pathfinding.blocked_mask(graph, req.use_walls)

    if req.model == "flood":
        timeline = fire_model.flood_timeline(graph, start, req.max_steps, blocked)
    else:
        timeline = fire_model.radial_timeline(graph, start, req.max_steps)

    return {
        "mode": req.mode,
        "graph_id": req.graph_id,
        "model": req.model,
        "start_id": start,
        "steps": len(timeline),
        "timeline": timeline,
    }


@router.get("/api/fire/stream", tags=["simulation"])
def fire_stream(
    graph_id: Optional[str] = None,
    mode: str = "ifc",
    model: str = "radial",
    start_id: Optional[str] = None,
    end_id: Optional[str] = None,
    start_x: Optional[float] = None,
    start_y: Optional[float] = None,
    start_z: Optional[float] = None,
    end_x: Optional[float] = None,
    end_y: Optional[float] = None,
    end_z: Optional[float] = None,
    max_steps: int = Query(60, ge=1, le=2000),
    delay_ms: int = Query(200, ge=0, le=10000),
    use_walls: bool = True,
    stream_path: bool = False,
    path_start_x: Optional[float] = None,
    path_start_y: Optional[float] = None,
    path_start_z: Optional[float] = None,
    path_recompute_interval: int = Query(5, ge=1, le=200),
    path_alpha: float = Query(0.5, ge=0.0, le=10.0),
    path_lethality_threshold: Optional[float] = None,
    path_engine: str = "fast",
) -> StreamingResponse:
    """Server-sent events: fire front, temperatures and live rerouting."""
    graph = _require_graph(graph_id, mode)

    start_point = _point(start_x, start_y, start_z)
    end_point = _point(end_x, end_y, end_z)
    component = pathfinding.largest_component(graph)
    seed = pathfinding.resolve_endpoint(graph, start_point, start_id, component)
    if seed is None:
        raise HTTPException(status_code=400, detail="Could not resolve the fire start point.")
    exit_node = pathfinding.resolve_endpoint(graph, end_point, end_id, component)

    # The evacuee's origin is independent of where the fire starts. Classic
    # defaulted the two to the same node, which put the walker inside the fire;
    # that default is preserved only when no explicit origin is supplied.
    walker_start = pathfinding.resolve_endpoint(
        graph, _point(path_start_x, path_start_y, path_start_z), None, component
    )
    if walker_start is None:
        walker_start = seed

    blocked_walls = pathfinding.blocked_mask(graph, use_walls)
    delay = max(delay_ms, 0) / 1000.0

    def generate() -> Generator[str, None, None]:
        yield _sse(
            {
                "type": "meta",
                "graph_id": graph_id,
                "model": model,
                "nodes": graph.node_count,
                "edges": graph.edge_count,
                "start_id": seed,
                "end_id": exit_node,
            }
        )

        if model == "temperature":
            options = fire_model.FireOptions(
                model="temperature", max_steps=max_steps, blocked=blocked_walls
            )
            last_path: List[int] = []
            last_recompute = -10**9
            for step, temps in enumerate(fire_model.temperature_steps(graph, seed, options)):
                yield _sse(
                    {
                        "type": "temperature_step",
                        "step": step,
                        "temperatures": fire_model.temperature_payload(temps),
                    }
                )

                if (
                    stream_path
                    and exit_node is not None
                    and step - last_recompute >= path_recompute_interval
                ):
                    last_recompute = step
                    weights = pathfinding.hazard_weights(graph, temps, path_alpha)
                    blocked = pathfinding.blocked_mask(
                        graph, use_walls, temps, path_lethality_threshold
                    )
                    result = pathfinding.shortest_path(
                        graph, walker_start, exit_node,
                        engine=path_engine, weights=weights, blocked=blocked,
                    )
                    if not result.found and blocked is not None:
                        result = pathfinding.shortest_path(
                            graph, walker_start, exit_node,
                            engine=path_engine, weights=weights, blocked=blocked_walls,
                        )
                        if result.found:
                            result.note = "Ignoring the lethality threshold; no safe route remained."
                    changed = result.node_ids != last_path
                    last_path = result.node_ids
                    yield _sse(
                        {
                            "type": "path_update",
                            "step": step,
                            "found": result.found,
                            "path": result.points,
                            "cost": None if not np.isfinite(result.cost) else round(result.cost, 3),
                            "changed": bool(changed),
                            "engine": result.engine,
                            "note": result.note,
                        }
                    )

                if delay:
                    time.sleep(delay)
        else:
            timeline = (
                fire_model.flood_timeline(graph, seed, max_steps, blocked_walls)
                if model == "flood"
                else fire_model.radial_timeline(graph, seed, max_steps)
            )
            for step, nodes in enumerate(timeline):
                yield _sse({"type": "step", "step": step, "nodes": nodes})
                if delay:
                    time.sleep(delay)

        yield _sse({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/rl/train", response_model=RLResponse, tags=["simulation"])
def rl_train(req: RLRequest) -> RLResponse:
    graph = _require_graph(req.graph_id, req.mode)
    component = pathfinding.largest_component(graph)
    start = pathfinding.resolve_endpoint(graph, req.start_point, req.start_id, component)
    exit_node = pathfinding.resolve_endpoint(graph, req.exit_point, req.exit_id, component)
    if start is None or exit_node is None:
        raise HTTPException(status_code=400, detail="Invalid start or exit point.")

    ignite = None
    if req.use_fire:
        timeline = fire_model.flood_timeline(
            graph, start, req.max_steps, pathfinding.blocked_mask(graph, True)
        )
        ignite = fire_model.ignition_times(timeline)

    result = q_learning_path(
        graph,
        start,
        exit_node,
        ignite_time=ignite,
        episodes=req.episodes,
        max_steps=req.max_steps,
        seed=req.seed,
    )
    return RLResponse(
        graph_id=req.graph_id,
        mode=req.mode,
        path=result.path,
        points=[graph.points[i].tolist() for i in result.path],
        reached_exit=result.reached_exit,
        episodes=result.episodes,
    )


@router.get("/graph-meta", tags=["legacy"])
def graph_meta(mode: str = "ifc", graph_id: Optional[str] = None) -> Dict:
    graph = _require_graph(graph_id, mode)
    return {
        "mode": mode,
        "graph_id": graph_id,
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "bounds": graph.bounds(),
        "cell_bboxes": graph.meta.get("bboxes", []),
    }


def _point(x, y, z) -> Optional[List[float]]:
    if x is None or y is None or z is None:
        return None
    return [x, y, z]
