# Backend Overview

The backend is a FastAPI app in `topologicpy-web-backend/app/main.py`. It combines request validation, TopologicPy conversion, IFC-derived graph generation, shortest path routing, fire simulation, and RL path training in one module.

## Runtime

- App title/version: `TopologicPy Web Backend`, `0.1.0`.
- Local verified imports: FastAPI `0.124.0`, Uvicorn `0.38.0`, Pydantic `2.12.5`, TopologicPy `0.8.93`.
- CORS allows `http://localhost:5173`, `http://127.0.0.1:5173`, plus comma-separated `CORS_ORIGINS`.
- The backend stores the active graph in process memory:

```python
LAST_GRAPHS = {"wire": None, "cell": None, "ifc": None}
```

This makes route order important. `/ifc-egress-path`, `/fire-sim`, `/fire-sim/stream`, `/rl/train`, and `/graph-meta` require a previous upload or graph build.

## Route Inventory

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check for Render and local tests. |
| `/ifc-egress-graph` | POST | Build an IFC navigation graph from browser-extracted floor, stair, door, and wall geometry. |
| `/ifc-egress-path` | POST | Resolve clicked points to graph nodes and compute a wall-aware path. |
| `/fire-sim` | POST | Return a precomputed fire timeline. |
| `/fire-sim/stream` | GET | Stream fire simulation and optional dynamic reroute events over SSE. |
| `/rl/train` | POST | Train a tabular Q-learning path over the active graph. |
| `/graph-meta` | GET | Return stored graph metadata such as cell bounding boxes. |
| `/upload-topology` | POST | Accept TopologicPy JSON or a pre-converted viewer contract. |
| `/upload-ifc` | POST | Upload an IFC file and convert it server-side through `ifcopenshell` if available. |

## Function Clusters

- Graph utility helpers: `_vertex_coord_map`, `_build_adjacency`, `_nearest_node_id`, `_resolve_start_id`.
- IFC sampling and obstacles: `_sample_walkable_points`, `_build_point_adjacency_hybrid`, `_build_point_adjacency_rectilinear`, `_extract_door_positions`, `_extract_wall_segments_2d`.
- Routing: `_shortest_path_ids`, `_dict_graph_to_topologic_graph`, `_shortest_path_topologic_hazard`.
- Fire and dynamic paths: `_compute_fire_timeline`, `_compute_temperature_fire_spread`, `DynamicPathState`, `_recompute_path_with_hazards`.
- RL: `_q_learning_path`.
- Upload conversion: `json_by_cluster`, `upload_topology`, `upload_ifc`, `convert_ifc_to_contract`.

## Main Constraint

The module is intentionally pragmatic but tightly coupled. When changing behavior, keep the route contract stable or update the matching chunk in [api/](../api/routes.md).

