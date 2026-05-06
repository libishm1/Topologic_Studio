# Fire, Dynamic Routing, And Reinforcement Learning

## Binary Fire Spread

The binary model uses graph traversal from a start node. It can operate as:

- radial distance buckets from the fire origin, or
- BFS neighborhood expansion.

`/fire-sim` returns a precomputed timeline. `/fire-sim/stream` can stream `step` SSE events.

Frontend behavior:

- precomputed mode accumulates all burning nodes seen so far.
- streaming node mode also accumulates `step` nodes into `fireAccumRef`.

## Temperature Fire Spread

Function: `_compute_temperature_fire_spread`

The temperature model initializes all nodes to ambient 20 C and the fire origin to 120 C. On each step, a node heats from the average of neighboring node temperatures. The fire origin stays at 120 C.

This is a graph heat-spread approximation. It is not CFD, smoke modeling, or validated fire engineering simulation.

SSE event:

```json
{
  "type": "temperature_step",
  "step": 0,
  "temperatures": {
    "ifc_0": 20.0
  }
}
```

Frontend temperature rendering maps node temperatures to graph edge endpoint colors by using `edge_ids`. This replaced an earlier sphere-per-node approach and is more visually coherent with the navigation graph.

## Dynamic Hazard Rerouting

Dynamic path rerouting is enabled only for IFC + temperature streaming when the frontend sends:

- `stream_path=true`
- `path_recompute_interval`
- `path_alpha`
- optional `path_lethality_threshold`

Backend flow:

1. Build a TopologicPy graph from dict-based coordinates/adjacency.
2. Attach temperature metadata to vertices.
3. Attach length, cost, average temperature, and hazard metadata to edges.
4. Use `Graph.ShortestPath(..., edgeKey="cost")`.
5. If a lethality threshold is set, first attempt a filtered graph.
6. If filtered graph fails, fall back to full graph.
7. If TopologicPy pathing fails, fall back to custom Dijkstra.

Cost formula currently implemented:

```text
cost = length * (1.0 + alpha * normalized_hazard)
normalized_hazard = max(0.0, (avg_temp - 20.0) / 100.0)
```

This is a practical scalar hazard weighting, not the full AHP/proximity-index method from the route-prioritization papers.

## Reinforcement Learning

Function: `_q_learning_path`

The backend exposes `/rl/train`, using a tabular Q-learning style process:

- state: current graph node
- actions: adjacent nodes
- terminal reward: high positive reward at exit
- per-step penalty
- optional fire penalty based on ignite time
- epsilon-greedy exploration

The endpoint returns a path as node IDs and optional cell bounding boxes.

## Current RL Risk

In `App.jsx`, `trainRlPath` sends `mode: graphMode`. For IFC workflows, fire simulation uses `effectiveFireMode = viewerMode === "ifc" ? "ifc" : graphMode`, but RL does not use that same effective mode. If the user is in IFC mode and `graphMode` is still `cell`, `/rl/train` may use the wrong graph. Before claiming IFC RL is fully validated, change or test RL mode selection.

## Research Mapping

- Jabi et al. supports the NMT + RL idea.
- Boguslawski dynamic routing supports time-varying hazard and evacuee distribution.
- Zverovich safest/balanced routing supports multi-criteria hazard-aware path selection.
- Tantowi supports future two-way fire-egress coupling.

The current system adopts a lightweight implementation slice: graph fire plus hazard-weighted path cost, with tabular RL as an experimental endpoint.

