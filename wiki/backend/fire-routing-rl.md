# Fire Routing And RL

This chunk covers fire spread, wall-aware shortest paths, dynamic hazard routing, and tabular Q-learning in `app/main.py`.

## Static IFC Path

`/ifc-egress-path` resolves clicked `start_point` and `end_point` to nearest graph nodes, then calls `_shortest_path_ids`.

`_shortest_path_ids` is a Dijkstra implementation:

- Edge cost is Euclidean length.
- If wall segments exist, an edge is skipped when its horizontal projection intersects a wall segment.
- Edges touching door nodes are exempt.
- The route returns node IDs, then the route endpoint maps IDs back to coordinates.

## Fire Timeline

`/fire-sim` returns precomputed timeline data. `/fire-sim/stream` streams each step over Server-Sent Events.

Two timeline modes exist:

- Radial mode uses distance buckets from the start node.
- BFS mode spreads over graph neighborhoods.

The frontend accumulates node IDs for visual display in non-temperature mode.

## Temperature Mode

`_compute_temperature_fire_spread` initializes every node at ambient temperature and pins the ignition node to fire temperature. Each step updates a node based on its neighbors' average temperature and `heat_transfer_rate`.

Default constants in code:

- Ambient: `20.0`
- Fire source: `120.0`
- Heat transfer: `1.20`

The SSE event shape for this mode is `temperature_step` with `step` and `temperatures`.

## Dynamic Hazard Rerouting

Dynamic path streaming only activates when:

- `mode == "ifc"`
- `use_temperature == true`
- `stream_path == true`
- start and end nodes can be resolved

The reroute pipeline:

1. Create `DynamicPathState`.
2. At each recompute interval, convert the dict graph to a TopologicPy `Graph`.
3. Store `temperature`, `length`, `cost`, and `hazard` metadata on vertices or edges.
4. Try a lethality-threshold graph if requested.
5. Fall back to a full hazard-weighted graph.
6. Fall back again to custom Dijkstra if TopologicPy pathfinding fails.
7. Emit `path_update` SSE events.

The cost formula is effectively:

```text
cost = length * (1 + alpha * normalized_hazard)
```

where `normalized_hazard` maps roughly from `20 C` to `120 C`.

## RL Path Training

`/rl/train` calls `_q_learning_path`.

The current RL is tabular:

- State: graph node ID.
- Action: neighbor node ID.
- Reward: positive for reaching exit, negative for fire/hazard timing, step penalty otherwise.
- Output: a node ID path, not a full learned policy artifact.

## Development Notes

- Dynamic hazard routing uses TopologicPy for the main route, not only custom Dijkstra.
- The fallback Dijkstra keeps the UI useful when TopologicPy graph conversion or path extraction fails.
- RL currently uses the selected `graphMode`, while IFC dynamic pathing uses the IFC graph path.

