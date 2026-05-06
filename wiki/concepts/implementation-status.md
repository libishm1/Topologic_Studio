# Current Implementation Status

## Git And Workspace Status

The active git repository is `TopologicStudio/`. Before this wiki was generated, the working tree already had modified app files and untracked runtime/support artifacts. This wiki did not intentionally modify app source files.

Important committed history from `git log --oneline -n 20` includes:

- `f936b86 Optimize hazard path recompute`
- `733f7b2 Document codebase overview`
- `16c0f50 Tune IFC egress graph thresholds`
- `e78a66f Add IFC fire temperature overlays and graph updates`
- `2982ac6 Add IFC temperature fire stream toggle`
- `6866749 Refine IFC egress adjacency and fire spread`
- `704a255 Tune IFC egress connectivity controls`
- `94ee99b Improve IFC viewer load stability`
- `1fa4fce Fix viewer fit handling and fire sim defaults`
- `7337fbf Add deploy config and env-based API base`

Current uncommitted work adds or refines:

- Render/GitHub Pages deployment docs and workflow variable fallback.
- `topologicpy` in backend requirements.
- door and wall geometry in IFC egress graph requests.
- wall-aware path computation.
- graph edge ID return values.
- per-edge fire coloring in `IFCViewer`.
- non-blocking/yielded IFC extraction behavior.

## Feature Matrix

| Feature | Status | Evidence |
|---|---|---|
| React/Vite frontend | Implemented | `package.json`, build succeeds |
| FastAPI backend | Implemented | backend import lists expected routes |
| TopologicPy JSON upload | Implemented with validation improvements | `/upload-topology` checks contract/list formats and wraps parse errors |
| Browser IFC loading | Implemented | `IFCViewer.jsx` uses That Open Components and web-ifc |
| IFC ID extraction | Implemented | slabs, stairs, coverings, doors, spaces, walls, storeys |
| IFC geometry extraction | Implemented | floors, stairs, doors, walls sent to backend |
| IFC graph generation | Implemented | `/ifc-egress-graph` |
| Hybrid distance graph | Implemented | `_build_point_adjacency_hybrid` |
| Grid-snap rectilinear graph | Implemented | `_build_point_adjacency_rectilinear` |
| Door waypoint injection | Implemented | `_extract_door_positions` and forced nearest-neighbor connection |
| Wall-aware pathing | Implemented | path-time segment intersection and door exemptions |
| Wall-blocked grid cells | Partial/dormant | helper exists, not wired into grid-snap call |
| Static IFC path rendering | Implemented | red line in `IFCViewer.jsx` |
| Binary fire spread | Implemented | `/fire-sim`, `/fire-sim/stream` step events |
| Temperature fire spread | Implemented | `_compute_temperature_fire_spread` and `temperature_step` SSE |
| Per-edge temperature coloring | Implemented | `edge_ids` + vertex color attributes |
| Dynamic hazard rerouting | Implemented | `DynamicPathState`, `Graph.ShortestPath`, `path_update` SSE |
| RL endpoint | Implemented | `/rl/train` |
| IFC-specific RL | Needs verification | frontend sends `mode: graphMode`, not necessarily `ifc` |
| GitHub Pages deployment | Implemented | workflow with `BASE_PATH` and `VITE_API_BASE` |
| Render backend deployment | Configured | `render.yaml`, backend Dockerfile |
| Lint health | Failing | see [[reports/verification]] |
| Production build | Passing with warning | see [[reports/verification]] |

## Implementation Posture

The system has moved beyond a static viewer. The major current implementation additions are semantic IFC extraction, a backend egress graph builder, interactive pathing, fire simulation, and dynamic rerouting. The main unfinished work is correctness hardening: tests, lint cleanup, graph/session isolation, IFC axis handling, route validation against known samples, and performance management for large IFCs.

