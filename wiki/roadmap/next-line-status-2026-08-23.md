# Topologic Studio Next — status against the handoff plan

Date: 2026-08-23
Branch: `next/studio-2` (worktree at `topologic_webapp/TopologicStudio-Next`)
Baseline: `claude/ifc-port-baseline` @ `38e08b7`, tagged `demo-safe-2026-08-24`

This records what the handoff asked for, what was done, and what was measured.
Everything below is on the Next line only; Classic is byte-for-byte unchanged.

---

## Handoff tickets

| Ticket | Status | Notes |
| --- | --- | --- |
| 1. Demo-safe freeze | Done | Tag `demo-safe-2026-08-24` on `38e08b7`. Uncommitted working-tree edits in the Classic checkout are **not** in the tag. |
| 2. Frontend runtime hardening | Done | `optimizeDeps.exclude`, StrictMode removed, WASM self-hosted. |
| 3. Dependency reconciliation | Done | `requirements.txt` is the single source of truth and now includes `ifcopenshell` and `topologic-core`. |
| 4. Import cleanup | Done | `sys.path.append("C:/Users/sarwj/...")` deleted. |
| 5. Pathfinding unification spike | Done | One engine interface, parity verified, both exposed via `/api/ifc/path/compare`. |

## Roadmap phases

| Phase | Status |
| --- | --- |
| 1. Stabilise | Done |
| 2. Harden frontend runtime | Done |
| 3. Measure before rewriting | Done — see numbers below |
| 4. Fragment-first IFC track | Done — IndexedDB `.frag` cache, no second parse |
| 5. Rationalise pathfinding | Done |
| 6. TopologicPy modernisation | Done for the graph/path surface; `Graph.ByIFCFile` not adopted (see below) |

---

## Measurements

Duplex model, 7,036 triangles, same machine, identical input payload.

### Graph build (Classic `/ifc-egress-graph` contract, both backends)

| | Classic | Next |
| --- | --- | --- |
| Build time (median of 3) | 0.380 s | 0.388 s |
| Nodes | 1,364 | 3,459 |
| Edges | 23,388 | 22,895 |
| Path query | 476 ms | 8 ms |
| Path found | **no** | yes |

Next produces a 2.5x denser graph for the same build cost, and routes on it
59x faster. Classic returned `found=false` on its own graph for this model.

### Next fast path (browser sampling → `/api/ifc/graph`)

| Stage | Time |
| --- | --- |
| Worker sampling (7,036 triangles → 2,637 points) | 5 ms |
| `POST /api/ifc/graph` | 225 ms |
| `POST /api/ifc/path` (fast engine) | 14 ms |

Result: 2,651 nodes, 17,677 edges, **one** connected component.

### Engine parity and cost

Both engines return the same route with a cost delta of `0.00000`.

| Engine | Cold (incl. graph build) | Warm |
| --- | --- | --- |
| `fast` | 14 ms | 14 ms |
| `topologicpy` | 56.7 s | 1.44 s |

`Graph.ByMeshData(ontology=False)` is 4x faster than `Graph.ByVerticesEdges`
and 11x faster than the default `ontology=True`, but a 2,651-node graph still
costs tens of seconds to construct. Hence: fast engine by default, TopologicPy
opt-in with an in-UI warning.

Micro-benchmark on a synthetic 1,600-node / 3,120-edge grid:

| Call | Time |
| --- | --- |
| `Graph.ByVerticesEdges(ontology=True)` | 12.04 s |
| `Graph.ByVerticesEdges(ontology=False)` | 4.26 s |
| `Graph.ByMeshData(ontology=False)` | **1.06 s** |
| `ShortestPath(useAStar=True)` | 0.41 s |
| `ShortestPath(edgeCostFunc=...)` | 0.33 s |

---

## TopologicPy 0.9.64 findings

Every API Classic used still exists. Nothing was removed.

**New and adopted:**

- `Graph.ShortestPath(useAStar=, edgeCostFunc=, edgeFilter=, returnVertices=)`.
  `edgeCostFunc(edge) -> float` and `edgeFilter(edge) -> bool` are called once
  per edge and receive the edge with its dictionary intact, so hazard
  reweighting and wall blocking work against a **cached** graph. Classic
  rebuilt the entire TopologicPy graph on every hazard recompute.
- Vertex dictionaries survive onto path vertices, which removes Classic's
  coordinate-rounding reverse map entirely.
- `Graph.ByMeshData(...)` — takes raw coordinate lists and index pairs.

**New and not adopted (deliberately):**

- `Graph.ByIFCFile` / `Graph.ByIFCPath` build a *topological relationship*
  graph from IFC entities and relationships. That is a different object from a
  walkable navigation lattice, so it does not replace the egress pipeline. It
  is worth a separate feature ("IFC relationship explorer"), not a swap.
- `Graph.NavigationGraph(face, ...)` operates on a single face with obstacles.
  Promising for per-storey routing, but it does not handle multi-storey stair
  connectivity, which is the core requirement here.

**Packaging change (the important one):**

TopologicPy 0.9 split the native backend into a separate `topologic-core`
distribution (currently 8.0.4). `pip install topologicpy` succeeds, `import
topologicpy` succeeds, and then the first geometry call dies with
`ImportError: Could not import topologic_core`. Both are pinned, and the
Docker build runs a probe that fails the image rather than the first request.

**Licence:** PyPI classifies topologicpy as
`GNU Lesser General Public License v3 or later` (LGPL-3.0-or-later), not
AGPL-3.0. This removes the network-copyleft concern previously recorded
against deepening server-side use.

---

## Risks from the handoff audit

| Risk | Resolution |
| --- | --- |
| A. Hardcoded `sys.path.append` | Removed. |
| B. Process-global graph state | Replaced by `GraphStore` (LRU + TTL, `graph_id` per build). |
| C. Two path stacks | Unified; parity verified in tests and via `/api/ifc/path/compare`. |
| D. Geometry extracted twice | Extracted once from fragments; the second IFC parse is gone. |
| E. StrictMode around viewer bootstrap | Removed. |
| F. `web-ifc` not excluded from Vite optimizer | Excluded, along with `@thatopen/fragments`. |
| G. Remote WASM CDN | Self-hosted via `scripts/sync-assets.mjs` on postinstall — this also covers the fragments worker, which That Open fetches from unpkg.com by default. |
| H. Dependency split | One `requirements.txt`, Docker installs only from it. |
| I. Partial axis handling | All geometry goes through `geometry/common.py`. Fixed a real bug: agent height was added to index 2 unconditionally, which displaced points sideways on the y-up models the frontend actually defaulted to. |
| J. `main.py` too large | Split into the module tree in the README. |

---

## Bugs found and fixed along the way

1. **Agent height on the wrong axis.** Classic added `agent_height` to
   coordinate index 2 regardless of `up_axis`, while the frontend defaulted to
   `ifcUpAxis = "y"`. Points were pushed sideways rather than upward.
2. **Wall touch counted as a crossing.** A rewrite of the segment intersection
   test using sign comparison treated a zero cross-product (endpoint lying on
   the wall line) as an intersection, which would wall off any node grazing a
   wall. Caught by a unit test; the strict form is restored.
3. **Endpoint snapping to orphan nodes.** Nearest-node lookup could snap a pick
   to a disconnected sample, producing "no path found" on a perfectly routable
   model. Endpoint resolution is now restricted to the largest connected
   component. This is the most likely cause of Classic's `found=false`.
4. **RL rollout emitted a repeated node.** The greedy rollout appended a node
   and *then* checked whether it had been visited, so a trapped policy returned
   a path ending in a duplicate. It now stops cleanly and reports whether the
   exit was reached.
5. **RL tie-breaking depended on adjacency order.** `max(neighbours, key=q)`
   resolved ties by list position, making results depend on graph construction
   order. Ties now break randomly under a caller-supplied seed.
6. **Evacuee started inside the fire.** The dynamic reroute defaulted the path
   start to the fire seed. An explicit walker origin is now accepted; the old
   default only applies when none is given.
7. **Edge explosion.** A pure radius query couples node density to edge count
   quadratically — 0.5 m sampling inside a 2.25 m radius gave 56 edges/node and
   a 195k-edge, 8.3 MB graph. Capping neighbours per node cut it to 22,895
   edges with no loss of route quality.

---

## Test coverage

- 73 backend tests (`pytest`): geometry, adjacency, obstacles, pathfinding,
  fire models, RL, graph store, engine parity, API contracts including the
  Classic compatibility routes.
- `tools/smoke-pipeline.mjs`: runs the real sampler worker source in Node
  against real IFC geometry and drives a live backend through graph build,
  routing, both engines, fire timeline and SSE rerouting.
- `tools/bench_egress.py`: Classic-vs-Next benchmark on any IFC file.
- Frontend: ESLint clean (including the React compiler rules), production build
  clean, dev-server module graph crawled and fully resolvable.

## Browser verification (2026-08-24)

Driven through real Chrome 151 with `tools/browser-test.mjs` (playwright-core
against the installed browser, no bundled download). 21 checks: app boot, WebGL
context, IFC load, category counts, 3D render, graph build, click-picking,
routing, fire streaming, theme, every panel tab, and the topology mode. All
pass with **zero console errors, zero uncaught exceptions and zero failed
requests**.

Measured in-browser on the Duplex model: IFC load 0.7 s, graph build 0.6 s
(2,577 nodes / 17,211 edges), worker sampling 2,563 points in 4 ms, route
5.7 m across two storeys via the stairs.

### Bugs the browser found that headless testing had not

1. **Cross-origin isolation broke IFC loading entirely.** The dev server set
   COOP/COEP headers so the multi-threaded web-ifc build could engage. That
   made the page cross-origin isolated, so web-ifc selected its pthread build,
   which spawned workers with an undefined script URL. Every worker fetched
   `/undefined`, got `index.html` back, and died on
   `Uncaught SyntaxError: Unexpected token '<'`. Headers removed; dev and
   production now run the same single-threaded path.
2. **The fragments worker was fetched from unpkg.com at runtime.**
   `FragmentsManager.getWorker()` downloads
   `unpkg.com/@thatopen/fragments@<version>/dist/worker/worker.mjs` on boot,
   the same CDN dependency already removed for the WASM. It is now copied into
   `public/fragments/` by `scripts/sync-assets.mjs`, with the CDN kept only as
   a fallback.
3. **A door produced one navigation node per mesh, not per door.** An IFC door
   is several meshes (frame, leaf, glazing), so 14 doors became 44 door nodes.
   Meshes now carry their `itemId` and the worker aggregates by it. The clone
   step in `buildGraph` was dropping the id, which is what kept the first
   attempt at this fix from taking effect.
4. **Missing favicon**, a 404 on every page load.

### UX problems the screenshots exposed

1. **Building a graph appeared to do nothing.** The navigation graph is inside
   the building, so solid walls and a roof hid all of it; the top view showed
   blue only where floor slabs overhang the walls. Added a Solid / Ghost /
   Hidden control for the IFC geometry, and the first graph build switches to
   Ghost automatically.
2. **The egress route was a one-pixel thread.** `LineBasicMaterial.linewidth`
   is ignored by every major browser, so the thing the app exists to show was
   lost among 17,000 graph edges. Routes are now tubes with real thickness,
   scaled to the model, and the graph fades while a route is displayed.
3. **Picks landed on walls and roofs.** A ray into the model hits whatever
   surface faces the camera. Routing snapped to the nearest node anyway, so the
   marker showed a point the walker could never stand on. Picks now snap to the
   graph, so the marker shows where the route will actually begin.
4. **Exit and fire origin were near-identical oranges.** The fire origin is now
   a spiked octahedron rather than a sphere, so the two differ in shape as well
   as hue.

## Not done

- The browser test covers one model on one machine. It is a smoke test, not a
  cross-browser or cross-model suite, and it is not wired into CI.
- `/upload-ifc` and `/upload-topology` are carried over verbatim and unprofiled.
- The `thatopen` chunk is still 5.4 MB; lazy-loaded but not slimmed.
- No CI wiring for the new test suites.
