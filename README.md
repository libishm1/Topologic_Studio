# Topologic Studio — Next

The modernisation line of Topologic Studio. IFC viewer, egress navigation
graph, fire simulation and hazard-aware rerouting.

This lives on branch **`next/studio-2`** in a git worktree beside the Classic
checkout. Classic (`claude/ifc-port-baseline`, tagged `demo-safe-2026-08-24`)
is untouched and still runs from its own folder.

```
topologic_webapp/
├── TopologicStudio/        # Classic - demo-safe, unchanged
└── TopologicStudio-Next/   # this line, branch next/studio-2
```

**Picking this up cold?** Start with
[wiki/roadmap/handoff-2026-08-24.md](wiki/roadmap/handoff-2026-08-24.md).

---

## What changed and why

Classic worked, but the IFC workflow re-did the same expensive work several
times per session. The headline changes are architectural, not tuning.

### 1. Fragment-first loading with a local cache

That Open's own guidance is that parsing IFC at runtime is too slow for
production and that models should be converted to fragments once and reused.

Next hashes the file (SHA-256), converts it to fragments on first sight, and
stores the `.frag` buffer in IndexedDB. Re-opening the same model skips IFC
parsing completely. The cache is LRU-bounded (512 MB / 24 models) and can be
cleared from the Model panel.

### 2. The second IFC parse is gone

Classic called `ifcLoader.readIfcFile(bytes)` *after* the model was already on
screen, purely to enumerate `IFCSLAB` / `IFCSTAIR` / `IFCDOOR` / `IFCWALL`
express IDs — a full second parse of the same file, every session.

Next reads the same categories from the fragments already in memory via
`FragmentsModel.getItemsOfCategories()`.

### 3. Sampling moved into the browser

Classic serialised every floor, stair, door and wall triangle to JSON, posted
it to Python, and Python resampled it down to a few thousand points.

Next samples walkable points in a Web Worker — the browser already holds the
geometry to draw it — and uploads only the point cloud. On the Duplex model
the worker turns 7,036 triangles into 2,637 points in **5 ms**.

### 4. Session-safe graph storage

Classic kept one module-global `LAST_GRAPHS` dict, so two browser tabs (or two
users) silently overwrote each other. Next returns a `graph_id` per build and
stores graphs in an LRU + TTL store.

### 5. One pathfinding layer, two engines

Classic used a hand-rolled Dijkstra for the normal route and TopologicPy for
hazard rerouting — two stacks, no parity check. Next has one interface:

| Engine | Use | Measured (2,651-node graph) |
| --- | --- | --- |
| `fast` | default; A* over CSR with a Euclidean heuristic | **14 ms** per query |
| `topologicpy` | `Graph.ShortestPath`, for TopologicPy semantics | 1.44 s warm, 56.7 s cold |

Both return **identical** routes and costs (verified in tests and by
`POST /api/ifc/path/compare`). The fast engine is the default because
TopologicPy is ~100x slower per query at this scale.

Hazard reweighting no longer rebuilds the TopologicPy graph. 0.9 added
`edgeCostFunc` / `edgeFilter`, so the graph is built once, cached, and
reweighted per query.

### 6. Frontend runtime hardening

- `optimizeDeps.exclude: ["web-ifc", "@thatopen/fragments"]`
- `React.StrictMode` removed around the viewer bootstrap
- web-ifc WASM **and** the fragments worker self-hosted from `public/`
  (`FragmentsManager.getWorker()` otherwise fetches the worker from unpkg.com
  on every boot)
- no COOP/COEP headers: cross-origin isolation makes web-ifc pick its pthread
  build, which spawns workers with an undefined URL and breaks IFC loading
- the engine is a lazy chunk, so the shell paints before ~6 MB of Three + That Open arrives
- an imperative `ViewerManager` owns the 3D lifecycle; React never re-creates the WebGL context

---

## Measured results

Duplex model (`Ifc2x3_Duplex_Architecture.ifc`, 7,036 triangles), same machine,
same input payload.

| | Classic | Next |
| --- | --- | --- |
| Graph build | 380 ms | 225 ms |
| Graph size | 1,364 nodes / 23,388 edges | 2,651 nodes / 17,677 edges |
| Connected components | fragmented | **1** |
| Path query | 476 ms, **no path found** | **14 ms**, found |
| Browser-side sampling | n/a (server-side) | 5 ms |

Classic could not find a route on its own graph for this model. Next produces a
denser, fully connected graph in less time and routes on it 30x faster.

Reproduce with:

```bash
.venv-next/Scripts/python  tools/bench_egress.py ../Ifc2x3_Duplex_Architecture.ifc --cache /tmp/duplex.json
node tools/smoke-pipeline.mjs /tmp/duplex.json
```

---

## Dependency versions

| Package | Classic | Next |
| --- | --- | --- |
| `@thatopen/components` | 3.2.6 | 3.4.8 |
| `@thatopen/fragments` | 3.2.13 | 3.4.7 |
| `web-ifc` | 0.0.73 | 0.0.77 |
| `three` | 0.181.2 | 0.185.1 |
| `topologicpy` | 0.8.93 | 0.9.64 |
| `topologic-core` | (bundled) | **8.0.4, now separate** |
| `ifcopenshell` | Dockerfile only | 0.8.5, in `requirements.txt` |

> **Packaging trap.** TopologicPy 0.9 split its native backend into a separate
> `topologic-core` distribution. Installing `topologicpy` alone imports fine and
> then fails at the first geometry call with `Could not import topologic_core`.
> Both are pinned in `requirements.txt`, and the Docker build fails fast if the
> backend is unusable.

> **Licence.** TopologicPy is published as **LGPL-3.0-or-later** (PyPI
> classifier `GNU Lesser General Public License v3 or later`), not AGPL. Server-
> side use does not trigger a network-copyleft obligation.

---

## Running locally

### Quickest

```
studio          start backend + frontend and open the browser
studio stop     stop both
studio status   show what is running
studio test     run the browser suite against the running app
```

`studio.cmd` lives at the repo root; a one-line shim in
`%LOCALAPPDATA%\Microsoft\WindowsApps` (already on PATH) makes `studio`
work from any terminal. Each service gets its own window.

### Backend

```bash
cd TopologicStudio-Next
python -m venv .venv-next
.venv-next/Scripts/python -m pip install -r topologicpy-web-backend/requirements-dev.txt

cd topologicpy-web-backend
../.venv-next/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

Check it came up cleanly — `usable` must be `true`:

```bash
curl http://localhost:8000/api/capabilities
```

### Frontend

```bash
cd TopologicStudio-Next/topologicpy-web-frontend
npm install          # postinstall copies the WASM and fragments worker into public/
npm run dev
```

Copy `.env.example` to `.env` if the backend is not on `localhost:8000`.

### Tests

```bash
# backend: 73 tests
cd topologicpy-web-backend && ../.venv-next/Scripts/python -m pytest tests -q

# frontend
cd topologicpy-web-frontend && npm run lint && npm run build

# end-to-end against a running backend
node tools/smoke-pipeline.mjs /tmp/duplex.json

# real Chrome, driving the actual UI (needs both servers running)
npm install            # once, at the repo root: installs playwright-core
npm run test:browser   # add --headed to watch it
```

`tools/browser-test.mjs` uses the Chrome already installed on the machine, so
there is no browser download. It fails on any console error, uncaught
exception or failed request, and writes screenshots to `tools/shots/`.

---

## Architecture

```
IFC file
  │
  ├─ SHA-256 hash ────────────► IndexedDB fragment cache      lib/fragmentCache.js
  │                                    │ hit → skip parsing
  ├─ IfcLoader.load ◄──────── miss ────┘                      viewer/ViewerManager.js
  │
  ├─ getItemsOfCategories() ─► floor / stair / door / wall ids  viewer/categories.js
  ├─ getItemsGeometry()     ─► transferable typed arrays
  │
  ├─ Web Worker ────────────► walkable point cloud + wall lines workers/sampler.worker.js
  │
  └─ POST /api/ifc/graph ───► NavGraph (numpy + KD-tree)      services/graph_builder.py
                                   │
                                   ├─ POST /api/ifc/path      graphs/pathfinding.py
                                   ├─ GET  /api/fire/stream   graphs/fire.py
                                   └─ POST /api/rl/train      graphs/rl.py
```

### Backend modules

`main.py` was 2,838 lines with a hard-coded `sys.path.append` to a developer's
home directory. It is now an app factory:

```
app/
├── main.py                    app factory, /health, /api/capabilities
├── config.py                  env-driven settings
├── models.py                  request/response schemas
├── store.py                   NavGraph + LRU/TTL GraphStore
├── perf.py                    timing markers
├── geometry/
│   ├── common.py              up-axis handling
│   ├── sampling.py            vectorised walkable sampling
│   ├── adjacency.py           KD-tree adjacency, grid snapping
│   └── obstacles.py           walls, doors, edge blocking
├── graphs/
│   ├── pathfinding.py         A*, hazard weights, components
│   ├── topologic_engine.py    Graph.ByMeshData + Graph.ShortestPath
│   ├── fire.py                radial / flood / temperature
│   └── rl.py                  tabular Q-learning
├── routers/
│   ├── ifc.py                 /api/ifc/* and Classic /ifc-egress-*
│   └── simulation.py          /api/fire/*, /api/rl/*
└── legacy/
    ├── contract.py            /upload-topology, /upload-ifc (moved verbatim)
    └── helpers.py             supporting helpers + store adapter
```

---

## API

### Next

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/ifc/graph` | Build a graph from a sampled point cloud |
| `GET` | `/api/ifc/graph/{id}` | Re-fetch a built graph |
| `DELETE` | `/api/ifc/graph/{id}` | Drop a graph |
| `GET` | `/api/graphs` | Store contents |
| `POST` | `/api/ifc/path` | Route, optionally hazard-weighted |
| `POST` | `/api/ifc/path/compare` | Run both engines and report agreement |
| `POST` | `/api/fire/timeline` | Precomputed ignition timeline |
| `GET` | `/api/fire/stream` | SSE: fire front, temperatures, live rerouting |
| `POST` | `/api/rl/train` | Q-learning policy |
| `GET` | `/api/capabilities` | What this server can actually do |

Graph node and edge arrays are returned base64-encoded (`float32` / `uint32`)
rather than as JSON numbers — roughly a sixth of the bytes and no per-number
parse on the main thread.

### Classic compatibility

`/ifc-egress-graph`, `/ifc-egress-path`, `/graph-meta`, `/upload-topology` and
`/upload-ifc` keep their original request and response shapes, served by the
new machinery. The Classic frontend can point at this backend unchanged, which
is what makes an A/B comparison on one machine possible.

---

## Verified in a browser

`npm run test:browser` drives Chrome through the full workflow (load, build,
pick, route, simulate) and passes with no console errors, uncaught exceptions
or failed requests. See
[wiki/roadmap/next-line-status-2026-08-23.md](wiki/roadmap/next-line-status-2026-08-23.md)
for the bugs that pass turned up.

## Known limitations

- `Graph.ShortestPath` does not scale to graphs of this size; the fast engine is
  the default and TopologicPy is opt-in with a warning in the UI.
- The `thatopen` bundle chunk is 5.4 MB (911 KB gzipped). It is lazy-loaded, but
  it is still a large first-visit download.
- Wall blocking uses 2D centrelines with a vertical extent, not true solids.
  Curved and non-orthogonal walls are approximated by their bounding centreline.
- `/upload-ifc` (server-side IFC) is carried over unchanged and has not been
  re-profiled.
- Grid-snap mode re-issues node indices, so a graph built with it cannot be
  compared node-for-node against one built without.
- The browser test covers one model on one machine. It is a smoke test, not a
  cross-browser or cross-model suite, and it is not wired into CI.
- `tools/bench_egress.py` counts walls twice on IFC2X3, because
  `by_type("IfcWall")` already includes `IfcWallStandardCase`. That inflates
  the wall count in the benchmark payload only; both backends receive the same
  input, so the comparison stands.

## Merging back

Nothing here touches Classic. When the time comes:

```bash
cd TopologicStudio
git merge next/studio-2          # or open a PR from the branch
```

The two frontends are separate directories in the same tree, so a merge is a
file-level addition rather than a rewrite. The backend `app/` package is
replaced wholesale — review `legacy/contract.py` first, since it is the only
part carried over verbatim.
