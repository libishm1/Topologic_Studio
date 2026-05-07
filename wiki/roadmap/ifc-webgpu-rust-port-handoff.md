# Claude Code Handoff: IFC Loader Modernization, Rust/WASM, And WebGPU

This handoff plans a staged modernization of Topologic Studio's IFC workflow. The user wants the IFC loader and rendering path made faster, updated to the latest library methods, and eventually moved toward Rust + WebAssembly + WebGPU where that actually helps performance. A separate concern — that the deployed backend is slow — is addressed in Phase 8.

This revision (2026-05-07) reflects a full repo audit and the fixes from the prior critique pass. It corrects the deployment platform (the backend runs on **Render**, not Railway), pins down the dependency table, gives Phases 5–7 numeric gates, and replaces the oversized "first PR" with a sequenced milestone plan.

## Core Clarification

Do not treat WebGPU as an IFC parser replacement. WebGPU can help with rendering and GPU-side compute, but IFC parsing and semantic extraction still need a loader such as `web-ifc`, That Open Components/Fragments, `topologicpy.IFC`, or `IfcOpenShell`.

The recommended strategy is:

1. Benchmark the current slow path.
2. Upgrade the current IFC libraries safely.
3. Pin or self-host WASM assets.
4. Cache parsed/fragmented IFC assets and introduce an IFC Lite contract.
5. Evaluate the new TopologicPy 0.9.x IFC module.
6. Port graph/path/fire compute to Rust/WASM.
7. Prototype WebGPU for graph/fire/path overlays first.
8. Decide on a full WebGPU IFC viewer rewrite only if profiling proves rendering is the bottleneck.
9. Decide on backend deployment migration based on measured request latency, cold-start, and image size.

## Repo Context

Working repository:

```text
c:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio
```

Key files (verified to exist as of 2026-05-07):

- `topologicpy-web-frontend/package.json`
- `topologicpy-web-frontend/package-lock.json`
- `topologicpy-web-frontend/src/IFCViewer.jsx`
- `topologicpy-web-frontend/src/App.jsx`
- `topologicpy-web-frontend/src/TopologyViewer.jsx`
- `topologicpy-web-backend/requirements.txt`
- `topologicpy-web-backend/app/main.py` (2,791 lines, single module — refactor target)
- `topologicpy-web-backend/app/schemas.py`
- `topologicpy-web-backend/app/utils.py`
- `topologicpy-web-backend/Dockerfile`
- `render.yaml` (root)
- `.github/workflows/deploy-frontend.yml`
- `wiki/roadmap/ifc-lite-profile.md`
- `wiki/sources/dependencies.md`
- `wiki/frontend/ifc-viewer.md`
- `wiki/backend/ifc-egress-graph.md`
- `wiki/deployment/render-backend.md`
- `wiki/deployment/vite-github-pages.md`

The worktree may contain uncommitted changes and generated local artifacts (`frontend-dev.log`, `uvicorn-out.log`, the bundled `node-v24.11.1-win-x64/` runtime, the backend `.venv/`). Do not reset or revert unrelated changes.

`main.py:11` hardcodes a `sys.path.append("C:/Users/sarwj/...")` to a TopologicPy checkout from a different developer's machine. This line is dead on the current machine and in Docker (the path doesn't exist there) but should be removed in the Phase 1 cleanup so it doesn't mask import errors after the TopologicPy upgrade.

## Current Architecture

### Browser IFC Path

`IFCViewer.jsx` is the main IFC browser pipeline:

- Imports IFC type constants from `web-ifc`.
- Initializes That Open `Components`, `Worlds`, `SimpleScene`/`SimpleCamera`/`SimpleRenderer`, `FragmentsManager`, `IfcLoader`, `Raycasters`.
- Uses a fragments worker:

```js
new URL("@thatopen/fragments/dist/Worker/worker.mjs", import.meta.url)
```

- Sets the `web-ifc` WASM path through `ifcLoader.setup` with `autoSetWasm: false`.
- Already applies the documented `web-ifc` performance flags: `COORDINATE_TO_ORIGIN = false`, `USE_FAST_BOOLS = true`, `OPTIMIZE_PROFILES = true`. **These are not future work.**
- Loads IFC buffers with `ifcLoader.load(new Uint8Array(buffer), false, file.name)`.
- Extracts IFC IDs for slabs (`IFCSLAB`, `IFCSLABSTANDARDCASE`, `IFCSLABELEMENTEDCASE`), stairs (`IFCSTAIR`, `IFCSTAIRFLIGHT`), coverings (`IFCCOVERING`), doors (`IFCDOOR`), spaces (`IFCSPACE`), walls (`IFCWALL`), storeys (`IFCBUILDINGSTOREY`).
- Calls `model.getItemsGeometry(localIds)` for floors, stairs, doors, and walls (each category is a separate sequential pass with a `requestAnimationFrame + setTimeout` yield between them — this is already there).
- Renders graph wire overlays, static path (red), dynamic hazard path (magenta), and per-vertex temperature gradient colors.

### Browser Orchestration

`App.jsx` posts extracted geometry and runs simulations against:

- `POST /upload-topology` (TopologicPy JSON)
- `POST /upload-ifc` (server-side IFC ingest, multipart)
- `POST /ifc-egress-graph` (build graph from extracted geometry)
- `POST /ifc-egress-path` (Dijkstra over last graph)
- `POST /fire-sim` (precomputed timeline)
- `GET /fire-sim/stream` (SSE via `EventSource`, supports temperature mode and dynamic path rerouting)
- `POST /rl/train` (Q-learning on graph)
- `GET /graph-meta` (cell bbox metadata)

`API_BASE` resolves to `import.meta.env.VITE_API_BASE || "http://localhost:8000"`. The deployed value is `https://topologicstudio-backend.onrender.com`.

Dynamic path rerouting is **already implemented** end-to-end: the SSE stream accepts `stream_path`, `path_alpha`, `path_recompute_interval`, `path_lethality_threshold` query params, and the frontend renders the magenta dynamic line with cost and "path changed" indicator. Treat this as a feature to preserve through the migration, not a feature to build.

### Backend IFC And Graph Path

`topologicpy-web-backend/app/main.py` is a single 2,791-line FastAPI module. It contains:

- TopologicPy imports (`Vertex`, `Edge`, `Wire`, `Face`, `Shell`, `Cluster`, `Topology`, `Graph`, `Dictionary`, `Color`).
- Point sampling for walkable IFC geometry (`_sample_walkable_points`).
- Distance-based hybrid adjacency (`_build_point_adjacency_hybrid`) and rectilinear grid-snap adjacency (`_build_point_adjacency_rectilinear`).
- Wall segment extraction in 2D (`_extract_wall_segments_2d`) and segment-vs-segment intersection pruning (`_prune_edges_through_walls`).
- Wall-aware Dijkstra (`_shortest_path_ids`).
- Radial and graph-flood fire spread (`_compute_radial_timeline`, `_compute_fire_timeline`).
- Temperature-based fire spread (`_compute_temperature_fire_spread`).
- Hazard-weighted dynamic path recomputation (`_recompute_path_with_hazards`) using TopologicPy `Graph.ShortestPath`.
- Q-learning RL path (`_q_learning_path`).
- Server-side `/upload-ifc` conversion using `ifcopenshell` when available.
- A module-global `LAST_GRAPHS = {"wire", "cell", "ifc"}` shared across requests — **not session-safe**.

The browser IFC egress path and server `/upload-ifc` path are different. Do not assume they produce the same graph state.

## Current Dependency State

Verified 2026-05-07 against `package.json`, `requirements.txt`, the backend `.venv`, the npm registry, and PyPI.

### Frontend (`topologicpy-web-frontend/package.json`)

| Package | Pinned | Latest | Notes |
|---|---:|---:|---|
| `web-ifc` | `0.0.73` | `0.0.77` | IFC WASM parser. `IFCViewer.jsx:27` hardcodes a CDN fallback to `web-ifc@0.0.73`. |
| `@thatopen/components` | `3.2.6` | `3.4.5` | IFC loader and world abstractions. |
| `@thatopen/fragments` | `3.2.13` | `3.4.5` | Fragment worker and model geometry. |
| `three` | `0.181.2` | `0.184.0` | Rendering and overlay geometry. WebGPU renderer is in this lineage. |
| `react` | `19.2.0` | reverify before upgrade | UI framework. |
| `react-dom` | `19.2.0` | reverify before upgrade | DOM mount. |
| `vite` | `7.2.4` | reverify before upgrade | Build/dev server. |
| `axios` | `1.13.2` | reverify before upgrade | HTTP client. |

### Backend (`topologicpy-web-backend/requirements.txt`)

| Package | Local venv | Latest | Notes |
|---|---:|---:|---|
| `topologicpy` | `0.8.93` | `0.9.26` | **Major bump.** 0.9.x exposes a `topologicpy.IFC` module (`IFC.MeshDataByPath`, `IFC.TopologiesByPath`, experimental `IFCFastTopology`). Not present in 0.8.93. |
| `ifcopenshell` | `0.8.4` (manual) | reverify | Installed locally and in Docker, **not declared in `requirements.txt`**. The Dockerfile installs it via a separate `pip install` step. |
| `fastapi` | `0.124.0` | reverify | Local venv. |
| `pydantic` | `2.12.5` | reverify | Local venv. |
| `uvicorn` | `0.38.0` | reverify | Local venv. |

The `python-multipart` dependency is declared but its installed version was not checked.

The version check commands are documented at the bottom of `wiki/sources/dependencies.md`. Re-run them on the day a Phase 1 upgrade PR is opened — npm/PyPI publish frequently and the table above will drift.

## Performance Hypotheses

Do not assume WebGPU is the first fix. The likely slow sections are:

- IFC parsing from STEP text into WASM structures.
- That Open fragment creation.
- `getItemsGeometry` over many elements (sequential, per-category).
- Sending large JSON geometry payloads to FastAPI (no compression today).
- Python `_sample_walkable_points` and `_build_point_adjacency_*`.
- Wall pruning (`_prune_edges_through_walls` is O(edges × walls)).
- Rendering many graph line segments on top of the IFC model.
- Render free-tier cold start when the backend has been idle (handled in Phase 8).

The first implementation task is measurement, not optimization.

## Phase 0: Add Baseline Instrumentation

Add explicit timings without changing behavior. After this phase, **rebaseline at the end of every later phase** (Phase 1 upgrades, Phase 3 caching, Phase 5 Rust port, Phase 8 deployment move) so improvements are measurable, not assumed.

### Frontend timings (`IFCViewer.jsx`)

Wrap the IFC load and egress extraction with `performance.now()`. Emit a single `console.table` of:

- File `arrayBuffer` read time.
- `ifcLoader.load` time.
- Time to first frame visible (after `fragments.core.update(true)` and `fitCameraToModel`).
- IFC ID collection time (`collectIfcIds`).
- `getItemsGeometry` time per category (floors, stairs, doors, walls) — these already run sequentially with frame yields, so per-category timing is meaningful.
- Payload size in bytes per category (sum of `vertices` + `indices` + `normals` Float32/Int32 lengths × element size).

### Frontend timings (`App.jsx`)

Wrap each axios call:

- `/ifc-egress-graph` request duration (build).
- `/ifc-egress-path` request duration (route).
- `/upload-ifc` request duration (server-side ingest).
- Fire stream startup time (interval from `new EventSource(url)` to first non-`meta` event).

### Backend timings (`main.py`)

Inside `/ifc-egress-graph`, log:

- Sampling time (sum of all `_sample_walkable_points` calls).
- Wall extraction time (`_extract_wall_segments_2d`).
- Adjacency build time (`_build_point_adjacency_*`).
- Door connection loop time.
- Returned `nodes` and `edges` count.

Inside `/ifc-egress-path`, log:

- `_resolve_start_id` time for start and end.
- `_shortest_path_ids` time.
- Resulting `points` length.

Use `time.perf_counter()`. Print one structured line per request to stdout (Render captures this in the service log) using a stable prefix like `IFC_TIMING` so it greps cleanly.

### Acceptance criteria

- A single IFC load produces a compact timing report in the dev console and the backend log.
- No UI behavior changes. All overlays (graph wires, static path, dynamic path, fire colors) still render.
- The combined report identifies whether parsing, extraction, payload transport, graph build, route compute, or rendering dominates.
- The instrumentation is left in place behind a `VITE_PERF_LOG` / `PERF_LOG` env flag (default on in dev, off in prod) so later phases can rebaseline cheaply.

## Phase 1: Upgrade Existing Libraries Safely

Use a branch. Upgrade one dependency group at a time. Re-run the version check commands in `wiki/sources/dependencies.md` first — the table above can drift between merge and ship.

Frontend upgrade order (one PR per row, smallest blast radius first):

1. `web-ifc` 0.0.73 → 0.0.77 (also covers Phase 2 since the WASM file ships with this).
2. `@thatopen/fragments` 3.2.13 → 3.4.5.
3. `@thatopen/components` 3.2.6 → 3.4.5 (often paired with fragments — bundle if the API surface changed).
4. `three` 0.181.2 → 0.184.0.
5. `axios` / `react` / `vite` only if the version check shows a meaningful gap.

Backend upgrade:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade topologicpy ifcopenshell
.\.venv\Scripts\python.exe -m pip show topologicpy ifcopenshell
```

Also update `requirements.txt`:

- Add `ifcopenshell` (so local and Docker match).
- Add a soft pin for `topologicpy` (e.g., `topologicpy>=0.9.26,<0.10`).
- Optionally pin the rest to the locally verified versions to make Docker builds reproducible.

### Rollback plan (per PR)

Each upgrade PR must include:

- A `git revert` recipe in the PR description (the merge commit hash and the command).
- A note of which fixture IFC was used and the Phase 0 timing delta vs. the previous baseline.
- The exact `package.json` / `requirements.txt` line changed, so reverting is one-line if the change introduced a regression.

If a That Open API rename breaks `IFCViewer.jsx` and the fix is non-trivial, **revert and open an issue** rather than carrying a half-migrated branch — that's the failure mode the prior critique flagged.

### Acceptance criteria

- Frontend build succeeds (`npm run build`).
- IFC viewer loads the sample IFC.
- `/ifc-egress-graph` returns nonzero `nodes` and `edges`.
- `/ifc-egress-path` returns a route between two reachable points.
- Dynamic path rerouting still emits magenta updates over SSE when temperature mode is enabled.
- Browser console has no fatal That Open or web-ifc API errors.
- Phase 0 timing report shows no regression > 20% on any line for the same fixture IFC (and ideally improvements).

## Phase 2: Pin Or Self-Host WASM Assets

`IFCViewer.jsx:27` falls back to `https://unpkg.com/web-ifc@0.0.73/`. This is slow on restricted networks and pins the runtime to an old version regardless of what `package.json` says.

Tasks:

- Copy `web-ifc.wasm` and `web-ifc-mt.wasm` (and any companion `.js` glue) for the upgraded `web-ifc` version into `topologicpy-web-frontend/public/wasm/`.
- Set `VITE_WEBIFC_WASM_PATH=/wasm/` in the Vite env defaults and in the GitHub Actions workflow so production picks it up.
- Replace `DEFAULT_WASM_PATH` with a build-time-derived constant, or remove the fallback entirely and fail loudly if the env is missing in dev.
- Document the expected WASM asset names and the upgrade procedure in `wiki/frontend/ifc-viewer.md`.

### Acceptance criteria

- IFC loading works offline once the app bundle is available.
- The bundled WASM hash matches the version pinned in `package.json` (verify with `sha256sum public/wasm/web-ifc.wasm` vs. the npm tarball's WASM file).
- `npm run build` fails the build if the WASM file is missing from `public/wasm/`.

## Phase 3: Cache Fragments And Introduce IFC Lite

This is likely a bigger speed win than WebGPU if repeated parsing is the pain.

Use the existing `wiki/roadmap/ifc-lite-profile.md` as the contract starting point.

Tasks:

- Make `IFCViewer.jsx` emit an IFC Lite payload before posting to the backend.
- Include `units`, `up_axis`, `source_schema` (e.g., `IFC4`), `transform`, floors, stairs, doors, walls, levels, and optional bounding boxes.
- Add backend Pydantic models for the IFC Lite contract in `app/schemas.py`.
- Make `/ifc-egress-graph` accept both the current payload and the new profile (parallel acceptance during migration; remove the legacy path once the frontend is updated).
- Cache extracted lite profiles by file SHA-256 in memory during a browser session (and by file hash in the backend if `LAST_GRAPHS` is replaced — see Phase 5 / Phase 8 risks).
- Investigate That Open fragment export/import (`fragments` 3.4.x has serialization helpers) for persistent cache, possibly via IndexedDB.

### Acceptance criteria

- Reloading or rebuilding the graph for the same IFC completes the second `getItemsGeometry` pass in **< 25% of the first-load time** per Phase 0 timings (IndexedDB or in-memory cache hit).
- The IFC Lite payload is at least 30% smaller than the current raw geometry arrays for the same fixture, or carries strictly more structure (typed elements, no opaque float arrays).
- Backend validation errors are explicit when floors/stairs/doors/walls are missing — no silent empty graphs.

## Phase 4: Evaluate Latest TopologicPy IFC Loader

After upgrading TopologicPy to 0.9.x, test the new IFC helpers in isolation before wiring them into production routes.

Prototype script target:

```text
topologicpy-web-backend/scripts/probe_topologicpy_ifc.py
```

Questions to answer:

- Does `from topologicpy.IFC import IFC` import successfully?
- Does `IFC.MeshDataByPath(...)` return usable mesh data for the sample IFC?
- Does `IFC.TopologiesByPath(...)` produce topologies useful for egress extraction?
- How does `IFCFastTopology` (experimental) compare on the same fixture?
- Is it faster or slower than the current browser-side `web-ifc` extraction?
- Does it preserve door / wall / stair semantics well enough for egress?
- AGPL exposure: does using `topologicpy.IFC` for an inbound user file place network-service obligations on the deployment? See the AGPL note at the bottom of this doc.

### Acceptance criteria

- A short benchmark compares browser `web-ifc` extraction, `ifcopenshell`, and `topologicpy.IFC` on the same sample IFC, with timings logged via the Phase 0 instrumentation.
- The production route is **not** switched until correctness and speed are clear and the AGPL implication is reviewed.

## Phase 5: Port Graph/Path/Fire Compute To Rust/WASM

This is the best first Rust target. Avoid porting TopologicCore or the IFC parser first.

Candidate Rust crate layout:

```text
topologicpy-web-frontend/
  wasm-egress/
    Cargo.toml
    src/lib.rs
```

Rust/WASM functions to expose via `wasm-bindgen`:

- `build_graph(ifc_lite_payload_or_typed_arrays, params) -> graph`
- `shortest_path(graph, start_point, end_point, wall_settings) -> path`
- `fire_timeline(graph, start_id, params) -> timeline`
- `hazard_path(graph, temperatures, start_id, end_id, alpha, threshold) -> path`

Implementation notes:

- Prefer typed arrays or compact binary payloads (e.g., FlatBuffers, postcard, or raw `Float32Array`/`Uint32Array`) once the IFC Lite contract is stable. Avoid sending huge JSON across the wasm boundary.
- Keep the current Python backend route as the **oracle** during migration — every Rust call is double-checked against Python output behind a feature flag.
- Add deterministic test fixtures comparing Python and Rust outputs on the same input.
- Move the temperature-to-color mapping (currently in `IFCViewer.jsx`) into the same crate if it ends up running per-step on large vertex counts.

### Acceptance criteria

- Rust graph builder returns the same node count and ≥ 95% edge overlap with Python on the documented fixture (wall pruning may legitimately differ at the boundary, hence the tolerance).
- Rust route matches Python route within a documented length tolerance (e.g., total path length within 5% and same start / end nodes).
- Rust implementation hits the documented improvement target on the largest fixture: graph build **≥ 3× faster** OR route compute **≥ 5× faster**, measured against the Phase 0 baseline. (The 2× gate from the prior draft is dropped — Dijkstra is already fast enough that 2× isn't worth a Rust port; hold the bar higher to justify the migration cost. If only sampling+adjacency clears the bar, port only that and leave Dijkstra in Python.)
- Memory: Rust peak heap on the same fixture is no more than 2× Python's peak. A "3× faster but 10× memory" port is rejected.
- Existing UI can switch between Python and WASM graph engines behind a feature flag (`VITE_USE_WASM_EGRESS`).

## Phase 6: WebGPU Prototype

Start with overlays, not the full IFC model.

Targets:

- Graph line segments (currently `THREE.LineSegments` with vertex colors).
- Fire temperature gradient.
- Static route line.
- Dynamic hazard route line.

Decision criteria for choosing a tier (use Phase 0 / Phase 5 telemetry):

- **Stay on Three/WebGL (no WebGPU)** if the largest fixture renders graph + path + fire overlays at ≥ 30 FPS during a fire stream. WebGPU is unwarranted.
- **Lower risk: Three.js WebGPU renderer for overlays only** if FPS drops below 30 with > 50k edge segments on the dominant target machine.
- **Medium risk: small custom WebGPU layer for overlays** if Three's WebGPU path doesn't meaningfully help (frame time still > 33 ms with overlays).
- **Higher risk: Rust `wgpu` + WASM overlay renderer** only if the medium-risk path doesn't close the gap.
- **Highest risk: replace That Open / Three IFC rendering entirely** is Phase 7, not 6.

### Acceptance criteria

- Overlay rendering stays interactive (≥ 30 FPS sustained) on the largest fixture during a fire stream with dynamic path rerouting active.
- No regression in start / exit point picking (raycaster behavior preserved).
- A WebGPU-unavailable fallback still works through the current Three / WebGL path.

## Phase 7: Decide On Full WebGPU IFC Viewer

Only start this if benchmark data shows rendering the IFC model is the **dominant** bottleneck after fragment caching (Phase 3) and Rust compute migration (Phase 5).

"Dominant" is defined numerically:

- IFC model rendering frame time accounts for **> 50% of total p95 frame time** during interactive orbit on the largest fixture, AND
- Overlay rendering (Phase 6) is already optimized to its tier's gate.

If the dominant cost is parsing, extraction, network, or compute, this phase does not start.

Full replacement risks:

- That Open Components are tied to Three / fragments abstractions; replacing the renderer means rebuilding the loader integration.
- IFC parsing remains separate from WebGPU.
- Reimplementing camera, picking, transforms, materials, disposal, model fitting is a large project.
- WebGPU support requires browser feature checks and a fallback path.

Estimated effort (assumes one engineer familiar with Rust/WASM and Three; halve or double accordingly):

- Phase 1 library upgrades and Phase 2 WASM hosting: 1–2 weeks.
- Phase 3 IFC Lite and cache: 1–2 weeks.
- Phase 4 TopologicPy probe: 2–4 days.
- Phase 5 Rust/WASM compute engine: 3–6 weeks.
- Phase 6 WebGPU overlay prototype (lowest tier): 1–2 weeks.
- Phase 6 medium-tier WebGPU layer: 3–6 weeks.
- Phase 7 full custom WebGPU IFC viewer: 2–4+ months.
- Phase 8 deployment migration (any single target): 2–5 days.

## Phase 8: Backend Deployment Migration

The user reported the deployed backend is too slow. The deployment **today is Render**, not Railway, per `render.yaml` and `wiki/deployment/render-backend.md` and the GitHub Actions `VITE_API_BASE` value `https://topologicstudio-backend.onrender.com`. This phase distinguishes the cheap fix from the platform move.

### Likely root cause: Render free-tier cold start

If the backend is on Render's free tier (the default for `render.yaml` without a `plan:` field), the service spins down after 15 minutes of inactivity. The next request triggers a cold start that for this image is plausibly 30–60+ seconds because:

- The Docker image is `python:3.11-slim` plus `topologicpy` plus `ifcopenshell` (both pull large native dependencies — ifcopenshell ships precompiled Open CASCADE wheels). Image is likely 400–800 MB.
- `ifcopenshell` and `topologicpy` are reinstalled on every build because they're outside the `requirements.txt` cache layer in the current Dockerfile.
- Uvicorn is started without preloading TopologicPy modules.

Verify cold-start vs. warm-start using Phase 0 instrumentation: a request after >15 min idle should show clearly higher first-request latency.

### Cheapest fix: stay on Render, upgrade the plan and the Dockerfile

Tasks:

- Add `plan: starter` (or higher) to `render.yaml`. Starter is currently $7/mo and removes the spin-down. This alone likely resolves the perceived slowness.
- Move `pip install ifcopenshell topologicpy` into `requirements.txt` so it sits inside the cached layer, not outside it. This shortens redeploys.
- Add a `--workers 2` (or autoscaling) flag to the uvicorn command if the instance has the RAM headroom — TopologicPy operations release the GIL inconsistently and a single worker bottlenecks SSE plus a parallel `/ifc-egress-graph`.
- Add a small warm-up route or do `from topologicpy.Graph import Graph` in module scope (already done) so the first request doesn't pay import cost.

This is the recommended **first** action. It's reversible, single-PR, and doesn't fork the deployment story.

### When to actually move off Render

Move only if any of these hold after the cheap fix:

- p50 request latency for `/ifc-egress-graph` on a representative fixture is still > 5 seconds with a warm instance.
- Memory pressure causes OOM kills on the largest realistic IFC.
- You need a region closer to Saudi Arabia / the Gulf (Render runs Oregon/Frankfurt/Singapore — Frankfurt is the closest; Fly.io has Bahrain/Dubai-adjacent regions).

### Platform comparison

Hard 'no' for the backend: **Vercel**. Vercel Functions are serverless with a 10-second default and 300-second hard cap, ~250 MB unzipped bundle limit, no persistent process state, and no SSE streaming guarantee. This backend has:

- `GET /fire-sim/stream` (SSE, holds the connection open across many fire steps).
- A module-global `LAST_GRAPHS` dict shared across requests (`/ifc-egress-graph` writes it, `/ifc-egress-path` and `/fire-sim` read it).
- `ifcopenshell` (~100 MB+ unpacked with Open CASCADE).
- Long compute paths for `/rl/train`.

Each of these is a hard incompatibility with Vercel's model. Vercel **is** fine for the frontend if you ever want to leave GitHub Pages — preview URLs and edge caching are real wins — but that's a separate decision and not the slowness the user is reporting.

Realistic candidates:

| Platform | Model | Cold start | Native deps | Streaming | Estimated $ | Best for |
|---|---|---|---|---|---|---|
| Render Starter | Persistent container | none on paid | yes (Docker) | SSE works | $7/mo | Cheapest fix, no migration |
| Fly.io | Persistent VM, scale-to-zero optional | ~1–3 s wake | yes (Docker) | SSE works | ~$2–5/mo hobby | Closer regions, similar effort |
| Railway | Persistent container | none on paid | yes | SSE works | ~$5/mo + usage | Dev experience parity with Render |
| Hetzner / DO droplet | Fixed VM | none ever | yes | SSE works | ~$4–6/mo | Predictable cost, no platform lock-in, requires manual TLS / deploy |
| AWS Fargate / GCP Cloud Run | Container with concurrency | seconds on cold | yes | Cloud Run SSE has caveats | usage-based | Overkill for this app's scale |
| **Vercel** | **Serverless functions** | **N/A** | **size-limited** | **SSE not supported reliably** | **N/A** | **Frontend only, do not use for this backend** |

### Recommended migration order

1. Apply the cheap fix on Render (paid plan + Dockerfile cleanup). Re-measure Phase 0 metrics.
2. If still slow, try Fly.io (closest model to Render, near-zero migration cost — a `fly.toml` plus the existing Dockerfile).
3. Only consider a self-managed VPS if the team accepts owning TLS, deploy automation, and process supervision.

### Acceptance criteria

- Cold-first-request latency for `/ifc-egress-graph` on the sample IFC drops below the user's pain threshold (define this once Phase 0 numbers exist).
- p50 warm-request latency does not regress vs. Render.
- CORS still permits the GitHub Pages origin.
- The frontend `VITE_API_BASE` is the only frontend change required (single env var bump in the GitHub Actions workflow).

## Regression Test Checklist

Use at least one small IFC and one larger realistic IFC. Each item is **pass/fail with a check**, not a vibe.

For each phase verify:

- IFC model appears (model bbox is non-empty after `fitCameraToModel`).
- Orbit/pan/zoom remain responsive (sustained ≥ 30 FPS during a 10-second orbit).
- Start and exit picking still work (raycaster returns a hit, marker repositions).
- Egress graph has plausible node/edge counts (within ±10% of the Phase 0 baseline for the same fixture).
- Doors are represented as waypoints (`graph.stats.door_nodes > 0` for fixtures with doors).
- Walls block paths only where intended (a path that should detour around a wall does, verified by fixture).
- Stairs connect floors (path between floor-A and floor-B nodes returns a route that traverses stair nodes).
- Static path returns at least two points.
- Temperature fire stream emits at least one `temperature_step` and one `path_update` SSE event for fixtures with `streamPath=true`.
- Dynamic path updates fire when temperature crosses the lethality threshold.
- RL mode (`/rl/train`) returns a non-empty `path` for a reachable pair on a topology fixture.

If the IFC RL path is broken or unverified for the migration target, mark it explicitly in the PR description rather than implicitly assume it works.

## Known Risks To Preserve

- Module-global `LAST_GRAPHS` makes the backend session-unsafe — a second concurrent user overwrites the first user's graph. This is a latent bug; do not hide it with a renderer migration. Replace with per-session keying (e.g., a graph ID returned from `/ifc-egress-graph` that the frontend passes back in `/ifc-egress-path` and `/fire-sim`) before scaling to multi-user.
- `main.py` line 11 hardcodes `sys.path.append("C:/Users/sarwj/...")` from a different developer's machine. Remove during Phase 1.
- Axis handling is inconsistent across `IFCViewer.jsx` (`upAxis` is forced to `"y"` in `App.jsx`) and the backend (which accepts `up_axis` per request). Don't bury this with a renderer migration.
- Server `/upload-ifc` and browser IFC egress are separate paths. They don't currently share the IFC Lite contract; Phase 3 is where they should converge.
- Existing lint may fail for unrelated reasons; do not mix lint cleanup with loader migration unless it blocks builds.
- TopologicPy is **AGPL-3.0**. Hosting it as a network service that processes user files (which is what `/upload-ifc` and `/ifc-egress-graph` do) triggers AGPL §13 — the source must be made available to users interacting with the service. Decide ownership of this before Phase 4 wires `topologicpy.IFC` deeper into production. This is a project-leadership decision, not an engineering one.
- The Dockerfile installs `topologicpy` and `ifcopenshell` outside of `requirements.txt`, breaking pip caching. Phase 1 fixes this incidentally.

## Recommended Milestone Plan (replaces "Preferred First PR")

The prior version of this doc had a "Preferred First Pull Request" that bundled instrumentation + WASM hosting + three major upgrades + docs into one PR. That's a milestone, not a PR. Sequence:

**Milestone A — Measurement and cleanup (1 PR, ~1–2 days)**

- Phase 0 instrumentation in `IFCViewer.jsx`, `App.jsx`, `main.py` behind a `PERF_LOG` flag.
- Remove the dead `sys.path.append` on `main.py:11`.
- Add `ifcopenshell` to `requirements.txt`.
- Move `pip install ifcopenshell topologicpy` into the cached requirements layer in the Dockerfile.
- Update `wiki/sources/dependencies.md` and `wiki/deployment/render-backend.md`.

**Milestone B — Render plan upgrade (1 PR, hours)**

- Add `plan: starter` to `render.yaml`.
- Re-measure Phase 0 numbers warm vs. cold.
- This is the cheapest possible fix to the user's slowness complaint and should ship before any larger effort.

**Milestone C — Frontend dependency upgrades (4 PRs, one per row of the table)**

- One per package: `web-ifc`, `@thatopen/fragments`, `@thatopen/components`, `three`. Each PR re-runs Phase 0 timings.

**Milestone D — Self-host WASM (1 PR)**

- Phase 2.

**Milestone E — IFC Lite (multi-PR)**

- Phase 3, split into payload contract, frontend emit, backend accept, cache.

After Milestone E, decide based on the rebaselined Phase 0 numbers whether Phase 4 (TopologicPy probe) or Phase 5 (Rust port) is the next target.

The first PR should **not** introduce WebGPU, Rust, or a deployment migration. It should be Milestone A only.
