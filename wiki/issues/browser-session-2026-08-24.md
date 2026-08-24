# Browser Test Session — 2026-08-24 (Next line)

Agent: Claude Code. Branch: `next/studio-2`.
Protocol: [agent-instructions-codex-claude.md](../testing/agent-instructions-codex-claude.md).

Harness: `tools/browser-test.mjs`, playwright-core driving the **installed**
Chrome 151 (`C:/Program Files/Google/Chrome/Application/chrome.exe`), headless,
no bundled browser download.

- URL: `http://127.0.0.1:5173` (Vite dev), backend `http://127.0.0.1:8000`
- Model: `Ifc2x3_Duplex_Architecture.ifc` (2.3 MB, IFC2X3)
- Screenshots: `tools/shots/` (gitignored; regenerate with `npm run test:browser`)
- Workflows exercised: app boot, IFC load, graph build, click-picking, routing,
  fire stream with live rerouting, theme toggle, all panel tabs, topology mode

## Final state

| Signal | Result |
| --- | --- |
| Harness checks | 21 / 21 pass |
| Browser console errors | 0 |
| Uncaught page exceptions | 0 |
| Failed requests | 0 |
| HTTP >= 400 | 0 |
| Backend unit tests | 73 / 73 pass |
| Pipeline smoke test | pass |
| ESLint / production build | clean |

In-browser timings on the Duplex model: IFC load 0.7 s, graph build 0.6 s
(2,577 nodes / 17,211 edges, one connected component), worker sampling
2,563 points in 4 ms, route 5.7 m across two storeys via the stairs.

---

## Bug 1 — Cross-origin isolation broke IFC loading entirely

- Date: 2026-08-24
- Reporter: agent
- Workflow: IFC load
- Severity: **fatal**
- Status: fixed

### Evidence

- Page error, repeated ~20x: `Unexpected token '<'`
- Console: `worker sent an error! http://127.0.0.1:5173/undefined:1: Uncaught SyntaxError: Unexpected token '<'`
- Worker construction traced by patching `window.Worker`:
  `allocateUnusedWorker → initMainThread → IfcAPI2.Init` inside
  `node_modules/web-ifc/web-ifc-api.js`
- Screenshot: model never appears; panel stays on "Converting IFC 0%"

### Expected

The IFC converts to fragments and renders.

### Actual

Loading hung indefinitely. Every spawned worker requested `/undefined`,
received `index.html`, and failed to parse it as JavaScript.

### Suspected cause

`topologicpy-web-frontend/vite.config.js` set
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: credentialless` on the dev server, added by this
agent as a speculative optimisation so the multi-threaded web-ifc build could
engage. Those headers make the page cross-origin isolated, which grants
`SharedArrayBuffer`, which makes web-ifc select its **pthread** build. That
build spawns workers with an undefined script URL.

### Fix

Headers removed. The single-threaded build is what production runs anyway,
since static hosts do not send these headers, so dev and production now share
one code path. A comment in `vite.config.js` records why they must not come
back without wiring the pthread worker properly.

### Verification

IFC now loads in 0.7 s with no page errors.

---

## Bug 2 — Fragments worker fetched from unpkg.com at runtime

- Date: 2026-08-24
- Reporter: agent
- Workflow: app boot
- Severity: **environment**
- Status: fixed

### Evidence

Source read of `@thatopen/components` → `FragmentsManager.getWorker()` →
`FragmentsModels.getWorker()`, which does:

```js
const url = `https://unpkg.com/@thatopen/fragments@3.4.7/dist/worker/worker.mjs`;
const response = await fetch(url);
```

### Expected

Booting the viewer should not depend on a third-party CDN.

### Actual

Every cold boot fetched the worker from unpkg.com. Offline and air-gapped
deployment fail, and when the fetch fails the symptom is an opaque worker
syntax error rather than a network message.

### Fix

`scripts/sync-wasm.mjs` became `scripts/sync-assets.mjs` and now copies both
the web-ifc WASM **and** the matching fragments worker out of `node_modules`
into `public/`. `ViewerManager.resolveWorkerUrl()` prefers the local copy,
verifies it is not an HTML error page, and falls back to the CDN only if the
local file is missing.

### Verification

`GET /fragments/worker.mjs` → `200 text/javascript`, 1.39 MB. No unpkg.com
requests during boot.

---

## Bug 3 — One navigation node per door mesh instead of per door

- Date: 2026-08-24
- Reporter: agent
- Workflow: graph build
- Severity: **functional**
- Status: fixed

### Evidence

Model panel reported `Doors 14`; graph stats reported `Doors 44`.

### Expected

One navigation waypoint per IFC door opening.

### Actual

An IFC door is several meshes (frame, leaf, glazing). The sampler emitted a
waypoint per mesh, so 14 doors produced 44 clustered nodes in the doorways.

### Suspected cause

`src/workers/sampler.worker.js::doorWaypoints` iterated meshes, and
`src/viewer/categories.js::extractMeshes` discarded which IFC item each mesh
came from.

### Fix

`extractMeshes` now tags each mesh with `itemId`, and `doorWaypoints`
aggregates by it. A second defect blocked the first attempt: the `clone()` step
in `App.buildGraph` (which copies buffers before transferring them to the
worker) dropped `itemId`, so the worker silently fell back to per-mesh keys.

### Verification

Graph now reports `Doors 14`, matching the model. Node count fell 2,607 → 2,577.

> **Human decision needed.** This changes graph sampling, which the HITL trigger
> list flags. The door count now matches the IFC, but whether one waypoint per
> opening is the right modelling choice for egress capacity is a research call,
> not a code call.

---

## Bug 4 — Missing favicon

- Date: 2026-08-24
- Severity: **environment**
- Status: fixed

`GET /favicon.ico` returned 404 on every page load. Added
`public/favicon.svg` and linked it from `index.html`.

---

## Visual finding 1 — Building a graph appeared to do nothing

- Severity: **visual**
- Status: fixed

### Evidence

`tools/shots/view-Top.png` (before): the roof slab occludes the entire graph;
blue is visible only where floor slabs overhang the walls, which reads as
"the graph was generated outside the building".
`tools/shots/occl-without-model.png` (geometry hidden): the graph correctly
fills the footprint, with purple stairs and orange door nodes.

Bounding boxes confirmed alignment rather than displacement:

```
MODEL  min [-2.76, -4.81, -16.02]  max [6.53, 5.74, 10.55]
GRAPH  min [-2.53, -2.65, -16.02]  max [6.30, 0.61, 10.55]
```

### Fix

Added a Solid / Ghost / Hidden control for the IFC geometry. Ghost sets
`opacity 0.18` and `depthWrite: false` so the graph reads through while the
building stays as context. The first graph build switches to Ghost
automatically; the choice is the user's afterwards. `M` cycles the three modes.

---

## Visual finding 2 — The egress route rendered as a one-pixel thread

- Severity: **visual**
- Status: fixed

`LineBasicMaterial.linewidth` is ignored by every major browser (capped at 1px
on most platforms), so the route — the app's primary output — was lost among
17,000 graph edges. Routes are now `TubeGeometry` with a radius scaled to the
model extent, and the graph fades from 0.45 to 0.16 opacity while a route is
displayed.

See `tools/shots/05-route.png`.

---

## Visual finding 3 — Picks landed on walls and roofs

- Severity: **research-validity**
- Status: fixed

Every pick returned `x = 6.29`, the model's outer wall face: a ray into the
model hits whatever surface faces the camera. Routing snapped to the nearest
graph node regardless, so the marker showed a position the walker could never
occupy, and the displayed start did not match the routed start.

Picks now snap to the nearest graph node on the client, so the marker is the
node the route will actually use.

> **Human decision needed.** Snapping is silent. If a pick lands far from any
> walkable node — clicking a roof, say — the marker jumps somewhere the user did
> not click. Whether to warn above a distance threshold, or refuse the pick, is
> a UX judgement.

---

## Visual finding 4 — Exit and fire origin were near-identical oranges

- Severity: **visual**
- Status: fixed

`--viz-exit #ea580c` against `--viz-fire #ff4500`. The fire origin is now a
spiked octahedron rather than a sphere, so the two differ in shape as well as
hue and remain distinguishable without colour vision.

---

## Bug 5 — Navigation graph was a space-frame truss, not a floor mesh

- Date: 2026-08-24
- Reporter: human (screenshot), diagnosed by agent
- Workflow: graph build
- Severity: **research-validity**
- Status: fixed

### Evidence

`tools/inspect-graph.mjs` on the Duplex, before the fix:

```
2577 nodes, height bands (5 cm buckets):
    y=-2.65  529      <- ground slab TOP
    y=-2.50  284      <- ground slab UNDERSIDE
    y= 0.15  401      <- first floor ceiling / slab
    y= 0.30  307      <- first floor slab
    y= 0.60  553      <- first floor finish

17211 edges | level (<=5cm) 7620 | sloped (5-35cm) 7490 | steep (>35cm) 2101
```

56% of edges were non-level, bracing two parallel sheets into a lattice.

### Expected

One flat walkable mesh per storey, joined between storeys only by stairs.

### Actual

Every slab produced two point sheets a slab-thickness apart, and the
neighbourhood search wired them together with diagonals, so the graph rendered
as a 3D space-frame truss. Routes could also cut through a slab.

### Suspected cause

Sampling accepted a surface whose normal was *near-horizontal in either
direction*:

```js
const upComponent = Math.abs(signedUp);   // worker
walkable = np.abs(face_normal_up) >= min_up   # backend
```

The absolute value was deliberate (IFC winding is not always trustworthy) but it
makes a slab's underside and a ceiling as walkable as a floor.

Measurement showed winding *is* reliable in the fragments output for this
model: floors split 158 up-facing / 226 down-facing at distinct heights, and
stairs split exactly 184 / 184 (treads vs soffit).

### Fix

Three layers, from most to least specific:

1. **Only sample upward-facing surfaces.** The absolute value is gone from both
   the browser worker and the Python sampler. If insisting on upward normals
   yields nothing at all, the model's winding cannot be trusted and sampling
   retries accepting either orientation, flagged as `windingFallback`.
2. **Collapse each vertical column onto its walking surface.** A safety net for
   coincident or duplicated surfaces: points in a horizontal cell are clustered
   by height and only the topmost of each cluster survives (`column_gap`,
   default 0.9 m).
3. **Floor links must stay near-level.** A floor-to-floor edge may span at most
   `max_edge_rise` (default 0.35 m), exposed in the UI as *Max vertical step*.
   Stair links are unaffected.

### Verification

```
1765 nodes | 10441 edges | level 9148 | sloped 1241 | steep 52

by node kind      level   sloped    steep
  floor-floor      7921        0        0     <- was 6098 / 2623 / 0
  stair-stair      1041     1093        0     <- stairs are meant to climb
  floor-stair         8      148       48     <- landings
```

**Zero non-level floor-to-floor edges.** Routes still cross storeys via the
stairs (verified start y=-1.93 to exit y=+0.61).

### Follow-on: connectivity dropped from 100% to 94.2%

The largest component now holds 1,663 of 1,765 nodes and spans both storeys.
The 12 orphans are one up-facing surface 0.5 m below the first floor, in
fragments of 57/25/4/3/3/2/2/2/1/1/1/1 nodes.

This is not a regression in substance: the previous 100% was manufactured by
the truss gluing together surfaces a person cannot walk between. Isolated floor
area with no route out is exactly what an egress tool should surface, so the
graph is not pruned. The panel now reports a **Reachable %** stat and explains
that raising *Max vertical step* will bridge surfaces at slightly different
heights.

> **Human decision needed.** Whether that isolated surface at 0.5 m below the
> first floor is real walkable area the model fails to connect with a step, or
> geometry that should not be walkable at all, is a model-interpretation call.

---

## Residual risks

Recorded in [open-risks.md](open-risks.md):

- Upward-only sampling relies on trustworthy face winding. The fallback
  catches a model with no upward faces at all, but not one with *partly*
  inverted winding, which would silently lose walkable area.

- One model, one browser, one machine. Not a cross-browser or cross-model suite.
- Not wired into CI.
- Ghost mode mutates material properties directly on the fragments model. If
  That Open reassigns materials on LOD changes, the ghost may not persist. Not
  observed in this session, not proven absent.
- `tools/bench_egress.py` double-counts walls on IFC2X3, because
  `by_type("IfcWall")` already includes `IfcWallStandardCase`. Benchmark
  payload only; both backends get identical input, so the comparison holds.
