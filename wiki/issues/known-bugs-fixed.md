# Known Bugs Fixed Or Mitigated

This page records bugs visible in logs or code comments that the current implementation appears to have fixed or mitigated.

## Documentation Requirement For Test Sessions

Every browser-based debugging or testing session must leave an audit trail:

- If a bug is fixed or mitigated, add it to this page.
- If a bug remains open, add it to [open-risks.md](open-risks.md) or a dedicated issue page.
- If the run is only verification, update [verification-status.md](verification-status.md).
- Use [bug-log-template.md](../testing/bug-log-template.md) for new or ambiguous findings.

This rule comes from the HITL workflow in [human-in-the-loop-principles.md](../testing/human-in-the-loop-principles.md): autonomous testing is allowed, but human review and bug documentation preserve accountability.

## `/upload-topology` 500 On Unsupported JSON

Evidence:

- `uvicorn-err.log` contains an older `TypeError: string indices must be integers, not 'str'` inside `Topology.ByJSONDictionary`.
- `uvicorn-out.log` records an old `POST /upload-topology` returning `500`.

Current mitigation:

- `upload_topology` now accepts pre-converted viewer contracts directly.
- Blender/Sverchok-style JSON is rejected with HTTP `400`.
- Unsupported non-list payloads are rejected with HTTP `400`.
- TopologicPy parse failures are caught and returned as HTTP `422`.

## Wall Obstacles Hiding The Display Graph

Current mitigation:

- `/ifc-egress-graph` stores wall segments but does not prune the displayed graph.
- `_shortest_path_ids` applies wall blocking only during path computation.

Why it matters:

- The full graph remains visible for debugging.
- Path constraints can change without rebuilding display geometry.

## Door Connectivity Through Walls

Current mitigation:

- Door positions are extracted as forced waypoints.
- Door nodes are force-connected to nearby floor nodes.
- Door-touching edges are exempt from wall blocking.

Why it matters:

- Without this, walls can block all edges around legitimate openings.

## Browser Responsiveness During IFC Extraction

Current mitigation:

- IFC geometry extraction is delayed until after the model is rendered.
- Heavy extraction yields between floor, stair, door, and wall extraction using animation-frame timing.

Why it matters:

- Users can see and navigate the model before the expensive egress extraction completes.

## GitHub Pages Asset Base

Current mitigation:

- Vite `base` comes from `BASE_PATH`.
- GitHub Actions sets `BASE_PATH` to `/${{ github.event.repository.name }}/`.

Why it matters:

- Project Pages deploys under a nested path, so absolute `/assets/...` links would otherwise break.

## Browser Session 2026-08-24 (Next Line)

Four bugs found by driving the app in real Chrome and fixed. Full logs,
evidence and screenshots in
[browser-session-2026-08-24.md](browser-session-2026-08-24.md).

- **fatal** — Cross-origin isolation headers on the Vite dev server made
  web-ifc select its pthread build, which spawned workers with an undefined
  script URL. IFC loading never completed. Headers removed.
- **environment** — `FragmentsManager.getWorker()` fetched the fragments worker
  from unpkg.com on every boot. Now self-hosted from `public/fragments/` by
  `scripts/sync-assets.mjs`, CDN kept only as a fallback.
- **functional** — An IFC door produced one navigation node per mesh rather
  than per door, so 14 doors became 44 nodes. Meshes now carry `itemId` and the
  sampler aggregates by it.
- **environment** — Missing favicon, a 404 on every page load.

Four visual/UX defects also fixed: graph invisible behind solid geometry
(added Solid/Ghost/Hidden), route drawn as a 1px line (now a tube),
picks landing on walls (now snap to the graph), and exit vs fire origin
sharing a hue (fire is now a distinct shape).

## Next line, 2026-10-27 review

Full evidence, measurements and tests: [review-2026-10-27.md](../roadmap/review-2026-10-27.md).

- **Temperature fire heated some nodes twice as fast.** `np.add.reduceat` with
  zeroed offsets summed only the first neighbour of any node stored before an
  isolated node. Now `np.bincount`; tested against Classic's loop.
- **RL could not reach the exit on a real model.** The fire was seeded at the
  evacuee's start; gamma 0.9 hid the exit; terminal transitions bootstrapped;
  walls were ignored; the state ignored time; the "reward" was a sum of
  Q-values. All fixed; 20/20 vs 0/20 on a tight fire race.
- **Flood and radial fires never rerouted.** Now they do, with burning nodes
  impassable.
- **Per-request connected-components BFS** cost 11 of 14 ms per route query.
  Cached.
- **Wall blocking was a per-edge Python loop.** Vectorised, bit-identical.
- **`ifcopenshell` was unbounded**, so the Linux image would have run 0.9.0
  against a suite tested on 0.8.5. Capped `<0.9`.
- **A cold-starting hosted backend was reported "offline"** after 8 s. The
  UI now shows "Waking backend…" and waits up to 90 s.

## State-machine audit, area 1: fire runs and compute workers (2026-11-10)

Full machines, event sequences and evidence: [state-machines.md](../architecture/state-machines.md). On `next/browser-core`; each fix has a regression test that fails on the code before it.

- **SM-1. A fire run outlived Stop, replacement or a restart.** A run stopped or
  replaced while its kept run was read from IndexedDB started computing anyway,
  with no Stop able to end it, and could write its numbers on the newer run's
  card. Start fire during a plan's run computed that run again from zero.
  Repeat runs went on after Stop and after a new model. After a GPU-to-CPU
  restart the tenability index kept the abandoned frames' doses.
- **SM-2. One forecast worker served the plan, the two-zone fire and the
  report.** The report's Stop ended a plan beside it and left the Plan button
  spinning, disabled, for good; a second Plan click lost a starting two-zone
  fire. Each now has its own worker, and the two-zone fire has a "starting"
  state that Start, Plan and Stop respect.
- **SM-3. Async actions without a starting state or a check that they were
  still wanted.** People leaving with the fire drew nobody after a fire was
  started again with no step delay; a double click on Record GIF made two
  recorders; Identify by shape stayed busy when its module failed to load;
  moving a point during agent training drew its answer for the old points; Stop
  in a guided run's fire stage counted as the fire ending.

## State-machine audit, area 2: network, backend and storage (2026-11-10)

Details and evidence: [state-machines.md](../architecture/state-machines.md), SM-4 to SM-6. On `next/browser-core`.

- **SM-4. The graph's copy on the server.** The upload's answer replaced the
  graph object, so a fire or plan running then ended silently and the code
  check, report and plans vanished; a graph built while the backend woke was
  never uploaded; a copy the server lost was never sent again; a saved
  TopologicPy engine refused routing with no server. The id is now kept beside
  the graph, the graph is uploaded when a TopologicPy feature first asks, and
  sent again once if the server lost it; with no server the browser engine runs.
- **SM-5. Answers for inputs that changed meanwhile.** TopologicPy routes,
  alternatives, comparisons and the Navigation cross-check were drawn after the
  points or the model changed; the comparison outlived its route; Stop did not
  end a wait for the human check; a description stayed under another model.
- **SM-6. The backend's state and the site's updates.** `/health` and
  `/api/graphs` listed every graph id (any visitor could read or delete another's
  graph): counts only now, live after the next manual Render deploy. A tab open
  across a deploy now says to reload. A timeout or a cancel no longer marks the
  backend offline.
