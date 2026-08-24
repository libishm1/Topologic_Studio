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
