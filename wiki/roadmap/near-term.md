# Near-Term Roadmap

## Stabilize Current Behavior

- Fix frontend lint errors without changing behavior.
- Add a small backend route test suite that imports the app and checks error codes for missing graph state.
- Add a frontend build smoke command to CI.
- Document whether the app contract is Y-up or axis-configurable.
- Pin or self-host `web-ifc` WASM assets.

## Improve IFC Egress

- Add session-scoped graph IDs instead of global `LAST_GRAPHS`.
- Add graph metadata to responses: axis, transform, sampling params, wall usage, creation timestamp.
- Return route node IDs with route coordinates for easier debugging.
- Add graph build warnings when floors, stairs, doors, or walls are absent.
- Add optional graph simplification after route correctness is stable.

## Improve Fire And Dynamic Routing

- Include temperature statistics in SSE events, such as min, max, and hot node count.
- Add deterministic seed handling for RL.
- Persist RL result metadata: reward curve, episodes, convergence, hazard settings.
- Add route comparison output: static shortest path versus hazard-weighted path.

## Improve Documentation Workflow

- Keep wiki chunks below a size that one agent can load comfortably.
- Update the matching chunk whenever route contracts or implementation behavior change.
- Add OCR output for the local PDF into a separate source chunk after review.

