# Open Risks

These are current implementation risks, not all confirmed bugs.

## Browser Testing Accountability

Agents must not run headless browser tests without updating issue or verification docs. Required evidence for browser workflow claims:

- console messages
- page errors
- failed requests
- API statuses
- screenshot paths when visual behavior matters
- human decision needed, if any

Use [agent browser testing instructions](../testing/agent-instructions-codex-claude.md).

## Frontend Lint Fails

Latest `npm.cmd run lint` reported 9 errors and 4 warnings:

- Unused state/vars in `App.jsx`: `fireTimeline`, `fireCellBboxes`, placeholder `_` args, `vertexList`.
- Unused helper in `IFCViewer.jsx`: `waitForModelReady`.
- Unused locals in `TopologyViewer.jsx`: `camera`, `controls`.
- `vite.config.js`: `process` is not defined for ESLint.
- React hooks warnings for missing dependencies.

Build still succeeds.

## Large Production Bundle

Vite build warns that a chunk is larger than 500 kB after minification:

- `assets/index-C7fSBkY8.js`: about `5,197.43 kB`, gzip about `872.14 kB`.
- IFC and 3D libraries are likely the main contributors.

Future mitigation: dynamic imports for the IFC viewer and manual chunks for heavy 3D/IFC libraries.

## In-Memory Graph State

`LAST_GRAPHS` is process-global. A second user or second browser tab can overwrite graph state for another session.

Future mitigation: per-session graph IDs or stateless graph payload references.

## Axis Handling

Current frontend hard-codes `ifcUpAxis = "y"`, but backend defaults and some helper logic are mixed:

- `IfcEgressRequest.up_axis` defaults to `"z"`.
- `_nearest_node_id` explicitly prefers same-floor Y distance.
- Agent-height offset is applied to `p[2]`, not `p[up_idx]`.

Future mitigation: either freeze Y-up as the contract or generalize all axis-sensitive math.

## Remote WASM Dependency

`IFCViewer.jsx` defaults to `https://unpkg.com/web-ifc@0.0.73/` for WASM files.

Future mitigation: serve pinned WASM files from the app's static assets for reproducible deployment and offline/local resilience.

## Server IFC Dependency Split

`ifcopenshell` is installed in Docker but not listed in `requirements.txt`. Local `/upload-ifc` may fail if the developer venv lacks it.

Future mitigation: add `ifcopenshell` to backend requirements if server-side IFC upload remains a supported feature.

## Scanned PDF

The local PDF has zero extractable text. Research documentation from it requires OCR before semantic parsing.
