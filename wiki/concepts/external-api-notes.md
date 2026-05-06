# External API And Library Notes

## OpenKB

Source: https://github.com/VectifyAI/OpenKB

OpenKB is an open-source CLI that compiles raw documents into a structured, interlinked wiki-style knowledge base. Relevant features for this wiki:

- broad document support through conversion tools
- long-document handling through PageIndex
- summaries and concept pages
- cross-links and contradiction/gap checks
- Obsidian-compatible Markdown wikilinks
- `wiki/`, `sources/`, `summaries/`, `concepts/`, `reports/`, `log.md`, and `AGENTS.md` style layout

Adoption in this repo: the wiki structure mirrors OpenKB, but the CLI was not run because this environment did not have an OpenKB LLM key workflow configured.

## Vite

Context7 library ID: `/vitejs/vite`

Relevant docs:

- `defineConfig` supports a `base` option for public asset base paths.
- GitHub Pages deployment under `/<repo>/` requires `base: '/<repo>/'`.
- only env variables with `VITE_` prefix are exposed to client code by default.
- `vite build` and `vite preview` are the standard production build/preview scripts.

Adoption in current system:

- `vite.config.js` reads `process.env.BASE_PATH || '/'`.
- GitHub Actions sets `BASE_PATH` to the repository subpath.
- `App.jsx` reads `import.meta.env.VITE_API_BASE`.

Risk:

- ESLint currently flags `process` as undefined in `vite.config.js`; config should either include Node globals for config files or import/read env in an ESLint-friendly way.

## FastAPI

Context7 library ID: `/fastapi/fastapi`

Relevant docs:

- Pydantic models declared as operation parameters are parsed from JSON request bodies and validated.
- validation schemas feed OpenAPI docs.
- `StreamingResponse` streams generator output chunk by chunk and can be used for `text/event-stream`.
- `UploadFile`/`File` are the standard pattern for uploaded files.
- CORS is configured through middleware.

Adoption in current system:

- Pydantic request models define fire, RL, IFC geometry, egress graph, and path payloads.
- `StreamingResponse(gen(), media_type="text/event-stream")` implements SSE.
- `UploadFile = File(...)` is used by `/upload-ifc`.
- `CORSMiddleware` allows local Vite origins and env-configured production origins.

## Three.js

Context7 library ID: `/mrdoob/three.js`

Relevant docs:

- `BufferGeometry` stores attributes such as `position`, `normal`, `uv`, and `color`.
- `Float32BufferAttribute` can store vertex positions and colors.
- materials can enable `vertexColors`.
- `LineSegments` is suitable for graph edge overlays.
- `Raycaster` can select/pick scene intersections.
- `OrbitControls` supports interactive camera navigation.
- geometry and materials should be disposed when removed.

Adoption in current system:

- `IFCViewer.jsx` builds graph overlays as `BufferGeometry` + `LineSegments`.
- per-edge endpoint colors are stored in a `color` attribute.
- removed path/graph objects dispose geometry/material.
- picking uses That Open raycasters rather than raw Three.js raycasting.
- `TopologyViewer.jsx` uses Three.js and `OrbitControls` for TopologicPy JSON rendering.

## web-ifc And That Open

Source: https://github.com/ThatOpen/engine_web-ifc

The web-ifc project describes itself as a JavaScript library to read and write IFC files at native speeds and part of That Open Company's open BIM tooling.

Adoption in current system:

- `web-ifc@0.0.73` supplies IFC type constants and low-level IFC API access.
- `@thatopen/components` supplies `IfcLoader`, scene/world/camera/renderer abstractions, fragments, and raycasters.
- `@thatopen/fragments` supplies the worker used by `FragmentsManager`.

## IFC / buildingSMART

Sources:

- https://technical.buildingsmart.org/standards/ifc/
- https://www.buildingsmart.org/standards/bsi-standards/industry-foundation-classes/

buildingSMART describes IFC as an open, international, vendor-neutral standard for digital descriptions of the built environment. It notes that IFC can be encoded in multiple formats and that the latest official IFC version is IFC 4.3.2.0, with prior official versions including IFC 4 and IFC 2x3.

Adoption in current system:

- sample IFC is IFC2X3.
- browser extraction currently targets common building entities needed for egress rather than full IFC semantics.

