# Context7 API Notes

This page records the library documentation context used for the wiki.

## Vite

Context7 library ID: `/vitejs/vite`

Relevant notes:

- `base` rewrites generated asset paths for nested deployments such as GitHub Pages project sites.
- `vite build --base=/path/` is an alternate CLI form.
- `import.meta.env.BASE_URL` exposes the configured public base path at runtime.
- Vite static deployment docs include a GitHub Actions pattern: checkout, setup Node, install, build, deploy `dist`.

Adopted in this system:

- `vite.config.js` sets `base` from `BASE_PATH`.
- GitHub Actions sets `BASE_PATH` to the repository path and uploads `dist`.
- App runtime backend URL uses `import.meta.env.VITE_API_BASE`.

## FastAPI

Context7 library ID: `/fastapi/fastapi`

Relevant notes:

- Pydantic models are standard request body contracts.
- `UploadFile` plus `File` supports multipart file uploads.
- `StreamingResponse` can stream generator output chunk by chunk, suitable for SSE-style data.
- Middleware is the expected place for CORS behavior.

Adopted in this system:

- Request contracts are Pydantic classes in `main.py`.
- `/upload-ifc` uses `UploadFile = File(...)`.
- `/fire-sim/stream` returns `StreamingResponse(..., media_type="text/event-stream")`.
- CORS origins are configured during app startup.

## Three.js

Context7 library ID: `/mrdoob/three.js`

Relevant notes:

- `BufferGeometry` with `Float32BufferAttribute` is the right primitive for large procedural geometry.
- Vertex colors require a `color` attribute and material `vertexColors: true`.
- `Raycaster.setFromCamera` and `intersectObjects` are standard picking tools.
- `OrbitControls` should be updated during render loops.
- Geometry and material disposal matters in single-page apps.

Adopted in this system:

- IFC graph overlays use `LineSegments` and vertex colors for fire temperatures.
- Topology rendering uses `BufferGeometry` for triangulated faces and lines.
- Both viewers clean up or dispose geometry/material objects during rebuilds or unmount.

