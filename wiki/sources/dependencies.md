# Dependencies

Last verified upstream: 2026-05-07. Re-run the version checks before any upgrade PR (commands at the bottom of this page).

## Frontend

Source: `topologicpy-web-frontend/package.json`

Runtime dependencies:

| Package | Pinned | Latest (2026-05-07) | Role |
|---|---:|---:|---|
| `react` | `^19.2.0` | reverify | UI state and rendering. |
| `react-dom` | `^19.2.0` | reverify | Browser mount. |
| `vite` | `^7.2.4` | reverify | Dev server and production build. |
| `@vitejs/plugin-react` | `^5.1.1` | reverify | React transform for Vite. |
| `three` | `^0.181.2` | `0.184.0` | 3D geometry, lines, colors, raycasting. |
| `@thatopen/components` | `^3.2.6` | `3.4.5` | IFC viewer components and loader. |
| `@thatopen/fragments` | `^3.2.13` | `3.4.5` | IFC fragment worker and model geometry. |
| `web-ifc` | `^0.0.73` | `0.0.77` | Browser-side IFC parsing. |
| `axios` | `^1.13.2` | reverify | HTTP client. |

Dev dependencies include ESLint `^9.39.1` and React Hooks lint plugin `^7.0.1`.

`IFCViewer.jsx` hardcodes a CDN fallback `https://unpkg.com/web-ifc@0.0.73/` for the `web-ifc` WASM. This must be moved to a self-hosted asset under `public/wasm/` keyed by the upgraded `web-ifc` version. See `wiki/roadmap/ifc-webgpu-rust-port-handoff.md` Phase 2.

## Backend

Source: `topologicpy-web-backend/requirements.txt` (no version pins):

```
fastapi
uvicorn[standard]
pydantic
python-multipart
topologicpy
```

`ifcopenshell` is **not** declared in `requirements.txt`. The Dockerfile installs it after `pip install -r requirements.txt` via a separate `pip install --no-cache-dir ifcopenshell topologicpy` step. The local venv must install it manually for `/upload-ifc` to work locally.

Local verified versions from the backend venv (2026-05-07):

| Package | Local venv | Latest (2026-05-07) |
|---|---:|---:|
| FastAPI | `0.124.0` | reverify |
| Uvicorn | `0.38.0` | reverify |
| Pydantic | `2.12.5` | reverify |
| TopologicPy | `0.8.93` | `0.9.26` |
| ifcopenshell | `0.8.4` (manual) | reverify |

The 0.8 → 0.9 TopologicPy bump is significant: 0.9.x adds a `topologicpy.IFC` module (`IFC.MeshDataByPath`, `IFC.TopologiesByPath`, experimental `IFCFastTopology`) which is **not** available in 0.8.93. Phase 4 of the roadmap evaluates whether to wire this into `/upload-ifc` or `/ifc-egress-graph`.

## Dependency Notes

- Browser IFC egress depends on That Open Components and `web-ifc`.
- Server IFC conversion depends on `ifcopenshell` (Docker-only today).
- Dynamic hazard rerouting depends on TopologicPy `Graph.ShortestPath`.
- The frontend production bundle is large because IFC/3D dependencies are heavy.

## Reverify Commands

PowerShell (frontend):

```powershell
& "..\node-v24.11.1-win-x64\npm.cmd" view web-ifc version
& "..\node-v24.11.1-win-x64\npm.cmd" view @thatopen/components version
& "..\node-v24.11.1-win-x64\npm.cmd" view @thatopen/fragments version
& "..\node-v24.11.1-win-x64\npm.cmd" view three version
& "..\node-v24.11.1-win-x64\npm.cmd" view react version
& "..\node-v24.11.1-win-x64\npm.cmd" view vite version
& "..\node-v24.11.1-win-x64\npm.cmd" view axios version
```

PowerShell (backend):

```powershell
.\.venv\Scripts\python.exe -m pip index versions topologicpy
.\.venv\Scripts\python.exe -m pip index versions ifcopenshell
.\.venv\Scripts\python.exe -m pip show topologicpy ifcopenshell fastapi pydantic uvicorn
```

