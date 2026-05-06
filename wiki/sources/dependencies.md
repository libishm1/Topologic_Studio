# Dependencies

## Frontend

Source: `topologicpy-web-frontend/package.json`

Runtime dependencies:

| Package | Version range | Role |
|---|---:|---|
| `react` | `^19.2.0` | UI state and rendering. |
| `react-dom` | `^19.2.0` | Browser mount. |
| `vite` | `^7.2.4` | Dev server and production build. |
| `@vitejs/plugin-react` | `^5.1.1` | React transform for Vite. |
| `three` | `^0.181.2` | 3D geometry, lines, colors, raycasting. |
| `@thatopen/components` | `^3.2.6` | IFC viewer components and loader. |
| `@thatopen/fragments` | `^3.2.13` | IFC fragment worker and model geometry. |
| `web-ifc` | `^0.0.73` | Browser-side IFC parsing. |
| `axios` | `^1.13.2` | HTTP client. |

Dev dependencies include ESLint `^9.39.1` and React Hooks lint plugin `^7.0.1`.

## Backend

Source: `topologicpy-web-backend/requirements.txt`

Declared:

- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `python-multipart`
- `topologicpy`

Local verified versions from the backend venv:

- FastAPI `0.124.0`
- Uvicorn `0.38.0`
- Pydantic `2.12.5`
- TopologicPy `0.8.93`

Docker additionally installs:

- `ifcopenshell`
- `topologicpy`

## Dependency Notes

- Browser IFC egress depends on That Open Components and `web-ifc`.
- Server IFC conversion depends on `ifcopenshell`.
- Dynamic hazard rerouting depends on TopologicPy `Graph.ShortestPath`.
- The frontend production bundle is large because IFC/3D dependencies are heavy.

