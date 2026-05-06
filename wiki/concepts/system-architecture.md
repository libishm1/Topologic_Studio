# System Architecture

## Overview

Topologic Studio is a full-stack research prototype for IFC fire egress workflows. The browser loads and visualizes IFC models, extracts relevant geometry, sends simplified mesh payloads to the backend, receives navigation graph/path/fire/RL results, and renders overlays with Three.js.

## Frontend

Primary files:

- `src/App.jsx`
- `src/IFCViewer.jsx`
- `src/TopologyViewer.jsx`

The frontend has two viewer paths:

1. TopologicPy JSON contract viewer, handled by `TopologyViewer.jsx`.
2. IFC Fragments viewer, handled by `IFCViewer.jsx`.

`App.jsx` owns most workflow state:

- loaded topology/IFC file state
- IFC graph statistics, coordinates, edge coordinate pairs, and edge ID pairs
- start/exit picked points
- static IFC egress path
- fire simulation controls
- temperature map and hot nodes
- dynamic hazard path
- RL request controls

`IFCViewer.jsx` owns the Three.js/That Open rendering lifecycle:

- creates a `Components` system, `Worlds`, `SimpleScene`, `SimpleRenderer`, `SimpleCamera`, and `Raycasters`
- initializes the `FragmentsManager` worker
- initializes `IfcLoader` with `web-ifc`
- loads IFC buffers
- extracts IFC entity IDs for slabs, stairs, doors, spaces, walls, and storeys
- extracts mesh geometry for floors, stairs, doors, and walls
- renders static path as red, dynamic path as magenta, and graph wires as per-vertex colored `LineSegments`

`TopologyViewer.jsx` is a generic Three.js renderer for TopologicPy export contracts. It renders faces, edges, vertices, selection state, parent hierarchy cycling, fire/path highlighting, and manual camera fitting.

## Backend

Primary file: `topologicpy-web-backend/app/main.py`

The backend is a single FastAPI app. It imports TopologicPy classes and exposes:

- `/health`
- `/upload-topology`
- `/upload-ifc`
- `/graph-meta`
- `/ifc-egress-graph`
- `/ifc-egress-path`
- `/fire-sim`
- `/fire-sim/stream`
- `/rl/train`

The backend stores most runtime graph state in the process-global `LAST_GRAPHS` dictionary. That keeps the implementation simple but means graph state is per-process, not durable, and not safe for multi-user production workloads.

## Data Flow

1. User loads an IFC file in the browser.
2. `IFCViewer.jsx` displays the model quickly and defers heavy egress extraction.
3. `IFCViewer.jsx` collects IFC IDs and then extracts floor, stair, door, and wall mesh payloads.
4. `App.jsx` posts the geometry payload to `/ifc-egress-graph`.
5. Backend samples walkable points from floors/stairs, injects door points, stores wall obstacle segments, builds adjacency, and returns graph edges plus edge IDs.
6. Viewer renders the graph overlay.
7. User picks start and exit points.
8. `App.jsx` posts to `/ifc-egress-path`, and backend resolves picked coordinates to nearest graph nodes.
9. Backend computes a path with wall checks and door exemptions.
10. Fire can run either as precomputed timeline or SSE stream.
11. Temperature fire stream can also recompute and stream hazard-weighted path updates.

## Runtime Constraints

- Frontend is Vite + React and can be deployed as static files.
- Backend is FastAPI/Uvicorn and must be deployed as a persistent API service.
- Production frontend/backend split requires correct `VITE_API_BASE` at build time and matching backend `CORS_ORIGINS`.
- Browser IFC parsing depends on `web-ifc` WASM availability.
- Backend TopologicPy functionality depends on correct Python environment and `topologicpy` installation.

## Production Readiness Assessment

The project is viable as a research prototype. It is not yet production-grade for multi-user analysis because:

- `LAST_GRAPHS` is global mutable process memory.
- no persistent project/session store exists.
- graph build can be CPU-heavy and single-process.
- no automated test suite currently covers path, fire, or IFC extraction.
- lint currently fails.
- frontend bundle is very large after minification.

