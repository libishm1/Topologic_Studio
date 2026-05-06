# System Knowledge Graph

This page is the human-readable graph. For machine routing, use [graph.json](graph.json).

## Core Nodes

| Node | Type | Meaning |
|---|---|---|
| `React App` | frontend | Owns state, route calls, mode switching. |
| `IFC Viewer` | frontend | Loads IFC fragments and extracts geometry. |
| `Topology Viewer` | frontend | Renders Topologic viewer contracts. |
| `FastAPI Backend` | backend | Serves upload, graph, path, fire, RL routes. |
| `IFC Egress Graph` | model | In-memory graph built from browser geometry. |
| `TopologicPy` | library | Converts topology and computes hazard paths. |
| `Three.js` | library | Renders geometry and overlays. |
| `That Open Components` | library | Browser IFC loading and fragments. |
| `web-ifc` | library | IFC parsing and entity IDs in browser. |
| `Vite` | tooling | Frontend dev/build/deploy. |
| `Render` | deployment | Backend runtime. |
| `GitHub Pages` | deployment | Frontend runtime. |

## High-Value Edges

- `React App` calls `FastAPI Backend`.
- `IFC Viewer` extracts geometry for `IFC Egress Graph`.
- `FastAPI Backend` stores `IFC Egress Graph` in `LAST_GRAPHS.ifc`.
- `IFC Egress Graph` feeds static path, fire spread, and dynamic hazard routing.
- `TopologicPy` computes hazard-weighted shortest paths when temperature mode streams path updates.
- `Three.js` renders topology, IFC graph wires, fire colors, static paths, and dynamic paths.
- `Vite` builds the React app for `GitHub Pages`.
- `Render` serves FastAPI and must allow the GitHub Pages origin by CORS.

## Failure Edges To Watch

- `GitHub Pages` to `Render`: CORS or wrong `VITE_API_BASE`.
- `IFC Viewer` to `web-ifc`: remote WASM path availability.
- `React App` to `FastAPI Backend`: route depends on in-memory graph state.
- `IFC Egress Graph` to `Path`: axis mismatch or wall overblocking.
- `PDF Sources` to `Research Docs`: OCR missing.

