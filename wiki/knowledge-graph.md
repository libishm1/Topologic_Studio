# Knowledge Graph

This page is the human-readable form of `knowledge_graph.json`.

## Core Nodes

- `Topologic Studio`: research web app for IFC fire egress analysis.
- `React/Vite frontend`: UI, state orchestration, deployment artifact.
- `IFCViewer`: browser IFC rendering and extraction.
- `TopologyViewer`: generic TopologicPy JSON renderer.
- `FastAPI backend`: API, graph generation, fire, pathing, RL.
- `TopologicPy`: backend graph/topology library and `Graph.ShortestPath`.
- `web-ifc`: browser IFC parsing engine.
- `That Open Components`: IFC Fragments viewer and geometry extraction stack.
- `Three.js`: rendering, graph overlays, vertex colors, picking support.
- `IFC egress graph`: sampled walkable graph plus door nodes and wall segments.
- `Fire simulation`: binary and temperature graph fire.
- `Dynamic rerouting`: hazard-weighted path recomputation.
- `Q-learning`: experimental RL route training.
- `GitHub Pages`: frontend hosting target.
- `Render`: backend hosting target.
- `OpenKB-style wiki`: current documentation structure.

## Important Relationships

- `IFCViewer` extracts geometry for `IFC egress graph`.
- `IFC egress graph` stores state in `LAST_GRAPHS["ifc"]`.
- `Dynamic rerouting` uses `TopologicPy` and current fire temperatures.
- `Fire simulation` colors `Three.js` graph overlays through `edge_ids`.
- `React/Vite frontend` deploys to `GitHub Pages`.
- `FastAPI backend` deploys to `Render`.
- `GitHub Pages` requires `VITE_API_BASE`.
- `Render` requires `CORS_ORIGINS`.
- `OpenKB-style wiki` summarizes PDFs and source implementation.

## Concept Clusters

1. Implementation: frontend, backend, APIs, graph, fire, RL.
2. Research: non-manifold topology, two-graph interior representation, safest routes, dynamic routing, coupled fire-egress modeling.
3. Operations: OneDrive runtime, bundled Node, backend `.venv`, deployment.
4. Quality: build, lint, known bugs, tests, roadmap.

