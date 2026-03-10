# Topologic_Studio — IFC Fire Egress

Extensions to [TopologicStudio](https://github.com/wassimj/TopologicStudio) adding IFC model loading, spatial graph generation, and fire egress path computation — including reinforcement learning-based path training and fire spread simulation.

Built on [TopologicPy](https://topologic.app/) — a non-manifold topology library for spatial reasoning in architecture.

**Live demo:** [libishm1.github.io/Topologic_Studio](https://libishm1.github.io/Topologic_Studio)

---

## What This Fork Adds

The upstream TopologicStudio handles general topology and graph operations. This fork adds a dedicated **Fire + Egress** workflow:

- **IFC loading** — import standard IFC models directly into the browser viewer
- **Egress graph generation** — convert IFC spatial data into a traversable navigation graph using Topologic's cell model
- **Shortest path computation** — compute egress paths between user-defined start and exit points
- **Fire spread simulation** — simulate fire propagation with optional temperature model
- **RL path training** — train reinforcement learning agents to find optimal egress paths under dynamic fire conditions
- **Wall obstacle integration** — use IFC wall geometry as navigation obstacles *(work in progress)*

---

## Workflow

### 1 — Load IFC
Import any IFC file into the browser-based 3D viewer.

![preview](https://github.com/user-attachments/assets/cb098eef-8260-4154-838c-c8c28f7e4edc)


### 2 — Build Egress Graph
Generate a spatial navigation graph from the IFC model. Topologic converts rooms, doors, stairs, and corridors into graph nodes and edges.

![2](https://github.com/user-attachments/assets/24184ffb-fb93-41b9-9c83-640becd1aa73)


*Graph: 1,866 nodes / 1,552 edges · 14 doors · 57 walls · 21 floors / 4 stairs*

### 3 — Set Start and Exit Points
Click to define the fire origin (orange) and egress exit (green) directly on the 3D model.

![3](https://github.com/user-attachments/assets/7caa0561-e433-44fc-952a-6839715c4410)
![4](https://github.com/user-attachments/assets/9cf129d2-078f-446a-a221-dd835f3d4348)


### 4 — Compute Egress Path
Run shortest-path computation. The path is highlighted in red through the building graph.

![5](https://github.com/user-attachments/assets/f4b4c2ac-d848-442f-96d8-25d60687571f)


### 5 — Wall Obstacles (WIP)
Toggle IFC walls as navigation obstacles for more spatially accurate routing.

![6](https://github.com/user-attachments/assets/014f5c03-d366-4844-a2ac-c65ca91b7b28)

---

## Fire + Egress Panel

| Parameter | Description |
|---|---|
| Graph mode | Cell model (simple) — Topologic spatial graph representation |
| Floor connectivity (m) | Maximum traversal distance between floor nodes. Stair threshold separate. |
| Connectivity (m) | Node-to-node connectivity radius for graph edges |
| Grid cell size (m) | Resolution of the rectilinear grid snapped to IFC geometry |
| Use walls as obstacles | Incorporate IFC wall geometry as navigation barriers |
| Fire steps | Number of simulation steps for fire spread |
| RL episodes / max steps | Reinforcement learning training parameters for path agent |
| Temperature model | Enable thermal propagation in fire simulation |
| Precompute timeline | Pre-bake fire spread timeline for step-through playback |

---

## Local Launch (Windows PowerShell)

### Backend (FastAPI)

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend (Vite)

If Node is installed globally:

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
npm run dev -- --host 0.0.0.0 --port 5173
```

If using the bundled Node in this repo:

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
$nodeDir = "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\node-v24.11.1-win-x64"
$env:Path = "$nodeDir;$env:Path"
& "$nodeDir\npm.cmd" run dev -- --host 0.0.0.0 --port 5173
```

Open in browser:
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

---

## Codebase Overview

### Backend (FastAPI + TopologicPy)
`topologicpy-web-backend/app/main.py` — API server exposing endpoints for health, IFC upload, graph construction, egress pathing, fire simulation (SSE + precompute), and RL training. Core geometry types (Vertex / Edge / Face / Shell / Cluster / Graph / Wire) come from TopologicPy and are used to build navigation graphs and straightened paths.

### IFC Processing and Egress Graph
`main.py` ingests IFC-derived floor and stair point clouds and builds a navigation graph with hybrid adjacency — short edges for stairs, longer for floors. Path endpoints are computed by snapping to nearest graph nodes, then returning path and metadata for frontend overlay.

### Fire Simulation and RL
Fire simulation endpoints (`/fire-sim`, `/fire-sim/stream`) emit time-steps or temperature steps for overlay. The RL endpoint (`/rl/train`) uses graph data to train an agent to find optimal paths under dynamic fire conditions. These feed overlays into the viewer and are designed to complement, not replace, shortest-path routing.

### Frontend (React)
`topologicpy-web-frontend/src/App.jsx` — main UI and state, coordinating file loading, viewer mode switching, slider inputs, and egress/fire controls. Sends IFC egress data to backend, receives graph and path, and pushes overlays into the viewer.

### IFC Viewer (Fragments)
`topologicpy-web-frontend/src/IFCViewer.jsx` — loads IFC with [@thatopen/components](https://github.com/ThatOpen/engine_components), handles selection and picking, and renders overlays (graph wires, paths, fire and temperature). Includes coordinate transforms (up-axis + flips) to align overlays with IFC geometry.

### Topology JSON Viewer
`topologicpy-web-frontend/src/TopologyViewer.jsx` — renders TopologicPy JSON (non-IFC) with its own camera controls and selection logic.

### UI and Styling
`src/styles.css` · `src/index.css` · `src/Sidebar.jsx`

### Build and Deploy
Frontend: Vite — `topologicpy-web-frontend/`  
Backend: Python venv + Uvicorn  
GitHub Pages: `.github/workflows/deploy-frontend.yml`

---

## Research Context

This work sits at the intersection of three problems:

**Graph-based spatial reasoning for AEC.** Topologic represents buildings as cell complexes — spaces as graph nodes, adjacencies and openings as edges. This encodes spatial topology explicitly, not as geometric proximity. Fire egress becomes a graph problem: shortest weighted path under dynamic edge weights driven by fire propagation.

**GraphML in AEC workflows.** The egress graph produced here is directly compatible with graph machine learning pipelines. Node features encode spatial properties (area, floor level, door width). Edge features encode connectivity type (door, stair, corridor). This structure supports GNN-based egress prediction and evacuation optimisation as downstream tasks.

**Reinforcement learning for dynamic evacuation.** Static shortest-path fails when fire is spreading — the optimal path changes as edges become blocked. The RL agent trains on the live simulation, learning policies that adapt to fire state rather than computing a fixed route.

This fork is part of active research into GraphML-based control workflows for the built environment.

---

## Test Model

Tested with `Ifc2x3_Duplex_Architecture.ifc` — the standard IFC reference model from buildingSMART. Any IFC 2x3 or IFC 4 model with defined spaces and doors should work.

---

## Status

- [x] IFC loading and 3D viewer
- [x] Egress graph generation from IFC spatial data
- [x] Shortest path computation with start/exit point picking
- [x] Fire spread simulation with step playback
- [x] RL-based path training
- [ ] Wall obstacles as navigation barriers *(in progress)*
- [ ] Multi-floor path visualisation
- [ ] Export egress graph as GraphML/GML

---

## Related

- [TopologicPy](https://github.com/wassimj/topologicpy) — upstream Python library
- [TopologicStudio](https://github.com/wassimj/TopologicStudio) — upstream browser application
- [Depth_Anything_3_Motifs](https://github.com/libishm1/Depth_Anything_3_Motifs) — companion work on monocular depth reconstruction

---

## Author

Libish Murugesan — researcher and lecturer in Robotics for Architecture, Riyadh.  
[Portfolio](https://libishm1.github.io/portfolio/) · [LinkedIn](https://www.linkedin.com/in/libish-murugesan/) · [GitHub](https://github.com/libishm1)
