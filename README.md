# Topologic Studio — IFC Fire Egress Simulation

A browser-based research tool for IFC building model analysis, spatial navigation graph generation, fire spread simulation, and dynamic evacuation path computation. Built on [TopologicPy](https://topologic.app) and deployed as a full-stack web application.

**Live demo:** https://libishm1.github.io/Topologic_Studio

> **Screenshot** — rectilinear grid-snap graph (1866 nodes / 1552 edges), 14 door waypoints, 57 wall segments, temperature fire simulation at step 49 with dynamic path rerouting (cost 27.48 m), all rendered simultaneously in the browser:
>
> ![Topologic Studio — fire egress simulation screenshot](docs/screenshot.png)

---

## Image Series

### 0. Full Workflow View
Overall browser view showing IFC model loading, graph overlay, fire simulation, and egress pathing in one interface.
![hero](https://github.com/user-attachments/assets/9401488b-7b58-4a80-9db7-c6614dc8cab9)

### 1. Load IFC
Import any IFC file into the browser-based 3D viewer.
![preview](https://github.com/user-attachments/assets/cb098eef-8260-4154-838c-c8c28f7e4edc)

### 2. Build Egress Graph
Generate a spatial navigation graph from the IFC model. Topologic converts rooms, doors, stairs, and corridors into graph nodes and edges.
![2](https://github.com/user-attachments/assets/24184ffb-fb93-41b9-9c83-640becd1aa73)
*Graph: 1,866 nodes / 1,552 edges · 14 doors · 57 walls · 21 floors / 4 stairs*

### 3. Set Start and Exit Points
Click to define the fire origin and egress exit directly on the 3D model.
![3](https://github.com/user-attachments/assets/7caa0561-e433-44fc-952a-6839715c4410)
![4](https://github.com/user-attachments/assets/9cf129d2-078f-446a-a221-dd835f3d4348)

### 4. Compute Egress Path
Run shortest-path computation. The path is highlighted in red through the building graph.
![5](https://github.com/user-attachments/assets/f4b4c2ac-d848-442f-96d8-25d60687571f)

### 5. Wall Obstacles (WIP)
Toggle IFC walls as navigation obstacles for more spatially accurate routing.
![6](https://github.com/user-attachments/assets/014f5c03-d366-4844-a2ac-c65ca91b7b28)

---

## Table of Contents

1. [Overview](#overview)
2. [What This Project Does](#what-this-project-does)
3. [Tech Stack](#tech-stack)
4. [Architecture](#architecture)
5. [Installation & Detailed Local Deployment (Windows)](#installation--detailed-local-deployment-windows)
6. [Configuration](#configuration)
7. [Usage Workflow](#usage-workflow)
8. [Algorithms](#algorithms)
9. [API Reference](#api-reference)
10. [UI Parameters](#ui-parameters)
11. [Feature Status](#feature-status)
12. [Known Limitations](#known-limitations)
13. [Troubleshooting](#troubleshooting)
14. [Research Context](#research-context)
15. [References](#references)
16. [License](#license)

---

## Overview

Topologic Studio extends [TopologicPy](https://topologic.app) with an IFC-native fire egress workflow. Given any IFC building model, the tool:

- Parses floor slabs, stairs, doors, and walls from IFC geometry in the browser
- Constructs a navigable spatial graph across all floors and stairwells
- Computes the shortest evacuation path between any two points with wall obstacle avoidance
- Simulates fire spreading through the graph (temperature diffusion or BFS model)
- Re-routes the evacuation path in real-time as fire spreads, streamed over SSE
- Visualises fire spread as a blue-to-red colour gradient directly on the navigation wires
- Trains a reinforcement learning agent to find escape routes under dynamic fire conditions

All computation runs in a Python/FastAPI backend; visualisation runs in a React/Three.js browser viewer with no plugin required.

---

## What This Project Does

### IFC Loading and Parsing

IFC models (`.ifc`) are loaded entirely in the browser via the [web-ifc](https://github.com/ThatOpenCompany/web-ifc) WASM library (part of [@thatopen/components](https://github.com/ThatOpenCompany/engine_components)). Loading is non-blocking — the model is navigable (pan, orbit, zoom) immediately after the fragment geometry appears, while IFC element extraction continues in the background.

The following IFC element types are extracted for navigation:

| IFC Type | Role |
|---|---|
| `IFCSLAB` | Floor and ceiling surfaces — sampled for walkable floor points |
| `IFCSTAIR` | Stair geometry — sampled at fine vertical resolution (≤ 0.15 m per tread) |
| `IFCDOOR` | Door openings — injected as forced waypoints in the navigation graph |
| `IFCWALL` / `IFCWALLSTANDARDCASE` | Wall centrelines — used as path obstacles during traversal |

Vertex geometry is extracted as flat float arrays, transformed by the model world matrix, and sent to the backend.

### Navigation Graph Generation

Two modes are available:

**Hybrid (distance-based):** Points sampled from floor and stair surfaces are connected when within a user-defined distance threshold. An optional rectilinear filter removes diagonal connections by comparing the horizontal minor/major extent ratio of each edge.

**Grid-snap (rectilinear):** Sampled points are snapped to a regular voxel grid. Only the six cardinal directions (±x, ±y, ±z) are connected — no diagonals. A gap-filling pass bridges cells separated by sparse sampling. Stairs use a fine vertical cell size (default 0.15 m) to capture individual treads.

Door positions (extracted as bounding-box bottom-centres) are appended to the full point set before the agent-height offset is applied, ensuring they sit at the same navigable height as floor nodes. Each door is then force-connected to its five nearest floor neighbours within 3 m.

### Wall Obstacles

Wall centrelines are extracted as 2D axis-aligned segments from IFC wall bounding boxes. Wall obstacles are applied **only during path computation** — the full graph is always displayed unchanged. During Dijkstra traversal, any edge whose horizontal projection intersects a wall segment is skipped. Door-adjacent edges are exempt from this rule, since doors are openings through walls.

### Fire Spread Simulation

Two models are supported:

**Binary (BFS) model:** Fire spreads breadth-first from the ignition node, one graph neighbourhood per step. All nodes reached so far are accumulated and shown as orange-red wires.

**Temperature diffusion model:** Each node holds a temperature value T. At every step, heat transfers from hot neighbours:

```text
T(n, t+1) = T(n, t) + k × (mean(T(neighbours, t)) − T(n, t))
```

where `k` is the heat transfer coefficient (default 1.20), adapted from (Murugesan and Jabi 2019). The ignition node is held at 120 °C. Temperatures below ambient (20 °C) are shown as blue; higher temperatures map through cyan → green → yellow → red. Fire spread and dynamic path re-routing are streamed and displayed simultaneously.

### Dynamic Path Re-routing

During temperature-mode fire simulation, the evacuation path is recomputed every N steps using hazard-weighted shortest-path via TopologicPy's `Graph.ShortestPath`. Edge weights are:

```text
w = distance × (1 + α × max(T_a, T_b) / T_ref)
```

where `α` (hazard weight) is user-configurable (default 1.4). If a lethality temperature threshold is set, a filtered graph (excluding edges above threshold) is tried first; the full graph is used as fallback. The re-routed path is drawn as a magenta line in the viewer, concurrent with the fire colour update.

### Reinforcement Learning

A tabular Q-learning agent (Watkins and Dayan 1992) is trained on-server to navigate from a start node to an exit node while fire spreads. The learned policy path is returned as a coordinate polyline and displayed in the viewer.

---

## Tech Stack

### Backend

| Library | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime |
| [FastAPI](https://fastapi.tiangolo.com) | REST + Server-Sent Events API |
| [Uvicorn](https://www.uvicorn.org) | ASGI server |
| [Pydantic v2](https://docs.pydantic.dev) | Request/response validation |
| [TopologicPy](https://topologic.app) | Cell complex graph, `Graph.ShortestPath`, edge weights |
| python-multipart | File upload support |

### Frontend

| Library | Purpose |
|---|---|
| [React 18](https://react.dev) | UI framework |
| [Vite 7](https://vite.dev) | Build tool and dev server |
| [Three.js](https://threejs.org) | 3D rendering, `BufferGeometry`, `vertexColors` |
| [@thatopen/components](https://github.com/ThatOpenCompany/engine_components) | IFC Fragments viewer engine |
| [@thatopen/fragments](https://github.com/ThatOpenCompany/engine_fragments) | IFC fragment worker and geometry extraction |
| [web-ifc](https://github.com/ThatOpenCompany/web-ifc) | IFC WASM parser (runs in browser) |
| [Axios](https://axios-http.com) | HTTP client |

---

## Architecture

```text
Browser (React + Vite)
│
├── IFCViewer.jsx     — Three.js scene, IFC fragment loader, navigation graph
│                       wire overlay (per-vertex fire colours), path lines,
│                       raycasting for point picking
├── App.jsx           — Application state, fire/path controls, SSE consumer,
│                       graph/edge data management
└── TopologyViewer.jsx — Generic TopologicPy JSON renderer
        │
        │  REST (JSON)  +  SSE (text/event-stream)
        ▼
FastAPI backend  (topologicpy-web-backend/app/main.py)
│
├── POST /ifc-egress-graph   — Build navigation graph from IFC geometry
├── POST /ifc-egress-path    — Compute wall-aware shortest path
├── GET  /fire-sim/stream    — SSE: fire spread + dynamic path updates
├── POST /fire-sim           — Precomputed fire timeline (non-streaming)
├── POST /rl/train           — Train Q-learning agent, return best path
└── POST /topology           — TopologicPy cell complex operations
```

REST is used for graph and path requests (synchronous, JSON). [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) are used for real-time fire simulation streaming — the connection stays open and the server pushes temperature and path events as they are computed.

---

## Installation & Detailed Local Deployment (Windows)

This section covers first-time setup and local launch.

### Requirements
- Windows 10 or 11
- Python 3.10+
- Node.js 18+ (or bundled Node runtime)
- Git
- PowerShell

### Expected Project Layout
```text
TopologicStudio/
├── topologicpy-web-backend/
├── topologicpy-web-frontend/
└── node-v24.11.1-win-x64/   # optional bundled Node runtime
```

### 1) Clone
```powershell
git clone [https://github.com/libishm1/Topologic_Studio.git](https://github.com/libishm1/Topologic_Studio.git)
cd Topologic_Studio
```

### 2) Backend Setup
```powershell
cd topologicpy-web-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Install TopologicPy wheel (required for this project):
# Download from: [https://github.com/wassimj/topologicpy/releases](https://github.com/wassimj/topologicpy/releases)
pip install path\to\TopologicPy-<version>-<pyver>-<platform>.whl
```

### 3) Frontend Setup
Open a **new** PowerShell terminal:
```powershell
cd topologicpy-web-frontend
```

**Option A: System Node**
```powershell
npm install
```

**Option B: Bundled Node**
```powershell
$nodeDir = "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\node-v24.11.1-win-x64"
$env:Path = "$nodeDir;$env:Path"
& "$nodeDir\npm.cmd" install
```

### 4) Run Backend
```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5) Run Frontend
**Option A: System Node**
```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
npm run dev -- --host 0.0.0.0 --port 5173
```

**Option B: Bundled Node**
```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
$nodeDir = "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\node-v24.11.1-win-x64"
$env:Path = "$nodeDir;$env:Path"
& "$nodeDir\npm.cmd" run dev -- --host 0.0.0.0 --port 5173
```

### 6) Open
* **Frontend:** http://localhost:5173
* **Backend:** http://localhost:8000
* **Backend docs:** http://localhost:8000/docs

---

## Configuration

### Frontend
`src/App.jsx` uses:
```javascript
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
```

Create `topologicpy-web-frontend/.env` for local override:
```env
VITE_API_BASE=http://localhost:8000
VITE_WEBIFC_WASM_PATH=/wasm/
```

### Backend CORS
`app/main.py` allows localhost frontend origins by default (`http://localhost:5173`, `http://127.0.0.1:5173`). You can add more via environment variables:
```env
CORS_ORIGINS=[https://your-frontend.example.com](https://your-frontend.example.com)
```

---

## Usage Workflow

1. **Load IFC** — Drag and drop or browse to an `.ifc` file. The model appears in the 3D viewer; the scene is immediately navigable.
2. **Configure graph** — Choose hybrid or grid-snap (rectilinear) mode. Toggle "Use walls as obstacles".
3. **Build egress graph** — Click "Build IFC egress graph." The navigation wire graph appears overlaid on the model (blue wires).
4. **Set start and exit points** — Click "Pick start point," then click a floor location in the model. Repeat for "Pick exit point." 
5. **Compute path** — Click "Compute IFC egress path."
6. **Run fire simulation** — Configure fire parameters (Precompute vs Temperature model, dynamic routing) and click "Start fire."
7. **Train RL agent** *(optional)* — Enable "Use fire in RL" and click "Train RL path." 

---

## Algorithms

### Graph construction — gap filling (grid-snap mode)
Between any two occupied grid cells on the same axis-aligned column, intermediate cells are filled when the gap is ≤ `max_gap` cells. This bridges the mismatch between coarse IFC surface sampling (~1.5 m) and fine grid cells (0.3–1.5 m). 

### Door injection
Each door's bottom-centre is computed from its bounding-box vertex array. Door points are added to the combined point set **before** the agent-height offset, so they receive the same height treatment as floor points. They are then force-connected to the five nearest floor nodes within 3 m.

### Wall obstacle pruning (path-time only)
Each wall's bounding box is collapsed to a 2D centreline segment along the wall's principal horizontal axis. During Dijkstra edge relaxation, an edge is skipped if its vertical range overlaps the wall's height range **and** its 2D horizontal projection intersects the wall segment.

### Fire spread — temperature diffusion
Discrete heat-diffusion, adapted from (Jabi et al. 2019):
```text
T(n, t+1) = T(n, t) + k * (mean(T(neighbours, t)) - T(n, t))
```
where `k = 1.20`, ambient `T_0 = 20°C`, and ignition `T_fire = 120°C`. Temperature maps to RGB color via a five-band gradient (blue to red).

### Dynamic Path Re-routing
Standard Dijkstra is augmented with hazard-weighted edge costs during temperature-mode fire simulation:
```text
w = distance * (1 + α * max(T_a, T_b) / T_ref)
```
where `α` is a user-configurable hazard weight.

### Reinforcement learning — tabular Q-learning
- **State:** current node ID
- **Actions:** adjacent node IDs
- **Reward:** +100 at exit, −1 per step, −50 for lethal nodes
- **Exploration:** ε-greedy, ε = 0.1
- **Update:**
```text
Q(s,a) = Q(s,a) + α(r + γ * max(Q(s',a')) - Q(s,a))
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/topology` | Process TopologicPy cell complex JSON |
| `POST` | `/upload-ifc` | Parses IFC for viewer graph/topology data upload flow |
| `POST` | `/ifc-egress-graph` | Build navigation graph from IFC geometry payload |
| `POST` | `/ifc-egress-path` | Compute wall-aware shortest path |
| `POST` | `/fire-sim` | Precompute full fire timeline (non-streaming) |
| `GET` | `/fire-sim/stream` | SSE: fire steps + dynamic path updates |
| `POST` | `/rl/train` | Train Q-learning agent, return best path |

### POST `/ifc-egress-graph` — key request fields
```json
{
  "floors":         [{"expressID": 1, "vertices": [], "indices": []}],
  "stairs":         [],
  "doors":          [],
  "walls":          [],
  "use_walls":      true,
  "up_axis":        "y",
  "agent_height":   0.75,
  "base_spacing":   0.5,
  "max_edge_floor": 2.25,
  "max_edge_stair": 0.4,
  "rectilinear":    false,
  "grid_snap":      false,
  "grid_cell_size": 1.5
}
```

### POST `/ifc-egress-graph` — response
```json
{
  "mode":    "ifc",
  "stats":   { "nodes": 426, "edges": 751, "door_nodes": 14, "wall_segments": 57 },
  "edges":   [ [[x1,y1,z1], [x2,y2,z2]] ],
  "edge_ids": [ ["ifc_0", "ifc_5"] ],
  "coords":  { "ifc_0": [x,y,z] }
}
```

### GET `/fire-sim/stream` — SSE event types
| `type` | Fields | Description |
|---|---|---|
| `meta` | `cell_bboxes` | Sent once at start |
| `temperature_step` | `step`, `temperatures: {nodeId: °C}` | Per-step temperature map |
| `path_update` | `step`, `path`, `cost`, `changed` | Re-routed path coordinates |
| `step` | `step`, `nodes: [nodeId]` | Newly burning nodes (binary model) |
| `done` | — | Simulation complete |

---

## UI Parameters

| Parameter | Description |
|---|---|
| Floor connectivity (m) | Maximum edge length between floor nodes (hybrid mode) |
| Stair connectivity (m) | Maximum edge length between stair nodes |
| Grid-snap (rectilinear) | Use regular voxel grid instead of distance-based graph |
| Grid cell size (m) | Voxel resolution for grid-snap mode |
| Use walls as obstacles | Incorporate IFC wall centrelines as path barriers |
| Precompute timeline | Pre-bake fire spread before playback (batch mode) |
| Temperature model | Enable thermal diffusion model instead of BFS spread |
| Dynamic path rerouting | Re-route evacuation path every N steps as fire spreads |
| Hazard weight (α) | How strongly temperature penalises edge cost |
| Lethality threshold (°C) | Nodes above this temperature are avoided if possible |
| Step delay (ms) | Playback speed |

---

## Feature Status

| Feature | Status |
|---|---|
| IFC loading and 3D viewer (non-blocking) | Done |
| Egress graph — hybrid (distance-based) | Done |
| Egress graph — grid-snap (rectilinear) | Done |
| Shortest path — wall-aware Dijkstra | Done |
| Wall obstacles (path-time 2D segment intersection) | Done |
| Door waypoints (forced graph nodes) | Done |
| Fire spread — binary BFS model | Done |
| Fire spread — temperature diffusion model | Done |
| Dynamic path re-routing during fire (SSE) | Done |
| RL path training (Q-learning) | Done |
| Export egress graph as GraphML / GML | Planned |
| Agent / crowd simulation | Planned |

---

## Known Limitations

- **IFC axis convention:** The agent-height offset is applied to the Z axis regardless of `up_axis`. Models with Y-up IFC convention show a slight vertical positional offset on floor nodes; connectivity is not affected.
- **Wall bounding-box accuracy:** Wall obstacle segments are derived from axis-aligned bounding boxes. Non-axis-aligned or curved walls may produce oversized obstacle segments.
- **Large models:** Graph construction is single-threaded Python. Models with more than 20,000 sampled points may be slow. 
- **RL scalability:** Tabular Q-learning does not scale beyond ~5,000 nodes.
- **Fire Simulation:** Fire simulation is graph-based topology, not CFD-based.

---

## Troubleshooting

### `npm` not recognized
Use bundled Node runtime commands from this README, or install Node LTS globally.

### Port 8000 already in use
```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Backend says no IFC graph available
Build graph first via `POST /ifc-egress-graph` (or UI button) before path/fire/RL calls.

### Frontend cannot reach backend
- Confirm backend is running at `http://localhost:8000`
- Confirm `.env` has `VITE_API_BASE=http://localhost:8000`

### TopologicPy import error
- Ensure TopologicPy wheel is installed in the active `.venv`
- **Critical:** If you are not using the original developer path, remove or update the hardcoded `sys.path.append(...)` line in `topologicpy-web-backend/app/main.py`.

### IFC loads but graph is sparse
Increase floor/stair connectivity sliders or use smaller grid cell size in grid-snap mode.

---

## Research Context

Topologic Studio sits within a line of research that treats building interiors as topological and graph-based spatial structures rather than only as geometric meshes or BIM objects. Topologic and related work showed that non-manifold topology and cell-complex reasoning can support richer architectural representations of adjacency, enclosure, circulation, and navigation (Aish et al. 2018; Jabi et al. 2018). This project adopts that view at the application level by converting IFC-derived floors, stairs, and doors into a navigation graph that can be queried and visualized in real time.

The graph-generation layer is also grounded in research on BIM-to-network conversion for indoor navigation and emergency response. That literature established approaches that convert building interiors into traversable graphs for indoor routing and emergency response (Boguslawski 2011; Liu and Zlatanova 2011; Isikdag, Zlatanova, and Underwood 2013; Boguslawski et al. 2015, 2016a, 2016b). The current implementation extends that direction through a browser-based workflow, explicit door-node injection, stair-aware sampling, and a rectilinear grid-snap strategy designed to preserve architectural alignment.

At the routing level, the project follows research on risk-aware and balanced route selection in hazardous buildings (Duckham and Kulik 2003; Park et al. 2009; Vanclooster et al. 2014; Zverovich et al. 2016, 2017). The path solver begins with Dijkstra shortest path, then adds a hazard-weighted cost term so the route can adapt as local graph temperatures change over time.

At the fire and learning level, the most relevant precedent is the study on the synergy of non-manifold topology and reinforcement learning for fire egress (Jabi et al. 2019). In this README, that paper is used as research context for the broader idea of graph-based fire propagation and adaptive route selection. It should not be read as a claim that this repository reproduces the same implementation in full. The current codebase, as documented in the repository, exposes shortest-path routing, streamed or precomputed fire overlays, and an RL training endpoint inside a browser-first IFC workflow with live visualization and server-side updates.

Taken together, the project should be read as a research prototype for IFC-native evacuation analysis rather than as a final validated fire-engineering simulator. Its value lies in linking architectural representation, graph construction, hazard-aware routing, and interactive visualization inside one reproducible software stack.

---

## References

**Graph-based spatial reasoning / Cell complexes / Topologic**
* Aish, R., Jabi, W., Lannon, S., Wardhana, N.M. and Chatzivasileiadi, A. 2018. "Topologic: Tools to Explore Architectural Topology." *Proceedings of eCAADe 2018*.
* Boguslawski, P. 2011. *Modelling and Analysing 3D Building Interiors with the Dual Half-Edge Data Structure*. PhD thesis, University of Glamorgan.
* Jabi, W., Aish, R., Lannon, S., Chatzivasileiadi, A. and Wardhana, N.M. 2018. "Topologic: Enhancing the Representation of Space in 3D Modelling Environments through Non-Manifold Topology." *Proceedings of eCAADe 2018*.
* Kwan, M.-P. and Lee, J. 2005. "Emergency Response after 9/11: The Potential of Real-Time 3D GIS for Quick Emergency Response in Micro-Spatial Environments." *Computers, Environment and Urban Systems* 29: 93–113.

**Navigable network generation from BIM**
* Boguslawski, P., Mahdjoubi, L., Zverovich, V., Barki, H. and Fadli, F. 2015. "BIM-GIS Modelling in Support of Emergency Response Applications." In *Building Information Modelling (BIM) in Design, Construction and Operations*, vol. 149, edited by L. Mahdjoubi, C. Brebbia, and R. Laing, 381–392. WIT Press.
* Boguslawski, P., Mahdjoubi, L., Zverovich, V. and Fadli, F. 2016a. "Automated Construction of Variable Density Navigable Networks in a 3D Indoor Environment for Emergency Response." *Automation in Construction* 72 (2): 115–128.
* Boguslawski, P., Mahdjoubi, L., Zverovich, V. and Fadli, F. 2016b. "Two-Graph Building Interior Representation for Emergency Response Applications." *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences* III-2: 9–14.
* Isikdag, U., Zlatanova, S. and Underwood, J. 2013. "A BIM-Oriented Model for Supporting Indoor Navigation Requirements." *Computers, Environment and Urban Systems* 41: 112–123.
* Liu, L. and Zlatanova, S. 2011. "A 'Door-to-Door' Path-Finding Approach for Indoor Navigation." In *Gi4DM 2011: GeoInformation for Disaster Management*. ISPRS.

**Shortest / safest / optimal path computation**
* Duckham, M. and Kulik, L. 2003. "'Simplest' Paths: Automated Route Selection for Navigation." In *Spatial Information Theory*, edited by W. Kuhn, M.F. Worboys, and S. Timpf, 169–185. Springer.
* Park, I., Jang, G., Park, S. and Lee, J. 2009. "Time-Dependent Optimal Routing in Micro-Scale Emergency Situation." In *Tenth International Conference on Mobile Data Management*, 714–719.
* Vanclooster, A., De Maeyer, P., Fack, V. and Van de Weghe, N. 2014. "Calculating Least Risk Paths in 3D Indoor Space." In *Innovations in 3D Geo-Information Sciences*, edited by U. Isikdag, 13–31. Springer.
* Zverovich, V., Mahdjoubi, L., Boguslawski, P., Fadli, F. and Barki, H. 2016. "Emergency Response in Complex Buildings: Automated Selection of Safest and Balanced Routes." *Computer-Aided Civil and Infrastructure Engineering* 31 (8): 617–632.
* Zverovich, V., Mahdjoubi, L., Boguslawski, P. and Fadli, F. 2017. "Analytic Prioritization of Indoor Routes for Search and Rescue Operations in Hazardous Environments." *Computer-Aided Civil and Infrastructure Engineering* 32 (9): 727–747.

**Fire spread simulation & RL-based path training**
* Jabi, W., Chatzivasileiadi, A., Wardhana, N.M., Lannon, S. and Aish, R. 2019. "The Synergy of Non-Manifold Topology and Reinforcement Learning for Fire Egress." In *Proceedings of eCAADe 37 / SIGraDi 23*, vol. 2, 85–94.
* Sutton, R. and Barton, A. 2018. *Reinforcement Learning: An Introduction*. MIT Press.
* Thombre, P. 2018. *Multi-Objective Path Finding Using Reinforcement Learning*. Master's thesis, San Jose State University.

**Agent-based evacuation simulation / dynamic density**
* Nelson, H.E. and MacLennan, H.A. 1995. "Emergency Movement." In *SFPE Handbook of Fire Protection Engineering*, edited by P.J. DiNenno, 3.286–3.295. NFPA.
* Pauls, J. 1995. "Movement of People." In *SFPE Handbook of Fire Protection Engineering*, edited by P.J. DiNenno, 3.263–3.285. NFPA.

**Multi-criteria decision making (AHP for route ranking)**
* Forcael, E., González, V., Orozco, F., Vargas, S., Pantoja, A. and Moscoso, P. 2014. "Ant Colony Optimization Model for Tsunamis Evacuation Routes." *Computer-Aided Civil and Infrastructure Engineering* 29 (10): 723–737.
* Saaty, T.L. 1980. *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*. McGraw-Hill.

---

## License and Third-Party Licenses

Primary project license: MIT.

Third-party components include:
- **TopologicPy** (AGPL-3.0)
- **@thatopen/components** (MIT)
- **web-ifc** (MIT)
- **Three.js** (MIT)
- **FastAPI** (MIT)
- **React** (MIT)

If you deploy network-accessible services with AGPL components (like TopologicPy), please review your AGPL obligations.

---

**Author:** Libish Murugesan  
Researcher and Lecturer in Computational Architecture and Robotics for Architecture  
Alfaisal University, Riyadh, Saudi Arabia  
GitHub: [@libishm1](https://github.com/libishm1)
