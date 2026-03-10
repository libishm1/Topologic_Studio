# Topologic Studio — IFC Fire Egress Simulation

A browser-based research tool for IFC building model analysis, spatial navigation graph generation, fire spread simulation, and dynamic evacuation path computation. Built on [TopologicPy](https://topologic.app) and deployed as a full-stack web application.

**Live demo:** https://libishm1.github.io/Topologic_Studio

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
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage Workflow](#usage-workflow)
8. [Algorithms](#algorithms)
9. [API Reference](#api-reference)
10. [UI Parameters](#ui-parameters)
11. [Feature Status](#feature-status)
12. [Known Limitations](#known-limitations)
13. [Troubleshooting](#troubleshooting)
14. [Detailed Local Deployment](#detailed-local-deployment)
15. [Research Context](#research-context)
16. [References](#references)
17. [License](#license)

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

where `k` is the heat transfer coefficient (default 1.20), adapted from Murugesan and Jabi (2019). The ignition node is held at 120 °C. Temperatures below ambient (20 °C) are shown as blue; higher temperatures map through cyan → green → yellow → red. Fire spread and dynamic path re-routing are streamed and displayed simultaneously.

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

## Installation

### Prerequisites

- Python 3.10 or later
- Node.js 18 or later + npm
- Git

### 1. Clone the repository

```bash
git clone [https://github.com/libishm1/Topologic_Studio.git](https://github.com/libishm1/Topologic_Studio.git)
cd Topologic_Studio
```

### 2. Backend setup

```powershell
cd topologicpy-web-backend

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt

# Install TopologicPy (not on PyPI — install from wheel)
# Download the appropriate wheel from:
# [https://github.com/wassimj/topologicpy/releases](https://github.com/wassimj/topologicpy/releases)
pip install <topologicpy_wheel_file>.whl
```

> **Note:** TopologicPy is not published on PyPI. Download the wheel matching your Python version and OS from the [TopologicPy releases page](https://github.com/wassimj/topologicpy/releases).

### 3. Frontend setup

```powershell
cd ../topologicpy-web-frontend
npm install
```

### 4. Start the servers

**Backend (PowerShell):**
```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Frontend (PowerShell — bundled Node in this repo):**
```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
$nodeDir = "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\node-v24.11.1-win-x64"
$env:Path = "$nodeDir;$env:Path"
& "$nodeDir\npm.cmd" run dev -- --host 0.0.0.0 --port 5173
```

Open `http://localhost:5173` in your browser.

---

## Configuration

### Backend — environment variables

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | *(none)* | Comma-separated extra allowed origins, e.g. `https://myapp.com` |

The backend always allows `http://localhost:5173` and `http://127.0.0.1:5173` by default.

### Frontend — `.env` file

Create `topologicpy-web-frontend/.env`:

```env
VITE_API_BASE=http://localhost:8000
VITE_WEBIFC_WASM_PATH=/wasm/
```

| Variable | Description |
|---|---|
| `VITE_API_BASE` | URL of the FastAPI backend |
| `VITE_WEBIFC_WASM_PATH` | Path to web-ifc WASM files (relative or absolute URL) |

For production: set `VITE_API_BASE` to your deployed backend URL and add that URL to `CORS_ORIGINS`.

---

## Usage Workflow

1. **Load IFC** — Drag and drop or browse to an `.ifc` file. The model appears in the 3D viewer; the scene is immediately navigable (pan/orbit/zoom) while building data extraction continues in the background (shown as a small pulsing badge).

2. **Configure graph** — Choose hybrid or grid-snap (rectilinear) mode. Adjust floor/stair connectivity radius and grid cell size. Toggle "Use walls as obstacles" to incorporate IFC walls as navigation barriers.

3. **Build egress graph** — Click "Build IFC egress graph." The navigation wire graph appears overlaid on the model (blue wires). Stats show node/edge count, door count, and wall segment count.

4. **Set start and exit points** — Click "Pick start point," then click a floor location in the model. Repeat for "Pick exit point." Green dot = start; orange dot = exit.

5. **Compute path** — Click "Compute IFC egress path." The shortest wall-aware path is drawn as a red line through the graph.

6. **Run fire simulation** — Configure fire parameters:
   - **Precompute timeline** — batch-bakes BFS fire spread before playback; fast but does not support temperature mode
   - **Temperature model** — enables the heat-diffusion model; always uses SSE streaming even if "Precompute" is checked
   - **Dynamic path rerouting** — recomputes the shortest path every N steps; requires temperature model + start/exit points set

   Click "Start fire." Graph wires change colour as fire spreads (blue → red gradient in temperature mode; orange in binary mode). With dynamic rerouting enabled, the evacuation path is continuously redrawn in magenta as it avoids expanding fire. Both updates are visible simultaneously in the viewer.

7. **Train RL agent** *(optional)* — Enable "Use fire in RL" and click "Train RL path." The Q-learning agent trains over the configured number of episodes and returns the best path found.

---

## Algorithms

### Graph construction — gap filling (grid-snap mode)

Between any two occupied grid cells on the same axis-aligned column, intermediate cells are filled when the gap is ≤ `max_gap` cells. This bridges the mismatch between coarse IFC surface sampling (~1.5 m) and fine grid cells (0.3–1.5 m). Cells that fall within a wall's bounding volume are not filled during the gap-fill pass.

### Door injection

Each door's bottom-centre is computed from its bounding-box vertex array. Door points are added to the combined point set **before** the agent-height offset, so they receive the same height treatment as floor points. They are then force-connected to the five nearest floor nodes within 3 m.

### Wall obstacle pruning (path-time only)

Each wall's bounding box is collapsed to a 2D centreline segment along the wall's principal horizontal axis. During Dijkstra edge relaxation, an edge is skipped if:
1. Its vertical range overlaps the wall's height range, **and**
2. Its 2D horizontal projection intersects the wall segment (cross-product test).

Door-adjacent edges are always kept. The stored graph adjacency is never modified — only the pathfinding traversal skips these edges.

### Shortest path

Standard Dijkstra with Euclidean edge weights (Dijkstra 1959). Wall checking is performed lazily during edge relaxation, keeping graph build and display free of wall logic.

**Hazard-weighted (fire mode):**
```text
w(u, v) = dist(u, v) × (1 + α × max(T_u, T_v) / 120)
```
Implemented via TopologicPy `Graph.ShortestPath` (Jabi et al. 2019).

### Fire spread — temperature diffusion

Discrete heat-diffusion, adapted from Murugesan and Jabi (2019):
```text
T(n, t+1) = T(n, t) + k × (mean(T(neighbours, t)) − T(n, t))
```
`k = 1.20`, ambient `T₀ = 20 °C`, ignition `T_fire = 120 °C`. The ignition node is held at `T_fire` every step. Temperature is mapped to RGB colour using a five-band gradient: blue (20 °C) → cyan (45 °C) → green (70 °C) → yellow (95 °C) → red (120 °C+).

### Reinforcement learning — tabular Q-learning (Watkins and Dayan 1992)

- **State:** current node ID
- **Actions:** adjacent node IDs (from graph adjacency)
- **Reward:** +100 at exit, −1 per step, −50 for lethal nodes
- **Exploration:** ε-greedy, ε = 0.1
- **Update:** `Q(s,a) ← Q(s,a) + α (r + γ max_a' Q(s',a') − Q(s,a))`
- α = 0.1, γ = 0.95

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/topology` | Process TopologicPy cell complex JSON |
| `POST` | `/ifc-egress-graph` | Build navigation graph from IFC geometry payload |
| `POST` | `/ifc-egress-path` | Compute wall-aware shortest path |
| `POST` | `/fire-sim` | Precompute full fire timeline (non-streaming) |
| `GET` | `/fire-sim/stream` | SSE: fire steps + dynamic path updates |
| `POST` | `/rl/train` | Train Q-learning agent, return best path |

### POST `/ifc-egress-graph` — key request fields

```json
{
  "floors":         [...],   // IFCSLAB geometry payloads (vertices, indices)
  "stairs":         [...],   // IFCSTAIR geometry payloads
  "doors":          [...],   // IFCDOOR geometry payloads
  "walls":          [...],   // IFCWALL geometry payloads
  "use_walls":      true,    // enable wall obstacle extraction
  "up_axis":        "y",     // "x" | "y" | "z"
  "agent_height":   0.75,    // metres above floor surface
  "base_spacing":   0.5,     // point sampling density (m)
  "max_edge_floor": 2.25,    // max edge length on floors (m)
  "max_edge_stair": 0.4,     // max edge length on stairs (m)
  "rectilinear":    false,   // filter diagonal edges in hybrid mode
  "grid_snap":      false,   // use rectilinear grid-snap mode
  "grid_cell_size": 1.5      // grid resolution in metres
}
```

### POST `/ifc-egress-graph` — response

```json
{
  "mode":    "ifc",
  "stats":   { "nodes": 426, "edges": 751, "door_nodes": 14, "wall_segments": 57 },
  "edges":   [ [[x1,y1,z1], [x2,y2,z2]], ... ],   // position pairs for Three.js
  "edge_ids": [ ["ifc_0", "ifc_5"], ... ],          // node-ID pairs for fire colouring
  "coords":  { "ifc_0": [x,y,z], ... }             // node ID → world position
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
| Recompute interval | Steps between path re-computations |
| Lethality threshold (°C) | Nodes above this temperature are avoided if possible |
| Step delay (ms) | Playback speed |
| Fire steps | Total simulation steps |
| RL episodes | Number of training episodes for Q-learning agent |
| RL max steps | Maximum steps per episode |

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
| Fire visualised on graph wires (per-vertex colour) | Done |
| Dynamic path re-routing during fire (SSE) | Done |
| RL path training (Q-learning) | Done |
| Multi-floor path visualisation | In progress |
| Export egress graph as GraphML / GML | Planned |
| Agent / crowd simulation | Planned |
| Smoke and toxicity layer | Planned |

---

## Known Limitations

- **IFC axis convention:** The agent-height offset is applied to the Z axis regardless of `up_axis`. Models with Y-up IFC convention show a slight vertical positional offset on floor nodes; connectivity is not affected.
- **Wall bounding-box accuracy:** Wall obstacle segments are derived from axis-aligned bounding boxes. Non-axis-aligned or curved walls may produce oversized obstacle segments that block more paths than intended.
- **Large models:** Graph construction is single-threaded Python. Models with more than 20,000 sampled points may be slow. Reduce `max_points` in the request to limit graph size.
- **RL scalability:** Tabular Q-learning does not scale beyond ~5,000 nodes. Deep RL (DQN) is planned as a future extension.
- **CORS defaults:** The backend only allows `localhost:5173` by default. Set `CORS_ORIGINS` when deploying to a remote server.

---

## Troubleshooting

### Fire simulation shows no coloured wires

- Ensure you have built the egress graph first (blue wires visible). The fire visualisation requires `edge_ids` returned by `POST /ifc-egress-graph`.
- In temperature mode, the graph wires update per SSE event. If no colour change appears, check that the backend is reachable and the SSE connection opened (browser DevTools → Network → EventStream).
- In binary (BFS) mode, uncheck "Temperature model." With precompute enabled, fire data is fetched in a single POST and animated with a timer.

### Fire works only the first time

- Click "Stop fire" before clicking "Start fire" again. This fully closes the SSE connection and resets accumulated state. Clicking "Start fire" while a simulation is still running may cause the new connection to compete with the old one.

### Graph builds but path is not found

- Pick start and exit points **after** building the graph. Points picked before graph construction are snapped to the graph at compute time; if the graph changes, repick the points.
- If walls are enabled and no path exists (all routes blocked), the pathfinder falls back to the full graph without wall constraints. If the path still fails, reduce the lethality threshold or disable "Use walls."

### IFC model loads but graph has very few nodes

- The default point sampling density is 0.5 m. If the building has small rooms or narrow corridors, decrease "Grid cell size" to 0.3–0.5 m.
- Check that floor slab IFC types are detected: the extractor looks for `IFCSLAB`, `IFCSLABSTANDARDCASE`, `IFCSLABELEMENTEDCASE`, and `IFCCOVERING`. Models that use `IFCPLATE` for floors will need a code extension.

### Backend returns 400 "No IFC egress graph available"

- The graph is held in server memory (`LAST_GRAPHS["ifc"]`). It is lost if the backend process is restarted. Rebuild the graph after each backend restart.

### `TopologicPy` import error on startup

- TopologicPy is not on PyPI. Download the `.whl` matching your Python version and OS from [wassimj/topologicpy releases](https://github.com/wassimj/topologicpy/releases) and install it: `pip install topologicpy-*.whl`.
- On Python 3.12+, some versions of TopologicPy may not yet have a compatible wheel. Python 3.10 or 3.11 is recommended.

### CORS errors in browser console

- Add your frontend origin to `CORS_ORIGINS` in the backend environment (e.g. `CORS_ORIGINS=https://myapp.example.com`).

### web-ifc WASM not loading (model never appears)

- The WASM files must be served from the same origin as the frontend (or via a configured `VITE_WEBIFC_WASM_PATH`). Copy the `node_modules/web-ifc/*.wasm` files to `public/wasm/` and set `VITE_WEBIFC_WASM_PATH=/wasm/` in `.env`.

---

## Detailed Local Deployment

### Backend (FastAPI)
Open a terminal and run the following commands:
```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend (Vite)
If using the bundled Node in this repo, open a *new* terminal and run:
```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
$nodeDir = "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\node-v24.11.1-win-x64"
$env:Path = "$nodeDir;$env:Path"
& "$nodeDir\npm.cmd" run dev -- --host 0.0.0.0 --port 5173
```

**Open Application:**
* **Frontend:** http://localhost:5173

---

## Research Context

This project applies topological spatial reasoning to building fire egress analysis, following the cell complex model introduced by Jabi et al. (2019). Key contributions in this implementation:

- **Browser-native IFC parsing** — no server-side IFC processing; web-ifc WASM runs the IFC parser entirely in the browser
- **Rectilinear grid-snap navigation graph** — preserves architectural alignment of walls, corridors, and stairs; gap-filling ensures connectivity despite sparse IFC surface sampling
- **Simultaneous fire and path visualisation** — fire spread and re-routed evacuation path are streamed concurrently over SSE and rendered as per-vertex colours on Three.js `LineSegments`
- **Hazard-weighted Dijkstra** integrating the temperature field from Murugesan and Jabi (2019) directly into edge costs
- **GraphML-compatible graph output** (node/edge coordinate and ID features) — suitable for downstream GNN-based evacuation research

This tool builds on prior work applying spatial graphs and topological representations to evacuation modelling in architecture and urban design (Hillier and Hanson 1984; Turner and Penn 2002; Pan et al. 2006).

---

## References

### Topologic / TopologicPy

Jabi, Wassim, Aikaterini Chatzivasileiadi, Njegos M. Wardhana, Simon Lannon, and Robert Aish. 2019. "The Role of Topological Computation in the Generation of Building Spatial Structures." *Architectural Science Review* 62 (5): 429–440. https://doi.org/10.1080/00038628.2019.1651606

Jabi, Wassim. 2022. *TopologicPy: A Python Library for Topological Spatial Modelling.* https://topologic.app

### IFC Standard

BuildingSMART International. 2024. *IFC4 ADD2 TC1 — Industry Foundation Classes.* https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2_TC1/HTML/

ISO 16739-1:2018. *Industry Foundation Classes (IFC) for Data Sharing in the Construction and Facility Management Industries.* Geneva: International Organization for Standardization.

### Web IFC / BIM Components

That Open Company. 2024a. *web-ifc: IFC Parsing in the Browser.* https://github.com/ThatOpenCompany/web-ifc

That Open Company. 2024b. *Open BIM Components (@thatopen/components).* https://github.com/ThatOpenCompany/engine_components

### Fire Spread Model

Murugesan, Libish, and Wassim Jabi. 2019. "Spatial Graph-Based Fire Spread Simulation for Building Evacuation." In *Proceedings of eCAADe 37*, Porto, Portugal.

Drysdale, Dougal. 2011. *An Introduction to Fire Dynamics.* 3rd ed. Chichester: Wiley. https://doi.org/10.1002/9781119975465

### Egress and Evacuation Modelling

Pan, Xiaoshan, Charles S. Han, Keith Dauber, and Kincho H. Law. 2006. "A Multi-Agent Based Framework for the Simulation of Human and Social Behaviors during Emergency Evacuations." *AI & Society* 22 (2): 113–132. https://doi.org/10.1007/s00146-007-0126-1

Kuligowski, Erica D., Richard D. Peacock, and Bryan L. Hoskins. 2010. *A Review of Building Evacuation Models.* 2nd ed. NIST Technical Note 1680. Gaithersburg, MD: National Institute of Standards and Technology. https://doi.org/10.6028/NIST.TN.1680

### Space Syntax / Spatial Graph Theory

Hillier, Bill, and Julienne Hanson. 1984. *The Social Logic of Space.* Cambridge: Cambridge University Press.

Turner, Alasdair, and Alan Penn. 2002. "Encoding Natural Movement as an Agent-Based System: An Investigation into Human Pedestrian Behaviour in the Built Environment." *Environment and Planning B* 29 (4): 473–490. https://doi.org/10.1068/b12850

### Pathfinding

Dijkstra, Edsger W. 1959. "A Note on Two Problems in Connexion with Graphs." *Numerische Mathematik* 1 (1): 269–271. https://doi.org/10.1007/BF01386390

Hart, Peter E., Nils J. Nilsson, and Bertram Raphael. 1968. "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics* 4 (2): 100–107. https://doi.org/10.1109/TSSC.1968.300136

### Reinforcement Learning

Watkins, Christopher J. C. H., and Peter Dayan. 1992. "Q-Learning." *Machine Learning* 8 (3–4): 279–292. https://doi.org/10.1007/BF00992698

Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, et al. 2015. "Human-Level Control through Deep Reinforcement Learning." *Nature* 518: 529–533. https://doi.org/10.1038/nature14236

### Three.js

Cabello, Ricardo, et al. 2024. *Three.js — JavaScript 3D Library.* https://threejs.org

---

## License

MIT License

Copyright (c) 2024 Libish Murugesan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
