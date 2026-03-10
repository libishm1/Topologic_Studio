# Topologic Studio - IFC Fire Egress Simulation

A browser-based research tool for IFC building model analysis, spatial navigation graph generation, fire spread simulation, and dynamic evacuation path computation. Built on [TopologicPy](https://topologic.app) and deployed as a full-stack web application.

**Live demo:** https://libishm1.github.io/Topologic_Studio

Topologic Studio combines browser-native IFC parsing, graph-based indoor navigation, hazard-aware routing, and streamed fire visualization in a single research workflow. The project builds on prior work in topological spatial modelling, BIM-derived navigable networks, safest-route computation, and reinforcement learning for fire egress.

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
5. [Installation & Detailed Local Deployment](#installation--detailed-local-deployment)
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

- parses floor slabs, stairs, doors, and walls from IFC geometry in the browser
- constructs a navigable spatial graph across all floors and stairwells
- computes the shortest evacuation path between any two points with wall obstacle avoidance
- simulates fire spreading through the graph using either a binary BFS model or a temperature diffusion model
- re-routes the evacuation path in real time as fire spreads, streamed over SSE
- visualizes fire spread as a blue-to-red colour gradient directly on the navigation graph
- trains a reinforcement learning agent to find escape routes under dynamic fire conditions

All computation runs in a Python/FastAPI backend. Visualization runs in a React/Three.js browser viewer with no plugin required.

---

## What This Project Does

### IFC Loading and Parsing

IFC models (`.ifc`) are loaded entirely in the browser via the [web-ifc](https://github.com/ThatOpenCompany/web-ifc) WASM library, used through [@thatopen/components](https://github.com/ThatOpenCompany/engine_components). Loading is non-blocking. The model is navigable as soon as fragment geometry appears, while IFC element extraction continues in the background.

The following IFC element types are extracted for navigation:

| IFC Type | Role |
|---|---|
| `IFCSLAB` | Floor and ceiling surfaces sampled for walkable floor points |
| `IFCSTAIR` | Stair geometry sampled at fine vertical resolution |
| `IFCDOOR` | Door openings injected as forced waypoints in the navigation graph |
| `IFCWALL` / `IFCWALLSTANDARDCASE` | Wall centrelines used as path obstacles during traversal |

Vertex geometry is extracted as flat float arrays, transformed by the model world matrix, and sent to the backend.

### Navigation Graph Generation

Two modes are available.

**Hybrid (distance-based):** Points sampled from floor and stair surfaces are connected when within a user-defined distance threshold. An optional rectilinear filter removes diagonal connections by comparing the horizontal minor-to-major extent ratio of each edge.

**Grid-snap (rectilinear):** Sampled points are snapped to a regular voxel grid. Only the six cardinal directions (`±x`, `±y`, `±z`) are connected. A gap-filling pass bridges cells separated by sparse sampling. Stairs use a fine vertical cell size to capture individual treads.

Door positions, extracted as bounding-box bottom centres, are appended to the point set before the agent-height offset is applied. Each door is then force-connected to its nearest floor neighbours.

### Wall Obstacles

Wall centrelines are extracted as 2D axis-aligned segments from IFC wall bounding boxes. Wall obstacles are applied during path computation, while the full graph remains visible in the viewer. During Dijkstra traversal, any edge whose horizontal projection intersects a wall segment is skipped. Door-adjacent edges are exempt because doors are treated as openings through walls.

### Fire Spread Simulation

Two models are supported.

**Binary (BFS) model:** Fire spreads breadth-first from the ignition node, one graph neighbourhood per step. All nodes reached so far are accumulated and shown as orange-red wires.

**Temperature diffusion model:** Each node holds a temperature value `T`. At every step, heat transfers from hot neighbours:

```text
T(n, t+1) = T(n, t) + k × (mean(T(neighbours, t)) - T(n, t))
```

where k is the heat transfer coefficient, adapted from (Jabi et al. 2019). The ignition node is held at a constant fire temperature. Temperatures below ambient are shown as blue, and higher temperatures map through cyan, green, yellow, and red. Fire spread and dynamic path re-routing are streamed and displayed simultaneously.

### Dynamic Path Re-routing

During temperature-mode fire simulation, the evacuation path is recomputed at a user-defined interval using hazard-weighted shortest path through TopologicPy's Graph.ShortestPath. Edge weights are:

```text
w = distance × (1 + α × max(T_a, T_b) / T_ref)
```

where α is a user-configurable hazard weight. If a lethality threshold is set, the system first tries a filtered graph that excludes edges above the threshold, then falls back to the full graph if needed.

### Reinforcement Learning

A tabular Q-learning agent is trained on the server to navigate from a start node to an exit node while fire spreads. The learned policy path is returned as a coordinate polyline and displayed in the viewer.

---

## Tech Stack

### Backend
| Library | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime |
| FastAPI | REST and Server-Sent Events API |
| Uvicorn | ASGI server |
| Pydantic v2 | Request and response validation |
| TopologicPy | Cell complex graph operations and weighted shortest path |
| python-multipart | File upload support |

### Frontend
| Library | Purpose |
|---|---|
| React 18 | UI framework |
| Vite 7 | Build tool and dev server |
| Three.js | 3D rendering and line-based graph display |
| @thatopen/components | IFC fragments viewer engine |
| @thatopen/fragments | IFC fragment worker and geometry extraction |
| web-ifc | IFC WASM parser running in the browser |
| Axios | HTTP client |

---

## Architecture

```text
Browser (React + Vite)
│
├── IFCViewer.jsx      -> Three.js scene, IFC fragment loader, navigation graph,
│                         fire-color overlay, path lines, point picking
├── App.jsx            -> application state, controls, SSE consumer,
│                         graph and edge data management
└── TopologyViewer.jsx -> generic TopologicPy JSON renderer
        │
        │  REST (JSON) + SSE (text/event-stream)
        ▼
FastAPI backend (topologicpy-web-backend/app/main.py)
│
├── POST /ifc-egress-graph -> build navigation graph from IFC geometry
├── POST /ifc-egress-path  -> compute wall-aware shortest path
├── GET  /fire-sim/stream  -> SSE fire spread and dynamic path updates
├── POST /fire-sim         -> precomputed fire timeline
├── POST /rl/train         -> train Q-learning agent and return best path
└── POST /topology         -> TopologicPy cell complex operations
```

REST is used for graph and path requests. Server-Sent Events are used for real-time fire simulation streaming, keeping the connection open while the server pushes temperature and path events as they are computed.

---

## Installation & Detailed Local Deployment

This section covers first-time setup and local launch for Windows.

### System Requirements
- Windows 10/11
- Python 3.10+
- Node.js 18+ (or bundled Node runtime in this repo)
- Git
- PowerShell

### Expected Project Layout
```text
TopologicStudio/
├── topologicpy-web-backend/
├── topologicpy-web-frontend/
└── node-v24.11.1-win-x64/   # optional bundled Node runtime
```

### 1. Clone
```powershell
git clone [https://github.com/libishm1/Topologic_Studio.git](https://github.com/libishm1/Topologic_Studio.git)
cd Topologic_Studio
```

### 2. Backend Setup
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

### 3. Frontend Setup
Open a new PowerShell terminal:
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

### 4. Frontend API Configuration
Create `topologicpy-web-frontend/.env`:
```env
VITE_API_BASE=http://localhost:8000
VITE_WEBIFC_WASM_PATH=/wasm/
```

### 5. Run Backend
```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run Frontend
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

### 7. Open
* **Frontend:** http://localhost:5173
* **Backend:** http://localhost:8000
* **Backend docs:** http://localhost:8000/docs

### 8. Verify End-to-End
1. Load an IFC file
2. Build IFC egress graph
3. Pick start and exit points
4. Compute IFC egress path
5. Start fire simulation

### Common Issues
* **`.\.venv\Scripts\python.exe` not found:** Run `python -m venv .venv` first to create the virtual environment.
* **`npm` not recognized:** Use the bundled Node commands above, or install Node.js LTS globally.
* **Port 8000 already in use:**
  ```powershell
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  ```
* **Frontend cannot reach backend:** Confirm the backend is running on `http://localhost:8000` and confirm your `.env` has `VITE_API_BASE=http://localhost:8000`.

---

## Configuration

Frontend defaults are stored near the top of `src/App.jsx`.

```javascript
const API_BASE = "http://localhost:8000";
```

If the backend runs elsewhere, update this URL.

Key defaults include:
* graph mode
* node connection threshold
* stair vertical sampling
* fire interval
* hazard weight
* ambient and fire temperatures
* lethality threshold
* agent height

---

## Usage Workflow

1. Load IFC using the browser picker.
2. Build graph to extract floor, stair, door, and wall geometry and send it to the backend.
3. Pick start and end points directly in the 3D scene.
4. Compute path to run shortest path with optional wall blocking.
5. Run fire simulation in BFS or temperature mode.
6. Observe dynamic updates as temperatures and escape paths evolve.
7. Train RL to obtain a learned policy path under spreading fire.

---

## Algorithms

**1. Point sampling from IFC**
Floor and slab triangles are sampled on a regular XY grid. Stair triangles are sampled with denser vertical spacing. Door bottom-centre points are appended as mandatory graph waypoints. All sampled points are offset upward by user-defined agent height.

**2. Graph construction**
* **Hybrid mode:** Euclidean neighbour connections under threshold `d`.
* **Grid-snap mode:** Points quantized into 3D cells. Six-axis neighbour connectivity only. Optional gap fill across sparse misses.

**3. Wall-aware shortest path**
Dijkstra runs over the graph while rejecting edges whose 2D segment intersects an extracted wall centreline, unless the edge is associated with a door node.

**4. Fire diffusion**
* **BFS spread:** Gives topological distance from ignition.
* **Temperature diffusion:** Updates each node using neighbour averaging and a fixed ignition temperature.

**5. Hazard-weighted re-routing**
Path cost increases with local node temperature, biasing the route away from hazardous zones.

**6. Reinforcement learning**
Tabular Q-learning uses graph nodes as states and adjacency choices as actions. Rewards favour reaching the exit quickly while avoiding hot regions.

---

## API Reference

**`POST /ifc-egress-graph`**
Builds a navigation graph from IFC-extracted floor, stair, door, and wall geometry. Returns graph vertices and edges, wall segments, and metadata counts.
Example response shape:
```json
{
  "points": [[x, y, z], [x, y, z]],
  "edges": [[0, 1], [1, 2]],
  "walls": [[[x1, y1], [x2, y2]]],
  "meta": {
    "node_count": 1866,
    "edge_count": 1552,
    "door_count": 14,
    "wall_count": 57,
    "floor_count": 21,
    "stair_count": 4
  }
}
```

**`POST /ifc-egress-path`**
Computes shortest path between two picked coordinates over the graph with optional wall blocking.

**`POST /fire-sim`**
Returns a complete fire timeline for precomputed playback.

**`GET /fire-sim/stream`**
Streams temperature updates and dynamic paths over Server-Sent Events.

**`POST /rl/train`**
Runs server-side Q-learning and returns the best learned path.

**`POST /topology`**
General TopologicPy endpoint for cell complex and graph operations.

---

## UI Parameters

| Parameter | Meaning |
|---|---|
| Graph mode | Hybrid or grid-snap graph generation |
| Connect threshold | Maximum distance for hybrid edge creation |
| Grid cell size | Quantization size for rectilinear mode |
| Stair Z step | Stair vertical sampling resolution |
| Fire mode | BFS or temperature diffusion |
| Fire interval | Time between streamed fire updates |
| Hazard weight | Extra path cost from temperature |
| Fire temperature | Temperature assigned to ignition |
| Ambient temperature | Baseline temperature |
| Lethality threshold | Optional upper temperature for edge filtering |
| Agent height | Vertical offset of navigation points |

---

## Feature Status

| Feature | Status |
|---|---|
| Browser IFC loading | Working |
| Graph generation from slabs, stairs, doors | Working |
| Door waypoint injection | Working |
| Wall-aware shortest path | Working |
| BFS fire spread | Working |
| Temperature diffusion | Working |
| SSE fire streaming | Working |
| Dynamic path rerouting | Working |
| RL route training | Prototype |
| Wall-solid obstacle modelling | Work in progress |

---

## Known Limitations

* Fire diffusion is graph-based, not CFD-based.
* Temperature updates are simplified and do not model smoke, ventilation, or material behaviour.
* Wall extraction currently uses simplified centreline obstacles, not full volumetric collision geometry.
* RL is tabular and intended for research-scale graph experiments, not large production deployments.
* IFC element interpretation depends on geometry quality and naming consistency in the source model.

---

## Troubleshooting

**Backend says "No IFC egress graph available"**
The fire or path endpoints require a graph to be built first. Load an IFC model and run `Build Graph` before starting fire simulation or RL.

**Frontend cannot reach backend**
Check that FastAPI is running on port `8000` and that `API_BASE` in `App.jsx` matches.

**CORS error in browser**
Ensure the backend CORS middleware includes your frontend origin, for example `http://localhost:5173`.

**IFC loads but graph is empty**
The IFC may lack properly triangulated slabs or stairs, or the sampling parameters may be too strict. Try larger graph thresholds or smaller grid cells.

**Frontend memory spikes on large IFC models**
Large fragment models and dense graph overlays can exhaust browser memory. Start with smaller IFC files or coarser sampling values.

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

## License

**MIT License**
Copyright (c) 2024 Libish Murugesan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Third-Party Acknowledgements
This project builds directly on the following open-source works. Their respective licenses govern their use and distribution:

* **TopologicPy**. Licensed under the AGPL-3.0. Source: https://github.com/wassimj/topologicpy
* **@thatopen/components**. Licensed under the MIT License. Source: https://github.com/ThatOpenCompany/engine_components
* **web-ifc**. Licensed under the MIT License. Source: https://github.com/ThatOpenCompany/web-ifc
* **Three.js**. Licensed under the MIT License. Source: https://github.com/mrdoob/three.js
* **FastAPI**. Licensed under the MIT License. Source: https://github.com/fastapi/fastapi
* **React**. Licensed under the MIT License. Source: https://github.com/facebook/react

*Note on TopologicPy licensing:* TopologicPy is distributed under AGPL-3.0. If you deploy this application as a network service, the AGPL requires that you make your complete modified source code available to users of that service.

**Author:** Libish Murugesan
Researcher and Lecturer in Computational Architecture and Robotics for Architecture
Alfaisal University, Riyadh, Saudi Arabia
GitHub: @libishm1
