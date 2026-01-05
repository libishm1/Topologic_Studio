# TopologicStudio
An interface for Topologic.

## Local launch (Windows PowerShell)

### Backend (FastAPI)
```
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend (Vite)
If Node is installed globally:
```
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
npm run dev -- --host 0.0.0.0 --port 5173
```

If using the bundled Node in this repo:
```
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
$nodeDir = "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\node-v24.11.1-win-x64"
$env:Path = "$nodeDir;$env:Path"
& "$nodeDir\npm.cmd" run dev -- --host 0.0.0.0 --port 5173
```

Open:
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

## Codebase overview

### Backend (FastAPI + TopologicPy)
- API server lives in `topologicpy-web-backend/app/main.py` and exposes endpoints for health, IFC upload, graph construction, egress pathing, fire simulation (SSE + precompute), and RL training.
- Core geometry types (Vertex/Edge/Face/Shell/Cluster/Graph/Wire) come from TopologicPy and are used to build navigation graphs and straightened paths.

### IFC processing & egress graph
- `topologicpy-web-backend/app/main.py` ingests IFC-derived floor/stair point clouds and builds a navigation graph with hybrid adjacency (short edges for stairs, longer for floors).
- It computes path endpoints by snapping to nearest graph nodes, then returns path + metadata for frontend overlay.

### Fire simulation & RL
- Fire sim endpoints (`/fire-sim`, `/fire-sim/stream`) emit time-steps or temperature steps for overlay; RL endpoint (`/rl/train`) uses graph data to learn paths.
- These are designed to feed overlays in the viewer, not to replace shortest-path routing.

### Frontend (React app)
- Main UI and state live in `topologicpy-web-frontend/src/App.jsx`; this coordinates file loading, viewer mode switching, slider inputs, and egress/fire controls.
- It sends IFC egress data to backend, receives graph/path, and pushes overlays into the viewer.

### IFC viewer (Fragments)
- `topologicpy-web-frontend/src/IFCViewer.jsx` loads IFC with @thatopen components, handles selection/picking, and renders overlays (graph wires, paths, fire/temperature).
- Includes coordinate transforms (up-axis + flips) so overlays align with IFC geometry.

### Topology JSON viewer
- `topologicpy-web-frontend/src/TopologyViewer.jsx` renders TopologicPy JSON (non-IFC) with its own camera controls and selection logic.

### UI & styling
- Global styles in `topologicpy-web-frontend/src/styles.css` and `topologicpy-web-frontend/src/index.css`.
- Sidebar controls in `topologicpy-web-frontend/src/Sidebar.jsx`.

### Build & deploy
- Frontend Vite setup in `topologicpy-web-frontend/`.
- Backend uses Python venv + Uvicorn.
- GitHub Pages deploy workflow in `.github/workflows/deploy-frontend.yml`.
