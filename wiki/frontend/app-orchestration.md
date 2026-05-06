# Frontend App Orchestration

`topologicpy-web-frontend/src/App.jsx` owns the main application state and decides which backend route to call.

## API Base

```js
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
```

Deployment must set `VITE_API_BASE` when the backend is not local.

## Main State Groups

- Topology upload and viewer: `topology`, `selection`, `fileName`, `showFaces`, `showVerts`, `wireframe`, `fitRequest`.
- IFC viewer and graph: `ifcFile`, `ifcEgress`, `ifcGraphStats`, `ifcGraphCoords`, `ifcGraphEdges`, `ifcGraphEdgeIds`.
- IFC graph controls: `ifcFloorEdge`, `ifcStairEdge`, `ifcGridSnap`, `ifcGridCellSize`, `ifcUseWalls`.
- Picking: `pickMode`, `startPoint`, `exitPoint`, `startId`, `exitId`.
- Fire: `fireRunning`, `fireNodes`, `fireUsePrecompute`, `fireDelayMs`, `fireMaxSteps`, `fireUseTemperature`, `fireTemperatures`.
- Dynamic path: `dynamicPath`, `dynamicPathCost`, `dynamicPathChanged`, `pathAlpha`, `pathRecomputeInterval`, `pathLethalityThreshold`, `streamPath`.
- RL: `rlEpisodes`, `rlMaxSteps`, `rlUseFire`, `rlPath`, `rlLoading`.

## Main Workflows

### JSON Topology

1. User loads `.json`.
2. `handleFileChange` parses JSON in the browser.
3. App posts to `/upload-topology`.
4. Returned viewer contract feeds `TopologyViewer`.

### Server IFC Upload

1. User loads `.ifc` through `handleIfcUpload`.
2. App posts the file to `/upload-ifc`.
3. Returned viewer contract feeds `TopologyViewer`.

This is separate from the browser IFC egress workflow.

### Browser IFC Egress

1. User loads `.ifc` and switches to IFC viewer mode.
2. `IFCViewer` loads fragments and extracts IFC IDs.
3. User clicks Build Egress Graph.
4. `IFCViewer` extracts floor, stair, door, and wall geometry.
5. `App.jsx` posts extracted geometry to `/ifc-egress-graph`.
6. The response stores render edges, edge IDs, and coordinates.
7. Picked start/exit points are posted to `/ifc-egress-path`.

### Fire Simulation

Precompute path:

- `App.jsx` posts to `/fire-sim`.
- Timeline is played locally with `setInterval`.

Streaming path:

- `App.jsx` opens `EventSource` to `/fire-sim/stream`.
- Handles `step`, `temperature_step`, `path_update`, `meta`, and `done`.

## Current Axis Assumption

`ifcUpAxis` is hard-coded as `"y"` in `App.jsx`, with `flipZ = true`. Keep this in mind before changing backend axis logic.

## Known Frontend Lint Issues

See [issues/open-risks.md](../issues/open-risks.md). Lint currently fails, but production build succeeds.

