# IFC Viewer

`topologicpy-web-frontend/src/IFCViewer.jsx` loads IFC files in the browser and renders them with That Open Components, fragments, web-ifc, and Three.js.

## Libraries Used

- `@thatopen/components`: `Components`, `Worlds`, `SimpleScene`, `SimpleRenderer`, `SimpleCamera`, `IfcLoader`, `FragmentsManager`, `Raycasters`.
- `@thatopen/fragments`: worker file used by `FragmentsManager`.
- `web-ifc`: IFC entity constants such as `IFCSLAB`, `IFCSTAIR`, `IFCDOOR`, `IFCWALL`.
- `three`: scene overlays, lines, markers, transformations, colors.

## Load Pipeline

1. Initialize That Open components and a world.
2. Initialize fragments with the worker URL:

```js
new URL("@thatopen/fragments/dist/Worker/worker.mjs", import.meta.url)
```

3. Configure `IfcLoader` with the `web-ifc` WASM path.
4. Load the uploaded file as `Uint8Array`.
5. Add the resolved model object to the Three.js scene.
6. Fit camera to the model.
7. Extract IFC IDs after the scene is responsive.
8. Extract geometry only when `egressRequestId` changes.

## Extracted IFC Classes

`collectIfcIds` extracts:

- Slabs: `IFCSLAB`, `IFCSLABSTANDARDCASE`, `IFCSLABELEMENTEDCASE`
- Stairs: `IFCSTAIR`, `IFCSTAIRFLIGHT`
- Coverings: `IFCCOVERING`
- Doors: `IFCDOOR`
- Spaces: `IFCSPACE`
- Walls: `IFCWALL`
- Storeys: `IFCBUILDINGSTOREY`

Only floors, stairs, doors, and walls are sent to the egress graph route today.

## Geometry Extraction

`buildGeometryPayload` calls `model.getItemsGeometry(localIds)`, merges mesh data, applies the model world matrix, and returns:

- `expressID`
- `vertices`
- `indices`
- `normals`

Heavy extraction yields between geometry groups through `requestAnimationFrame`, which keeps the viewer more responsive during extraction.

## Visual Overlays

- Static path: red line from `/ifc-egress-path`.
- Dynamic hazard path: magenta line from `path_update` SSE.
- Graph: blue `LineSegments`, recolored by fire state.
- Fire temperature mode: per-vertex line colors from blue/cyan/green/yellow/red.
- Picked start and exit: sphere markers.

## Implementation Notes

- `DEFAULT_WASM_PATH` is `https://unpkg.com/web-ifc@0.0.73/`.
- That Open docs recommend serving WASM files statically or from a remote server. The current app uses remote fallback unless `VITE_WEBIFC_WASM_PATH` is set.
- The component disposes That Open `components` on unmount.
- If changing model transforms, also update graph, path, marker, and fire overlay transforms together.

