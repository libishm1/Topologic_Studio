# Topology Viewer

`topologicpy-web-frontend/src/TopologyViewer.jsx` renders generic TopologicPy viewer contracts with Three.js.

## Responsibilities

- Initialize a Three.js scene, camera, renderer, grid, axes, and `OrbitControls`.
- Render faces as `MeshStandardMaterial` objects.
- Render topology edges as `Line`.
- Render vertices and extra cell nodes as small spheres.
- Raycast faces, edges, and vertices for selection.
- Highlight selected topology levels through parent links.
- Highlight fire and RL path vertices/edges.

## Data Contract

The viewer expects:

- `vertices`: objects with `uid` or `uuid`, plus `coordinates` or `Coordinates`.
- `edges`: objects with `uid` or `uuid`, plus `vertices`.
- `faces`: objects with `uid` or `uuid`, plus `triangles`.
- `raw`: optional fallback source for edges and sidebar details.
- `parents`: per-object parent maps used for multi-level selection.

## Selection Behavior

Clicking an object creates a parent chain in a fixed order:

```text
clusters -> cellComplexes -> cells -> shells -> faces -> wires -> edges -> vertices
```

Repeated clicks in the same region cycle through the chain. A new region tries to preserve the current selection level when possible.

## Rendering Notes

- Scene uses Z-up.
- Face opacity and color are read from direct fields or dictionary variants.
- `depthTest: false` is used for edges and vertex markers so graph overlays remain visible.
- Geometry and materials are disposed when groups are rebuilt.

## When To Edit

Edit this file for:

- Topologic JSON viewer behavior.
- Selection hierarchy behavior.
- Generic fire/RL highlighting on topology contracts.

Do not edit it for browser IFC fragments rendering; use [IFC Viewer](ifc-viewer.md).

