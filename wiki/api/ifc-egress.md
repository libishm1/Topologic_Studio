# IFC Egress API

## `POST /ifc-egress-graph`

Builds the active IFC graph from browser-extracted geometry.

Request fields:

| Field | Type | Notes |
|---|---|---|
| `floors` | `IfcGeometry[]` | Slabs and slab-like surfaces. |
| `stairs` | `IfcGeometry[]` | Stairs and stair flights. |
| `doors` | `IfcGeometry[]` | Door geometry for waypoints. |
| `walls` | `IfcGeometry[]` | Wall geometry for path blocking. |
| `use_walls` | boolean | Store wall segments for route-time filtering. |
| `agent_height` | number | Current frontend sends `0.75`. |
| `base_spacing` | number | Backend clamps to at least `1.5`. |
| `max_edge_floor` | number | Frontend-controlled floor connection threshold. |
| `max_edge_stair` | number | Frontend-controlled stair connection threshold. |
| `up_axis` | string | Current frontend sends `"y"`. |
| `rectilinear` | boolean | Hybrid mode diagonal suppression. |
| `grid_snap` | boolean | Use voxel/cardinal graph. |
| `grid_cell_size` | number | Used only in grid-snap mode. |

`IfcGeometry` contains:

- `expressID`
- `vertices`: flat float array
- `indices`: triangle indices
- `normals`: optional flat float array

Response fields:

- `mode`
- `stats`
- `edges`: coordinate pairs for rendering
- `edge_ids`: node ID pairs aligned with `edges`
- `coords`: node ID to coordinate

## `POST /ifc-egress-path`

Request:

```json
{
  "start_point": [0, 0, 0],
  "end_point": [1, 1, 1]
}
```

Response:

```json
{
  "mode": "ifc",
  "points": [[0, 0, 0], [1, 1, 1]]
}
```

Failure cases:

- `400`: graph has not been built.
- `400`: start or end point cannot resolve to a node.
- `404`: no route found.

