# Upload Contracts

## `POST /upload-topology`

Accepts either a viewer contract or a TopologicPy export.

Viewer contract minimum:

```json
{
  "vertices": [],
  "faces": []
}
```

TopologicPy export:

- Must be a list.
- Parsed with `Topology.ByJSONDictionary`.
- Converted into a viewer contract through `json_by_cluster`.

Expected error behavior:

- Unsupported Blender/Sverchok-like JSON returns `400`.
- Non-list unsupported payload returns `400`.
- TopologicPy parse failure returns `422`.

## `POST /upload-ifc`

Multipart upload:

- Field: `file`
- Query params: `include_path`, `tilt_min`, `max_z_span`, `min_floor_area`
- File extension must be `.ifc`

This route uses `ifcopenshell`. The Dockerfile installs it explicitly, while `requirements.txt` does not list it.

## Frontend Call Sites

- JSON upload: `src/App.jsx` posts to `/upload-topology`.
- Server-side IFC upload: `src/App.jsx` posts to `/upload-ifc`.
- Browser IFC egress path: `src/IFCViewer.jsx` extracts geometry and `src/App.jsx` posts it to `/ifc-egress-graph`.

