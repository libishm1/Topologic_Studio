# Topology And IFC Upload

This chunk covers `/upload-topology` and `/upload-ifc`.

## `/upload-topology`

Accepted payloads:

- A pre-converted viewer contract with `vertices` and `faces`.
- A TopologicPy JSON export represented as a list of topology dictionaries.

Rejected payloads:

- Blender/Sverchok-style graph JSON with `export_version` and `main_tree`.
- Non-list payloads that are not already viewer contracts.

Current implementation catches TopologicPy parse exceptions and returns HTTP `422` instead of an unhandled `500`.

## Viewer Contract

The viewer contract generally contains:

- `vertices`: UID plus coordinates and dictionary metadata.
- `edges`: UID plus vertex references and metadata.
- `faces`: triangulated face coordinates plus metadata.
- `raw`: combined topology entities for sidebar inspection.

`json_by_cluster` uses TopologicPy to enumerate vertices, edges, wires, faces, shells, cells, and cell complexes, then writes normalized `uid`, `type`, parent links, color, and opacity metadata.

## `/upload-ifc`

This is server-side IFC conversion:

1. Accept only `.ifc` uploads.
2. Write the upload to a temp file.
3. Call `convert_ifc_to_contract`.
4. Delete the temp file in `finally`.

`convert_ifc_to_contract` imports `ifcopenshell` and `ifcopenshell.geom`. If the library is unavailable, it returns HTTP `500`.

## Important Difference From Browser IFC

There are two IFC paths:

- Browser IFC path: `IFCViewer.jsx` uses That Open Components, `web-ifc`, and fragments; it extracts geometry and posts to `/ifc-egress-graph`.
- Server IFC path: `/upload-ifc` uses `ifcopenshell` to convert an uploaded IFC into the generic Topologic viewer contract.

Do not assume both paths produce the same graph state. The browser path is the main path for IFC egress graph work.

