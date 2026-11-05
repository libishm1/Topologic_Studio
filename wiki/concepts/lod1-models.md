# LOD1 models

A LOD1 model is one vertical prism per room, extruded per storey. Studio Next reads three kinds of LOD1 source into one model (`topologicpy-web-frontend/src/core/lod1.js`). One converter turns that model into the egress geometry the lattice is built from, so the graph, routes, fire models and Flamescapes run on it unchanged. Nothing is sliced into plans.

| source | function | walls stand |
|---|---|---|
| TopologicPy CellComplex (`Topology.ExportToJSON`), with its circulation and adjacency graphs (`Graph.ExportToJSON`) | `parseTopologicLOD1(cc, {circulation, adjacency})` | on the rooms' shared outlines |
| Studio GeoJSON plan (below) | `parseGeoJSONPlan(fc)` | shared outlines, or `"inner"` |
| IFC spaces, doors, stairs and storeys only | `lod1FromIfcSpaces({spaces, doors, stairs, semantics})` | outside each space (`"inner"`) |

`lod1ToEgressGeometry(model, {upAxis})` returns what `ViewerManager.extractEgressGeometry` returns, plus `semantics` (exit flags and door widths, as `ifcSemantics` gives them) and `lod1` (rooms, doors and links by itemId, for the viewer). `lod1ToGeoJSON(model)` writes the GeoJSON plan.

## The model

IFC axes (x and y in plan, z up), metres.

- `levels`: `{id, name, ffl, height}`, lowest first.
- `rooms`: `{id, name, level, ffl, height, class, area, footprint, holes, walkable}`. The footprint runs anticlockwise and holes clockwise. Shaft, Void and Tank rooms are not walked in.
- `doors`: `{id, position: [x, y, sill], direction, level, kind, rooms: [a, b], width, assumed}`. The kind is `door`, `exit` (a final exit to the street) or `opening` (no leaf). A room id of `null` is the outside.
- `links`: `{id, kind: stair | lift | ramp, rooms: [lower, upper], position, width, footprint, line?}`.
- `meta`: `{source, name, north, boundary: shared | inner, counts, warnings}`.

## The GeoJSON plan schema

The file is one `FeatureCollection`. Coordinates are plan metres in the building's own frame (x and y as drawn, the IFC's axes), not longitude and latitude.

- **`FeatureCollection.properties`** (all optional): `name`, `north`, `units: "m"`, and `levels: [{id, name, ffl, height}]`. `boundary` is `"shared"` (the default: rooms meet on the walls' centre lines) or `"inner"` (rooms end at the walls' faces, as IfcSpaces do).
- **Rooms** are `Polygon` features; a `MultiPolygon` gives one room per part. Their properties are `{id, name, level, ffl, height, class}`:
  - `level` is required unless the plan has only one level;
  - `ffl` and `height` default to the level's;
  - `class` defaults from the name (Stair, Corridor, Toilet, Shaft, Unit).
- **Doors and openings** are `Point` or `LineString` features with `{level, kind: door | exit | opening, width, rooms?}`:
  - a LineString runs from jamb to jamb: its middle is the door, and its length is the width unless `width` is given;
  - without `rooms`, the door joins the rooms whose outlines pass within 0.3 m of it.
- **Stairs, lifts and ramps** are `Point` or `LineString` features with `{level, kind: stair | lift | ramp, toLevel?, levels?, width?, rooms?}`:
  - the feature joins the room under it on `level` to the room under it on `toLevel` (by default the next level up);
  - `levels` lists every level a run serves;
  - a LineString is the walking line, from foot to head.

A problem in a plan, such as a room without a level or a door with no room beside it, becomes a warning in `meta.warnings` rather than an exception.

## How a model becomes egress geometry

- **Room prisms** are `spaces`. The lattice samples their bottoms as floor, and the two-zone model uses them as its rooms.
- **Floor slabs** (`floors`, 0.2 m thick) and **ceiling slabs** (`roofMeshes`) are closed solids, as the Flamescapes voxeliser needs.
  - A room a stair comes up into has no slab, so the stairwell stays open to smoke.
  - On a floor level where a room has a door to the outside that is not an exit, the uncovered roofs below become walkable terraces: a flat on a roof can be reached from its stair only across one. The terrace is cut round any tanks or shafts standing on it.
- **Walls** are 0.2 m boxes. Collinear edges are merged, so a shared wall is a single box: two coincident boxes would read as open in the lattice's crossing-count test. Each door gets a gap 2.1 m high (an opening gets a full-height gap), so the lattice links the rooms through it.
- **Doors** are boxes across their wall at the door point. Each carries its `kind`; `opening` meshes have no leaf to close.
- **Stairs** are treads along the link's walking line, or along cores.js's assumed dog-leg line in the stair room.
  - A stair's first tread stands at least 0.22 m up, so stairs.js (which links treads up to 0.36 m apart) cannot run past a floor.
  - Landings are kept under 0.6 m deep, so they stay treads rather than floors that would merge with the floor above.
  - They do not use cores, because `coreStacks` stacks any cores that overlap in plan and meet in height: a parking ramp and two stair shafts became one core.
- **Ramps** that rise at most 12° become a sloping slab. **Lifts** are cores.js lifts, used only with `useLifts`.

## Checks

The headless check is `node tools/lod1-check.mjs [folder | .geojson | .ifc ...]`. For each input it reports parts, joined storeys, exits, a route from a top-floor guest room, a Flamescapes scene, a GeoJSON round trip and, for an IFC, the full-geometry graph for comparison. Committed results hold aggregates only. The lab's buildings are private: their data, names and drawings stay out of the repository.

Results on 2026-11-05:

| model | parts | main part | storeys joined | route from the top floor |
|---|---|---|---|---|
| LOD1 model A (586 rooms, 15 storeys) | 1 | 100% | 15 of 15 | 143 m |
| LOD1 model B (54 rooms, 4 storeys) | 1 | 100% | 4 of 4 | 35 m |

The IfcSpace path agrees with the full-geometry graph within about 6% on route length: the Duplex 18.6 m against 17.5 m, FZK-Haus 8.4 against 8.5, Office A 52.5 against 54.1, and the dental clinic 59.4 against 58.0.

Limits of the IfcSpace path:

- It cannot see doorways without an IfcDoor (it does not read IfcOpeningElement), so spaces joined only by such openings stay apart.
- Spaces that touch, within 3 cm, are taken as open to each other.

IFCs without IfcSpace are not supported, and that is deliberate. Rooms taken from the lattice's wall-bounded regions matched 4 to 6 in 10 IfcSpaces on the Duplex and FZK-Haus: open plans merge, and the regions carry no names or classes. The regions also need the full walls and slabs, the very geometry the fast path avoids reading, and for those files the full IFC graph already serves egress.
