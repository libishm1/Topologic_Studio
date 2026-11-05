// Run with: node --test src/core
import assert from "node:assert/strict";
import { test } from "node:test";

import { KIND_DOOR, KIND_STAIR, buildLatticeGraph } from "./lattice.js";
import { astar, componentLabels, distancesFrom } from "./route.js";
import { exitDoors } from "./ifcSemantics.js";
import { buildScene } from "./flamescapesRun.js";
import { fuelsFor } from "./fireClasses.js";
import {
  classFromName,
  doorOnBoundary,
  lod1FromIfcSpaces,
  lod1FromJsonFiles,
  lod1JsonKind,
  lod1ToEgressGeometry,
  lod1ToGeoJSON,
  outlineFromSegments,
  parseGeoJSONPlan,
  parseTopologicLOD1,
  ringArea,
  triangulate,
} from "./lod1.js";

// ------------------------------------------------------------- fixtures

/**
 * A TopologicPy CellComplex export (Topology.ExportToJSON) of axis-aligned
 * boxes: Vertex, Edge, Wire, Face, Shell and Cell entries linked by uuid, a
 * face two boxes share written once, and one CellComplex.
 */
function cellComplexJson(boxes, dictionary = { name: "test" }) {
  const records = [];
  const ids = new Map();
  let next = 0;
  const uuid = () => `u${(next += 1)}`;
  const vertex = (x, y, z) => {
    const k = `v${x},${y},${z}`;
    if (!ids.has(k)) {
      const u = uuid();
      ids.set(k, u);
      records.push({ type: "Vertex", uuid: u, coordinates: [x, y, z], dictionary: { uuid: u }, apertures: [] });
    }
    return ids.get(k);
  };
  const edge = (a, b) => {
    const k = a < b ? `e${a}|${b}` : `e${b}|${a}`;
    if (!ids.has(k)) {
      const u = uuid();
      ids.set(k, u);
      records.push({ type: "Edge", uuid: u, vertices: [a, b], dictionary: { uuid: u }, apertures: [] });
    }
    return ids.get(k);
  };
  const face = (corners) => {
    const vs = corners.map((c) => vertex(...c));
    const k = `f${[...vs].sort().join("|")}`;
    if (!ids.has(k)) {
      const wire = uuid(), u = uuid();
      records.push({ type: "Wire", uuid: wire, edges: vs.map((v, i) => edge(v, vs[(i + 1) % vs.length])), dictionary: { uuid: wire }, apertures: [] });
      records.push({ type: "Face", uuid: u, wires: [wire], dictionary: { uuid: u }, apertures: [] });
      ids.set(k, u);
    }
    return ids.get(k);
  };
  const cells = [];
  for (const { x0, y0, z0, x1, y1, z1, dict } of boxes) {
    const faces = [
      [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
      [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
      [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
      [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
      [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
      [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
    ].map(face);
    const shell = uuid(), cell = uuid();
    records.push({ type: "Shell", uuid: shell, faces, dictionary: { uuid: shell }, apertures: [] });
    records.push({ type: "Cell", uuid: cell, shells: [shell], dictionary: { ...dict, uuid: cell }, apertures: [] });
    cells.push(cell);
  }
  records.push({ type: "CellComplex", uuid: uuid(), cells, dictionary: { ...dictionary, toplevel: true }, apertures: [] });
  return records;
}

/** Two rooms side by side, a door between them and an exterior door from the first to the street. */
function twoRooms() {
  const dict = (id, name, cls) => ({ id, name, level: "GF", level_name: "Ground floor", ffl: 0, height: 3, space_class: cls, area: 12 });
  const cc = cellComplexJson([
    { x0: 0, y0: 0, z0: 0, x1: 4, y1: 3, z1: 3, dict: dict("GF-A", "Lobby", "Corridor") },
    { x0: 4, y0: 0, z0: 0, x1: 8, y1: 3, z1: 3, dict: dict("GF-B", "Guest room", "Unit") },
  ]);
  const space = (id, x, y) => ({ id, node_type: "space", space_class: "Unit", x, y, z: 1.5 });
  const circulation = {
    properties: {},
    vertices: {
      Vertex_0000: space("GF-A", 2, 1.5),
      Vertex_0001: space("GF-B", 6, 1.5),
      Vertex_0002: { id: "OP1", node_type: "door", space_class: "Opening", x: 4, y: 1.2, z: 1, width: 0.9 },
      Vertex_0003: { id: "OP2", node_type: "exterior door", space_class: "Opening", x: 0, y: 1.5, z: 1, width: 1.2 },
      Vertex_0004: { id: "GF-STREET1", node_type: "street", space_class: "Exterior", x: -2, y: 1.5, z: 0 },
    },
    edges: {
      Edge_0000: { src: 0, dst: 2, type: "door", walkable: 1 },
      Edge_0001: { src: 1, dst: 2, type: "door", walkable: 1 },
      Edge_0002: { src: 0, dst: 3, type: "exterior door", walkable: 1 },
      Edge_0003: { src: 4, dst: 3, type: "exterior door", walkable: 1 },
    },
  };
  const adjacency = { properties: {}, vertices: { Vertex_0000: { id: "GF-A" }, Vertex_0001: { id: "GF-B" } }, edges: { Edge_0000: { source: "Vertex_0000", target: "Vertex_0001", shared: "wall" } } };
  return { cc, circulation, adjacency };
}

/** A two-storey GeoJSON plan: a hall and a stair room on each floor, a stair between them, an exit from the hall. */
function twoStoreyPlan() {
  const rect = (x0, y0, x1, y1) => [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]];
  const room = (id, level, coords, name) => ({ type: "Feature", geometry: { type: "Polygon", coordinates: coords }, properties: { id, level, name } });
  const point = (x, y, props) => ({ type: "Feature", geometry: { type: "Point", coordinates: [x, y] }, properties: props });
  return {
    type: "FeatureCollection",
    properties: { name: "two storeys", levels: [{ id: "G", ffl: 0, height: 3 }, { id: "1", ffl: 3, height: 3 }] },
    features: [
      room("hall", "G", rect(0, 0, 6, 5), "Hall"),
      room("st-g", "G", rect(6, 0, 9, 5), "Stair"),
      room("corr", "1", rect(0, 0, 6, 5), "Corridor"),
      room("st-1", "1", rect(6, 0, 9, 5), "Stair"),
      room("bed", "1", rect(0, 5, 6, 9), "Bedroom"),
      point(6, 2.5, { level: "G", kind: "door", width: 1 }),
      point(6, 2.5, { level: "1", kind: "door", width: 1 }),
      point(3, 5, { level: "1", kind: "door", width: 0.9, rooms: ["corr", "bed"] }),
      { type: "Feature", geometry: { type: "LineString", coordinates: [[0, 2], [0, 3.2]] }, properties: { level: "G", kind: "exit" } },
      point(7.5, 2.5, { level: "G", kind: "stair", toLevel: "1" }),
    ],
  };
}

/** The viewer-frame graph of a model, as the site builds it. */
function graphOf(geo) {
  const built = buildLatticeGraph({ ...geo, upAxis: "y", spacing: 0.5, agentHeight: 0.75 });
  return { built, graph: { nodes: built.points, edges: built.edges, blocked: built.blocked } };
}

/** The floor node nearest an IFC plan point at a floor level (viewer frame: x, y up, -y). */
function nodeAt(built, x, y, ffl) {
  let best = -1, d = Infinity;
  for (let v = 0; v < built.points.length / 3; v += 1) {
    if (built.kinds[v] !== 0) continue;
    const e = Math.hypot(built.points[v * 3] - x, built.points[v * 3 + 2] + y, (built.points[v * 3 + 1] - 0.75 - ffl) * 4);
    if (e < d) [best, d] = [v, e];
  }
  return best;
}

// ------------------------------------------------------------ geometry

test("triangulate covers a polygon with a hole exactly, every triangle anticlockwise", () => {
  const outer = [[0, 0], [10, 0], [10, 6], [0, 6]];
  const hole = [[3, 2], [3, 4], [7, 4], [7, 2]];
  const { points, indices } = triangulate(outer, [hole]);
  let area = 0;
  for (let k = 0; k < indices.length; k += 3) {
    const tri = [points[indices[k]], points[indices[k + 1]], points[indices[k + 2]]];
    const a = ringArea(tri);
    assert.ok(a > -1e-9, "anticlockwise");
    area += a;
  }
  assert.ok(Math.abs(area - (60 - 8)) < 1e-9);
});

test("an outline from split floor faces and a T-junction is the room's outline", () => {
  // An L of three rectangles; the third's edge meets the second's in its middle.
  const segs = [];
  const quad = (x0, y0, x1, y1) => {
    const c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    c.forEach((p, i) => segs.push([...p, ...c[(i + 1) % 4]]));
  };
  quad(0, 0, 4, 2);
  quad(4, 0, 6, 2);
  quad(0, 2, 2, 5);
  const out = outlineFromSegments(segs);
  assert.equal(out.holes.length, 0);
  assert.ok(Math.abs(ringArea(out.footprint) - (8 + 4 + 6)) < 1e-9);
  assert.equal(out.footprint.length, 6, "collinear corners dropped");
});

test("a door drawn at a corner moves along its wall until it clears it", () => {
  const a = { footprint: [[0, 0], [3, 0], [3, 3], [0, 3]], holes: [] };
  const b = { footprint: [[3, 0], [6, 0], [6, 3], [3, 3]], holes: [] };
  const placed = doorOnBoundary(a, b, [3, 0.05], 0.8);
  assert.equal(placed.touching, true);
  assert.ok(Math.abs(placed.at[0] - 3) < 1e-9);
  assert.ok(placed.at[1] >= 0.5 - 1e-9, "half the width and 10 cm from the end");
});

test("room classes follow the names in the corpus's languages", () => {
  assert.equal(classFromName("1-17 Treppenhaus Ost"), "Stair");
  assert.equal(classFromName("E1 ELEVATOR"), "Shaft");
  assert.equal(classFromName("Flur"), "Corridor");
  assert.equal(classFromName("WC Damen"), "Toilet");
  assert.equal(classFromName("Bedroom 2"), "Unit");
});

// ------------------------------------------------------- TopologicPy JSON

test("a two-room CellComplex with a door parses into rooms, a door and an exit", () => {
  const { cc, circulation, adjacency } = twoRooms();
  const model = parseTopologicLOD1(cc, { circulation, adjacency });
  assert.equal(model.rooms.length, 2);
  assert.deepEqual(model.levels.map((l) => [l.id, l.ffl, l.height]), [["GF", 0, 3]]);
  const [a, b] = model.rooms;
  assert.equal(a.id, "GF-A");
  assert.ok(Math.abs(a.area - 12) < 1e-9 && Math.abs(b.area - 12) < 1e-9);
  assert.ok(ringArea(a.footprint) > 0, "anticlockwise");
  assert.equal(model.meta.warnings.length, 0);
  assert.deepEqual(model.meta.adjacency, { wallPairs: 1, wallPairsFound: 1, slabPairs: 0 });
  const door = model.doors.find((d) => d.kind === "door");
  assert.deepEqual(door.rooms, ["GF-A", "GF-B"]);
  assert.ok(Math.abs(door.position[0] - 4) < 1e-9, "on the shared wall");
  const exit = model.doors.find((d) => d.kind === "exit");
  assert.deepEqual(exit.rooms, ["GF-A", null]);
});

test("dropped files are told apart and make one model", () => {
  const { cc, circulation, adjacency } = twoRooms();
  assert.equal(lod1JsonKind(cc), "cellcomplex");
  assert.equal(lod1JsonKind(circulation), "circulation");
  assert.equal(lod1JsonKind(adjacency), "adjacency");
  assert.equal(lod1JsonKind(twoStoreyPlan()), "geojson");
  assert.equal(lod1JsonKind({ hello: 1 }), null);
  const model = lod1FromJsonFiles([adjacency, circulation, cc]);
  assert.equal(model.doors.length, 2);
  assert.equal(model.meta.adjacency.wallPairsFound, 1);
  assert.equal(lod1FromJsonFiles([twoStoreyPlan()]).rooms.length, 5);
  assert.throws(() => lod1FromJsonFiles([circulation]));
});

test("the two rooms convert to geometry the lattice joins through the door, with the exit flagged", () => {
  const { cc, circulation } = twoRooms();
  const model = parseTopologicLOD1(cc, { circulation });
  const geo = lod1ToEgressGeometry(model);
  assert.equal(geo.spaces.length, 2);
  assert.equal(geo.floors.length, 2);
  assert.equal(geo.doors.length, 2);
  assert.equal(geo.semantics.doors.filter((d) => d.exit).length, 1);
  // Viewer frame: IFC (x, y, z) is (x, z, -y), so floors face +y.
  for (const mesh of geo.floors) for (let i = 1; i < mesh.positions.length; i += 3) assert.ok(mesh.positions[i] <= 1e-6);
  // A gap in the shared wall at the door: no wall box stands there at 1 m.
  const blocks = geo.walls.some((w) => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (let i = 0; i < w.positions.length; i += 3) {
      [x0, x1] = [Math.min(x0, w.positions[i]), Math.max(x1, w.positions[i])];
      [y0, y1] = [Math.min(y0, w.positions[i + 1]), Math.max(y1, w.positions[i + 1])];
      [z0, z1] = [Math.min(z0, -w.positions[i + 2]), Math.max(z1, -w.positions[i + 2])];
    }
    return x0 <= 4 && x1 >= 4 && y0 <= 1 && y1 >= 1 && z0 <= 1.2 && z1 >= 1.2;
  });
  assert.equal(blocks, false);
  const { built, graph } = graphOf(geo);
  const { labels } = componentLabels(graph);
  const a = nodeAt(built, 1, 1.5, 0), b = nodeAt(built, 7, 1.5, 0);
  assert.equal(labels[a], labels[b], "one part");
  const exits = exitDoors(built.doorReport, geo.semantics, graph);
  assert.equal(exits.nodes.length, 1);
  assert.equal(exits.source, "LOD1 exit");
  const route = astar(graph, b, exits.nodes[0]);
  assert.ok(route && route.cost > 6 && route.cost < 10, `route ${route?.cost}`);
  assert.ok(route.nodes.some((v) => built.kinds[v] === KIND_DOOR));
});

// ------------------------------------------------------------- GeoJSON

test("a GeoJSON two-storey plan with a stair parses, joins its storeys and routes down the stair", () => {
  const model = parseGeoJSONPlan(twoStoreyPlan());
  assert.equal(model.meta.warnings.length, 0, model.meta.warnings.join("; "));
  assert.equal(model.rooms.length, 5);
  assert.deepEqual(model.levels.map((l) => l.id), ["G", "1"]);
  assert.equal(model.rooms.find((r) => r.id === "st-g").class, "Stair");
  assert.deepEqual(model.links.map((l) => [l.kind, ...l.rooms]), [["stair", "st-g", "st-1"]]);
  assert.deepEqual(model.doors.map((d) => d.rooms), [["hall", "st-g"], ["corr", "st-1"], ["corr", "bed"], ["hall", null]]);
  const exit = model.doors.find((d) => d.kind === "exit");
  assert.ok(Math.abs(exit.width - 1.2) < 1e-9, "a LineString's length is its width");

  const geo = lod1ToEgressGeometry(model);
  assert.equal(geo.stairs.length, 1, "the stair as treads");
  assert.equal(geo.floors.length, 4, "no slab where the stair comes up");
  const { built, graph } = graphOf(geo);
  assert.equal(built.stats.stairEnds, built.stats.stairEndsLinked, "both ends of the stair on a floor");
  const { labels, sizes } = componentLabels(graph);
  assert.equal(sizes.length, 1, "one part");
  const start = nodeAt(built, 3, 7, 3);
  const exits = exitDoors(built.doorReport, geo.semantics, graph).nodes;
  const dist = distancesFrom(graph, start, { blocked: graph.blocked });
  const end = exits.reduce((p, q) => (dist[q] < dist[p] ? q : p));
  const route = astar(graph, start, end);
  assert.ok(route, "a route from the bedroom upstairs to the exit");
  assert.ok(route.nodes.some((v) => built.kinds[v] === KIND_STAIR), "down the stair");
  // At least the stair's going (3 m of risers at 0.175 on 0.28 m goings) and the walk to it.
  assert.ok(route.cost > (3 / 0.175) * 0.28, `route ${route.cost}`);
  assert.equal(labels[start], labels[end]);
});

test("a model goes through GeoJSON and back unchanged", () => {
  const model = parseGeoJSONPlan(twoStoreyPlan());
  const back = parseGeoJSONPlan(JSON.parse(JSON.stringify(lod1ToGeoJSON(model))));
  assert.deepEqual(back.levels, model.levels);
  assert.deepEqual(back.rooms.map((r) => [r.id, r.level, r.ffl, r.height, r.class, r.area]), model.rooms.map((r) => [r.id, r.level, r.ffl, r.height, r.class, r.area]));
  assert.deepEqual(back.doors.map((d) => [d.kind, d.rooms]), model.doors.map((d) => [d.kind, d.rooms]));
  assert.deepEqual(back.links.map((l) => l.rooms), model.links.map((l) => l.rooms));
  const { cc, circulation } = twoRooms();
  const fromTopologic = parseTopologicLOD1(cc, { circulation });
  const again = parseGeoJSONPlan(lod1ToGeoJSON(fromTopologic));
  assert.deepEqual(again.doors.map((d) => [d.kind, d.rooms, d.width]), fromTopologic.doors.map((d) => [d.kind, d.rooms, d.width]));
});

test("a GeoJSON plan's problems come back as warnings, not exceptions", () => {
  const plan = twoStoreyPlan();
  plan.features.push({ type: "Feature", geometry: { type: "Polygon", coordinates: [[[0, 0], [1, 0], [1, 1], [0, 0]]] }, properties: { id: "x", level: "nowhere" } });
  plan.features.push({ type: "Feature", geometry: { type: "Point", coordinates: [50, 50] }, properties: { level: "G", kind: "door" } });
  const model = parseGeoJSONPlan(plan);
  assert.equal(model.rooms.length, 5);
  assert.equal(model.meta.warnings.length, 2);
  assert.throws(() => parseGeoJSONPlan({ type: "Feature" }));
});

// ------------------------------------------------------------ IfcSpace

test("IfcSpace volumes and a door mesh make rooms the lattice joins across their wall", () => {
  // Viewer frame (y up): a box from IFC (x0, y0, z0) to (x1, y1, z1).
  const box = (x0, y0, z0, x1, y1, z1, itemId) => {
    const c = [];
    for (const z of [z0, z1]) for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) c.push(x, z, -y);
    const f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]];
    return { itemId, positions: Float32Array.from(c), indices: Uint32Array.from(f.flat()) };
  };
  // Two spaces with a 0.2 m wall between them, a door in it, and a third space
  // touching the second (no wall between them).
  const spaces = [box(0, 0, 0, 4, 3, 2.7, 1), box(4.2, 0, 0, 8, 3, 2.7, 2), box(8, 0, 0, 10, 3, 2.7, 3)];
  const doors = [box(4, 1, 0, 4.2, 1.9, 2.1, 10)];
  const semantics = { spaces: [{ id: 1, name: "101", longName: "Office" }, { id: 2, name: "102", longName: "Corridor" }, { id: 3, name: "103", longName: "Lobby" }], doors: [{ id: 10, widthM: 0.9, exit: null }], storeys: [{ name: "EG", elevation: 0 }] };
  const model = lod1FromIfcSpaces({ spaces, doors, semantics });
  assert.equal(model.meta.boundary, "inner");
  assert.deepEqual(model.rooms.map((r) => [r.id, r.name, r.class]), [["1", "Office", "Unit"], ["2", "Corridor", "Corridor"], ["3", "Lobby", "Corridor"]]);
  assert.equal(model.levels[0].name, "EG");
  assert.equal(model.doors.length, 1);
  assert.deepEqual([...model.doors[0].rooms].sort(), ["1", "2"]);
  assert.ok(Math.abs(model.doors[0].width - 0.9) < 1e-9, "width from the IFC");
  const { built, graph } = graphOf(lod1ToEgressGeometry(model));
  const { labels } = componentLabels(graph);
  const office = nodeAt(built, 2, 1.5, 0), lobby = nodeAt(built, 9, 1.5, 0);
  assert.equal(labels[office], labels[lobby], "through the door, and between the touching spaces");
});

// ---------------------------------------------------------- Flamescapes

test("the converted geometry voxelises into a Flamescapes scene with walls, slabs and a fire", () => {
  const model = parseGeoJSONPlan(twoStoreyPlan());
  const geo = lod1ToEgressGeometry(model);
  const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
  const scene = buildScene({ walls: geo.walls, floors: geo.floors, roofs: geo.roofMeshes, stairs: geo.stairs, obstacles: geo.obstacles }, { upAxis: "y", fire: [3, 3.1, -7], maxCells: 8000, contentsFuel, ignitionFuel, item });
  let solid = 0;
  for (const s of scene.solid) solid += s;
  assert.ok(solid > 0.05 * scene.grid.cells, "walls and slabs are solid");
  assert.ok(scene.contents.cells.length > 0, "contents on the floors");
  assert.ok(scene.ignition, "the fire has a place");
});
