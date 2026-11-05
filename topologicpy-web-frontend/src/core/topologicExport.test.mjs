/**
 * The TopologicPy export (topologicExport.js): the frame change undone
 * exactly, and the JSON in the layout Graph.ByJSONFile reads.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { frameChange, topologicGraph, viewerToIfc } from "./topologicExport.js";

/** Column-major 4x4 (three.js) of a turn by `angle` about the viewer's Y, then a move. */
function coordination(angle, [tx, ty, tz]) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, tx, ty, tz, 1];
}

/** The viewer's own mapping, as the backend writes it: A ifc + b. */
function forward(points, change) {
  const out = [];
  for (let i = 0; i < points.length; i += 3) {
    for (let r = 0; r < 3; r += 1) out.push(change.A[r][0] * points[i] + change.A[r][1] * points[i + 1] + change.A[r][2] * points[i + 2] + change.b[r]);
  }
  return out;
}

test("the viewer's frame change is undone: the turn, the coordination matrix and the lift", () => {
  const ifc = [1, 2, 3, -4.5, 0.25, 7, 10, -3, 0];
  for (const matrix of [coordination(0, [-2.514, -3.262, -11.635]), coordination(0.7, [5, -1, 3]), null]) {
    const change = frameChange(matrix, "y", 0.75);
    const back = viewerToIfc(Float32Array.from(forward(ifc, change)), change);
    for (let i = 0; i < ifc.length; i += 1) assert.ok(Math.abs(back[i] - ifc[i]) < 1e-5, `${i}: ${back[i]} vs ${ifc[i]}`);
    assert.equal(change.exact, Boolean(matrix));
  }
  // The Duplex's matrix (a pure move): IFC (1, 2, 3) shows at (1 - 2.514, 3 - 3.262 + 0.75, -2 - 11.635).
  const duplex = frameChange(coordination(0, [-2.514, -3.262, -11.635]), "y", 0.75);
  const v = forward([1, 2, 3], duplex);
  assert.deepEqual(v.map((x) => Math.round(x * 1000) / 1000), [-1.514, 0.488, -13.635]);
});

test("the export is TopologicPy's graph JSON: vertices with coordinates, edges by vertex id, the run's results", () => {
  // A path of three nodes along IFC x, shown in the viewer with the Duplex's move and the lift.
  const change = frameChange(coordination(0, [-2.514, -3.262, -11.635]), "y", 0.75);
  const graph = {
    points: Float32Array.from(forward([0, 0, 0, 1, 0, 0, 2, 0, 0], change)),
    edges: Uint32Array.from([0, 1, 1, 2]),
    kinds: Uint8Array.from([0, 2, 1]),
    upAxis: "y",
    agentHeight: 0.75,
  };
  const frames = [
    { time: 5, T: [20, 20, 20], vis: [100, 100, 100], co: [0, 0, 0], hrr: 10 },
    { time: 10, T: [80, 30, 20], vis: [50, 8, 100], co: [500, 5, 0], hrr: 40 },
  ];
  const json = topologicGraph({
    graph,
    coordination: coordination(0, [-2.514, -3.262, -11.635]),
    route: [2, 1, 0],
    frames,
    tenability: { untenableC: 60, smoke: true, visibilityLimit: 10 },
    properties: { source_file: "test.ifc" },
  });
  assert.deepEqual(Object.keys(json), ["properties", "vertices", "edges"]);
  assert.equal(Object.keys(json.vertices).length, 3);
  for (const e of Object.values(json.edges)) assert.ok(json.vertices[e.source] && json.vertices[e.target]);
  const [a, b, c] = ["Vertex_0", "Vertex_1", "Vertex_2"].map((id) => json.vertices[id]);
  assert.ok(Math.abs(b.x - 1) < 1e-3 && Math.abs(b.y) < 1e-3 && Math.abs(b.z) < 1e-3, JSON.stringify(b));
  assert.deepEqual([a.kind, b.kind, c.kind], ["floor", "door", "stair"]);
  assert.deepEqual([a.route_order, b.route_order, c.route_order], [2, 1, 0]);
  // Node 0 passes 60 C at 10 s, node 1 falls below 10 m visibility at 10 s, node 2 never.
  assert.deepEqual([a.untenable_s, b.untenable_s, c.untenable_s], [10, 10, -1]);
  assert.deepEqual(a.gas_T_C, [20, 80]);
  assert.equal(json.edges.Edge_0.length_m, 1);
  assert.equal(json.edges.Edge_0.kind, "door");
  assert.equal(json.edges.Edge_0.on_route, 1);
  assert.deepEqual(json.properties.hrr_kW, [10, 40]);
  assert.equal(json.properties.source_file, "test.ifc");
});
