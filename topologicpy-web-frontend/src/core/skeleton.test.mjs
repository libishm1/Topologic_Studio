// Skeletons of the walkable floor (core/skeleton.js): clearance, the medial
// axis, the slime mould's desire lines and the medial way out.
import test from "node:test";
import assert from "node:assert/strict";

import { agreement, clearanceOf, desireLines, medialAxis, medialRoute, openAdjacency, ridgeOf } from "./skeleton.js";
import { KIND_DOOR, KIND_FLOOR } from "./lattice.js";

/** A floor nx x nz nodes 0.5 m apart, eight-linked, less `hole(i, k)`; `kind(i, k)` marks doors. */
function floor(nx, nz, { hole = () => false, kind = () => KIND_FLOOR } = {}) {
  const nodes = [], kinds = [], edges = [], id = new Map();
  for (let i = 0; i < nx; i += 1) for (let k = 0; k < nz; k += 1) {
    if (hole(i, k)) continue;
    id.set(`${i},${k}`, nodes.length / 3);
    nodes.push(i * 0.5, 0.75, k * 0.5);
    kinds.push(kind(i, k));
  }
  for (let i = 0; i < nx; i += 1) for (let k = 0; k < nz; k += 1) {
    const a = id.get(`${i},${k}`);
    if (a === undefined) continue;
    for (const [di, dk] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
      const b = id.get(`${i + di},${k + dk}`);
      if (b !== undefined) edges.push(a, b);
    }
  }
  const graph = { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: Uint8Array.from(kinds), spacing: 0.5, upAxis: "y" };
  return { graph, id, adj: openAdjacency(graph) };
}

test("clearance rises from the walls; the medial axis runs down a corridor's middle", () => {
  const { graph, id, adj } = floor(30, 5);
  const { clearance, axis, links } = medialAxis(graph, adj);
  // The border rows are half a link from the wall; the middle row two links and a half.
  assert.equal(clearance[id.get("10,0")], 0.25);
  assert.equal(clearance[id.get("10,2")], 1.25);
  // One line, not a band: the rows either side of the middle are off it.
  for (let i = 3; i < 27; i += 1) {
    assert.equal(axis[id.get(`${i},2`)], 1, `middle (${i}, 2) on the axis`);
    for (const k of [0, 1, 3, 4]) assert.equal(axis[id.get(`${i},${k}`)], 0, `(${i}, ${k}) off it`);
  }
  assert.ok(links.length >= 2 * 20, "the axis's links join it up");
});

test("the medial axis passes through a doorway between two rooms", () => {
  // Two rooms 9 x 9 with a wall between at i = 9, a door at (9, 4).
  const { graph, id, adj } = floor(19, 9, { hole: (i, k) => i === 9 && k !== 4, kind: (i, k) => (i === 9 && k === 4 ? KIND_DOOR : KIND_FLOOR) });
  const { axis } = medialAxis(graph, adj);
  assert.equal(axis[id.get("9,4")], 1, "the door");
  assert.equal(axis[id.get("4,4")], 1, "the first room's centre");
  assert.equal(axis[id.get("14,4")], 1, "the second's");
});

test("desire lines: the ways out from the whole floor bundle into the slime's tubes, each link towards the exit", () => {
  const { graph, id, adj } = floor(20, 9);
  const n = graph.nodes.length / 3;
  // A tube along k = 6 to the exit at (19, 6).
  const weights = new Float32Array(n);
  for (let i = 0; i < 20; i += 1) for (let k = 0; k < 9; k += 1) weights[id.get(`${i},${k}`)] = Math.max(0, 1 - 0.4 * Math.abs(k - 6));
  const exit = id.get("19,6");
  const { links, flow, on, maxFlow } = desireLines(graph, adj, weights, [exit], { cell: 1, minFlow: 2 });
  assert.ok(links.length > 0 && maxFlow >= 10);
  // Every link points to a node nearer the exit along its line of x.
  for (let k = 0; k < links.length; k += 2) assert.ok(graph.nodes[links[k + 1] * 3] >= graph.nodes[links[k] * 3] - 1e-6, "towards the exit");
  // The trunk: the tube's row from the middle on carries the most.
  const trunk = [], rest = [];
  for (let k = 0; k < links.length; k += 2) (Math.round(graph.nodes[links[k] * 3 + 2] / 0.5) === 6 && graph.nodes[links[k] * 3] > 5 ? trunk : rest).push(flow[k / 2]);
  const mean = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
  assert.ok(trunk.length > 5 && mean(trunk) > 2 * mean(rest), `trunk ${mean(trunk).toFixed(1)} against ${mean(rest).toFixed(1)}`);
  assert.equal(on[exit], 1);
  // Against the medial axis (k = 4 mid-room, with branches to the corners):
  // in the middle of the room the trunk runs two links off it.
  const { axis } = medialAxis(graph, adj);
  assert.equal(axis[id.get("10,4")], 1);
  assert.equal(axis[id.get("10,6")], 0);
  const trunkOnly = new Uint8Array(n);
  for (let i = 6; i < 14; i += 1) trunkOnly[id.get(`${i},6`)] = 1;
  assert.equal(agreement(adj, trunkOnly, axis, { reach: 1 }), 0);
  assert.equal(agreement(adj, trunkOnly, axis, { reach: 2 }), 1);
});

test("the medial way out keeps clearer of the walls than the shortest walk", () => {
  const { graph, id, adj } = floor(30, 15);
  const { clearance } = clearanceOf(graph, adj);
  const start = id.get("0,0"), exit = id.get("29,14");
  const medial = medialRoute(adj, clearance, start, [exit], { beta: 2 });
  const direct = medialRoute(adj, clearance, start, [exit], { beta: 0 });
  const meanClear = (r) => r.nodes.reduce((s, v) => s + clearance[v], 0) / r.nodes.length;
  assert.ok(meanClear(medial) > meanClear(direct), `${meanClear(medial).toFixed(2)} m against ${meanClear(direct).toFixed(2)} m`);
  assert.ok(medial.length >= direct.length - 1e-9);
  // A flat field's ridge lies only where a neighbour is missing (a wall counts as none): along the floor's edge.
  const flat = ridgeOf(adj, new Float64Array(graph.nodes.length / 3).fill(1));
  for (let v = 0; v < flat.length; v += 1) if (flat[v]) assert.ok(adj.start[v + 1] - adj.start[v] < 8, `node ${v} inside`);
  assert.equal(flat[id.get("15,7")], 0);
});
