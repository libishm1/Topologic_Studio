// Run with: node --test src/core
import assert from "node:assert/strict";
import { test } from "node:test";

import { KIND_DOOR, KIND_STAIR, buildLatticeGraph, dominantAngle, wallSegments } from "./lattice.js";

/** A horizontal rectangle at height z, as two triangles (z up). */
function rect(x0, y0, x1, y1, z, rotate = 0) {
  const c = Math.cos(rotate), s = Math.sin(rotate);
  const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].flatMap(([x, y]) => [c * x - s * y, s * x + c * y, z]);
  return { positions: Float32Array.from(pts), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) };
}

/** The same rectangle as a fan of thin triangles from one corner, the way
 *  exporters often tessellate a slab. */
function fan(x0, y0, x1, y1, z, slices = 17) {
  const pts = [x0, y0, z];
  const rim = [];
  for (let k = 0; k <= slices; k += 1) rim.push([x1, y0 + ((y1 - y0) * k) / slices]);
  for (let k = slices - 1; k >= 0; k -= 1) rim.push([x0 + ((x1 - x0) * k) / slices, y1]);
  rim.forEach(([x, y]) => pts.push(x, y, z));
  const idx = [];
  for (let k = 1; k < rim.length; k += 1) idx.push(0, k, k + 1);
  return { positions: Float32Array.from(pts), indices: Uint32Array.from(idx) };
}

/** A box, for walls and doors. */
function box(x0, y0, z0, x1, y1, z1, rotate = 0, itemId) {
  const c = Math.cos(rotate), s = Math.sin(rotate);
  const corners = [];
  for (const z of [z0, z1]) for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) corners.push(c * x - s * y, s * x + c * y, z);
  const f = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]];
  return { positions: Float32Array.from(corners), indices: Uint32Array.from(f.flat()), itemId };
}

function edgeAngles(g) {
  const out = [];
  for (let e = 0; e < g.edges.length / 2; e += 1) {
    const a = g.edges[e * 2], b = g.edges[e * 2 + 1];
    const dx = g.points[b * 3] - g.points[a * 3];
    const dy = g.points[b * 3 + 1] - g.points[a * 3 + 1];
    out.push((((Math.atan2(dy, dx) * 180) / Math.PI) % 180 + 180) % 180);
  }
  return out;
}

const base = { upAxis: "z", spacing: 0.5, agentHeight: 0 };

test("a 10 x 10 m floor is a 21 x 21 lattice with axis and 45-degree links only", () => {
  const g = buildLatticeGraph({ ...base, floors: [rect(0, 0, 10, 10, 0)] });
  assert.equal(g.stats.nodes, 21 * 21);
  // 2 * 20 * 21 axis links + 2 * 20 * 20 diagonals.
  assert.equal(g.stats.edges, 840 + 800);
  for (const a of edgeAngles(g)) assert.ok([0, 45, 90, 135, 180].some((t) => Math.abs(a - t) < 1e-3), `edge at ${a} deg`);
  const degree = new Uint8Array(g.stats.nodes);
  for (const n of g.edges) degree[n] += 1;
  assert.ok(Math.max(...degree) <= 8);
});

test("the lattice does not depend on how the slab was triangulated", () => {
  const two = buildLatticeGraph({ ...base, floors: [rect(0, 0, 10, 6, 0)] });
  const thin = buildLatticeGraph({ ...base, floors: [fan(0, 0, 10, 6, 0)] });
  assert.equal(thin.stats.nodes, two.stats.nodes);
  assert.equal(thin.stats.edges, two.stats.edges);
});

test("rectilinear drops the diagonals", () => {
  const g = buildLatticeGraph({ ...base, rectilinear: true, floors: [rect(0, 0, 10, 10, 0)] });
  assert.equal(g.stats.edges, 840);
});

test("the lattice turns with the building", () => {
  const turn = (30 * Math.PI) / 180;
  const walls = [box(0, -0.1, 0, 10, 0.1, 3, turn), box(-0.1, 0, 0, 0.1, 10, 3, turn)];
  assert.ok(Math.abs((dominantAngle(wallSegments(walls, "z")) * 180) / Math.PI - 30) < 0.5);
  const g = buildLatticeGraph({ ...base, floors: [rect(0, 0, 10, 10, 0, turn)], walls });
  assert.ok(Math.abs(g.stats.angleDeg - 30) < 0.5);
  for (const a of edgeAngles(g)) {
    const rel = (((a - 30) % 45) + 45) % 45;
    assert.ok(rel < 1e-2 || 45 - rel < 1e-2, `edge at ${a} deg is off the building's grid`);
  }
});

/** Whether a node left of `x` and one right of it are connected, walls on. */
function crossesAt(g, x) {
  const parent = Int32Array.from({ length: g.stats.nodes }, (_, i) => i);
  const find = (n) => (parent[n] === n ? n : (parent[n] = find(parent[n])));
  for (let e = 0; e < g.edges.length / 2; e += 1) {
    if (!g.blocked[e]) parent[find(g.edges[e * 2])] = find(g.edges[e * 2 + 1]);
  }
  const left = [...Array(g.stats.nodes).keys()].find((n) => g.points[n * 3] < x - 1);
  const right = [...Array(g.stats.nodes).keys()].find((n) => g.points[n * 3] > x + 1);
  return find(left) === find(right);
}

for (const at of [5.0, 5.25]) {
  // At 5.0 the wall sits on a lattice line and removes those nodes; at 5.25
  // it falls between lines and blocks the links that cross it.
  test(`a wall at x = ${at} separates the two sides, and its door reconnects them`, () => {
    // The wall runs past both floor edges, as a dividing wall meets the outer
    // walls; one ending flush with the slab edge really can be walked round.
    const floors = [rect(0, 0.25, 10, 9.75, 0)];
    const walls = [box(at - 0.1, 0, 0, at + 0.1, 10, 3)];
    const closed = buildLatticeGraph({ ...base, floors, walls });
    assert.equal(crossesAt(closed, at), false);
    if (at === 5.25) assert.ok(closed.stats.blockedEdges > 0);
    const open = buildLatticeGraph({ ...base, floors, walls, doors: [box(at - 0.1, 4.55, 0, at + 0.1, 5.45, 2.1, 0, 7)] });
    assert.equal(crossesAt(open, at), true, "the doorway is open");
    assert.equal(open.stats.doorNodes, 1);
    assert.ok(open.kinds.includes(KIND_DOOR));
  });
}

test("nodes inside a wall are removed", () => {
  const g = buildLatticeGraph({ ...base, floors: [rect(0, 0.25, 10, 9.75, 0)], walls: [box(4.8, 0, 0, 5.2, 10, 3)] });
  for (let n = 0; n < g.stats.nodes; n += 1) assert.ok(Math.abs(g.points[n * 3] - 5) >= 0.2 - 1e-6);
});

test("an opening with no IfcDoor in it is open", () => {
  // Two wall pieces with a 1 m gap between them: an archway. The section at
  // walking height has no material there, so the rooms connect without any
  // door object. The old centreline walls closed such openings.
  const floors = [rect(0, 0.25, 10, 9.75, 0)];
  const walls = [box(4.9, 0, 0, 5.1, 4.5, 3), box(4.9, 5.5, 0, 5.1, 10, 3)];
  const g = buildLatticeGraph({ ...base, floors, walls });
  assert.equal(crossesAt(g, 5), true);
});

test("a wall with a door void is open at the void, closed elsewhere", () => {
  // One wall mesh with a hole from the floor to 2.1 m, the way IFC exports a
  // wall with an opening: a ring of four boxes stands in for it here.
  const floors = [rect(0, 0.25, 10, 9.75, 0)];
  const walls = [box(4.9, 0, 0, 5.1, 4.5, 3), box(4.9, 5.5, 0, 5.1, 10, 3), box(4.9, 4.5, 2.1, 5.1, 5.5, 3)];
  const g = buildLatticeGraph({ ...base, floors, walls });
  assert.equal(crossesAt(g, 5), true, "open under the lintel");
  for (let e = 0; e < g.edges.length / 2; e += 1) {
    const a = g.edges[e * 2], b = g.edges[e * 2 + 1];
    const crosses = (g.points[a * 3] - 5) * (g.points[b * 3] - 5) < 0;
    const y = (g.points[a * 3 + 1] + g.points[b * 3 + 1]) / 2;
    if (crosses && !g.blocked[e]) assert.ok(y > 4.4 && y < 5.6, `link through the wall at y = ${y}`);
  }
});

test("a stair joins two floors into one connected graph", () => {
  // Ground 0..10 x 0..4 at z 0; upper floor 6..10 at z 3; a flight from
  // x = 1 to x = 5.75 in 16 risers of 0.1875 m with 0.25 m goings... use 19
  // treads of 0.25 m so the run reaches the upper floor's edge at x = 6.
  const stairs = [];
  for (let k = 1; k <= 15; k += 1) stairs.push(rect(2.25 + (k - 1) * 0.25, 0, 2.25 + k * 0.25, 2, 0.1875 * k));
  const floors = [rect(0, 0, 2.25, 4, 0), rect(2.25, 2, 10, 4, 0), rect(6, 0, 10, 2, 3)];
  // A wall standing on the upper floor: without one, a top surface with
  // nothing on it reads as a roof and is excluded.
  const walls = [box(6, -0.2, 3, 10, 0, 5.8)];
  const g = buildLatticeGraph({ ...base, floors, stairs, walls });
  assert.ok(g.kinds.includes(KIND_STAIR));
  // Union-find over edges.
  const parent = Int32Array.from({ length: g.stats.nodes }, (_, i) => i);
  const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  for (let e = 0; e < g.edges.length; e += 2) parent[find(g.edges[e])] = find(g.edges[e + 1]);
  const upper = [...Array(g.stats.nodes).keys()].find((n) => g.points[n * 3 + 2] > 2.9);
  const ground = [...Array(g.stats.nodes).keys()].find((n) => g.points[n * 3 + 2] < 0.1);
  assert.equal(find(upper), find(ground), "the upper floor is reachable from the ground by the stair");
});
