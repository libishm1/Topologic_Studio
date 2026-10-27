// Run with: npm test
import assert from "node:assert/strict";
import { test } from "node:test";

import { KIND_DOOR, buildLatticeGraph } from "./lattice.js";
import { astar, findRoute, snap } from "./route.js";
import { straightenRoute, withStraight } from "./straighten.js";

function rect(x0, y0, x1, y1, z) {
  const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].flatMap(([x, y]) => [x, y, z]);
  return { positions: Float32Array.from(pts), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) };
}

function box(x0, y0, z0, x1, y1, z1, itemId) {
  const corners = [];
  for (const z of [z0, z1]) for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) corners.push(x, y, z);
  const f = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]];
  return { positions: Float32Array.from(corners), indices: Uint32Array.from(f.flat()), itemId };
}

/** A lattice in the shape the app routes on (route.js reads `nodes`). */
function graphOf(input) {
  const g = buildLatticeGraph({ upAxis: "z", spacing: 0.5, agentHeight: 0, ...input });
  return { ...g, nodes: g.points, upAxis: "z", agentHeight: 0 };
}

function segmentHitsBox(p, q, [x0, y0, x1, y1]) {
  for (let k = 0; k <= 200; k += 1) {
    const t = k / 200;
    const x = p[0] + (q[0] - p[0]) * t, y = p[1] + (q[1] - p[1]) * t;
    if (x > x0 && x < x1 && y > y0 && y < y1) return true;
  }
  return false;
}

test("across an open room the route becomes one straight line", () => {
  const g = graphOf({ floors: [rect(0, 0, 10, 10, 0)] });
  const r = findRoute(g, [0.5, 0.5, 0], [9.5, 3.5, 0]);
  assert.ok(r.found);
  const s = straightenRoute(g, r.node_ids);
  assert.equal(s.points.length, 2);
  assert.ok(Math.abs(s.length - Math.hypot(9, 3)) < 1e-6);
  assert.ok(s.length < r.length - 0.5, `${s.length} vs lattice ${r.length}`);
});

test("a straightened route goes round a wall, not through it", () => {
  // A 10 x 10 room split by a wall at x = 5 up to y = 7, running past the
  // slab edge as real walls do.
  const wall = [4.9, -0.5, 5.1, 7];
  const g = graphOf({ floors: [rect(0, 0, 10, 10, 0)], walls: [box(wall[0], wall[1], 0, wall[2], wall[3], 3)] });
  const r = findRoute(g, [2, 2, 0], [8, 2, 0]);
  assert.ok(r.found);
  const s = straightenRoute(g, r.node_ids);
  for (let i = 1; i < s.points.length; i += 1) {
    assert.ok(!segmentHitsBox(s.points[i - 1], s.points[i], wall), `segment ${i} crosses the wall`);
  }
  // Taut string round the wall end: at least 2 * hypot(3, 5), at most the lattice route.
  assert.ok(s.length >= 2 * Math.hypot(3, 5) - 1e-6);
  assert.ok(s.length <= r.length + 1e-9);
  assert.ok(s.points.length < r.points.length / 3);
});

test("without wall sections a route is left as found unless walls are off", () => {
  const g = graphOf({ floors: [rect(0, 0, 10, 10, 0)] });
  delete g.sections;
  const r = findRoute(g, [0.5, 0.5, 0], [9.5, 3.5, 0]);
  assert.deepEqual(straightenRoute(g, r.node_ids).node_ids, r.node_ids);
  assert.equal(straightenRoute(g, r.node_ids, { useWalls: false }).points.length, 2);
});

test("a straightened route stays over the floor, not across a void", () => {
  // A C-shaped floor round a 7 x 4 m void, no walls.
  const void_ = [3, 3, 10, 7];
  const g = graphOf({ floors: [rect(0, 0, 10, 3, 0), rect(0, 3, 3, 7, 0), rect(0, 7, 10, 10, 0)] });
  const r = findRoute(g, [9, 1.5, 0], [9, 8.5, 0]);
  assert.ok(r.found);
  const s = straightenRoute(g, r.node_ids);
  const inner = [void_[0] + 0.15, void_[1] + 0.15, void_[2], void_[3] - 0.15];
  for (let i = 1; i < s.points.length; i += 1) {
    assert.ok(!segmentHitsBox(s.points[i - 1], s.points[i], inner), `segment ${i} crosses the void`);
  }
  assert.ok(s.length < r.length);
});

test("doors stay on the straightened route", () => {
  // Two rooms either side of a wall at x = 5 with a door at y = 7..8.
  const walls = [box(4.9, 0, 0, 5.1, 7, 3), box(4.9, 8, 0, 5.1, 10, 3)];
  const doors = [box(4.9, 7, 0, 5.1, 8, 2.1, "door")];
  const g = graphOf({ floors: [rect(0, 0, 4.9, 10, 0), rect(5.1, 0, 10, 10, 0)], walls, doors });
  const r = findRoute(g, [1, 1, 0], [9, 1, 0]);
  assert.ok(r.found, r.note);
  const door = r.node_ids.find((n) => g.kinds[n] === KIND_DOOR);
  assert.ok(door !== undefined, "the lattice route uses the door");
  const s = straightenRoute(g, r.node_ids);
  assert.ok(s.node_ids.includes(door));
  assert.ok(s.points.length <= 5, `${s.points.length} vertices`);
});

test("a burning node keeps the straight line away", () => {
  const g = graphOf({ floors: [rect(0, 0, 10, 10, 0)] });
  // Burn the node in the middle of the straight line, and route round it as
  // the fire's rerouting does.
  const fire = snap(g, [5, 2, 0]);
  const avoid = new Uint8Array(g.nodes.length / 3);
  avoid[fire] = 1;
  const blocked = new Uint8Array(g.edges.length / 2);
  for (let e = 0; e < blocked.length; e += 1) {
    if (g.edges[e * 2] === fire || g.edges[e * 2 + 1] === fire) blocked[e] = 1;
  }
  const r = astar(g, snap(g, [0.5, 0.5, 0]), snap(g, [9.5, 3.5, 0]), { blocked });
  const s = straightenRoute(g, r.nodes, { avoid });
  assert.ok(s.points.length > 2);
  assert.ok(!s.node_ids.includes(fire));
});

test("withStraight reads node_ids or an RL path and leaves empty routes alone", () => {
  const g = graphOf({ floors: [rect(0, 0, 10, 10, 0)] });
  const r = findRoute(g, [0.5, 0.5, 0], [9.5, 3.5, 0]);
  assert.equal(withStraight(g, r).straight_points.length, 2);
  assert.equal(withStraight(g, { path: r.node_ids }).straight_points.length, 2);
  const none = { found: false, node_ids: [], points: [] };
  assert.equal(withStraight(g, none), none);
});
