// Run with: node --test src/core/lattice-metric.property.test.mjs
//
// Property test for the lattice's geometry, the JavaScript side of
// FireEgress/Lattice.lean (Lean project fireegress):
//   octile_is_shortest - on the open 8-connected lattice the shortest route
//     between two lattice nodes is exactly oct(di, dj) * spacing, with
//     oct(a, b) = max(|a|,|b|) + (sqrt2 - 1) * min(|a|,|b|);
//   ratio_bounds - that length over the straight line lies in [1, 1.0824).
// Checked against the real buildLatticeGraph and findRoute, on random pairs
// drawn from a seeded mulberry32 (as in rl.js), so a failure reproduces.
import assert from "node:assert/strict";
import { test } from "node:test";

import { buildLatticeGraph } from "./lattice.js";
import { findRoute } from "./route.js";

/** mulberry32, as rl.js seeds it. */
function rngFrom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A horizontal rectangle at height z (z up), turned by `rotate` about the origin. */
function rect(x0, y0, x1, y1, z, rotate = 0) {
  const c = Math.cos(rotate), s = Math.sin(rotate);
  const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].flatMap(([x, y]) => [c * x - s * y, s * x + c * y, z]);
  return { positions: Float32Array.from(pts), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) };
}

/** A box (walls), turned by `rotate` about the origin. */
function box(x0, y0, z0, x1, y1, z1, rotate = 0) {
  const c = Math.cos(rotate), s = Math.sin(rotate);
  const corners = [];
  for (const z of [z0, z1]) for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) corners.push(c * x - s * y, s * x + c * y, z);
  const f = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]];
  return { positions: Float32Array.from(corners), indices: Uint32Array.from(f.flat()) };
}

/** A lattice in the shape the app routes on (route.js reads `nodes`). */
function graphOf(input) {
  const g = buildLatticeGraph({ upAxis: "z", agentHeight: 0, ...input });
  return { ...g, nodes: g.points };
}

const SQRT2 = Math.SQRT2;
const oct = (a, b) => {
  const x = Math.abs(a), y = Math.abs(b);
  return Math.max(x, y) + (SQRT2 - 1) * Math.min(x, y);
};
/** The Lean constant K = sqrt(4 - 2 sqrt 2) ~ 1.082392, and the bound 1.0824 above it. */
const K = Math.sqrt(4 - 2 * SQRT2);
const UPPER = 1.0824;

// Open lattices: no walls inside the floor. The turned one has two walls far
// outside the slab, only to turn the lattice to the building (30 degrees);
// they cut no link and hug nothing.
const turn = (30 * Math.PI) / 180;
const CASES = [
  { name: "10 x 10 m at 0.5 m", input: { spacing: 0.5, floors: [rect(0, 0, 10, 10, 0)] }, pairs: 120 },
  { name: "12 x 5 m at 0.5 m", input: { spacing: 0.5, floors: [rect(0, 0, 12, 5, 0)] }, pairs: 80 },
  { name: "7 x 9 m at 0.3 m", input: { spacing: 0.3, floors: [rect(0, 0, 7.2, 9, 0)] }, pairs: 80 },
  {
    name: "10 x 6 m turned 30 degrees at 0.5 m",
    input: {
      spacing: 0.5,
      floors: [rect(0, 0, 10, 6, 0, turn)],
      walls: [box(-20, 30, 0, 30, 30.2, 3, turn), box(30, -20, 0, 30.2, 30, 3, turn)],
    },
    pairs: 80,
  },
];

test("the constant is the Lean K, and 1.0824 bounds it", () => {
  assert.ok(K > 1.0823 && K < UPPER, `K = ${K}`);
  // K_attained: at 22.5 degrees the octile/Euclid ratio is exactly K.
  const t = Math.tan(Math.PI / 8);
  assert.ok(Math.abs(t - (SQRT2 - 1)) < 1e-15);
  assert.ok(Math.abs(oct(1, t) / Math.hypot(1, t) - K) < 1e-12);
});

for (const { name, input, pairs } of CASES) {
  test(`open lattice ${name}: grid route = octile x spacing, ratio in [1, 1.0824)`, () => {
    const g = graphOf(input);
    const s = g.spacing;
    const n = g.stats.nodes;
    assert.ok(n > 100, `${n} nodes`);
    assert.equal(g.stats.blockedEdges, 0, "an open lattice has no blocked links");
    assert.equal(g.stats.wallFaceNodes, 0, "and no wall-face nodes");
    const cos = Math.cos(g.angle), sin = Math.sin(g.angle);
    // Lattice coordinates of every node, checked to be integral.
    const ij = (k) => {
      const x = g.nodes[k * 3], y = g.nodes[k * 3 + 1];
      const u = (x * cos + y * sin) / s, v = (-x * sin + y * cos) / s;
      const i = Math.round(u), j = Math.round(v);
      assert.ok(Math.abs(u - i) < 1e-4 && Math.abs(v - j) < 1e-4, `node ${k} is off the lattice`);
      return [i, j];
    };
    const rand = rngFrom(0x1a77 ^ n);
    let worst = 1;
    for (let c = 0; c < pairs; c += 1) {
      const a = Math.floor(rand() * n);
      let b = Math.floor(rand() * n);
      if (b === a) b = (a + 1) % n;
      const pa = [g.nodes[a * 3], g.nodes[a * 3 + 1], g.nodes[a * 3 + 2]];
      const pb = [g.nodes[b * 3], g.nodes[b * 3 + 1], g.nodes[b * 3 + 2]];
      const r = findRoute(g, pa, pb);
      assert.ok(r.found, `no route ${a} -> ${b}`);
      assert.equal(r.node_ids[0], a);
      assert.equal(r.node_ids[r.node_ids.length - 1], b);
      const [ia, ja] = ij(a), [ib, jb] = ij(b);
      const want = oct(ib - ia, jb - ja) * s;
      assert.ok(
        Math.abs(r.length - want) <= 1e-5 * Math.max(1, want),
        `${name}: route ${a} -> ${b} (d = ${ib - ia}, ${jb - ja}) is ${r.length}, octile x spacing is ${want}`,
      );
      const euclid = Math.hypot(pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]);
      const ratio = r.length / euclid;
      assert.ok(ratio >= 1 - 1e-6, `ratio ${ratio} below 1`);
      assert.ok(ratio < UPPER, `ratio ${ratio} at or above ${UPPER}`);
      if (ratio > worst) worst = ratio;
    }
    // The Lean bound is strict on the lattice; the worst sampled pair stays under K.
    assert.ok(worst <= K + 1e-6, `worst ratio ${worst} above K = ${K}`);
  });
}
