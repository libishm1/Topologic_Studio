// Run with: node --test src/core/eikonal.property.test.mjs
//
// Property test for the graph fire front, the JavaScript side of
// FireEgress/Eikonal.lean (Lean project fireegress). frontArrival
// (deadline.js) is graph distance / speed, the "Dijkstra on l_e / F" front of
// the review (section 10). Checked against the real buildLatticeGraph:
//   latArrival_never_early, latArrival_lt - on an open 8-connected lattice
//     euclid / F <= T < 1.0824 euclid / F, and T = octile * spacing / F
//     (open_lattice_dijkstra_exact); oct_12_5 - the 8.2% is reached;
//   rectArrival_bounds, rectArrival_diagonal - on the rectilinear lattice
//     T = taxicab * spacing / F, in [euclid / F, sqrt2 euclid / F], with sqrt2
//     at the diagonal neighbours;
//   graph_front_never_early, lattice_walls_delay - with random wall boxes T is
//     never below euclid / F, and at lattice nodes never below the open-lattice
//     octile time: walls only delay the front;
//   dijkstra_bellman, dijkstra_discrete_eikonal - the min-plus equation
//     T(v) = min_u (T(u) + l(u, v) / F) at every node, i.e. the steepest
//     backward difference quotient is exactly 1 / F;
//   arrival_speed_mono - a faster front is never later.
// Random choices come from a seeded mulberry32 (as in rl.js), so a failure
// reproduces. Coordinates are float32, so comparisons carry a relative
// tolerance.
import assert from "node:assert/strict";
import { test } from "node:test";

import { frontArrival } from "./deadline.js";
import { buildLatticeGraph } from "./lattice.js";
import { csr, edgeLengths } from "./route.js";

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
/** Lean's K = sqrt(4 - 2 sqrt 2) ~ 1.082392 and the bound 1.0824 above it. */
const K = Math.sqrt(4 - 2 * SQRT2);
const UPPER = 1.0824;
const REL = 1e-6;

const euclid = (g, a, b) =>
  Math.hypot(g.nodes[b * 3] - g.nodes[a * 3], g.nodes[b * 3 + 1] - g.nodes[a * 3 + 1], g.nodes[b * 3 + 2] - g.nodes[a * 3 + 2]);

/** Lattice coordinates of node k, or null when it is off the lattice (a wall-face node). */
function latticeIndex(g, k) {
  const cos = Math.cos(g.angle), sin = Math.sin(g.angle), s = g.spacing;
  const x = g.nodes[k * 3], y = g.nodes[k * 3 + 1];
  const u = (x * cos + y * sin) / s, v = (-x * sin + y * cos) / s;
  const i = Math.round(u), j = Math.round(v);
  return Math.abs(u - i) < 1e-3 && Math.abs(v - j) < 1e-3 ? [i, j] : null;
}

/**
 * dijkstra_bellman and dijkstra_discrete_eikonal at every node: T(origin) = 0;
 * a reached node has T = min over open links of T(u) + len / F, so the largest
 * (T(v) - T(u)) / len is exactly 1 / F; an unreached node has no reached
 * neighbour over an open link.
 */
function checkMinPlus(g, T, origin, speed, blocked, label) {
  const { indptr, neighbours, edgeIds, n } = csr(g);
  const len = edgeLengths(g);
  assert.equal(T[origin], 0, `${label}: T(origin) = ${T[origin]}`);
  for (let v = 0; v < n; v += 1) {
    let best = Infinity, steepest = -Infinity;
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      const e = edgeIds[k];
      if (blocked?.[e]) continue;
      const u = neighbours[k];
      if (!Number.isFinite(T[u])) continue;
      best = Math.min(best, T[u] + len[e] / speed);
      if (len[e] > 0 && Number.isFinite(T[v])) steepest = Math.max(steepest, (T[v] - T[u]) / len[e]);
    }
    if (!Number.isFinite(T[v])) {
      assert.equal(best, Infinity, `${label}: unreached node ${v} has a reached neighbour`);
      continue;
    }
    if (v === origin) continue;
    assert.ok(
      Math.abs(T[v] - best) <= 1e-9 * Math.max(1, T[v]),
      `${label}: node ${v} T = ${T[v]}, min over neighbours of T(u) + l / F = ${best}`,
    );
    assert.ok(
      Math.abs(steepest * speed - 1) <= REL,
      `${label}: node ${v} steepest backward quotient ${steepest} against 1 / F = ${1 / speed}`,
    );
  }
}

// Open lattices (no walls inside the floor). The turned one has two walls far
// outside the slab, only to turn the lattice by 30 degrees; they cut no link.
const turn = (30 * Math.PI) / 180;
const OPEN = [
  { name: "10 x 10 m at 0.5 m", input: { spacing: 0.5, floors: [rect(0, 0, 10, 10, 0)] } },
  { name: "12 x 5 m at 0.5 m", input: { spacing: 0.5, floors: [rect(0, 0, 12, 5, 0)] } },
  { name: "7 x 9 m at 0.3 m", input: { spacing: 0.3, floors: [rect(0, 0, 7.2, 9, 0)] } },
  {
    name: "10 x 6 m turned 30 degrees at 0.5 m",
    input: {
      spacing: 0.5,
      floors: [rect(0, 0, 10, 6, 0, turn)],
      walls: [box(-20, 30, 0, 30, 30.2, 3, turn), box(30, -20, 0, 30.2, 30, 3, turn)],
    },
  },
];

test("the Lean constants: K < 1.0824, and the (12, 5) offset already exceeds 1.0823", () => {
  assert.ok(K > 1.0823 && K < UPPER, `K = ${K}`);
  assert.ok(Math.abs(oct(12, 5) - (7 + 5 * SQRT2)) < 1e-12);
  assert.ok(oct(12, 5) / 13 > 1.0823 && oct(12, 5) / 13 < K, `oct(12, 5) / 13 = ${oct(12, 5) / 13}`);
});

for (const { name, input } of OPEN) {
  test(`open lattice ${name}: euclid / F <= frontArrival = octile * s / F < 1.0824 euclid / F`, () => {
    const g = graphOf(input);
    const n = g.stats.nodes;
    assert.ok(n > 100, `${n} nodes`);
    assert.equal(g.stats.blockedEdges, 0, "an open lattice has no blocked links");
    assert.equal(g.stats.wallFaceNodes, 0, "and no wall-face nodes");
    const ij = Array.from({ length: n }, (_, k) => latticeIndex(g, k));
    assert.ok(ij.every(Boolean), "every node is a lattice node");
    const rand = rngFrom(0xe1c0 ^ n);
    for (let trial = 0; trial < 3; trial += 1) {
      const origin = Math.floor(rand() * n);
      const speed = 0.05 + 2 * rand();
      const T = frontArrival(g, origin, { speed, blocked: g.blocked });
      const [i0, j0] = ij[origin];
      for (let v = 0; v < n; v += 1) {
        assert.ok(Number.isFinite(T[v]), `${name}: node ${v} unreached on an open floor`);
        const e = euclid(g, origin, v) / speed;
        const want = (oct(ij[v][0] - i0, ij[v][1] - j0) * g.spacing) / speed;
        assert.ok(Math.abs(T[v] - want) <= REL * Math.max(1, want), `${name}: T(${v}) = ${T[v]}, octile time ${want}`);
        assert.ok(T[v] >= e * (1 - 1e-9), `${name}: T(${v}) = ${T[v]} is earlier than euclid / F = ${e}`);
        assert.ok(T[v] <= UPPER * e + 1e-9, `${name}: T(${v}) = ${T[v]} is later than 1.0824 euclid / F = ${UPPER * e}`);
      }
      checkMinPlus(g, T, origin, speed, g.blocked, name);
      // arrival_speed_mono: twice the speed is never later.
      const fast = frontArrival(g, origin, { speed: 2 * speed, blocked: g.blocked });
      for (let v = 0; v < n; v += 1) assert.ok(fast[v] <= T[v], `${name}: faster front later at ${v}`);
    }
  });
}

test("open lattice: from a corner the worst ratio exceeds 1.0823 (oct_12_5), never K", () => {
  const g = graphOf({ spacing: 0.5, floors: [rect(0, 0, 10, 10, 0)] });
  const n = g.stats.nodes;
  let corner = 0;
  for (let k = 1; k < n; k += 1) if (g.nodes[k * 3] + g.nodes[k * 3 + 1] < g.nodes[corner * 3] + g.nodes[corner * 3 + 1]) corner = k;
  const speed = 0.4;
  const T = frontArrival(g, corner, { speed, blocked: g.blocked });
  let worst = 1;
  for (let v = 0; v < n; v += 1) if (v !== corner) worst = Math.max(worst, T[v] / (euclid(g, corner, v) / speed));
  assert.ok(worst > 1.0823, `worst ratio ${worst}`);
  assert.ok(worst <= K + 1e-6, `worst ratio ${worst} above K = ${K}`);
});

test("rectilinear lattice: frontArrival = taxicab * s / F, in [euclid, sqrt2 euclid] / F, sqrt2 on the diagonal", () => {
  const g = graphOf({ spacing: 0.5, rectilinear: true, floors: [rect(0, 0, 10, 10, 0)] });
  const n = g.stats.nodes;
  const ij = Array.from({ length: n }, (_, k) => latticeIndex(g, k));
  const rand = rngFrom(0x4c0 ^ n);
  for (let trial = 0; trial < 3; trial += 1) {
    const origin = Math.floor(rand() * n);
    const speed = 0.05 + 2 * rand();
    const T = frontArrival(g, origin, { speed, blocked: g.blocked });
    const [i0, j0] = ij[origin];
    let diagonals = 0;
    for (let v = 0; v < n; v += 1) {
      const di = ij[v][0] - i0, dj = ij[v][1] - j0;
      const e = euclid(g, origin, v) / speed;
      const want = ((Math.abs(di) + Math.abs(dj)) * g.spacing) / speed;
      assert.ok(Math.abs(T[v] - want) <= REL * Math.max(1, want), `T(${v}) = ${T[v]}, taxicab time ${want}`);
      assert.ok(T[v] >= e * (1 - 1e-9), `T(${v}) earlier than euclid / F`);
      assert.ok(T[v] <= SQRT2 * e * (1 + REL) + 1e-9, `T(${v}) later than sqrt2 euclid / F`);
      if (Math.abs(di) === 1 && Math.abs(dj) === 1) {
        diagonals += 1;
        assert.ok(Math.abs(T[v] / e - SQRT2) <= REL, `diagonal neighbour ratio ${T[v] / e}`);
      }
    }
    assert.ok(diagonals > 0, "the origin has a diagonal neighbour");
    checkMinPlus(g, T, origin, speed, g.blocked, "rectilinear");
  }
});

/** Random thin wall boxes inside a 12 x 10 m floor, axis-aligned. */
function wallBoxes(rand, count) {
  const out = [];
  for (let w = 0; w < count; w += 1) {
    const length = 2 + 4 * rand();
    if (rand() < 0.5) {
      const x0 = 0.5 + rand() * (11 - length), y0 = 1 + rand() * 8;
      out.push(box(x0, y0, 0, x0 + length, y0 + 0.2, 3));
    } else {
      const x0 = 1 + rand() * 10, y0 = 0.5 + rand() * (9 - length);
      out.push(box(x0, y0, 0, x0 + 0.2, y0 + length, 3));
    }
  }
  return out;
}

test("walls: frontArrival is never below euclid / F, and at lattice nodes never below the open octile time", () => {
  const rand = rngFrom(0xf12e);
  let delayed = 0, checkedLattice = 0, checkedOff = 0;
  for (let c = 0; c < 6; c += 1) {
    const g = graphOf({ spacing: 0.5, floors: [rect(0, 0, 12, 10, 0)], walls: wallBoxes(rand, 3 + Math.floor(3 * rand())) });
    const n = g.stats.nodes;
    assert.ok(g.stats.blockedEdges > 0 || g.stats.wallFaceNodes > 0, `case ${c}: the walls cut something`);
    const ij = Array.from({ length: n }, (_, k) => latticeIndex(g, k));
    const onLattice = [];
    for (let k = 0; k < n; k += 1) if (ij[k]) onLattice.push(k);
    for (let trial = 0; trial < 3; trial += 1) {
      const origin = onLattice[Math.floor(rand() * onLattice.length)];
      const speed = 0.05 + 2 * rand();
      const T = frontArrival(g, origin, { speed, blocked: g.blocked });
      const [i0, j0] = ij[origin];
      for (let v = 0; v < n; v += 1) {
        if (!Number.isFinite(T[v])) continue;
        const e = euclid(g, origin, v) / speed;
        assert.ok(T[v] >= e * (1 - 1e-9), `case ${c}: T(${v}) = ${T[v]} is earlier than euclid / F = ${e}`);
        if (v !== origin && T[v] > UPPER * e) delayed += 1;
        if (ij[v]) {
          const open = (oct(ij[v][0] - i0, ij[v][1] - j0) * g.spacing) / speed;
          assert.ok(T[v] >= open * (1 - REL), `case ${c}: T(${v}) = ${T[v]} is earlier than the open-lattice time ${open}`);
          checkedLattice += 1;
        } else {
          checkedOff += 1;
        }
      }
      checkMinPlus(g, T, origin, speed, g.blocked, `walls case ${c}`);
    }
  }
  assert.ok(checkedLattice > 1000 && checkedOff > 0, `checked ${checkedLattice} lattice and ${checkedOff} wall-face nodes`);
  // The walls matter: some nodes arrive later than the open-floor bound allows.
  assert.ok(delayed > 0, "no node was delayed beyond 1.0824 euclid / F by the walls");
});

test("walls ignored: without the blocked flags the front passes through walls (frontArrival's default)", () => {
  // frontArrival defaults to blocked = null, which ignores wall-blocked links.
  // A wall thin enough to fall between two lattice columns (x 4.6 to 4.9, the
  // columns at 4.5 and 5.0) removes no node; it only blocks the links across
  // it. Passing g.blocked must never make the front earlier than the default,
  // and here makes it strictly later.
  const g = graphOf({ spacing: 0.5, floors: [rect(0, 0.25, 10, 9.75, 0)], walls: [box(4.6, 0, 0, 4.9, 9.0, 3)] });
  const n = g.stats.nodes;
  assert.ok(g.stats.blockedEdges > 0, "the wall blocks links");
  let origin = 0;
  for (let k = 1; k < n; k += 1) if (g.nodes[k * 3] < g.nodes[origin * 3]) origin = k;
  const speed = 0.5;
  const through = frontArrival(g, origin, { speed });
  const walled = frontArrival(g, origin, { speed, blocked: g.blocked });
  let later = 0;
  for (let v = 0; v < n; v += 1) {
    assert.ok(walled[v] >= through[v] * (1 - 1e-12), `node ${v}: walls made the front earlier`);
    if (walled[v] > through[v] * (1 + 1e-9)) later += 1;
  }
  assert.ok(later > 0, "the wall delays some node only when blocked is passed");
});
