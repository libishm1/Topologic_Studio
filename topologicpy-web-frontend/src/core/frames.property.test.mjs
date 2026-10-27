// Run with: node --test src/core/frames.property.test.mjs
//
// Property tests for the frame change IFC -> viewer, the statements proved in
// FireEgress/Frames.lean (C:\Users\lmurugesan\lean\fireegress):
//   (1) (x, y, z) -> (x, z, -y) is orthogonal with determinant +1 and keeps
//       the cross product (handedness);
//   (2) any rigid motion R p + t (R orthogonal, det +1) leaves the lengths the
//       real routing code reports unchanged: edgeLengths, findRoute(...).length
//       and the polyline length in straightenRoute;
//   (3) a uniform scale k multiplies those lengths by exactly k;
//   plus That Open's coordination block, built as in @thatopen/fragments
//   CoordinatesManager.getCoordinationMatrix (columns xd, yd, xd x yd), never
//   has a negative determinant, and with xd, yd rounded the way fragments
//   rounds them (1e-3, 1e-5) lengths drift by at most the proved bound.
// Pseudo-random cases come from a seeded mulberry32, as in rl.js.
import assert from "node:assert/strict";
import { test } from "node:test";

import { edgeLengths, findRoute } from "./route.js";
import { straightenRoute } from "./straighten.js";

/** mulberry32, the same generator as rl.js `rngFrom`. */
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

// ---- 3 x 3 helpers (row-major arrays of rows) -----------------------------

const det3 = (m) =>
  m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
  m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
  m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
const mulVec = (m, v) => [0, 1, 2].map((i) => m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]);
const mulMat = (a, b) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]));
const transpose = (m) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => m[j][i]));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const fromColumns = (c0, c1, c2) => [0, 1, 2].map((i) => [c0[i], c1[i], c2[i]]);

/** TURNS["y"] in navigation_check.py: web-ifc's Z-up -> Y-up turn. */
const TURN = [
  [1, 0, 0],
  [0, 0, 1],
  [0, -1, 0],
];

/** A uniformly random rotation from a random unit quaternion (Shoemake). */
function randomRotation(rng) {
  const u1 = rng(), u2 = rng(), u3 = rng();
  const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
  const [x, y, z, w] = [a * Math.sin(2 * Math.PI * u2), a * Math.cos(2 * Math.PI * u2), b * Math.sin(2 * Math.PI * u3), b * Math.cos(2 * Math.PI * u3)];
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

/** That Open's block: columns xd, yd, xd x yd (CoordinatesManager.getCoordinationMatrix). */
const fragmentsBlock = (xd, yd) => fromColumns(xd, yd, cross(xd, yd));
const round = (v, p) => Math.round(v * p) / p; // GeomsFbUtils.round

/**
 * A connected random graph of n nodes (a random spanning tree plus extra
 * edges), coordinates within +-span metres, stored in float64 so the test
 * measures the maths and not float32 storage.
 */
function randomGraph(rng, n, span) {
  const nodes = new Float64Array(n * 3);
  for (let i = 0; i < n * 3; i += 1) nodes[i] = (rng() * 2 - 1) * span;
  const edges = [];
  for (let i = 1; i < n; i += 1) edges.push(Math.floor(rng() * i), i);
  for (let k = 0; k < n; k += 1) {
    const a = Math.floor(rng() * n), b = Math.floor(rng() * n);
    if (a !== b) edges.push(a, b);
  }
  // spacing only sizes straighten.js's cell index (no sections here, so routes
  // are kept as they are); a coarse one keeps the test fast
  return { nodes, edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2), spacing: 5 };
}

/** The graph's nodes moved by p -> L p + offset (a fresh graph, no caches). */
function moved(graph, L, offset, scale = 1) {
  const n = graph.nodes.length / 3;
  const nodes = new Float64Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    const q = mulVec(L, [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]]);
    for (let k = 0; k < 3; k += 1) nodes[i * 3 + k] = q[k] + offset[k];
  }
  return { nodes, edges: graph.edges, blocked: graph.blocked, spacing: graph.spacing * scale };
}

const nodePoint = (graph, i) => [graph.nodes[i * 3], graph.nodes[i * 3 + 1], graph.nodes[i * 3 + 2]];
const relErr = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);
const TOL = 1e-9;

// ---- (1) the turn ---------------------------------------------------------

test("(1) the turn (x, y, z) -> (x, z, -y) is orthogonal, det +1, and keeps the cross product", () => {
  assert.equal(det3(TURN), 1);
  assert.deepEqual(mulMat(transpose(TURN), TURN), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  const rng = rngFrom(20261027);
  for (let c = 0; c < 300; c += 1) {
    const p = [0, 1, 2].map(() => (rng() * 2 - 1) * 100);
    const q = [0, 1, 2].map(() => (rng() * 2 - 1) * 100);
    const tp = mulVec(TURN, p);
    assert.deepEqual(tp, [p[0], p[2], -p[1]].map((v) => v + 0)); // +0 folds -0
    const lhs = mulVec(TURN, cross(p, q));
    const rhs = cross(tp, mulVec(TURN, q));
    for (let k = 0; k < 3; k += 1) assert.ok(Math.abs(lhs[k] - rhs[k]) <= 1e-9 * (1 + Math.abs(lhs[k])));
    const d0 = Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    const tq = mulVec(TURN, q);
    const d1 = Math.hypot(tp[0] - tq[0], tp[1] - tq[1], tp[2] - tq[2]);
    // a signed permutation of coordinates; Math.hypot may round differently when
    // its arguments are reordered, so compare to 1e-15 relative
    assert.ok(relErr(d1, d0) <= 1e-15, `${d1} vs ${d0}`);
  }
});

// ---- (2) rigid motions ----------------------------------------------------

test("(2) random rotation + translation (after the turn, plus lift) leaves route lengths unchanged within 1e-9", () => {
  const rng = rngFrom(7);
  let worst = 0;
  for (let c = 0; c < 200; c += 1) {
    const R = randomRotation(rng);
    assert.ok(Math.abs(det3(R) - 1) < 1e-12, "random rotation has det +1");
    const L = mulMat(R, TURN); // from_coordination: m[:3, :3] @ TURN
    assert.ok(Math.abs(det3(L) - 1) < 1e-12, "the composite keeps handedness");
    const lift = 0.75;
    const offset = [0, 1, 2].map((k) => (rng() * 2 - 1) * 50 + (k === 1 ? lift : 0));

    const g = randomGraph(rng, 30 + Math.floor(rng() * 30), 20);
    const h = moved(g, L, offset);

    // edge lengths (what A* and Dijkstra weigh)
    const la = edgeLengths(g), lb = edgeLengths(h);
    for (let e = 0; e < la.length; e += 1) {
      const err = relErr(lb[e], la[e]);
      worst = Math.max(worst, err);
      assert.ok(err <= TOL, `edge ${e}: ${lb[e]} vs ${la[e]}`);
    }

    // a route between two nodes, picked at their own positions
    const n = g.nodes.length / 3;
    const s = Math.floor(rng() * n), t = Math.floor(rng() * n);
    const ra = findRoute(g, nodePoint(g, s), nodePoint(g, t), { useWalls: false });
    const rb = findRoute(h, nodePoint(h, s), nodePoint(h, t), { useWalls: false });
    assert.ok(ra.found && rb.found);
    if (ra.length > 0) {
      const err = relErr(rb.length, ra.length);
      worst = Math.max(worst, err);
      assert.ok(err <= TOL, `route ${s}->${t}: ${rb.length} vs ${ra.length}`);
    } else {
      assert.ok(rb.length <= TOL);
    }

    // the polyline length straightenRoute reports (no sections: path kept as is)
    const ids = [];
    for (let k = 0, len = 3 + Math.floor(rng() * 20); k < len; k += 1) ids.push(Math.floor(rng() * n));
    const pa = straightenRoute(g, ids, { useWalls: true });
    const pb = straightenRoute(h, ids, { useWalls: true });
    assert.deepEqual(pb.node_ids, pa.node_ids);
    const err = relErr(pb.length, pa.length);
    worst = Math.max(worst, err);
    assert.ok(err <= TOL, `polyline: ${pb.length} vs ${pa.length}`);
  }
  assert.ok(worst <= TOL);
});

// ---- (3) uniform scale ----------------------------------------------------

test("(3) a uniform scale k multiplies route lengths by exactly k (within 1e-9)", () => {
  const rng = rngFrom(11);
  for (let c = 0; c < 100; c += 1) {
    const k = [0.001, 0.0254, 0.3048, 1, 1000][c % 5] * (0.5 + rng());
    const R = randomRotation(rng);
    const L = mulMat(R, TURN).map((row) => row.map((v) => k * v));
    const g = randomGraph(rng, 25, 20);
    const h = moved(g, L, [rng() * 10, rng() * 10, rng() * 10], k);
    const ids = Array.from({ length: 12 }, () => Math.floor(rng() * 25));
    const pa = straightenRoute(g, ids, { useWalls: true });
    const pb = straightenRoute(h, ids, { useWalls: true });
    assert.ok(relErr(pb.length, k * pa.length) <= TOL, `k=${k}: ${pb.length} vs ${k * pa.length}`);
  }
});

// ---- That Open's coordination block ---------------------------------------

test("That Open's block [xd | yd | xd x yd] never has a negative determinant; orthonormal -> det +1", () => {
  const rng = rngFrom(3);
  for (let c = 0; c < 300; c += 1) {
    const xd = [0, 1, 2].map(() => rng() * 2 - 1);
    const yd = [0, 1, 2].map(() => rng() * 2 - 1);
    const M = fragmentsBlock(xd, yd);
    const d = det3(M);
    const cc = dot(cross(xd, yd), cross(xd, yd));
    assert.ok(d >= -1e-12, `det ${d}`);
    assert.ok(Math.abs(d - cc) <= 1e-12 * (1 + cc));
    // orthonormal columns from a rotation
    const R = randomRotation(rng);
    const e = det3(fragmentsBlock(transpose(R)[0], transpose(R)[1]));
    assert.ok(Math.abs(e - 1) < 1e-12);
  }
});

test("a unit scale k in xd, yd would scale z by k^2, not k (non-uniform)", () => {
  const k = 0.001;
  const M = fragmentsBlock([k, 0, 0], [0, k, 0]);
  assert.deepEqual(mulVec(M, [1, 0, 0]), [k, 0, 0]);
  assert.deepEqual(mulVec(M, [0, 0, 1]), [0, 0, k * k]);
});

test("fragments' rounding of xd (1e-3) and yd (1e-5): lengths drift, bounded by the block's stretch", () => {
  const rng = rngFrom(99);
  let worst = 0;
  for (let c = 0; c < 200; c += 1) {
    const R = randomRotation(rng);
    const cols = transpose(R);
    const xd = cols[0].map((v) => round(v, 1e3));
    const yd = cols[1].map((v) => round(v, 1e5));
    const C = fragmentsBlock(xd, yd);
    // the block's stretch range: square roots of the extreme eigenvalues of CᵀC,
    // bounded here by Gershgorin on CᵀC
    const G = mulMat(transpose(C), C);
    let lo = Infinity, hi = 0;
    for (let i = 0; i < 3; i += 1) {
      const off = [0, 1, 2].filter((j) => j !== i).reduce((s, j) => s + Math.abs(G[i][j]), 0);
      lo = Math.min(lo, G[i][i] - off);
      hi = Math.max(hi, G[i][i] + off);
    }
    const g = randomGraph(rng, 25, 20);
    const h = moved(g, mulMat(C, TURN), [0, 0, 0]);
    const ids = Array.from({ length: 12 }, () => Math.floor(rng() * 25));
    const a = straightenRoute(g, ids, { useWalls: true }).length;
    const b = straightenRoute(h, ids, { useWalls: true }).length;
    worst = Math.max(worst, relErr(b, a));
    assert.ok(b >= Math.sqrt(Math.max(lo, 0)) * a - 1e-9 && b <= Math.sqrt(hi) * a + 1e-9);
  }
  // A finding, not a pass/fail: the "exact" coordination alignment is exact
  // only up to this drift for a rotated model.
  console.log(`  rounded block: worst relative length drift ${worst.toExponential(2)}`);
  assert.ok(worst < 5e-3);
});
