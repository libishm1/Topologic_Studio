// Property test for wall blocking's segment test, `segmentsCross` in lattice.js.
// Mirrors FireEgress/Segments.lean (Lean project C:\Users\lmurugesan\lean\fireegress):
//   sound + complete : test(a,b,c,d) <=> the lines are not parallel and the
//                      parametric intersection has 0 < t < 1 and 0 < u < 1
//   crossTest_iff    : test <=> the closed segments meet and no endpoint of
//                      either lies on the other closed segment
//   touching_*       : touching but false => (non-parallel) the contact is an
//                      endpoint, (parallel) all four points are collinear
// The oracle is independent of the code: it solves A + t(B-A) = C + u(D-C) for t, u
// with exact rational arithmetic (BigInt) on the exact binary values of the inputs.
// Run with: node --test src/core/segments.property.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { SegmentIndex } from "./lattice.js";

// segmentsCross is module-private; load the real function text from lattice.js.
const source = readFileSync(new URL("./lattice.js", import.meta.url), "utf8");
const match = source.match(/function segmentsCross\(([^)]*)\)\s*\{([\s\S]*?)\n\}/);
assert.ok(match, "segmentsCross not found in lattice.js");
const segmentsCross = new Function(...match[1].split(",").map((s) => s.trim()), match[2]);

/** mulberry32, as in rl.js. */
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

// ------------------------------------------------------------ exact oracle

/** A finite double as m * 2^e with BigInt m, exactly. */
function decompose(x) {
  if (x === 0) return { m: 0n, e: 0 };
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const hi = view.getUint32(0), lo = view.getUint32(4);
  const sign = hi >>> 31 ? -1n : 1n;
  const exp = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e;
  if (exp === 0) e = -1074;
  else {
    mant |= 1n << 52n;
    e = exp - 1075;
  }
  return { m: sign * mant, e };
}

/** All eight coordinates as BigInts on one common power-of-two scale (exact). */
function exactInts(coords) {
  const parts = coords.map(decompose);
  const emin = Math.min(...parts.map((p) => p.e));
  return parts.map((p) => p.m << BigInt(p.e - emin));
}

const sgn = (v) => (v > 0n ? 1 : v < 0n ? -1 : 0);
const crs = (ux, uy, vx, vy) => ux * vy - uy * vx;
const dot = (ux, uy, vx, vy) => ux * vx + uy * vy;

/** p on the closed segment x-y (exact). */
function onSeg(px, py, xx, xy, yx, yy) {
  if (xx === yx && xy === yy) return px === xx && py === xy;
  if (crs(yx - xx, yy - xy, px - xx, py - xy) !== 0n) return false;
  return dot(px - xx, py - xy, yx - xx, yy - xy) >= 0n && dot(px - yx, py - yy, xx - yx, xy - yy) >= 0n;
}

/**
 * Exact facts about segments ab, cd from the parametric solution
 * A + t r = C + u s, r = B - A, s = D - C, t = ((C-A) x s) / (r x s), u = ((C-A) x r) / (r x s).
 */
function oracle(coords) {
  const [ax, ay, bx, by, cx, cy, dx, dy] = exactInts(coords);
  const rx = bx - ax, ry = by - ay, sx = dx - cx, sy = dy - cy;
  const wx = cx - ax, wy = cy - ay;
  let den = crs(rx, ry, sx, sy);
  let tn = crs(wx, wy, sx, sy), un = crs(wx, wy, rx, ry);
  if (den < 0n) [den, tn, un] = [-den, -tn, -un];
  let proper = false, closed = false, contactIsEndpoint = null;
  if (den !== 0n) {
    proper = 0n < tn && tn < den && 0n < un && un < den;
    closed = 0n <= tn && tn <= den && 0n <= un && un <= den;
    if (closed) contactIsEndpoint = tn === 0n || tn === den || un === 0n || un === den;
  } else if (rx === 0n && ry === 0n && sx === 0n && sy === 0n) {
    closed = ax === cx && ay === cy;
  } else if (rx === 0n && ry === 0n) {
    closed = onSeg(ax, ay, cx, cy, dx, dy);
  } else if (sx === 0n && sy === 0n) {
    closed = onSeg(cx, cy, ax, ay, bx, by);
  } else if (crs(rx, ry, wx, wy) === 0n) {
    // Collinear: overlap of the 1-D intervals along r.
    const b = dot(rx, ry, rx, ry), c = dot(wx, wy, rx, ry), d = dot(dx - ax, dy - ay, rx, ry);
    const lo = c < d ? c : d, hi = c < d ? d : c;
    closed = lo <= b && hi >= 0n;
  }
  const endpointOn =
    onSeg(ax, ay, cx, cy, dx, dy) || onSeg(bx, by, cx, cy, dx, dy) ||
    onSeg(cx, cy, ax, ay, bx, by) || onSeg(dx, dy, ax, ay, bx, by);
  const d = [
    crs(dx - cx, dy - cy, ax - cx, ay - cy), crs(dx - cx, dy - cy, bx - cx, by - cy),
    crs(bx - ax, by - ay, cx - ax, cy - ay), crs(bx - ax, by - ay, dx - ax, dy - ay),
  ].map(sgn);
  return { parallel: den === 0n, proper, closed, endpointOn, contactIsEndpoint, d };
}

/**
 * Whether the code's float determinants all have a certain sign: Shewchuk's
 * orient2d bound (3 + 16 eps) eps (|l| + |r|), with the same float differences.
 */
function floatSignsCertain([ax, ay, bx, by, cx, cy, dx, dy]) {
  const eps = 2 ** -53, bound = (3 + 16 * eps) * eps;
  const det = (ux, uy, vx, vy) => {
    const l = ux * vy, r = uy * vx;
    return Math.abs(l - r) > bound * (Math.abs(l) + Math.abs(r)) || (l === 0 && r === 0);
  };
  return (
    det(dx - cx, dy - cy, ax - cx, ay - cy) && det(dx - cx, dy - cy, bx - cx, by - cy) &&
    det(bx - ax, by - ay, cx - ax, cy - ay) && det(bx - ax, by - ay, dx - ax, dy - ay)
  );
}

// ------------------------------------------------------------ case generators

function cases(seed, n, coord) {
  const rand = rngFrom(seed);
  const pick = (k) => Math.floor(rand() * k);
  const out = [];
  for (let i = 0; i < n; i += 1) {
    const p = () => [coord(rand), coord(rand)];
    let A = p(), B = p(), C = p(), D = p();
    // P + (Q - P) k / n: exact for the integer generators (coordinates are multiples of 12).
    const lerp = (P, Q, [k, n]) => [P[0] + ((Q[0] - P[0]) * k) / n, P[1] + ((Q[1] - P[1]) * k) / n];
    const frac = () => [[1, 2], [1, 4], [3, 4], [1, 3]][pick(4)];
    switch (pick(10)) {
      case 0: B = A.slice(); break; // degenerate link
      case 1: D = C.slice(); break; // degenerate wall segment
      case 2: C = A.slice(); break; // shared vertex
      case 3: C = lerp(A, B, frac()); break; // T-junction on ab
      case 4: A = lerp(C, D, frac()); break; // T-junction on cd
      case 5: { // collinear, overlapping or end to end
        const k = [[[-1, 2], [1, 2]], [[1, 4], [3, 4]], [[1, 1], [2, 1]], [[-1, 1], [0, 1]], [[1, 2], [3, 2]], [[2, 1], [3, 1]]][pick(6)];
        C = lerp(A, B, k[0]); D = lerp(A, B, k[1]);
        break;
      }
      case 6: D = [C[0] + (B[0] - A[0]), C[1] + (B[1] - A[1])]; break; // parallel
      case 7: { // a link through an outline vertex
        const m = lerp(A, B, [1, 2]);
        C = m; break;
      }
      case 8: { // a proper crossing: cd's midpoint is inside ab
        const m = lerp(A, B, frac());
        C = [2 * m[0] - D[0], 2 * m[1] - D[1]];
        break;
      }
      default: break; // generic
    }
    out.push([...A, ...B, ...C, ...D]);
  }
  return out;
}

/** The code, both directly and through the exported SegmentIndex.crosses. */
function runCode(q, cell) {
  const direct = segmentsCross(...q);
  const index = new SegmentIndex(Float64Array.from(q.slice(4)), cell);
  const viaIndex = index.crosses(q[0], q[1], q[2], q[3]);
  return { direct, viaIndex };
}

/** Checks the theorems' statements on one case, given the code's answer. */
function checkTheorems(q, got, o) {
  // sound + complete
  assert.equal(got, o.proper, `sound/complete: ${q}`);
  // crossTest_iff
  assert.equal(got, o.closed && !o.endpointOn, `crossTest_iff: ${q}`);
  if (o.closed && !got) {
    // false_while_touching_iff
    assert.ok(o.endpointOn, `false_while_touching_iff: ${q}`);
    // touching_nonparallel_at_endpoint
    if (!o.parallel) assert.ok(o.contactIsEndpoint, `touching_nonparallel_at_endpoint: ${q}`);
    // touching_parallel_collinear
    else assert.deepEqual(o.d, [0, 0, 0, 0], `touching_parallel_collinear: ${q}`);
  }
  // degenerate_false
  if ((q[0] === q[2] && q[1] === q[3]) || (q[4] === q[6] && q[5] === q[7])) assert.equal(got, false);
}

test("integer coordinates: code equals the exact theorems on every case", () => {
  // |coord| <= 12 * 4096 < 2^16: every determinant is an exact integer in doubles, and the sign
  // of a product of doubles is exact, so the code must agree exactly.
  const small = (r) => 12 * (Math.floor(r() * 21) - 10);
  const big = (r) => 12 * (Math.floor(r() * 8193) - 4096);
  let proper = 0, touching = 0;
  for (const [coord, cell, seed] of [[small, 48, 1], [big, 24576, 2]]) {
    for (const q of cases(seed, 400, coord)) {
      const { direct, viaIndex } = runCode(q, cell);
      assert.equal(viaIndex, direct, `SegmentIndex.crosses disagrees with segmentsCross: ${q}`);
      const o = oracle(q);
      checkTheorems(q, direct, o);
      if (o.proper) proper += 1;
      if (o.closed && !o.proper) touching += 1;
    }
  }
  // The generators must exercise both directions and the touching cases.
  console.log(`integer cases: ${proper} proper crossings, ${touching} touching but not crossing`);
  assert.ok(proper > 50, `only ${proper} proper crossings`);
  assert.ok(touching > 150, `only ${touching} touching-but-false cases`);
});

test("float coordinates: code equals the exact theorems wherever float signs are certain", () => {
  const coord = (r) => (r() - 0.5) * 200 + r() * 1e-3;
  let certain = 0, uncertain = 0, disagree = 0;
  for (const q of cases(3, 600, coord)) {
    const { direct, viaIndex } = runCode(q, 32);
    assert.equal(viaIndex, direct, `SegmentIndex.crosses disagrees with segmentsCross: ${q}`);
    const o = oracle(q);
    if (floatSignsCertain(q)) {
      certain += 1;
      checkTheorems(q, direct, o);
    } else {
      // Near-degenerate in doubles (an endpoint within rounding of the other
      // line): the code answers for the rounded determinants, not the exact ones.
      uncertain += 1;
      if (direct !== o.proper) disagree += 1;
    }
  }
  assert.ok(certain > 300, `only ${certain} certain cases`);
  // Informational: how often rounding decides the answer on near-degenerate input.
  console.log(`float cases: ${certain} certain (all agree), ${uncertain} near-degenerate, ${disagree} of those disagree with exact arithmetic`);
});

test("touching configurations the strict test does not block (documented gaps)", () => {
  // Link through an outline vertex, T-junction, collinear overlap: all false, as proved.
  assert.equal(segmentsCross(0, 0, 4, 0, 2, 0, 2, 3), false); // wall ends on the link
  assert.equal(segmentsCross(0, 0, 4, 0, 2, -3, 2, 0), false);
  assert.equal(segmentsCross(0, 0, 4, 0, 1, 0, 3, 0), false); // runs along a wall face
  assert.equal(segmentsCross(0, 0, 4, 0, 4, 0, 4, 3), false); // link ends on the wall
  // A link through two vertices of a solid diamond crosses solid with no proper
  // crossing; only the `inside` samples can catch it.
  const diamond = Float64Array.from([-1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, -1, 0, -1, -1, 0]);
  const index = new SegmentIndex(diamond, 1);
  assert.equal(index.crosses(-1.2, 0, 9, 0), false);
});
