// Property test for straightenRoute, checked against the Lean file
// FireEgress/Straighten.lean (C:\Users\lmurugesan\lean\fireegress).
// Run with: node --test src/core/straighten.property.test.mjs
//
// (1) Straightening keeps a subsequence of the lattice route's nodes with the
//     same first and last node, so (generalised triangle inequality, Lean
//     `subseq_length_le` / `pathLen_sublist_le`) it is never longer.
// (2) Every point of an accepted shortcut keeps at least
//     sqrt(m^2 - (delta/2)^2) off the wall sections, where m is the clearance
//     the samples were required to keep (min(c, ca, cb) - 1e-6) and delta the
//     sample gap (at most s/4); Lean `euclid_shortcut_clearance`. The generic
//     metric bound m - delta/2 (Lean `shortcut_clearance`) is weaker and
//     implied. The full 0.25 m is NOT guaranteed between samples; the test
//     only records how often a shortcut dips below it.
import assert from "node:assert/strict";
import { test } from "node:test";

import { buildLatticeGraph } from "./lattice.js";
import { findRoute } from "./route.js";
import { straightenRoute } from "./straighten.js";

/** mulberry32, as rngFrom in rl.js. */
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

function graphOf(input) {
  const g = buildLatticeGraph({ upAxis: "z", spacing: 0.5, agentHeight: 0, ...input });
  return { ...g, nodes: g.points, upAxis: "z", agentHeight: 0 };
}

function polyLength(g, ids) {
  let L = 0;
  for (let i = 1; i < ids.length; i += 1) {
    const a = ids[i - 1] * 3, b = ids[i] * 3;
    L += Math.hypot(g.nodes[b] - g.nodes[a], g.nodes[b + 1] - g.nodes[a + 1], g.nodes[b + 2] - g.nodes[a + 2]);
  }
  return L;
}

function distanceToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
  return Math.hypot(px - ax - t * dx, py - ay - t * dy);
}

function wallDist(segs, x, y) {
  let best = Infinity;
  for (let k = 0; k < segs.length; k += 4) best = Math.min(best, distanceToSegment(x, y, segs[k], segs[k + 1], segs[k + 2], segs[k + 3]));
  return best;
}

/** Positions of `sub` in `full`, increasing, or null if not a subsequence. */
function subsequenceIndices(sub, full) {
  const at = [];
  let j = 0;
  for (const v of sub) {
    while (j < full.length && full[j] !== v) j += 1;
    if (j === full.length) return null;
    at.push(j);
    j += 1;
  }
  return at;
}

/** A random 10 x 10 floor with 1-4 random wall boxes (as in straighten.test.mjs). */
function randomCase(rng) {
  const W = 6 + Math.floor(rng() * 7), H = 6 + Math.floor(rng() * 7);
  const walls = [];
  const nWalls = 1 + Math.floor(rng() * 4);
  for (let w = 0; w < nWalls; w += 1) {
    const vertical = rng() < 0.5;
    const thick = 0.1 + rng() * 0.2;
    const along0 = -0.5 + rng() * (vertical ? H : W) * 0.6;
    const along1 = along0 + 1 + rng() * (vertical ? H : W) * 0.7;
    const at = 1 + rng() * ((vertical ? W : H) - 2);
    if (vertical) walls.push(box(at - thick / 2, along0, 0, at + thick / 2, along1, 3));
    else walls.push(box(along0, at - thick / 2, 0, along1, at + thick / 2, 3));
  }
  // Sometimes a free-standing column (a box, short and square).
  if (rng() < 0.5) {
    const cx = 1 + rng() * (W - 2), cy = 1 + rng() * (H - 2), r = 0.1 + rng() * 0.3;
    walls.push(box(cx - r, cy - r, 0, cx + r, cy + r, 3));
  }
  const g = graphOf({ floors: [rect(0, 0, W, H, 0)], walls });
  const start = [0.3 + rng() * (W - 0.6), 0.3 + rng() * (H - 0.6), 0];
  const end = [0.3 + rng() * (W - 0.6), 0.3 + rng() * (H - 0.6), 0];
  return { g, start, end };
}

const CASES = 300;
const s = 0.5, c = 0.25, eps = 1e-6;
const delta = s / 4;

test("straightened route: subsequence with same endpoints, never longer, keeps the sampled clearance bound", () => {
  const rng = rngFrom(0x5eed2026);
  let routes = 0, shortcuts = 0, fpExcess = 0, belowFull = 0, worstFullShortcut = Infinity;
  let worstMargin = Infinity;
  for (let n = 0; n < CASES; n += 1) {
    const { g, start, end } = randomCase(rng);
    const r = findRoute(g, start, end);
    if (!r.found) continue;
    routes += 1;
    const st = straightenRoute(g, r.node_ids);

    // (1) subsequence, same endpoints, no longer.
    const at = subsequenceIndices(st.node_ids, r.node_ids);
    assert.ok(at, `case ${n}: straightened ids are not a subsequence of the lattice route`);
    assert.equal(st.node_ids[0], r.node_ids[0], `case ${n}: first node changed`);
    assert.equal(st.node_ids.at(-1), r.node_ids.at(-1), `case ${n}: last node changed`);
    if (r.node_ids.length >= 1) assert.equal(at[0], 0);
    assert.equal(at.at(-1), r.node_ids.length - 1);
    // Real-number theorem; floating point rounding may differ by a few ulps
    // when a shortcut is collinear with the edges it replaces.
    assert.ok(st.length <= r.length + 1e-9 * Math.max(1, r.length), `case ${n}: ${st.length} > ${r.length}`);
    if (st.length > r.length) fpExcess += 1;
    assert.ok(Math.abs(polyLength(g, r.node_ids) - r.length) < 1e-9);

    // (2) clearance along each accepted shortcut (consecutive kept nodes that
    // were not adjacent on the lattice route).
    const sec = (g.sections || []).find((x) => x.key === Math.round(0.75 * 10));
    if (!sec) continue;
    const segs = sec.segs;
    for (let i = 1; i < st.node_ids.length; i += 1) {
      if (at[i] - at[i - 1] < 2) continue;
      shortcuts += 1;
      const a = st.node_ids[i - 1] * 3, b = st.node_ids[i] * 3;
      const ax = g.nodes[a], ay = g.nodes[a + 1], bx = g.nodes[b], by = g.nodes[b + 1];
      const L = Math.hypot(bx - ax, by - ay);
      const N = Math.ceil(L / (s / 4));
      assert.ok(L / N <= delta + 1e-12, "sample gap exceeds s/4");
      const ca = Math.min(c, wallDist(segs, ax, ay)), cb = Math.min(c, wallDist(segs, bx, by));
      const m = Math.min(c, ca, cb) - eps;
      const bound = m > 0 ? Math.sqrt(Math.max(0, m * m - (L / N / 2) ** 2)) : 0;
      let worst = Infinity;
      const dense = Math.max(400, Math.ceil(L / 0.002));
      for (let k = 0; k <= dense; k += 1) {
        const t = k / dense;
        worst = Math.min(worst, wallDist(segs, ax + (bx - ax) * t, ay + (by - ay) * t));
      }
      worstMargin = Math.min(worstMargin, worst - bound);
      assert.ok(worst >= bound - 1e-9, `case ${n}: shortcut ${i} comes ${worst} off a wall, bound ${bound}`);
      // Generic metric-space bound (weaker).
      assert.ok(worst >= m - L / N / 2 - 1e-9);
      if (ca >= c && cb >= c) {
        worstFullShortcut = Math.min(worstFullShortcut, worst);
        if (worst < c - eps) belowFull += 1;
      }
    }
  }
  assert.ok(routes > CASES / 2, `only ${routes} routes found`);
  assert.ok(shortcuts > 100, `only ${shortcuts} shortcuts`);
  console.log(
    `routes ${routes}, shortcuts ${shortcuts}, worst margin over bound ${worstMargin.toFixed(4)} m, ` +
      `shortcuts with both ends >= 0.25 m that dip below 0.25 m: ${belowFull} (closest ${worstFullShortcut.toFixed(4)} m), ` +
      `float-only length excess: ${fpExcess}`,
  );
});
