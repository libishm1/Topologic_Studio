// Run with: npm test
//
// Routing-ready surrogates (surrogate.js): the hop front is a conservative
// surrogate of heat diffusion, and a conservative surrogate never makes the
// routers or the deadlines more optimistic than the full forecast.
import assert from "node:assert/strict";
import { test } from "node:test";

import { buildForecast, earliestArrival, doseFront } from "./forecast.js";
import { escapeDeadlines, doseDeadline } from "./deadline.js";
import { compareForecasts, hopSurrogate, withClosingMargin } from "./surrogate.js";

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A w x h lattice with diagonals, some links walled off. */
function lattice(w, h, rand, wallShare = 0.12) {
  const nodes = [], edges = [];
  for (let j = 0; j < h; j += 1) for (let i = 0; i < w; i += 1) nodes.push(i * 0.5, 0, j * 0.5);
  const at = (i, j) => j * w + i;
  for (let j = 0; j < h; j += 1) {
    for (let i = 0; i < w; i += 1) {
      if (i < w - 1) edges.push(at(i, j), at(i + 1, j));
      if (j < h - 1) edges.push(at(i, j), at(i, j + 1));
      if (i < w - 1 && j < h - 1) edges.push(at(i, j), at(i + 1, j + 1));
    }
  }
  const blocked = new Uint8Array(edges.length / 2);
  for (let e = 0; e < blocked.length; e += 1) if (rand() < wallShare) blocked[e] = 1;
  return { nodes: Float64Array.from(nodes), edges: Uint32Array.from(edges), blocked };
}

test("the hop front is a conservative surrogate of heat diffusion (walls on and off, any threshold)", () => {
  const rand = mulberry32(2026);
  let cases = 0, gapSum = 0;
  for (let c = 0; c < 30; c += 1) {
    const g = lattice(5 + Math.floor(rand() * 10), 4 + Math.floor(rand() * 8), rand);
    const n = g.nodes.length / 3;
    const seed = Math.floor(rand() * n);
    const blocked = rand() < 0.7 ? g.blocked : null;
    const opts = { seed, steps: 10 + Math.floor(rand() * 50), dt: 0.5 + rand() * 6, threshold: 20.5 + rand() * 99.5, blocked };
    const full = buildForecast(g, { ...opts, model: "temperature" });
    const sur = hopSurrogate(g, opts);
    const cmp = compareForecasts(full, sur);
    assert.ok(cmp.conservative, `case ${c}: ${JSON.stringify(cmp)}`);
    gapSum += cmp.meanEarlier;
    cases += 1;
  }
  console.log(JSON.stringify({ cases, meanEarlierS: +(gapSum / cases).toFixed(2) }));
});

test("a conservative surrogate never gives a safer route, a lower dose or a later deadline", () => {
  const rand = mulberry32(77);
  let compared = 0;
  for (let c = 0; c < 25; c += 1) {
    const g = lattice(5 + Math.floor(rand() * 7), 4 + Math.floor(rand() * 5), rand);
    const n = g.nodes.length / 3;
    const opts = { seed: Math.floor(rand() * n), steps: 30, dt: 1 + rand() * 4, threshold: 40 + rand() * 60, blocked: g.blocked };
    const full = buildForecast(g, { ...opts, model: "temperature" });
    const sur = hopSurrogate(g, opts);
    const exits = [Math.floor(rand() * n)];
    const speed = 0.6 + rand();
    const Lf = escapeDeadlines(g, { exits, untenable: full.closes, speed, blocked: g.blocked });
    const Ls = escapeDeadlines(g, { exits, untenable: sur.closes, speed, blocked: g.blocked });
    for (let v = 0; v < n; v += 1) {
      assert.ok(Ls[v] <= Lf[v] + 1e-9, `case ${c} node ${v}: surrogate deadline ${Ls[v]} after ${Lf[v]}`);
      // A route the surrogate finds is safe under the full forecast.
      const ea = earliestArrival(g, v, 0, { exits, closes: sur.closes, speed, blocked: g.blocked });
      if (ea.exit >= 0) {
        for (let w = ea.exit; w >= 0; w = ea.parent[w]) assert.ok(ea.arrive[w] < full.closes[w], `case ${c}: surrogate route unsafe at ${w}`);
        const fs = doseFront(g, v, { exits, forecast: sur, speed, blocked: g.blocked }).front;
        const ff = doseFront(g, v, { exits, forecast: full, speed, blocked: g.blocked }).front;
        assert.ok(ff.length > 0, "a route safe under the surrogate is safe under the full forecast");
        assert.ok(fs[fs.length - 1].dose >= ff[ff.length - 1].dose - 1e-12, `case ${c}: surrogate dose lower`);
        compared += 1;
      }
    }
    // The dose-limited deadline at one start.
    const v = Math.floor(rand() * n);
    const ds = doseDeadline(g, v, { exits, forecast: sur, limit: 0.01, speed, blocked: g.blocked, tolerance: 1e-3 });
    const df = doseDeadline(g, v, { exits, forecast: full, limit: 0.01, speed, blocked: g.blocked, tolerance: 1e-3 });
    if (ds) assert.ok(df && ds.good <= df.bad + 1e-9, `case ${c}: surrogate dose deadline later`);
  }
  assert.ok(compared > 100);
});

test("a margin makes a two-sided surrogate conservative, and moves the hard deadline by at most twice it", () => {
  const rand = mulberry32(5);
  for (let c = 0; c < 20; c += 1) {
    const g = lattice(6, 5, rand);
    const n = g.nodes.length / 3;
    const full = buildForecast(g, { model: "temperature", seed: Math.floor(rand() * n), steps: 40, dt: 2, threshold: 50, blocked: g.blocked });
    // A surrogate off by up to e either way.
    const e = 0.5 + rand() * 3;
    const noisy = { ...full, closes: Float64Array.from(full.closes, (x) => Math.max(0, x + (rand() * 2 - 1) * e)) };
    const safe = withClosingMargin(noisy, e);
    for (let v = 0; v < n; v += 1) assert.ok(safe.closes[v] <= full.closes[v] + 1e-12);
    const exits = [0];
    const Lf = escapeDeadlines(g, { exits, untenable: full.closes, speed: 1, blocked: g.blocked });
    const Ls = escapeDeadlines(g, { exits, untenable: safe.closes, speed: 1, blocked: g.blocked });
    for (let v = 0; v < n; v += 1) {
      if (!Number.isFinite(Lf[v])) continue;
      assert.ok(Ls[v] <= Lf[v] + 1e-9 && Ls[v] >= Lf[v] - 2 * e - 1e-9, `node ${v}: ${Ls[v]} vs ${Lf[v]} (e ${e})`);
    }
  }
});
