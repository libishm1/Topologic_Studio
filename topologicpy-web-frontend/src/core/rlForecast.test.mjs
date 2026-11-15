// Run with: node --test src/core/rlForecast.test.mjs
//
// The learners against a forecast in seconds (src/core/rlForecast.js), each
// against a statement of formal/fireegress/FireEgress/SafeClock.lean:
//   - the conversion: a move at step t into v passes the learners' test
//     t < ig(v) exactly when d + (t + 1) tau < s c(v) (move_safe_iff);
//   - the clock: every move takes at most tau seconds, on both clocks, and a
//     split link's virtual nodes close with the sooner of its ends;
//   - the seconds guarantee (clock_safe): on random graphs with random link
//     lengths, closing times, departures and margins, every route the
//     learners return, and every earliest escape on the clock, reaches every
//     node before its closing time when walked in seconds (forecast.js
//     walkRoute) - no late arrival, ever;
//   - the request: what it is certified against, the start already
//     untenable, and the hop front left as it was.
import assert from "node:assert/strict";
import { test } from "node:test";

import { walkRoute } from "./forecast.js";
import { edgeLengths } from "./route.js";
import { earliestSafeEscape, learnSafeEscape, openUntil, trainSafeRl } from "./rlSafe.js";
import { qLearningPath, trainRl } from "./rl.js";
import {
  STEP_SLACK,
  clockClosings,
  forecastIgnition,
  linkStats,
  realPath,
  stepClock,
  stepsToHorizon,
  trainForecastRl,
} from "./rlForecast.js";

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A forecast with these closing times and no dose (walkRoute's shape). */
function forecastOf(closes, horizon) {
  const n = closes.length;
  return { n, K: 1, stride: 1, frameSeconds: horizon, horizon, rate: new Float32Array(n), untenable: Float64Array.from(closes), closes: Float64Array.from(closes, (c) => Math.min(c, horizon)) };
}

/**
 * A connected random graph: nodes scattered in a box (links of random
 * lengths, some long, some up a "stair" in y), a random tree plus chords,
 * some links walled.
 */
function randomGraph(r, n) {
  const nodes = new Float32Array(n * 3);
  for (let v = 0; v < n; v += 1) {
    nodes[v * 3] = r() * 6;
    nodes[v * 3 + 1] = r() < 0.2 ? r() * 3 : 0;
    nodes[v * 3 + 2] = r() * 4;
  }
  const edges = [];
  const has = new Set();
  const add = (a, b) => {
    if (a === b) return;
    const k = a < b ? `${a},${b}` : `${b},${a}`;
    if (has.has(k)) return;
    has.add(k);
    edges.push(a, b);
  };
  for (let v = 1; v < n; v += 1) add(v, Math.floor(r() * v));
  const chords = Math.floor(r() * n * 1.5);
  for (let i = 0; i < chords; i += 1) add(Math.floor(r() * n), Math.floor(r() * n));
  const m = edges.length / 2;
  const blocked = new Uint8Array(m);
  for (let e = n - 1; e < m; e += 1) if (r() < 0.15) blocked[e] = 1; // only chords: the tree stays open
  return { nodes, edges: Uint32Array.from(edges), blocked };
}

/** A walked route's arrivals, as walkRoute computes them, and whether any is late. */
function late(graph, nodes, closes, depart, speed) {
  return walkRoute(graph, nodes, { forecast: forecastOf(closes, Infinity), depart, speed }).firstUnsafe >= 0;
}

test("the conversion: t < ig(v) exactly when d + (t + 1) tau < s c(v) (move_safe_iff)", () => {
  const r = mulberry32(7);
  for (let i = 0; i < 20000; i += 1) {
    const tau = 0.05 + r() * 2;
    const depart = r() < 0.3 ? 0 : r() * 60;
    const margin = r() < 0.5 ? 1 : 0.1 + 0.9 * r();
    const c = r() * 200;
    const [ig] = forecastIgnition([c], { depart, tau, margin });
    const H = 1 + Math.floor(r() * 400);
    const open = openUntil(Float64Array.of(ig), H)[0];
    for (let t = 0; t < H; t += 1) {
      // rl.js's test, and rlSafe.js's form of it (openUntil, within the step limit).
      const onClock = t < ig;
      assert.equal(t + 1 <= open, onClock);
      // In seconds, with a margin of a few rounding errors either side.
      const arrive = depart + (t + 1) * tau, closes = margin * c;
      if (Math.abs(arrive - closes) > 1e-9 * Math.max(1, closes)) assert.equal(onClock, arrive < closes);
    }
  }
});

test("the clock: every move takes at most tau seconds; virtual nodes close with the sooner end", () => {
  const r = mulberry32(11);
  for (let i = 0; i < 300; i += 1) {
    const g = randomGraph(r, 4 + Math.floor(r() * 20));
    const speed = 0.4 + r() * 1.6;
    for (const clock of ["split", "longest"]) {
      const stepMetres = clock === "split" && r() < 0.5 ? 0.2 + r() : null;
      const clk = stepClock(g, { speed, blocked: g.blocked, clock, stepMetres });
      const len = edgeLengths(clk.graph);
      const m = clk.graph.edges.length / 2;
      for (let e = 0; e < m; e += 1) {
        if (clk.blocked?.[e]) continue;
        assert.ok(len[e] / speed <= clk.tau * (1 + 1e-6), `move of ${len[e] / speed} s over tau ${clk.tau}`);
      }
      if (clock === "longest") {
        assert.equal(clk.virtual, 0);
        assert.ok(Math.abs(clk.tau - (linkStats(g, g.blocked).longest / speed) * (1 + STEP_SLACK)) < 1e-12);
      }
      // The real nodes keep their ids and places; a virtual node lies on its link.
      assert.deepEqual(Array.from(clk.graph.nodes.subarray(0, g.nodes.length)), Array.from(g.nodes));
      const closes = Float64Array.from({ length: g.nodes.length / 3 }, () => r() * 50);
      const c = clockClosings(clk, closes);
      for (let k = 0; k < clk.virtual; k += 1) assert.equal(c[clk.n + k], Math.min(closes[clk.ends[k * 2]], closes[clk.ends[k * 2 + 1]]));
      // Walls are never split, and a split link's length is kept.
      const total = (graph, flags) => {
        const l = edgeLengths(graph);
        let s = 0;
        for (let e = 0; e < l.length; e += 1) if (!flags?.[e]) s += l[e];
        return s;
      };
      assert.ok(Math.abs(total(clk.graph, clk.blocked) - total(g, g.blocked)) < 1e-3);
    }
  }
});

test("realPath drops virtual nodes and merges repeats", () => {
  const clk = { n: 3 };
  assert.deepEqual(realPath(clk, [0, 3, 4, 1, 5, 1, 2]), [0, 1, 2]);
  assert.deepEqual(realPath(clk, [0, 3, 0, 1]), [0, 1]);
});

test("random graphs: no route the learners return, and no earliest escape, is late in seconds (clock_safe)", () => {
  const r = mulberry32(2026);
  let routes = 0, escapes = 0;
  for (let i = 0; i < 400; i += 1) {
    const n = 5 + Math.floor(r() * 25);
    const g = randomGraph(r, n);
    const speed = 0.5 + r() * 1.5;
    const depart = r() < 0.4 ? 0 : r() * 20;
    const margin = r() < 0.5 ? 1 : 0.3 + 0.7 * r();
    const horizon = 20 + r() * 60;
    const closes = Float64Array.from({ length: n }, () => (r() < 0.3 ? horizon : depart + r() * (horizon - depart)));
    const exits = [Math.floor(r() * n)];
    if (r() < 0.5) exits.push(Math.floor(r() * n));
    let start = Math.floor(r() * n);
    while (exits.includes(start)) start = (start + 1) % n;
    const scaled = closes.map((c) => margin * c);
    for (const clock of ["split", "longest"]) {
      const clk = stepClock(g, { speed, blocked: g.blocked, clock, stepMetres: clock === "split" && r() < 0.5 ? 0.3 + r() : null });
      const ig = forecastIgnition(clockClosings(clk, closes), { depart, tau: clk.tau, margin });
      const H = Math.max(1, Math.min(300, stepsToHorizon(horizon, { depart, tau: clk.tau, margin })));
      if (!(depart < scaled[start])) continue;
      const exact = earliestSafeEscape(clk.graph, start, { exits, ignite: ig, horizon: H, blocked: clk.blocked });
      if (exact) {
        escapes += 1;
        assert.equal(late(g, realPath(clk, exact.path), scaled, depart, speed), false, "an earliest escape on the clock is late in seconds");
      }
      const s = learnSafeEscape(clk.graph, start, { exits, ignite: ig, horizon: H, blocked: clk.blocked, trials: 50 });
      if (s.reachedExit) {
        routes += 1;
        assert.equal(late(g, realPath(clk, s.path), scaled, depart, speed), false, "a shielded route is late in seconds");
      }
      const ignite = new Map();
      ig.forEach((x, v) => ignite.set(v, x));
      const q = qLearningPath(clk.graph, start, exits[0], { ignite, episodes: 30, maxSteps: H, seed: i, blocked: clk.blocked });
      if (q.reachedExit) {
        routes += 1;
        assert.equal(late(g, realPath(clk, q.path), scaled, depart, speed), false, "a Q-learning route that escapes on the clock is late in seconds");
      }
    }
  }
  assert.ok(routes > 200 && escapes > 100, `too few cases: ${routes} routes, ${escapes} escapes`);
});

/** A 1 m lattice, w x h, with diagonals (0.5 m links up a stair column if asked). */
function lattice(w, h) {
  const nodes = [], edges = [];
  for (let j = 0; j < h; j += 1) for (let i = 0; i < w; i += 1) nodes.push(i * 0.5, 0, j * 0.5);
  for (let j = 0; j < h; j += 1) {
    for (let i = 0; i < w; i += 1) {
      const v = j * w + i;
      if (i < w - 1) edges.push(v, v + 1);
      if (j < h - 1) edges.push(v, v + w);
      if (i < w - 1 && j < h - 1) edges.push(v, v + w + 1);
    }
  }
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2) };
}

test("the request: certified against the forecast, walked in seconds, never late", () => {
  const g = lattice(12, 6);
  const n = 72;
  const horizon = 60;
  // The fire closes the row j = 0 from x = 3 m on at 4 s; everything else at the horizon.
  const closes = new Float64Array(n).fill(horizon);
  for (let i = 6; i < 12; i += 1) closes[i] = 4;
  const forecast = forecastOf(closes, horizon);
  const opts = { forecast, depart: 0, walkSpeed: 1.2, startPoint: [0, 0, 0], exitPoints: [[5.5, 0, 0], [5.5, 0, 2.5]], firePoint: [3, 0, 0], trials: 500, maxSteps: 200 };
  const r = trainForecastRl(g, opts);
  assert.equal(r.fire_source, "forecast");
  assert.equal(r.method, "shielded");
  assert.ok(r.reached_exit);
  assert.ok(r.certified);
  assert.equal(r.moves, r.escape_steps);
  // The near exit, on the row the fire closes, is not taken.
  assert.equal(r.exit_id, 71);
  assert.equal(r.seconds.late, false);
  assert.ok(r.seconds.slack > 0);
  assert.equal(r.forecast.horizon, horizon);
  assert.equal(r.forecast.clock, "split");
  // The default step is the diagonal: nothing on this lattice is split.
  assert.equal(r.forecast.virtual, 0);
  assert.ok(Math.abs(r.forecast.stepMetres - Math.SQRT1_2) < 1e-6);
  for (const v of r.path) assert.ok(v < n);
  // Split at the median link, diagonals take two moves through a virtual node;
  // the drawn route is still made of real nodes.
  const half = trainForecastRl(g, { ...opts, stepMetres: 0.5 });
  assert.ok(half.forecast.virtual > 0);
  assert.ok(half.reached_exit && half.certified && !half.seconds.late);
  for (const v of half.path) assert.ok(v < n);
  // Every move of the drawn route is a real link.
  const links = new Set();
  for (let e = 0; e < g.edges.length; e += 2) links.add(`${g.edges[e]},${g.edges[e + 1]}`).add(`${g.edges[e + 1]},${g.edges[e]}`);
  for (let i = 1; i < r.path.length; i += 1) assert.ok(links.has(`${r.path[i - 1]},${r.path[i]}`));
  // On the longest-link clock the same.
  const a = trainForecastRl(g, { ...opts, clock: "longest" });
  assert.ok(a.reached_exit && a.certified && !a.seconds.late);
  // Q-learning to the far exit: walked in seconds, not certified (it has no certificate).
  const q = trainForecastRl(g, { ...opts, method: "qlearning", exitPoint: [5.5, 0, 2.5], episodes: 300, seed: 3 });
  assert.equal(q.method, "qlearning");
  assert.equal(q.certified, null);
  if (q.reached_exit) assert.equal(q.seconds.late, false);
  // Leaving after the start has closed: no escape, and the note says why.
  const shut = new Float64Array(closes);
  shut[0] = 2;
  const s = trainForecastRl(g, { ...opts, forecast: forecastOf(shut, horizon), depart: 3 });
  assert.equal(s.start_untenable, true);
  assert.equal(s.reached_exit, false);
  assert.equal(s.escape_steps, null);
  assert.match(s.note, /untenable when the evacuee leaves/);
  // A margin makes the forecast stricter: closing everything by 0.1 x 60 s
  // leaves no time to cross 6 m.
  const m = trainForecastRl(g, { ...opts, margin: 0.1 });
  assert.equal(m.reached_exit, false);
  assert.match(m.note, /No route escapes the forecast fire/);
  assert.throws(() => trainForecastRl(g, { ...opts, margin: 0 }), /margin/);
});

test("the step limit binds before the horizon, and the note says so", () => {
  const g = lattice(30, 2);
  const closes = new Float64Array(60).fill(600);
  const r = trainForecastRl(g, { forecast: forecastOf(closes, 600), startPoint: [0, 0, 0], exitPoint: [14.5, 0, 0], trials: 50, maxSteps: 10 });
  assert.equal(r.reached_exit, false);
  assert.ok(r.forecast.limitBinds);
  assert.match(r.note, /raise Max steps/);
});

test("the hop front is left as it was: trainSafeRl and trainRl take no forecast", () => {
  // The forecast request is a separate function; the hop-front requests are
  // main's (tools/rl-bench.mjs checks their 848 rows against the study's).
  const g = lattice(10, 6);
  const s = trainSafeRl(g, { startPoint: [0, 0, 0], exitPoints: [[4.5, 0, 0], [4.5, 0, 2.5]], firePoint: [2.5, 0, 1], trials: 100, maxSteps: 60 });
  assert.equal(s.fire_source, undefined);
  assert.equal(s.forecast, undefined);
  assert.equal(s.seconds, undefined);
  const q = trainRl(g, { startPoint: [0, 0, 0], exitPoint: [4.5, 0, 2.5], firePoint: [2.5, 0, 1], episodes: 50, maxSteps: 60, seed: 5 });
  assert.equal(q.fire_source, undefined);
  assert.equal(q.seconds, undefined);
});
