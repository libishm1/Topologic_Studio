// Run with: node --test src/core/forecast.test.mjs
//
// The forecast routers (forecast.js) against hand-made cases and against brute
// force over every simple route on random graphs with random heat that never
// falls (the hypothesis of TimeDependent.lean and MinDose.lean).
import test from "node:test";
import assert from "node:assert/strict";

import { buildForecast, convectiveDoseRate, doseFront, earliestArrival, linkDose, planWithForecast, walkRoute } from "./forecast.js";
import { csr, edgeLengths } from "./route.js";

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

/** A graph from 2-D points (y = 0) and an edge list. */
function graphOf(points, edges) {
  return {
    nodes: Float64Array.from(points.flatMap(([x, z]) => [x, 0, z])),
    edges: Int32Array.from(edges.flat()),
  };
}

/** A forecast with given per-step rates (K x n, never falling) and closing times. */
function forecastOf(rates, untenable, frameSeconds) {
  const K = rates.length, n = rates[0].length;
  const rate = new Float32Array(K * n);
  rates.forEach((row, k) => row.forEach((r, v) => (rate[k * n + v] = r)));
  const horizon = K * frameSeconds;
  const closes = Float64Array.from(untenable, (u) => Math.min(u, horizon));
  return { n, K, stride: 1, dt: frameSeconds, frameSeconds, horizon, rate, untenable: Float64Array.from(untenable), closes };
}

test("convective FED rate matches t_I = 5e7 T^-3.4 min", () => {
  for (const T of [20, 60, 100, 120]) {
    const tI = 5e7 * Math.pow(T, -3.4); // minutes
    assert.ok(Math.abs(1 / convectiveDoseRate(T) / 60 - tI) / tI < 1e-12);
  }
  assert.ok(convectiveDoseRate(120) > convectiveDoseRate(60));
  // About 4.3 minutes to FED 1 at 120 C.
  assert.ok(Math.abs(1 / convectiveDoseRate(120) / 60 - 4.27) < 0.01);
});

test("a flood forecast closes each node when the fire reaches it", () => {
  // A line of 6 nodes, 1 m apart, fire at node 0.
  const g = graphOf([0, 1, 2, 3, 4, 5].map((x) => [x, 0]), [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]);
  const f = buildForecast(g, { model: "flood", seed: 0, steps: 4, dt: 10 });
  assert.deepEqual(Array.from(f.untenable), [0, 10, 20, 30, Infinity, Infinity]);
  assert.equal(f.horizon, 40);
  assert.deepEqual(Array.from(f.closes), [0, 10, 20, 30, 40, 40]);
  // A burning node's rate is the 120 C rate from its ignition step on.
  assert.equal(f.rate[2 * f.n + 1], Math.fround(convectiveDoseRate(120)));
  assert.equal(f.rate[0 * f.n + 1], Math.fround(convectiveDoseRate(20)));
});

test("the temperature forecast never cools, and coarsening holds the hottest state", () => {
  const n = 30;
  const pts = Array.from({ length: n }, (_, i) => [i % 6, Math.floor(i / 6)]);
  const edges = [];
  for (let i = 0; i < n; i += 1) {
    if (i % 6 < 5) edges.push([i, i + 1]);
    if (i + 6 < n) edges.push([i, i + 6]);
  }
  const g = graphOf(pts, edges);
  const f = buildForecast(g, { model: "temperature", seed: 0, steps: 40, dt: 2, threshold: 60 });
  for (let v = 0; v < n; v += 1) for (let k = 1; k < f.K; k += 1) assert.ok(f.rate[k * n + v] >= f.rate[(k - 1) * n + v]);
  assert.equal(f.untenable[0], 0);
  assert.ok(f.untenable[1] > 0 && f.untenable[1] < Infinity);
});

test("the forecast router walks past a fire that arrives after it has gone", () => {
  // Short way 0-1-2 (2 m) through node 1, which closes at 3 s; long way
  // 0-3-4-2 (about 4.8 m). At 1 m/s the walker is at node 1 at 1 s: safe.
  const g = graphOf([[0, 0], [1, 0], [2, 0], [0, 1.4], [2, 1.4]], [[0, 1], [1, 2], [0, 3], [3, 4], [4, 2]]);
  const f = forecastOf([[0, 0, 0, 0, 0]], [Infinity, 3, Infinity, Infinity, Infinity], 100);
  const r = earliestArrival(g, 0, 0, { exits: [2], closes: f.closes, speed: 1 });
  assert.equal(r.exit, 2);
  assert.ok(Math.abs(r.arrive[2] - 2) < 1e-9);
  // Leaving at 2.5 s the walker would reach node 1 at 3.5 s: the long way.
  const late = earliestArrival(g, 0, 2.5, { exits: [2], closes: f.closes, speed: 1 });
  assert.ok(late.arrive[2] > 2.5 + 4.7);
  assert.equal(late.parent[2], 4);
});

test("a route planned on the fire at departure can walk into it; the forecast route does not", () => {
  // Node 1 is fine now but closes at 1.5 s, before the walker reaches it at 2 s.
  const g = graphOf([[0, 0], [2, 0], [4, 0], [0, 3], [4, 3]], [[0, 1], [1, 2], [0, 3], [3, 4], [4, 2]]);
  const f = forecastOf([[0, 0, 0, 0, 0]], [Infinity, 1.5, Infinity, Infinity, Infinity], 100);
  const plan = planWithForecast(g, { start: 0, exits: [2], forecast: f, speed: 1 });
  assert.equal(plan.snapshot.safe, false);
  assert.ok(Math.abs(plan.snapshot.lateBy - 0.5) < 1e-9);
  assert.deepEqual(plan.earliest.nodes, [0, 3, 4, 2]);
  assert.ok(Math.abs(plan.earliest.arrival - 10) < 1e-9);
  assert.equal(walkRoute(g, plan.earliest.nodes, { forecast: f, speed: 1 }).firstUnsafe, -1);
});

test("the least-dose route can be slower than the fastest", () => {
  // Fast way 0-1-2 past a hot node 1; slow way 0-3-4-2 stays cool.
  const g = graphOf([[0, 0], [1, 0], [2, 0], [0, 2], [2, 2]], [[0, 1], [1, 2], [0, 3], [3, 4], [4, 2]]);
  const cool = 1e-5, hot = 1e-2;
  const f = forecastOf([[cool, hot, cool, cool, cool]], [Infinity, Infinity, Infinity, Infinity, Infinity], 1000);
  const { front } = doseFront(g, 0, { exits: [2], forecast: f, speed: 1 });
  assert.equal(front.length, 2);
  assert.deepEqual(front[0].nodes, [0, 1, 2]);
  assert.deepEqual(front[1].nodes, [0, 3, 4, 2]);
  assert.ok(front[1].dose < front[0].dose && front[1].arrival > front[0].arrival);
});

/** Every simple route from s to an exit, walked without waiting. */
function bruteForce(g, s, exits, f, speed, depart) {
  const { indptr, neighbours, edgeIds } = csr(g);
  const len = edgeLengths(g);
  const isExit = new Set(exits);
  const out = [];
  const onPath = new Set([s]);
  const walk = (v, t, d) => {
    if (isExit.has(v)) {
      out.push({ arrival: t, dose: d });
      return;
    }
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      const w = neighbours[k];
      if (onPath.has(w)) continue;
      const t2 = t + len[edgeIds[k]] / speed;
      if (!(t2 < f.closes[w])) continue;
      onPath.add(w);
      walk(w, t2, d + linkDose(f, v, w, t, t2));
      onPath.delete(w);
    }
  };
  if (depart < f.closes[s]) walk(s, depart, 0);
  return out;
}

test("random graphs: the front is exactly the time-dose trade-off of all simple routes", () => {
  const rng = mulberry32(20261027);
  let cases = 0, fronts = 0, multi = 0;
  for (let trial = 0; trial < 400; trial += 1) {
    const n = 4 + Math.floor(rng() * 6);
    const pts = Array.from({ length: n }, () => [rng() * 6, rng() * 6]);
    const edges = [];
    for (let a = 0; a < n; a += 1) for (let b = a + 1; b < n; b += 1) if (rng() < 0.45) edges.push([a, b]);
    if (!edges.length) continue;
    const g = graphOf(pts, edges);
    const K = 1 + Math.floor(rng() * 5);
    const frame = 0.5 + rng() * 3;
    // Heat that never falls: running maximum of random rates.
    const rates = Array.from({ length: K }, () => new Array(n).fill(0));
    for (let v = 0; v < n; v += 1) {
      let r = rng() * 1e-3;
      for (let k = 0; k < K; k += 1) {
        r = Math.max(r, rng() < 0.4 ? rng() * 0.05 : 0);
        rates[k][v] = r;
      }
    }
    const untenable = Array.from({ length: n }, () => (rng() < 0.3 ? rng() * K * frame : Infinity));
    const f = forecastOf(rates, untenable, frame);
    const s = 0;
    const exits = [n - 1, ...(rng() < 0.3 ? [n - 2] : [])].filter((x) => x !== s);
    const speed = 0.8 + rng();
    const depart = rng() < 0.3 ? rng() * frame : 0;

    const all = bruteForce(g, s, exits, f, speed, depart);
    const { front, truncated } = doseFront(g, s, { exits, forecast: f, speed, depart });
    const ea = earliestArrival(g, s, depart, { exits, closes: f.closes, speed });
    assert.equal(truncated, false);
    cases += 1;
    if (!all.length) {
      assert.equal(front.length, 0, `trial ${trial}: no route exists but the front is not empty`);
      assert.equal(ea.exit, -1);
      continue;
    }
    fronts += 1;
    if (front.length > 1) multi += 1;
    const tol = (x) => 1e-9 * Math.max(1, Math.abs(x));
    const minDose = Math.min(...all.map((r) => r.dose));
    const minTime = Math.min(...all.map((r) => r.arrival));
    const last = front[front.length - 1];
    assert.ok(Math.abs(last.dose - minDose) <= tol(minDose), `trial ${trial}: least dose ${last.dose} vs ${minDose}`);
    assert.ok(Math.abs(front[0].arrival - minTime) <= tol(minTime), `trial ${trial}: fastest ${front[0].arrival} vs ${minTime}`);
    assert.ok(Math.abs(ea.arrive[ea.exit] - minTime) <= tol(minTime), `trial ${trial}: TD Dijkstra ${ea.arrive[ea.exit]} vs ${minTime}`);
    // Every route is matched or beaten on both counts by a front point, and
    // every front point is some route's (arrival, dose).
    for (const r of all) {
      assert.ok(front.some((p) => p.arrival <= r.arrival + tol(r.arrival) && p.dose <= r.dose + tol(r.dose)), `trial ${trial}: a route escapes the front`);
    }
    for (const p of front) {
      assert.ok(all.some((r) => Math.abs(r.arrival - p.arrival) <= tol(p.arrival) && Math.abs(r.dose - p.dose) <= tol(p.dose)), `trial ${trial}: front point is no route`);
      // The reported route walks to the reported numbers.
      const w = walkRoute(g, p.nodes, { forecast: f, depart, speed });
      assert.equal(w.firstUnsafe, -1);
      assert.ok(Math.abs(w.times[w.times.length - 1] - p.arrival) <= tol(p.arrival));
      assert.ok(Math.abs(w.dose - p.dose) <= tol(p.dose));
    }
    for (let i = 1; i < front.length; i += 1) assert.ok(front[i].arrival >= front[i - 1].arrival && front[i].dose < front[i - 1].dose);
  }
  console.log(JSON.stringify({ cases, fronts, multi }));
  assert.ok(multi > 10, "the random cases should include real trade-offs");
});
