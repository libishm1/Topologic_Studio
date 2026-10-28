// Run with: node --test src/core/forecast.property.test.mjs
//
// Property test for FireEgress/Forecast.lean (LG1, LG2; formal/fireegress): the
// forecast routers of forecast.js checked against the statements the Lean file
// proves, on seeded random cases (mulberry32) and on Studio's three fire models.
//   1. doseRate_monotone / doseRate_eq_tConv: convectiveDoseRate is 0 up to 0,
//      non-decreasing, and 1 / (60 tConv).
//   2. linkDose / linkDose_monotone / linkDose_nonneg / window_monotone: the
//      code's frame-by-frame sum equals the integral of the hotter end's step
//      rate (computed here independently; the one identity Lean does not
//      prove), is >= 0, and never falls as the departure time grows.
//   3. dominance_preserved: (t1, D1) <= (t2, D2) and the move allowed from t2
//      => allowed from t1 and the results stay ordered.
//   4. runMax_* / forecast_rate_monotone / closed_iff / closeStep_runMax: for
//      buildForecast with model temperature, flood and radial on small lattices
//      (lattice.js) and hand graphs, the raw fields never cool, the stored
//      rates are the FED rate of the running maximum and never fall, and each
//      node's untenable time is the first step its heat reaches the threshold.
//   5. coarse_conservative / fine_le_coarse / coarse_dose_ge: when the forecast
//      is coarsened (stride > 1), each frame holds its window's last step, and
//      no rate or link dose is below the full-resolution forecast's.
//   6. front_complete / front_sound / front_head_earliest / front_last_leastDose
//      / front_sorted / front_minimal / keepStrict_front: doseFront against
//      brute force over all simple walks, random walks with cycles, walks that
//      pass through exits, and walks that wait (waiting_never_helps), on random
//      graphs, including integer grids where routes tie in arrival time.
//   7. earliestArrival_exact / _settled / _none: the exit label and every time
//      on the returned route are exact earliest safe arrivals; front[0] has the
//      same arrival.
//   8. snapshot_walks_into_fire / snapshot_plan_unsafe on the Lean network, and
//      on random graphs: the snapshot plan is a shortest route of the
//      departure-time graph, and its safety report agrees with the forecast.
//   9. tie_counterexample: on the Lean tie network the front satisfies
//      front_minimal. The code used to keep both tied points (its "keep the
//      strict front" filter was a no-op, Forecast.keepStrict_front); it now
//      keeps only the last of a tie, and the regression tests at the end check
//      the strict trade-off and a departure before time 0.
import assert from "node:assert/strict";
import { test } from "node:test";

import { buildForecast, convectiveDoseRate, doseFront, earliestArrival, linkDose, planWithForecast, walkRoute } from "./forecast.js";
import { AMBIENT, FIRE, floodTimeline, radialTimeline, temperatureSteps } from "./fire.js";
import { buildLatticeGraph } from "./lattice.js";
import { csr, distancesFrom, edgeLengths } from "./route.js";

/** mulberry32, as rngFrom in src/core/rl.js. */
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

const tol = (x) => 1e-9 * Math.max(1, Math.abs(x));

/** A graph from 2-D points (y = 0) and an edge list. */
function graphOf(points, edges) {
  return {
    nodes: Float64Array.from(points.flatMap(([x, z]) => [x, 0, z])),
    edges: Int32Array.from(edges.flat()),
  };
}

/** A forecast with given per-frame rates (K x n) and untenable times. */
function forecastOf(rates, untenable, frameSeconds) {
  const K = rates.length, n = rates[0].length;
  const rate = new Float32Array(K * n);
  rates.forEach((row, k) => row.forEach((r, v) => (rate[k * n + v] = r)));
  const horizon = K * frameSeconds;
  const closes = Float64Array.from(untenable, (u) => Math.min(u, horizon));
  return { n, K, stride: 1, dt: frameSeconds, frameSeconds, horizon, rate, untenable: Float64Array.from(untenable), closes };
}

/** Random rates that never fall (the running maximum of random draws). */
function monotoneRates(rng, K, n, scale = 0.05) {
  const rates = Array.from({ length: K }, () => new Array(n).fill(0));
  for (let v = 0; v < n; v += 1) {
    let r = rng() * scale * 0.02;
    for (let k = 0; k < K; k += 1) {
      r = Math.max(r, rng() < 0.4 ? rng() * scale : 0);
      rates[k][v] = r;
    }
  }
  return rates;
}

/**
 * The integral over [t0, t1] of max(r_v, r_w), r piecewise constant per frame,
 * cut at frame boundaries and read at each piece's midpoint (independent of
 * linkDose's loop). Valid for 0 <= t0 <= t1 <= K f.
 */
function refIntegral(f, v, w, t0, t1) {
  const fs = f.frameSeconds;
  const cuts = [t0];
  for (let k = Math.ceil(t0 / fs); k * fs < t1; k += 1) if (k * fs > t0) cuts.push(k * fs);
  cuts.push(t1);
  let sum = 0;
  for (let i = 0; i + 1 < cuts.length; i += 1) {
    const a = cuts[i], b = cuts[i + 1];
    if (!(b > a)) continue;
    const k = Math.min(f.K - 1, Math.floor((a + b) / 2 / fs));
    sum += (b - a) * Math.max(f.rate[k * f.n + v], f.rate[k * f.n + w]);
  }
  return sum;
}

/** A horizontal rectangle at height z (z up), as two triangles. */
function rect(x0, y0, x1, y1, z) {
  return { positions: Float32Array.from([x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z]), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]) };
}

/** A lattice in the shape the app routes on (route.js reads `nodes`). */
function latticeOf(w, h, spacing = 0.5) {
  const g = buildLatticeGraph({ upAxis: "z", agentHeight: 0, spacing, floors: [rect(0, 0, w, h, 0)] });
  return { ...g, nodes: g.points };
}

/** A random connected-ish graph; integer grids make many routes tie in time. */
function randomGraph(rng, n, { grid = false, p = 0.45 } = {}) {
  if (grid) {
    const cols = 2 + Math.floor(rng() * 3);
    const rows = Math.ceil(n / cols);
    const pts = Array.from({ length: n }, (_, i) => [i % cols, Math.floor(i / cols)]);
    const edges = [];
    for (let i = 0; i < n; i += 1) {
      if (i % cols < cols - 1 && i + 1 < n && rng() < 0.9) edges.push([i, i + 1]);
      if (i + cols < n && rng() < 0.9) edges.push([i, i + cols]);
    }
    void rows;
    return graphOf(pts, edges);
  }
  const pts = Array.from({ length: n }, () => [rng() * 6, rng() * 6]);
  const edges = [];
  for (let a = 0; a < n; a += 1) for (let b = a + 1; b < n; b += 1) if (rng() < p) edges.push([a, b]);
  return graphOf(pts, edges);
}

/** Every simple safe walk from s (no waiting), not stopping at exits. */
function simpleWalks(g, s, f, speed, depart) {
  const { indptr, neighbours, edgeIds, n } = csr(g);
  const len = edgeLengths(g);
  const out = [];
  const onPath = new Uint8Array(n);
  const path = [s];
  const walk = (v, t, d) => {
    out.push({ node: v, arrival: t, dose: d, nodes: path.slice() });
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      const w = neighbours[k];
      if (onPath[w]) continue;
      const t2 = t + len[edgeIds[k]] / speed;
      if (!(t2 < f.closes[w])) continue;
      onPath[w] = 1;
      path.push(w);
      walk(w, t2, d + linkDose(f, v, w, t, t2));
      path.pop();
      onPath[w] = 0;
    }
  };
  if (depart < f.closes[s]) {
    onPath[s] = 1;
    walk(s, depart, 0);
  }
  return out;
}

/**
 * Random safe walks from s that may revisit nodes, pass through exits and wait
 * (waiting at v from t to t' < closes[v] accrues the node's own rate, which is
 * linkDose(v, v, t, t')).
 */
function randomWalks(rng, g, s, f, speed, depart, count, { waits = false } = {}) {
  const { indptr, neighbours, edgeIds } = csr(g);
  const len = edgeLengths(g);
  const out = [];
  if (!(depart < f.closes[s])) return out;
  for (let c = 0; c < count; c += 1) {
    let v = s, t = depart, d = 0;
    const steps = 1 + Math.floor(rng() * 12);
    for (let i = 0; i < steps; i += 1) {
      if (waits && rng() < 0.3) {
        const t2 = t + rng() * (f.closes[v] - t);
        if (t2 > t && t2 < f.closes[v]) {
          d += linkDose(f, v, v, t, t2);
          t = t2;
        }
      }
      const deg = indptr[v + 1] - indptr[v];
      if (!deg) break;
      const k = indptr[v] + Math.floor(rng() * deg);
      const w = neighbours[k];
      const t2 = t + len[edgeIds[k]] / speed;
      if (!(t2 < f.closes[w])) break;
      d += linkDose(f, v, w, t, t2);
      t = t2;
      v = w;
      out.push({ node: v, arrival: t, dose: d });
    }
  }
  return out;
}

const dominated = (front, r) => front.some((p) => p.arrival <= r.arrival + tol(r.arrival) && p.dose <= r.dose + tol(r.dose));

// ---------------------------------------------------------------------------

test("FED rate: 0 up to 0, non-decreasing, and 1 / (60 tConv) (doseRate_*)", () => {
  const rng = mulberry32(0x5eed01);
  for (const T of [-50, -1, 0]) assert.equal(convectiveDoseRate(T), 0);
  const xs = Array.from({ length: 2000 }, () => (rng() - 0.2) * 400).sort((a, b) => a - b);
  for (let i = 1; i < xs.length; i += 1) assert.ok(convectiveDoseRate(xs[i]) >= convectiveDoseRate(xs[i - 1]));
  for (const T of xs.filter((x) => x > 0)) {
    const tConv = 5e7 * Math.pow(T, -3.4); // minutes
    assert.ok(Math.abs(convectiveDoseRate(T) - 1 / tConv / 60) <= 1e-12 * convectiveDoseRate(T));
  }
});

test("linkDose is the integral of the hotter end's step rate, >= 0, non-decreasing in the departure time", () => {
  const rng = mulberry32(0x11d05e);
  let checks = 0;
  for (let trial = 0; trial < 300; trial += 1) {
    const n = 2 + Math.floor(rng() * 4);
    const K = 1 + Math.floor(rng() * 8);
    const fs = 0.25 + rng() * 4;
    const f = forecastOf(monotoneRates(rng, K, n), new Array(n).fill(Infinity), fs);
    const v = Math.floor(rng() * n), w = Math.floor(rng() * n);
    const tau = rng() < 0.1 ? 0 : rng() * f.horizon * 0.6;
    // Sweep departures over [0, horizon - tau], with frame boundaries hit exactly.
    const ts = [];
    for (let i = 0; i <= 60; i += 1) ts.push(((f.horizon - tau) * i) / 60);
    for (let k = 0; k * fs <= f.horizon - tau; k += 1) ts.push(k * fs);
    ts.sort((a, b) => a - b);
    let prev = -Infinity;
    for (const t of ts) {
      const d = linkDose(f, v, w, t, t + tau);
      const ref = refIntegral(f, v, w, t, t + tau);
      assert.ok(Math.abs(d - ref) <= 1e-12 * Math.max(1, ref) + 1e-15, `trial ${trial}: sum ${d} vs integral ${ref}`);
      assert.ok(d >= 0);
      assert.ok(d >= prev - 1e-12 * Math.max(1, Math.abs(d)), `trial ${trial}: dose fell from ${prev} to ${d} at t=${t}`);
      prev = d;
      checks += 1;
    }
  }
  assert.ok(checks > 15000);
});

test("dominance is preserved by extension (dominance_preserved)", () => {
  const rng = mulberry32(0xd0d0);
  let allowedBoth = 0, allowedOnlyEarly = 0;
  for (let trial = 0; trial < 4000; trial += 1) {
    const n = 2 + Math.floor(rng() * 3);
    const K = 1 + Math.floor(rng() * 6);
    const fs = 0.5 + rng() * 3;
    const untenable = Array.from({ length: n }, () => (rng() < 0.6 ? rng() * K * fs : Infinity));
    const f = forecastOf(monotoneRates(rng, K, n), untenable, fs);
    const v = Math.floor(rng() * n), w = Math.floor(rng() * n);
    const tau = rng() * fs * 2;
    const t2 = rng() * f.horizon, t1 = t2 * rng();
    const D2 = rng(), D1 = D2 * rng();
    const ok2 = t2 + tau < f.closes[w], ok1 = t1 + tau < f.closes[w];
    if (ok2) {
      assert.ok(ok1, "a move allowed later must be allowed earlier");
      assert.ok(t1 + tau <= t2 + tau);
      const e1 = D1 + linkDose(f, v, w, t1, t1 + tau), e2 = D2 + linkDose(f, v, w, t2, t2 + tau);
      assert.ok(e1 <= e2 + tol(e2), `trial ${trial}: ${e1} > ${e2}`);
      allowedBoth += 1;
    } else if (ok1) allowedOnlyEarly += 1;
  }
  assert.ok(allowedBoth > 500 && allowedOnlyEarly > 100);
});

/** The model's raw field per step, as buildForecast reads it. */
function rawFields(graph, model, seed, K0, blocked) {
  const n = graph.nodes.length / 3;
  if (model === "temperature") {
    const frames = [];
    for (const frame of temperatureSteps(graph, seed, { maxSteps: K0, blocked })) frames.push(frame);
    return (k, v) => frames[k][v];
  }
  const timeline = model === "flood" ? floodTimeline(graph, seed, K0, blocked) : radialTimeline(graph, seed, K0);
  const ignite = new Int32Array(n).fill(-1);
  timeline.forEach((nodes, k) => nodes.forEach((v) => (ignite[v] = k)));
  return (k, v) => (ignite[v] >= 0 && ignite[v] <= k ? FIRE : AMBIENT);
}

test("Studio's three fire models: fields never cool, rates never fall, closing is first passage", () => {
  const rng = mulberry32(0xf12e);
  const graphs = [
    latticeOf(4, 3),
    latticeOf(3, 3, 0.75),
    graphOf([[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1], [3, 0]], [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [1, 4], [2, 6]]),
    graphOf([0, 1, 2, 3, 4, 5, 6].map((x) => [x, 0]), [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]),
  ];
  let cases = 0;
  for (const g of graphs) {
    const n = g.nodes.length / 3;
    for (const model of ["temperature", "flood", "radial"]) {
      for (let rep = 0; rep < 3; rep += 1) {
        const seed = Math.floor(rng() * n);
        const steps = 5 + Math.floor(rng() * 40);
        const dt = 0.5 + rng() * 5;
        const threshold = model === "temperature" ? 30 + rng() * 90 : FIRE;
        const blocked = g.blocked ?? null;
        const f = buildForecast(g, { model, seed, steps, dt, threshold, blocked });
        assert.equal(f.stride, 1);
        assert.equal(f.K, steps);
        const field = rawFields(g, model, seed, steps, blocked);
        for (let v = 0; v < n; v += 1) {
          let peak = -Infinity, first = Infinity;
          for (let k = 0; k < steps; k += 1) {
            const T = field(k, v);
            // The header's claim: the models never cool a node (runMax_of_monotone).
            if (k > 0) assert.ok(T >= field(k - 1, v), `${model}: node ${v} cooled at step ${k}`);
            peak = Math.max(peak, T);
            if (first === Infinity && peak >= threshold) first = k * dt;
            // Stored rate = FED rate of the running maximum (forecast_rate_monotone).
            assert.equal(f.rate[k * n + v], Math.fround(convectiveDoseRate(peak)));
            assert.ok(f.rate[k * n + v] >= 0);
            if (k > 0) assert.ok(f.rate[k * n + v] >= f.rate[(k - 1) * n + v]);
          }
          // closeStep_runMax / closed_iff: untenable is the first passage of the raw field.
          assert.equal(f.untenable[v], first);
          assert.equal(f.closes[v], Math.min(first, steps * dt));
          // closed_at_iff: before the horizon, closes <= t iff the step holding t has reached the threshold.
          for (let i = 0; i < 10; i += 1) {
            const t = rng() * steps * dt;
            const k = Math.floor(t / dt);
            if (k >= steps) continue;
            let pk = -Infinity;
            for (let j = 0; j <= k; j += 1) pk = Math.max(pk, field(j, v));
            assert.equal(f.closes[v] <= t, pk >= threshold, `${model}: node ${v} at t=${t}`);
          }
        }
        cases += 1;
      }
    }
  }
  assert.equal(cases, 36);
});

test("coarsening holds each window's last state: no rate or link dose below the full forecast", () => {
  // K0 * n above 16M forces stride 2 (buildForecast's MAX_CELLS).
  const g = latticeOf(20, 25);
  const n = g.nodes.length / 3;
  const K0 = Math.floor(16_000_000 / n) + 50;
  const rng = mulberry32(0xc0a25e);
  for (const model of ["temperature", "flood", "radial"]) {
    const dt = 0.5 + rng();
    const seed = Math.floor(rng() * n);
    const threshold = model === "temperature" ? 60 : FIRE;
    const f = buildForecast(g, { model, seed, steps: K0, dt, threshold, blocked: g.blocked });
    assert.ok(f.stride >= 2, `${model}: expected coarsening, stride ${f.stride}`);
    assert.equal(f.K, Math.ceil(K0 / f.stride));
    assert.equal(f.horizon, K0 * dt);
    const field = rawFields(g, model, seed, K0, g.blocked);
    // Fine rates for a few nodes, for the dose comparison.
    const watch = [seed, (seed + 1) % n, Math.floor(n / 2), n - 1];
    const fine = new Map(watch.map((v) => [v, new Float32Array(K0)]));
    const peak = new Float64Array(n).fill(-Infinity);
    let ties = 0;
    for (let k = 0; k < K0; k += 1) {
      const frame = Math.floor(k / f.stride);
      const end = Math.min((frame + 1) * f.stride - 1, K0 - 1); // windowEnd
      for (let v = 0; v < n; v += 1) {
        const T = field(k, v);
        if (T > peak[v]) peak[v] = T;
        const r = Math.fround(convectiveDoseRate(peak[v]));
        // coarse_conservative: the frame is at least every step of its window.
        assert.ok(f.rate[frame * n + v] >= r);
        // ... and equals the window's last step.
        if (k === end) {
          assert.equal(f.rate[frame * n + v], r);
          ties += 1;
        }
        if (fine.has(v)) fine.get(v)[k] = r;
      }
    }
    assert.ok(ties >= f.K * n);
    // coarse_dose_ge: link doses (hotter of two ends) never below the full forecast's.
    const fineF = { n: 1, K: K0, frameSeconds: dt, rate: null };
    const fineDose = (v, w, t0, t1) => {
      const rv = fine.get(v), rw = fine.get(w);
      const pair = new Float32Array(K0);
      for (let k = 0; k < K0; k += 1) pair[k] = Math.max(rv[k], rw[k]);
      return refIntegral({ ...fineF, rate: pair }, 0, 0, t0, t1);
    };
    for (let i = 0; i < 40; i += 1) {
      const v = watch[Math.floor(rng() * watch.length)], w = watch[Math.floor(rng() * watch.length)];
      const t0 = rng() * f.horizon * 0.9, t1 = t0 + rng() * (f.horizon - t0);
      const coarse = linkDose(f, v, w, t0, t1);
      const ref = fineDose(v, w, t0, t1);
      assert.ok(coarse >= ref - tol(ref), `${model}: coarse dose ${coarse} below fine ${ref}`);
    }
  }
});

test("doseFront against brute force: complete, sound, first = earliest, last = least dose, minimal up to ties", () => {
  const rng = mulberry32(0xbeef2026);
  let cases = 0, fronts = 0, multi = 0, tiedFronts = 0, extraWalks = 0;
  for (let trial = 0; trial < 500; trial += 1) {
    const grid = trial % 3 === 0;
    const n = 5 + Math.floor(rng() * 5);
    const g = randomGraph(rng, n, { grid });
    if (!g.edges.length) continue;
    const K = 1 + Math.floor(rng() * 6);
    const fs = 0.5 + rng() * 3;
    const untenable = Array.from({ length: n }, () => (rng() < 0.3 ? rng() * K * fs : Infinity));
    const f = forecastOf(monotoneRates(rng, K, n), untenable, fs);
    const s = 0;
    const exits = [n - 1, ...(rng() < 0.4 ? [Math.floor(n / 2)] : [])].filter((x) => x !== s);
    const isExit = new Set(exits);
    const speed = grid ? 1 : 0.8 + rng();
    const depart = rng() < 0.3 ? rng() * fs : 0;

    const walks = simpleWalks(g, s, f, speed, depart);
    const atExit = walks.filter((r) => isExit.has(r.node));
    const { front, truncated } = doseFront(g, s, { exits, forecast: f, speed, depart });
    const ea = earliestArrival(g, s, depart, { exits, closes: f.closes, speed });
    assert.equal(truncated, false);
    cases += 1;
    if (!atExit.length) {
      // front_empty_iff, earliestArrival_none.
      assert.equal(front.length, 0, `trial ${trial}: no safe walk but a front`);
      assert.equal(ea.exit, -1);
      continue;
    }
    fronts += 1;
    if (front.length > 1) multi += 1;
    const minDose = Math.min(...atExit.map((r) => r.dose));
    const minTime = Math.min(...atExit.map((r) => r.arrival));
    // front_head_earliest, front_last_leastDose, earliestArrival_exact.
    assert.ok(Math.abs(front[0].arrival - minTime) <= tol(minTime), `trial ${trial}: fastest`);
    assert.ok(Math.abs(front[front.length - 1].dose - minDose) <= tol(minDose), `trial ${trial}: least dose`);
    assert.ok(ea.exit >= 0 && Math.abs(ea.arrive[ea.exit] - minTime) <= tol(minTime));
    // earliestArrival_settled: every time on the returned route is that node's earliest safe arrival.
    const earliestAt = new Float64Array(n).fill(Infinity);
    for (const r of walks) earliestAt[r.node] = Math.min(earliestAt[r.node], r.arrival);
    let c = ea.exit;
    while (c >= 0) {
      assert.ok(Math.abs(ea.arrive[c] - earliestAt[c]) <= tol(earliestAt[c]), `trial ${trial}: route time at ${c}`);
      c = ea.parent[c];
    }
    // front_complete, against simple walks (through exits too), random walks with
    // cycles, and walks that wait (waiting_never_helps).
    const extra = [...randomWalks(rng, g, s, f, speed, depart, 60), ...randomWalks(rng, g, s, f, speed, depart, 60, { waits: true })]
      .filter((r) => isExit.has(r.node));
    extraWalks += extra.length;
    for (const r of [...atExit, ...extra]) assert.ok(dominated(front, r), `trial ${trial}: a walk escapes the front`);
    // front_sound: every point is a safe walk's, and its route walks to it.
    for (const p of front) {
      assert.ok(atExit.some((r) => Math.abs(r.arrival - p.arrival) <= tol(p.arrival) && Math.abs(r.dose - p.dose) <= tol(p.dose)));
      const w = walkRoute(g, p.nodes, { forecast: f, depart, speed });
      assert.equal(w.firstUnsafe, -1);
      assert.ok(Math.abs(w.times[w.times.length - 1] - p.arrival) <= tol(p.arrival));
      assert.ok(Math.abs(w.dose - p.dose) <= tol(p.dose));
      assert.ok(isExit.has(p.nodes[p.nodes.length - 1]));
    }
    // front_sorted (and so keepStrict_front: the final filter removed nothing).
    for (let i = 1; i < front.length; i += 1) assert.ok(front[i].arrival >= front[i - 1].arrival && front[i].dose < front[i - 1].dose);
    // front_minimal: a point is beaten only by a later point with the same arrival.
    for (let i = 0; i < front.length; i += 1) {
      const p = front[i];
      for (const r of atExit) {
        const beats = r.arrival <= p.arrival && r.dose <= p.dose && (r.arrival < p.arrival || r.dose < p.dose);
        if (!beats) continue;
        assert.ok(r.arrival === p.arrival, `trial ${trial}: front point beaten in time`);
        assert.ok(front.slice(i + 1).some((q) => q.arrival === p.arrival && q.dose <= r.dose), `trial ${trial}: front_minimal`);
      }
    }
    if (front.some((p, i) => i + 1 < front.length && front[i + 1].arrival === p.arrival)) tiedFronts += 1;
  }
  console.log(JSON.stringify({ cases, fronts, multi, tiedFronts, extraWalks }));
  assert.ok(multi > 20, "the random cases should include real trade-offs");
});

/** The network of snapshot_walks_into_fire (and of forecast.test.mjs). */
function snapshotCase() {
  const g = graphOf([[0, 0], [2, 0], [4, 0], [0, 3], [4, 3]], [[0, 1], [1, 2], [0, 3], [3, 4], [4, 2]]);
  const f = forecastOf([[0, 0, 0, 0, 0]], [Infinity, 1.5, Infinity, Infinity, Infinity], 100);
  return { g, f };
}

test("snapshot plan on the Lean network: every route under 10 s passes node 1 late; the earliest safe arrival is 10 s", () => {
  const { g, f } = snapshotCase();
  const plan = planWithForecast(g, { start: 0, exits: [2], forecast: f, speed: 1 });
  assert.equal(plan.snapshot.safe, false);
  assert.deepEqual(plan.snapshot.nodes, [0, 1, 2]);
  assert.equal(plan.snapshot.arrival, 4);
  assert.ok(Math.abs(plan.snapshot.lateBy - 0.5) < 1e-12);
  assert.equal(plan.earliest.arrival, 10);
  assert.deepEqual(plan.earliest.nodes, [0, 3, 4, 2]);
  // snapshot_plan_unsafe: every simple route to the exit shorter than 10 s is unsafe, at node 1.
  const open = forecastOf([[0, 0, 0, 0, 0]], [Infinity, Infinity, Infinity, Infinity, Infinity], 100);
  const routes = simpleWalks(g, 0, open, 1, 0).filter((r) => r.node === 2);
  assert.ok(routes.length >= 2);
  for (const r of routes) {
    const w = walkRoute(g, r.nodes, { forecast: f, speed: 1 });
    if (r.arrival < 10) {
      assert.ok(w.firstUnsafe >= 0 && r.nodes[w.firstUnsafe] === 1 && w.times[w.firstUnsafe] > 1.5);
    }
  }
});

test("snapshot plan on random graphs: a shortest route of the departure-time graph, honestly reported", () => {
  const rng = mulberry32(0x5a95);
  let unsafe = 0, safe = 0;
  for (let trial = 0; trial < 300; trial += 1) {
    const n = 5 + Math.floor(rng() * 6);
    const g = randomGraph(rng, n);
    if (!g.edges.length) continue;
    const K = 1 + Math.floor(rng() * 4);
    const fs = 0.5 + rng() * 3;
    const untenable = Array.from({ length: n }, () => (rng() < 0.5 ? rng() * K * fs : Infinity));
    const f = forecastOf(monotoneRates(rng, K, n), untenable, fs);
    const exits = [n - 1, ...(rng() < 0.5 ? [n - 2] : [])];
    const depart = rng() < 0.5 ? rng() * fs : 0;
    const speed = 0.8 + rng();
    const plan = planWithForecast(g, { start: 0, exits, forecast: f, speed, depart });
    // The snapshot graph: links touching a node untenable at departure are closed.
    const m = g.edges.length / 2;
    const closedNow = new Uint8Array(m);
    for (let e = 0; e < m; e += 1) if (!(depart < f.untenable[g.edges[2 * e]]) || !(depart < f.untenable[g.edges[2 * e + 1]])) closedNow[e] = 1;
    const dist = distancesFrom(g, 0, { blocked: closedNow });
    const nearest = Math.min(...exits.map((x) => dist[x]));
    if (!Number.isFinite(nearest)) {
      assert.equal(plan.snapshot, null);
      continue;
    }
    const sn = plan.snapshot;
    assert.ok(sn, `trial ${trial}: snapshot missing`);
    const w = walkRoute(g, sn.nodes, { forecast: f, depart, speed });
    // A shortest route to the nearest exit (static Dijkstra: TimeDependent.static_hyps).
    assert.ok(Math.abs((w.times[w.times.length - 1] - depart) * speed - nearest) <= 1e-9 * Math.max(1, nearest));
    assert.equal(sn.safe, w.firstUnsafe < 0);
    if (sn.safe) {
      safe += 1;
      // A safe snapshot route is a safe walk: the front beats or matches it, and earliest is no later.
      assert.ok(dominated(plan.front, { arrival: sn.arrival, dose: w.dose }));
      assert.ok(plan.earliest && plan.earliest.arrival <= sn.arrival + tol(sn.arrival));
    } else {
      unsafe += 1;
      assert.ok(Math.abs(sn.lateBy - (w.times[w.firstUnsafe] - f.closes[sn.nodes[w.firstUnsafe]])) < 1e-12);
      assert.ok(sn.lateBy >= 0);
    }
  }
  console.log(JSON.stringify({ safe, unsafe }));
  assert.ok(safe > 20 && unsafe > 20);
});

/** The network of tie_counterexample: 0 start, 1 hot room, 2 cool room, 3 exit; every link 1 s. */
function tieCase() {
  const g = graphOf([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1], [0, 2], [1, 3], [2, 3]]);
  // A move touching node 1 costs dose 1 (rate 1 per second for 1 s), any other 0.
  const f = forecastOf([[0, 1, 0, 0]], [Infinity, Infinity, Infinity, Infinity], 100);
  return { g, f };
}

test("tie network (tie_counterexample): the front is sound, complete and minimal up to the tie", () => {
  const { g, f } = tieCase();
  const { front } = doseFront(g, 0, { exits: [3], forecast: f, speed: 1 });
  const all = simpleWalks(g, 0, f, 1, 0).filter((r) => r.node === 3);
  for (const r of all) assert.ok(dominated(front, r));
  assert.equal(front[0].arrival, 2);
  assert.equal(front[front.length - 1].dose, 0);
  for (let i = 0; i < front.length; i += 1) {
    for (const r of all) {
      if (r.arrival <= front[i].arrival && r.dose < front[i].dose) {
        assert.ok(front.slice(i + 1).some((q) => q.arrival === front[i].arrival && q.dose <= r.dose));
      }
    }
  }
  console.log(JSON.stringify(front.map((p) => ({ nodes: p.nodes, arrival: p.arrival, dose: p.dose }))));
});

test(
  "regression (tie_counterexample): no front point is beaten by another at the same arrival",
  () => {
    const { g, f } = tieCase();
    const { front } = doseFront(g, 0, { exits: [3], forecast: f, speed: 1 });
    for (let i = 1; i < front.length; i += 1) assert.ok(front[i].arrival > front[i - 1].arrival, `points ${i - 1} and ${i} tie at ${front[i].arrival} s`);
    const plan = planWithForecast(g, { start: 0, exits: [3], forecast: f, speed: 1 });
    assert.equal(plan.fastest.dose, 0, `the plan's fastest route [${plan.fastest.nodes}] takes dose ${plan.fastest.dose} when [0,2,3] arrives as early with dose 0`);
  },
);

test(
  "regression: a departure before time 0 still finds the route earliestArrival finds",
  () => {
    const { g, f } = tieCase();
    const ea = earliestArrival(g, 0, -1, { exits: [3], closes: f.closes, speed: 1 });
    assert.equal(ea.exit, 3);
    const { front } = doseFront(g, 0, { exits: [3], forecast: f, speed: 1, depart: -1 });
    assert.ok(front.length > 0);
  },
);
