// Run with: node --test src/core/deadline.property.test.mjs
//
// Property test for FireEgress/Deadline.lean (formal/fireegress): the escape
// deadlines of deadline.js against the statements the Lean file proves, on
// random small lattices under Studio's three fire models (buildForecast) and on
// random graphs with hand-made forecasts whose heat never falls.
//   1. escapeDeadlines satisfies the Bellman equation at every node and is
//      -Infinity exactly where no exit is reachable (deadline_bellman,
//      deadline_exit, deadline_eq_bot_iff);
//   2. departureSafe agrees with L just before and just after it
//      (mem_safeSet_iff, departureSafe_correct);
//   3. the least dose D*(v, t) of doseFront never falls as t grows
//      (leastDose_mono), and the good departures are downward closed
//      (goodSet_downward);
//   4. doseDeadline's bracket contains the boundary of that sweep
//      (bisect_correct, doseDeadline_bisection);
//   5. doseDeadlineMap is never later than doseDeadline, and leaving at the map's
//      deadline works under doseFront (grid_conservative, map_le_doseDeadline);
//   6. with no dose limit the dose deadline is the hard deadline
//      (doseDeadline_top);
//   7. the map's V, recomputed here from the recursion in deadline.js's header,
//      never falls as j grows, and the map is its largest good grid time
//      (gridV_mono, gridV_eq_val); the single-frame shortcut equals linkDose;
//   8. the one-link case where the limit binds, T* = 4.5 < 9 = L
//      (example_dose_limit_binds);
//   9. tau = ceil(a / delta - 1e-9) could round a link time down, so the map
//      could be later than the exact deadline (epsilon_rounds_down); the code
//      now puts such links back up, and this is a regression test.
// Seeds and sizes differ from deadline.test.mjs (seeds 7 and 11 there).
import assert from "node:assert/strict";
import { test } from "node:test";

import { departureSafe, doseDeadline, doseDeadlineMap, escapeDeadlines, frontArrival } from "./deadline.js";
import { buildForecast, doseFront, linkDose } from "./forecast.js";
import { csr, edgeLengths } from "./route.js";

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

const MODELS = ["temperature", "flood", "radial"];

/** A w x h unit lattice (4-neighbour), plus a cell diagonal with probability pDiag. */
function lattice(w, h, rand, pDiag = 0) {
  const nodes = [], edges = [];
  for (let j = 0; j < h; j += 1) for (let i = 0; i < w; i += 1) nodes.push(i, j, 0);
  for (let j = 0; j < h; j += 1) {
    for (let i = 0; i < w; i += 1) {
      const n = j * w + i;
      if (i < w - 1) edges.push(n, n + 1);
      if (j < h - 1) edges.push(n, n + w);
      if (i < w - 1 && j < h - 1 && rand() < pDiag) edges.push(n, n + w + 1);
    }
  }
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges) };
}

/** A graph from 2-D points (y = 0) and an edge list. */
function graphOf(points, edges) {
  return { nodes: Float64Array.from(points.flatMap(([x, z]) => [x, 0, z])), edges: Int32Array.from(edges.flat()) };
}

/** A forecast with given per-frame rates (K x n, never falling) and untenability times. */
function forecastOf(rates, untenable, frameSeconds) {
  const K = rates.length, n = rates[0].length;
  const rate = new Float32Array(K * n);
  rates.forEach((row, k) => row.forEach((r, v) => (rate[k * n + v] = r)));
  const horizon = K * frameSeconds;
  const closes = Float64Array.from(untenable, (u) => Math.min(u, horizon));
  return { n, K, stride: 1, dt: frameSeconds, frameSeconds, horizon, rate, untenable: Float64Array.from(untenable), closes };
}

function randomBlocked(g, rand, p) {
  const blocked = new Uint8Array(g.edges.length / 2);
  for (let e = 0; e < blocked.length; e += 1) if (rand() < p) blocked[e] = 1;
  return blocked;
}

/** Nodes joined to an exit by open links, whatever the fire. */
function reachesExit(g, exits, blocked) {
  const { indptr, neighbours, edgeIds, n } = csr(g);
  const seen = new Uint8Array(n);
  const stack = [];
  for (const x of exits) if (!seen[x]) (seen[x] = 1), stack.push(x);
  while (stack.length) {
    const v = stack.pop();
    for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
      if (blocked?.[edgeIds[k]]) continue;
      const w = neighbours[k];
      if (!seen[w]) (seen[w] = 1), stack.push(w);
    }
  }
  return seen;
}

const close = (a, b, rel = 1e-9) => a === b || (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= rel * Math.max(1, Math.abs(a), Math.abs(b)));

// ---------------------------------------------------------------------------
// Hard deadlines: fronts and forecast closing times on lattices.

const HARD = (() => {
  const rand = mulberry32(20261028);
  const cases = [];
  for (let c = 0; c < 48; c += 1) {
    const g = lattice(5 + Math.floor(rand() * 6), 4 + Math.floor(rand() * 5), rand, c % 2 ? 0.35 : 0);
    const n = g.nodes.length / 3;
    const blocked = randomBlocked(g, rand, 0.12);
    const exits = [Math.floor(rand() * n)];
    if (rand() < 0.4) exits.push(Math.floor(rand() * n));
    const speed = 0.5 + rand() * 1.5;
    const origin = Math.floor(rand() * n);
    // Every eighth case has no fire at all (u = Infinity: never cut off).
    const untenable =
      c % 8 === 7
        ? new Float64Array(n).fill(Infinity)
        : c % 3 === 0
          ? frontArrival(g, origin, { speed: 0.15 + rand(), blocked })
          : buildForecast(g, { model: MODELS[c % 3], seed: origin, steps: 25, dt: 1 + rand() * 2, threshold: 60 + rand() * 60, blocked }).closes;
    cases.push({ g, blocked, exits, speed, untenable });
  }
  return cases;
})();

test("hard deadlines satisfy the Bellman equation at every node (deadline_bellman, deadline_exit, deadline_eq_bot_iff)", () => {
  let nodes = 0, finite = 0, never = 0, none = 0;
  HARD.forEach(({ g, blocked, exits, speed, untenable }, c) => {
    const L = escapeDeadlines(g, { exits, untenable, speed, blocked });
    const { indptr, neighbours, edgeIds, n } = csr(g);
    const len = edgeLengths(g);
    const isExit = new Uint8Array(n);
    for (const x of exits) isExit[x] = 1;
    const reach = reachesExit(g, exits, blocked);
    for (let v = 0; v < n; v += 1) {
      let rhs = -Infinity;
      if (isExit[v]) rhs = untenable[v];
      else {
        for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
          const e = edgeIds[k];
          if (blocked[e]) continue;
          rhs = Math.max(rhs, Math.min(untenable[v], L[neighbours[k]] - len[e] / speed));
        }
      }
      assert.ok(close(L[v], rhs), `case ${c} node ${v}: L = ${L[v]} but the Bellman right side is ${rhs}`);
      assert.equal(L[v] === -Infinity, !reach[v], `case ${c} node ${v}: L = ${L[v]}, exit reachable: ${!!reach[v]}`);
      nodes += 1;
      if (Number.isFinite(L[v])) finite += 1;
      else if (L[v] === Infinity) never += 1;
      else none += 1;
    }
  });
  console.log(JSON.stringify({ cases: HARD.length, nodes, finite, never, none }));
  assert.ok(finite > 500 && none > 0 && never > 50);
});

// In exact arithmetic leaving at L itself is unsafe (the safe set is open). In
// doubles (L[w] - a) + a can round below L[w], so leaving at exactly the computed L
// may pass the forward check: e.g. a 0.2 s link into an exit closing at 0.9 s gives
// L = 0.7, and 0.7 + 0.2 = 0.8999999999999999 < 0.9. Those are counted, not failed.
test("departureSafe agrees with L just before and just after it (mem_safeSet_iff, departureSafe_correct)", () => {
  let before = 0, after = 0, none = 0, never = 0, atL = 0;
  HARD.forEach(({ g, blocked, exits, speed, untenable }, c) => {
    const opts = { exits, untenable, speed, blocked };
    const L = escapeDeadlines(g, opts);
    for (let v = 0; v < L.length; v += 1) {
      if (L[v] === -Infinity) {
        assert.equal(departureSafe(g, v, -1e9, opts), false, `case ${c} node ${v}: no deadline but escapes`);
        none += 1;
      } else if (L[v] === Infinity) {
        assert.equal(departureSafe(g, v, 1e9, opts), true, `case ${c} node ${v}: never cut off but fails late`);
        never += 1;
      } else {
        const eps = 1e-7 * Math.max(1, Math.abs(L[v]));
        assert.equal(departureSafe(g, v, L[v] - eps, opts), true, `case ${c} node ${v}: just before L = ${L[v]} fails`);
        if (departureSafe(g, v, L[v], opts)) atL += 1;
        assert.equal(departureSafe(g, v, L[v] + eps, opts), false, `case ${c} node ${v}: just after L = ${L[v]} succeeds`);
        before += 1;
        after += 1;
      }
    }
  });
  console.log(JSON.stringify({ before, after, none, never, safeAtExactlyL: atL }));
  assert.ok(before > 500);
});

// ---------------------------------------------------------------------------
// Dose deadlines: buildForecast on lattices, hand forecasts on random graphs.

const LIMITS = [1e-3, 1e-2, 0.1, 1, Infinity];

const DOSE = (() => {
  const rand = mulberry32(31337);
  const cases = [];
  for (let c = 0; c < 18; c += 1) {
    const g = lattice(4 + Math.floor(rand() * 4), 3 + Math.floor(rand() * 3), rand, c % 2 ? 0.3 : 0);
    const n = g.nodes.length / 3;
    const blocked = randomBlocked(g, rand, 0.08);
    const exits = [Math.floor(rand() * n)];
    if (rand() < 0.3) exits.push(Math.floor(rand() * n));
    const model = MODELS[c % 3];
    const forecast = buildForecast(g, { model, seed: Math.floor(rand() * n), steps: 20 + Math.floor(rand() * 20), dt: 1 + rand() * 3, threshold: 60 + rand() * 60, blocked });
    const step = c % 4 === 3 ? 0.2 + rand() * 0.6 : null;
    cases.push({ g, blocked, exits, forecast, speed: 0.6 + rand(), limit: LIMITS[c % 5], step, label: `${model} ${c}` });
  }
  for (let c = 0; c < 14; c += 1) {
    const n = 5 + Math.floor(rand() * 5);
    const pts = Array.from({ length: n }, () => [rand() * 5, rand() * 5]);
    const edges = [];
    for (let a = 0; a < n; a += 1) for (let b = a + 1; b < n; b += 1) if (rand() < 0.4) edges.push([a, b]);
    if (!edges.length) edges.push([0, 1]);
    const g = graphOf(pts, edges);
    const K = 2 + Math.floor(rand() * 5);
    const frame = 1 + rand() * 3;
    const rates = Array.from({ length: K }, () => new Array(n).fill(0));
    for (let v = 0; v < n; v += 1) {
      let r = rand() * 1e-3;
      for (let k = 0; k < K; k += 1) {
        r = Math.max(r, rand() < 0.4 ? rand() * 0.05 : 0);
        rates[k][v] = r;
      }
    }
    const untenable = Array.from({ length: n }, () => (rand() < 0.3 ? rand() * K * frame : Infinity));
    const forecast = forecastOf(rates, untenable, frame);
    const exits = [n - 1, ...(rand() < 0.3 ? [n - 2] : [])];
    const step = c % 3 === 2 ? 0.2 + rand() * 0.5 : null;
    cases.push({ g, blocked: null, exits, forecast, speed: 0.8 + rand(), limit: LIMITS[(c + 2) % 5], step, label: `hand ${c}` });
  }
  return cases;
})();

/** Good: a route exists (finite least dose) and its dose is within the limit (works in doseDeadline). */
const within = (dose, limit) => dose < Infinity && dose <= limit;

/** The least dose of doseFront leaving v at t (Infinity: no safe route). */
function leastDoseAt(g, v, t, { exits, forecast, speed, blocked }) {
  const { front, truncated } = doseFront(g, v, { exits, forecast, depart: t, speed, blocked });
  assert.equal(truncated, false);
  return front.length ? front[front.length - 1].dose : Infinity;
}

/** Per case, up to four starts with a positive hard deadline, each swept over [0, 1.05 L]. */
const SWEEPS = (() => {
  const rand = mulberry32(777);
  const out = [];
  DOSE.forEach((cs, c) => {
    const { g, blocked, exits, forecast, speed } = cs;
    const L = escapeDeadlines(g, { exits, untenable: forecast.closes, speed, blocked });
    const starts = [];
    for (let v = 0; v < L.length; v += 1) if (L[v] > 0) starts.push(v);
    for (let i = 0; i < 4 && starts.length; i += 1) {
      const v = starts.splice(Math.floor(rand() * starts.length), 1)[0];
      const times = Array.from({ length: 41 }, (_, i) => (i / 40) * 1.05 * L[v]);
      const doses = times.map((t) => leastDoseAt(g, v, t, cs));
      out.push({ c, v, L: L[v], times, doses });
    }
  });
  return out;
})();

test("the least dose never falls as the departure gets later (leastDose_mono, goodSet_downward)", () => {
  let points = 0, finiteRises = 0;
  for (const { c, v, times, doses } of SWEEPS) {
    const { limit } = DOSE[c];
    for (let i = 1; i < times.length; i += 1) {
      const a = doses[i - 1], b = doses[i];
      assert.ok(a <= b + 1e-12 * Math.max(1, Math.abs(b)), `${DOSE[c].label} node ${v}: D*(${times[i - 1]}) = ${a} > D*(${times[i]}) = ${b}`);
      // Good at t means good at every earlier time (downward closed).
      if (within(b, limit)) assert.ok(within(a, limit), `${DOSE[c].label} node ${v}: good at ${times[i]} but not at ${times[i - 1]}`);
      if (Number.isFinite(b) && b > a) finiteRises += 1;
      points += 1;
    }
    // Past the hard deadline nothing is safe.
    assert.equal(doses[doses.length - 1], Infinity);
  }
  console.log(JSON.stringify({ sweeps: SWEEPS.length, points, finiteRises }));
  assert.ok(SWEEPS.length > 60 && finiteRises > 200);
});

test("doseDeadline's bracket contains the boundary of the sweep (bisect_correct, doseDeadline_bisection)", () => {
  let brackets = 0, nulls = 0, binding = 0, badAtLGood = 0;
  const tolerance = 1e-3;
  for (const { c, v, L, times, doses } of SWEEPS) {
    const cs = DOSE[c];
    const exact = doseDeadline(cs.g, v, { ...cs, tolerance });
    const good = (t) => within(leastDoseAt(cs.g, v, t, cs), cs.limit);
    if (!within(doses[0], cs.limit)) {
      assert.equal(exact, null, `${cs.label} node ${v}: leaving at 0 is bad but doseDeadline is ${JSON.stringify(exact)}`);
      nulls += 1;
      continue;
    }
    assert.ok(exact !== null, `${cs.label} node ${v}: leaving at 0 is good but doseDeadline is null`);
    assert.ok(good(exact.good), `${cs.label} node ${v}: good = ${exact.good} is not good`);
    // `bad` starts at L and is never tested by the code; leaving at exactly the
    // computed L can pass in doubles (see the departureSafe test). A midpoint the
    // code moved `bad` to was tested bad, and must stay bad.
    if (good(exact.bad)) {
      assert.equal(exact.bad, L, `${cs.label} node ${v}: bad = ${exact.bad} (a tested midpoint) is good`);
      badAtLGood += 1;
    }
    assert.ok(exact.bad - exact.good <= tolerance, `${cs.label} node ${v}: gap ${exact.bad - exact.good}`);
    // At most a part in 10^9 past L: in doubles leaving at exactly L can pass,
    // and doseDeadline then moves `bad` just past it so that it really is bad.
    assert.ok(exact.bad <= L + 4e-9 * Math.max(1, Math.abs(L)), `${cs.label} node ${v}: bad = ${exact.bad} is past L = ${L}`);
    times.forEach((t, i) => {
      const ok = within(doses[i], cs.limit);
      if (t <= exact.good) assert.ok(ok, `${cs.label} node ${v}: ${t} <= good = ${exact.good} but bad`);
      if (t >= exact.bad) assert.ok(!ok, `${cs.label} node ${v}: ${t} >= bad = ${exact.bad} but good`);
    });
    if (exact.bad < L - tolerance) binding += 1;
    brackets += 1;
  }
  console.log(JSON.stringify({ brackets, nulls, binding, badAtLJudgedGood: badAtLGood }));
  assert.ok(brackets > 40 && binding > 5);
});

test("doseDeadlineMap is never later than doseDeadline and its departure works (grid_conservative, map_le_doseDeadline)", () => {
  let checked = 0, unreached = 0, worstGap = 0;
  for (const cs of DOSE) {
    const { g, blocked, exits, forecast, limit, speed, step } = cs;
    const { deadline } = doseDeadlineMap(g, { exits, forecast, limit, speed, blocked, step });
    for (let v = 0; v < deadline.length; v += 1) {
      if (deadline[v] === -Infinity) {
        unreached += 1;
        continue;
      }
      const dose = leastDoseAt(g, v, deadline[v], cs);
      assert.ok(within(dose, limit), `${cs.label} node ${v}: leaving at the map's ${deadline[v]} gives dose ${dose} > ${limit}`);
      const exact = doseDeadline(g, v, { ...cs, tolerance: 1e-3 });
      assert.ok(exact !== null && deadline[v] <= exact.bad + 1e-9, `${cs.label} node ${v}: map ${deadline[v]} is later than exact ${JSON.stringify(exact)}`);
      worstGap = Math.max(worstGap, exact.good - deadline[v]);
      checked += 1;
    }
  }
  // The gap T* - map is not bounded by the proofs; it is reported, not asserted.
  console.log(JSON.stringify({ checked, unreached, worstGap: +worstGap.toFixed(3) }));
  assert.ok(checked > 300);
});

test("with no dose limit the dose deadline is the hard deadline (doseDeadline_top)", () => {
  let checked = 0, nulls = 0;
  const tolerance = 1e-4;
  for (const cs of DOSE) {
    if (cs.limit !== Infinity) continue;
    const { g, blocked, exits, forecast, speed, step } = cs;
    const L = escapeDeadlines(g, { exits, untenable: forecast.closes, speed, blocked });
    const { deadline } = doseDeadlineMap(g, { exits, forecast, limit: Infinity, speed, blocked, step });
    for (let v = 0; v < L.length; v += 1) {
      const exact = doseDeadline(g, v, { ...cs, tolerance });
      if (!(0 < L[v])) {
        assert.equal(exact, null, `${cs.label} node ${v}: L = ${L[v]} but doseDeadline is ${JSON.stringify(exact)}`);
        nulls += 1;
        continue;
      }
      assert.ok(exact.good < L[v] && exact.bad <= L[v] && exact.bad >= L[v] - tolerance - 1e-9, `${cs.label} node ${v}: ${exact.good}..${exact.bad} vs L = ${L[v]}`);
      if (deadline[v] !== -Infinity) assert.ok(deadline[v] < L[v], `${cs.label} node ${v}: map ${deadline[v]} >= L = ${L[v]}`);
      checked += 1;
    }
  }
  console.log(JSON.stringify({ checked, nulls }));
  assert.ok(checked > 40);
});

/** The map's backward pass, recomputed from deadline.js's header: the whole table V[j][v]. */
function gridTable(g, { exits, forecast, speed, blocked }, delta, stats) {
  const { indptr, neighbours, edgeIds, n, m } = csr(g);
  const len = edgeLengths(g);
  const closes = forecast.closes;
  const J = Math.floor(forecast.horizon / delta);
  const tau = new Int32Array(m);
  for (let e = 0; e < m; e += 1) tau[e] = Math.max(1, Math.ceil(len[e] / speed / delta - 1e-9));
  const isExit = new Uint8Array(n);
  for (const x of exits) isExit[x] = 1;
  const V = Array.from({ length: J + 1 }, () => new Float64Array(n).fill(Infinity));
  const { rate, frameSeconds, K } = forecast;
  for (let j = J; j >= 0; j -= 1) {
    const t = j * delta;
    const frame = Math.min(K - 1, Math.floor(t / frameSeconds));
    const frameEnd = (frame + 1) * frameSeconds;
    for (let v = 0; v < n; v += 1) {
      if (!(t < closes[v])) continue;
      if (isExit[v]) {
        V[j][v] = 0;
        continue;
      }
      let best = Infinity;
      for (let k = indptr[v]; k < indptr[v + 1]; k += 1) {
        const e = edgeIds[k];
        if (blocked?.[e]) continue;
        const jj = j + tau[e];
        if (jj > J) continue;
        const w = neighbours[k];
        const t2 = jj * delta;
        if (!(t2 < closes[w]) || V[jj][w] === Infinity) continue;
        let dose = linkDose(forecast, v, w, t, t2);
        if (t2 <= frameEnd) {
          // The code's single-frame shortcut must equal linkDose.
          const short = (t2 - t) * Math.max(rate[frame * n + v], rate[frame * n + w]);
          assert.ok(close(short, dose, 1e-12), `shortcut ${short} vs linkDose ${dose} at ${t}..${t2}`);
          stats.shortcuts += 1;
          dose = short;
        }
        best = Math.min(best, V[jj][w] + dose);
      }
      V[j][v] = best;
    }
  }
  return { V, J };
}

test("the map's V never falls as j grows, and the map is its largest good grid time (gridV_mono, gridV_eq_val)", () => {
  const stats = { shortcuts: 0 };
  let pairs = 0, deadlines = 0, finitePairs = 0;
  for (const cs of DOSE) {
    const { g, blocked, exits, forecast, limit, speed, step } = cs;
    const map = doseDeadlineMap(g, { exits, forecast, limit, speed, blocked, step });
    const { V, J } = gridTable(g, cs, map.step, stats);
    assert.equal(map.layers, J + 1);
    for (let v = 0; v < V[0].length; v += 1) {
      for (let j = 0; j < J; j += 1) {
        const a = V[j][v], b = V[j + 1][v];
        assert.ok(a <= b + 1e-9 * Math.max(1, Math.abs(b)), `${cs.label} node ${v}: V_${j} = ${a} > V_${j + 1} = ${b}`);
        if (Number.isFinite(b)) finitePairs += 1;
        pairs += 1;
      }
      let last = -1;
      for (let j = J; j >= 0; j -= 1) {
        if (V[j][v] < Infinity && V[j][v] <= limit) {
          last = j;
          break;
        }
      }
      const expect = last < 0 ? -Infinity : last * map.step;
      assert.equal(map.deadline[v], expect, `${cs.label} node ${v}: map ${map.deadline[v]} vs recomputed ${expect}`);
      // Every earlier grid time is good too.
      for (let j = 0; j < last; j += 1) assert.ok(within(V[j][v], limit), `${cs.label} node ${v}: grid time ${j} bad below the deadline ${last}`);
      deadlines += 1;
    }
  }
  console.log(JSON.stringify({ pairs, finitePairs, deadlines, shortcuts: stats.shortcuts }));
  assert.ok(finitePairs > 5000 && stats.shortcuts > 1000);
});

test("one link where the dose limit binds: T* = 4.5 < L = 9 (example_dose_limit_binds)", () => {
  // Start 0, exit 1, 1 m apart at 1 m/s; both close at the horizon, 10 s. The
  // exit's rate steps from 0 to 1 per second at 5 s, so leaving at t accrues
  // clamp(t - 4, 0, 1); with limit 1/2 the last good departure is 4.5 s.
  const g = graphOf([[0, 0], [1, 0]], [[0, 1]]);
  const rates = Array.from({ length: 10 }, (_, k) => [0, k >= 5 ? 1 : 0]);
  const forecast = forecastOf(rates, [Infinity, Infinity], 1);
  const opts = { exits: [1], forecast, limit: 0.5, speed: 1 };
  const L = escapeDeadlines(g, { exits: [1], untenable: forecast.closes, speed: 1 });
  assert.deepEqual(Array.from(L), [9, 10]);
  const exact = doseDeadline(g, 0, { ...opts, tolerance: 1e-6 });
  assert.ok(exact.good <= 4.5 && 4.5 < exact.bad && exact.bad - exact.good <= 1e-6, JSON.stringify(exact));
  assert.ok(exact.bad < L[0]);
  const { deadline } = doseDeadlineMap(g, { ...opts, step: 0.25 });
  // The dose reaches the limit exactly at 4.5 s; the map keeps a part in 10^9
  // in hand on the limit, so it may report the grid step before, never later.
  assert.ok(deadline[0] <= 4.5 && deadline[0] >= 4.5 - 0.25 - 1e-12, `map ${deadline[0]}`);
});

test(
  "leaving just past L is unsafe, and doseDeadline's bad end is bad (mem_safeSet_iff, doseDeadline_bisection)",
  () => {
    // A 0.2 m link at 1 m/s into an exit that closes at 0.9 s: L(0) = 0.9 - 0.2 = 0.7,
    // but 0.7 + 0.2 = 0.8999999999999999 < 0.9, so the forward checks accept a
    // departure at L, and doseDeadline's initial bad = L is in fact good.
    const g = { nodes: Float64Array.from([0, 0, 0, 0.2, 0, 0]), edges: Int32Array.from([0, 1]) };
    const forecast = {
      n: 2, K: 1, stride: 1, dt: 10, frameSeconds: 10, horizon: 10,
      rate: new Float32Array(2), untenable: Float64Array.from([Infinity, 0.9]), closes: Float64Array.from([10, 0.9]),
    };
    const L = escapeDeadlines(g, { exits: [1], untenable: forecast.closes, speed: 1 });
    // At exactly L the forward check can pass in doubles (documented in
    // deadline.js); a part in 10^9 later it must fail.
    const past = L[0] + 1e-9 * Math.max(1, Math.abs(L[0]));
    assert.equal(departureSafe(g, 0, past, { exits: [1], untenable: forecast.closes, speed: 1 }), false, `departureSafe just past L = ${L[0]}`);
    const exact = doseDeadline(g, 0, { exits: [1], forecast, limit: Infinity, speed: 1, tolerance: 0.05 });
    assert.equal(leastDoseAt(g, 0, exact.bad, { exits: [1], forecast, speed: 1 }), Infinity, `bad = ${exact.bad} works`);
  },
);

test(
  "the map's departure works even when a link time sits just above a whole number of steps (epsilon_rounds_down)",
  () => {
    // A link of 1 + 5e-10 m at 1 m/s on a 1 s grid takes one grid step, which is
    // shorter than the walk; the exit closes at 1 + 3e-10 s, between the two. The
    // grid admits the move from t = 0 ((0 + 1) * 1 < closes), the real walker
    // arrives at 1 + 5e-10 >= closes. Expected (Deadline.lean): no grid deadline at
    // node 0, since no departure from 0 works.
    const g = { nodes: Float64Array.from([0, 0, 0, 1 + 5e-10, 0, 0]), edges: Int32Array.from([0, 1]) };
    const forecast = {
      n: 2, K: 1, stride: 1, dt: 10, frameSeconds: 10, horizon: 10,
      rate: new Float32Array(2), untenable: Float64Array.from([Infinity, 1 + 3e-10]), closes: Float64Array.from([10, 1 + 3e-10]),
    };
    const opts = { exits: [1], forecast, limit: 1, speed: 1 };
    const { deadline } = doseDeadlineMap(g, { ...opts, step: 1 });
    assert.equal(doseDeadline(g, 0, opts), null);
    assert.equal(deadline[0], -Infinity, `map deadline ${deadline[0]} but leaving node 0 never works`);
  },
);
