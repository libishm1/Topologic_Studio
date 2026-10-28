// Run with: npm test
import assert from "node:assert/strict";
import { test } from "node:test";

import { departureSafe, doseDeadline, doseDeadlineMap, escapeDeadlines, frontArrival } from "./deadline.js";
import { buildForecast, doseFront } from "./forecast.js";

/** A w x h unit lattice (4-neighbour), z = 0. */
function lattice(w, h) {
  const nodes = [], edges = [];
  for (let j = 0; j < h; j += 1) for (let i = 0; i < w; i += 1) nodes.push(i, j, 0);
  for (let j = 0; j < h; j += 1) {
    for (let i = 0; i < w; i += 1) {
      const n = j * w + i;
      if (i < w - 1) edges.push(n, n + 1);
      if (j < h - 1) edges.push(n, n + w);
    }
  }
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2) };
}

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("without fire every node of a connected graph is never cut off", () => {
  const g = lattice(6, 4);
  const L = escapeDeadlines(g, { exits: [0] });
  assert.ok(L.every((x) => x === Infinity));
});

test("a node with no way to an exit has no deadline at all", () => {
  const g = lattice(4, 1);
  g.blocked[1] = 1; // cut the line between nodes 1 and 2
  const L = escapeDeadlines(g, { exits: [0], blocked: g.blocked });
  assert.equal(L[3], -Infinity);
  assert.equal(L[1], Infinity);
});

test("on a corridor the deadline is the fire's arrival minus the walk", () => {
  // Exit at 0, fire reaching node i at time 10 - i (from the far end), walking 1 m/s.
  const g = lattice(6, 1);
  const u = Float64Array.from([10, 9, 8, 7, 6, 5]);
  const L = escapeDeadlines(g, { exits: [0], untenable: u, speed: 1 });
  // From node i the exit is i metres away; the binding constraint is the exit itself.
  for (let i = 0; i < 6; i += 1) assert.ok(Math.abs(L[i] - Math.min(u[i], 10 - i)) < 1e-12);
});

test("deadlines agree with the forward check on random graphs and fires", () => {
  const rand = mulberry32(7);
  let checked = 0;
  for (let c = 0; c < 60; c += 1) {
    const g = lattice(3 + Math.floor(rand() * 6), 2 + Math.floor(rand() * 5));
    for (let e = 0; e < g.blocked.length; e += 1) if (rand() < 0.15) g.blocked[e] = 1;
    const n = g.nodes.length / 3;
    const exits = [Math.floor(rand() * n)];
    if (rand() < 0.5) exits.push(Math.floor(rand() * n));
    const untenable = frontArrival(g, Math.floor(rand() * n), { speed: 0.2 + rand(), blocked: g.blocked });
    const opts = { exits, untenable, speed: 0.5 + rand() * 1.5, blocked: g.blocked };
    const L = escapeDeadlines(g, opts);
    for (let v = 0; v < n; v += 1) {
      if (L[v] === -Infinity) {
        assert.equal(departureSafe(g, v, -1e9, opts), false, `node ${v} has no deadline but can escape`);
      } else if (Number.isFinite(L[v])) {
        assert.equal(departureSafe(g, v, L[v] - 1e-6, opts), true, `just before L fails at ${v}`);
        assert.equal(departureSafe(g, v, L[v] + 1e-6, opts), false, `just after L succeeds at ${v}`);
        checked += 1;
      }
    }
  }
  assert.ok(checked > 200, `only ${checked} finite deadlines checked`);
});

test("dose deadlines: the map never says later than the exact deadline, and is close to it", () => {
  const rand = mulberry32(11);
  let compared = 0, worstGap = 0, unlimited = 0;
  for (let c = 0; c < 25; c += 1) {
    const g = lattice(4 + Math.floor(rand() * 6), 3 + Math.floor(rand() * 4));
    for (let e = 0; e < g.blocked.length; e += 1) if (rand() < 0.1) g.blocked[e] = 1;
    const n = g.nodes.length / 3;
    const exits = [Math.floor(rand() * n)];
    const model = ["temperature", "flood", "radial"][c % 3];
    const forecast = buildForecast(g, { model, seed: Math.floor(rand() * n), steps: 30, dt: 1 + rand() * 3, threshold: 60 + rand() * 60, blocked: g.blocked });
    const speed = 0.6 + rand();
    const limit = [1e-3, 1e-2, 0.1, Infinity][c % 4];
    const opts = { exits, forecast, limit, speed, blocked: g.blocked };
    const { deadline, step } = doseDeadlineMap(g, opts);
    for (let v = 0; v < n; v += 1) {
      const exact = doseDeadline(g, v, { ...opts, tolerance: 1e-4 });
      if (deadline[v] === -Infinity) continue;
      // The map's departure works in the exact model.
      const { front } = doseFront(g, v, { exits, forecast, depart: deadline[v], speed, blocked: g.blocked });
      assert.ok(front.length && front[front.length - 1].dose <= limit, `map deadline at ${v} does not work`);
      assert.ok(exact !== null && deadline[v] <= exact.bad + 1e-9, `map is later than exact at ${v}`);
      worstGap = Math.max(worstGap, exact.good - deadline[v]);
      compared += 1;
      if (limit === Infinity) {
        // Without a dose limit the exact deadline is the hard one.
        const L = escapeDeadlines(g, { exits, untenable: forecast.closes, speed, blocked: g.blocked })[v];
        if (Number.isFinite(L)) {
          assert.ok(exact.good < L && exact.bad >= L - 1e-4 - 1e-9, `node ${v}: ${exact.good}..${exact.bad} vs ${L}`);
          unlimited += 1;
        }
      }
    }
    void step;
  }
  console.log(JSON.stringify({ compared, unlimited, worstGap: +worstGap.toFixed(3) }));
  assert.ok(compared > 300 && unlimited > 50);
});

test("the dose deadline says when the end of the forecast, not the fire, is the limit (review M11)", () => {
  // A corridor of 11 nodes 1 m apart, exit at node 0, walking 1 m/s. The
  // fire starts at node 10 and takes a node per 5 s step; the forecast is 6
  // steps (30 s), so it reaches node 5 at 25 s and node 4 only after it ends.
  const g = lattice(11, 1);
  const forecast = buildForecast(g, { model: "flood", seed: 10, steps: 6, dt: 5 });
  const opts = { exits: [0], forecast, limit: 1, speed: 1, tolerance: 1e-3 };
  // From node 2 the fire never matters: only the forecast's end does.
  const far = doseDeadline(g, 2, opts);
  assert.equal(far.horizon, true);
  assert.ok(Math.abs(far.good - 28) < 0.01, `${far.good}`);
  // From node 5 the fire is the limit: the start itself closes at 25 s.
  const near = doseDeadline(g, 5, opts);
  assert.equal(near.horizon, false);
  assert.ok(Math.abs(near.good - 25) < 0.01, `${near.good}`);
});

test("the deadline map's work stays under its cap for long horizons (review K2)", () => {
  // 2,000 fire steps of 5 s: at a quarter of the link time a 30 x 30 lattice
  // would take 48,000 layers, 2e8 link updates.
  const g = lattice(30, 30);
  const n = g.nodes.length / 3;
  const exits = [0, n - 1];
  const speed = 1.2, limit = 0.3;
  // Per layer: both ends of every (open) link, and every node.
  const perLayer = g.edges.length + n;
  for (const model of ["temperature", "flood"]) {
    const forecast = buildForecast(g, { model, seed: 465, steps: 2000, dt: 5, threshold: 60 });
    for (const maxWork of [2e5, 1e6, 5e6]) {
      const map = doseDeadlineMap(g, { exits, forecast, limit, speed, maxWork });
      const work = map.layers * perLayer;
      assert.ok(work <= maxWork, `${model}: ${work} link updates against a cap of ${maxWork}`);
      assert.equal(map.work, work);
      // A wider step is still never later than the exact deadline.
      for (const v of [31, 200, 450, 700]) {
        if (map.deadline[v] === -Infinity) continue;
        const exact = doseDeadline(g, v, { exits, forecast, limit, speed, tolerance: 1e-3 });
        assert.ok(exact && map.deadline[v] <= exact.bad + 1e-9, `${model}: map later than exact at ${v}`);
      }
    }
  }
});
