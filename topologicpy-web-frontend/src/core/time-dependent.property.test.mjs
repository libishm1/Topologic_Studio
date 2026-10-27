// Run with: node --test src/core/time-dependent.property.test.mjs
//
// Property test for FireEgress/TimeDependent.lean (L10c, formal/fireegress).
// Studio has no time-dependent router yet, so this file carries a reference
// implementation of the label-setting loop the Lean file specifies (Dreyfus
// 1969: settle the unsettled node of least label, relax every link out of it
// with label' = A_e(label u); Infinity = never) and checks it on random small
// FIFO networks (closure windows, smoke slow-down, doors that open at a time,
// links that speed up no faster than the clock, zero-time links, self-loops,
// parallel links, cycles) against
//   1. brute force over all simple walks (td_dijkstra_glb, simple_walk_suffices);
//   2. a time-expanded search over all walks that may wait and loop
//      (waiting_never_helps, cycle_never_helps);
//   3. Bellman-Ford iterated down from Infinity (bellman_subsolution_le);
//   4. the parent pointers: every label is the arrival time of a real walk;
//   5. replanning from each node of the earliest-arrival route, at the time the
//      route reaches it, gives the same arrival (replan_consistent).
// It also checks the non-FIFO counterexample (fifo_necessary_every_run) and
// that on random non-FIFO networks the labels stay sound, never earlier than a
// real walk (exhausted_without_fifo), while some are later than the optimum.
// Times are integer ticks, so every comparison is exact.
import assert from "node:assert/strict";
import { test } from "node:test";

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

const INF = Infinity;

/** A link that is never entered is never left: A(Infinity) = Infinity. */
const lift = (f) => (t) => (t === INF ? INF : f(t));

/**
 * Reference time-dependent label setting (the `Step` relation of
 * TimeDependent.lean, with min-updates as in `relaxState`). `links[u]` lists
 * `{ v, arr }`. `pick` chooses among nodes tied at the least label; every
 * choice is a valid run. Stops when the least label is Infinity
 * (td_dijkstra_stop_at_top). Returns labels and, per node, the link that set
 * its label.
 */
function labelSetting(n, links, s, t0, pick = (ties) => ties[0]) {
  const d = new Array(n).fill(INF);
  const via = new Array(n).fill(null);
  const settled = new Uint8Array(n);
  d[s] = t0;
  for (;;) {
    let best = INF;
    for (let v = 0; v < n; v += 1) if (!settled[v] && d[v] < best) best = d[v];
    if (best === INF) break;
    const ties = [];
    for (let v = 0; v < n; v += 1) if (!settled[v] && d[v] === best) ties.push(v);
    const u = pick(ties);
    settled[u] = 1;
    for (const link of links[u]) {
      if (settled[link.v]) continue;
      const a = link.arr(d[u]);
      if (a < d[link.v]) {
        d[link.v] = a;
        via[link.v] = { u, arr: link.arr };
      }
    }
  }
  return { d, via };
}

/** Earliest arrival over all simple walks from s, leaving at t0 (no waiting). */
function bruteSimple(n, links, s, t0) {
  const best = new Array(n).fill(INF);
  const on = new Uint8Array(n);
  const visit = (u, t) => {
    if (t < best[u]) best[u] = t;
    on[u] = 1;
    for (const { v, arr } of links[u]) {
      if (on[v]) continue;
      const a = arr(t);
      if (a < INF) visit(v, a);
    }
    on[u] = 0;
  };
  visit(s, t0);
  return best;
}

/**
 * Earliest arrival over all walks that may loop and wait (one tick at a time),
 * by a time-expanded search over ticks t0..H.
 */
function timeExpanded(n, links, s, t0, H) {
  const ea = new Array(n).fill(INF);
  const reached = Array.from({ length: n }, () => new Uint8Array(H + 1));
  reached[s][t0] = 1;
  for (let t = t0; t <= H; t += 1) {
    const stack = [];
    for (let u = 0; u < n; u += 1) if (reached[u][t]) stack.push(u);
    while (stack.length) {
      const u = stack.pop();
      if (t < ea[u]) ea[u] = t;
      if (t + 1 <= H) reached[u][t + 1] = 1;
      for (const { v, arr } of links[u]) {
        const a = arr(t);
        assert.ok(a >= t, "a link arrives before it is entered");
        if (a > H || reached[v][a]) continue;
        reached[v][a] = 1;
        if (a === t) stack.push(v); // zero-time link: same tick
      }
    }
  }
  return ea;
}

/** Bellman-Ford on d(v) <- min(d(v), A(d(u))), started from Infinity. */
function bellmanFord(n, links, s, t0) {
  const d = new Array(n).fill(INF);
  d[s] = t0;
  for (let round = 0; round < n; round += 1) {
    let changed = false;
    for (let u = 0; u < n; u += 1) {
      for (const { v, arr } of links[u]) {
        const a = arr(d[u]);
        if (a < d[v]) {
          d[v] = a;
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  return d;
}

/** The route the parent links give to v, as [{ node, time }], source first. */
function routeTo(via, s, t0, v, n) {
  const back = [];
  let x = v;
  while (x !== s) {
    assert.ok(back.length < n, "parent links form a cycle");
    back.push(via[x]);
    x = via[x].u;
  }
  back.reverse();
  const route = [{ node: s, time: t0 }];
  let t = t0;
  for (let i = 0; i < back.length; i += 1) {
    t = back[i].arr(t);
    route.push({ node: i + 1 < back.length ? back[i + 1].u : v, time: t });
  }
  return route;
}

/** A random FIFO arrival function in ticks (monotone, never earlier than t). */
function randomFifoArr(rand) {
  const w = Math.floor(rand() * 6); // 0..5 ticks; 0 = zero-time link
  const kind = Math.floor(rand() * 5);
  let f;
  if (kind === 0) {
    f = (t) => t + w; // static
  } else if (kind === 1) {
    const ts = Math.floor(rand() * 20);
    const k = [0.5, 1, 2][Math.floor(rand() * 3)];
    f = (t) => t + w + Math.floor(k * Math.max(0, t - ts)); // smoke slows the link
  } else if (kind === 2) {
    const open = Math.floor(rand() * 15);
    f = (t) => Math.max(t, open) + w; // a door that opens at a fixed time
  } else if (kind === 3) {
    const w0 = w + Math.floor(rand() * 8);
    f = (t) => t + Math.max(w, w0 - Math.floor(t / 2)); // speeds up at half the clock rate
  } else {
    const ts = Math.floor(rand() * 10);
    const w0 = w + Math.floor(rand() * 8);
    f = (t) => t + Math.max(w, w0 - Math.max(0, t - ts)); // speeds up at the clock rate: flat arrival
  }
  const b = rand() < 0.5 ? INF : Math.floor(rand() * 40); // fire closes the link at b
  return lift((t) => {
    const a = f(t);
    return a < b ? a : INF; // closure window, GraphNumerics.closeArc
  });
}

/** A random non-FIFO arrival function in ticks (never earlier than t). */
function randomNonFifoArr(rand) {
  const p = 2 + Math.floor(rand() * 5);
  const table = Array.from({ length: p }, () => Math.floor(rand() * 10));
  return lift((t) => t + table[t % p]);
}

/** A random network on 2..7 nodes, source 0. */
function randomNetwork(rand, makeArr) {
  const n = 2 + Math.floor(rand() * 6);
  const pLink = 0.2 + rand() * 0.4;
  const links = Array.from({ length: n }, () => []);
  for (let u = 0; u < n; u += 1) {
    for (let v = 0; v < n; v += 1) {
      if (u === v ? rand() < 0.1 : rand() < pLink) links[u].push({ v, arr: makeArr(rand) });
      if (u !== v && rand() < 0.05) links[u].push({ v, arr: makeArr(rand) }); // parallel link
    }
  }
  return { n, links, s: 0, t0: Math.floor(rand() * 10) };
}

const randomPick = (rand) => (ties) => ties[Math.floor(rand() * ties.length)];

test("generators: FIFO links are monotone and never arrive early", () => {
  const rand = mulberry32(7);
  for (let i = 0; i < 500; i += 1) {
    const arr = randomFifoArr(rand);
    let prev = -INF;
    for (let t = 0; t <= 200; t += 1) {
      const a = arr(t);
      assert.ok(a >= t);
      assert.ok(a >= prev, `not FIFO at t=${t}`);
      prev = a;
    }
    assert.equal(arr(INF), INF);
  }
});

test("L10c: label setting equals the earliest arrival on random FIFO networks", () => {
  const rand = mulberry32(20261027);
  let reachedPairs = 0;
  let closedSomewhere = 0;
  for (let c = 0; c < 400; c += 1) {
    const { n, links, s, t0 } = randomNetwork(rand, randomFifoArr);
    const { d, via } = labelSetting(n, links, s, t0, randomPick(rand));

    // 1. Brute force over simple walks: output = inf over walks, Infinity if none.
    const brute = bruteSimple(n, links, s, t0);
    assert.deepEqual(d, brute, `case ${c}: label setting vs simple walks`);

    // 2. Walks that may wait and loop do no better (cycles and waiting never help).
    const finite = brute.filter((x) => x < INF);
    const H = Math.max(t0, ...finite) + 30;
    const te = timeExpanded(n, links, s, t0, H);
    assert.deepEqual(te, brute, `case ${c}: waiting/looping walks vs simple walks`);
    if (brute.some((x) => x === INF)) closedSomewhere += 1;

    // 3. Bellman-Ford iterated down from Infinity reaches the same labels.
    assert.deepEqual(bellmanFord(n, links, s, t0), d, `case ${c}: Bellman-Ford`);

    for (let v = 0; v < n; v += 1) {
      if (d[v] === INF) continue;
      reachedPairs += 1;
      // 4. The label is the arrival time of the parent-pointer route.
      const route = routeTo(via, s, t0, v, n);
      assert.equal(route[route.length - 1].node, v);
      assert.equal(route[route.length - 1].time, d[v], `case ${c}: label of ${v} is not real`);
      // 5. Replanning from any point of the route, at the time the route gets
      //    there, under the same forecast, arrives at the same time.
      for (const { node, time } of route) {
        const again = labelSetting(n, links, node, time, randomPick(rand));
        assert.equal(again.d[v], d[v], `case ${c}: replanning from ${node} at ${time}`);
      }
    }
  }
  assert.ok(reachedPairs > 1000, "too few reached nodes to be a meaningful test");
  assert.ok(closedSomewhere > 20, "too few networks where fire cuts a node off");
});

test("FIFO is necessary: the Lean counterexample (fifo_necessary_every_run)", () => {
  // 0 = start, 1 = corridor, 2 = side room, 3 = exit. Every link takes one tick
  // except 1 -> 3: entered before tick 2 it arrives at tick 10, later it takes
  // one tick. Not FIFO: A(1) = 10 > A(2) = 3.
  const one = lift((t) => t + 1);
  const slowDoor = lift((t) => (t < 2 ? 10 : t + 1));
  const links = [
    [{ v: 1, arr: one }, { v: 2, arr: one }],
    [{ v: 3, arr: slowDoor }],
    [{ v: 1, arr: one }],
    [],
  ];
  assert.ok(slowDoor(1) > slowDoor(2));
  // Every tie-break among 1 and 2 (both labelled 1) gives the exit label 10.
  for (const pick of [(ties) => ties[0], (ties) => ties[ties.length - 1]]) {
    assert.equal(labelSetting(4, links, 0, 0, pick).d[3], 10);
  }
  // The detour 0 -> 2 -> 1 -> 3 arrives at 3; so does waiting at 1.
  assert.equal(bruteSimple(4, links, 0, 0)[3], 3);
  assert.equal(timeExpanded(4, links, 0, 0, 20)[3], 3);
});

test("without FIFO the labels stay sound but may be late (exhausted_without_fifo)", () => {
  const rand = mulberry32(1969);
  let late = 0;
  for (let c = 0; c < 300; c += 1) {
    const { n, links, s, t0 } = randomNetwork(rand, randomNonFifoArr);
    const { d, via } = labelSetting(n, links, s, t0, randomPick(rand));
    const brute = bruteSimple(n, links, s, t0);
    for (let v = 0; v < n; v += 1) {
      // Arrival functions here never return Infinity, so reachability is exact.
      assert.equal(d[v] === INF, brute[v] === INF, `case ${c}: reachability of ${v}`);
      if (d[v] === INF) continue;
      assert.ok(d[v] >= brute[v], `case ${c}: label earlier than any walk`);
      const route = routeTo(via, s, t0, v, n);
      assert.equal(route[route.length - 1].time, d[v], `case ${c}: label of ${v} is not real`);
      if (d[v] > brute[v]) late += 1;
    }
  }
  assert.ok(late > 0, "no random non-FIFO network showed a late label");
});
