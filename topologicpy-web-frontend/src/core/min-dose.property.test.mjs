// Run with: node --test src/core/min-dose.property.test.mjs
//
// Property test for FireEgress/MinDose.lean (T17, formal/fireegress): minimum-dose
// egress routing by a dynamic programme on the time-expanded graph. Studio does not
// implement this yet, so the DP below is the reference specification (it is the Lean
// `val`, line for line): V_H(v) = 0 at an exit, Infinity elsewhere;
// V_t(v) = min(wait: w_v(t) + V_{t+1}(v), move: min_e d_e(t) + V_{t+tau_e}(head e)),
// a move that would end after H is not allowed, exits are absorbing. Infinity stands
// for Lean's top (a closed link or an untenable wait).
//
// On small random time-expanded graphs (seeded mulberry32) it checks, against brute
// force enumeration of every schedule (walk in the time-expanded graph):
//   1. dp_exact / dp_isLeast / val_eq_top_iff: V equals the least schedule dose, with
//      and without holdover, from every start node and several start steps;
//   2. greedy_valid / safe_iff / safe_iff_le / safe_prefix / earliest_safe_arrival:
//      the DP's argmin schedule is valid with dose V; a schedule with dose < theta
//      (<= theta) exists iff V < theta (V <= theta); prefixes stay below theta; the
//      earliest arrival under the dose limit is the least horizon with V < theta;
//   3. strip_le / no_holdover / optimal_never_waits: with move doses non-decreasing in
//      the departure step (wait doses arbitrary but >= 0) deleting the waits from ANY
//      schedule never raises its dose, V without holdover equals V with holdover, and
//      with wait doses > 0 no optimal schedule waits; on non-monotone instances
//      holdover sometimes helps, and only when some move dose decreases;
//   4. oneLink_no_holdover_iff, decay_counterexample, relay_counterexample: the exact
//      hypothesis and the two counterexamples, with the numbers proved in Lean;
//   5. cost_eq_arrive / dp_earliest_arrival / earliest_escape_no_wait /
//      dp_is_earliest_escape / bfs_layer_iff_dp: with dose = time, V + t is the
//      earliest arrival; with unit links and the burn check t < ign(head) it is the
//      exit's layer in the earliest-escape BFS (GraphNumerics.bfs, L7).
// Doses are small integers so every sum is exact in doubles.
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

const WAIT = -1;
const randInt = (rand, lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));

// ---------------------------------------------------------------------------
// Reference specification: the backward induction (Lean `val`) and its argmin.
// ---------------------------------------------------------------------------

/** V[t][v] for t = 0..H+1 (row H+1 is all Infinity, Lean's `val_gt`). */
function minDoseDP(net, H, hold) {
  const n = net.exit.length;
  const V = Array.from({ length: H + 2 }, () => new Array(n).fill(Infinity));
  for (let t = H; t >= 0; t -= 1) {
    for (let v = 0; v < n; v += 1) {
      if (net.exit[v]) { V[t][v] = 0; continue; }
      let best = Infinity;
      if (hold && t + 1 <= H) best = Math.min(best, net.wait[v][t] + V[t + 1][v]);
      net.links.forEach((L, e) => {
        if (L.tail === v && t + L.tau <= H) best = Math.min(best, net.dose[e][t] + V[t + L.tau][L.head]);
      });
      V[t][v] = best;
    }
  }
  return V;
}

/** Follow the DP's argmins (Lean `Greedy`) from (v, t); ties broken by `preferWait`. */
function greedySchedule(net, H, hold, V, v, t, preferWait) {
  const acts = [];
  while (!net.exit[v]) {
    const waitOk = hold && t + 1 <= H && net.wait[v][t] + V[t + 1][v] === V[t][v];
    let move = -1;
    net.links.forEach((L, e) => {
      if (move < 0 && L.tail === v && t + L.tau <= H && net.dose[e][t] + V[t + L.tau][L.head] === V[t][v]) move = e;
    });
    if (waitOk && (preferWait || move < 0)) { acts.push(WAIT); t += 1; continue; }
    assert.ok(move >= 0, `no argmin action at node ${v}, step ${t} with V = ${V[t][v]}`);
    acts.push(move);
    t += net.links[move].tau;
    v = net.links[move].head;
  }
  return acts;
}

// ---------------------------------------------------------------------------
// Independent schedule semantics (Lean `Valid`, `cost`, `arrive`, `strip`).
// ---------------------------------------------------------------------------

/** Validity, dose (right-nested like Lean `cost`), arrival step and dose of every prefix. */
function evalSchedule(net, H, hold, v, t, acts) {
  let valid = true;
  const doses = [];
  for (const a of acts) {
    if (net.exit[v]) valid = false; // exits are absorbing
    if (a === WAIT) {
      if (!hold || t + 1 > H) valid = false;
      doses.push(net.wait[v][t] ?? Infinity);
      t += 1;
    } else {
      const L = net.links[a];
      if (L.tail !== v || t + L.tau > H) valid = false;
      doses.push(net.dose[a][t] ?? Infinity);
      t += L.tau;
      v = L.head;
    }
  }
  if (!net.exit[v] || t > H) valid = false;
  let cost = 0;
  for (let i = doses.length - 1; i >= 0; i -= 1) cost = doses[i] + cost;
  const prefix = [0];
  for (const d of doses) prefix.push(prefix[prefix.length - 1] + d);
  return { valid, cost, arrive: t, prefix };
}

const strip = (acts) => acts.filter((a) => a !== WAIT);

/** Brute force: call visit(acts, cost, arrive) for every valid schedule from (v0, t0). */
function enumerateSchedules(net, H, hold, v0, t0, visit) {
  const acts = [];
  const rec = (v, t, cost) => {
    if (net.exit[v]) { if (t <= H) visit(acts, cost, t); return; }
    if (hold && t + 1 <= H) { acts.push(WAIT); rec(v, t + 1, cost + net.wait[v][t]); acts.pop(); }
    net.links.forEach((L, e) => {
      if (L.tail !== v || t + L.tau > H) return;
      acts.push(e);
      rec(L.head, t + L.tau, cost + net.dose[e][t]);
      acts.pop();
    });
  };
  rec(v0, t0, 0);
}

// ---------------------------------------------------------------------------
// Random instances.
// ---------------------------------------------------------------------------

function randomTopology(rand, opts = {}) {
  const n = randInt(rand, 2, 5);
  const exit = Array.from({ length: n }, (_, v) => v > 0 && rand() < 0.3);
  if (!exit.some(Boolean)) exit[n - 1] = true;
  const m = randInt(rand, 1, 3 * n);
  const links = [];
  for (let e = 0; e < m; e += 1) {
    const tail = randInt(rand, 0, n - 1);
    let head = randInt(rand, 0, n - 1);
    if (head === tail && rand() < 0.7) head = (tail + 1) % n; // a few self-loops survive
    const tau = opts.unitTau ? 1 : rand() < 0.6 ? 1 : randInt(rand, 2, 3);
    links.push({ tail, head, tau });
  }
  return { n, exit, links };
}

/** Move doses for steps 0..H: "random", "monotone" (fire only grows; closures permanent) or "decay". */
function moveDoses(rand, links, H, mode) {
  return links.map(() => {
    const d = new Array(H + 1);
    if (mode === "monotone") {
      let x = randInt(rand, 0, 4);
      const close = rand() < 0.3 ? randInt(rand, 0, H) : Infinity;
      for (let t = 0; t <= H; t += 1) { d[t] = t >= close ? Infinity : x; x += randInt(rand, 0, 3); }
    } else if (mode === "decay") {
      let x = randInt(rand, 3, 12);
      for (let t = 0; t <= H; t += 1) { d[t] = x; x = Math.max(0, x - randInt(rand, 0, 4)); }
    } else {
      for (let t = 0; t <= H; t += 1) d[t] = rand() < 0.1 ? Infinity : randInt(rand, 0, 9);
    }
    return d;
  });
}

/** Wait doses for steps 0..H: arbitrary non-negative (not monotone), `minWait` lower bound. */
function waitDoses(rand, n, H, minWait = 0, pTop = 0.05) {
  return Array.from({ length: n }, () =>
    Array.from({ length: H + 1 }, () => (rand() < pTop ? Infinity : randInt(rand, minWait, 6))));
}

const isMonotone = (d, upTo) => d.slice(0, upTo + 1).every((x, i, a) => i === 0 || a[i - 1] <= x);

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

test("T17 exactness and dose constraint: DP = brute-force minimum; argmin schedule is safe", () => {
  const rand = mulberry32(20261027);
  const stats = { cases: 0, starts: 0, finite: 0, infinite: 0, schedules: 0, thetaChecks: 0, earliestChecks: 0 };
  for (let c = 0; c < 300; c += 1) {
    const top = randomTopology(rand);
    const H = randInt(rand, 2, 7);
    const mode = ["random", "monotone", "decay"][randInt(rand, 0, 2)];
    const net = { ...top, dose: moveDoses(rand, top.links, H, mode), wait: waitDoses(rand, top.n, H) };
    stats.cases += 1;
    for (const hold of [true, false]) {
      const V = minDoseDP(net, H, hold);
      const starts = [];
      for (let v = 0; v < net.n; v += 1) starts.push([v, 0]);
      starts.push([randInt(rand, 0, net.n - 1), randInt(rand, 1, H + 1)]);
      for (const [v, t] of starts) {
        stats.starts += 1;
        // 1. Exactness (dp_exact, dp_isLeast, val_eq_top_iff).
        let best = Infinity;
        const costs = [];
        const arrivals = [];
        enumerateSchedules(net, H, hold, v, t, (acts, cost, arrive) => {
          const ev = evalSchedule(net, H, hold, v, t, acts);
          assert.ok(ev.valid, `case ${c}: enumerated schedule is not valid`);
          assert.equal(ev.cost, cost);
          assert.equal(ev.arrive, arrive);
          assert.ok(V[t][v] <= cost, `case ${c}: V ${V[t][v]} above a schedule of dose ${cost} (val_le_cost)`);
          best = Math.min(best, cost);
          costs.push(cost);
          arrivals.push(arrive);
          stats.schedules += 1;
        });
        assert.equal(V[t][v], best, `case ${c} hold=${hold} (${v},${t}): DP ${V[t][v]} vs brute force ${best}`);
        if (!Number.isFinite(V[t][v])) { stats.infinite += 1; continue; }
        stats.finite += 1;

        // 2. The argmin schedule (greedy_valid), both tie-breaks.
        for (const preferWait of [false, true]) {
          const acts = greedySchedule(net, H, hold, V, v, t, preferWait);
          const ev = evalSchedule(net, H, hold, v, t, acts);
          assert.ok(ev.valid, `case ${c}: argmin schedule invalid`);
          assert.equal(ev.cost, V[t][v], `case ${c}: argmin schedule dose ${ev.cost} vs V ${V[t][v]}`);
          // safe_iff / safe_iff_le / greedy_safe / safe_prefix.
          for (const theta of [V[t][v] - 0.5, V[t][v], V[t][v] + 0.5, randInt(rand, 0, 25)]) {
            stats.thetaChecks += 1;
            assert.equal(costs.some((x) => x < theta), V[t][v] < theta, `case ${c}: safe_iff at theta ${theta}`);
            assert.equal(costs.some((x) => x <= theta), V[t][v] <= theta, `case ${c}: safe_iff_le at theta ${theta}`);
            if (V[t][v] < theta) {
              assert.ok(ev.cost < theta);
              for (const p of ev.prefix) assert.ok(p < theta, `case ${c}: prefix dose ${p} reaches theta ${theta}`);
            }
          }
        }

        // earliest_safe_arrival and val_antitone_horizon (start at step 0 only).
        if (t === 0) {
          const theta = V[0][v] + randInt(rand, 0, 6) + 0.5;
          let K = -1;
          let prev = Infinity;
          for (let h = 0; h <= H; h += 1) {
            const Vh = minDoseDP(net, h, hold)[0][v];
            assert.ok(Vh <= prev, `case ${c}: longer horizon raised the dose`);
            prev = Vh;
            if (K < 0 && Vh < theta) K = h;
          }
          const safeArrivals = arrivals.filter((_, i) => costs[i] < theta);
          assert.ok(K >= 0 && safeArrivals.length > 0);
          assert.equal(K, Math.min(...safeArrivals), `case ${c}: earliest safe arrival`);
          stats.earliestChecks += 1;
        }
      }
    }
  }
  console.log(JSON.stringify(stats));
  assert.ok(stats.finite > 500 && stats.infinite > 50, "the generator covers both reachable and cut-off starts");
});

test("T17 holdover: no holdover under monotone move doses; holdover can help otherwise", () => {
  const rand = mulberry32(7741);
  const stats = { monotoneCases: 0, stripChecks: 0, strictOptimalChecks: 0, nonMonotoneCases: 0, holdoverHelped: 0 };
  for (let c = 0; c < 300; c += 1) {
    const top = randomTopology(rand);
    const H = randInt(rand, 2, 7);
    // strip_le / no_holdover / exists_optimal_nowait: monotone move doses, ANY wait doses >= 0.
    {
      const strictWait = rand() < 0.5;
      const net = { ...top, dose: moveDoses(rand, top.links, H, "monotone"), wait: waitDoses(rand, top.n, H, strictWait ? 1 : 0) };
      stats.monotoneCases += 1;
      const Vh = minDoseDP(net, H, true);
      const Vn = minDoseDP(net, H, false);
      for (let t = 0; t <= H; t += 1) {
        for (let v = 0; v < net.n; v += 1) assert.equal(Vn[t][v], Vh[t][v], `case ${c}: no_holdover at (${v},${t})`);
      }
      for (let v = 0; v < net.n; v += 1) {
        enumerateSchedules(net, H, true, v, 0, (acts, cost) => {
          // strip_le, also started earlier (t' <= t): here t' = t = 0.
          const ev = evalSchedule(net, H, false, v, 0, strip(acts));
          assert.ok(ev.valid, `case ${c}: stripped schedule invalid`);
          assert.ok(ev.cost <= cost, `case ${c}: stripping raised the dose ${cost} -> ${ev.cost}`);
          stats.stripChecks += 1;
          // optimal_never_waits: with every wait dose > 0, an optimal schedule never waits.
          if (strictWait && Number.isFinite(cost) && cost === Vh[0][v]) {
            assert.ok(!acts.includes(WAIT), `case ${c}: an optimal schedule waits although waits cost > 0`);
            stats.strictOptimalChecks += 1;
          }
        });
      }
    }
    // Non-monotone move doses: when holdover helps, some move dose must decrease.
    {
      const mode = rand() < 0.5 ? "random" : "decay";
      const net = { ...top, dose: moveDoses(rand, top.links, H, mode), wait: waitDoses(rand, top.n, H, 0, 0) };
      stats.nonMonotoneCases += 1;
      const Vh = minDoseDP(net, H, true);
      const Vn = minDoseDP(net, H, false);
      let helped = false;
      for (let v = 0; v < net.n; v += 1) {
        assert.ok(Vh[0][v] <= Vn[0][v], `case ${c}: holdover raised the minimum`);
        if (Vh[0][v] < Vn[0][v]) helped = true;
      }
      if (helped) {
        stats.holdoverHelped += 1;
        assert.ok(net.dose.some((d) => !isMonotone(d, H)), `case ${c}: holdover helped with monotone move doses`);
      }
    }
  }
  console.log(JSON.stringify(stats));
  assert.ok(stats.holdoverHelped > 10, "holdover strictly helps on some non-monotone instances");
  assert.ok(stats.strictOptimalChecks > 100);
});

test("T17 exact hypothesis on one link and the two counterexamples (numbers as in Lean)", () => {
  // oneLink: node 0 (start) -> node 1 (exit), tau 1, free waiting.
  const oneLink = (d) => ({
    n: 2, exit: [false, true], links: [{ tail: 0, head: 1, tau: 1 }],
    dose: [d], wait: [d.map(() => 0), d.map(() => 0)],
  });
  // decay_counterexample: d = (1, 0, 0), H = 2: with holdover 0, without 1.
  const decay = oneLink([1, 0, 0]);
  assert.equal(minDoseDP(decay, 2, true)[0][0], 0);
  assert.equal(minDoseDP(decay, 2, false)[0][0], 1);

  // oneLink_no_holdover_iff: no holdover at every H <= Hmax, t <= H iff d monotone on 0..Hmax-1.
  const rand = mulberry32(99);
  const Hmax = 6;
  let monotoneSeen = 0, nonMonotoneSeen = 0;
  for (let c = 0; c < 400; c += 1) {
    const d = Array.from({ length: Hmax + 1 }, () => randInt(rand, 0, 4));
    if (rand() < 0.4) d.sort((a, b) => a - b);
    if (rand() < 0.1) for (let t = randInt(rand, 0, Hmax); t <= Hmax; t += 1) d[t] = Infinity; // permanent closure
    const net = oneLink(d);
    let same = true;
    for (let H = 0; H <= Hmax; H += 1) {
      const Vh = minDoseDP(net, H, true), Vn = minDoseDP(net, H, false);
      for (let t = 0; t <= H; t += 1) if (Vh[t][0] !== Vn[t][0]) same = false;
    }
    const mono = isMonotone(d, Hmax - 1);
    if (mono) monotoneSeen += 1; else nonMonotoneSeen += 1;
    assert.equal(same, mono, `one link ${JSON.stringify(d)}: no-holdover ${same} vs monotone ${mono}`);
  }
  assert.ok(monotoneSeen > 50 && nonMonotoneSeen > 50);

  // relay_counterexample: 0 -> 1 (dose 0), 1 -> 2 (dose 3 if departing at t <= 1, else 0),
  // waiting free at node 0 and 5 at node 1; node 2 is the exit; H = 3.
  const H = 3;
  const relay = {
    n: 3, exit: [false, false, true],
    links: [{ tail: 0, head: 1, tau: 1 }, { tail: 1, head: 2, tau: 1 }],
    dose: [[0, 0, 0, 0, 0], [3, 3, 0, 0, 0]],
    wait: [[0, 0, 0, 0, 0], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]],
  };
  // The local exchange test d_e(t) <= w_tail(e)(t) + d_e(t+1) holds for every link and step...
  relay.links.forEach((L, e) => {
    for (let t = 0; t < 4; t += 1) assert.ok(relay.dose[e][t] <= relay.wait[L.tail][t] + relay.dose[e][t + 1]);
  });
  // ...yet holdover lowers the optimum from 3 to 0 (wait at node 0, then go).
  assert.equal(minDoseDP(relay, H, true)[0][0], 0);
  assert.equal(minDoseDP(relay, H, false)[0][0], 3);
  const ev = evalSchedule(relay, H, true, 0, 0, [WAIT, 0, 1]);
  assert.ok(ev.valid);
  assert.equal(ev.cost, 0);
});

test("T17 dose = time: the DP is earliest arrival and reproduces the earliest-escape BFS (L7)", () => {
  const rand = mulberry32(31337);
  const stats = { cases: 0, finite: 0, bfsCases: 0, bfsReached: 0 };
  for (let c = 0; c < 300; c += 1) {
    const top = randomTopology(rand);
    const H = randInt(rand, 2, 7);
    // cost_eq_arrive / dp_earliest_arrival: d = tau on open links (open at random), w in {1, Infinity}.
    const permanent = rand() < 0.5;
    const close = top.links.map(() => (rand() < 0.4 ? randInt(rand, 0, H) : Infinity));
    const dose = top.links.map((L, e) => Array.from({ length: H + 1 }, (_, t) => {
      const open = permanent ? t < close[e] : rand() < 0.8;
      return open ? L.tau : Infinity;
    }));
    const wait = Array.from({ length: top.n }, () => Array.from({ length: H + 1 }, () => (permanent || rand() < 0.9 ? 1 : Infinity)));
    const net = { ...top, dose, wait };
    stats.cases += 1;
    for (const hold of [true, false]) {
      const V = minDoseDP(net, H, hold);
      for (let v = 0; v < net.n; v += 1) {
        let earliest = Infinity;
        enumerateSchedules(net, H, hold, v, 0, (acts, cost, arrive) => {
          if (!Number.isFinite(cost)) return;
          assert.equal(cost, arrive, `case ${c}: dose ${cost} is not the travel time ${arrive}`);
          earliest = Math.min(earliest, arrive);
        });
        assert.equal(V[0][v], earliest, `case ${c} hold=${hold}: DP ${V[0][v]} vs earliest arrival ${earliest}`);
        if (Number.isFinite(earliest)) stats.finite += 1;
      }
    }
    // earliest_escape_no_wait: permanent closures and unit waits -> holdover does not help.
    if (permanent) {
      const Vh = minDoseDP(net, H, true), Vn = minDoseDP(net, H, false);
      for (let v = 0; v < net.n; v += 1) assert.equal(Vn[0][v], Vh[0][v], `case ${c}: waiting helped earliest arrival`);
    }

    // dp_is_earliest_escape / bfs_layer_iff_dp: unit links, burn check t < ign(head), one exit.
    const u = randomTopology(rand, { unitTau: true });
    const x = u.n - 1;
    const exit = u.exit.map((_, v) => v === x);
    const ign = Array.from({ length: u.n }, () => (rand() < 0.3 ? Infinity : randInt(rand, 0, H + 1)));
    const unet = {
      ...u, exit,
      dose: u.links.map((L) => Array.from({ length: H + 1 }, (_, t) => (t < ign[L.head] ? 1 : Infinity))),
      wait: Array.from({ length: u.n }, () => new Array(H + 1).fill(1)),
    };
    const s0 = randInt(rand, 0, u.n - 1);
    // GraphNumerics.bfs: layer 0 = {s0}; layer k+1 = unseen v with a link from layer k and k < ign(v).
    let frontier = new Set([s0]);
    const seen = new Set([s0]);
    let layer = frontier.has(x) ? 0 : Infinity;
    for (let k = 0; k < H && !Number.isFinite(layer); k += 1) {
      const next = new Set();
      for (const L of u.links) if (frontier.has(L.tail) && !seen.has(L.head) && k < ign[L.head]) next.add(L.head);
      for (const w of next) seen.add(w);
      frontier = next;
      if (frontier.has(x)) layer = k + 1;
    }
    stats.bfsCases += 1;
    if (Number.isFinite(layer)) stats.bfsReached += 1;
    for (const hold of [false, true]) {
      assert.equal(minDoseDP(unet, H, hold)[0][s0], layer, `case ${c} hold=${hold}: DP vs earliest-escape BFS layer`);
    }
  }
  console.log(JSON.stringify(stats));
  assert.ok(stats.bfsReached > 50 && stats.bfsReached < stats.bfsCases, "BFS cases cover reached and unreached exits");
});
