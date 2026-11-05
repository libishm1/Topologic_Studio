// Run with: node --test src/core/rlSafe.test.mjs
//
// Tests for the shielded learned escape (src/core/rlSafe.js), each against a
// statement of formal/fireegress/FireEgress/SafeEscape.lean, on small random
// grids (seeded) with walls, several exits, fire origins, fire speeds and step
// limits, where everything can be checked by brute force:
//   - deadline_iff: the deadline map satisfies the max-min recursion, and
//     t <= D(v) exactly when v is tenable on arrival at t and an exit can
//     still be reached (finite-horizon value iteration, unpruned, as oracle);
//   - vi_eq_earliest: value iteration equals the earliest safe escape, which
//     equals rl.js's earliestEscape for a single exit;
//   - shield_safe, shield_nonempty, shield_reaches_exit: random walks inside
//     the shield never burn, never get stuck and reach an exit by step H;
//   - shield_complete: a move is masked only when no escape goes through it;
//   - lrta_admissible, certificate_optimal: learned values never exceed the
//     exact ones, and a certified result is an earliest escape;
//   - arrival_mask_traps, node_policy_fails: the two counterexamples.
import assert from "node:assert/strict";
import { test } from "node:test";

import { earliestEscape } from "./rl.js";
import {
  NEVER,
  StepTable,
  earliestSafeEscape,
  escapeField,
  escapeValues,
  exitHops,
  hopDeadlines,
  ignitionSteps,
  learnSafeEscape,
  openLinks,
  openUntil,
  trainSafeRl,
} from "./rlSafe.js";

function grid(w, h) {
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

/** A graph from an edge list, nodes on a line (lengths do not matter here). */
function fromEdges(count, pairs) {
  const nodes = [];
  for (let i = 0; i < count; i += 1) nodes.push(i, 0, 0);
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(pairs.flat()), blocked: new Uint8Array(pairs.length) };
}

function lcg(seed) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648;
  };
}

/**
 * Random cases: grid, walls, exits, fire (or none), speed and step limit.
 * Odd case numbers are races: start on the left edge, an exit on the right
 * edge, the fire near the middle and slower than the walker, a limit long
 * enough for a detour. Those are the cases that need more than one trial and
 * that trap an agent masking only untenable nodes.
 */
function* cases(count, seed) {
  const rand = lcg(seed);
  for (let c = 0; c < count; c += 1) {
    if (c % 2) {
      const w = 5 + Math.floor(rand() * 8), h = 4 + Math.floor(rand() * 6);
      const g = grid(w, h);
      for (let e = 0; e < g.blocked.length; e += 1) if (rand() < 0.25) g.blocked[e] = 1;
      const n = w * h;
      const start = Math.floor(rand() * h) * w;
      const exits = [Math.floor(rand() * h) * w + w - 1];
      if (rand() < 0.5) exits.push(Math.floor(rand() * n));
      const H = 2 * (w + h) + Math.floor(rand() * 20);
      const fire = Math.floor(rand() * h) * w + Math.floor(w / 2) + Math.floor(rand() * 3) - 1;
      const speed = [0.25, 0.34, 0.5, 0.75, 1][Math.floor(rand() * 5)];
      const ig = ignitionSteps(g, fire, { fireSpeed: speed, maxSteps: H, blocked: g.blocked });
      yield { c, g, n, exits, start, H, ig, fire, speed };
      continue;
    }
    const w = 3 + Math.floor(rand() * 6), h = 2 + Math.floor(rand() * 5);
    const g = grid(w, h);
    for (let e = 0; e < g.blocked.length; e += 1) if (rand() < 0.15) g.blocked[e] = 1;
    const n = w * h;
    const exits = [Math.floor(rand() * n)];
    if (rand() < 0.4) exits.push(Math.floor(rand() * n));
    const start = Math.floor(rand() * n);
    const H = 2 + Math.floor(rand() * 25);
    const fire = rand() < 0.85 ? Math.floor(rand() * n) : null;
    const speed = [0.25, 0.5, 1, 1.5, 2][Math.floor(rand() * 5)];
    const ig = ignitionSteps(g, fire, { fireSpeed: speed, maxSteps: H, blocked: g.blocked });
    yield { c, g, n, exits, start, H, ig, fire, speed };
  }
}

test("the deadline map solves the max-min recursion and is exactly viability (deadline_iff)", () => {
  for (const { c, g, n, exits, H, ig } of cases(150, 11)) {
    const D = hopDeadlines(g, { exits, ignite: ig, horizon: H, blocked: g.blocked });
    const open = openUntil(ig, H);
    const { first, to } = openLinks(g, g.blocked);
    const isExit = new Set(exits);
    for (let v = 0; v < n; v += 1) {
      if (isExit.has(v)) {
        assert.equal(D[v], open[v], `case ${c} exit ${v}`);
        continue;
      }
      let best = -1;
      for (let j = first[v]; j < first[v + 1]; j += 1) best = Math.max(best, D[to[j]]);
      assert.equal(D[v], Math.max(-1, Math.min(open[v], best - 1)), `case ${c} node ${v}`);
    }
    const { rows } = escapeValues(g, { exits, ignite: ig, horizon: H, blocked: g.blocked, prune: false, rows: true });
    for (let v = 0; v < n; v += 1) {
      for (let t = 0; t <= H; t += 1) {
        const viable = t <= open[v] && rows[t][v] < NEVER;
        assert.equal(t <= D[v], viable, `case ${c} node ${v} step ${t}: D ${D[v]}, V ${rows[t][v]}`);
      }
    }
    // Pruning by the deadline changes no value.
    const pruned = escapeValues(g, { exits, ignite: ig, horizon: H, blocked: g.blocked });
    assert.deepEqual(Array.from(pruned.moves), Array.from(rows[0]));
  }
});

test("value iteration equals the earliest safe escape, and rl.js's for one exit (vi_eq_earliest)", () => {
  for (const { c, g, exits, start, H, ig } of cases(200, 23)) {
    const { moves } = escapeValues(g, { exits, ignite: ig, horizon: H, blocked: g.blocked });
    const bfs = earliestSafeEscape(g, start, { exits, ignite: ig, horizon: H, blocked: g.blocked });
    assert.equal(bfs ? bfs.moves : NEVER, moves[start], `case ${c}`);
    if (bfs) {
      // The route is a walk of open links, safe on every arrival.
      const open = openUntil(ig, H);
      assert.equal(bfs.path.length - 1, bfs.moves);
      for (let i = 1; i < bfs.path.length; i += 1) assert.ok(i <= open[bfs.path[i]], `case ${c}: unsafe at ${i}`);
      assert.ok(exits.includes(bfs.path[bfs.path.length - 1]));
    }
    if (exits.length === 1 && start !== exits[0]) {
      const map = new Map();
      ig.forEach((s, v) => Number.isFinite(s) && map.set(v, s));
      const old = earliestEscape(g, start, exits[0], map, H, g.blocked);
      assert.equal(old ?? NEVER, moves[start], `case ${c}: rl.js ${old}`);
    }
  }
});

test("random walks inside the shield never burn, never stall and reach an exit by the limit (shield_*)", () => {
  const rand = lcg(5);
  for (const { c, g, exits, start, H, ig } of cases(150, 37)) {
    const D = hopDeadlines(g, { exits, ignite: ig, horizon: H, blocked: g.blocked });
    if (D[start] < 0 || exits.includes(start)) continue;
    const open = openUntil(ig, H);
    const { first, to } = openLinks(g, g.blocked);
    for (let trial = 0; trial < 20; trial += 1) {
      let v = start, t = 0;
      while (!exits.includes(v)) {
        const allowed = [];
        for (let j = first[v]; j < first[v + 1]; j += 1) if (t + 1 <= D[to[j]]) allowed.push(to[j]);
        assert.ok(allowed.length > 0, `case ${c}: stalled at ${v}, step ${t}`);
        v = allowed[Math.floor(rand() * allowed.length)];
        t += 1;
        assert.ok(t <= open[v], `case ${c}: burned at ${v}, step ${t}`);
        assert.ok(t <= H, `case ${c}: past the limit`);
      }
    }
  }
});

test("the shield masks a move only when no escape goes through it (shield_complete)", () => {
  for (const { c, g, n, exits, H, ig } of cases(120, 41)) {
    const D = hopDeadlines(g, { exits, ignite: ig, horizon: H, blocked: g.blocked });
    const open = openUntil(ig, H);
    const { rows } = escapeValues(g, { exits, ignite: ig, horizon: H, blocked: g.blocked, prune: false, rows: true });
    for (let w = 0; w < n; w += 1) {
      for (let t = 0; t < H; t += 1) {
        const throughW = t + 1 <= open[w] && rows[t + 1][w] < NEVER;
        assert.equal(t + 1 <= D[w], throughW, `case ${c}: move into ${w} at ${t}`);
      }
    }
  }
});

test("learned values never exceed the exact ones, and a certified result is an earliest escape (lrta_admissible)", () => {
  let certified = 0, total = 0, learned = 0, trapped = 0;
  for (const { c, g, n, exits, start, H, ig } of cases(300, 53)) {
    const { rows } = escapeValues(g, { exits, ignite: ig, horizon: H, blocked: g.blocked, prune: false, rows: true });
    const exact = rows[0][start];
    for (const shield of ["deadline", "arrival", "none"]) {
      const r = learnSafeEscape(g, start, { exits, ignite: ig, horizon: H, blocked: g.blocked, trials: 200, shield });
      if (shield === "arrival" && exact < NEVER && r.curve[0] == null) trapped += 1;
      // A breakpoint holds at its step and every later one; the exact value
      // never falls with the step (val_mono_time), so checking its own step
      // and the last one suffices, and the monotonicity is checked too.
      for (const [v, t, value] of r.values.entries()) {
        assert.ok(v < n);
        // FAIL (H + 1) stands for "no escape": admissible against NEVER.
        for (const s of [t, H]) assert.ok(value <= rows[s][v] || (value === H + 1 && rows[s][v] === NEVER), `case ${c} ${shield}: V(${v}, ${s}) = ${value} > ${rows[s][v]}`);
      }
      for (let v = 0; v < n; v += 1) for (let s = 0; s < H; s += 1) assert.ok(rows[s][v] <= rows[s + 1][v], `case ${c}: V*(${v}) falls at ${s}`);
      if (r.reachedExit) assert.ok(r.moves >= exact, `case ${c} ${shield}: ${r.moves} < ${exact}`);
      if (r.certified) assert.equal(r.moves, exact, `case ${c} ${shield}`);
      if (shield === "deadline") {
        // With the deadline shield the policy escapes whenever an escape exists.
        assert.equal(r.reachedExit, exact < NEVER, `case ${c}`);
        if (exact < NEVER) {
          total += 1;
          if (r.certified) certified += 1;
          if (r.trials > 1) learned += 1;
          for (const m of r.curve) assert.ok(m != null, `case ${c}: a shielded trial failed`);
        }
      }
    }
  }
  // On these small cases learning always ends with the certificate, some of
  // them after several trials, and the arrival mask alone traps some first
  // trials that the deadline shield does not.
  assert.equal(certified, total);
  assert.ok(learned >= 5, `only ${learned} cases needed a second trial`);
  assert.ok(trapped >= 5, `only ${trapped} first trials trapped under the arrival mask`);
});

test("one value table serves every start: learning from several starts stays admissible", () => {
  for (const { c, g, n, exits, H, ig } of cases(40, 61)) {
    const { rows } = escapeValues(g, { exits, ignite: ig, horizon: H, blocked: g.blocked, prune: false, rows: true });
    const values = new StepTable();
    for (let s = 0; s < n; s += 1) {
      const r = learnSafeEscape(g, s, { exits, ignite: ig, horizon: H, blocked: g.blocked, trials: 40, values });
      if (r.certified) assert.equal(r.moves, rows[0][s], `case ${c} start ${s}`);
    }
    for (const [v, t, value] of values.entries()) {
      for (const s of [t, H]) assert.ok(value <= rows[s][v] || (value === H + 1 && rows[s][v] === NEVER), `case ${c}`);
    }
  }
});

test("without fire the first trial is a shortest route and is certified", () => {
  const g = grid(12, 9);
  const r = learnSafeEscape(g, 0, { exits: [107], horizon: 200, blocked: g.blocked });
  assert.equal(r.trials, 1);
  assert.ok(r.certified);
  assert.equal(r.moves, exitHops(g, { exits: [107] })[0]);
});

test("masking only nodes untenable on arrival can trap the agent; the deadline shield cannot (arrival_mask_traps)", () => {
  // S = 0, P = 1, X = 2 (exit), F = 3 (fire origin): S - P, S - X, S - F.
  // The fire reaches S at step 1 and P and X at step 2. Leaving S at once for
  // X escapes; stepping into P is safe on arrival, but then S has burned.
  const g = fromEdges(4, [[0, 1], [0, 2], [0, 3]]);
  const ig = ignitionSteps(g, 3, { fireSpeed: 1, maxSteps: 10 });
  assert.deepEqual(Array.from(ig), [1, 2, 2, 0]);
  const exits = [2];
  const D = hopDeadlines(g, { exits, ignite: ig, horizon: 10 });
  assert.deepEqual(Array.from(D), [1, 0, 2, 0]);
  assert.equal(earliestSafeEscape(g, 0, { exits, ignite: ig, horizon: 10 }).moves, 1);
  // The arrival mask allows S -> P at step 0, and then nothing at (P, 1).
  const open = openUntil(ig, 10);
  assert.ok(0 + 1 <= open[1]);
  assert.ok(!(1 + 1 <= open[0]));
  // The deadline shield masks S -> P.
  assert.ok(!(0 + 1 <= D[1]));
});

test("no policy on the node alone is right from every start (node_policy_fails)", () => {
  // S1 = 0 - v = 2, S2 = 1 - a = 3 - v, v - X1 = 4 (exit, burns at step 2),
  // v - B = 5 - X2 = 6 (exit). Fire origin F = 7 attached to X1 through
  // F - 8 - X1, so X1 is tenable on arrival up to step 2.
  const g = fromEdges(9, [[0, 2], [1, 3], [3, 2], [2, 4], [2, 5], [5, 6], [7, 8], [8, 4]]);
  const ig = new Float64Array(9).fill(Infinity);
  ig[7] = 0;
  ig[8] = 1;
  ig[4] = 2;
  const exits = [4, 6];
  const H = 10;
  const { rows } = escapeValues(g, { exits, ignite: ig, horizon: H, prune: false, rows: true });
  assert.equal(rows[0][0], 2); // S1: v at 1, X1 at 2
  assert.equal(rows[0][1], 4); // S2: a, v at 2, B, X2 at 4
  // At (v, 1) the only optimal move is X1; at (v, 2) it is unsafe.
  const open = openUntil(ig, H);
  assert.equal(rows[1][2], 1);
  assert.equal(rows[2][5] + 1, 2); // via B from (v, 1) takes 2 moves
  assert.ok(!(2 + 1 <= open[4]));
  const field = escapeField(g, { exits, ignite: ig, horizon: H });
  assert.deepEqual([field.steps[0], field.steps[1]], [2, 4]);
});

test("trainSafeRl: the RL response shape, several exits, and the notes", () => {
  const g = grid(10, 6);
  const r = trainSafeRl(g, { startPoint: [0, 0, 0], exitPoints: [[0, 5, 0], [9, 0, 0]], firePoint: [3, 5, 0], trials: 100, maxSteps: 60, fireSpeed: 1 });
  assert.equal(r.method, "shielded");
  assert.ok(r.reached_exit);
  assert.equal(r.burned, false);
  assert.equal(r.moves, r.escape_steps);
  assert.ok(r.certified);
  assert.equal(r.points.length, r.path.length);
  // The nearer exit (0, 5) burns before it can be reached; the far one does not.
  assert.equal(r.exit_id, 9);
  assert.equal(r.moves, 9);
  // A fire on the exit itself: no escape, and the note says so.
  const none = trainSafeRl(g, { startPoint: [0, 0, 0], exitPoint: [9, 0, 0], firePoint: [9, 0, 0], trials: 20, maxSteps: 30 });
  assert.equal(none.reached_exit, false);
  assert.equal(none.escape_steps, null);
  assert.match(none.note, /No route escapes/);
  // Without fire and with too short a limit.
  const far = trainSafeRl(g, { startPoint: [0, 0, 0], exitPoint: [9, 5, 0], useFire: false, maxSteps: 5 });
  assert.match(far.note, /raise the step limit/);
});
