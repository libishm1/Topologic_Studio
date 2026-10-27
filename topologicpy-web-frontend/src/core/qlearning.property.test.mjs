// Run with: node --test src/core/qlearning.property.test.mjs
//
// Property test for FireEgress/QLearning.lean (L15, formal/fireegress): tabular Q-learning
// as src/core/rl.js runs it, in the deterministic setting. A problem is written on its
// state-action pairs, as in the Lean `DetMDP`: pair p is taken at state st[p], pays r[p], and
// is followed by a choice among succ[p] (the actions of the next state; [] = terminal). The
// Bellman operator is (T Q)(p) = r[p] + gamma * max_{q in succ[p]} Q[q], with max [] = 0,
// exactly rl.js's target. Q* is computed exactly by policy iteration (a linear solve per
// policy), and on levelled (time-indexed) problems also by backward induction.
//
// On small random problems (seeded mulberry32) it checks:
//   1. bellman_contracting / value_iteration_error: ||T^n Q0 - Q*|| <= gamma^n ||Q0 - Q*||;
//   2. update_dist_le / sweep_contracts / qlearning_tendsto: one update never increases the
//      sup distance to Q*, a window that updates every pair with steps >= alpha_min shrinks it
//      by 1 - alpha_min (1 - gamma), and repeated sweeps converge to Q*; on levelled problems
//      (dag_value_iteration_exact / qlearning_tendsto_dag) the same holds with gamma = 1;
//   3. shaped_qstar_eq / shaped_same_greedy / shaped_fixed_general: with Phi(terminal) = 0,
//      Q*_shaped = Q* - Phi(st), so the greedy actions are the same; with Phi(terminal) != 0,
//      Q*_shaped = Q*_bonus - Phi(st) and the greedy action can change (the Lean
//      counterexamples nonzero_terminal_flips_qstar and horizon_potential_prefers_fire);
//   4. Studio's own trainRl, without fire, on tiny graphs: it reaches the exit by a shortest
//      route.
import assert from "node:assert/strict";
import { test } from "node:test";

import { trainRl } from "./rl.js";

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

const randInt = (rand, lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
const uniform = (rand, lo, hi) => lo + (hi - lo) * rand();

// ---------------------------------------------------------------------------
// Problems on state-action pairs (the Lean DetMDP).
// ---------------------------------------------------------------------------

/**
 * A random deterministic problem. States 0..nStates-1, each with 1..maxActions actions; an
 * action is terminal with probability pTerminal, else it leads to a random state. With
 * `levelled`, actions only lead to higher-numbered states (a DAG, like time-indexed states
 * with the horizon terminal), so gamma = 1 is allowed.
 */
function randomProblem(rand, { nStates, maxActions, pTerminal, levelled = false }) {
  const st = [], r = [], next = [], actionsOf = [];
  for (let s = 0; s < nStates; s += 1) {
    actionsOf.push([]);
    const k = randInt(rand, 1, maxActions);
    for (let a = 0; a < k; a += 1) {
      const p = st.length;
      st.push(s);
      r.push(Math.round(uniform(rand, -10, 10) * 10) / 10);
      let ns = -1;
      if (rand() >= pTerminal) ns = levelled ? (s + 1 < nStates ? randInt(rand, s + 1, nStates - 1) : -1) : randInt(rand, 0, nStates - 1);
      next.push(ns);
      actionsOf[s].push(p);
    }
  }
  const succ = next.map((ns) => (ns >= 0 ? actionsOf[ns] : []));
  return { st, r, next, actionsOf, succ, n: st.length };
}

const best = (Q, s) => (s.length ? Math.max(...s.map((q) => Q[q])) : 0);
const bellman = (P, gamma, Q) => P.r.map((rp, p) => rp + gamma * best(Q, P.succ[p]));
const supDist = (A, B) => A.reduce((m, a, i) => Math.max(m, Math.abs(a - B[i])), 0);

/** Solve A x = b by Gaussian elimination with partial pivoting (small n). */
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c += 1) {
    let piv = c;
    for (let i = c + 1; i < n; i += 1) if (Math.abs(M[i][c]) > Math.abs(M[piv][c])) piv = i;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let i = c + 1; i < n; i += 1) {
      const f = M[i][c] / M[c][c];
      if (f !== 0) for (let j = c; j <= n; j += 1) M[i][j] -= f * M[c][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i -= 1) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j += 1) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

/** Q* by policy iteration: evaluate the policy exactly, improve, until it is stable. */
function exactQstar(P, gamma, r = P.r) {
  const n = P.n;
  const pi = P.succ.map((s) => (s.length ? s[0] : -1));
  for (let iter = 0; iter < 1000; iter += 1) {
    const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let p = 0; p < n; p += 1) if (pi[p] >= 0) A[p][pi[p]] -= gamma;
    const Q = solve(A, r);
    let changed = false;
    for (let p = 0; p < n; p += 1) {
      let b = pi[p];
      for (const q of P.succ[p]) if (Q[q] > Q[b] + 1e-12) b = q;
      if (b !== pi[p]) {
        pi[p] = b;
        changed = true;
      }
    }
    if (!changed) return Q;
  }
  throw new Error("policy iteration did not stabilise");
}

/** Backward induction on a levelled problem (actions only lead to higher states). */
function backwardInduction(P, gamma, r = P.r) {
  const Q = new Array(P.n).fill(0);
  const V = new Array(P.actionsOf.length).fill(0);
  for (let s = P.actionsOf.length - 1; s >= 0; s -= 1) {
    for (const p of P.actionsOf[s]) Q[p] = r[p] + (P.next[p] >= 0 ? gamma * V[P.next[p]] : 0);
    V[s] = Math.max(...P.actionsOf[s].map((p) => Q[p]));
  }
  return Q;
}

/** One Q-learning update of pair p with step alpha (Lean qUpdate; rl.js q.set). */
function qUpdate(P, gamma, alpha, Q, p) {
  const target = P.r[p] + gamma * best(Q, P.succ[p]);
  Q[p] = Q[p] + alpha * (target - Q[p]);
}

function shuffled(rand, n) {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Greedy pairs among the actions of state s, with a tolerance for exact ties. */
function greedySet(P, Q, s, tol = 1e-9) {
  const top = Math.max(...P.actionsOf[s].map((p) => Q[p]));
  return P.actionsOf[s].filter((p) => Q[p] >= top - tol);
}

// ---------------------------------------------------------------------------
// 1. Value iteration (L15.1).
// ---------------------------------------------------------------------------

test("Q* from policy iteration solves the Bellman equation, and on levelled problems equals backward induction", () => {
  const rand = mulberry32(0x5eed01);
  for (let trial = 0; trial < 300; trial += 1) {
    const levelled = trial % 2 === 1;
    const P = randomProblem(rand, { nStates: randInt(rand, 1, 6), maxActions: 3, pTerminal: 0.3, levelled });
    const gamma = levelled && trial % 4 === 1 ? 1 : uniform(rand, 0, 0.95);
    const Qs = exactQstar(P, gamma);
    assert.ok(supDist(bellman(P, gamma, Qs), Qs) < 1e-9, `trial ${trial}: Bellman residual`);
    if (levelled) {
      assert.ok(supDist(backwardInduction(P, gamma), Qs) < 1e-9, `trial ${trial}: backward induction`);
      // dag_value_iteration_exact: nStates sweeps from any table are exact.
      let Q = P.r.map(() => uniform(rand, -50, 50));
      for (let k = 0; k < P.actionsOf.length; k += 1) Q = bellman(P, gamma, Q);
      assert.ok(supDist(Q, Qs) < 1e-9, `trial ${trial}: value iteration exact after H sweeps`);
    }
  }
});

test("value iteration: ||T^n Q0 - Q*|| <= gamma^n ||Q0 - Q*|| (value_iteration_error)", () => {
  const rand = mulberry32(0x5eed02);
  for (let trial = 0; trial < 300; trial += 1) {
    const P = randomProblem(rand, { nStates: randInt(rand, 1, 7), maxActions: 3, pTerminal: 0.25 });
    const gamma = uniform(rand, 0, 0.95);
    const Qs = exactQstar(P, gamma);
    let Q = P.r.map(() => uniform(rand, -40, 40));
    const e0 = supDist(Q, Qs);
    for (let n = 1; n <= 40; n += 1) {
      const before = supDist(Q, Qs);
      Q = bellman(P, gamma, Q);
      const after = supDist(Q, Qs);
      assert.ok(after <= gamma * before + 1e-9, `trial ${trial}, n ${n}: one sweep contracts by gamma`);
      assert.ok(after <= gamma ** n * e0 + 1e-9, `trial ${trial}, n ${n}: ${after} > gamma^n e0`);
    }
  }
});

// ---------------------------------------------------------------------------
// 2. Asynchronous Q-learning with a constant or bounded-below step (L15.2, L15.3).
// ---------------------------------------------------------------------------

function runSweeps(rand, P, gamma, Qs, { alphaMin, constantAlpha = null, maxWindows, label }) {
  const Q = P.r.map(() => uniform(rand, -40, 40));
  const rho = 1 - alphaMin * (1 - gamma);
  let windows = 0;
  while (supDist(Q, Qs) > 1e-10 && windows < maxWindows) {
    const start = supDist(Q, Qs);
    // A window: every pair once in random order, plus random extra updates.
    const order = shuffled(rand, P.n);
    for (let extra = randInt(rand, 0, P.n); extra > 0; extra -= 1) order.splice(randInt(rand, 0, order.length), 0, randInt(rand, 0, P.n - 1));
    for (const p of order) {
      const alpha = constantAlpha ?? uniform(rand, alphaMin, 1);
      const before = supDist(Q, Qs);
      qUpdate(P, gamma, alpha, Q, p);
      assert.ok(supDist(Q, Qs) <= before + 1e-9, `${label}: an update increased the distance`);
    }
    if (gamma < 1) assert.ok(supDist(Q, Qs) <= rho * start + 1e-9, `${label}: window did not contract by ${rho}`);
    windows += 1;
  }
  return supDist(Q, Qs);
}

test("one update never moves Q away from Q*, a covering window contracts by 1 - alpha_min (1 - gamma), sweeps converge (sweep_contracts, qlearning_tendsto)", () => {
  const rand = mulberry32(0x5eed03);
  for (let trial = 0; trial < 120; trial += 1) {
    const P = randomProblem(rand, { nStates: randInt(rand, 1, 6), maxActions: 3, pTerminal: 0.25 });
    const gamma = uniform(rand, 0, 0.9);
    const alphaMin = uniform(rand, 0.3, 1);
    const Qs = exactQstar(P, gamma);
    const d = runSweeps(rand, P, gamma, Qs, { alphaMin, maxWindows: 4000, label: `trial ${trial}` });
    assert.ok(d <= 1e-8, `trial ${trial}: did not converge (${d})`);
  }
});

test("Studio's constant step alpha = 0.5 with gamma = 0.99 converges to Q* (no Robbins-Monro needed)", () => {
  const rand = mulberry32(0x5eed04);
  for (let trial = 0; trial < 40; trial += 1) {
    const P = randomProblem(rand, { nStates: randInt(rand, 1, 5), maxActions: 3, pTerminal: 0.4 });
    const Qs = exactQstar(P, 0.99);
    const d = runSweeps(rand, P, 0.99, Qs, { alphaMin: 0.5, constantAlpha: 0.5, maxWindows: 20000, label: `trial ${trial}` });
    assert.ok(d <= 1e-8, `trial ${trial}: did not converge (${d})`);
  }
});

test("levelled (time-indexed) problems: Q-learning converges to backward induction also with gamma = 1 (qlearning_tendsto_dag)", () => {
  const rand = mulberry32(0x5eed05);
  for (let trial = 0; trial < 120; trial += 1) {
    const P = randomProblem(rand, { nStates: randInt(rand, 1, 6), maxActions: 3, pTerminal: 0.2, levelled: true });
    const gamma = trial % 2 ? 1 : uniform(rand, 0, 1);
    const Qs = backwardInduction(P, gamma);
    const d = runSweeps(rand, P, gamma, Qs, { alphaMin: uniform(rand, 0.3, 1), maxWindows: 4000, label: `trial ${trial}` });
    assert.ok(d <= 1e-8, `trial ${trial}: did not converge (${d})`);
  }
});

// ---------------------------------------------------------------------------
// 3. Shaping at the level of Q* (L15.4, L15.5).
// ---------------------------------------------------------------------------

/** Shaped rewards r + gamma psi - Phi(st), with psi = Phi(next), or `terminal[p]` if terminal. */
function shapedRewards(P, gamma, Phi, terminal) {
  return P.r.map((rp, p) => rp + gamma * (P.next[p] >= 0 ? Phi[P.next[p]] : terminal[p]) - Phi[P.st[p]]);
}

test("Phi(terminal) = 0: Q*_shaped = Q* - Phi(st) and the greedy actions are unchanged (shaped_qstar_eq, shaped_same_greedy)", () => {
  const rand = mulberry32(0x5eed06);
  for (let trial = 0; trial < 300; trial += 1) {
    const levelled = trial % 3 === 0;
    const P = randomProblem(rand, { nStates: randInt(rand, 1, 6), maxActions: 3, pTerminal: 0.3, levelled });
    const gamma = levelled ? (trial % 2 ? 1 : uniform(rand, 0, 1)) : uniform(rand, 0, 0.95);
    const Phi = Array.from({ length: P.actionsOf.length }, () => Math.round(uniform(rand, -15, 5) * 10) / 10);
    const Qs = exactQstar(P, gamma);
    const Qsh = exactQstar(P, gamma, shapedRewards(P, gamma, Phi, P.r.map(() => 0)));
    for (let p = 0; p < P.n; p += 1) {
      assert.ok(Math.abs(Qsh[p] - (Qs[p] - Phi[P.st[p]])) < 1e-8, `trial ${trial}, pair ${p}: Q*_shaped != Q* - Phi`);
    }
    for (let s = 0; s < P.actionsOf.length; s += 1) {
      assert.deepEqual(greedySet(P, Qsh, s), greedySet(P, Qs, s), `trial ${trial}, state ${s}: greedy set changed`);
    }
  }
});

test("Phi(terminal) != 0: Q*_shaped = Q*_bonus - Phi(st), and the greedy action can change (shaped_fixed_general)", () => {
  const rand = mulberry32(0x5eed07);
  let flips = 0;
  for (let trial = 0; trial < 300; trial += 1) {
    const P = randomProblem(rand, { nStates: randInt(rand, 1, 6), maxActions: 3, pTerminal: 0.4 });
    const gamma = uniform(rand, 0.5, 0.95);
    const Phi = Array.from({ length: P.actionsOf.length }, () => Math.round(uniform(rand, -15, 5) * 10) / 10);
    const terminal = P.r.map(() => Math.round(uniform(rand, -20, 20) * 10) / 10);
    const Qs = exactQstar(P, gamma);
    const Qsh = exactQstar(P, gamma, shapedRewards(P, gamma, Phi, terminal));
    const bonus = P.r.map((rp, p) => rp + (P.next[p] >= 0 ? 0 : gamma * terminal[p]));
    const Qb = exactQstar(P, gamma, bonus);
    for (let p = 0; p < P.n; p += 1) {
      assert.ok(Math.abs(Qsh[p] - (Qb[p] - Phi[P.st[p]])) < 1e-8, `trial ${trial}, pair ${p}: Q*_shaped != Q*_bonus - Phi`);
    }
    for (let s = 0; s < P.actionsOf.length; s += 1) {
      if (greedySet(P, Qsh, s).join() !== greedySet(P, Qs, s).join()) flips += 1;
    }
  }
  assert.ok(flips > 0, "a non-zero terminal potential never changed a greedy action");
});

test("the Lean counterexamples: L14 lifted to Q*, and Studio's horizon potential preferring the fire", () => {
  // nonzero_terminal_flips_qstar: one state, two terminal actions, gamma = 1, Phi(start) = 0.
  const flip = { st: [0, 0], r: [-10, -9], next: [-1, -1], actionsOf: [[0, 1]], succ: [[], []], n: 2 };
  const Qs = exactQstar(flip, 1);
  const Qsh = exactQstar(flip, 1, shapedRewards(flip, 1, [0], [5, 0]));
  assert.deepEqual(Qs, [-10, -9]);
  assert.deepEqual(Qsh, [-5, -9]);
  assert.deepEqual(greedySet(flip, Qs, 0), [1]);
  assert.deepEqual(greedySet(flip, Qsh, 0), [0]);
  // horizon_potential_prefers_fire: rl.js's last step, gamma = 0.99. Pair 0 enters a burning
  // node (-10, potential zeroed because done); pair 1 takes a median link (-0.1) to a node
  // 150 median links from the exit, where the horizon stops the episode but rl.js keeps
  // Phi(next) = -0.1 * 150 = -15. The current node's potential is -14.9.
  const last = { st: [0, 0], r: [-10, -0.1], next: [-1, -1], actionsOf: [[0, 1]], succ: [[], []], n: 2 };
  const Q0 = exactQstar(last, 0.99);
  const Q1 = exactQstar(last, 0.99, shapedRewards(last, 0.99, [-14.9], [0, -15]));
  assert.deepEqual(greedySet(last, Q0, 0), [1], "unshaped: survive to the horizon");
  assert.deepEqual(greedySet(last, Q1, 0), [0], "rl.js's shaped target: walk into the fire");
});

// ---------------------------------------------------------------------------
// 4. Studio's own trainRl, without fire.
// ---------------------------------------------------------------------------

function graphFrom(nodes, edges) {
  return { nodes: Float32Array.from(nodes.flat()), edges: Uint32Array.from(edges.flat()), blocked: new Uint8Array(edges.length) };
}

test("Studio's trainRl without fire reaches the exit by a shortest route on tiny graphs", () => {
  // A ring of 8 unit-spaced nodes: from 0 to 3 the short way is 3 links, the long way 5.
  const ring = Array.from({ length: 8 }, (_, i) => [Math.cos((2 * Math.PI * i) / 8), Math.sin((2 * Math.PI * i) / 8), 0]);
  const ringEdges = Array.from({ length: 8 }, (_, i) => [i, (i + 1) % 8]);
  // A 4 x 3 grid with one detour: the middle row is walled except at x = 3.
  const gridNodes = [];
  for (let j = 0; j < 3; j += 1) for (let i = 0; i < 4; i += 1) gridNodes.push([i, j, 0]);
  const gridEdges = [];
  for (let j = 0; j < 3; j += 1) for (let i = 0; i < 4; i += 1) {
    const n = j * 4 + i;
    if (i < 3) gridEdges.push([n, n + 1]);
    if (j < 2) gridEdges.push([n, n + 4]);
  }
  const cases = [
    { graph: graphFrom(ring, ringEdges), start: ring[0], exit: ring[3] },
    { graph: graphFrom(gridNodes, gridEdges), start: [0, 0, 0], exit: [0, 2, 0] },
  ];
  for (const { graph, start, exit } of cases) {
    for (const seed of [1, 2, 3]) {
      const r = trainRl(graph, { startPoint: start, exitPoint: exit, useFire: false, episodes: 300, maxSteps: 40, seed });
      assert.equal(r.reached_exit, true, `seed ${seed}: did not reach the exit`);
      assert.ok(Math.abs(r.length - r.optimal_length) < 1e-6, `seed ${seed}: length ${r.length} vs shortest ${r.optimal_length}`);
    }
  }
});
