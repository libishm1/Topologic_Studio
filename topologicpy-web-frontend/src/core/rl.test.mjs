// Run with: node --test src/core/rl.test.mjs
// Mirrors the server's RL tests (tests/test_pathfinding.py, TestRL).
import assert from "node:assert/strict";
import { test } from "node:test";

import { earliestEscape, qLearningPath, trainRl } from "./rl.js";

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

test("reaches the exit on a long route in few episodes, along a shortest path", () => {
  const g = grid(12, 12);
  const r = qLearningPath(g, 0, 143, { episodes: 60, maxSteps: 200, seed: 1 });
  assert.ok(r.reachedExit);
  assert.equal(r.length, 22);
  assert.equal(r.reward, Math.round((10 - 21 * 0.1) * 1e4) / 1e4);
});

test("never crosses a blocked link", () => {
  const g = grid(5, 5);
  // Wall off column 2 except its top cell.
  for (let e = 0; e < g.edges.length / 2; e += 1) {
    const a = g.edges[e * 2], b = g.edges[e * 2 + 1];
    const [ax, ay] = [a % 5, Math.floor(a / 5)], [bx, by] = [b % 5, Math.floor(b / 5)];
    if (ay === by && Math.min(ax, bx) === 1 && ay < 4) g.blocked[e] = 1;
  }
  const r = qLearningPath(g, 0, 4, { episodes: 200, maxSteps: 100, seed: 2, blocked: g.blocked });
  assert.ok(r.reachedExit);
  const set = new Set(r.path);
  assert.ok(set.has(21) || set.has(22), "goes round through the top row");
});

test("escape feasibility is exact on a race", () => {
  // A 12 x 1 corridor: fire starts at the exit end and spreads one hop per move.
  const g = grid(12, 1);
  const ignite = new Map(Array.from({ length: 12 }, (_, i) => [11 - i, i]));
  assert.equal(earliestEscape(g, 0, 11, ignite, 50, null), null);
  // Fire far behind the walker: free run of 11 moves.
  const late = new Map(Array.from({ length: 12 }, (_, i) => [i, 100]));
  assert.equal(earliestEscape(g, 0, 11, late, 50, null), 11);
});

test("terminal states carry zero potential: a burned rollout reports a note, not a crash (W5)", () => {
  const g = grid(6, 1);
  const r = trainRl(g, { startPoint: [0, 0, 0], exitPoint: [5, 0, 0], firePoint: [5, 0, 0], episodes: 50, maxSteps: 20, seed: 3 });
  assert.equal(r.reached_exit, false);
  assert.equal(r.escape_steps, null);
  assert.match(r.note, /No route escapes/);
});

test("a seed reproduces a run", () => {
  const g = grid(8, 8);
  const a = qLearningPath(g, 0, 63, { episodes: 30, seed: 9 });
  const b = qLearningPath(g, 0, 63, { episodes: 30, seed: 9 });
  assert.deepEqual(a.path, b.path);
  assert.deepEqual(a.curve, b.curve);
});

test("the exact optimum bounds the policy and matches brute force on small graphs (LG6)", async () => {
  const { qLearningPath } = await import("./rl.js");
  // A 4 x 3 lattice, exit in a corner, fire spreading from the far corner.
  const w = 4, h = 3, nodes = [], edges = [];
  for (let j = 0; j < h; j += 1) for (let i = 0; i < w; i += 1) nodes.push(i, 0, j);
  for (let j = 0; j < h; j += 1) {
    for (let i = 0; i < w; i += 1) {
      const v = j * w + i;
      if (i < w - 1) edges.push(v, v + 1);
      if (j < h - 1) edges.push(v, v + w);
    }
  }
  const g = { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2) };
  const exit = 0, start = w * h - 2, gamma = 0.99, maxSteps = 6;
  for (const ignite of [null, new Map([[w * h - 1, 0], [w * h - 2, 2], [w * h - 3, 3], [w * h - 1 - w, 2]])]) {
    const r = qLearningPath(g, start, exit, { ignite, episodes: 300, maxSteps, seed: 3, gamma });
    assert.ok(r.policyReturn <= r.optimalReturn + 1e-9);
    // Brute force: every walk of at most maxSteps moves.
    const adj = Array.from({ length: w * h }, () => []);
    for (let e = 0; e < edges.length; e += 2) {
      adj[edges[e]].push(edges[e + 1]);
      adj[edges[e + 1]].push(edges[e]);
    }
    const burns = (v) => (ignite ? ignite.get(v) ?? Infinity : Infinity);
    const best = (u, t) => {
      if (t === maxSteps) return 0;
      let b = -Infinity;
      for (const v of adj[u]) {
        const len = Math.hypot(nodes[v * 3] - nodes[u * 3], nodes[v * 3 + 2] - nodes[u * 3 + 2]);
        const value = t >= burns(v) ? -10 : v === exit ? 10 : -0.1 * len + gamma * best(v, t + 1);
        if (value > b) b = value;
      }
      return b;
    };
    assert.ok(Math.abs(best(start, 0) - r.optimalReturn) < 1e-9, `${best(start, 0)} vs ${r.optimalReturn}`);
  }
});

test("the policy's return is scored in the optimum's MDP, so it is never above it (review K1)", () => {
  // The review's line X - S - A - B - E, fire from B one hop per step. The
  // drawn rollout stops at X once every neighbour is visited; scored there it
  // booked -0.1 against an optimum of -10.
  const line = {
    nodes: Float32Array.from([-1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0]),
    edges: Uint32Array.from([0, 1, 1, 2, 2, 3, 3, 4]),
    blocked: new Uint8Array(4),
  };
  const ignite = new Map([[3, 0], [2, 1], [4, 1], [1, 2], [0, 3]]);
  for (let seed = 0; seed < 20; seed += 1) {
    const r = qLearningPath(line, 1, 4, { ignite, episodes: 300, maxSteps: 10, seed });
    assert.ok(r.policyReturn <= r.optimalReturn + 1e-9, `seed ${seed}: ${r.policyReturn} > ${r.optimalReturn}`);
  }

  // Random grids: walls, start, exit, fire (or none), its speed, the step
  // limit and the training length all drawn at random.
  let s = 7;
  const rand = () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648;
  };
  for (let c = 0; c < 80; c += 1) {
    const w = 3 + Math.floor(rand() * 6), h = 2 + Math.floor(rand() * 6);
    const g = grid(w, h);
    for (let e = 0; e < g.blocked.length; e += 1) if (rand() < 0.15) g.blocked[e] = 1;
    const at = () => {
      const i = Math.floor(rand() * w * h);
      return [i % w, Math.floor(i / w), 0];
    };
    const r = trainRl(g, {
      startPoint: at(),
      exitPoint: at(),
      firePoint: rand() < 0.75 ? at() : null,
      useFire: true,
      fireSpeed: [0.25, 0.5, 1, 2][Math.floor(rand() * 4)],
      maxSteps: 4 + Math.floor(rand() * 40),
      episodes: 20 + Math.floor(rand() * 200),
      seed: c,
    });
    if (r.optimal_return == null) continue;
    assert.ok(r.policy_return <= r.optimal_return + 1e-9, `case ${c}: ${r.policy_return} > ${r.optimal_return}`);
  }
});
