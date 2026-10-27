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
