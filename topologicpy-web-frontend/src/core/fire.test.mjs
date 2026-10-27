// Run with: node --test src/core/fire.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { floodTimeline, radialTimeline, runFire, temperatureSteps } from "./fire.js";

/** A path graph 0-1-...-(n-1) with unit spacing, z up. */
function chain(n) {
  const nodes = [], edges = [];
  for (let i = 0; i < n; i += 1) nodes.push(i, 0, 0);
  for (let i = 0; i < n - 1; i += 1) edges.push(i, i + 1);
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: new Uint8Array(n - 1) };
}

test("flood spreads one hop per step and stops at a blocked link", () => {
  const g = chain(6);
  assert.deepEqual(floodTimeline(g, 0, 10), [[0], [1], [2], [3], [4], [5]]);
  g.blocked[2] = 1; // between 2 and 3
  assert.deepEqual(floodTimeline(g, 0, 10, g.blocked), [[0], [1], [2]]);
});

test("radial bands follow straight-line distance", () => {
  const bands = radialTimeline(chain(5), 2, 10);
  assert.deepEqual(bands, [[2], [1, 3], [0, 4]]);
});

test("the temperature step never pushes a node past its hotter neighbour (W1)", () => {
  // Node 0 is a dead end (degree 1) beside the seed's neighbour. Classic's
  // 1.2 / degree overshot here; the clamped coefficient must not.
  const g = chain(4);
  let previous = null;
  for (const t of temperatureSteps(g, 3, { maxSteps: 40 })) {
    if (previous) {
      for (let i = 0; i < 3; i += 1) {
        const hottest = Math.max(previous[i], previous[i + 1], i > 0 ? previous[i - 1] : -Infinity);
        assert.ok(t[i] <= hottest + 1e-9, `node ${i} rose to ${t[i]} above its neighbourhood's ${hottest}`);
      }
    }
    for (const v of t) assert.ok(v >= 20 && v <= 120);
    previous = t;
  }
});

test("a full run emits steps, reroutes round the fire, and finishes", async () => {
  // A 7 x 3 lattice; fire in the middle of the top row, walker bottom-left.
  const nodes = [], edges = [];
  for (let j = 0; j < 3; j += 1) for (let i = 0; i < 7; i += 1) nodes.push(i, j, 0);
  for (let j = 0; j < 3; j += 1) {
    for (let i = 0; i < 7; i += 1) {
      const n = j * 7 + i;
      if (i < 6) edges.push(n, n + 1);
      if (j < 2) edges.push(n, n + 7);
    }
  }
  const g = { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2) };
  const messages = [];
  await new Promise((resolve) =>
    runFire(
      g,
      { model: "flood", firePoint: [3, 2, 0], startPoint: [0, 0, 0], exitPoint: [6, 0, 0], maxSteps: 2, delayMs: 0, reroute: true, interval: 1 },
      { onMessage: (m) => messages.push(m), onDone: resolve },
    ),
  );
  const steps = messages.filter((m) => m.type === "step");
  const paths = messages.filter((m) => m.type === "path_update");
  assert.equal(steps.length, 2);
  assert.ok(paths.length >= 1);
  assert.ok(paths.every((p) => p.engine === "browser"));
  // One hop from the top middle leaves the bottom row clear: a safe route.
  assert.ok(paths[paths.length - 1].found);
});

test("when the fire cuts every route the drawn route is not reported as found (W4)", async () => {
  // Two hops from the top middle reach the bottom middle: every route is cut.
  const nodes = [], edges = [];
  for (let j = 0; j < 3; j += 1) for (let i = 0; i < 7; i += 1) nodes.push(i, j, 0);
  for (let j = 0; j < 3; j += 1) {
    for (let i = 0; i < 7; i += 1) {
      const n = j * 7 + i;
      if (i < 6) edges.push(n, n + 1);
      if (j < 2) edges.push(n, n + 7);
    }
  }
  const g = { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2) };
  const paths = [];
  await new Promise((resolve) =>
    runFire(
      g,
      { model: "flood", firePoint: [3, 2, 0], startPoint: [0, 0, 0], exitPoint: [6, 0, 0], maxSteps: 3, delayMs: 0, reroute: true, interval: 1 },
      { onMessage: (m) => m.type === "path_update" && paths.push(m), onDone: resolve },
    ),
  );
  const last = paths[paths.length - 1];
  assert.equal(last.found, false);
  assert.ok(last.path.length > 0, "the least-hazard route is still drawn");
  assert.match(last.note, /not a safe route/);
});
