// Run with: npm test
import assert from "node:assert/strict";
import { test } from "node:test";

import { departureSafe, escapeDeadlines, frontArrival } from "./deadline.js";

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
