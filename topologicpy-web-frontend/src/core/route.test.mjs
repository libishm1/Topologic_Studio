// Run with: node --test src/core/route.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { astar, findRoute, snap } from "./route.js";

/** A w x h unit lattice (4-neighbour) starting at x = ox, z up. */
function lattice(w, h, ox = 0) {
  const nodes = [], edges = [];
  for (let j = 0; j < h; j += 1) for (let i = 0; i < w; i += 1) nodes.push(ox + i, j, 0);
  for (let j = 0; j < h; j += 1) {
    for (let i = 0; i < w; i += 1) {
      const n = j * w + i;
      if (i < w - 1) edges.push(n, n + 1);
      if (j < h - 1) edges.push(n, n + w);
    }
  }
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2) };
}

function join(a, b) {
  const offset = a.nodes.length / 3;
  return {
    nodes: Float32Array.from([...a.nodes, ...b.nodes]),
    edges: Uint32Array.from([...a.edges, ...Array.from(b.edges, (e) => e + offset)]),
    blocked: new Uint8Array((a.edges.length + b.edges.length) / 2),
  };
}

test("A* finds the Manhattan optimum on a unit lattice", () => {
  const g = lattice(12, 12);
  const r = astar(g, 0, 143);
  assert.equal(r.cost, 22);
  assert.equal(r.nodes.length, 23);
});

test("a blocked link forces a detour", () => {
  const g = lattice(5, 5);
  // Block both links into node 2 from the bottom row's left side.
  for (let e = 0; e < g.edges.length / 2; e += 1) {
    const pair = [g.edges[e * 2], g.edges[e * 2 + 1]].sort((x, y) => x - y).join(",");
    if (pair === "1,2") g.blocked[e] = 1;
  }
  assert.equal(astar(g, 0, 2).cost, 4);
  assert.equal(astar(g, 0, 2, { useWalls: false }).cost, 2);
});

test("a pick in the second of two equal parts snaps into that part", () => {
  const g = join(lattice(6, 6), lattice(6, 6, 20));
  const n = snap(g, [21, 1, 0]);
  assert.ok(g.nodes[n * 3] >= 20);
});

test("a route between separate parts explains why there is none", () => {
  const g = join(lattice(6, 6), lattice(6, 6, 20));
  const r = findRoute(g, [0, 0, 0], [22, 2, 0]);
  assert.equal(r.found, false);
  assert.match(r.note, /separate parts/);
  const ok = findRoute(g, [20, 0, 0], [25, 5, 0]);
  assert.equal(ok.found, true);
  assert.equal(ok.length, 10);
  assert.equal(ok.engine, "browser");
});
