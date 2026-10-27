// Run with: node --test src/core/yen.property.test.mjs
//
// Property test for FireEgress/Yen.lean (L12, formal/fireegress): Yen's k shortest
// loopless paths as `alternativeRoutes(..., { method: "yen" })` computes them. On a few
// hundred small random graphs (unit lattices with ties, random planar points, diagonals,
// parallel links, self-loops, blocked links, walls on and off) it checks the theorem
// `yen_kshortest` (KShortest) against a brute-force enumeration of every simple path:
//   1. every route starts at the start node, ends at the exit, repeats no node, and steps
//      along usable links (Simple);
//   2. the reported cost is the route's cost, and the routes are distinct (nodup);
//   3. costs are non-decreasing (sorted);
//   4. the costs equal the min(k, N) smallest costs over all N simple paths, so no simple
//      path left out is cheaper than the last route (last_le), and fewer than k routes
//      means there are no more simple paths (yen_dry_complete).
import assert from "node:assert/strict";
import { test } from "node:test";

import { alternativeRoutes } from "./alternatives.js";
import { edgeLengths } from "./route.js";

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

/**
 * A random graph of at most 12 nodes (under the 20 that snap() treats as a real part, so a
 * node's own point snaps to it): a unit lattice (many equal-cost routes) or random points
 * in the plane, with random links, some parallel duplicates, some self-loops and some
 * blocked links. Positions are distinct.
 */
function randomGraph(rand) {
  const nodes = [];
  const pairs = [];
  if (rand() < 0.5) {
    const W = 2 + Math.floor(rand() * 3), H = 2 + Math.floor(rand() * 2);
    for (let j = 0; j < H; j += 1) for (let i = 0; i < W; i += 1) nodes.push(i, j, 0);
    const pKeep = 0.6 + rand() * 0.4, pDiag = rand() * 0.7;
    for (let j = 0; j < H; j += 1) {
      for (let i = 0; i < W; i += 1) {
        const n = j * W + i;
        if (i < W - 1 && rand() < pKeep) pairs.push([n, n + 1]);
        if (j < H - 1 && rand() < pKeep) pairs.push([n, n + W]);
        if (i < W - 1 && j < H - 1 && rand() < pDiag) pairs.push([n, n + W + 1]);
        if (i > 0 && j < H - 1 && rand() < pDiag) pairs.push([n, n + W - 1]);
      }
    }
  } else {
    const n = 2 + Math.floor(rand() * 8);
    for (let i = 0; i < n; i += 1) nodes.push(i * 0.37 + rand() * 10, rand() * 10, rand() < 0.2 ? rand() : 0);
    const p = 0.25 + rand() * 0.6;
    for (let a = 0; a < n; a += 1) for (let b = a + 1; b < n; b += 1) if (rand() < p) pairs.push(rand() < 0.5 ? [a, b] : [b, a]);
  }
  const n = nodes.length / 3;
  const edges = [];
  for (const [a, b] of pairs) {
    edges.push(a, b);
    if (rand() < 0.1) edges.push(b, a); // a parallel link
  }
  for (let i = 0; i < n; i += 1) if (rand() < 0.05) edges.push(i, i); // a self-loop
  const m = edges.length / 2;
  const pBlock = rand() < 0.5 ? 0 : rand() * 0.3;
  const blocked = new Uint8Array(m);
  for (let e = 0; e < m; e += 1) blocked[e] = rand() < pBlock ? 1 : 0;
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked };
}

/** Usable adjacency: for each node, neighbour -> cheapest usable link weight. */
function adjacency(graph, weights, blocked) {
  const n = graph.nodes.length / 3, m = graph.edges.length / 2;
  const adj = Array.from({ length: n }, () => new Map());
  for (let e = 0; e < m; e += 1) {
    if (blocked?.[e]) continue;
    const a = graph.edges[e * 2], b = graph.edges[e * 2 + 1];
    if (a === b) continue;
    for (const [u, v] of [[a, b], [b, a]]) {
      const old = adj[u].get(v);
      if (old === undefined || weights[e] < old) adj[u].set(v, weights[e]);
    }
  }
  return adj;
}

/** Every simple path from s to t (node lists), with its cost, by depth-first search. */
function allSimplePaths(adj, s, t) {
  const out = [];
  const path = [s];
  const on = new Set([s]);
  const walk = (u, cost) => {
    if (u === t) {
      out.push({ nodes: [...path], cost });
      return;
    }
    for (const [v, w] of adj[u]) {
      if (on.has(v)) continue;
      on.add(v);
      path.push(v);
      walk(v, cost + w);
      path.pop();
      on.delete(v);
    }
  };
  walk(s, 0);
  return out;
}

const TOL = 1e-9;
const close = (a, b) => Math.abs(a - b) <= TOL * Math.max(1, Math.abs(a), Math.abs(b));

test("Yen returns the k shortest loopless paths (L12, yen_kshortest)", () => {
  const rand = mulberry32(20261027);
  const stats = { cases: 0, routes: 0, none: 0, fewerThanK: 0, deep: 0, ties: 0, maxPaths: 0 };
  for (let c = 0; c < 500; c += 1) {
    const graph = randomGraph(rand);
    const n = graph.nodes.length / 3;
    const start = Math.floor(rand() * n);
    const end = rand() < 0.04 ? start : Math.floor(rand() * n);
    const k = 1 + Math.floor(rand() * 8); // the panel allows 2 to 8
    const useWalls = rand() < 0.7;
    const P = graph.nodes;
    const point = (i) => [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]];

    const r = alternativeRoutes(graph, point(start), point(end), { k, method: "yen", useWalls });
    assert.ok(!/time budget/.test(r.note ?? ""), `case ${c}: hit the time budget`);

    const w = edgeLengths(graph);
    const adj = adjacency(graph, w, useWalls ? graph.blocked : null);
    const all = allSimplePaths(adj, start, end);
    const best = all.map((p) => p.cost).sort((a, b) => a - b);
    stats.cases += 1;
    stats.maxPaths = Math.max(stats.maxPaths, all.length);
    stats.routes += r.routes.length;
    if (!all.length) stats.none += 1;
    if (r.routes.length < k) stats.fewerThanK += 1;
    if (r.routes.length >= 5) stats.deep += 1;

    // 4 (count): min(k, N) routes; fewer than k only when there are no more simple paths.
    assert.equal(r.routes.length, Math.min(k, all.length), `case ${c}: ${r.routes.length} routes, k=${k}, N=${all.length}`);

    const keys = new Set();
    let prevCost = -Infinity;
    r.routes.forEach((route, i) => {
      const ids = route.node_ids;
      // 1. Simple s-t path along usable links.
      assert.equal(ids[0], start, `case ${c} route ${i}: starts at ${ids[0]}, not ${start}`);
      assert.equal(ids[ids.length - 1], end, `case ${c} route ${i}: ends at ${ids[ids.length - 1]}, not ${end}`);
      assert.equal(new Set(ids).size, ids.length, `case ${c} route ${i}: repeats a node (${ids})`);
      let cost = 0;
      for (let j = 1; j < ids.length; j += 1) {
        const step = adj[ids[j - 1]].get(ids[j]);
        assert.ok(step !== undefined, `case ${c} route ${i}: ${ids[j - 1]}-${ids[j]} is not a usable link`);
        cost += step;
      }
      // 2. Reported cost is the route's cost; routes are distinct.
      assert.ok(close(route.cost, cost), `case ${c} route ${i}: reported ${route.cost}, recomputed ${cost}`);
      keys.add(ids.join(","));
      // 3. Non-decreasing costs.
      assert.ok(cost >= prevCost - TOL * Math.max(1, Math.abs(cost)), `case ${c} route ${i}: cost ${cost} after ${prevCost}`);
      prevCost = Math.max(prevCost, cost);
      // 4. The i-th route costs the i-th smallest simple-path cost.
      assert.ok(close(cost, best[i]), `case ${c} route ${i}: cost ${cost}, but the ${i + 1}-th smallest is ${best[i]}`);
    });
    assert.equal(keys.size, r.routes.length, `case ${c}: repeated route`);

    // 4 (last_le, as stated in KShortest): no simple path left out is cheaper than the last route.
    if (r.routes.length) {
      const last = r.routes[r.routes.length - 1].cost;
      for (const p of all) {
        if (keys.has(p.nodes.join(","))) continue;
        assert.ok(p.cost >= last - TOL * Math.max(1, last), `case ${c}: left-out path ${p.nodes} costs ${p.cost} < ${last}`);
      }
      if (best.length > r.routes.length && close(best[r.routes.length], last)) stats.ties += 1;
    }
  }
  console.log(JSON.stringify(stats));
  assert.ok(
    stats.none > 10 && stats.fewerThanK > 30 && stats.deep > 50 && stats.ties > 10,
    "the generator covers no route, k > N, five or more routes, and ties at the cut",
  );
});
