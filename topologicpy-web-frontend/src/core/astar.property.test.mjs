// Run with: node --test src/core/astar.property.test.mjs
//
// Property test for FireEgress/AStar.lean (L9, formal/fireegress): A* with the
// straight-line heuristic and weights at least the link lengths returns the
// optimum. On random weighted grid graphs (jittered positions, random
// diagonals, parallel links, blocked links, hazard weights
// length * (1 + alpha * hazard)) it checks
//   1. every reduced weight w(u,v) - h(u) + h(v) is non-negative (L9.1);
//   2. astar's cost equals an independent Dijkstra written here, and equals
//      route.js's distancesFrom when the weights are plain lengths (L9.3);
//   3. the returned node list is a real walk over open links whose cost is the
//      reported cost (L9.2: the label is the cost of a walk);
//   4. astar returns null exactly when the reference finds no walk.
import assert from "node:assert/strict";
import { test } from "node:test";

import { astar, distancesFrom, edgeLengths } from "./route.js";

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

/** A random W x H grid: jittered nodes, 4-neighbour links, some diagonals and duplicates. */
function randomGrid(rand) {
  const W = 2 + Math.floor(rand() * 9), H = 2 + Math.floor(rand() * 9);
  const nodes = [], edges = [];
  const jitter = rand() < 0.5 ? 0 : 0.4;
  for (let j = 0; j < H; j += 1) {
    for (let i = 0; i < W; i += 1) {
      nodes.push(i + jitter * (rand() - 0.5), j + jitter * (rand() - 0.5), rand() < 0.2 ? rand() * 0.5 : 0);
    }
  }
  const pDiag = rand() * 0.6;
  for (let j = 0; j < H; j += 1) {
    for (let i = 0; i < W; i += 1) {
      const n = j * W + i;
      if (i < W - 1) edges.push(n, n + 1);
      if (j < H - 1) edges.push(n, n + W);
      if (i < W - 1 && j < H - 1 && rand() < pDiag) edges.push(n, n + W + 1);
      if (i > 0 && j < H - 1 && rand() < pDiag) edges.push(n, n + W - 1);
    }
  }
  // Parallel links: route.js keeps both, A* must use the cheaper.
  const m0 = edges.length / 2;
  for (let e = 0; e < m0; e += 1) if (rand() < 0.05) edges.push(edges[e * 2 + 1], edges[e * 2]);
  const m = edges.length / 2;
  const pBlock = rand() * 0.45;
  const blocked = new Uint8Array(m);
  for (let e = 0; e < m; e += 1) blocked[e] = rand() < pBlock ? 1 : 0;
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked };
}

/** Hazard weights length * (1 + alpha * hazard), hazard in [0, 1], or null for plain lengths. */
function randomWeights(rand, graph) {
  if (rand() < 0.3) return null;
  const len = edgeLengths(graph);
  const alpha = rand() * 4;
  return Float64Array.from(len, (l) => l * (1 + alpha * (rand() < 0.5 ? 0 : rand())));
}

/** Reference: textbook O(n^2) Dijkstra over open links, no heap, no heuristic. */
function referenceDijkstra(graph, weights, blocked, source) {
  const n = graph.nodes.length / 3, m = graph.edges.length / 2;
  const adj = Array.from({ length: n }, () => []);
  for (let e = 0; e < m; e += 1) {
    if (blocked?.[e]) continue;
    const a = graph.edges[e * 2], b = graph.edges[e * 2 + 1];
    adj[a].push([b, weights[e]]);
    adj[b].push([a, weights[e]]);
  }
  const dist = new Array(n).fill(Infinity);
  const done = new Array(n).fill(false);
  dist[source] = 0;
  for (;;) {
    let u = -1;
    for (let i = 0; i < n; i += 1) if (!done[i] && dist[i] < Infinity && (u < 0 || dist[i] < dist[u])) u = i;
    if (u < 0) break;
    done[u] = true;
    for (const [v, w] of adj[u]) if (dist[u] + w < dist[v]) dist[v] = dist[u] + w;
  }
  return dist;
}

/** Cost of a node list over open links (cheapest parallel link), or NaN if a step has no open link. */
function walkCost(graph, weights, blocked, nodes) {
  const m = graph.edges.length / 2;
  let cost = 0;
  for (let i = 1; i < nodes.length; i += 1) {
    let best = Infinity;
    for (let e = 0; e < m; e += 1) {
      if (blocked?.[e]) continue;
      const a = graph.edges[e * 2], b = graph.edges[e * 2 + 1];
      if ((a === nodes[i - 1] && b === nodes[i]) || (b === nodes[i - 1] && a === nodes[i])) best = Math.min(best, weights[e]);
    }
    if (!Number.isFinite(best)) return NaN;
    cost += best;
  }
  return cost;
}

const close = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));

test("A* returns the Dijkstra optimum and reduced weights are non-negative (L9)", () => {
  const rand = mulberry32(20261027);
  const stats = { cases: 0, routes: 0, none: 0, exact: 0, reducedChecked: 0, reducedFpNegative: 0, minReduced: Infinity };
  for (let c = 0; c < 400; c += 1) {
    const graph = randomGrid(rand);
    const n = graph.nodes.length / 3, m = graph.edges.length / 2;
    const weights = randomWeights(rand, graph);
    const w = weights || edgeLengths(graph);
    const useWalls = rand() < 0.8;
    const blocked = useWalls ? graph.blocked : null;
    const start = Math.floor(rand() * n);
    const end = rand() < 0.05 ? start : Math.floor(rand() * n);

    // L9.1: reduced weights with h = straight-line distance to `end`, computed as route.js does.
    const P = graph.nodes;
    const h = (i) => Math.hypot(P[i * 3] - P[end * 3], P[i * 3 + 1] - P[end * 3 + 1], P[i * 3 + 2] - P[end * 3 + 2]);
    for (let e = 0; e < m; e += 1) {
      if (blocked?.[e]) continue;
      const a = graph.edges[e * 2], b = graph.edges[e * 2 + 1];
      for (const [u, v] of [[a, b], [b, a]]) {
        const r = w[e] - h(u) + h(v);
        stats.reducedChecked += 1;
        stats.minReduced = Math.min(stats.minReduced, r);
        if (r < 0) stats.reducedFpNegative += 1;
        // Non-negative in exact arithmetic; doubles may round a zero to -1 ulp.
        assert.ok(r >= -1e-9 * Math.max(1, w[e]), `case ${c}: reduced weight ${r} on ${u}->${v}`);
      }
    }

    // L9.3: A* against the reference.
    const ref = referenceDijkstra(graph, w, blocked, start);
    const result = astar(graph, start, end, { weights: weights || undefined, useWalls });
    stats.cases += 1;
    if (!Number.isFinite(ref[end])) {
      assert.equal(result, null, `case ${c}: A* found a route the reference says does not exist`);
      stats.none += 1;
      continue;
    }
    assert.ok(result, `case ${c}: A* found no route but the reference has cost ${ref[end]}`);
    stats.routes += 1;
    assert.ok(close(result.cost, ref[end]), `case ${c}: A* cost ${result.cost} vs optimum ${ref[end]}`);
    if (result.cost === ref[end]) stats.exact += 1;

    // L9.2: the label is the cost of the walk it returns.
    assert.equal(result.nodes[0], start);
    assert.equal(result.nodes[result.nodes.length - 1], end);
    const wc = walkCost(graph, w, blocked, result.nodes);
    assert.ok(close(wc, result.cost), `case ${c}: walk cost ${wc} vs reported ${result.cost}`);

    // route.js's own Dijkstra agrees when the weights are plain lengths.
    if (!weights) {
      const d = distancesFrom(graph, start, { blocked });
      assert.ok(close(d[end], result.cost), `case ${c}: distancesFrom ${d[end]} vs A* ${result.cost}`);
    }
  }
  console.log(JSON.stringify(stats));
  assert.ok(stats.routes > 100 && stats.none > 10, "the generator covers both reachable and cut-off goals");
});
