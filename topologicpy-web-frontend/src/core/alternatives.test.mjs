// Run with: node --test src/core/alternatives.test.mjs
// The same cases as the server's tests/test_alternatives.py, so the two
// implementations are held to one standard.
import assert from "node:assert/strict";
import { test } from "node:test";

import { alternativeRoutes } from "./alternatives.js";

/** A unit lattice over the given cells, z up. */
function lattice(cells) {
  const index = new Map(cells.map((c, n) => [c.join(","), n]));
  const nodes = cells.flatMap(([i, j]) => [i, j, 0]);
  const edges = [];
  for (const [i, j] of cells) {
    for (const [di, dj] of [[1, 0], [0, 1]]) {
      const other = index.get(`${i + di},${j + dj}`);
      if (other !== undefined) edges.push(index.get(`${i},${j}`), other);
    }
  }
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: new Uint8Array(edges.length / 2) };
}

const block = new Set();
for (let i = 3; i < 9; i += 1) for (let j = 2; j < 6; j += 1) block.add(`${i},${j}`);
const ringCells = [];
for (let i = 0; i < 12; i += 1) for (let j = 0; j < 8; j += 1) if (!block.has(`${i},${j}`)) ringCells.push([i, j]);
const ring = () => lattice(ringCells);
const corridor = () => lattice(Array.from({ length: 60 }, (_, n) => [n % 20, Math.floor(n / 20)]));

test("distinct finds both ways round the block, and no third in the same corridor", () => {
  const r = alternativeRoutes(ring(), [0, 4, 0], [11, 4, 0], { k: 3 });
  assert.deepEqual(r.routes.map((x) => x.cost), [15, 17]);
  assert.ok(r.routes[1].overlap <= 0.5);
  assert.match(r.note, /Only 2 of 3/);
});

test("a single corridor has one way through", () => {
  const r = alternativeRoutes(corridor(), [0, 1, 0], [19, 1, 0], { k: 3 });
  assert.equal(r.routes.length, 1);
});

test("the stretch limit is respected", () => {
  const r = alternativeRoutes(ring(), [0, 4, 0], [11, 4, 0], { k: 2, maxStretch: 1.1 });
  assert.equal(r.routes.length, 1);
});

test("Yen returns routes in increasing cost, loopless and distinct", () => {
  const r = alternativeRoutes(ring(), [0, 4, 0], [11, 4, 0], { k: 4, method: "yen" });
  const costs = r.routes.map((x) => x.cost);
  assert.equal(costs.length, 4);
  assert.deepEqual([...costs].sort((a, b) => a - b), costs);
  assert.equal(costs[0], 15);
  for (const route of r.routes) assert.equal(new Set(route.node_ids).size, route.node_ids.length);
  assert.equal(new Set(r.routes.map((x) => x.node_ids.join(","))).size, 4);
});

test("on a corridor Yen's runners-up tie and overlap the best route", () => {
  const r = alternativeRoutes(corridor(), [0, 1, 0], [19, 1, 0], { k: 4, method: "yen" });
  assert.equal(r.routes.length, 4);
  for (const route of r.routes.slice(1)) assert.ok(route.overlap > 0.9);
});
